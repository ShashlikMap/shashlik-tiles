//! Camera-driven tile loading with an LRU cache.
//!
//! [`TileCache`] sits on top of any [`TileSource`]: given a [`Camera`] it works
//! out the visible tiles at the right zoom, loads the missing ones concurrently,
//! caches them under a byte budget, and returns what to draw — falling back to a
//! cached coarser ancestor for any tile not yet loaded so there are no blank
//! gaps. A ring of neighbouring tiles is prefetched in the background for smooth
//! panning.
//!
//! The cache is generic over its payload via `TilePayload`: today that's raw
//! `Vec<u8>` tile bytes; once the tile-format decoder exists, a decoded
//! render-ready type implements the same trait with no interface change.

use crate::reader::{TileRange, TileSource};
use crate::{Mercator, Tile};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Assumed on-screen tile size in pixels (matches the tiler's `TILE_RENDER_PX`).
pub const TILE_RENDER_PX: f64 = 512.0;

/// The map point-of-view. `bearing`/`pitch` are radians; `pitch` is carried for
/// a future 2.5D frustum but ignored by the 2D visible-tile computation.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub center: Mercator,
    pub zoom: f64,
    pub bearing: f64,
    pub pitch: f64,
    pub viewport_px: [f32; 2],
}

impl Camera {
    /// A north-up, top-down camera at `center` and `zoom` for a `w`×`h` viewport.
    pub fn new(center: Mercator, zoom: f64, viewport_px: [f32; 2]) -> Self {
        Self {
            center,
            zoom,
            bearing: 0.0,
            pitch: 0.0,
            viewport_px,
        }
    }

    /// The slippy zoom the map is actually displayed at. Tile ids and zoom
    /// numbering use the standard 256-px convention, but tiles are drawn
    /// [`TILE_RENDER_PX`] px each — so a `TILE_RENDER_PX` of 512 shows one level
    /// finer than `zoom`. Selecting the tile level and the display-filter zoom
    /// from this (not the raw `zoom`) keeps loaded detail and class visibility in
    /// step with what's on screen.
    #[inline]
    pub fn effective_zoom(&self) -> f64 {
        self.zoom + (TILE_RENDER_PX / 256.0).log2()
    }

    /// Build a reusable projection from scene coordinates (as produced by
    /// `Scene`, at `target_zoom` with tile `extent`) to screen pixels. Precomputes
    /// the mercator centre, scale, and rotation once — call
    /// [`SceneProjection::to_screen`] per vertex.
    pub fn scene_projection(&self, target_zoom: u8, extent: u16) -> SceneProjection {
        let e = extent as f64;
        // Camera centre in scene units (tile fraction × extent).
        let (fx, fy) = self.center.to_fractional_tile(target_zoom);
        // One scene unit = 1/extent of a target-zoom tile; that tile is
        // `TILE_RENDER_PX * 2^(zoom-target_zoom)` px on screen (overzoom).
        let scale = TILE_RENDER_PX * 2f64.powf(self.zoom - target_zoom as f64) / e;
        let (sin, cos) = self.bearing.sin_cos();
        SceneProjection {
            center: [fx * e, fy * e],
            scale,
            sin,
            cos,
            half: [
                self.viewport_px[0] as f64 / 2.0,
                self.viewport_px[1] as f64 / 2.0,
            ],
        }
    }

    /// Convert a single scene coordinate to a screen pixel. For many points build
    /// a `SceneProjection` once with `scene_projection`
    /// and reuse it — this rebuilds the projection each call.
    pub fn scene_to_screen(&self, point: [i32; 2], target_zoom: u8, extent: u16) -> [f32; 2] {
        self.scene_projection(target_zoom, extent).to_screen(point)
    }
}

/// A precomputed scene→screen mapping for one camera + tile layout. Scene units
/// are absolute (`tile_index × extent`); `to_screen` subtracts the camera centre,
/// rotates by the bearing, scales to pixels, and offsets to the viewport centre.
/// Both scene-y and screen-y grow downward, so there is no flip.
#[derive(Clone, Copy, Debug)]
pub struct SceneProjection {
    center: [f64; 2],
    scale: f64,
    sin: f64,
    cos: f64,
    half: [f64; 2],
}

impl SceneProjection {
    /// Project one scene coordinate to a screen pixel (origin top-left).
    #[inline]
    pub fn to_screen(&self, point: [i32; 2]) -> [f32; 2] {
        let rx = point[0] as f64 - self.center[0];
        let ry = point[1] as f64 - self.center[1];
        // Rotate by -bearing (the map is rotated `bearing` clockwise from north).
        let x = rx * self.cos + ry * self.sin;
        let y = -rx * self.sin + ry * self.cos;
        [
            (self.half[0] + x * self.scale) as f32,
            (self.half[1] + y * self.scale) as f32,
        ]
    }
}

/// A cacheable tile payload. Implemented for `Vec<u8>` (raw tile bytes); a
/// decoded render-ready tile type would implement it too.
pub trait TilePayload: Send + Sync + 'static {
    /// Build the payload from a tile's stored bytes (as returned by the source).
    fn from_tile_bytes(bytes: Vec<u8>) -> Self;
    /// Approximate heap footprint, used for the cache's byte budget.
    fn byte_size(&self) -> usize;
}

impl TilePayload for Vec<u8> {
    fn from_tile_bytes(bytes: Vec<u8>) -> Self {
        bytes
    }
    fn byte_size(&self) -> usize {
        self.len()
    }
}

/// One tile the renderer should draw this frame.
pub struct VisibleTile<T: ?Sized> {
    /// The tile slot to fill on screen.
    pub tile: Tile,
    /// The payload to draw it with.
    pub data: Arc<T>,
    /// The tile the payload actually belongs to — equal to `tile`, or a coarser
    /// ancestor used as a fallback (the renderer draws the matching sub-rect
    /// scaled up).
    pub source: Tile,
}

impl<T> VisibleTile<T> {
    /// Whether this tile is drawn from a coarser ancestor rather than itself.
    pub fn is_fallback(&self) -> bool {
        self.source != self.tile
    }
}

/// High-level, camera-driven tile access
#[async_trait]
pub trait TileView<T: TilePayload + ?Sized>: Send + Sync {
    /// Recompute the visible set for `cam`, load any missing tiles, and return
    /// what to draw (loaded tiles, coarser fallbacks where not yet available).
    async fn update(&self, cam: &Camera) -> Vec<VisibleTile<T>>;

    /// Non-blocking peek at a cached tile.
    fn get(&self, tile: Tile) -> Option<Arc<T>>;
}
#[async_trait]
impl<A: TilePayload, T: TileView<A>> TileView<A> for Box<T> {
    async fn update(&self, cam: &Camera) -> Vec<VisibleTile<A>> {
        (**self).update(cam).await
    }

    fn get(&self, tile: Tile) -> Option<Arc<A>> {
        (**self).get(tile)
    }
}

/// A `TileView` backed by a `TileSource` and an LRU cache.
pub struct TileCache<S, T> {
    source: Arc<S>,
    cache: Arc<Mutex<Lru<T>>>,
    /// Materialized zoom levels in the archive, ascending (e.g. `[3,4,6,8,10,12,14]`).
    zoom_levels: Vec<u8>,
    /// Extra tiles beyond the viewport to prefetch in the background.
    prefetch_ring: u32,
}

impl<S, T> TileCache<S, T>
where
    S: TileSource + 'static,
    T: TilePayload,
{
    /// Create a cache over `source`. `zoom_levels` are the archive's materialized
    /// levels (ascending); `budget_bytes` caps cached payload size;
    /// `prefetch_ring` is how many extra tile rings to load around the viewport.
    pub fn new(source: S, zoom_levels: Vec<u8>, budget_bytes: usize, prefetch_ring: u32) -> Self {
        let mut zoom_levels = zoom_levels;
        zoom_levels.sort_unstable();
        Self {
            source: Arc::new(source),
            cache: Arc::new(Mutex::new(Lru::new(budget_bytes))),
            zoom_levels,
            prefetch_ring,
        }
    }

    /// The materialized level nearest the camera zoom (ties go to the coarser
    /// level, which is cheaper and overzooms cleanly).
    fn target_level(&self, zoom: f64) -> u8 {
        *self
            .zoom_levels
            .iter()
            .min_by(|&&a, &&b| {
                let (da, db) = ((a as f64 - zoom).abs(), (b as f64 - zoom).abs());
                da.partial_cmp(&db).unwrap().then(a.cmp(&b))
            })
            .expect("zoom_levels must be non-empty")
    }

    /// Tiles covering the viewport at `level`, expanded by `margin` rings. The
    /// rotated viewport rectangle is bounded by an axis-aligned box (a small
    /// over-approximation at non-zero bearing).
    fn visible_range(&self, cam: &Camera, level: u8, margin: u32) -> TileRange {
        let (cx, cy) = cam.center.to_fractional_tile(level);
        let scale = 2f64.powf(cam.zoom - level as f64);
        let px_per_unit = (TILE_RENDER_PX * scale).max(1e-6);
        let half_w = (cam.viewport_px[0] as f64 / 2.0) / px_per_unit;
        let half_h = (cam.viewport_px[1] as f64 / 2.0) / px_per_unit;

        let (sin, cos) = cam.bearing.sin_cos();
        let hx = half_w * cos.abs() + half_h * sin.abs() + margin as f64;
        let hy = half_w * sin.abs() + half_h * cos.abs() + margin as f64;

        let last = (1u32 << level).saturating_sub(1);
        let clamp = |v: f64| v.floor().clamp(0.0, last as f64) as u32;
        TileRange::new(
            level,
            clamp(cx - hx),
            clamp(cx + hx),
            clamp(cy - hy),
            clamp(cy + hy),
        )
    }

    /// Load any of `tiles` not already cached, concurrently, and insert them.
    /// `protected` tiles are never evicted to make room.
    async fn load_missing(&self, tiles: &[Tile], protected: &HashSet<Tile>) {
        let missing: Vec<Tile> = {
            let cache = self.cache.lock().unwrap();
            tiles
                .iter()
                .copied()
                .filter(|t| !cache.contains(*t))
                .collect()
        };
        if missing.is_empty() {
            return;
        }
        // One batched fetch: the source coalesces adjacent tiles' byte ranges into
        // a few reads (crucial on HTTP), instead of a request per tile.
        let loaded = match self.source.tiles(&missing).await {
            Ok(loaded) => loaded,
            // Transient I/O on the batch — skip; a later frame retries.
            Err(_) => return,
        };

        let mut cache = self.cache.lock().unwrap();
        for (tile, opt) in missing.into_iter().zip(loaded) {
            match opt {
                Some(bytes) => {
                    let payload = T::from_tile_bytes(bytes);
                    let size = payload.byte_size();
                    cache.insert(tile, Arc::new(payload), size, protected);
                }
                // Genuinely absent in the archive (e.g. an all-land water tile the
                // build dropped): cache a tombstone so it isn't re-requested every
                // update. Renders as nothing; LRU-evicted like any entry.
                None => cache.insert_absent(tile, protected),
            }
        }
    }

    /// Background-load a ring of tiles around the viewport (best effort).
    fn spawn_prefetch(&self, cam: &Camera, level: u8) {
        if self.prefetch_ring == 0 {
            return;
        }
        let ring: Vec<Tile> = self
            .visible_range(cam, level, self.prefetch_ring)
            .tiles()
            .collect();
        let source = self.source.clone();
        let cache = self.cache.clone();
        tokio::spawn(async move {
            let missing: Vec<Tile> = {
                let c = cache.lock().unwrap();
                ring.into_iter().filter(|t| !c.contains(*t)).collect()
            };
            if missing.is_empty() {
                return;
            }
            // Batched, coalesced fetch for the whole ring (best effort).
            let Ok(loaded) = source.tiles(&missing).await else {
                return;
            };
            let mut c = cache.lock().unwrap();
            for (tile, opt) in missing.into_iter().zip(loaded) {
                if let Some(bytes) = opt {
                    let payload = T::from_tile_bytes(bytes);
                    let size = payload.byte_size();
                    // Prefetched tiles aren't pinned (empty protected set).
                    c.insert(tile, Arc::new(payload), size, &HashSet::new());
                }
            }
        });
    }
}

#[async_trait]
impl<S, T> TileView<T> for TileCache<S, T>
where
    S: TileSource + 'static,
    T: TilePayload,
{
    async fn update(&self, cam: &Camera) -> Vec<VisibleTile<T>> {
        let level = self.target_level(cam.effective_zoom());
        let tiles: Vec<Tile> = self.visible_range(cam, level, 0).tiles().collect();
        let protected: HashSet<Tile> = tiles.iter().copied().collect();

        self.load_missing(&tiles, &protected).await;

        let mut out = Vec::with_capacity(tiles.len());
        {
            let mut cache = self.cache.lock().unwrap();
            // Only substitute a coarser ancestor when the target level is beyond
            // the archive (genuine overzoom). A miss at a *materialized* level
            // means the feature is genuinely absent there — e.g. an all-land tile
            // the water build dropped — so render nothing. Falling back per-absent-
            // tile would pull in whichever coarser tile happens to be cached, which
            // varies by pan history and makes detail change as you move at fixed zoom
            let overzoom = level > self.source.max_zoom();
            for tile in tiles {
                if let Some(data) = cache.get(tile) {
                    out.push(VisibleTile {
                        tile,
                        data,
                        source: tile,
                    });
                } else if overzoom {
                    if let Some((source, data)) = cache.nearest_ancestor(tile) {
                        out.push(VisibleTile { tile, data, source });
                    }
                }
                // else: absent at a materialized level -> blank (no feature here).
            }
        }

        self.spawn_prefetch(cam, level);
        out
    }

    fn get(&self, tile: Tile) -> Option<Arc<T>> {
        self.cache.lock().unwrap().get(tile)
    }
}

/// LRU cache with a byte budget. Eviction never removes a `protected` tile.
struct Lru<T> {
    map: HashMap<Tile, Entry<T>>,
    budget: usize,
    used: usize,
    clock: u64,
}

struct Entry<T> {
    /// `None` is a tombstone: the tile is genuinely absent in the archive, cached
    /// so it isn't re-requested from the source on every update.
    data: Option<Arc<T>>,
    bytes: usize,
    touched: u64,
}

impl<T> Lru<T> {
    fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            budget,
            used: 0,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn contains(&self, tile: Tile) -> bool {
        self.map.contains_key(&tile)
    }

    /// Fetch and mark most-recently-used.
    fn get(&mut self, tile: Tile) -> Option<Arc<T>> {
        let now = self.tick();
        let entry = self.map.get_mut(&tile)?;
        entry.touched = now;
        entry.data.clone() // None for a tombstone (absent tile)
    }

    fn insert(&mut self, tile: Tile, data: Arc<T>, bytes: usize, protected: &HashSet<Tile>) {
        let now = self.tick();
        if let Some(old) = self.map.insert(
            tile,
            Entry {
                data: Some(data),
                bytes,
                touched: now,
            },
        ) {
            self.used -= old.bytes;
        }
        self.used += bytes;
        self.evict(protected);
    }

    /// Record that `tile` is absent in the archive (a tombstone) so it isn't
    /// re-requested every update. Counts a nominal size toward the budget so
    /// tombstones stay LRU-bounded like real entries.
    fn insert_absent(&mut self, tile: Tile, protected: &HashSet<Tile>) {
        const NOMINAL: usize = 64;
        let now = self.tick();
        if let Some(old) = self.map.insert(
            tile,
            Entry {
                data: None,
                bytes: NOMINAL,
                touched: now,
            },
        ) {
            self.used -= old.bytes;
        }
        self.used += NOMINAL;
        self.evict(protected);
    }

    fn evict(&mut self, protected: &HashSet<Tile>) {
        while self.used > self.budget {
            let victim = self
                .map
                .iter()
                .filter(|(t, _)| !protected.contains(*t))
                .min_by_key(|(_, e)| e.touched)
                .map(|(t, _)| *t);
            match victim {
                Some(tile) => {
                    if let Some(e) = self.map.remove(&tile) {
                        self.used -= e.bytes;
                    }
                }
                None => break, // everything left is protected
            }
        }
    }

    /// Nearest cached ancestor (walking parents; skips non-materialized zooms,
    /// which are never cached). Marks it most-recently-used.
    fn nearest_ancestor(&mut self, tile: Tile) -> Option<(Tile, Arc<T>)> {
        let mut cur = tile;
        while let Some(parent) = cur.parent() {
            cur = parent;
            if let Some(data) = self.get(cur) {
                return Some((cur, data));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    /// A source that returns `z/x/y` bytes for every requested tile.
    struct DenseSource;

    #[async_trait]
    impl TileSource for DenseSource {
        async fn tile(&self, t: Tile) -> io::Result<Option<Vec<u8>>> {
            Ok(Some(format!("{}/{}/{}", t.z, t.x, t.y).into_bytes()))
        }
        fn min_zoom(&self) -> u8 {
            0
        }
        fn max_zoom(&self) -> u8 {
            14
        }
    }

    /// A source that only has tiles at zoom 12 (so z14 requests miss → fallback).
    struct CoarseOnly;

    #[async_trait]
    impl TileSource for CoarseOnly {
        async fn tile(&self, t: Tile) -> io::Result<Option<Vec<u8>>> {
            Ok((t.z == 12).then(|| vec![t.z]))
        }
        fn min_zoom(&self) -> u8 {
            12
        }
        fn max_zoom(&self) -> u8 {
            12
        }
    }

    fn camera_at(zoom: f64, px: f32) -> Camera {
        // Null-island centre; a small square viewport.
        Camera::new(Mercator::new(0.0, 0.0), zoom, [px, px])
    }

    #[tokio::test]
    async fn update_loads_and_caches_visible_tiles() {
        let cache: TileCache<_, Vec<u8>> =
            TileCache::new(DenseSource, vec![12, 14], 8 * 1024 * 1024, 0);

        // 256px viewport at z14 -> a 2×2 block around the centre.
        let visible = cache.update(&camera_at(14.0, 256.0)).await;
        assert_eq!(visible.len(), 4);
        assert!(visible.iter().all(|v| !v.is_fallback()));

        // All visible tiles are now cached and resolve to their own data.
        for v in &visible {
            let cached = cache.get(v.tile).expect("cached");
            assert_eq!(&*cached, format!("14/{}/{}", v.tile.x, v.tile.y).as_bytes());
        }
    }

    #[tokio::test]
    async fn zoom_snaps_to_nearest_materialized_level() {
        let cache: TileCache<_, Vec<u8>> = TileCache::new(DenseSource, vec![12, 14], 1 << 20, 0);
        // Selection uses the *effective* zoom (camera + 1 with 512px tiles), so the
        // 12↔14 midpoint of effective 13 lands at camera 12: above snaps to 14,
        // below to 12.
        let visible = cache.update(&camera_at(12.5, 256.0)).await;
        assert!(visible.iter().all(|v| v.tile.z == 14));
        let visible = cache.update(&camera_at(11.4, 256.0)).await;
        assert!(visible.iter().all(|v| v.tile.z == 12));
    }

    #[tokio::test]
    async fn missing_tiles_fall_back_to_coarser_ancestor() {
        let cache: TileCache<_, Vec<u8>> = TileCache::new(CoarseOnly, vec![12, 14], 1 << 20, 0);
        // Warm the cache with the z12 coverage first.
        cache.update(&camera_at(12.0, 256.0)).await;
        // Now ask at z14: those tiles are absent in the source, so each visible
        // slot is served by its cached z12 ancestor.
        let visible = cache.update(&camera_at(14.0, 256.0)).await;
        assert!(!visible.is_empty());
        assert!(visible.iter().all(|v| v.is_fallback() && v.source.z == 12));
    }

    #[tokio::test]
    async fn budget_evicts_unprotected_tiles() {
        // Budget for ~2 tiles (each payload is a few bytes); pan across many.
        let cache: TileCache<_, Vec<u8>> = TileCache::new(DenseSource, vec![14], 12, 0);
        cache.update(&camera_at(14.0, 64.0)).await; // small viewport -> few tiles
        // Pan far away; the earlier tiles are unprotected and should be evicted.
        let cam = Camera::new(Mercator::new(1_000_000.0, 1_000_000.0), 14.0, [64.0, 64.0]);
        let visible = cache.update(&cam).await;
        assert!(!visible.is_empty());
        // Cache stays within a small multiple of the budget (not unbounded).
        let used = cache.cache.lock().unwrap().used;
        assert!(used <= 64, "cache grew unbounded: {used} bytes");
    }
}
