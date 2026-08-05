//! Tile decoding and camera-view scene assembly

use crate::view::{Camera, TilePayload, TileView, VisibleTile};
use geo::{Coord, LineString, Polygon};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
pub use util::feature::{AreaKind, EdgeNode, LabelClass, Lanes, PoiKind, RoadKind};
use util::tileformat::{
    AreaMeta, FLAG_RING_CLIP_MASK, LabelMeta, NAME_NONE, PoiMeta, RingMeta, RoadMeta, TileHeader,
    dezigzag,
};

// ─────────────────────────── decoded tile (cache payload) ───────────────────

/// A tile decoded into owned, tile-local geometry (`[0, extent]` coordinates).
#[derive(Default, Clone)]
pub struct DecodedTile {
    pub extent: u16,
    pub roads: Vec<DecRoad>,
    pub areas: Vec<DecArea>,
    pub labels: Vec<DecLabel>,
    pub pois: Vec<DecPoi>,
}

#[derive(Clone)]
pub struct DecRoad {
    pub kind: RoadKind,
    pub layer: i8,
    pub lanes: Lanes,
    pub start: EdgeNode,
    pub end: EdgeNode,
    pub name: Option<String>,
    pub coords: Vec<[i16; 2]>,
}

#[derive(Clone)]
pub struct DecArea {
    pub kind: AreaKind,
    pub layer: i8,
    pub floors: u8,
    /// Ring 0 is the outer ring; the rest are holes.
    pub rings: Vec<Vec<[i16; 2]>>,
    /// One flag per ring vertex (rings concatenated): clip-produced boundary
    /// vertex. Empty means none.
    pub clip: Vec<bool>,
}

#[derive(Clone)]
pub struct DecLabel {
    pub class: LabelClass,
    pub rank: u8,
    pub anchor: [i16; 2],
    pub name: String,
}

#[derive(Clone)]
pub struct DecPoi {
    pub kind: PoiKind,
    pub anchor: [i16; 2],
}

/// Decode a compressed tile blob (as stored in the archive) into geometry.
pub fn decode_tile(compressed: &[u8]) -> io::Result<DecodedTile> {
    let raw = zstd::decode_all(compressed)?;
    decode_uncompressed(&raw)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed tile"))
}

fn decode_uncompressed(raw: &[u8]) -> Option<DecodedTile> {
    let h: TileHeader = pod_at(raw, 0)?;
    let mut off = size_of::<TileHeader>();
    let road_metas: Vec<RoadMeta> = read_vec(raw, &mut off, h.road_count as usize)?;
    let area_metas: Vec<AreaMeta> = read_vec(raw, &mut off, h.area_count as usize)?;
    let ring_metas: Vec<RingMeta> = read_vec(raw, &mut off, h.ring_count as usize)?;
    let label_metas: Vec<LabelMeta> = read_vec(raw, &mut off, h.label_count as usize)?;
    let poi_metas: Vec<PoiMeta> = read_vec(raw, &mut off, h.poi_count as usize)?;

    // Coordinate pool: road chains first, then ring chains (each zig-zag delta).
    let mut road_coords = Vec::with_capacity(road_metas.len());
    for m in &road_metas {
        road_coords.push(read_chain(raw, &mut off, m.vertex_count as usize)?);
    }
    let mut rings: Vec<Vec<[i16; 2]>> = Vec::with_capacity(ring_metas.len());
    let mut ring_prefix = Vec::with_capacity(ring_metas.len() + 1); // vertex offset per ring
    let mut acc = 0usize;
    ring_prefix.push(0);
    for m in &ring_metas {
        rings.push(read_chain(raw, &mut off, m.vertex_count as usize)?);
        acc += m.vertex_count as usize;
        ring_prefix.push(acc);
    }
    let ring_verts = acc;

    let strings = read_strings(raw, &mut off, h.string_count as usize)?;
    let clip_bits = if h.flags & FLAG_RING_CLIP_MASK != 0 {
        read_bits(raw, &mut off, ring_verts)?
    } else {
        Vec::new()
    };

    let name_of = |idx: u16| -> Option<String> {
        (idx != NAME_NONE)
            .then(|| strings.get(idx as usize).cloned())
            .flatten()
    };

    let mut roads = Vec::with_capacity(road_metas.len());
    for (m, coords) in road_metas.iter().zip(road_coords) {
        roads.push(DecRoad {
            kind: RoadKind::from_u8(m.kind)?,
            layer: m.layer,
            lanes: Lanes::new(m.lanes_forward, m.lanes_backward),
            start: EdgeNode::from_u8(m.caps & 0b11)?,
            end: EdgeNode::from_u8((m.caps >> 2) & 0b11)?,
            name: name_of(m.name),
            coords,
        });
    }

    let mut areas = Vec::with_capacity(area_metas.len());
    for m in &area_metas {
        let (fr, rc) = (m.first_ring as usize, m.ring_count as usize);
        let area_rings = rings.get(fr..fr + rc)?.to_vec();
        let clip = if clip_bits.is_empty() {
            Vec::new()
        } else {
            clip_bits
                .get(*ring_prefix.get(fr)?..*ring_prefix.get(fr + rc)?)?
                .to_vec()
        };
        areas.push(DecArea {
            kind: AreaKind::from_u8(m.kind)?,
            layer: m.layer,
            floors: m.floors,
            rings: area_rings,
            clip,
        });
    }

    let mut labels = Vec::with_capacity(label_metas.len());
    for m in &label_metas {
        labels.push(DecLabel {
            class: LabelClass::from_u8(m.class)?,
            rank: m.rank,
            anchor: [m.anchor_x, m.anchor_y],
            name: strings.get(m.name as usize).cloned().unwrap_or_default(),
        });
    }

    let pois = poi_metas
        .iter()
        .map(|m| {
            Some(DecPoi {
                kind: PoiKind::from_u8(m.kind)?,
                anchor: [m.anchor_x, m.anchor_y],
            })
        })
        .collect::<Option<Vec<_>>>()?;

    Some(DecodedTile {
        extent: h.extent,
        roads,
        areas,
        labels,
        pois,
    })
}

impl TilePayload for DecodedTile {
    fn from_tile_bytes(bytes: Vec<u8>) -> Self {
        decode_tile(&bytes).unwrap_or_default()
    }

    fn byte_size(&self) -> usize {
        let roads: usize = self
            .roads
            .iter()
            .map(|r| r.coords.len() * 4 + r.name.as_ref().map_or(0, String::len) + 32)
            .sum();
        let areas: usize = self
            .areas
            .iter()
            .map(|a| a.rings.iter().map(|r| r.len() * 4).sum::<usize>() + a.clip.len() / 8 + 16)
            .sum();
        let labels: usize = self.labels.iter().map(|l| l.name.len() + 16).sum();
        let pois = self.pois.len() * 8;
        roads + areas + labels + pois + 64
    }
}

// ─────────────────────────────── scene assembly ─────────────────────────────

/// All geometry in the current view, lifted into a shared `i32` **scene frame**
/// (tile index × extent at `Scene::target_zoom`) with road segments welded
/// across tile boundaries. Coordinates are absolute; a renderer subtracts the
/// camera centre (in the same frame) before projecting.
pub struct Scene {
    pub target_zoom: u8,
    pub extent: u16,
    roads: Vec<SceneRoad>,
    areas: Vec<SceneArea>,
    labels: Vec<SceneLabel>,
    pois: Vec<ScenePoi>,
}

pub struct SceneRoad {
    pub kind: RoadKind,
    pub layer: i8,
    pub lanes: Lanes,
    pub start: EdgeNode,
    pub end: EdgeNode,
    pub name: Option<String>,
    pub coords: Vec<[i32; 2]>,
}

pub struct SceneArea {
    pub kind: AreaKind,
    pub layer: i8,
    pub floors: u8,
    pub rings: Vec<Vec<[i32; 2]>>,
    pub clip: Vec<bool>,
}

pub struct SceneLabel {
    pub class: LabelClass,
    pub rank: u8,
    pub anchor: [i32; 2],
    pub name: String,
}

pub struct ScenePoi {
    pub kind: PoiKind,
    /// Absolute scene-frame anchor (tile index × extent at `target_zoom`).
    pub anchor: [i32; 2],
}

impl SceneArea {
    /// The area as a `geo::Polygon<f64>` in absolute scene coordinates (ring 0 =
    /// exterior, the rest = holes) — for `geo` algorithms (area, centroid, ...).
    /// Scene `i32` coordinates cast losslessly to `f64`
    pub fn to_polygon(&self) -> Polygon<f64> {
        build_polygon(&self.rings, |[x, y]| Coord {
            x: x as f64,
            y: y as f64,
        })
    }
}

impl From<&SceneArea> for Polygon<f64> {
    fn from(area: &SceneArea) -> Self {
        area.to_polygon()
    }
}

/// Build a `geo::Polygon` from scene rings (ring 0 = exterior, rest = holes),
/// mapping each vertex with `map` and closing each ring (geo wants closed rings;
/// ours are stored open).
fn build_polygon<T: geo::CoordNum>(
    rings: &[Vec<[i32; 2]>],
    map: impl Fn([i32; 2]) -> Coord<T>,
) -> Polygon<T> {
    let to_ring = |ring: &[[i32; 2]]| -> LineString<T> {
        let mut cs: Vec<Coord<T>> = ring.iter().map(|&c| map(c)).collect();
        match (cs.first().copied(), cs.last().copied()) {
            (Some(f), Some(l)) if f != l => cs.push(f),
            _ => {}
        }
        LineString::new(cs)
    };
    match rings.split_first() {
        Some((outer, holes)) => {
            Polygon::new(to_ring(outer), holes.iter().map(|h| to_ring(h)).collect())
        }
        None => Polygon::new(LineString::new(Vec::new()), Vec::new()),
    }
}

/// One geometry yielded from a [`Scene`].
pub enum Feature<'a> {
    Road(&'a SceneRoad),
    Area(&'a SceneArea),
    Label(&'a SceneLabel),
    Poi(&'a ScenePoi),
}

impl Scene {
    /// Update `view` for `cam`, decode + place every visible tile into the scene
    /// frame, and weld cut road segments. Works over any [`TileView`] whose
    /// payload is `DecodedTile` (e.g. `TileCache<S, DecodedTile>`).
    pub async fn from_view<V>(view: &V, cam: &Camera) -> Scene
    where
        V: TileView<DecodedTile> + ?Sized,
    {
        Scene::from_visible(view.update(cam).await, cam.zoom.round() as u8)
    }

    /// Assemble a scene from an already-fetched visible tile set (coordinate
    /// transform into the scene frame + cross-tile road welding). Shapes whose
    /// class isn't shown at `display_zoom` (the live camera zoom) are dropped
    /// before welding, so minor roads / detail features vanish when finer tiles
    /// are viewed at a coarser zoom (overzoom-out)
    pub fn from_visible(visible: Vec<VisibleTile<DecodedTile>>, display_zoom: u8) -> Scene {
        // Target zoom = the exact (non-fallback) tiles' zoom; fall back to any.
        let target_zoom = visible
            .iter()
            .find(|v| v.source == v.tile)
            .or_else(|| visible.first())
            .map(|v| v.tile.z)
            .unwrap_or(0);

        // Never filter below the level of the tiles we actually loaded. The tile
        // level snaps to the nearest materialized zoom (switches at x.0), while
        // `display_zoom` is `round(camera)` (switches at x.5) — so in the half-zoom
        // band between them we'd load, say, z10 tiles yet filter as if at z9,
        // hiding every class materialized at z10 (motorway/trunk/…). Clamping up to
        // `target_zoom` closes that gap; overzoom beyond the finest level still
        // reveals more, since there `display_zoom > target_zoom`.
        let display_zoom = display_zoom.max(target_zoom);

        let mut extent = 0u16;
        let mut weldable: Vec<SceneRoad> = Vec::new(); // exact-tile roads to join
        let mut roads: Vec<SceneRoad> = Vec::new(); // fallback roads (kept as-is)
        let mut areas: Vec<SceneArea> = Vec::new();
        let mut labels: Vec<SceneLabel> = Vec::new();
        let mut pois: Vec<ScenePoi> = Vec::new();

        for v in &visible {
            let d = &v.data;
            extent = d.extent;
            // Coarser fallback tiles are scaled up into the target-zoom frame.
            let shift = target_zoom.saturating_sub(v.source.z) as u32;
            let ext = d.extent as i64;
            let (sx, sy) = (v.source.x as i64, v.source.y as i64);
            let to_scene = |[lx, ly]: [i16; 2]| -> [i32; 2] {
                [
                    ((sx * ext + lx as i64) << shift) as i32,
                    ((sy * ext + ly as i64) << shift) as i32,
                ]
            };
            let exact = v.source == v.tile;

            for r in &d.roads {
                if r.kind.display_min_zoom() > display_zoom {
                    continue; // class not shown at this zoom
                }
                let road = SceneRoad {
                    kind: r.kind,
                    layer: r.layer,
                    lanes: r.lanes,
                    start: r.start,
                    end: r.end,
                    name: r.name.clone(),
                    coords: r.coords.iter().map(|&c| to_scene(c)).collect(),
                };
                // Only same-zoom (exact) tiles weld; fallbacks differ in scale.
                if exact {
                    weldable.push(road)
                } else {
                    roads.push(road)
                }
            }
            for a in &d.areas {
                if a.kind.display_min_zoom() > display_zoom {
                    continue; // class not shown at this zoom
                }
                areas.push(SceneArea {
                    kind: a.kind,
                    layer: a.layer,
                    floors: a.floors,
                    rings: a
                        .rings
                        .iter()
                        .map(|ring| ring.iter().map(|&c| to_scene(c)).collect())
                        .collect(),
                    clip: a.clip.clone(),
                });
            }
            for l in &d.labels {
                if l.class.display_min_zoom() > display_zoom {
                    continue; // class not shown at this zoom
                }
                labels.push(SceneLabel {
                    class: l.class,
                    rank: l.rank,
                    anchor: to_scene(l.anchor),
                    name: l.name.clone(),
                });
            }
            for p in &d.pois {
                if p.kind.display_min_zoom() > display_zoom {
                    continue; // class not shown at this zoom
                }
                pois.push(ScenePoi {
                    kind: p.kind,
                    anchor: to_scene(p.anchor),
                });
            }
        }

        roads.extend(weld_scene_roads(weldable));
        // Areas are emitted per tile — no cross-tile union. Edge-exact clipping
        // makes same-kind neighbours abut seamlessly when filled, and the per-
        // vertex `clip` flags carry the tile-cut edges so a border pass can skip
        // them. Sort by (layer, kind, floors) for a stable, layer-ordered draw
        // sequence (so e.g. water draws under forest, buildings last).
        let mut areas = areas;
        areas.sort_by_key(|a| (a.layer, a.kind as u8, a.floors));
        Scene {
            target_zoom,
            extent,
            roads,
            areas,
            labels,
            pois,
        }
    }

    pub fn roads(&self) -> &[SceneRoad] {
        &self.roads
    }
    pub fn areas(&self) -> &[SceneArea] {
        &self.areas
    }
    pub fn labels(&self) -> &[SceneLabel] {
        &self.labels
    }
    pub fn pois(&self) -> &[ScenePoi] {
        &self.pois
    }

    /// Iterate every geometry in the view (roads, then areas, then labels, then POIs).
    pub fn features(&self) -> impl Iterator<Item = Feature<'_>> {
        self.roads
            .iter()
            .map(Feature::Road)
            .chain(self.areas.iter().map(Feature::Area))
            .chain(self.labels.iter().map(Feature::Label))
            .chain(self.pois.iter().map(Feature::Poi))
    }
}

/// A camera-driven scene producer that rebuilds only when the visible tile
/// composition changes — so a slight pan within the same tiles costs just the
/// (cheap) cache `update` plus a hash, not a full re-transform + re-weld.
pub struct SceneView<V> {
    view: V,
    last_key: Option<u64>,
}

impl<V: TileView<DecodedTile>> SceneView<V> {
    pub fn new(view: V) -> Self {
        Self {
            view,
            last_key: None,
        }
    }

    /// The underlying view (e.g. to inspect the cache).
    pub fn view(&self) -> &V {
        &self.view
    }

    /// Update for `cam` and return a fresh `Scene` only if the visible tile
    /// composition changed since the last call (new tiles, tiles leaving view,
    /// or a fallback upgrading to its exact tile). `None` means the previously
    /// returned scene is still valid — reuse it.
    pub async fn update(&mut self, cam: &Camera) -> Option<Scene> {
        let visible = self.view.update(cam).await;
        // Filter by the live camera zoom, so the rebuild key must include it:
        // crossing a class's min_zoom threshold changes the scene even when the
        // visible tile set is unchanged.
        let display_zoom = cam.zoom.round() as u8;
        let key = visible_key(&visible, display_zoom);
        if self.last_key == Some(key) {
            return None;
        }
        self.last_key = Some(key);
        Some(Scene::from_visible(visible, display_zoom))
    }

    /// Force the next [`update`](Self::update) to rebuild (e.g. after a style
    /// change that isn't reflected in the tile set).
    pub fn invalidate(&mut self) {
        self.last_key = None;
    }
}

/// Hash the visible composition: each slot's tile and the tile its data comes
/// from (so fallback→exact upgrades count as a change). Order is deterministic.
fn visible_key(visible: &[VisibleTile<DecodedTile>], display_zoom: u8) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    display_zoom.hash(&mut h);
    visible.len().hash(&mut h);
    for v in visible {
        v.tile.hash(&mut h);
        v.source.hash(&mut h);
    }
    h.finish()
}

/// Weld road segments split at tile boundaries into continuous polylines. Same
/// rule as the builder: within a group of matching attributes, join where a
/// coordinate is shared by exactly two `Cut` endpoints (so junctions and
/// viewport-edge cuts stay split).
fn weld_scene_roads(roads: Vec<SceneRoad>) -> Vec<SceneRoad> {
    let mut groups: HashMap<(i8, u8, u8, u8, Option<String>), Vec<usize>> = HashMap::new();
    for (i, r) in roads.iter().enumerate() {
        let key = (
            r.layer,
            r.kind as u8,
            r.lanes.forward,
            r.lanes.backward,
            r.name.clone(),
        );
        groups.entry(key).or_default().push(i);
    }
    let mut out = Vec::new();
    for idxs in groups.into_values() {
        weld_group(&roads, &idxs, &mut out);
    }
    out
}

fn weld_group(roads: &[SceneRoad], idxs: &[usize], out: &mut Vec<SceneRoad>) {
    let mut at: HashMap<[i32; 2], Vec<(usize, bool)>> = HashMap::new();
    for (k, &gi) in idxs.iter().enumerate() {
        let r = &roads[gi];
        if r.start == EdgeNode::Cut {
            at.entry(r.coords[0]).or_default().push((k, true));
        }
        if r.end == EdgeNode::Cut {
            at.entry(*r.coords.last().unwrap())
                .or_default()
                .push((k, false));
        }
    }

    let mut used = vec![false; idxs.len()];
    for seed in 0..idxs.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let r0 = &roads[idxs[seed]];
        let mut coords = r0.coords.clone();
        let (mut start, mut end) = (r0.start, r0.end);

        for _ in 0..2 {
            while end == EdgeNode::Cut {
                let tail = *coords.last().unwrap();
                let next = at.get(&tail).and_then(|ports| {
                    (ports.len() == 2).then(|| ports.iter().copied().find(|&(k, _)| !used[k]))?
                });
                let Some((j, j_is_start)) = next else { break };
                used[j] = true;
                let rj = &roads[idxs[j]];
                if j_is_start {
                    coords.extend_from_slice(&rj.coords[1..]);
                    end = rj.end;
                } else {
                    coords.extend(rj.coords[..rj.coords.len() - 1].iter().rev().copied());
                    end = rj.start;
                }
            }
            coords.reverse();
            std::mem::swap(&mut start, &mut end);
        }

        let r = &roads[idxs[seed]];
        out.push(SceneRoad {
            kind: r.kind,
            layer: r.layer,
            lanes: r.lanes,
            start,
            end,
            name: r.name.clone(),
            coords,
        });
    }
}

// ───────────────────────────────── byte helpers ─────────────────────────────

fn pod_at<T: bytemuck::Pod>(raw: &[u8], off: usize) -> Option<T> {
    let end = off.checked_add(size_of::<T>())?;
    raw.get(off..end).map(bytemuck::pod_read_unaligned)
}

fn read_vec<T: bytemuck::Pod>(raw: &[u8], off: &mut usize, n: usize) -> Option<Vec<T>> {
    let size = size_of::<T>();
    let end = off.checked_add(n.checked_mul(size)?)?;
    let bytes = raw.get(*off..end)?;
    *off = end;
    Some(
        bytes
            .chunks_exact(size)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
    )
}

/// Read `n` zig-zag delta `[u16;2]` and prefix-sum them to absolute `[i16;2]`.
fn read_chain(raw: &[u8], off: &mut usize, n: usize) -> Option<Vec<[i16; 2]>> {
    let end = off.checked_add(n.checked_mul(4)?)?;
    let bytes = raw.get(*off..end)?;
    *off = end;
    let (mut x, mut y) = (0i32, 0i32);
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| {
                x += dezigzag(u16::from_le_bytes([c[0], c[1]]));
                y += dezigzag(u16::from_le_bytes([c[2], c[3]]));
                [x as i16, y as i16]
            })
            .collect(),
    )
}

fn read_strings(raw: &[u8], off: &mut usize, n: usize) -> Option<Vec<String>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len_end = off.checked_add(2)?;
        let len = u16::from_le_bytes(raw.get(*off..len_end)?.try_into().ok()?) as usize;
        let s_end = len_end.checked_add(len)?;
        out.push(
            std::str::from_utf8(raw.get(len_end..s_end)?)
                .ok()?
                .to_owned(),
        );
        *off = s_end;
    }
    Some(out)
}

fn read_bits(raw: &[u8], off: &mut usize, n: usize) -> Option<Vec<bool>> {
    let end = off.checked_add(n.div_ceil(8))?;
    let data = raw.get(*off..end)?;
    *off = end;
    Some((0..n).map(|i| (data[i / 8] >> (i % 8)) & 1 == 1).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{Camera, TileCache};
    use crate::{Mercator, Tile};

    #[test]
    fn scene_area_to_geo_polygon() {
        use geo::Area;
        let area = SceneArea {
            kind: AreaKind::Water,
            layer: 0,
            floors: 1,
            rings: vec![
                vec![[0, 0], [10, 0], [10, 10], [0, 10]], // 10×10 outer
                vec![[2, 2], [4, 2], [4, 4], [2, 4]],     // 2×2 hole
            ],
            clip: vec![],
        };
        let poly = area.to_polygon();
        // Rings closed (first == last appended).
        assert_eq!(poly.exterior().0.len(), 5);
        assert_eq!(poly.interiors().len(), 1);
        assert_eq!(poly.interiors()[0].0.len(), 5);
        // Area = outer 100 minus hole 4.
        assert_eq!(poly.unsigned_area(), 96.0);
    }

    /// A source with a (garbage) tile at every coord — decode fails to an empty
    /// tile, which is fine for exercising the change-gate.
    struct Dense;

    #[async_trait::async_trait]
    impl crate::reader::TileSource for Dense {
        async fn tile(&self, t: Tile) -> io::Result<Option<Vec<u8>>> {
            Ok(Some(vec![t.z, t.x as u8, t.y as u8]))
        }
        fn min_zoom(&self) -> u8 {
            0
        }
        fn max_zoom(&self) -> u8 {
            22
        }
    }

    #[tokio::test]
    async fn scene_view_rebuilds_only_on_change() {
        let mut sv = SceneView::new(TileCache::<_, DecodedTile>::new(
            Dense,
            vec![14],
            1 << 20,
            0,
        ));
        let cam = Camera::new(Mercator::new(0.0, 0.0), 14.0, [256.0, 256.0]);

        assert!(sv.update(&cam).await.is_some(), "first call builds");
        assert!(sv.update(&cam).await.is_none(), "unchanged view → reuse");

        let far = Camera::new(
            Mercator::new(5_000_000.0, 5_000_000.0),
            14.0,
            [256.0, 256.0],
        );
        assert!(
            sv.update(&far).await.is_some(),
            "moved to new tiles → rebuild"
        );

        sv.invalidate();
        assert!(
            sv.update(&far).await.is_some(),
            "invalidate forces a rebuild"
        );
    }

    fn scene_road(coords: Vec<[i32; 2]>, start: EdgeNode, end: EdgeNode) -> SceneRoad {
        SceneRoad {
            kind: RoadKind::Primary,
            layer: 0,
            lanes: Lanes::new(1, 1),
            start,
            end,
            name: None,
            coords,
        }
    }

    #[test]
    fn welds_road_pieces_across_a_shared_edge() {
        use EdgeNode::{Cut, Disconnected};
        // Two pieces meeting at the tile edge [8192,50] → one polyline.
        let a = scene_road(vec![[4000, 50], [8192, 50]], Disconnected, Cut);
        let b = scene_road(vec![[8192, 50], [12000, 50]], Cut, Disconnected);
        let welded = weld_scene_roads(vec![a, b]);
        assert_eq!(welded.len(), 1);
        assert_eq!(welded[0].coords, vec![[4000, 50], [8192, 50], [12000, 50]]);
        assert_eq!(
            (welded[0].start, welded[0].end),
            (Disconnected, Disconnected)
        );
    }

    #[test]
    fn leaves_a_lone_viewport_edge_cut_unjoined() {
        use EdgeNode::{Cut, Disconnected};
        // Cut at the viewport edge with no neighbour loaded → stays cut.
        let a = scene_road(vec![[0, 50], [8192, 50]], Disconnected, Cut);
        let welded = weld_scene_roads(vec![a]);
        assert_eq!(welded.len(), 1);
        assert_eq!(welded[0].end, Cut);
    }
}
