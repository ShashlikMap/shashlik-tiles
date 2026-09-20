//! Tile decoding and camera-view scene assembly

use crate::view::{Camera, TilePayload, TileView, VisibleTile};
use geo::{Coord, LineString, Polygon};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
pub use util::feature::{AreaKind, EdgeNode, LabelClass, Lanes, PoiKind, RoadKind, RoadStructure};
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
    pub structure: RoadStructure,
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
            structure: RoadStructure::from_u8((m.caps >> 4) & 0b11)?,
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
    pub start: EdgeNode,
    pub end: EdgeNode,
    pub structure: RoadStructure,
    pub name: Option<String>,
    pub coords: Vec<[i32; 2]>,
    /// Per-vertex lane split (forward/backward), parallel to `coords`. The total
    /// (`forward + backward`) drives the stroke width and varies it smoothly at
    /// lane changes; the split places the direction divider (center line) for
    /// lane-line rendering. `forward == backward == 0` means untagged.
    pub lanes: Vec<Lanes>,
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
        Scene::from_visible(view.update(cam).await, cam.zoom.floor() as u8)
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

        // `display_zoom` is the live camera zoom (see the caller): a class shows
        // once the camera reaches its `display_min_zoom`. It is deliberately NOT
        // clamped up to `target_zoom` — a class materialized at the loaded tile
        // level must still stay hidden until the camera reaches its threshold
        // (e.g. rail is stored in z12 tiles, which the 512px display loads around
        // camera 10, but must not appear until camera 12). Continuity of the
        // backbone (water/land/major roads) is ensured by giving those classes a
        // low `display_min_zoom`, not by clamping.
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
                    start: r.start,
                    end: r.end,
                    structure: r.structure,
                    name: r.name.clone(),
                    lanes: vec![r.lanes; r.coords.len()],
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

        roads.extend(weld_scene_roads(weldable, extent));
        blend_join_widths(&mut roads, extent);
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
        // Class visibility keys off the *raw* camera zoom (floored), so
        // `display_min_zoom = N` means "appears at camera N" — not the effective
        // (tile-selection) zoom. The rebuild key must include it: crossing a
        // class's threshold changes the scene even when the tile set is unchanged.
        let display_zoom = cam.zoom.floor() as u8;
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

/// Fraction of the tile extent used as the endpoint-welding tolerance: two piece
/// ends within `extent / WELD_TOL_DIV` scene units are treated as the same node.
/// At extent 8192 this is 8 units ≈ a couple of metres at the base grid zoom —
/// far below the spacing between distinct or parallel roads (tens of metres), so
/// it only ever stitches a shared node that OSM digitised as two near-coincident
/// points, never merges separate roads.
const WELD_TOL_DIV: i32 = 1024;

/// Weld road segments into continuous polylines — pieces split across tile
/// boundaries, pieces split within a tile (OSM ways sharing an interior node,
/// e.g. at a lane-count change), and pieces whose shared node drifted by sub-metre
/// digitising noise. Within a group of matching geometry attributes, join wherever
/// exactly two piece-ends meet (so real junctions and lone dead-ends / viewport
/// cuts stay split).
fn weld_scene_roads(roads: Vec<SceneRoad>, extent: u16) -> Vec<SceneRoad> {
    // Group by `kind` only — not lanes, name, or **layer**. A road that changes
    // lane count, name, or dips through a bridge / tunnel portal (a layer change)
    // is one continuous carriageway and must weld into a single stroke. Dropping
    // `layer` is safe because welding is connectivity-based: a bridge flying over
    // a road shares no node with it, so only genuinely end-to-end-connected pieces
    // of the same road merge. The welded stroke takes the max layer of its pieces
    // (bridges keep draw-order priority); tunnel styling is a later milestone.
    let tol = (extent as i32 / WELD_TOL_DIV).max(1);
    let mut groups: HashMap<u8, Vec<usize>> = HashMap::new();
    for (i, r) in roads.iter().enumerate() {
        groups.entry(r.kind as u8).or_default().push(i);
    }
    let mut out = Vec::new();
    for idxs in groups.into_values() {
        weld_group(&roads, &idxs, tol, &mut out);
    }
    out
}

/// Minimum cos(turn) for a piece to count as *continuing* a line through a
/// junction (degree > 2). ~0.5 ≈ 60°: a sharper piece is a branch, so the line
/// ends and the branch stays a separate stroke. The straightest candidate always
/// wins, so this only gates the case where no near-straight continuation exists.
const CONTINUATION_MIN_COS: f64 = 0.5;

/// Unit direction from `a` to `b` in scene units.
fn dirf(a: [i32; 2], b: [i32; 2]) -> [f64; 2] {
    let (dx, dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
    let l = (dx * dx + dy * dy).sqrt().max(1e-9);
    [dx / l, dy / l]
}

fn weld_group(roads: &[SceneRoad], idxs: &[usize], tol: i32, out: &mut Vec<SceneRoad>) {
    // Spatial hash of every piece endpoint. Two ends "meet" when they lie within
    // `tol` scene units (chebyshev) of each other; the cell size equals `tol`, so
    // any such pair falls in the same or an adjacent cell. Registering all
    // endpoints (not only tile-boundary `Cut`s) welds interior shared nodes too,
    // and the tolerance stitches sub-metre gaps where a shared node was digitised
    // as two near-coincident points.
    let cell = tol.max(1);
    let key = |c: [i32; 2]| (c[0].div_euclid(cell), c[1].div_euclid(cell));
    // Flat port list: (piece slot in `idxs`, is_start, coord).
    let mut ports: Vec<(usize, bool, [i32; 2])> = Vec::with_capacity(idxs.len() * 2);
    for (k, &gi) in idxs.iter().enumerate() {
        let r = &roads[gi];
        ports.push((k, true, r.coords[0]));
        ports.push((k, false, *r.coords.last().unwrap()));
    }
    let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (pid, &(_, _, c)) in ports.iter().enumerate() {
        cells.entry(key(c)).or_default().push(pid);
    }
    // Port ids within `tol` (chebyshev) of `c`, scanning the 3×3 cell block.
    let cluster = |c: [i32; 2]| -> Vec<usize> {
        let (cx, cy) = key(c);
        let mut hits = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(v) = cells.get(&(cx + dx, cy + dy)) {
                    for &pid in v {
                        let pc = ports[pid].2;
                        if (pc[0] - c[0]).abs().max((pc[1] - c[1]).abs()) <= tol {
                            hits.push(pid);
                        }
                    }
                }
            }
        }
        hits
    };

    let mut used = vec![false; idxs.len()];
    for seed in 0..idxs.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let r0 = &roads[idxs[seed]];
        let mut coords = r0.coords.clone();
        let mut lanes = r0.lanes.clone(); // parallel to coords
        let (mut start, mut end) = (r0.start, r0.end);
        // Pieces may span structural layers (bridge/tunnel portals). The welded
        // stroke draws at whichever layer covers the most of its length (by vertex
        // count), so a mostly-surface road with a short bridge stays at surface
        // level — and a mostly-elevated expressway with a short tunnel stays
        // elevated. Per-section tunnel/bridge styling is a later milestone.
        let mut layer_len: Vec<(i8, usize)> = vec![(r0.layer, r0.coords.len())];

        for _ in 0..2 {
            loop {
                let tail = *coords.last().unwrap();
                let here = cluster(tail);
                // Unused pieces meeting at the tail (the tail's own end is used).
                let cands: Vec<usize> =
                    here.iter().copied().filter(|&p| !used[ports[p].0]).collect();
                if cands.is_empty() {
                    break; // dead end
                }
                let pid = if here.len() <= 2 {
                    // Degree ≤ 2: a single continuation — weld it at any angle, so
                    // both gentle and sharp bends of one road join.
                    cands[0]
                } else {
                    // Degree > 2 (a junction): weld only the piece that best
                    // continues the line (smallest turn) so the road runs through
                    // while branches stay split; if nothing continues closely
                    // enough, the line ends here.
                    let m = coords.len();
                    let incoming = dirf(coords[m - 2], coords[m - 1]);
                    let mut best = None;
                    let mut best_score = CONTINUATION_MIN_COS;
                    for &p in &cands {
                        let (j, j_is_start, _) = ports[p];
                        let rj = &roads[idxs[j]];
                        let nn = rj.coords.len();
                        let out = if j_is_start {
                            dirf(rj.coords[0], rj.coords[1])
                        } else {
                            dirf(rj.coords[nn - 1], rj.coords[nn - 2])
                        };
                        let score = incoming[0] * out[0] + incoming[1] * out[1];
                        if score > best_score {
                            best_score = score;
                            best = Some(p);
                        }
                    }
                    match best {
                        Some(p) => p,
                        None => break,
                    }
                };
                let (j, j_is_start, _) = ports[pid];
                used[j] = true;
                let rj = &roads[idxs[j]];
                // Skip the coincident joining point when it's exact; keep it (a
                // short bridge segment) when the two ends only nearly touch.
                match layer_len.iter_mut().find(|(l, _)| *l == rj.layer) {
                    Some(e) => e.1 += rj.coords.len(),
                    None => layer_len.push((rj.layer, rj.coords.len())),
                }
                if j_is_start {
                    let skip = usize::from(*coords.last().unwrap() == rj.coords[0]);
                    coords.extend_from_slice(&rj.coords[skip..]);
                    lanes.extend_from_slice(&rj.lanes[skip..]);
                    end = rj.end;
                } else {
                    // Piece appended in reverse: reverse its lane order AND swap
                    // forward/backward, since its travel direction is now flipped.
                    let n = rj.coords.len();
                    let take = n - usize::from(*coords.last().unwrap() == rj.coords[n - 1]);
                    coords.extend(rj.coords[..take].iter().rev().copied());
                    lanes.extend(rj.lanes[..take].iter().rev().map(|l| l.swapped()));
                    end = rj.start;
                }
            }
            coords.reverse();
            // Flipping the whole polyline flips every vertex's travel direction.
            lanes.reverse();
            lanes.iter_mut().for_each(|l| *l = l.swapped());
            std::mem::swap(&mut start, &mut end);
        }

        // Draw layer = the one covering the most length (ties → higher layer).
        let layer = layer_len
            .iter()
            .max_by_key(|(l, c)| (*c, *l as i16))
            .map(|&(l, _)| l)
            .unwrap();
        let r = &roads[idxs[seed]];
        out.push(SceneRoad {
            kind: r.kind,
            layer,
            start,
            end,
            structure: r.structure,
            name: r.name.clone(),
            lanes,
            coords,
        });
    }
}

/// Smooth stroke-width discontinuities where the welder correctly leaves same-kind
/// polylines split — real junctions, divided-carriageway Y-splits, bridge/tunnel
/// (layer) transitions. Welding the geometry there would fabricate a through-line
/// or overshoot the node; instead we keep the pieces split and only match their
/// **width** at the shared node: every incident same-kind endpoint is raised to
/// the local maximum lane count, so the variable-width stroke tapers each piece's
/// end segment up to a common join width instead of butting two different-width
/// rectangles together. Endpoint lane counts are read before any are written, so
/// the max is computed from original widths.
fn blend_join_widths(roads: &mut [SceneRoad], extent: u16) {
    let tol = (extent as i32 / WELD_TOL_DIV).max(1);
    let cell = tol.max(1);
    let key = |c: [i32; 2]| (c[0].div_euclid(cell), c[1].div_euclid(cell));
    // Ports: (road idx, is_start). Coord/kind/lanes are looked up from `roads`.
    let mut ports: Vec<(usize, bool)> = Vec::with_capacity(roads.len() * 2);
    for (ri, r) in roads.iter().enumerate() {
        if r.coords.len() >= 2 {
            ports.push((ri, true));
            ports.push((ri, false));
        }
    }
    let coord = |&(ri, start): &(usize, bool)| -> [i32; 2] {
        let r = &roads[ri];
        if start {
            r.coords[0]
        } else {
            *r.coords.last().unwrap()
        }
    };
    // Total lanes (forward + backward) at a port's endpoint — the fill width.
    let lane = |&(ri, start): &(usize, bool)| -> u8 {
        let r = &roads[ri];
        let l = if start { r.lanes[0] } else { *r.lanes.last().unwrap() };
        l.forward.saturating_add(l.backward)
    };
    let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (pid, p) in ports.iter().enumerate() {
        cells.entry(key(coord(p))).or_default().push(pid);
    }

    // Phase 1: compute the target lane count for each port (max among same-kind
    // ports of *other* roads within `tol`, including itself). None = unchanged.
    let mut targets: Vec<Option<u8>> = vec![None; ports.len()];
    for (pid, p) in ports.iter().enumerate() {
        let (c, kind, own) = (coord(p), roads[p.0].kind, lane(p));
        let (cx, cy) = key(c);
        let mut best = own;
        let mut joined = false;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(v) = cells.get(&(cx + dx, cy + dy)) else {
                    continue;
                };
                for &qid in v {
                    let q = &ports[qid];
                    if q.0 == p.0 || roads[q.0].kind != kind {
                        continue; // same road, or a different class — leave the step
                    }
                    let qc = coord(q);
                    if (qc[0] - c[0]).abs().max((qc[1] - c[1]).abs()) > tol {
                        continue;
                    }
                    joined = true;
                    best = best.max(lane(q));
                }
            }
        }
        if joined && best > own {
            targets[pid] = Some(best);
        }
    }

    // Phase 2: apply. Hit the target total by adjusting only `forward` (keeping
    // `backward`), so the widened endpoint's direction divider stays put; the
    // endpoint is junction-adjacent where lane lines fade anyway.
    for (pid, &(ri, start)) in ports.iter().enumerate() {
        if let Some(t) = targets[pid] {
            let r = &mut roads[ri];
            let idx = if start { 0 } else { r.lanes.len() - 1 };
            let b = r.lanes[idx].backward;
            r.lanes[idx].forward = t.saturating_sub(b);
        }
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
            start,
            end,
            structure: RoadStructure::None,
            name: None,
            lanes: vec![Lanes::new(1, 1); coords.len()], // total 2
            coords,
        }
    }

    /// Per-vertex total lane count (forward + backward), for assertions.
    fn totals(r: &SceneRoad) -> Vec<u8> {
        r.lanes.iter().map(|l| l.forward + l.backward).collect()
    }

    #[test]
    fn welds_road_pieces_across_a_shared_edge() {
        use EdgeNode::{Cut, Disconnected};
        // Two pieces meeting at the tile edge [8192,50] → one polyline.
        let a = scene_road(vec![[4000, 50], [8192, 50]], Disconnected, Cut);
        let b = scene_road(vec![[8192, 50], [12000, 50]], Cut, Disconnected);
        let welded = weld_scene_roads(vec![a, b], 8192);
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
        let welded = weld_scene_roads(vec![a], 8192);
        assert_eq!(welded.len(), 1);
        assert_eq!(welded[0].end, Cut);
    }

    #[test]
    fn welds_lane_change_at_an_interior_connected_node() {
        use EdgeNode::{Connected, Disconnected};
        // One road that changes lane count is two OSM ways meeting at an interior
        // Connected node (never a Cut). They must weld into one polyline whose
        // per-vertex lane_counts carry the change, so the stroke tapers smoothly.
        let mut a = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        a.lanes = vec![Lanes::new(2, 0); 2];
        let mut b = scene_road(vec![[100, 0], [200, 0]], Connected, Disconnected);
        b.lanes = vec![Lanes::new(3, 0); 2];
        let welded = weld_scene_roads(vec![a, b], 8192);
        assert_eq!(welded.len(), 1, "adjacent ways must weld into one stroke");
        assert_eq!(welded[0].coords, vec![[0, 0], [100, 0], [200, 0]]);
        assert_eq!(welded[0].lanes.len(), welded[0].coords.len());
        // The lane count steps at the shared node — width ramps across the segment.
        assert_eq!(totals(&welded[0]), vec![2, 2, 3]);
    }

    #[test]
    fn weld_keeps_lane_direction_consistent_when_reversing() {
        use EdgeNode::{Connected, Disconnected};
        // Two halves of one asymmetric road (2 forward / 1 backward). Piece B is
        // stored in the opposite direction, so the welder appends it reversed — its
        // forward/backward must swap so every welded vertex names the same physical
        // side (otherwise the center line flips mid-road / with the weld orientation).
        let mut a = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        a.lanes = vec![Lanes::new(2, 1); 2];
        let mut b = scene_road(vec![[200, 0], [100, 0]], Disconnected, Connected);
        b.lanes = vec![Lanes::new(1, 2); 2]; // stored reversed → split swapped
        let welded = weld_scene_roads(vec![a, b], 8192);
        assert_eq!(welded.len(), 1);
        let first = welded[0].lanes[0];
        assert!(
            welded[0].lanes.iter().all(|&l| l == first),
            "split stays consistent along the weld: {:?}",
            welded[0].lanes
        );
        assert_eq!(first.forward + first.backward, 3);
        assert_ne!(first.forward, first.backward, "asymmetric split preserved");
    }

    #[test]
    fn welds_the_through_line_at_a_junction() {
        use EdgeNode::{Connected, Disconnected};
        // A straight road (a—b) with a branch (c) at [100,0]: the straight line
        // welds through the junction; the perpendicular branch stays split.
        let a = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        let b = scene_road(vec![[100, 0], [200, 0]], Connected, Disconnected);
        let c = scene_road(vec![[100, 0], [100, 100]], Connected, Disconnected);
        let welded = weld_scene_roads(vec![a, b, c], 8192);
        assert_eq!(welded.len(), 2, "through-line welds, branch stays split");
        let through = welded.iter().find(|r| r.coords.len() == 3).unwrap();
        assert_eq!(through.coords, vec![[0, 0], [100, 0], [200, 0]]);
    }

    #[test]
    fn welds_both_lines_through_a_crossing() {
        use EdgeNode::{Connected, Disconnected};
        // Two straight same-kind roads crossing at [0,0]: each welds through, so
        // the result is two crossing polylines with no butt-cap gap at the node.
        let ew1 = scene_road(vec![[-100, 0], [0, 0]], Disconnected, Connected);
        let ew2 = scene_road(vec![[0, 0], [100, 0]], Connected, Disconnected);
        let ns1 = scene_road(vec![[0, -100], [0, 0]], Disconnected, Connected);
        let ns2 = scene_road(vec![[0, 0], [0, 100]], Connected, Disconnected);
        let welded = weld_scene_roads(vec![ew1, ew2, ns1, ns2], 8192);
        assert_eq!(welded.len(), 2, "each straight line welds through the crossing");
        assert!(welded.iter().all(|r| r.coords.len() == 3));
    }

    #[test]
    fn stitches_a_sub_tolerance_gap() {
        use EdgeNode::{Connected, Disconnected};
        // A shared node digitised as two near-coincident points (5 units apart,
        // under the 8-unit tolerance at extent 8192) must still weld into one
        // stroke, bridging the gap.
        let a = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        let b = scene_road(vec![[105, 0], [200, 0]], Connected, Disconnected);
        let welded = weld_scene_roads(vec![a, b], 8192);
        assert_eq!(welded.len(), 1, "sub-tolerance gap stitches");
        // Bridge segment kept: both near-coincident points are present.
        assert_eq!(welded[0].coords, vec![[0, 0], [100, 0], [105, 0], [200, 0]]);
    }

    #[test]
    fn leaves_distinct_roads_beyond_tolerance_split() {
        use EdgeNode::{Connected, Disconnected};
        // Ends 40 units apart (well beyond the 8-unit tolerance) are distinct
        // roads — e.g. parallel carriageways — and must not merge.
        let a = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        let b = scene_road(vec![[100, 40], [200, 40]], Connected, Disconnected);
        let welded = weld_scene_roads(vec![a, b], 8192);
        assert_eq!(welded.len(), 2, "gap beyond tolerance stays split");
    }

    #[test]
    fn blend_matches_width_at_a_split_join() {
        use EdgeNode::{Connected, Disconnected};
        // A 2-lane piece and a 1-lane piece of the same kind meet at a node the
        // welder left split (here modelled as two separate roads). Their widths
        // must match at the join: the 1-lane end is raised to 2, the 2-lane end
        // stays 2. Interior vertices are untouched.
        let mut wide = scene_road(vec![[0, 0], [50, 0], [100, 0]], Disconnected, Connected);
        wide.lanes = vec![Lanes::new(2, 0); 3];
        let mut thin = scene_road(vec![[100, 0], [150, 0], [200, 0]], Connected, Disconnected);
        thin.lanes = vec![Lanes::new(1, 0); 3];
        let mut roads = vec![wide, thin];
        blend_join_widths(&mut roads, 8192);
        assert_eq!(totals(&roads[0]), vec![2, 2, 2], "wide end unchanged");
        assert_eq!(
            totals(&roads[1]),
            vec![2, 1, 1],
            "thin join end raised to match; interior untouched"
        );
    }

    #[test]
    fn blend_leaves_different_kinds_alone() {
        use EdgeNode::{Connected, Disconnected};
        // A minor road ending into a major road of a different kind keeps its own
        // width — no ballooning at the junction.
        let mut major = scene_road(vec![[0, 0], [100, 0]], Disconnected, Connected);
        major.kind = RoadKind::Primary;
        major.lanes = vec![Lanes::new(2, 2); 2];
        let mut minor = scene_road(vec![[100, 0], [200, 0]], Connected, Disconnected);
        minor.kind = RoadKind::Service;
        minor.lanes = vec![Lanes::new(1, 0); 2];
        let mut roads = vec![major, minor];
        blend_join_widths(&mut roads, 8192);
        assert_eq!(totals(&roads[1]), vec![1, 1], "cross-kind join not blended");
    }
}
