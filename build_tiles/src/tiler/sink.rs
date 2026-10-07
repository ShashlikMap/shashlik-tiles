//! Tiler sink: clip each streamed shape into every materialized zoom and
//! spill the resulting per-tile records to the bucketed spill.
//!
//! Implements `ShapeSink` so the OSM extractor — and the water pre-merge step —
//! can stream shapes in concurrently. Per `push`, the shape is projected,
//! simplified, and clipped once per zoom (the base grid zoom is kept
//! full-detail), and one record per covered tile is spilled. No cascade: every
//! zoom derives straight from the source shape.

use super::record::{TileGeometry, TileRecord};
use super::spill::SpillWriter;
use super::{TileParams, grid};
use crate::shapes::{Area, AreaKind, EdgeNode, Label, Poi, Road, Shape};
use crate::sink::ShapeSink;
use geo::{Coord, LineString, MapCoords, SimplifyVw};
use std::io;
use std::path::Path;

/// Whether a kind is materialized from the global aggregation mask at coarse
#[inline]
fn is_aggregated(kind: AreaKind) -> bool {
    matches!(kind, AreaKind::Forest)
}

/// Streaming tiler: shape stream -> per-tile spill records, across all
/// materialized zooms.
pub struct TileSink {
    /// Extent / margin / bucket config; each zoom derives its own grid params.
    base: TileParams,
    /// Materialized zoom levels (client overzooms the gaps).
    zooms: Vec<u8>,
    spill: SpillWriter,
    /// Coarsest zoom at which a shape is materialized from raw per-feature
    /// geometry (areas). Was raised from 8 to 10 to fix a busy z10 band: at
    /// the old boundary, raw (unmerged) forest switched on right at z10 —
    /// the same tile level `target_level` snaps to across most of the
    /// camera 8-10 range — dumping thousands of small polygons in at once.
    /// Keeping the light aggregated forest through this zoom defers that to
    /// the next finer materialized zoom.
    agg_max_zoom: u8,
    /// Merged major roads (`RoadKind::MajorRoad`, from `route_roads`) are
    /// clipped into tiles up to this zoom; raw `Motorway`/`Trunk`/`Primary`
    /// ways take over starting at their own `RoadKind::min_zoom()` (`10`),
    /// regardless of this value -- see `major_roads.rs`'s module doc
    /// comment for why the two ranges *touch* at `z == 10` by design, and
    /// what widening this past `10` actually changes (extends the touching
    /// range rather than removing it).
    road_agg_max_zoom: u8,
}

impl TileSink {
    /// Create a sink spilling under `dir`, materializing `zooms`. Routing to spill
    /// buckets uses `base` (its `bucket_zoom`); each zoom's clip uses grid params
    /// derived from `base` with that zoom.
    pub fn new(
        dir: impl AsRef<Path>,
        base: TileParams,
        zooms: Vec<u8>,
        agg_max_zoom: u8,
        road_agg_max_zoom: u8,
    ) -> Self {
        Self {
            spill: SpillWriter::new(dir, base),
            base,
            zooms,
            agg_max_zoom,
            road_agg_max_zoom,
        }
    }

    /// Per-zoom grid params: grid zoom = `z`, `bucket_zoom` 0 (bucket routing is
    /// the `SpillWriter`'s job, via `base`).
    #[inline]
    fn zoom_params(&self, z: u8) -> TileParams {
        TileParams::new(
            z,
            self.base.min_zoom,
            0,
            self.base.extent,
            self.base.simplify_px,
            self.base.tile_render_px,
        )
    }

    /// Simplification tolerance (fractional-tile units) and whether this zoom is
    /// kept full-detail -- always at the base/finest grid zoom, or when
    /// `force_full_detail` says so (see `clip_area_at`'s `AreaKind::Building`
    /// exemption).
    #[inline]
    fn simplify(&self, zp: &TileParams, z: u8, force_full_detail: bool) -> (f64, bool) {
        if force_full_detail || z == self.base.grid_zoom {
            return (0.0, true);
        }
        let simplify_px = self.base.simplify_px;
        let full_detail = simplify_px <= 0.0;
        let eps = if full_detail {
            0.0
        } else {
            zp.simplify_tolerance(simplify_px) / zp.extent as f64
        };
        (eps, full_detail)
    }

    fn push_area(&self, area: &Area, z: u8) {
        if z < area.kind.min_zoom() {
            return; // class not shown at this (coarse) zoom
        }
        // Aggregated kinds are materialized from the mask at coarse zooms; the raw
        // geometry is only used at finer ones (see `push_aggregated`).
        if is_aggregated(area.kind) && z <= self.agg_max_zoom {
            return;
        }
        self.clip_area_at(area, z);
    }

    /// Clip the merged geometry of an aggregated kind into the coarse tiles
    pub fn push_aggregated(&self, area: &Area) {
        for &z in &self.zooms {
            if z > self.agg_max_zoom || z < area.kind.min_zoom() {
                continue;
            }
            self.clip_area_at(area, z);
        }
    }

    /// Project → simplify → clip → spill one area at one zoom (no visibility
    /// policy; callers gate the zoom).
    fn clip_area_at(&self, area: &Area, z: u8) {
        let zp = self.zoom_params(z);
        // Buildings are exempt from simplification unconditionally -- a
        // real, hardcoded exemption (not another tunable constant), see
        // `simplify`'s own doc comment for why. Visvalingam-Whyatt's one
        // fixed absolute tolerance devastates small building footprints
        // (often collapsing them to bare triangles) even at a tolerance
        // that barely dents a large land/water/forest polygon or a road.
        let (eps, full_detail) = self.simplify(&zp, z, area.kind == AreaKind::Building);
        let min_area = 4.0 * (self.base.extent as f64 / self.base.tile_render_px).powi(2);

        let world = project(&area.geometry, &zp);
        // Simplify the whole polygon once per zoom so every tile cuts from the
        // same cleaned geometry (seamless); the base grid zoom stays full-detail.
        let simp = if full_detail {
            world
        } else {
            world.simplify_vw(eps * eps)
        };

        let ext_ring = simp.exterior();
        if ext_ring.0.len() < 3 {
            return;
        }
        let outer: Vec<[f64; 2]> = ext_ring.0.iter().map(|c| [c.x, c.y]).collect();
        if outer.len() < 3 {
            return;
        }
        let holes: Vec<Vec<[f64; 2]>> = simp
            .interiors()
            .iter()
            .map(|r| r.0.iter().map(|c| [c.x, c.y]).collect())
            .filter(|h: &Vec<[f64; 2]>| h.len() >= 3)
            .filter(|h| polygon_area(h) > 20.0 / (self.base.tile_render_px * self.base.tile_render_px))
            .collect();

        grid::clip_area(&outer, &holes, &zp, min_area, &mut |tx, ty, piece| {
            self.spill.append(TileRecord {
                tile_key: zp.tile_id(tx, ty, z),
                layer: area.layer,
                geometry: TileGeometry::Area {
                    kind: area.kind,
                    floors: area.floors,
                    rings: piece.rings,
                    clip: piece.clip,
                },
            });
        });
    }

    fn push_road(&self, road: &Road, z: u8) {
        if z < road.kind.min_zoom() {
            return; // class not shown at this (coarse) zoom
        }
        self.clip_road_at(road, z);
    }

    /// Clip a merged **major** road (continuous line assembled from many ways)
    /// into the coarse tiles (`z <= self.road_agg_max_zoom`) — the counterpart
    /// to the raw-way path in [`push_road`], which runs starting at
    /// `RoadKind::min_zoom()` (`10` for `Motorway`/`Trunk`/`Primary`) --
    /// see `major_roads.rs`'s module doc comment for why the two ranges
    /// *touch* at `z == 10` rather than being strictly disjoint. Called once
    /// per merged line after extraction.
    pub fn push_aggregated_road(&self, road: &Road) {
        for &z in &self.zooms {
            if z > self.road_agg_max_zoom {
                continue;
            }
            self.clip_road_at(road, z);
        }
    }

    /// Project → simplify → clip → spill one road at one zoom (no visibility
    /// policy; callers gate the zoom).
    fn clip_road_at(&self, road: &Road, z: u8) {
        let zp = self.zoom_params(z);
        // Full detail only at the base/finest zoom -- a hardcoded
        // exemption, independent of whatever `simplify_px` is set to. Every
        // other (non-base) zoom now simplifies roads too -- see
        // `simplify_road_segments`'s doc comment for how it avoids
        // reintroducing the previously-confirmed, real-hardware
        // network-disconnection bug that used to make this an
        // unconditional exemption for roads at every zoom.
        let force_full_detail = z == self.base.grid_zoom;
        let (eps, full_detail) = self.simplify(&zp, z, force_full_detail);

        let world = project(&road.geometry, &zp);
        let simp = if full_detail {
            world
        } else {
            simplify_road_segments(&world.0, &road.junctions, eps * eps)
        };
        if simp.0.len() < 2 {
            return;
        }
        let pts: Vec<[f64; 2]> = simp.0.iter().map(|c| [c.x, c.y]).collect();

        grid::clip_line(&pts, &zp, &mut |tx, ty, piece| {
            // Clip-created endpoints become Cut so the renderer continues the line
            // into the neighbour; true ends keep the road's own node state.
            let start = if piece.start_cut {
                EdgeNode::Cut
            } else {
                road.start
            };
            let end = if piece.end_cut {
                EdgeNode::Cut
            } else {
                road.end
            };
            self.spill.append(TileRecord {
                tile_key: zp.tile_id(tx, ty, z),
                layer: road.layer,
                geometry: TileGeometry::Road {
                    kind: road.kind,
                    start,
                    end,
                    coords: piece.points,
                    lanes_forward: road.lanes.forward,
                    lanes_backward: road.lanes.backward,
                    name: road.name.clone(),
                    structure: road.structure,
                },
            });
        });
    }

    fn push_label(&self, label: &Label, z: u8) {
        if z < label.class.min_zoom() {
            return;
        }
        let zp = self.zoom_params(z);
        let (fx, fy) = zp.fractional(label.anchor);
        grid::clip_point([fx, fy], &zp, &mut |tx, ty, anchor| {
            self.spill.append(TileRecord {
                tile_key: zp.tile_id(tx, ty, z),
                layer: 0,
                geometry: TileGeometry::Label {
                    class: label.class,
                    anchor,
                    name: label.name.clone(),
                },
            });
        });
    }

    fn push_poi(&self, poi: &Poi, z: u8) {
        if z < poi.kind.min_zoom() {
            return;
        }
        let zp = self.zoom_params(z);
        let (fx, fy) = zp.fractional(poi.anchor);
        grid::clip_point([fx, fy], &zp, &mut |tx, ty, anchor| {
            self.spill.append(TileRecord {
                tile_key: zp.tile_id(tx, ty, z),
                layer: 0,
                geometry: TileGeometry::Poi {
                    kind: poi.kind,
                    anchor,
                },
            });
        });
    }
}

impl ShapeSink for TileSink {
    fn push(&self, shape: Shape) {
        for &z in &self.zooms {
            match &shape {
                Shape::Area(area) => self.push_area(area, z),
                Shape::Road(road) => self.push_road(road, z),
                Shape::Label(label) => self.push_label(label, z),
                Shape::Poi(poi) => self.push_poi(poi, z),
            }
        }
    }

    fn finish(&self) -> io::Result<()> {
        self.spill.flush();
        Ok(())
    }
}

/// Mercator -> fractional-tile projection. Kept small-magnitude (≤ 2^z) so the
/// boolean ops stay numerically robust; scaling to tile-local happens at the leaf.
#[inline]
fn project<G: MapCoords<f64, f64, Output = G>>(geom: &G, zp: &TileParams) -> G {
    geom.map_coords(|c| {
        let (fx, fy) = zp.fractional(c);
        Coord { x: fx, y: fy }
    })
}

/// Simplifies a projected road line (`points`) for a non-base zoom without
/// reintroducing the network-disconnection bug plain whole-line
/// Visvalingam-Whyatt simplification caused (see `clip_road_at`'s doc
/// comment): splits `points` into sub-segments at every index flagged in
/// `junctions` (one-to-one with `points`, see `Road::junctions`'s doc
/// comment), simplifies each sub-segment independently, then rejoins them.
/// This works because VW never moves or drops a line's own first/last
/// point -- splitting at a junction turns it into the shared boundary
/// between two segments, so it's preserved as *both* segments' endpoint
/// rather than being just another droppable interior point of one long
/// line.
///
/// Falls back to simplifying `points` as a single line (no junction
/// protection) when `junctions` doesn't line up with `points` -- either
/// genuinely empty (a merged/synthetic stroke with no per-point node ids,
/// see `Road::junctions`'s doc comment) or, defensively, any other length
/// mismatch that would otherwise index out of bounds below.
fn simplify_road_segments(points: &[Coord], junctions: &[bool], eps_sq: f64) -> LineString {
    if junctions.len() != points.len() || points.len() < 2 {
        return LineString::from(points.to_vec()).simplify_vw(eps_sq);
    }

    let mut out: Vec<Coord> = Vec::with_capacity(points.len());
    let mut start = 0usize;
    for i in 1..points.len() {
        if junctions[i] || i == points.len() - 1 {
            let segment = &points[start..=i];
            let simplified: Vec<Coord> = if segment.len() < 3 {
                segment.to_vec()
            } else {
                LineString::from(segment.to_vec()).simplify_vw(eps_sq).0
            };
            if out.is_empty() {
                out.extend(simplified);
            } else {
                // Skip this segment's first point -- it's the same
                // junction point `out`'s own last entry already holds,
                // the shared boundary with the previous segment.
                out.extend(simplified.into_iter().skip(1));
            }
            start = i;
        }
    }
    LineString::from(out)
}

/// Shoelace area of a ring (fractional-tile units^2), used to drop tiny holes.
fn polygon_area(points: &[[f64; 2]]) -> f64 {
    let n = points.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut j = n - 1;
    for i in 0..n {
        sum += (points[j][0] + points[i][0]) * (points[j][1] - points[i][1]);
        j = i;
    }
    sum.abs() * 0.5
}

#[cfg(test)]
mod simplify_road_segments_tests {
    use super::*;

    #[test]
    fn without_junction_data_middle_point_can_be_dropped() {
        // Exactly collinear -- zero triangle area, so VW drops the middle
        // point at any positive epsilon when nothing protects it.
        let points = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
        ];
        let simplified = simplify_road_segments(&points, &[false, false, false], 1.0);
        assert_eq!(simplified.0.len(), 2);
    }

    #[test]
    fn junction_point_survives_even_though_it_would_otherwise_be_dropped() {
        // Same exactly-collinear line, but the middle point is now a
        // junction -- another road's endpoint sits exactly there, so it
        // must survive regardless of how geometrically "unimportant" VW
        // alone would consider it.
        let points = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
        ];
        let simplified = simplify_road_segments(&points, &[false, true, false], 1.0);
        assert_eq!(simplified.0, points);
    }

    #[test]
    fn mismatched_junction_length_falls_back_to_whole_line_simplify() {
        let points = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 2.0, y: 0.0 },
        ];
        let simplified = simplify_road_segments(&points, &[], 1.0);
        assert_eq!(simplified.0.len(), 2);
    }
}
