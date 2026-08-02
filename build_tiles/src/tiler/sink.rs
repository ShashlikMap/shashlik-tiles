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
use super::{TILE_RENDER_PX, TileParams, grid};
use crate::shapes::{Area, AreaKind, EdgeNode, Label, Road, Shape};
use crate::sink::ShapeSink;
use geo::{Coord, MapCoords, SimplifyVw};
use std::io;
use std::path::Path;

/// Coarsest zoom at which a shape is materialized from raw per-feature geometry.
const AGG_MAX_ZOOM: u8 = 8;

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
}

impl TileSink {
    /// Create a sink spilling under `dir`, materializing `zooms`. Routing to spill
    /// buckets uses `base` (its `bucket_zoom`); each zoom's clip uses grid params
    /// derived from `base` with that zoom.
    pub fn new(dir: impl AsRef<Path>, base: TileParams, zooms: Vec<u8>) -> Self {
        Self {
            spill: SpillWriter::new(dir, base),
            base,
            zooms,
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
            self.base.margin,
            self.base.simplify_px,
        )
    }

    /// Simplification tolerance (fractional-tile units) and whether this zoom is
    /// kept full-detail (the finest / base grid zoom).
    #[inline]
    fn simplify(&self, zp: &TileParams, z: u8) -> (f64, bool) {
        let eps = zp.simplify_tolerance() / zp.extent as f64;
        (eps, z == self.base.grid_zoom)
    }

    fn push_area(&self, area: &Area, z: u8) {
        if z < area.kind.min_zoom() {
            return; // class not shown at this (coarse) zoom
        }
        // Aggregated kinds are materialized from the mask at coarse zooms; the raw
        // geometry is only used at finer ones (see `push_aggregated`).
        if is_aggregated(area.kind) && z <= AGG_MAX_ZOOM {
            return;
        }
        self.clip_area_at(area, z);
    }

    /// Clip the merged geometry of an aggregated kind into the coarse tiles
    pub fn push_aggregated(&self, area: &Area) {
        for &z in &self.zooms {
            if z > AGG_MAX_ZOOM || z < area.kind.min_zoom() {
                continue;
            }
            self.clip_area_at(area, z);
        }
    }

    /// Project → simplify → clip → spill one area at one zoom (no visibility
    /// policy; callers gate the zoom).
    fn clip_area_at(&self, area: &Area, z: u8) {
        let zp = self.zoom_params(z);
        let (eps, full_detail) = self.simplify(&zp, z);
        let min_area = 4.0 * (self.base.extent as f64 / TILE_RENDER_PX).powi(2);

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
            .filter(|h| polygon_area(h) > 20.0 / (512.0 * 512.0))
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
        let zp = self.zoom_params(z);
        let (eps, full_detail) = self.simplify(&zp, z);

        let world = project(&road.geometry, &zp);
        let simp = if full_detail {
            world
        } else {
            world.simplify_vw(eps * eps)
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
}

impl ShapeSink for TileSink {
    fn push(&self, shape: Shape) {
        for &z in &self.zooms {
            match &shape {
                Shape::Area(area) => self.push_area(area, z),
                Shape::Road(road) => self.push_road(road, z),
                Shape::Label(label) => self.push_label(label, z),
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
