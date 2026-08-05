//! Clip whole shapes into a tile grid

use crate::tiler::TileParams;
use crate::tiler::clip::clip_line as clip_polyline;
use geo::{BooleanOps, Coord, LineString, Polygon};

/// One polygon clipped to one tile. `rings[0]` is the outer ring, the rest are
/// holes, all in tile-local `i16`. `clip` is parallel to the rings' vertices
/// (concatenated in ring order): `true` where a vertex lies on the tile boundary
/// (an artificial cut edge the renderer should not outline).
#[derive(Debug, Clone, PartialEq)]
pub struct AreaPiece {
    pub rings: Vec<Vec<[i16; 2]>>,
    pub clip: Vec<bool>,
}

/// One polyline piece clipped to one tile, in tile-local `i16`. `start_cut` /
/// `end_cut` mark endpoints created by the clip (tile boundary) rather than the
/// road's true ends.
#[derive(Debug, Clone, PartialEq)]
pub struct LinePiece {
    pub points: Vec<[i16; 2]>,
    pub start_cut: bool,
    pub end_cut: bool,
}

/// Clip a polygon (`outer` + `holes`, fractional-tile coords) into the tile grid,
/// invoking `emit(x, y, piece)` once per surviving polygon per covered tile.
/// `min_area` drops clipped pieces whose bbox area (tile-local units^2) is below
/// it — pass `0.0` to keep every piece.
pub fn clip_area(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    zp: &TileParams,
    min_area: f64,
    emit: &mut dyn FnMut(u32, u32, AreaPiece),
) {
    if outer.len() < 3 {
        return;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in outer {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let range = zp.tiles_covering(x0, y0, x1, y1);
    clip_area_range(outer, holes, range, zp, min_area, emit);
}

/// Recursive worker for `clip_area`: quadtree-split the tile `range`, SH-reduce
/// into each half, and i_overlay-clip at single-tile leaves.
fn clip_area_range(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    (tx0, ty0, tx1, ty1): (u32, u32, u32, u32),
    zp: &TileParams,
    min_area: f64,
    emit: &mut dyn FnMut(u32, u32, AreaPiece),
) {
    let extent = zp.extent as f64;

    if tx0 == tx1 && ty0 == ty1 {
        // Exact tile rect in fractional coords (edge-exact, no buffer), and the
        // local-frame transform.
        let (rx0, ry0, rx1, ry1) = (
            tx0 as f64,
            ty0 as f64,
            (tx0 + 1) as f64,
            (ty0 + 1) as f64,
        );
        let (ox, oy) = (tx0 as f64, ty0 as f64);
        let to_local = |ring: &[[f64; 2]]| -> Vec<[i16; 2]> {
            let mut v: Vec<[i16; 2]> = ring
                .iter()
                .map(|p| zp.quantize([(p[0] - ox) * extent, (p[1] - oy) * extent]))
                .collect();
            if v.len() >= 2 && v.first() == v.last() {
                v.pop();
            }
            canonicalize(v)
        };

        // Prune: parent bbox vs this tile's buffered rect.
        let (mut bx0, mut by0, mut bx1, mut by1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in outer {
            bx0 = bx0.min(p[0]);
            by0 = by0.min(p[1]);
            bx1 = bx1.max(p[0]);
            by1 = by1.max(p[1]);
        }
        if bx1 < rx0 || bx0 > rx1 || by1 < ry0 || by0 > ry1 {
            return;
        }

        let ls = |pts: &[[f64; 2]]| -> LineString {
            LineString::from(pts.iter().map(|&[x, y]| Coord { x, y }).collect::<Vec<_>>())
        };
        let subject = Polygon::new(ls(outer), holes.iter().map(|h| ls(h)).collect());
        let rect = Polygon::new(
            ls(&[[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1], [rx0, ry0]]),
            Vec::new(),
        );

        let lo = 0i16;
        let hi = zp.extent as i16;
        for p in subject.intersection(&rect).0 {
            let ext: Vec<[f64; 2]> = p.exterior().0.iter().map(|c| [c.x, c.y]).collect();
            let e = to_local(&ext);
            if e.len() < 3 || bbox_area(&e) < min_area {
                continue;
            }
            let mut rings = vec![e];
            for h in p.interiors() {
                let hv: Vec<[f64; 2]> = h.0.iter().map(|c| [c.x, c.y]).collect();
                let hr = to_local(&hv);
                if hr.len() >= 3 {
                    rings.push(hr);
                }
            }
            let clip: Vec<bool> = rings
                .iter()
                .flatten()
                .map(|v| v[0] == lo || v[0] == hi || v[1] == lo || v[1] == hi)
                .collect();
            emit(tx0, ty0, AreaPiece { rings, clip });
        }
        return;
    }

    // Split the longer axis; recurse into each non-empty half.
    let halves = if tx1 - tx0 >= ty1 - ty0 {
        let mid = tx0 + (tx1 - tx0) / 2;
        [(tx0, ty0, mid, ty1), (mid + 1, ty0, tx1, ty1)]
    } else {
        let mid = ty0 + (ty1 - ty0) / 2;
        [(tx0, ty0, tx1, mid), (tx0, mid + 1, tx1, ty1)]
    };
    for (ax0, ay0, ax1, ay1) in halves {
        if ax0 == ax1 && ay0 == ay1 {
            // Single-tile half: pass the PARENT geometry unclipped so the leaf
            // i_overlay-clips the same input its sibling does (seamless).
            clip_area_range(outer, holes, (ax0, ay0, ax1, ay1), zp, min_area, emit);
            continue;
        }
        // Multi-tile half: SH-reduce to shrink the geometry before recursing.
        let (rx0, ry0, rx1, ry1) = (
            ax0 as f64,
            ay0 as f64,
            (ax1 + 1) as f64,
            (ay1 + 1) as f64,
        );
        let so = sh_rect(outer, rx0, ry0, rx1, ry1);
        if so.len() < 3 {
            continue;
        }
        let sh: Vec<Vec<[f64; 2]>> = holes
            .iter()
            .map(|h| sh_rect(h, rx0, ry0, rx1, ry1))
            .filter(|h| h.len() >= 3)
            .collect();
        clip_area_range(&so, &sh, (ax0, ay0, ax1, ay1), zp, min_area, emit);
    }
}

/// Clip a polyline (`points`, fractional-tile coords) into the tile grid,
/// emitting each visible piece per covered tile. Roads are assumed local (a small
/// covering range); the whole line is transformed per candidate tile, which is
/// fine for bounded features but O(tiles × len).
pub fn clip_line(points: &[[f64; 2]], zp: &TileParams, emit: &mut dyn FnMut(u32, u32, LinePiece)) {
    if points.len() < 2 {
        return;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in points {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let (tx0, ty0, tx1, ty1) = zp.tiles_covering(x0, y0, x1, y1);
    let extent = zp.extent as f64;
    let rect = zp.clip_rect(); // local frame: [0, extent]^2

    for ty in ty0..=ty1 {
        for tx in tx0..=tx1 {
            let (ox, oy) = (tx as f64, ty as f64);
            let local: Vec<[f64; 2]> = points
                .iter()
                .map(|p| [(p[0] - ox) * extent, (p[1] - oy) * extent])
                .collect();
            for piece in clip_polyline(&local, rect) {
                let pts: Vec<[i16; 2]> = piece.points.iter().map(|&p| zp.quantize(p)).collect();
                if pts.len() >= 2 {
                    emit(
                        tx,
                        ty,
                        LinePiece {
                            points: pts,
                            start_cut: piece.start_cut,
                            end_cut: piece.end_cut,
                        },
                    );
                }
            }
        }
    }
}

/// "Clip" a point label: emit its tile-local anchor in the single tile that
/// contains it. Never split or dropped.
pub fn clip_point(anchor: [f64; 2], zp: &TileParams, emit: &mut dyn FnMut(u32, u32, [i16; 2])) {
    let last = (1u32 << zp.grid_zoom) - 1;
    let tx = (anchor[0].floor().max(0.0) as u32).min(last);
    let ty = (anchor[1].floor().max(0.0) as u32).min(last);
    let extent = zp.extent as f64;
    let local = zp.quantize([
        (anchor[0] - tx as f64) * extent,
        (anchor[1] - ty as f64) * extent,
    ]);
    emit(tx, ty, local);
}

/// Sutherland–Hodgman clip of a ring against an axis-aligned rectangle (used only
/// to reduce geometry while recursing — the leaf cut is i_overlay).
fn sh_rect(ring: &[[f64; 2]], xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<[f64; 2]> {
    let a = sh_pass(ring, 0, xmin, true);
    let b = sh_pass(&a, 0, xmax, false);
    let c = sh_pass(&b, 1, ymin, true);
    sh_pass(&c, 1, ymax, false)
}

fn sh_pass(input: &[[f64; 2]], axis: usize, bound: f64, keep_ge: bool) -> Vec<[f64; 2]> {
    if input.len() < 3 {
        return Vec::new();
    }
    let other = 1 - axis;
    let inside = |p: &[f64; 2]| {
        if keep_ge {
            p[axis] >= bound
        } else {
            p[axis] <= bound
        }
    };
    let cross = |a: &[f64; 2], b: &[f64; 2]| -> [f64; 2] {
        let t = (bound - a[axis]) / (b[axis] - a[axis]);
        let mut r = [0.0; 2];
        r[axis] = bound;
        r[other] = a[other] + t * (b[other] - a[other]);
        r
    };
    let n = input.len();
    let mut out = Vec::with_capacity(n + 2);
    for i in 0..n {
        let cur = input[i];
        let prev = input[(i + n - 1) % n];
        match (inside(&cur), inside(&prev)) {
            (true, true) => out.push(cur),
            (true, false) => {
                out.push(cross(&prev, &cur));
                out.push(cur);
            }
            (false, true) => out.push(cross(&prev, &cur)),
            (false, false) => {}
        }
    }
    out
}

/// Rotate a ring to start at its lexicographically-smallest vertex so identical
/// (e.g. full-ocean) tiles encode to identical bytes and content-dedup collapses
/// them.
fn canonicalize(mut ring: Vec<[i16; 2]>) -> Vec<[i16; 2]> {
    if ring.len() < 2 {
        return ring;
    }
    let mut mi = 0;
    for i in 1..ring.len() {
        if ring[i] < ring[mi] {
            mi = i;
        }
    }
    ring.rotate_left(mi);
    ring
}

/// Bounding-box area of a ring in tile-local units^2.
fn bbox_area(ring: &[[i16; 2]]) -> f64 {
    let (mut a, mut b, mut c, mut d) = (i16::MAX, i16::MAX, i16::MIN, i16::MIN);
    for &[x, y] in ring {
        a = a.min(x);
        b = b.min(y);
        c = c.max(x);
        d = d.max(y);
    }
    if c < a {
        return 0.0;
    }
    (c - a) as f64 * (d - b) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiler::TileParams;

    // z2 grid (4×4 tiles), extent 4096, no margin.
    fn params() -> TileParams {
        TileParams::new(2, 0, 0, 4096, 2.0)
    }

    #[test]
    fn area_covering_four_tiles_emits_each() {
        // A square spanning tiles (0,0)…(1,1) in fractional coords.
        let outer = [[0.2, 0.2], [1.8, 0.2], [1.8, 1.8], [0.2, 1.8], [0.2, 0.2]];
        let mut hit: Vec<(u32, u32)> = Vec::new();
        clip_area(&outer, &[], &params(), 0.0, &mut |x, y, piece| {
            assert!(piece.rings[0].len() >= 3);
            assert_eq!(
                piece.rings.iter().flatten().count(),
                piece.clip.len(),
                "one clip flag per vertex"
            );
            hit.push((x, y));
        });
        hit.sort_unstable();
        assert_eq!(hit, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn line_across_tile_boundary_splits_and_marks_cuts() {
        // Horizontal line from tile 0 into tile 1 at y = 0.5.
        let pts = [[0.25, 0.5], [1.75, 0.5]];
        let mut pieces: Vec<(u32, u32, LinePiece)> = Vec::new();
        clip_line(&pts, &params(), &mut |x, y, p| pieces.push((x, y, p)));
        pieces.sort_by_key(|(x, _, _)| *x);
        assert_eq!(pieces.len(), 2, "one piece per crossed tile");
        // Left tile: real start, cut end at the shared boundary.
        assert!(!pieces[0].2.start_cut && pieces[0].2.end_cut);
        // Right tile: cut start, real end.
        assert!(pieces[1].2.start_cut && !pieces[1].2.end_cut);
    }

    #[test]
    fn point_emitted_in_home_tile() {
        let mut out: Option<(u32, u32, [i16; 2])> = None;
        clip_point([2.5, 3.25], &params(), &mut |x, y, a| out = Some((x, y, a)));
        let (x, y, a) = out.unwrap();
        assert_eq!((x, y), (2, 3));
        assert_eq!(a, [2048, 1024]); // 0.5·4096, 0.25·4096
    }
}
