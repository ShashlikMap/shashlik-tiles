//! Clipping geometry to a tile rectangle

/// A square clip region `[min, max]` on both axes (a buffered tile).
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub min: f64,
    pub max: f64,
}

impl Rect {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

/// One piece of a clipped polyline.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippedLine {
    pub points: Vec<[f64; 2]>,
    /// The first point is a clip intersection (not the polyline's original start).
    pub start_cut: bool,
    /// The last point is a clip intersection (not the polyline's original end).
    pub end_cut: bool,
}

/// Clip a polyline to `rect`, returning the visible pieces in order. A polyline
/// that leaves and re-enters yields several pieces; one fully inside yields a
/// single piece equal to the input with both cut flags false.
pub fn clip_line(points: &[[f64; 2]], rect: Rect) -> Vec<ClippedLine> {
    let mut out = Vec::new();
    if points.len() < 2 {
        return out;
    }

    let mut current: Vec<[f64; 2]> = Vec::new();
    let mut start_cut = false;

    for pair in points.windows(2) {
        let (p, q) = (pair[0], pair[1]);
        match liang_barsky(p, q, rect) {
            None => flush(&mut out, &mut current, start_cut, true),
            Some((t0, t1)) => {
                if current.is_empty() {
                    start_cut = t0 > 0.0;
                    current.push(lerp(p, q, t0));
                }
                current.push(lerp(p, q, t1));
                if t1 < 1.0 {
                    flush(&mut out, &mut current, start_cut, true);
                }
            }
        }
    }
    // Trailing piece ends at the original last vertex (t1 == 1), so not a cut.
    flush(&mut out, &mut current, start_cut, false);

    out
}

fn flush(out: &mut Vec<ClippedLine>, current: &mut Vec<[f64; 2]>, start_cut: bool, end_cut: bool) {
    if current.len() >= 2 {
        out.push(ClippedLine {
            points: std::mem::take(current),
            start_cut,
            end_cut,
        });
    } else {
        current.clear();
    }
}

/// Liang–Barsky: the visible parameter range `(t0, t1)` of segment `p->q`, or
/// `None` if the segment is entirely outside `rect`.
fn liang_barsky(p: [f64; 2], q: [f64; 2], rect: Rect) -> Option<(f64, f64)> {
    let dx = q[0] - p[0];
    let dy = q[1] - p[1];
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;

    let edges = [
        (-dx, p[0] - rect.min),
        (dx, rect.max - p[0]),
        (-dy, p[1] - rect.min),
        (dy, rect.max - p[1]),
    ];

    for (denom, dist) in edges {
        if denom == 0.0 {
            if dist < 0.0 {
                return None; // parallel to edge and outside
            }
        } else {
            let r = dist / denom;
            if denom < 0.0 {
                if r > t1 {
                    return None;
                }
                if r > t0 {
                    t0 = r;
                }
            } else {
                if r < t0 {
                    return None;
                }
                if r < t1 {
                    t1 = r;
                }
            }
        }
    }
    Some((t0, t1))
}

#[inline]
fn lerp(p: [f64; 2], q: [f64; 2], t: f64) -> [f64; 2] {
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/// A polygon ring clipped to a tile, with per-vertex provenance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClippedRing {
    pub points: Vec<[f64; 2]>,
    /// Parallel to `points`: `true` where the vertex was produced by the clip
    /// (lies on the tile boundary), `false` for an original geometry vertex. An
    /// edge between two consecutive clip vertices is an artificial tile-border
    /// edge — the renderer skips it as an outline and welds it to the neighbour.
    pub clip: Vec<bool>,
}

impl ClippedRing {
    /// Number of ring vertices (kept ⇒ `is_empty` lint waived).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.points.len()
    }
}

/// Clip a polygon ring to `rect` (Sutherland–Hodgman). Input may be explicitly
/// closed (first == last); output is an open ring (implicitly closed), empty if
/// nothing survives. Tracks which output vertices the clip created.
pub fn clip_ring(points: &[[f64; 2]], rect: Rect) -> ClippedRing {
    let closed = points.len() >= 2 && points.first() == points.last();
    let poly: &[[f64; 2]] = if closed {
        &points[..points.len() - 1]
    } else {
        points
    };
    if poly.len() < 3 {
        return ClippedRing::default();
    }

    // Carry a clip flag alongside each vertex; input vertices start as original.
    let mut ring: Vec<([f64; 2], bool)> = poly.iter().map(|&p| (p, false)).collect();
    for edge in [
        Edge::MinX(rect.min),
        Edge::MaxX(rect.max),
        Edge::MinY(rect.min),
        Edge::MaxY(rect.max),
    ] {
        if ring.is_empty() {
            break;
        }
        ring = clip_ring_edge(&ring, edge);
    }

    let (points, clip) = ring.into_iter().map(|(p, c)| (p, c)).unzip();
    ClippedRing { points, clip }
}

#[derive(Clone, Copy)]
enum Edge {
    MinX(f64),
    MaxX(f64),
    MinY(f64),
    MaxY(f64),
}

impl Edge {
    #[inline]
    fn inside(&self, p: [f64; 2]) -> bool {
        match *self {
            Edge::MinX(c) => p[0] >= c,
            Edge::MaxX(c) => p[0] <= c,
            Edge::MinY(c) => p[1] >= c,
            Edge::MaxY(c) => p[1] <= c,
        }
    }

    /// Intersection of segment `a->b` with this edge (only called when the two
    /// endpoints straddle it, so the relevant delta is non-zero).
    #[inline]
    fn intersect(&self, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        match *self {
            Edge::MinX(c) | Edge::MaxX(c) => {
                let t = (c - a[0]) / (b[0] - a[0]);
                [c, a[1] + t * (b[1] - a[1])]
            }
            Edge::MinY(c) | Edge::MaxY(c) => {
                let t = (c - a[1]) / (b[1] - a[1]);
                [a[0] + t * (b[0] - a[0]), c]
            }
        }
    }
}

fn clip_ring_edge(ring: &[([f64; 2], bool)], edge: Edge) -> Vec<([f64; 2], bool)> {
    let mut out = Vec::new();
    let mut prev = ring[ring.len() - 1];
    for &curr in ring {
        let curr_in = edge.inside(curr.0);
        let prev_in = edge.inside(prev.0);
        if curr_in {
            if !prev_in {
                out.push((edge.intersect(prev.0, curr.0), true)); // clip-created
            }
            out.push(curr); // carries its existing flag
        } else if prev_in {
            out.push((edge.intersect(prev.0, curr.0), true)); // clip-created
        }
        prev = curr;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: Rect = Rect {
        min: 0.0,
        max: 100.0,
    };

    #[test]
    fn line_fully_inside() {
        let pts = [[10.0, 10.0], [50.0, 50.0], [90.0, 10.0]];
        let out = clip_line(&pts, R);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].points, pts);
        assert!(!out[0].start_cut && !out[0].end_cut);
    }

    #[test]
    fn line_crossing_both_ends_cut() {
        let out = clip_line(&[[-50.0, 50.0], [150.0, 50.0]], R);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].points, vec![[0.0, 50.0], [100.0, 50.0]]);
        assert!(out[0].start_cut && out[0].end_cut);
    }

    #[test]
    fn line_fully_outside() {
        assert!(clip_line(&[[200.0, 200.0], [300.0, 300.0]], R).is_empty());
    }

    #[test]
    fn line_exits_and_reenters() {
        // Inside → out right → back in: two pieces, cuts only at the boundary.
        let pts = [[50.0, 50.0], [150.0, 50.0], [150.0, 80.0], [50.0, 80.0]];
        let out = clip_line(&pts, R);
        assert_eq!(out.len(), 2);

        assert_eq!(out[0].points, vec![[50.0, 50.0], [100.0, 50.0]]);
        assert!(!out[0].start_cut && out[0].end_cut);

        assert_eq!(out[1].points, vec![[100.0, 80.0], [50.0, 80.0]]);
        assert!(out[1].start_cut && !out[1].end_cut);
    }

    #[test]
    fn ring_fully_inside_unchanged() {
        let ring = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]];
        let out = clip_ring(&ring, R);
        assert_eq!(out.points, ring.to_vec());
        assert!(out.clip.iter().all(|&c| !c)); // nothing clipped
    }

    #[test]
    fn ring_fully_outside_empty() {
        let ring = [
            [200.0, 200.0],
            [300.0, 200.0],
            [300.0, 300.0],
            [200.0, 300.0],
        ];
        assert!(clip_ring(&ring, R).points.is_empty());
    }

    #[test]
    fn ring_clipped_to_rect() {
        // A square enclosing the clip rect collapses to the rect's corners —
        // every surviving vertex sits on the boundary and is clip-produced.
        let ring = [
            [-50.0, -50.0],
            [150.0, -50.0],
            [150.0, 150.0],
            [-50.0, 150.0],
        ];
        let out = clip_ring(&ring, R);
        assert!(out.len() >= 4);
        assert_eq!(out.points.len(), out.clip.len());
        for (p, &c) in out.points.iter().zip(&out.clip) {
            assert!((0.0..=100.0).contains(&p[0]) && (0.0..=100.0).contains(&p[1]));
            assert!(c, "corner vertex should be marked clip-produced");
        }
    }

    #[test]
    fn ring_partially_clipped_marks_only_boundary_vertices() {
        // A triangle poking out the right edge: the two intersection points on
        // x=100 are clip-produced, the interior vertices are not.
        let ring = [[50.0, 50.0], [150.0, 50.0], [50.0, 90.0]];
        let out = clip_ring(&ring, R);
        let on_edge: Vec<bool> = out.points.iter().map(|p| p[0] == 100.0).collect();
        assert_eq!(
            out.clip, on_edge,
            "clip flags should match vertices on x=100"
        );
    }

    #[test]
    fn closed_ring_input_strips_duplicate() {
        // Explicitly-closed input (first == last) is handled without a dup.
        let ring = [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 10.0]];
        let out = clip_ring(&ring, R);
        assert_eq!(out.len(), 3);
    }
}
