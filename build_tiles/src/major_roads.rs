//! Polyline merging for the low-zoom major-road network.
//!
//! Given a set of road polylines (in Web-Mercator metres), join them by shared
//! endpoints into continuous **strokes** that simplify cleanly, threading through
//! interchanges by good-continuation, then drop one carriageway of each parallel
//! pair. The caller ([`crate::route_roads`]) selects *which* polylines to feed in
//! (member ways of motorway-bearing route relations) and clips the result into
//! the coarse tiles (`z <= ROAD_AGG_MAX_ZOOM`); raw ways cover `z >= 10`, so the
//! two zoom ranges don't overlap.

use hashbrown::HashMap;
use tiles::EARTH_RADIUS;

/// Two strokes closer than this (Web-Mercator metres) count as the same road for
/// parallel-carriageway dedup — comfortably above a dual carriageway's median
/// width plus low-zoom simplification wobble.
const MAX_PARALLEL_DIST_M: f64 = 150.0;
/// Fraction of the shorter stroke that must run within [`MAX_PARALLEL_DIST_M`] of
/// the longer one for it to be dropped as a parallel duplicate.
const MIN_PARALLEL_OVERLAP: f64 = 0.6;

/// Join `polylines` into continuous strokes (good-continuation chaining) and drop
/// one stroke of every parallel-carriageway pair. Coordinates are Web-Mercator
/// metres throughout; output strokes have `len >= 2`.
pub fn merge_polylines(polylines: Vec<Vec<[f64; 2]>>) -> Vec<Vec<[f64; 2]>> {
    let chains: Vec<Vec<[f64; 2]>> = chain_group(&polylines)
        .into_iter()
        .filter(|c| c.len() >= 2)
        .collect();
    dedup_parallel(chains)
}

/// Drop one stroke of every parallel-carriageway pair: keeping longer strokes,
/// remove any shorter stroke that runs alongside a kept one for most of its
/// length. The retained line sits ~½ the median off the true centre — sub-pixel
/// at the low zooms these serve.
fn dedup_parallel(chains: Vec<Vec<[f64; 2]>>) -> Vec<Vec<[f64; 2]>> {
    let n = chains.len();
    let boxes: Vec<[f64; 4]> = chains.iter().map(|c| bbox(c)).collect();
    // Longest first, so the survivor of each pair is the longer stroke.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ground_length(&chains[b]).total_cmp(&ground_length(&chains[a])));

    let mut removed = vec![false; n];
    for (rank, &i) in order.iter().enumerate() {
        if removed[i] {
            continue;
        }
        for &j in &order[rank + 1..] {
            if removed[j] || !bbox_overlap(&boxes[i], &boxes[j], MAX_PARALLEL_DIST_M) {
                continue;
            }
            // j is no longer than i; drop j if it hugs i for most of its length.
            if parallel_overlap(&chains[j], &chains[i]) >= MIN_PARALLEL_OVERLAP {
                removed[j] = true;
            }
        }
    }
    chains
        .into_iter()
        .zip(removed)
        .filter_map(|(c, r)| (!r).then_some(c))
        .collect()
}

/// Fraction of `a`'s (subsampled) vertices lying within [`MAX_PARALLEL_DIST_M`]
/// of polyline `b`.
fn parallel_overlap(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let samples = a.len().min(64).max(2);
    let step = (a.len() - 1) as f64 / (samples - 1) as f64;
    let mut near = 0usize;
    for s in 0..samples {
        let p = a[(s as f64 * step).round() as usize];
        if b.windows(2).any(|w| point_seg_dist(p, w[0], w[1]) <= MAX_PARALLEL_DIST_M) {
            near += 1;
        }
    }
    near as f64 / samples as f64
}

/// Distance from point `p` to segment `a`–`b`.
fn point_seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let (wx, wy) = (p[0] - a[0], p[1] - a[1]);
    let c1 = vx * wx + vy * wy;
    let c2 = vx * vx + vy * vy;
    let t = if c2 > 0.0 { (c1 / c2).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
    (dx * dx + dy * dy).sqrt()
}

/// `[min_x, min_y, max_x, max_y]` of a polyline.
fn bbox(line: &[[f64; 2]]) -> [f64; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in line {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    [x0, y0, x1, y1]
}

/// Whether two bounding boxes come within `margin` of each other.
fn bbox_overlap(a: &[f64; 4], b: &[f64; 4], margin: f64) -> bool {
    a[0] - margin <= b[2] && b[0] - margin <= a[2] && a[1] - margin <= b[3] && b[1] - margin <= a[3]
}

/// Approximate ground length of a polyline given in Web-Mercator metres. Mercator
/// overstates distance by the secant of the latitude, so divide by the scale
/// factor `cosh(y/R)` at the line's mid-latitude to recover real metres — keeps
/// the length threshold meaningful across the planet, not just at the equator.
fn ground_length(coords: &[[f64; 2]]) -> f64 {
    let mut merc = 0.0;
    let mut y_sum = 0.0;
    for w in coords.windows(2) {
        let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
        merc += (dx * dx + dy * dy).sqrt();
        y_sum += w[0][1];
    }
    let y_mid = y_sum / (coords.len().saturating_sub(1).max(1)) as f64;
    merc / (y_mid / EARTH_RADIUS).cosh()
}

/// Minimum cosine between the incoming heading and a candidate continuation for
/// it to count as "the same road going straight through" — `0.5` ≈ a 60° turn.
/// Stroke building threads through interchanges (degree > 2 nodes) by taking the
/// straightest branch, so a highway stays one continuous line instead of
/// fragmenting at every junction (which would let the length cull punch gaps).
const MIN_CONTINUATION_COS: f64 = 0.5;

/// Chain a group's ways into continuous "strokes" by good-continuation: at each
/// node pick the unused way that best continues the current heading (straightest,
/// within [`MIN_CONTINUATION_COS`]). Branches that turn off sharply (ramps, cross
/// roads) are left for their own strokes.
fn chain_group(lines: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    // Quantize endpoints to 0.1 m so lines sharing an OSM node match despite
    // float noise (endpoints derive from the same lat/lon, so this is exact).
    let key = |c: [f64; 2]| ((c[0] * 10.0).round() as i64, (c[1] * 10.0).round() as i64);

    // node -> [(line index, is_start)]
    let mut at: HashMap<(i64, i64), Vec<(usize, bool)>> = HashMap::new();
    for (li, c) in lines.iter().enumerate() {
        if c.len() < 2 {
            continue;
        }
        at.entry(key(c[0])).or_default().push((li, true));
        at.entry(key(*c.last().unwrap())).or_default().push((li, false));
    }

    let mut used = vec![false; lines.len()];
    let mut out = Vec::new();
    for seed in 0..lines.len() {
        if used[seed] || lines[seed].len() < 2 {
            continue;
        }
        used[seed] = true;
        let mut coords = lines[seed].clone();

        // Extend the tail, then reverse and extend again (grows both directions).
        for _ in 0..2 {
            loop {
                let n = coords.len();
                if n < 2 {
                    break;
                }
                // Heading into the tail node (unit-ish direction of the last seg).
                let inc = [
                    coords[n - 1][0] - coords[n - 2][0],
                    coords[n - 1][1] - coords[n - 2][1],
                ];
                let inc_len = (inc[0] * inc[0] + inc[1] * inc[1]).sqrt();
                if inc_len == 0.0 {
                    break;
                }
                let Some(ports) = at.get(&key(coords[n - 1])) else {
                    break;
                };

                // Pick the unused branch whose departure best continues `inc`.
                let mut best: Option<(usize, bool)> = None;
                let mut best_cos = MIN_CONTINUATION_COS;
                for &(li, is_start) in ports {
                    if used[li] {
                        continue;
                    }
                    let c = &lines[li];
                    // Direction leaving the node along this candidate.
                    let dep = if is_start {
                        [c[1][0] - c[0][0], c[1][1] - c[0][1]]
                    } else {
                        let m = c.len();
                        [c[m - 2][0] - c[m - 1][0], c[m - 2][1] - c[m - 1][1]]
                    };
                    let dep_len = (dep[0] * dep[0] + dep[1] * dep[1]).sqrt();
                    if dep_len == 0.0 {
                        continue;
                    }
                    let cos = (inc[0] * dep[0] + inc[1] * dep[1]) / (inc_len * dep_len);
                    if cos >= best_cos {
                        best_cos = cos;
                        best = Some((li, is_start));
                    }
                }

                let Some((j, j_is_start)) = best else {
                    break;
                };
                used[j] = true;
                let cj = &lines[j];
                if j_is_start {
                    coords.extend_from_slice(&cj[1..]);
                } else {
                    coords.extend(cj[..cj.len() - 1].iter().rev().copied());
                }
            }
            coords.reverse();
        }
        out.push(coords);
    }
    out
}
