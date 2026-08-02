//! Global polygon raster mask for low-zoom aggregation

use crate::shapes::{AreaKind, Shape};
use crate::sink::ShapeSink;
use crate::tiler::sink::TileSink;
use geo::{Coord, LineString, Polygon};
use std::io;
use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
use tiles::{MERCATOR_EXTENT, MERCATOR_MAX};

/// Tile zoom whose grid sets the mask resolution: one pixel = one tile at this
/// zoom (`2^MASK_ZOOM` pixels per axis). z13 ≈ 4.9 km/px.
const MASK_ZOOM: u8 = 13;
/// Morphological closing radius, in pixels; bridges gaps up to `2 * radius`.
/// `0` disables closing (aggregation comes only from the resolution coarsening).
const CLOSE_RADIUS: usize = 0;
/// Pixels per axis.
const W: usize = 1 << MASK_ZOOM;

/// A global polygon coverage bitmap, written concurrently during extraction.
pub struct PolygonMask {
    /// `W * W` cells, each `0` or `1`. Atomic so extraction workers can burn
    /// concurrently; a plain store of `1` is idempotent so ordering is `Relaxed`.
    bits: Vec<AtomicU8>,
}

impl Default for PolygonMask {
    fn default() -> Self {
        Self::new()
    }
}

impl PolygonMask {
    pub fn new() -> Self {
        let mut bits = Vec::new();
        bits.resize_with(W * W, || AtomicU8::new(0));
        Self { bits }
    }

    #[inline]
    fn set(&self, x: usize, y: usize) {
        self.bits[y * W + x].store(1, Relaxed);
    }

    /// Burn one polygon's exterior into the mask (holes ignored — filling
    /// clearings only helps aggregation). Even-odd scan fill sets a pixel only
    /// when a polygon covers its row centre, so small patches that don't cover a
    /// cell don't claim it.
    pub fn burn(&self, poly: &Polygon) {
        let ring: Vec<[f64; 2]> = poly
            .exterior()
            .0
            .iter()
            .map(|c| merc_to_pixel(c.x, c.y))
            .collect();
        if ring.len() < 3 {
            return;
        }
        scanline_fill(&ring, |x, y| self.set(x, y));
    }

    /// Close the mask and vectorize it into merged polygons, in Web
    /// Mercator meters (holes dropped — solid blobs). Reads the mask without
    /// consuming it. Contours only the occupied bounding box.
    pub fn merged_polygons(&self) -> Vec<Polygon> {
        let raw: Vec<u8> = self.bits.iter().map(|c| c.load(Relaxed)).collect();
        let Some((x0, y0, x1, y1)) = occupied_bbox(&raw) else {
            return Vec::new();
        };
        // Pad by the closing radius (+1 for the contour's boundary cell) so a
        // dilation near the crop edge, and the outer contour, aren't clipped.
        let pad = CLOSE_RADIUS + 1;
        let cx0 = x0.saturating_sub(pad);
        let cy0 = y0.saturating_sub(pad);
        let cx1 = (x1 + pad).min(W - 1);
        let cy1 = (y1 + pad).min(W - 1);
        let (cw, ch) = (cx1 - cx0 + 1, cy1 - cy0 + 1);

        let mut crop = vec![0u8; cw * ch];
        for y in 0..ch {
            let s = (cy0 + y) * W + cx0;
            crop[y * cw..y * cw + cw].copy_from_slice(&raw[s..s + cw]);
        }
        let closed = if CLOSE_RADIUS > 0 {
            morph_close(&crop, cw, ch, CLOSE_RADIUS)
        } else {
            crop
        };

        let values: Vec<f64> = closed.iter().map(|&b| b as f64).collect();
        let contours = contour::ContourBuilder::new(cw, ch, true)
            .contours(&values, &[0.5])
            .expect("contour");

        let mut out = Vec::new();
        for contour in &contours {
            for poly in &contour.geometry().0 {
                let ext: Vec<Coord> = poly
                    .exterior()
                    .0
                    .iter()
                    .map(|c| pixel_to_merc(c.x + cx0 as f64, c.y + cy0 as f64))
                    .collect();
                if ext.len() >= 4 {
                    out.push(Polygon::new(LineString::from(ext), Vec::new()));
                }
            }
        }
        out
    }
}

/// A `ShapeSink` that tees areas into `PolygonMask` (for coarse-zoom aggregation)
/// while forwarding **every** shape to the inner tiler — which
/// skips raw polygons at coarse zooms, so the two don't overlap.
pub struct PolygonTee<'a> {
    inner: &'a TileSink,
    mask: &'a PolygonMask,
}

impl<'a> PolygonTee<'a> {
    pub fn new(inner: &'a TileSink, mask: &'a PolygonMask) -> Self {
        Self { inner, mask }
    }
}

impl ShapeSink for PolygonTee<'_> {
    fn push(&self, shape: Shape) {
        if let Shape::Area(area) = &shape
            && area.kind == AreaKind::Forest
        {
            self.mask.burn(&area.geometry);
        }
        self.inner.push(shape);
    }

    fn finish(&self) -> io::Result<()> {
        // The inner sink is finished by the caller after aggregated polygons are
        // pushed; nothing to flush here.
        Ok(())
    }
}

/// Web-Mercator meters → mask pixel coordinates (fractional; one pixel = one tile at `MASK_ZOOM`)
#[inline]
fn merc_to_pixel(x: f64, y: f64) -> [f64; 2] {
    let n = W as f64;
    [
        (x + MERCATOR_MAX) / MERCATOR_EXTENT * n,
        (MERCATOR_MAX - y) / MERCATOR_EXTENT * n,
    ]
}

/// Inverse of `merc_to_pixel`: mask grid coordinate → Web-Mercator meters.
#[inline]
fn pixel_to_merc(gx: f64, gy: f64) -> Coord {
    let n = W as f64;
    Coord {
        x: gx / n * MERCATOR_EXTENT - MERCATOR_MAX,
        y: MERCATOR_MAX - gy / n * MERCATOR_EXTENT,
    }
}

/// Bounding box (inclusive) of set pixels, or `None` if the mask is empty.
fn occupied_bbox(raw: &[u8]) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut any = false;
    for y in 0..W {
        for x in 0..W {
            if raw[y * W + x] != 0 {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    any.then_some((x0, y0, x1, y1))
}

/// Even-odd scanline fill of a ring (pixel coords) into a per-pixel setter.
fn scanline_fill(ring: &[[f64; 2]], mut set: impl FnMut(usize, usize)) {
    let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
    for p in ring {
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    let y0 = ymin.floor().max(0.0) as usize;
    let y1 = (ymax.ceil().min((W - 1) as f64)) as usize;
    let n = ring.len();
    let mut xs: Vec<f64> = Vec::new();
    for y in y0..=y1 {
        let yc = y as f64 + 0.5;
        xs.clear();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            if (a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc) {
                let t = (yc - a[1]) / (b[1] - a[1]);
                xs.push(a[0] + t * (b[0] - a[0]));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i + 1 < xs.len() {
            // Fill pixels whose centre lies inside the span [xs[i], xs[i+1]].
            let xa = (xs[i] - 0.5).ceil().max(0.0) as usize;
            let xb = (xs[i + 1] - 0.5).floor().min((W - 1) as f64);
            if xb >= xa as f64 {
                for x in xa..=xb as usize {
                    set(x, y);
                }
            }
            i += 2;
        }
    }
}

/// Morphological closing: dilate by `r` then erode by `r` (separable per axis).
fn morph_close(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let dil = morph_pass(&morph_pass(src, w, h, r, true, true), w, h, r, false, true);
    morph_pass(
        &morph_pass(&dil, w, h, r, true, false),
        w,
        h,
        r,
        false,
        false,
    )
}

/// One separable morphology pass. `horizontal` picks the axis; `dilate` picks OR
/// (max) vs AND (min). Out-of-bounds neighbours count as `0`.
fn morph_pass(src: &[u8], w: usize, h: usize, r: usize, horizontal: bool, dilate: bool) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let r = r as isize;
    for y in 0..h {
        for x in 0..w {
            let mut acc: u8 = if dilate { 0 } else { 1 };
            for d in -r..=r {
                let v = if horizontal {
                    let xx = x as isize + d;
                    if xx >= 0 && (xx as usize) < w {
                        src[y * w + xx as usize]
                    } else {
                        0
                    }
                } else {
                    let yy = y as isize + d;
                    if yy >= 0 && (yy as usize) < h {
                        src[yy as usize * w + x]
                    } else {
                        0
                    }
                };
                if dilate {
                    acc |= v;
                } else {
                    acc &= v;
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}
