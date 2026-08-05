//! Tiling logic

pub mod clip;
pub mod format;
pub mod grid;
pub mod pmtiles;
pub mod record;
pub mod sink;
pub mod spill;
pub mod writer;

use geo::Coord;
use tiles::{Mercator, Tile};

/// Number of tiles at all local zooms below `lz` in a bucket: `(4^lz − 1) / 3`.
/// Also the compact id of the first tile at local zoom `lz` (its zoom offset).
#[inline]
fn local_zoom_offset(lz: u8) -> u64 {
    ((1u64 << (2 * lz as u64)) - 1) / 3
}

/// Pack a tile into a **bucket-local, all-zoom** compact id.
///
/// The id is a mini PMTiles address rebased to the bucket: zoom-major, Morton
/// within a zoom, offset by the tile count of all coarser local zooms. It is
/// dense from 0 and unique among one bucket's tiles across every zoom
/// `bucket_zoom..=z`, so a per-bucket leaf directory can key on it directly.
///
/// `bucket_index` is the bucket's Morton code at `bucket_zoom`
/// (`morton2(x >> lz, y >> lz)`); `x`,`y`,`z` are the global tile coords.
pub fn bucket_local_id(bucket_zoom: u8, bucket_index: u32, x: u32, y: u32, z: u8) -> u64 {
    debug_assert!(z >= bucket_zoom, "tile zoom is coarser than the bucket");
    let lz = z - bucket_zoom; // local zoom within the bucket subtree
    debug_assert_eq!(
        morton2(x >> lz, y >> lz),
        bucket_index as u64,
        "tile does not belong to this bucket",
    );
    let (bx, by) = demorton2(bucket_index as u64); // bucket origin, in bucket_zoom tiles
    let local_x = x - (bx << lz); // rebase into [0, 2^lz)
    let local_y = y - (by << lz);
    local_zoom_offset(lz) + morton2(local_x, local_y)
}

/// Recover global `(x, y, z)` from a bucket-local compact id, given the bucket it came from.
pub fn bucket_local_xyz(bucket_zoom: u8, bucket_index: u32, id: u64) -> (u32, u32, u8) {
    // Which local-zoom band does `id` fall in? (bands: [0,1), [1,5), [5,21), …)
    let mut lz = 0u8;
    while id >= local_zoom_offset(lz + 1) {
        lz += 1; // bounded by max local zoom (~15); no overflow in practice
    }
    let (local_x, local_y) = demorton2(id - local_zoom_offset(lz));
    let (bx, by) = demorton2(bucket_index as u64);
    let x = (bx << lz) | local_x; // local_* < 2^lz, so OR == add
    let y = (by << lz) | local_y;
    (x, y, bucket_zoom + lz)
}

/// Fixed parameters for a tiling run
#[derive(Debug, Clone, Copy)]
pub struct TileParams {
    /// Finest zoom tiles are cut at. Higher display zooms come from overzoom.
    pub grid_zoom: u8,
    /// Coarsest materialized zoom (the pyramid floor); below it the client
    /// overzooms.
    pub min_zoom: u8,
    /// Coarse Morton prefix zoom that groups tiles into spill buckets.
    pub bucket_zoom: u8,
    /// Tile-local coordinate range; one tile spans `[0, extent]`.
    pub extent: i32,
    /// Simplification tolerance for pyramid levels, in *rendered pixels*
    /// (converted to tile-local units via `TILE_RENDER_PX`). Only affects
    /// merged/coarse zooms — the base grid zoom is emitted at full detail.
    pub simplify_px: f64,
}

/// Assumed on-screen tile size in pixels, used to convert `TileParams::simplify_px`
/// into tile-local simplification tolerance.
pub const TILE_RENDER_PX: f64 = 512.0;

/// Default configuration: z14 grid (z19 via client overzoom), z3 pyramid floor,
/// z6 buckets, 8192 extent (~0.3 m at z14), **edge-exact clipping** — clipped
/// pieces from adjacent tiles align exactly on the shared boundary and stitch
/// with no overlap/artifacts — 2 px simplification on coarse zooms.
pub const DEFAULT: TileParams = TileParams::new(14, 3, 6, 8192, 2.0);

impl TileParams {
    pub const fn new(
        grid_zoom: u8,
        min_zoom: u8,
        bucket_zoom: u8,
        extent: i32,
        simplify_px: f64,
    ) -> Self {
        Self {
            grid_zoom,
            min_zoom,
            bucket_zoom,
            extent,
            simplify_px,
        }
    }

    /// Simplification tolerance in tile-local units.
    #[inline]
    pub fn simplify_tolerance(&self) -> f64 {
        (self.simplify_px * self.extent as f64 / TILE_RENDER_PX).max(1.0)
    }

    /// Total number of spill buckets
    #[inline]
    pub fn bucket_count(&self) -> u32 {
        1u32 << (2 * self.bucket_zoom)
    }

    /// Global, self-describing tile id
    #[inline]
    pub fn tile_id(&self, x: u32, y: u32, z: u8) -> u64 {
        local_zoom_offset(z) + morton2(x, y)
    }

    /// Route a global `tile_id` to its spill bucket
    #[inline]
    pub fn route(&self, tile_id: u64) -> u32 {
        let (z, morton) = Self::split_id(tile_id);
        let shift = 2 * (z as i32 - self.bucket_zoom as i32);
        (if shift >= 0 {
            morton >> shift
        } else {
            morton << -shift
        }) as u32
    }

    /// Decode a global tile_id back to the exact `Tile`
    #[inline]
    pub fn tile(&self, tile_id: u64) -> Tile {
        let (z, morton) = Self::split_id(tile_id);
        let (x, y) = demorton2(morton);
        Tile::new(x, y, z)
    }

    /// Split a global tile id into `(zoom, morton)` by finding the zoom-offset
    #[inline]
    fn split_id(tile_id: u64) -> (u8, u64) {
        let mut z = 0u8;
        while tile_id >= local_zoom_offset(z + 1) {
            z += 1; // bounded by max zoom (~15); no overflow in practice
        }
        (z, tile_id - local_zoom_offset(z))
    }

    /// Fractional tile coordinates of a Web-Mercator-meters point at the grid zoom
    #[inline]
    pub fn fractional(&self, point: Coord) -> (f64, f64) {
        Mercator::new(point.x, point.y).to_fractional_tile(self.grid_zoom)
    }

    /// Quantize a clipped tile-local `f64` point to `i16`, clamped to the tile
    /// extent `[0, extent]` (edge-exact — no buffer).
    #[inline]
    pub fn quantize(&self, p: [f64; 2]) -> [i16; 2] {
        [
            self.clamp_local(p[0].round()),
            self.clamp_local(p[1].round()),
        ]
    }

    #[inline]
    fn clamp_local(&self, v: f64) -> i16 {
        (v as i32).clamp(0, self.extent) as i16
    }

    /// The clip rectangle for a tile: exactly `[0, extent]` on both axes.
    pub fn clip_rect(&self) -> clip::Rect {
        clip::Rect::new(0.0, self.extent as f64)
    }

    /// Candidate tile `(x, y)` range (inclusive) covering a fractional-tile
    /// bounding box — the tiles a shape might touch, refined by clipping.
    pub fn tiles_covering(
        &self,
        fx_min: f64,
        fy_min: f64,
        fx_max: f64,
        fy_max: f64,
    ) -> (u32, u32, u32, u32) {
        let last = (1u32 << self.grid_zoom) - 1;
        let idx = |v: f64| (v.floor().max(0.0) as u32).min(last);
        (idx(fx_min), idx(fy_min), idx(fx_max), idx(fy_max))
    }
}

/// Interleave the low bits of `x` and `y` into a Morton (z-order) code, keeping
/// spatially close tiles close in id space.
#[inline]
pub fn morton2(x: u32, y: u32) -> u64 {
    spread(x) | (spread(y) << 1)
}

/// Decode a Morton code back to tile `(x, y)`.
#[inline]
pub fn demorton2(d: u64) -> (u32, u32) {
    (compact(d), compact(d >> 1))
}

/// Spread the 32 bits of `n` so bit `i` moves to position `2*i`.
#[inline]
fn spread(n: u32) -> u64 {
    let mut n = n as u64;
    n = (n | (n << 16)) & 0x0000_ffff_0000_ffff;
    n = (n | (n << 8)) & 0x00ff_00ff_00ff_00ff;
    n = (n | (n << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    n = (n | (n << 2)) & 0x3333_3333_3333_3333;
    n = (n | (n << 1)) & 0x5555_5555_5555_5555;
    n
}

/// Gather the even bits of `n` back into a 32-bit value (inverse of `spread`).
#[inline]
fn compact(mut n: u64) -> u32 {
    n &= 0x5555_5555_5555_5555;
    n = (n | (n >> 1)) & 0x3333_3333_3333_3333;
    n = (n | (n >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    n = (n | (n >> 4)) & 0x00ff_00ff_00ff_00ff;
    n = (n | (n >> 8)) & 0x0000_ffff_0000_ffff;
    n = (n | (n >> 16)) & 0x0000_0000_ffff_ffff;
    n as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_known_values() {
        assert_eq!(morton2(0, 0), 0);
        assert_eq!(morton2(1, 0), 1);
        assert_eq!(morton2(0, 1), 2);
        assert_eq!(morton2(1, 1), 3);
        assert_eq!(morton2(2, 0), 4);
        assert_eq!(morton2(3, 3), 15);
    }

    #[test]
    fn morton_roundtrips() {
        for &(x, y) in &[(0u32, 0u32), (1, 2), (12345, 6789), (16383, 16383)] {
            assert_eq!(demorton2(morton2(x, y)), (x, y));
        }
    }

    #[test]
    fn tile_id_decodes_and_routes_across_zooms() {
        let p = DEFAULT; // grid z14, bucket z6
        for z in p.min_zoom..=p.grid_zoom {
            let max = (1u32 << z) - 1;
            for &(x, y) in &[(0u32, 0u32), (1, 2), (max, max), (max / 3, max / 7)] {
                let id = p.tile_id(x, y, z);
                // Global id decodes to the exact tile with no bucket context.
                let t = p.tile(id);
                assert_eq!((t.x, t.y, t.z), (x, y, z), "decode at z{z}");
                // Routes into range from the id alone; at/below the grid it's
                // exactly the tile's coarse Morton-prefix cell.
                let b = p.route(id);
                assert!(b < p.bucket_count(), "bucket in range at z{z}");
                if z >= p.bucket_zoom {
                    let lz = z - p.bucket_zoom;
                    assert_eq!(b, morton2(x >> lz, y >> lz) as u32, "prefix at z{z}");
                }
            }
        }
    }

    #[test]
    fn bucket_local_id_roundtrips() {
        let bz = 6u8;
        for z in bz..=14 {
            let lz = z - bz;
            for &(bx, by) in &[(0u32, 0u32), (13, 27), (63, 63)] {
                let bucket = morton2(bx, by) as u32;
                let max = (1u32 << lz) - 1;
                for &(ox, oy) in &[(0u32, 0u32), (1, 0), (0, 1), (max, max)] {
                    // Mask offsets into [0, 2^lz) so lz==0 (single-tile bucket)
                    // doesn't fabricate coords in a neighbouring bucket.
                    let (ox, oy) = (ox & max, oy & max);
                    let (x, y) = ((bx << lz) | ox, (by << lz) | oy);
                    let id = bucket_local_id(bz, bucket, x, y, z);
                    assert_eq!(bucket_local_xyz(bz, bucket, id), (x, y, z));
                }
            }
        }
    }
}
