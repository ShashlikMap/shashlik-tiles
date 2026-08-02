//! Async tile reading

mod file;
mod http;
mod pmtiles;

pub use file::FileRangeReader;
pub use http::HttpRangeReader;
pub use pmtiles::PmTilesReader;

use crate::{LatLon, Tile};
use async_trait::async_trait;
use std::io;

/// Low-level random-access byte transport: `read_range` returns the bytes at
/// `[offset, offset + len]`. `&self` + `Send + Sync` so a single reader can
/// serve concurrent requests (positioned reads, no shared cursor).
#[async_trait]
pub trait RangeReader: Send + Sync {
    async fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>>;
}

/// High-level, format-agnostic access to a tile archive.
#[async_trait]
pub trait TileSource: Send + Sync {
    /// The stored bytes for `tile`, or `None` if the archive has no such tile.
    /// Bytes are returned verbatim (still compressed / in the archive's tile
    /// format) — decoding is the caller's concern.
    async fn tile(&self, tile: Tile) -> io::Result<Option<Vec<u8>>>;

    /// Coarsest and finest zoom levels present.
    fn min_zoom(&self) -> u8;
    fn max_zoom(&self) -> u8;

    /// Fetch many tiles concurrently. The default fans out over tile(); a
    /// backend with a batch API (a SQL `IN (...)`, a coalesced range request)
    /// may override this.
    async fn tiles(&self, tiles: &[Tile]) -> io::Result<Vec<Option<Vec<u8>>>> {
        futures::future::try_join_all(tiles.iter().map(|&t| self.tile(t))).await
    }

    /// Fetch every tile in a rectangular tiles(), row-major (matching)
    async fn tile_range(&self, range: TileRange) -> io::Result<Vec<Option<Vec<u8>>>> {
        let tiles: Vec<Tile> = range.tiles().collect();
        self.tiles(&tiles).await
    }
}

/// An inclusive rectangular block of tiles at one zoom — e.g. a map viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRange {
    pub z: u8,
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
}

impl TileRange {
    pub fn new(z: u8, min_x: u32, max_x: u32, min_y: u32, max_y: u32) -> Self {
        Self {
            z,
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// The tile range covering a geographic bounding box at zoom `z`
    /// (`sw` = south-west corner, `ne` = north-east corner).
    pub fn covering(z: u8, sw: LatLon, ne: LatLon) -> Self {
        // Tile y grows southward, so the north-east corner gives min_y.
        let sw_t = sw.to_mercator().to_tile(z);
        let ne_t = ne.to_mercator().to_tile(z);
        Self {
            z,
            min_x: sw_t.x.min(ne_t.x),
            max_x: sw_t.x.max(ne_t.x),
            min_y: sw_t.y.min(ne_t.y),
            max_y: sw_t.y.max(ne_t.y),
        }
    }

    /// Number of tiles in the range.
    pub fn len(&self) -> usize {
        (self.max_x - self.min_x + 1) as usize * (self.max_y - self.min_y + 1) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.max_x < self.min_x || self.max_y < self.min_y
    }

    /// Iterate the tiles row-major (y outer, x inner).
    pub fn tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        (self.min_y..=self.max_y)
            .flat_map(move |y| (self.min_x..=self.max_x).map(move |x| Tile::new(x, y, self.z)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_range_enumerates_row_major() {
        let r = TileRange::new(3, 1, 2, 4, 5);
        assert_eq!(r.len(), 4);
        let tiles: Vec<(u32, u32)> = r.tiles().map(|t| (t.x, t.y)).collect();
        assert_eq!(tiles, vec![(1, 4), (2, 4), (1, 5), (2, 5)]);
    }
}
