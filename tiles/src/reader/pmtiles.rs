//! PMTiles v3 `TileSource`, built on any `RangeReader`.
//!
//! On open it fetches the 127-byte header and the root directory (bounded to
//! ~16 KiB by the writer), both cached for the reader's lifetime. Each `tile`
//! lookup computes the Hilbert tile id, searches the root, descends into a leaf
//! directory if needed, then reads the tile bytes from the data section. Parsed
//! leaf directories are held in a small LRU so repeated lookups into the same
//! leaf cost one ranged read, not one per lookup. The on-disk codec is shared
//! with the writer via `util::pmtiles`.

use super::{RangeReader, TileSource};
use crate::Tile;
use async_trait::async_trait;
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use util::pmtiles::{DirEntry, Header, find_entry, parse_directory, tile_id};

/// How many parsed leaf directories to keep resident.
const LEAF_CACHE_CAP: usize = 256;

/// When batching tiles, data ranges separated by at most
/// this many bytes are fetched as a single ranged read. PMTiles clusters tiles by
/// Hilbert id, so a viewport's tiles are usually contiguous (or nearly) in the
/// data section; bridging small gaps trades a little over-read for far fewer
/// round-trips — the win on an HTTP backend. Tune per transport latency.
const COALESCE_GAP: u64 = 64 * 1024;

pub struct PmTilesReader<R> {
    reader: R,
    header: Header,
    root: Vec<DirEntry>,
    /// Materialized zoom levels, parsed from the archive metadata at open time
    /// (falls back to `min_zoom..=max_zoom` for archives without the field).
    zoom_levels: Vec<u8>,
    /// LRU of parsed leaf directories, keyed by their offset in the leaf section.
    leaves: Mutex<LeafCache>,
}

impl<R: RangeReader> PmTilesReader<R> {
    /// Open a PMTiles archive: read + parse the header, root directory, and
    /// metadata (for the materialized zoom-level set).
    pub async fn open(reader: R) -> io::Result<Self> {
        let head = reader.read_range(0, util::pmtiles::HEADER_LEN).await?;
        let header = Header::parse(&head)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid PMTiles header"))?;
        let root_bytes = reader
            .read_range(header.root_offset, header.root_length as usize)
            .await?;
        let root = parse_directory(&root_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corrupt root directory"))?;
        let metadata = reader
            .read_range(header.metadata_offset, header.metadata_length as usize)
            .await?;
        let zoom_levels = util::pmtiles::parse_zoom_levels(&metadata)
            .unwrap_or_else(|| (header.min_zoom..=header.max_zoom).collect());
        Ok(Self {
            reader,
            header,
            root,
            zoom_levels,
            leaves: Mutex::new(LeafCache::new(LEAF_CACHE_CAP)),
        })
    }

    /// The parsed archive header (bounds, zoom range, section offsets).
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Fetch a leaf directory, using the cache. Never holds the cache lock across
    /// the ranged read.
    async fn leaf(&self, offset: u64, length: u32) -> io::Result<Arc<Vec<DirEntry>>> {
        if let Some(dir) = self.leaves.lock().unwrap().get(offset) {
            return Ok(dir);
        }
        let bytes = self
            .reader
            .read_range(self.header.leaf_offset + offset, length as usize)
            .await?;
        let dir =
            Arc::new(parse_directory(&bytes).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "corrupt leaf directory")
            })?);
        self.leaves.lock().unwrap().insert(offset, dir.clone());
        Ok(dir)
    }

    /// Resolve `tile` to its data-section-relative `(offset, length)`, descending
    /// leaf directories (LRU-cached) as needed. `None` if the archive has no such
    /// tile. Does not read tile data — that's split out so `tiles` can
    /// coalesce many tiles' data ranges into a few reads.
    async fn locate(&self, tile: Tile) -> io::Result<Option<(u64, usize)>> {
        let id = tile_id(tile.z, tile.x, tile.y);
        let mut entry = match find_entry(&self.root, id) {
            Some(e) => *e,
            None => return Ok(None),
        };
        while entry.is_leaf() {
            let leaf = self.leaf(entry.offset, entry.length).await?;
            match find_entry(&leaf, id) {
                Some(e) => entry = *e,
                None => return Ok(None),
            }
        }
        Ok(Some((entry.offset, entry.length as usize)))
    }
}

/// Merge sorted, unique data ranges into fetch blocks, bridging any gap of at
/// most `gap` bytes (adjacent/overlapping ranges always merge; a larger gap ends
/// the current block). Input must be sorted by offset. Returns `(offset, len)`
/// blocks in the same order, non-overlapping and ascending.
fn coalesce_blocks(ranges: &[(u64, usize)], gap: u64) -> Vec<(u64, usize)> {
    let mut blocks: Vec<(u64, usize)> = Vec::new();
    for &(off, len) in ranges {
        if let Some(last) = blocks.last_mut() {
            let last_end = last.0 + last.1 as u64;
            if off <= last_end.saturating_add(gap) {
                let end = last_end.max(off + len as u64);
                last.1 = (end - last.0) as usize;
                continue;
            }
        }
        blocks.push((off, len));
    }
    blocks
}

/// A tiny LRU over parsed leaf directories.
struct LeafCache {
    map: HashMap<u64, (Arc<Vec<DirEntry>>, u64)>, // offset -> (dir, last_used)
    clock: u64,
    cap: usize,
}

impl LeafCache {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            clock: 0,
            cap,
        }
    }

    fn get(&mut self, offset: u64) -> Option<Arc<Vec<DirEntry>>> {
        self.clock += 1;
        let now = self.clock;
        let (dir, used) = self.map.get_mut(&offset)?;
        *used = now;
        Some(dir.clone())
    }

    fn insert(&mut self, offset: u64, dir: Arc<Vec<DirEntry>>) {
        self.clock += 1;
        self.map.insert(offset, (dir, self.clock));
        while self.map.len() > self.cap {
            if let Some(&victim) = self
                .map
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k)
            {
                self.map.remove(&victim);
            } else {
                break;
            }
        }
    }
}

#[async_trait]
impl<R: RangeReader> TileSource for PmTilesReader<R> {
    async fn tile(&self, tile: Tile) -> io::Result<Option<Vec<u8>>> {
        match self.locate(tile).await? {
            Some((offset, len)) => {
                let data = self
                    .reader
                    .read_range(self.header.data_offset + offset, len)
                    .await?;
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    /// Coalescing batch fetch
    async fn tiles(&self, tiles: &[Tile]) -> io::Result<Vec<Option<Vec<u8>>>> {
        // Resolve each tile to its data range (no data read yet).
        let mut located: Vec<Option<(u64, usize)>> = Vec::with_capacity(tiles.len());
        for &t in tiles {
            located.push(self.locate(t).await?);
        }

        // Distinct ranges, sorted, then coalesced into fetch blocks. Duplicates
        // arise naturally from PMTiles' content dedup (many tiles -> same bytes).
        let mut ranges: Vec<(u64, usize)> = located.iter().flatten().copied().collect();
        ranges.sort_unstable();
        ranges.dedup();
        let blocks = coalesce_blocks(&ranges, COALESCE_GAP);

        // Fetch the blocks concurrently (order preserved), then slice each tile.
        let fetched = futures::future::try_join_all(
            blocks
                .iter()
                .map(|&(off, len)| self.reader.read_range(self.header.data_offset + off, len)),
        )
        .await?;

        Ok(located
            .into_iter()
            .map(|loc| {
                loc.map(|(off, len)| {
                    let bi = blocks.partition_point(|b| b.0 <= off) - 1;
                    let start = (off - blocks[bi].0) as usize;
                    fetched[bi][start..start + len].to_vec()
                })
            })
            .collect())
    }

    fn min_zoom(&self) -> u8 {
        self.header.min_zoom
    }

    fn max_zoom(&self) -> u8 {
        self.header.max_zoom
    }

    fn zoom_levels(&self) -> Vec<u8> {
        self.zoom_levels.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::coalesce_blocks;

    #[test]
    fn merges_adjacent_and_within_gap() {
        // 0..100 and 100..150 are adjacent; 170..200 is a 20-byte gap (< 64).
        // All fold into one block; the far range stays separate.
        let ranges = [(0u64, 100usize), (100, 50), (170, 30), (100_000, 10)];
        let blocks = coalesce_blocks(&ranges, 64);
        assert_eq!(blocks, vec![(0, 200), (100_000, 10)]);
    }

    #[test]
    fn gap_larger_than_threshold_splits() {
        let ranges = [(0u64, 10usize), (100, 10)]; // gap 90 > 64
        assert_eq!(coalesce_blocks(&ranges, 64), vec![(0, 10), (100, 10)]);
    }

    #[test]
    fn nested_range_does_not_shrink_block() {
        // A fully-contained range must not truncate the enclosing block.
        let ranges = [(0u64, 100usize), (10, 20)];
        assert_eq!(coalesce_blocks(&ranges, 0), vec![(0, 100)]);
    }

    #[test]
    fn zero_gap_only_merges_touching() {
        let ranges = [(0u64, 50usize), (50, 50), (101, 10)];
        assert_eq!(coalesce_blocks(&ranges, 0), vec![(0, 100), (101, 10)]);
    }
}
