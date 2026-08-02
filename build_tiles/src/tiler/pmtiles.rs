//! PMTiles v3 writer — first `TileWriter` backend.
//!
//! Spec-correct: tiles are addressed by PMTiles' Hilbert-based tile id, content-
//! deduplicated, run-length encoded, and split into a root + leaf directories.
//! Tiles stream to a temp data file as they arrive; `finish` assembles the final
//! archive: `header(127) | root_dir | metadata | leaf_dirs | tile_data`.
//!
//! The on-disk codec (header, directories, tile-id math) lives in
//! `util::pmtiles` and is shared with the reader in the `tiles` crate; this
//! file only handles the write-side bookkeeping (streaming, dedup, bounds).
//!
//! Deferred: clustered layout, directory gzip.

use super::writer::TileWriter;
use hashbrown::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tiles::{Mercator, Tile};
use util::pmtiles::{
    self, COMPRESSION_NONE, DirEntry, HEADER_LEN, Header, ROOT_MAX, TILE_TYPE_UNKNOWN, tile_id,
};
use xxhash_rust::xxh3::xxh3_128;

pub struct PmTilesWriter {
    out_path: PathBuf,
    tmp_path: PathBuf,
    data: BufWriter<File>,
    data_len: u64,
    entries: Vec<DirEntry>,
    /// Content hash -> (offset, length) of already-stored tile bytes, so
    /// identical tiles (open water, desert, empty) are written once.
    dedup: HashMap<u128, (u64, u32)>,
    min_zoom: u8,
    max_zoom: u8,
    /// Union of written tile bounds, in Web Mercator meters.
    bbox: Option<[f64; 4]>, // [min_x, min_y, max_x, max_y]
}

impl PmTilesWriter {
    /// Create a writer for the PMTiles archive at `path`. Tile bytes stream to
    /// `<path>.data.tmp` until `finish` assembles the final archive.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let out_path = path.as_ref().to_path_buf();
        let tmp_path = out_path.with_extension("data.tmp");
        let data = BufWriter::new(File::create(&tmp_path)?);
        Ok(Self {
            out_path,
            tmp_path,
            data,
            data_len: 0,
            entries: Vec::new(),
            dedup: HashMap::new(),
            min_zoom: u8::MAX,
            max_zoom: 0,
            bbox: None,
        })
    }

    fn accumulate(&mut self, tile: Tile) {
        self.min_zoom = self.min_zoom.min(tile.z);
        self.max_zoom = self.max_zoom.max(tile.z);
        let b = tile.bounds();
        let (min, max) = (b.min(), b.max());
        self.bbox = Some(match self.bbox {
            None => [min.x, min.y, max.x, max.y],
            Some([x0, y0, x1, y1]) => [x0.min(min.x), y0.min(min.y), x1.max(max.x), y1.max(max.y)],
        });
    }

    /// Written-tile bounds as `[min_lon, min_lat, max_lon, max_lat]` in units of
    /// 1e-7 degrees for the PMTiles header.
    fn lonlat_bounds(&self) -> [i32; 4] {
        let Some([min_x, min_y, max_x, max_y]) = self.bbox else {
            return [0; 4];
        };
        let sw = Mercator::new(min_x, min_y).to_latlon();
        let ne = Mercator::new(max_x, max_y).to_latlon();
        let e7 = |v: f64| (v * 1e7) as i32;
        [e7(sw.lon()), e7(sw.lat()), e7(ne.lon()), e7(ne.lat())]
    }
}

impl TileWriter for PmTilesWriter {
    fn write(&mut self, tile: Tile, data: &[u8]) -> io::Result<()> {
        // Store identical tile bytes once (128-bit content hash; collision is
        // negligible). Duplicates just add a directory entry to the shared data.
        let (offset, length) = match self.dedup.get(&xxh3_128(data)) {
            Some(&loc) => loc,
            None => {
                let loc = (self.data_len, data.len() as u32);
                self.data.write_all(data)?;
                self.data_len += data.len() as u64;
                self.dedup.insert(xxh3_128(data), loc);
                loc
            }
        };
        self.entries.push(DirEntry {
            tile_id: tile_id(tile.z, tile.x, tile.y),
            offset,
            length,
            run_length: 1,
        });
        self.accumulate(tile);
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        self.data.flush()?;

        // Directory entries ordered by tile id, then run-length encoded: a run
        // of consecutive tile ids sharing an offset (identical content, e.g. a
        // contiguous water region — Hilbert order keeps it contiguous) collapses
        // into one entry.
        self.entries.sort_unstable_by_key(|e| e.tile_id);

        // Cluster the data section: lay tile bytes out in tile-id order so that
        // spatially adjacent (Hilbert-adjacent) tiles become contiguous on disk,
        // and a viewport fetches in a few coalesced ranged reads instead of one
        // per tile — the whole point for an HTTP backend. Each distinct content is
        // copied once, at its first appearance in tile-id order; entry offsets are
        // remapped to the new layout and `plan` records the copy order/source.
        let mut remap: HashMap<u64, u64> = HashMap::new(); // old data offset -> new
        let mut plan: Vec<(u64, u32)> = Vec::new(); // (old offset, len) in new order
        let mut clustered_len: u64 = 0;
        for e in &mut self.entries {
            let new_off = match remap.get(&e.offset) {
                Some(&o) => o,
                None => {
                    let o = clustered_len;
                    plan.push((e.offset, e.length));
                    clustered_len += e.length as u64;
                    remap.insert(e.offset, o);
                    o
                }
            };
            e.offset = new_off;
        }

        let mut entries: Vec<DirEntry> = Vec::with_capacity(self.entries.len());
        for e in self.entries.drain(..) {
            match entries.last_mut() {
                Some(last)
                    if last.offset == e.offset
                        && last.tile_id + last.run_length as u64 == e.tile_id =>
                {
                    last.run_length += 1;
                }
                _ => entries.push(e),
            }
        }

        let num_addressed: u64 = entries.iter().map(|e| e.run_length as u64).sum();
        let num_entries = entries.len() as u64;
        let num_contents = self.dedup.len() as u64;

        let (root_dir, leaf_dirs) = pmtiles::build_directories(&entries, ROOT_MAX);
        let metadata = br#"{"format":"shashlik-maptile"}"#.to_vec();

        // Section layout: header | root_dir | metadata | leaf_dirs | tile_data.
        let root_off = HEADER_LEN as u64;
        let meta_off = root_off + root_dir.len() as u64;
        let leaf_off = meta_off + metadata.len() as u64;
        let data_off = leaf_off + leaf_dirs.len() as u64;

        let (min_zoom, max_zoom) = if num_entries == 0 {
            (0, 0)
        } else {
            (self.min_zoom, self.max_zoom)
        };
        let bounds = self.lonlat_bounds();
        let header = Header {
            root_offset: root_off,
            root_length: root_dir.len() as u64,
            metadata_offset: meta_off,
            metadata_length: metadata.len() as u64,
            leaf_offset: leaf_off,
            leaf_length: leaf_dirs.len() as u64,
            data_offset: data_off,
            data_length: self.data_len,
            num_addressed,
            num_entries,
            num_contents,
            clustered: true,
            internal_compression: COMPRESSION_NONE,
            tile_compression: COMPRESSION_NONE,
            tile_type: TILE_TYPE_UNKNOWN,
            min_zoom,
            max_zoom,
            bounds,
            center_zoom: min_zoom,
            center: [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2],
        };

        let mut out = BufWriter::new(File::create(&self.out_path)?);
        out.write_all(&header.to_bytes())?;
        out.write_all(&root_dir)?;
        out.write_all(&metadata)?;
        out.write_all(&leaf_dirs)?;
        // Append the tile data in clustered (tile-id) order: copy each distinct
        // blob once from the temp file, seeking to its original position.
        let mut tmp = File::open(&self.tmp_path)?;
        for &(old_off, len) in &plan {
            tmp.seek(SeekFrom::Start(old_off))?;
            let mut buf = vec![0u8; len as usize];
            tmp.read_exact(&mut buf)?;
            out.write_all(&buf)?;
        }
        out.flush()?;

        drop(tmp);
        let _ = fs::remove_file(&self.tmp_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use util::pmtiles::{MAGIC, VERSION};

    #[test]
    fn writes_readable_header() {
        let path =
            std::env::temp_dir().join(format!("pmtiles-test-{}.pmtiles", std::process::id()));
        let mut w = PmTilesWriter::create(&path).unwrap();
        w.write(Tile::new(0, 0, 0), b"tile-a").unwrap();
        w.write(Tile::new(1, 1, 1), b"tile-b-bigger").unwrap();
        w.write(Tile::new(3, 5, 3), b"c").unwrap();
        w.finish().unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[0..7], MAGIC);
        assert_eq!(bytes[7], VERSION);
        let header = Header::parse(&bytes).unwrap();
        assert_eq!(header.num_entries, 3);
        // Tile data section holds exactly the three blobs.
        assert_eq!(
            header.data_length as usize,
            "tile-a".len() + "tile-b-bigger".len() + "c".len()
        );
        assert_eq!(header.data_offset + header.data_length, bytes.len() as u64);
        // No temp file left behind.
        assert!(!path.with_extension("data.tmp").exists());

        let _ = fs::remove_file(&path);
    }

    /// Write an archive large enough to force leaf directories, then read tiles
    /// back through the `tiles` crate's async reader.
    #[tokio::test]
    async fn writer_reader_roundtrip() {
        use tiles::reader::{FileRangeReader, PmTilesReader, TileSource};

        let path = std::env::temp_dir().join(format!("pmtiles-rt-{}.pmtiles", std::process::id()));

        // ~6000 distinct tiles (unique content) — well past ROOT_MAX, so the
        // writer emits leaf directories and the reader must descend them.
        const N: u32 = 6000;
        let mut w = PmTilesWriter::create(&path).unwrap();
        for x in 0..N {
            w.write(Tile::new(x, 0, 13), format!("tile-{x}").as_bytes())
                .unwrap();
        }
        w.finish().unwrap();

        let reader = FileRangeReader::open(&path).await.unwrap();
        let src = PmTilesReader::open(reader).await.unwrap();
        assert_eq!(src.min_zoom(), 13);
        assert_eq!(src.max_zoom(), 13);

        // Sample present tiles across the id space (exercises leaf descent).
        for x in [0u32, 1, 1234, 4095, 4096, 5999] {
            let got = src.tile(Tile::new(x, 0, 13)).await.unwrap();
            assert_eq!(got.as_deref(), Some(format!("tile-{x}").as_bytes()));
        }
        // A tile that was never written.
        assert!(src.tile(Tile::new(N, 0, 13)).await.unwrap().is_none());

        // Batch fetch preserves order.
        let batch = src
            .tiles(&[
                Tile::new(3, 0, 13),
                Tile::new(N, 0, 13),
                Tile::new(7, 0, 13),
            ])
            .await
            .unwrap();
        assert_eq!(batch[0].as_deref(), Some(b"tile-3".as_ref()));
        assert_eq!(batch[1], None);
        assert_eq!(batch[2].as_deref(), Some(b"tile-7".as_ref()));

        let _ = fs::remove_file(&path);
    }
}
