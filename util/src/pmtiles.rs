//! PMTiles v3 on-disk codec, shared by the writer (in `build_tiles`) and the reader (in `tiles`)
//! Spec: <https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md>

use crate::varint::{read_uvarint, write_uvarint};

pub const MAGIC: &[u8; 7] = b"PMTiles";
pub const VERSION: u8 = 3;
pub const HEADER_LEN: usize = 127;

/// `internal_compression` / `tile_compression`: none (we store opaque bytes).
pub const COMPRESSION_NONE: u8 = 1;
/// `tile_type`: unknown — a custom, non-MVT tile format.
pub const TILE_TYPE_UNKNOWN: u8 = 0;

/// Target maximum root-directory size so the header + root fit one initial fetch.
pub const ROOT_MAX: usize = 16 * 1024;

/// A directory entry. A *tile* entry (`run_length >= 1`) addresses
/// `[tile_id, tile_id + run_length)` at `(offset, length)` within the tile-data
/// section. A leaf pointer (`run_length == 0`) points at a leaf directory at
/// `(offset, length)` within the leaf-directory section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirEntry {
    pub tile_id: u64,
    pub offset: u64,
    pub length: u32,
    pub run_length: u32,
}

impl DirEntry {
    /// Whether this entry points at a leaf directory rather than tile data.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.run_length == 0
    }
}

/// PMTiles ZXY-> tile id: zoom offset (count of all tiles below `z`) + the
/// Hilbert-curve index within zoom `z`.
pub fn tile_id(z: u8, x: u32, y: u32) -> u64 {
    let zoom_offset = ((1u64 << (2 * z as u64)) - 1) / 3; // sum_{i<z} 4^i
    zoom_offset + hilbert(z, x, y)
}

/// Standard Hilbert-curve xy→d index at zoom `z`.
pub fn hilbert(z: u8, mut x: u32, mut y: u32) -> u64 {
    let n: u32 = 1 << z;
    let mut d: u64 = 0;
    let mut s: u32 = n / 2;
    while s > 0 {
        let rx = ((x & s) > 0) as u32;
        let ry = ((y & s) > 0) as u32;
        d += (s as u64) * (s as u64) * ((3 * rx) ^ ry) as u64;
        if ry == 0 {
            if rx == 1 {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}

/// Serialize directory entries (sorted by tile id) in the PMTiles v3 columnar
/// varint layout: count, delta tile ids, run lengths, lengths, offsets
/// (0 = contiguous with the previous entry).
pub fn serialize_directory(entries: &[DirEntry]) -> Vec<u8> {
    let mut buf = Vec::new();
    write_uvarint(&mut buf, entries.len() as u64);

    let mut last = 0u64;
    for e in entries {
        write_uvarint(&mut buf, e.tile_id - last);
        last = e.tile_id;
    }
    for e in entries {
        write_uvarint(&mut buf, e.run_length as u64);
    }
    for e in entries {
        write_uvarint(&mut buf, e.length as u64);
    }
    for (i, e) in entries.iter().enumerate() {
        if i > 0 && e.offset == entries[i - 1].offset + entries[i - 1].length as u64 {
            write_uvarint(&mut buf, 0);
        } else {
            write_uvarint(&mut buf, e.offset + 1);
        }
    }
    buf
}

/// Parse a columnar-varint directory (inverse of `serialize_directory`).
/// Returns `None` on a truncated or malformed buffer.
pub fn parse_directory(mut buf: &[u8]) -> Option<Vec<DirEntry>> {
    let cursor = &mut buf;
    let n = read_uvarint(cursor)? as usize;
    let mut entries = vec![
        DirEntry {
            tile_id: 0,
            offset: 0,
            length: 0,
            run_length: 0,
        };
        n
    ];

    let mut last = 0u64;
    for e in entries.iter_mut() {
        last += read_uvarint(cursor)?;
        e.tile_id = last;
    }
    for e in entries.iter_mut() {
        e.run_length = read_uvarint(cursor)? as u32;
    }
    for e in entries.iter_mut() {
        e.length = read_uvarint(cursor)? as u32;
    }
    for i in 0..n {
        let v = read_uvarint(cursor)?;
        entries[i].offset = if v == 0 {
            entries[i - 1].offset + entries[i - 1].length as u64
        } else {
            v - 1
        };
    }
    Some(entries)
}

/// Find the entry addressing `tile_id`: the entry with the greatest `tile_id`
/// not exceeding the query. A tile entry matches only if the query falls
/// within its run; a leaf pointer always matches its subtree and must be
/// descended. Returns `None` if the query precedes all entries.
pub fn find_entry(entries: &[DirEntry], tile_id: u64) -> Option<&DirEntry> {
    // partition_point gives the first entry with tile_id > query; step back one.
    let idx = entries.partition_point(|e| e.tile_id <= tile_id);
    if idx == 0 {
        return None;
    }
    let e = &entries[idx - 1];
    if e.is_leaf() || tile_id < e.tile_id + e.run_length as u64 {
        Some(e)
    } else {
        None // past the end of a tile run with no covering entry
    }
}

/// Split entries (sorted by tile id) into a root directory and leaf directories.
///
/// If all entries fit in `root_max`, the root holds them and there are no
/// leaves. Otherwise entries are chunked into leaf directories and the root
/// holds one leaf pointer per leaf (`run_length = 0`, `offset` relative to the
/// leaf section). Leaf size grows until the root itself fits `root_max`.
/// Returns `(root_bytes, leaf_bytes)`.
pub fn build_directories(entries: &[DirEntry], root_max: usize) -> (Vec<u8>, Vec<u8>) {
    let root = serialize_directory(entries);
    if root.len() <= root_max {
        return (root, Vec::new());
    }

    let mut leaf_size = (entries.len() / 3500).max(4096);
    loop {
        let mut root_entries: Vec<DirEntry> = Vec::new();
        let mut leaves: Vec<u8> = Vec::new();
        for chunk in entries.chunks(leaf_size) {
            let leaf = serialize_directory(chunk);
            root_entries.push(DirEntry {
                tile_id: chunk[0].tile_id,
                offset: leaves.len() as u64, // relative to the leaf section
                length: leaf.len() as u32,
                run_length: 0, // leaf pointer
            });
            leaves.extend_from_slice(&leaf);
        }
        let root = serialize_directory(&root_entries);
        if root.len() <= root_max || root_entries.len() <= 1 {
            return (root, leaves);
        }
        leaf_size *= 2;
    }
}

/// The fixed 127-byte PMTiles v3 header. Offsets/lengths locate the four
/// sections; `bounds`/`center` are in units of 1e-7 degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub root_offset: u64,
    pub root_length: u64,
    pub metadata_offset: u64,
    pub metadata_length: u64,
    pub leaf_offset: u64,
    pub leaf_length: u64,
    pub data_offset: u64,
    pub data_length: u64,
    pub num_addressed: u64,
    pub num_entries: u64,
    pub num_contents: u64,
    pub clustered: bool,
    pub internal_compression: u8,
    pub tile_compression: u8,
    pub tile_type: u8,
    pub min_zoom: u8,
    pub max_zoom: u8,
    /// `[min_lon, min_lat, max_lon, max_lat]` in 1e-7 degrees.
    pub bounds: [i32; 4],
    pub center_zoom: u8,
    /// `[lon, lat]` in 1e-7 degrees.
    pub center: [i32; 2],
}

impl Header {
    /// Serialize to the fixed-width 127-byte header.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[0..7].copy_from_slice(MAGIC);
        h[7] = VERSION;
        let put = |h: &mut [u8; HEADER_LEN], at: usize, v: u64| {
            h[at..at + 8].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut h, 8, self.root_offset);
        put(&mut h, 16, self.root_length);
        put(&mut h, 24, self.metadata_offset);
        put(&mut h, 32, self.metadata_length);
        put(&mut h, 40, self.leaf_offset);
        put(&mut h, 48, self.leaf_length);
        put(&mut h, 56, self.data_offset);
        put(&mut h, 64, self.data_length);
        put(&mut h, 72, self.num_addressed);
        put(&mut h, 80, self.num_entries);
        put(&mut h, 88, self.num_contents);
        h[96] = self.clustered as u8;
        h[97] = self.internal_compression;
        h[98] = self.tile_compression;
        h[99] = self.tile_type;
        h[100] = self.min_zoom;
        h[101] = self.max_zoom;
        for (i, v) in self.bounds.iter().enumerate() {
            h[102 + i * 4..106 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        h[118] = self.center_zoom;
        h[119..123].copy_from_slice(&self.center[0].to_le_bytes());
        h[123..127].copy_from_slice(&self.center[1].to_le_bytes());
        h
    }

    /// Parse a 127-byte header. Returns `None` if too short or the magic/version
    /// don't match.
    pub fn parse(buf: &[u8]) -> Option<Header> {
        if buf.len() < HEADER_LEN || &buf[0..7] != MAGIC || buf[7] != VERSION {
            return None;
        }
        let u64_at = |at: usize| u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        let i32_at = |at: usize| i32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
        Some(Header {
            root_offset: u64_at(8),
            root_length: u64_at(16),
            metadata_offset: u64_at(24),
            metadata_length: u64_at(32),
            leaf_offset: u64_at(40),
            leaf_length: u64_at(48),
            data_offset: u64_at(56),
            data_length: u64_at(64),
            num_addressed: u64_at(72),
            num_entries: u64_at(80),
            num_contents: u64_at(88),
            clustered: buf[96] != 0,
            internal_compression: buf[97],
            tile_compression: buf[98],
            tile_type: buf[99],
            min_zoom: buf[100],
            max_zoom: buf[101],
            bounds: [i32_at(102), i32_at(106), i32_at(110), i32_at(114)],
            center_zoom: buf[118],
            center: [i32_at(119), i32_at(123)],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hilbert_and_tile_id() {
        assert_eq!(tile_id(0, 0, 0), 0);
        let mut z1: Vec<u64> = [(0, 0), (0, 1), (1, 1), (1, 0)]
            .iter()
            .map(|&(x, y)| tile_id(1, x, y))
            .collect();
        z1.sort_unstable();
        assert_eq!(z1, vec![1, 2, 3, 4]);
    }

    #[test]
    fn directory_roundtrip_with_runs_and_shared_offsets() {
        let entries = vec![
            DirEntry {
                tile_id: 0,
                offset: 0,
                length: 10,
                run_length: 1,
            },
            DirEntry {
                tile_id: 1,
                offset: 10,
                length: 20,
                run_length: 3,
            }, // run
            DirEntry {
                tile_id: 4,
                offset: 10,
                length: 20,
                run_length: 1,
            }, // shared offset? no, dedup
            DirEntry {
                tile_id: 5,
                offset: 30,
                length: 5,
                run_length: 1,
            },
        ];
        let bytes = serialize_directory(&entries);
        assert_eq!(parse_directory(&bytes).unwrap(), entries);
    }

    #[test]
    fn find_entry_covers_runs_and_leaves() {
        let entries = vec![
            DirEntry {
                tile_id: 10,
                offset: 0,
                length: 5,
                run_length: 3,
            }, // covers 10,11,12
            DirEntry {
                tile_id: 100,
                offset: 0,
                length: 40,
                run_length: 0,
            }, // leaf pointer
        ];
        assert!(find_entry(&entries, 9).is_none()); // before all
        assert_eq!(find_entry(&entries, 11).unwrap().tile_id, 10); // inside run
        assert!(find_entry(&entries, 13).is_none()); // past the run, before leaf
        assert!(find_entry(&entries, 500).unwrap().is_leaf()); // descend leaf
    }

    #[test]
    fn header_roundtrip() {
        let h = Header {
            root_offset: 127,
            root_length: 40,
            metadata_offset: 167,
            metadata_length: 12,
            leaf_offset: 179,
            leaf_length: 0,
            data_offset: 179,
            data_length: 4096,
            num_addressed: 100,
            num_entries: 90,
            num_contents: 88,
            clustered: false,
            internal_compression: COMPRESSION_NONE,
            tile_compression: COMPRESSION_NONE,
            tile_type: TILE_TYPE_UNKNOWN,
            min_zoom: 3,
            max_zoom: 14,
            bounds: [-1, -2, 3, 4],
            center_zoom: 3,
            center: [1, 1],
        };
        assert_eq!(Header::parse(&h.to_bytes()), Some(h));
    }
}
