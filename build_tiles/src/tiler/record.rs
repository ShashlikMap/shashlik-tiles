//! Intermediate spill record codec.
//!
//! One record = one shape clipped to one tile, in tile-local `i16` coordinates.
//! Byte-packed, little-endian, unaligned — optimized for write throughput and a
//! single sequential read (not for zero-copy; that's the final tile).
//!
//! Layout:
//! ```text
//! tile_key : u16
//! tag      : u8      bit7 = 0 Road / 1 Area ; bits0..6 = kind ordinal
//! layer    : i8
//! Road:  caps : u8 (bits0..1 start, bits2..3 end) ; n : uvarint ; [i16 x, i16 y]·n
//! Area:  rings: uvarint ; per ring { n : uvarint ; [i16 x, i16 y]·n }  (ring 0 = outer)
//! ```
//! Counts are varints (1 byte for the common small fragment, unbounded for giant
//! rings — no `u16` ceiling).

use crate::shapes::{AreaKind, EdgeNode, LabelClass, RoadKind};
use util::varint::{read_uvarint, write_uvarint};

/// Tag byte: `AREA_FLAG` → area, else `LABEL_FLAG` → label, else road. The low
/// bits carry the kind/class ordinal.
const AREA_FLAG: u8 = 0x80;
const LABEL_FLAG: u8 = 0x40;
const KIND_MASK: u8 = 0x3f;

/// One clipped shape destined for a single tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TileRecord {
    pub tile_key: u64,
    pub layer: i8,
    pub geometry: TileGeometry,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TileGeometry {
    Road {
        kind: RoadKind,
        start: EdgeNode,
        end: EdgeNode,
        coords: Vec<[i16; 2]>,
        /// Lanes forward / backward relative to the coordinate direction.
        lanes_forward: u8,
        lanes_backward: u8,
        /// Name rendered along the line, if any.
        name: Option<String>,
    },
    Area {
        kind: AreaKind,
        /// Building floor count (`building:levels`, ≥1); meaningful only for `AreaKind::Building`.
        floors: u8,
        /// Ring 0 is the outer ring; the rest are holes.
        rings: Vec<Vec<[i16; 2]>>,
        /// One flag per ring vertex (rings concatenated in order): `true` where
        /// the vertex was produced by tile-boundary clipping. Empty means "none"
        /// (unclipped area).
        clip: Vec<bool>,
    },
    /// A point label: an anchor and text, emitted only in its home tile.
    Label {
        class: LabelClass,
        anchor: [i16; 2],
        name: String,
    },
}

/// Append `record`'s bytes to `buf`.
pub fn encode(buf: &mut Vec<u8>, record: &TileRecord) {
    buf.extend_from_slice(&record.tile_key.to_le_bytes());

    match &record.geometry {
        TileGeometry::Road {
            kind,
            start,
            end,
            coords,
            lanes_forward,
            lanes_backward,
            name,
        } => {
            buf.push(*kind as u8);
            buf.push(record.layer as u8);
            buf.push((*start as u8) | ((*end as u8) << 2));
            buf.push(*lanes_forward);
            buf.push(*lanes_backward);
            write_uvarint(buf, coords.len() as u64);
            write_coords(buf, coords);
            write_opt_str(buf, name.as_deref());
        }
        TileGeometry::Area {
            kind,
            floors,
            rings,
            clip,
        } => {
            buf.push(AREA_FLAG | *kind as u8);
            buf.push(record.layer as u8);
            buf.push(*floors);
            write_uvarint(buf, rings.len() as u64);
            for ring in rings {
                write_uvarint(buf, ring.len() as u64);
                write_coords(buf, ring);
            }
            write_bitmask(buf, clip);
        }
        TileGeometry::Label {
            class,
            anchor,
            name,
        } => {
            buf.push(LABEL_FLAG | *class as u8);
            buf.push(record.layer as u8);
            buf.extend_from_slice(&anchor[0].to_le_bytes());
            buf.extend_from_slice(&anchor[1].to_le_bytes());
            write_opt_str(buf, Some(name));
        }
    }
}

/// Decode one record from the front of `cursor`, advancing it. Returns `None`
/// on a truncated or malformed record.
pub fn decode(cursor: &mut &[u8]) -> Option<TileRecord> {
    let tile_key = read_u64(cursor)?;
    let tag = read_u8(cursor)?;
    let layer = read_u8(cursor)? as i8;
    let kind_ord = tag & KIND_MASK;

    let geometry = if tag & AREA_FLAG != 0 {
        let kind = AreaKind::from_u8(kind_ord)?;
        let floors = read_u8(cursor)?;
        let ring_count = read_uvarint(cursor)? as usize;
        let mut rings = Vec::with_capacity(ring_count);
        let mut total = 0usize;
        for _ in 0..ring_count {
            let ring = read_coords(cursor)?;
            total += ring.len();
            rings.push(ring);
        }
        let clip = read_bitmask(cursor, total)?;
        TileGeometry::Area {
            kind,
            floors,
            rings,
            clip,
        }
    } else if tag & LABEL_FLAG != 0 {
        let class = LabelClass::from_u8(kind_ord)?;
        let anchor = [read_i16(cursor)?, read_i16(cursor)?];
        let name = read_opt_str(cursor)?.unwrap_or_default();
        TileGeometry::Label {
            class,
            anchor,
            name,
        }
    } else {
        let kind = RoadKind::from_u8(kind_ord)?;
        let caps = read_u8(cursor)?;
        let start = EdgeNode::from_u8(caps & 0b11)?;
        let end = EdgeNode::from_u8((caps >> 2) & 0b11)?;
        let lanes_forward = read_u8(cursor)?;
        let lanes_backward = read_u8(cursor)?;
        let coords = read_coords(cursor)?;
        let name = read_opt_str(cursor)?;
        TileGeometry::Road {
            kind,
            start,
            end,
            coords,
            lanes_forward,
            lanes_backward,
            name,
        }
    };

    Some(TileRecord {
        tile_key,
        layer,
        geometry,
    })
}

fn write_coords(buf: &mut Vec<u8>, coords: &[[i16; 2]]) {
    for c in coords {
        buf.extend_from_slice(&c[0].to_le_bytes());
        buf.extend_from_slice(&c[1].to_le_bytes());
    }
}

/// Write an optional string as `uvarint len` + UTF-8 bytes (`None` = len 0).
fn write_opt_str(buf: &mut Vec<u8>, s: Option<&str>) {
    let s = s.unwrap_or("");
    write_uvarint(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

/// Read a length-prefixed string; length 0 yields `None`.
fn read_opt_str(cursor: &mut &[u8]) -> Option<Option<String>> {
    let n = read_uvarint(cursor)? as usize;
    let (bytes, rest) = cursor.split_at_checked(n)?;
    *cursor = rest;
    if n == 0 {
        return Some(None);
    }
    Some(Some(String::from_utf8(bytes.to_vec()).ok()?))
}

/// Write a bit-per-flag mask: a leading `0` byte if all flags are false (the
/// common unclipped case), else `1` followed by `ceil(len/8)` packed bytes.
fn write_bitmask(buf: &mut Vec<u8>, bits: &[bool]) {
    if bits.iter().all(|&b| !b) {
        buf.push(0);
        return;
    }
    buf.push(1);
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << i;
            }
        }
        buf.push(byte);
    }
}

/// Read a mask of `n` flags written by `write_bitmask`. Returns an empty vec
/// when the leading byte is `0` (all false).
fn read_bitmask(cursor: &mut &[u8], n: usize) -> Option<Vec<bool>> {
    let flag = read_u8(cursor)?;
    if flag == 0 {
        return Some(Vec::new());
    }
    let (data, rest) = cursor.split_at_checked(n.div_ceil(8))?;
    *cursor = rest;
    Some((0..n).map(|i| (data[i / 8] >> (i % 8)) & 1 == 1).collect())
}

fn read_coords(cursor: &mut &[u8]) -> Option<Vec<[i16; 2]>> {
    let n = read_uvarint(cursor)? as usize;
    let mut coords = Vec::with_capacity(n);
    for _ in 0..n {
        coords.push([read_i16(cursor)?, read_i16(cursor)?]);
    }
    Some(coords)
}

#[inline]
fn read_u8(cursor: &mut &[u8]) -> Option<u8> {
    let (&byte, rest) = cursor.split_first()?;
    *cursor = rest;
    Some(byte)
}

#[inline]
fn read_u64(cursor: &mut &[u8]) -> Option<u64> {
    let (bytes, rest) = cursor.split_at_checked(8)?;
    *cursor = rest;
    Some(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

#[inline]
fn read_i16(cursor: &mut &[u8]) -> Option<i16> {
    let (bytes, rest) = cursor.split_at_checked(2)?;
    *cursor = rest;
    Some(i16::from_le_bytes([bytes[0], bytes[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(record: &TileRecord) {
        let mut buf = Vec::new();
        encode(&mut buf, record);
        let mut cursor = buf.as_slice();
        let decoded = decode(&mut cursor).expect("decode");
        assert_eq!(&decoded, record);
        assert!(cursor.is_empty(), "record left trailing bytes");
    }

    #[test]
    fn road_roundtrip() {
        roundtrip(&TileRecord {
            tile_key: 0xBEEF,
            layer: -2,
            geometry: TileGeometry::Road {
                kind: RoadKind::Motorway,
                start: EdgeNode::Connected,
                end: EdgeNode::Cut,
                coords: vec![[0, 0], [8192, -256], [-256, 8448], [123, -45]],
                lanes_forward: 3,
                lanes_backward: 0,
                name: Some("東名高速道路".to_string()),
            },
        });
    }

    #[test]
    fn unnamed_road_roundtrip() {
        roundtrip(&TileRecord {
            tile_key: 1,
            layer: 0,
            geometry: TileGeometry::Road {
                kind: RoadKind::Residential,
                start: EdgeNode::Disconnected,
                end: EdgeNode::Disconnected,
                coords: vec![[0, 0], [10, 10]],
                lanes_forward: 1,
                lanes_backward: 1,
                name: None,
            },
        });
    }

    #[test]
    fn building_roundtrip() {
        roundtrip(&TileRecord {
            tile_key: 3,
            layer: 0,
            geometry: TileGeometry::Area {
                kind: AreaKind::Building,
                floors: 12,
                rings: vec![vec![[0, 0], [50, 0], [50, 80], [0, 80], [0, 0]]],
                clip: vec![],
            },
        });
    }

    #[test]
    fn label_roundtrip() {
        roundtrip(&TileRecord {
            tile_key: 42,
            layer: 0,
            geometry: TileGeometry::Label {
                class: LabelClass::City,
                anchor: [4096, 2048],
                name: "Tokyo".to_string(),
            },
        });
    }

    #[test]
    fn area_roundtrip() {
        roundtrip(&TileRecord {
            tile_key: 7,
            layer: 0,
            geometry: TileGeometry::Area {
                kind: AreaKind::Water,
                floors: 1,
                rings: vec![
                    vec![[0, 0], [4096, 0], [4096, 4096], [0, 4096], [0, 0]],
                    vec![[100, 100], [200, 100], [150, 200], [100, 100]],
                ],
                clip: vec![],
            },
        });
    }

    #[test]
    fn multi_record_stream() {
        // Records concatenate and decode back-to-back.
        let a = TileRecord {
            tile_key: 1,
            layer: 1,
            geometry: TileGeometry::Road {
                kind: RoadKind::Residential,
                start: EdgeNode::Disconnected,
                end: EdgeNode::Disconnected,
                coords: vec![[1, 2], [3, 4]],
                lanes_forward: 2,
                lanes_backward: 2,
                name: None,
            },
        };
        let b = TileRecord {
            tile_key: 2,
            layer: 0,
            geometry: TileGeometry::Area {
                kind: AreaKind::Forest,
                floors: 1,
                rings: vec![vec![[0, 0], [10, 0], [10, 10], [0, 0]]],
                clip: vec![],
            },
        };
        let mut buf = Vec::new();
        encode(&mut buf, &a);
        encode(&mut buf, &b);
        let mut cur = buf.as_slice();
        assert_eq!(decode(&mut cur).unwrap(), a);
        assert_eq!(decode(&mut cur).unwrap(), b);
        assert!(cur.is_empty());
    }
}
