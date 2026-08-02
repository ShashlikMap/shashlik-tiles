//! LEB128 unsigned varints, shared by the spill record codec and the PMTiles
//! directory codec.

/// Append `v` as an LEB128 unsigned varint.
pub fn write_uvarint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

/// Read one LEB128 unsigned varint, advancing `cursor`. Returns `None` on a
/// truncated or overlong (> 64-bit) encoding.
pub fn read_uvarint(cursor: &mut &[u8]) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let (&byte, rest) = cursor.split_first()?;
        *cursor = rest;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None; // overlong
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let cases = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        for v in cases {
            let mut buf = Vec::new();
            write_uvarint(&mut buf, v);
            let mut cur = buf.as_slice();
            assert_eq!(read_uvarint(&mut cur), Some(v));
            assert!(cur.is_empty(), "leftover bytes for {v}");
        }
    }

    #[test]
    fn truncated_is_none() {
        let mut cur: &[u8] = &[0x80]; // continuation bit set, no follow-up
        assert_eq!(read_uvarint(&mut cur), None);
    }
}
