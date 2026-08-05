//! On-wire layout of a decoded tile — the flat, little-endian, 2-byte-aligned
//! struct-of-arrays shared by the builder (`build_tiles`) and the decoder

use bytemuck::{Pod, Zeroable};

/// Current tile format version (`TileHeader::version`).
pub const TILE_VERSION: u8 = 2;

/// `RoadMeta.name` / `LabelMeta.name` sentinel meaning "no name".
pub const NAME_NONE: u16 = u16::MAX;

/// `TileHeader::flags` bit: a ring clip-mask section is present at the end of
/// the tile (one bit per ring vertex; `true` = clip-produced boundary vertex).
pub const FLAG_RING_CLIP_MASK: u8 = 1;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TileHeader {
    pub version: u8,
    pub flags: u8,
    /// Tile-local coordinate extent (coords span `[0, extent]`).
    pub extent: u16,
    pub road_count: u16,
    pub area_count: u16,
    pub ring_count: u16,
    pub label_count: u16,
    pub poi_count: u16,
    pub string_count: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RoadMeta {
    pub kind: u8,
    /// bits 0..1 = start `EdgeNode`, bits 2..3 = end `EdgeNode`.
    pub caps: u8,
    pub layer: i8,
    /// Lanes forward / backward relative to the coordinate direction. A one-way
    /// road has `lanes_backward == 0`.
    pub lanes_forward: u8,
    pub lanes_backward: u8,
    pub _pad: u8,
    pub vertex_count: u16,
    /// Index into the string table, or [`NAME_NONE`] — the label drawn along the
    /// line.
    pub name: u16,
}

/// A point label: an anchor and a name, drawn by the renderer with collision
/// against other labels (priority = `rank`, lower wins).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LabelMeta {
    pub class: u8,
    pub rank: u8,
    /// Absolute tile-local anchor (not delta-encoded — labels are single points).
    pub anchor_x: i16,
    pub anchor_y: i16,
    /// Index into the string table.
    pub name: u16,
}

/// A point-of-interest symbol: a kind and an absolute tile-local anchor (no
/// text, not delta-encoded — POIs are single points).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PoiMeta {
    pub kind: u8,
    pub _pad: u8,
    pub anchor_x: i16,
    pub anchor_y: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AreaMeta {
    pub kind: u8,
    pub layer: i8,
    /// Building floor count (`building:levels`, >=1); 1 for non-buildings.
    pub floors: u8,
    pub _pad: u8,
    /// Index of this area's first ring in the ring-meta array (ring 0 = outer).
    pub first_ring: u16,
    pub ring_count: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RingMeta {
    pub vertex_count: u16,
}

/// Map a signed delta to an unsigned value where small magnitudes (either sign)
/// stay small, so the high byte is usually zero and the stream compresses well.
#[inline]
pub fn zigzag(d: i32) -> u16 {
    ((d << 1) ^ (d >> 31)) as u16
}

/// Inverse of [`zigzag`].
#[inline]
pub fn dezigzag(z: u16) -> i32 {
    let z = z as i32;
    (z >> 1) ^ -(z & 1)
}
