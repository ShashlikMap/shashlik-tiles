use super::record::{TileGeometry, TileRecord};
use bytemuck::{bytes_of, cast_slice};
use hashbrown::HashMap;
use util::tileformat::{
    AreaMeta, FLAG_RING_CLIP_MASK, LabelMeta, NAME_NONE, PoiMeta, RingMeta, RoadMeta, TILE_VERSION,
    TileHeader, zigzag,
};

/// Append `coords` as a zig-zag delta chain (first vertex = delta from origin).
fn push_deltas(out: &mut Vec<[u16; 2]>, coords: &[[i16; 2]]) {
    let (mut px, mut py) = (0i32, 0i32);
    for &[x, y] in coords {
        let (x, y) = (x as i32, y as i32);
        out.push([zigzag(x - px), zigzag(y - py)]);
        px = x;
        py = y;
    }
}

/// Intern a name into the per-tile string table, returning its index.
fn intern<'a>(s: &'a str, list: &mut Vec<&'a str>, map: &mut HashMap<&'a str, u16>) -> u16 {
    if let Some(&i) = map.get(s) {
        return i;
    }
    let i = list.len() as u16;
    list.push(s);
    map.insert(s, i);
    i
}

/// Serialize a tile's records into the flat tile format (uncompressed bytes;
/// the caller compresses).
pub fn build_tile<'a>(records: impl IntoIterator<Item = &'a TileRecord>, extent: u16) -> Vec<u8> {
    let mut road_metas: Vec<RoadMeta> = Vec::new();
    let mut area_metas: Vec<AreaMeta> = Vec::new();
    let mut ring_metas: Vec<RingMeta> = Vec::new();
    let mut label_metas: Vec<LabelMeta> = Vec::new();
    let mut poi_metas: Vec<PoiMeta> = Vec::new();
    let mut road_coords: Vec<[u16; 2]> = Vec::new();
    let mut ring_coords: Vec<[u16; 2]> = Vec::new();
    // One flag per ring vertex (rings in emission order): clip-produced?
    let mut ring_clip: Vec<bool> = Vec::new();

    // Per-tile string table: intern names so a road split into many segments (or
    // a repeated place name) stores its text once.
    let mut string_list: Vec<&str> = Vec::new();
    let mut interned: HashMap<&str, u16> = HashMap::new();

    for record in records {
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
                debug_assert!(
                    coords.len() <= u16::MAX as usize,
                    "road exceeds u16 vertices"
                );
                road_metas.push(RoadMeta {
                    kind: *kind as u8,
                    caps: (*start as u8) | ((*end as u8) << 2),
                    layer: record.layer,
                    lanes_forward: *lanes_forward,
                    lanes_backward: *lanes_backward,
                    _pad: 0,
                    vertex_count: coords.len() as u16,
                    name: match name {
                        Some(s) => intern(s, &mut string_list, &mut interned),
                        None => NAME_NONE,
                    },
                });
                push_deltas(&mut road_coords, coords);
            }
            TileGeometry::Area {
                kind,
                floors,
                rings,
                clip,
            } => {
                let first_ring = ring_metas.len() as u16;
                let area_verts: usize = rings.iter().map(Vec::len).sum();
                for ring in rings {
                    debug_assert!(ring.len() <= u16::MAX as usize, "ring exceeds u16 vertices");
                    ring_metas.push(RingMeta {
                        vertex_count: ring.len() as u16,
                    });
                    push_deltas(&mut ring_coords, ring);
                }
                // Append this area's clip flags (empty mask = all original).
                if clip.is_empty() {
                    ring_clip.resize(ring_clip.len() + area_verts, false);
                } else {
                    ring_clip.extend_from_slice(clip);
                }
                area_metas.push(AreaMeta {
                    kind: *kind as u8,
                    layer: record.layer,
                    floors: *floors,
                    _pad: 0,
                    first_ring,
                    ring_count: rings.len() as u16,
                });
            }
            TileGeometry::Label {
                class,
                anchor,
                name,
            } => {
                label_metas.push(LabelMeta {
                    class: *class as u8,
                    rank: class.rank(),
                    anchor_x: anchor[0],
                    anchor_y: anchor[1],
                    name: intern(name, &mut string_list, &mut interned),
                });
            }
            TileGeometry::Poi { kind, anchor } => {
                poi_metas.push(PoiMeta {
                    kind: *kind as u8,
                    _pad: 0,
                    anchor_x: anchor[0],
                    anchor_y: anchor[1],
                });
            }
        }
    }

    let has_clip = ring_clip.iter().any(|&c| c);
    let header = TileHeader {
        version: TILE_VERSION,
        flags: if has_clip { FLAG_RING_CLIP_MASK } else { 0 },
        extent,
        road_count: road_metas.len() as u16,
        area_count: area_metas.len() as u16,
        ring_count: ring_metas.len() as u16,
        label_count: label_metas.len() as u16,
        poi_count: poi_metas.len() as u16,
        string_count: string_list.len() as u16,
    };

    let mut out = Vec::new();
    out.extend_from_slice(bytes_of(&header));
    out.extend_from_slice(cast_slice(&road_metas));
    out.extend_from_slice(cast_slice(&area_metas));
    out.extend_from_slice(cast_slice(&ring_metas));
    out.extend_from_slice(cast_slice(&label_metas));
    out.extend_from_slice(cast_slice(&poi_metas));
    out.extend_from_slice(cast_slice(&road_coords));
    out.extend_from_slice(cast_slice(&ring_coords));
    for s in &string_list {
        out.extend_from_slice(&(s.len() as u16).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    // Ring clip-mask (only when any vertex is clip-produced): one bit per ring
    // vertex in emission order, LSB-first, at the very end of the tile.
    if has_clip {
        for chunk in ring_clip.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                if b {
                    byte |= 1 << i;
                }
            }
            out.push(byte);
        }
    }
    out
}
