//! Populates the reusable on-disk caches (`cache` crate) from an OSM PBF file.
//!
//! Currently builds the node location cache: an mmap-backed
//! `SparseStore` mapping every node referenced by a kept way to its packed
//! coordinate. Two passes over the PBF, both driven off the section offsets in `OsmPbfIndex`:
//!
//! 1. `build_node_bitset` — ways pass. Set a bit for every node id referenced
//!    by a way that matches the tag filter or is pulled in by a relation.
//! 2. `build_node_store` — nodes pass. Project each referenced node and write
//!    it into the store at `rank(id)`.

use crate::classifier::{BlockShapeClassifier, ShapeClassification};
use bytemuck::{Pod, Zeroable};
use cache::bitset::{BitSet, RankedBitSet};
use cache::index::OsmPbfIndex;
use cache::store::{SparseStore, SparseStoreBuilder};
use hashbrown::HashSet;
use osm_pbf::{
    protos::{DecodeNodes, Primitives, osmpbf},
    reader::OsmReader,
    tags::TagFilter,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A node coordinate packed as fixed-point lon/lat (degrees × 1e7).
///
/// 8 bytes, projection-agnostic (~1 cm precision) — geometry is projected to
/// tiles later, not baked into the cache. Lon range +-180 -> +-1.8e9 fits `i32`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedCoord {
    pub lon_e7: i32,
    pub lat_e7: i32,
}

impl PackedCoord {
    #[inline]
    pub fn from_lonlat(lon: f64, lat: f64) -> Self {
        Self {
            lon_e7: (lon * 1e7).round() as i32,
            lat_e7: (lat * 1e7).round() as i32,
        }
    }

    #[inline]
    pub fn lon(&self) -> f64 {
        self.lon_e7 as f64 / 1e7
    }

    #[inline]
    pub fn lat(&self) -> f64 {
        self.lat_e7 as f64 / 1e7
    }
}

/// The node location cache: `node id -> PackedCoord`.
pub type NodeStore = SparseStore<PackedCoord>;

/// Default on-disk path for the node cache (`<pbf>.nodes`).
pub fn node_cache_path(pbf: impl AsRef<Path>) -> PathBuf {
    pbf.as_ref().with_extension("nodes")
}

/// Whether the node store at `cache_path` can be reused with a bitset of
/// `node_count` set bits. The store is addressed by `rank(id)`, so it is only
/// valid for the exact referenced-node set it was written with; we check that
/// cheaply via file length (`node_count * size_of::<PackedCoord>()`). A filter
/// change alters the node count and so invalidates a stale store automatically —
/// delete the file to force a rebuild in the rare same-count case.
///
/// Callers should also check `load_bbox(bbox_cache_path(cache_path))` before
/// treating this as a full cache hit (see `main.rs`) -- a `.nodes` file
/// written before the bbox sidecar existed at all would otherwise match here
/// but have no bbox available, silently disabling
/// `--clip-water-to-extract-bbox` (a real bug this project hit).
pub fn node_store_matches(cache_path: &Path, node_count: usize) -> bool {
    let expected = (node_count * size_of::<PackedCoord>()).max(1) as u64;
    std::fs::metadata(cache_path).map(|m| m.len()).ok() == Some(expected)
}

/// On-disk path of the bbox sidecar for the node cache at `cache_path` (the
/// return value of `node_cache_path`) -- `<cache_path>.bbox`. See
/// `save_bbox`/`load_bbox`.
pub fn bbox_cache_path(cache_path: &Path) -> PathBuf {
    let mut os_string = cache_path.as_os_str().to_owned();
    os_string.push(".bbox");
    PathBuf::from(os_string)
}

/// Persists `bbox` (WGS84 degrees) as 4 little-endian `f64`s: min_lon,
/// min_lat, max_lon, max_lat. Written alongside the node store itself so a
/// later cache-reuse run (which skips `build_node_store`, the only place
/// the bbox is otherwise computed) can still load it back -- see
/// `node_store_matches`'s doc comment for the bug this fixes.
pub fn save_bbox(path: &Path, bbox: geo::Rect<f64>) -> io::Result<()> {
    let min = bbox.min();
    let max = bbox.max();
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&min.x.to_le_bytes());
    bytes[8..16].copy_from_slice(&min.y.to_le_bytes());
    bytes[16..24].copy_from_slice(&max.x.to_le_bytes());
    bytes[24..32].copy_from_slice(&max.y.to_le_bytes());
    std::fs::write(path, bytes)
}

/// Loads a bbox previously written by `save_bbox` -- `None` if the sidecar
/// doesn't exist, or isn't exactly 32 bytes (a corrupt/truncated/foreign
/// file, treated the same as "missing" rather than a hard error: the
/// caller falls back to treating this as a cache miss and rebuilds fresh).
pub fn load_bbox(path: &Path) -> Option<geo::Rect<f64>> {
    let bytes: [u8; 32] = std::fs::read(path).ok()?.try_into().ok()?;
    let min_lon = f64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let min_lat = f64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let max_lon = f64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let max_lat = f64::from_le_bytes(bytes[24..32].try_into().unwrap());
    Some(geo::Rect::new(
        geo::Coord { x: min_lon, y: min_lat },
        geo::Coord { x: max_lon, y: max_lat },
    ))
}

/// Decode PBF blocks in `[offset, offset + len)` as a parallel iterator.
pub(crate) fn iter_blocks(
    reader: &mut OsmReader<BufReader<File>>,
    offset: u64,
    len: u64,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> impl ParallelIterator<Item = osmpbf::PrimitiveBlock> {
    let end = offset + len;
    reader.set_position(offset).unwrap();

    reader
        .map(move |res| {
            let (blob, pos) = res.unwrap();
            progress(pos - offset, len);

            (blob, pos)
        })
        .take_while(move |(_, pos)| *pos <= end)
        .par_bridge()
        .map(|(blob, _)| blob.decode().unwrap())
}

/// Build the node-id bitset: one bit per node actually needed by extraction —
/// a way that **classifies** as a road/area (not merely matches the coarse tag
/// filter) or is referenced by a kept relation. Using the classifier here, not
/// just `way_tag_filter`, keeps the store from caching nodes of footways/paths/
/// service roads etc. that match `highway=*` but are never emitted — which
/// otherwise bloats the store several-fold and evicts itself from the cache.
pub fn build_node_bitset(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    way_tag_filter: &TagFilter,
    relation_ways: &HashSet<i64>,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> RankedBitSet {
    let bitset = BitSet::with_capacity(index.max_node_id as u64 + 1);
    let len = index.relations - index.ways;

    iter_blocks(reader, index.ways, len, progress).for_each(|block| {
        let string_map = block.stringtable.lookup_map();
        let classifier = BlockShapeClassifier::new(&string_map);
        let way_filter = way_tag_filter.get_sid_filter(&string_map);

        for grp in block.primitivegroup {
            if let Primitives::Ways(ways) = grp.primitives() {
                for way in ways {
                    let way = way.decode();
                    // A relation member (any tags), or a way the classifier will
                    // actually emit — the coarse filter is just a fast reject.
                    let keep = relation_ways.contains(&way.id)
                        || (way_filter.matches(&way.tags)
                            && !matches!(
                                classifier.classify_way(&way.tags),
                                ShapeClassification::Unknown
                            ));
                    if keep {
                        for node_id in way.refs {
                            bitset.set(node_id as u64);
                        }
                    }
                }
            }
        }
    });

    bitset.into_ranked()
}

/// `(min_lon, min_lat, max_lon, max_lat)` running accumulator, in WGS84
/// degrees -- `None` until the first node is seen.
type LonLatBoundsAcc = Option<(f64, f64, f64, f64)>;

#[inline]
fn fold_bounds(acc: LonLatBoundsAcc, lon: f64, lat: f64) -> LonLatBoundsAcc {
    Some(match acc {
        None => (lon, lat, lon, lat),
        Some((min_lon, min_lat, max_lon, max_lat)) => {
            (min_lon.min(lon), min_lat.min(lat), max_lon.max(lon), max_lat.max(lat))
        }
    })
}

#[inline]
fn merge_bounds(a: LonLatBoundsAcc, b: LonLatBoundsAcc) -> LonLatBoundsAcc {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some((a_lon0, a_lat0, a_lon1, a_lat1)), Some((b_lon0, b_lat0, b_lon1, b_lat1))) => Some((
            a_lon0.min(b_lon0),
            a_lat0.min(b_lat0),
            a_lon1.max(b_lon1),
            a_lat1.max(b_lat1),
        )),
    }
}

/// Fill the node location store: project each referenced node and write it at
/// `rank(id)`. Consumes `node_bitset` (the returned store owns it for lookups).
///
/// Also returns a bounding box (WGS84 degrees) over *every* decoded node in
/// the file -- unconditionally, not gated by `node_bitset` membership the
/// way the store itself is. A real OSM regional extract's own node set is
/// already clipped to the region of interest by whatever service produced
/// it, so this is a simpler, equally-representative proxy for "the
/// extract's coverage area" than restricting to just the tag-filtered
/// subset that ends up in the store -- and it's accumulated in this same
/// already-existing full pass over every node's lon/lat, not a new
/// dedicated scan. See `main.rs`'s `--clip-water-to-extract-bbox`, the
/// only current consumer.
pub fn build_node_store(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    node_bitset: RankedBitSet,
    cache_path: &Path,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> io::Result<(NodeStore, geo::Rect<f64>)> {
    let builder = SparseStoreBuilder::<PackedCoord>::create(cache_path, node_bitset)?;

    let bounds: LonLatBoundsAcc = iter_blocks(reader, 0, index.ways, progress)
        .fold(
            || None,
            |mut acc: LonLatBoundsAcc, block| {
                let granularity = block.granularity();
                let lat_offset = block.lat_offset();
                let lon_offset = block.lon_offset();

                for grp in block.primitivegroup {
                    // `set` is a no-op for ids not in the bitset, so we can feed every
                    // decoded node without a separate membership check. `acc` folds in
                    // every decoded node unconditionally (see this function's own doc
                    // comment for why that's deliberate, unlike the store itself).
                    match grp.primitives() {
                        Primitives::Nodes(nodes) => {
                            for node in nodes.decode_nodes(granularity, lat_offset, lon_offset) {
                                builder.set(node.id as u64, PackedCoord::from_lonlat(node.lon, node.lat));
                                acc = fold_bounds(acc, node.lon, node.lat);
                            }
                        }
                        Primitives::DenseNodes(dense) => {
                            for node in dense.decode_nodes(granularity, lat_offset, lon_offset) {
                                builder.set(node.id as u64, PackedCoord::from_lonlat(node.lon, node.lat));
                                acc = fold_bounds(acc, node.lon, node.lat);
                            }
                        }
                        _ => {}
                    }
                }
                acc
            },
        )
        .reduce(|| None, merge_bounds);

    let store = builder.finish()?;
    // Degenerate (zero nodes) fallback -- should never happen for a real
    // `.osm.pbf`, but a zero-size `Rect` at the origin is a harmless, obviously-
    // wrong-if-seen sentinel rather than a panic.
    let bbox = bounds
        .map(|(min_lon, min_lat, max_lon, max_lat)| {
            geo::Rect::new(geo::Coord { x: min_lon, y: min_lat }, geo::Coord { x: max_lon, y: max_lat })
        })
        .unwrap_or_else(|| geo::Rect::new(geo::Coord { x: 0.0, y: 0.0 }, geo::Coord { x: 0.0, y: 0.0 }));

    // Persisted so a later cache-reuse run (which never calls this function)
    // can still load the bbox back -- see `node_store_matches`'s doc comment.
    save_bbox(&bbox_cache_path(cache_path), bbox)?;

    Ok((store, bbox))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_roundtrips_through_sidecar_file() {
        let path = std::env::temp_dir().join(format!("node-cache-bbox-test-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let bbox = geo::Rect::new(
            geo::Coord { x: 139.0, y: 35.0 },
            geo::Coord { x: 141.0, y: 36.5 },
        );
        save_bbox(&path, bbox).unwrap();
        let loaded = load_bbox(&path).unwrap();
        assert_eq!(loaded.min(), bbox.min());
        assert_eq!(loaded.max(), bbox.max());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_or_corrupt_sidecar_loads_as_none() {
        let path = std::env::temp_dir().join(format!("node-cache-bbox-missing-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(load_bbox(&path).is_none());

        std::fs::write(&path, b"not a real bbox file").unwrap();
        assert!(load_bbox(&path).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bbox_cache_path_is_a_sidecar_of_the_node_cache_path() {
        let node_path = Path::new("/tmp/example.osm.nodes");
        assert_eq!(bbox_cache_path(node_path), Path::new("/tmp/example.osm.nodes.bbox"));
    }
}
