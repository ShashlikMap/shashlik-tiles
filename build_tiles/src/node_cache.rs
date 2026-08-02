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
pub fn node_store_matches(cache_path: &Path, node_count: usize) -> bool {
    let expected = (node_count * size_of::<PackedCoord>()).max(1) as u64;
    std::fs::metadata(cache_path).map(|m| m.len()).ok() == Some(expected)
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

/// Fill the node location store: project each referenced node and write it at
/// `rank(id)`. Consumes `node_bitset` (the returned store owns it for lookups).
pub fn build_node_store(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    node_bitset: RankedBitSet,
    cache_path: &Path,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> io::Result<NodeStore> {
    let builder = SparseStoreBuilder::<PackedCoord>::create(cache_path, node_bitset)?;

    iter_blocks(reader, 0, index.ways, progress).for_each(|block| {
        let granularity = block.granularity();
        let lat_offset = block.lat_offset();
        let lon_offset = block.lon_offset();

        for grp in block.primitivegroup {
            // `set` is a no-op for ids not in the bitset, so we can feed every
            // decoded node without a separate membership check.
            match grp.primitives() {
                Primitives::Nodes(nodes) => {
                    for node in nodes.decode_nodes(granularity, lat_offset, lon_offset) {
                        builder.set(node.id as u64, PackedCoord::from_lonlat(node.lon, node.lat));
                    }
                }
                Primitives::DenseNodes(dense) => {
                    for node in dense.decode_nodes(granularity, lat_offset, lon_offset) {
                        builder.set(node.id as u64, PackedCoord::from_lonlat(node.lon, node.lat));
                    }
                }
                _ => {}
            }
        }
    });

    builder.finish()
}
