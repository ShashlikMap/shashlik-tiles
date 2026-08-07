//! Shared harness for feature types built from OSM relations + their member
//! ways (country borders, route-derived major roads, and future ones).
//!
//! Each such feature otherwise needs two full section scans of the archive — a
//! relations scan and a ways scan — and the ways scan is the expensive one. This
//! harness runs them once, shared across all features

use crate::classifier::{BlockShapeClassifier, ShapeClassification};
use crate::node_cache::{NodeStore, PackedCoord, iter_blocks};
use crate::tiler::sink::TileSink;
use cache::index::OsmPbfIndex;
use hashbrown::{HashMap, HashSet};
use osm_pbf::protos::{MemberType, Primitives};
use osm_pbf::reader::OsmReader;
use osm_pbf::tags::Tag;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tiles::LatLon;

/// Per-block interned string map: `tag bytes -> string id`. Keys are borrowed
/// from the block's string table.
pub type StringMap<'a> = HashMap<&'a [u8], u32>;

/// A member way resolved in the shared ways pass.
pub struct ResolvedWay {
    pub refs: Vec<i64>,
    pub class: ShapeClassification,
}

/// All resolved member ways, keyed by way id.
pub type ResolvedWays = HashMap<i64, ResolvedWay>;

/// A feature type assembled from OSM relations and their member ways.
pub trait RelationFeature: Sync {
    /// Whether a relation (its `tags`, with the block's interned `smap`) belongs
    /// to this feature.
    fn claims(&self, tags: &[Tag], smap: &StringMap) -> bool;

    /// Assemble geometry and push to `sink`. `groups[i]` is the member way ids of
    /// the i-th relation this feature claimed; `resolved` gives every needed way's
    /// refs + classification.
    fn build(
        &self,
        groups: &[Vec<i64>],
        resolved: &ResolvedWays,
        nodes: &NodeStore,
        sink: &TileSink,
    );
}

/// Run all relation-driven `features` over the archive with a single relations
/// pass and a single (shared) ways pass.
pub fn run(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    nodes: &NodeStore,
    sink: &TileSink,
    features: &[&dyn RelationFeature],
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) {
    // 1. One relations pass → (feature index, member way ids) per claimed relation.
    let claimed = scan_relations(reader, index, features, progress.clone());
    if claimed.is_empty() {
        return;
    }

    // 2. One ways pass over the union of all claimed member ways.
    let needed: HashSet<i64> = claimed
        .iter()
        .flat_map(|(_, ways)| ways.iter().copied())
        .collect();
    let resolved = resolve_ways(reader, index, &needed, progress);

    // 3. Hand each feature its own relation groups + the shared resolution.
    for (i, feature) in features.iter().enumerate() {
        let groups: Vec<Vec<i64>> = claimed
            .iter()
            .filter(|(fi, _)| *fi == i)
            .map(|(_, ways)| ways.clone())
            .collect();
        if !groups.is_empty() {
            feature.build(&groups, &resolved, nodes, sink);
        }
    }
}

/// Resolve member way ids to Web-Mercator polylines via the node store, dropping
/// ways with fewer than two resolved vertices. Shared by feature `build`s.
pub fn polylines(
    ids: impl Iterator<Item = i64>,
    resolved: &ResolvedWays,
    nodes: &NodeStore,
) -> Vec<Vec<[f64; 2]>> {
    ids.filter_map(|id| resolved.get(&id))
        .filter_map(|w| {
            let line: Vec<[f64; 2]> = w
                .refs
                .iter()
                .filter_map(|n| nodes.get(*n as u64))
                .map(merc)
                .collect();
            (line.len() >= 2).then_some(line)
        })
        .collect()
}

/// Web-Mercator metres of a cached node coordinate.
#[inline]
fn merc(c: PackedCoord) -> [f64; 2] {
    let m = LatLon::new(c.lat(), c.lon()).to_mercator();
    [m.x(), m.y()]
}

/// One relations pass: decode each relation once, offer it to every feature, and
/// emit `(feature index, member way ids)` for each claim.
fn scan_relations(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    features: &[&dyn RelationFeature],
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> Vec<(usize, Vec<i64>)> {
    let len = index.end - index.relations;
    iter_blocks(reader, index.relations, len, progress)
        .map(|block| {
            let smap = block.stringtable.lookup_map();
            let mut out: Vec<(usize, Vec<i64>)> = Vec::new();
            for grp in block.primitivegroup {
                let Primitives::Relations(rels) = grp.primitives() else {
                    continue;
                };
                for rel in rels {
                    let rel = rel.decode();
                    let members: Vec<i64> = rel
                        .members
                        .iter()
                        .filter(|m| matches!(m.member_type, MemberType::Way))
                        .map(|m| m.id)
                        .collect();
                    if members.is_empty() {
                        continue;
                    }
                    for (fi, feature) in features.iter().enumerate() {
                        if feature.claims(&rel.tags, &smap) {
                            out.push((fi, members.clone()));
                        }
                    }
                }
            }
            out
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
}

/// One ways pass: for every way in `needed`, record its node refs and its
/// classification (some features filter members by class, e.g. motorway).
fn resolve_ways(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    needed: &HashSet<i64>,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> ResolvedWays {
    let len = index.relations - index.ways;
    iter_blocks(reader, index.ways, len, progress)
        .map(|block| {
            let smap = block.stringtable.lookup_map();
            let classifier = BlockShapeClassifier::new(&smap);
            let mut out: ResolvedWays = HashMap::new();
            for grp in block.primitivegroup {
                let Primitives::Ways(ways) = grp.primitives() else {
                    continue;
                };
                for way in ways {
                    if !needed.contains(&way.id) {
                        continue; // cheap check before decoding
                    }
                    let way = way.decode();
                    let class = classifier.classify_way(&way.tags);
                    out.insert(
                        way.id,
                        ResolvedWay {
                            refs: way.refs,
                            class,
                        },
                    );
                }
            }
            out
        })
        .reduce(HashMap::new, |mut a, b| {
            a.extend(b);
            a
        })
}
