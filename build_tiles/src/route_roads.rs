//! Low-zoom important-road network from OSM route relations

use crate::classifier::{BlockShapeClassifier, ShapeClassification};
use crate::node_cache::{NodeStore, iter_blocks};
use crate::shapes::{EdgeNode, Lanes, Road, RoadKind};
use crate::tiler::sink::TileSink;
use crate::{major_roads, node_cache::PackedCoord};
use cache::index::OsmPbfIndex;
use geo::{Coord, LineString};
use hashbrown::{HashMap, HashSet};
use osm_pbf::protos::{MemberType, Primitives};
use osm_pbf::reader::OsmReader;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tiles::LatLon;

/// Build the low-zoom major-road network from route relations and push it to
/// `sink` (as merged `RoadKind::MajorRoad` lines clipped into coarse tiles).
pub fn build(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    nodes: &NodeStore,
    sink: &TileSink,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) {
    // Every road route's member way ids.
    let routes = scan_routes(reader, index, progress.clone());
    let members: HashSet<i64> = routes.iter().flatten().copied().collect();
    if members.is_empty() {
        return;
    }

    // For each member way: is it a motorway, and what are its node refs?
    let (motorway, refs) = scan_members(reader, index, &members, progress);

    // Keep every member way of a route that touches a motorway.
    let mut important: HashSet<i64> = HashSet::new();
    for route in &routes {
        if route.iter().any(|w| motorway.contains(w)) {
            important.extend(route.iter().copied());
        }
    }

    // Resolve those ways to Web-Mercator polylines via the node store.
    let polylines: Vec<Vec<[f64; 2]>> = important
        .iter()
        .filter_map(|id| refs.get(id))
        .filter_map(|refs| {
            let line: Vec<[f64; 2]> = refs
                .iter()
                .filter_map(|n| nodes.get(*n as u64))
                .map(merc)
                .collect();
            (line.len() >= 2).then_some(line)
        })
        .collect();

    // Chain + dedup into strokes, then clip into the coarse tiles.
    let strokes = major_roads::merge_polylines(polylines);
    for s in strokes {
        let line = LineString::from(
            s.into_iter()
                .map(|[x, y]| Coord { x, y })
                .collect::<Vec<_>>(),
        );
        sink.push_aggregated_road(&Road::new(
            RoadKind::MajorRoad,
            line,
            EdgeNode::Disconnected,
            EdgeNode::Disconnected,
            0,
            Lanes::default(),
            None,
        ));
    }
}

/// Web-Mercator metres of a cached node coordinate.
#[inline]
fn merc(c: PackedCoord) -> [f64; 2] {
    let m = LatLon::new(c.lat(), c.lon()).to_mercator();
    [m.x(), m.y()]
}

/// Collect the member way ids of every `type=route`, `route=road` relation, one
/// `Vec<way_id>` per route.
fn scan_routes(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> Vec<Vec<i64>> {
    let len = index.end - index.relations;
    iter_blocks(reader, index.relations, len, progress)
        .map(|block| {
            let smap = block.stringtable.lookup_map();
            let route_key = smap.get("route".as_bytes()).copied();
            let road_val = smap.get("road".as_bytes()).copied();
            let mut routes: Vec<Vec<i64>> = Vec::new();
            // A block with neither the `route` key nor `road` value can't hold a
            // road route.
            if let (Some(rk), Some(rv)) = (route_key, road_val) {
                for grp in block.primitivegroup {
                    let Primitives::Relations(rels) = grp.primitives() else {
                        continue;
                    };
                    for rel in rels {
                        let rel = rel.decode();
                        if !rel.tags.iter().any(|t| t.key == rk && t.value == rv) {
                            continue;
                        }
                        let ways: Vec<i64> = rel
                            .members
                            .into_iter()
                            .filter(|m| matches!(m.member_type, MemberType::Way))
                            .map(|m| m.id)
                            .collect();
                        if !ways.is_empty() {
                            routes.push(ways);
                        }
                    }
                }
            }
            routes
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        })
}

/// For each way in `members`, record whether it classifies as a motorway and its
/// node refs (for geometry).
fn scan_members(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    members: &HashSet<i64>,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> (HashSet<i64>, HashMap<i64, Vec<i64>>) {
    let len = index.relations - index.ways;
    iter_blocks(reader, index.ways, len, progress)
        .map(|block| {
            let smap = block.stringtable.lookup_map();
            let classifier = BlockShapeClassifier::new(&smap);
            let mut motorway: HashSet<i64> = HashSet::new();
            let mut refs: HashMap<i64, Vec<i64>> = HashMap::new();
            for grp in block.primitivegroup {
                let Primitives::Ways(ways) = grp.primitives() else {
                    continue;
                };
                for way in ways {
                    if !members.contains(&way.id) {
                        continue; // cheap check before decoding
                    }
                    let way = way.decode();
                    // Only keep member ways that are actually roads. A `route=road`
                    // relation can include a ferry crossing (a `route=ferry` way,
                    // no `highway` tag) where the road continues by boat — those
                    // must not be drawn as roads across the water.
                    let ShapeClassification::Road(kind) = classifier.classify_way(&way.tags) else {
                        continue;
                    };
                    refs.insert(way.id, way.refs);
                    if kind == RoadKind::Motorway {
                        motorway.insert(way.id);
                    }
                }
            }
            (motorway, refs)
        })
        .reduce(
            || (HashSet::new(), HashMap::new()),
            |mut a, b| {
                a.0.extend(b.0);
                a.1.extend(b.1);
                a
            },
        )
}
