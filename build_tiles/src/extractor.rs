//! Extract geometries from OSM data

use crate::classifier::{BlockShapeClassifier, ShapeClassification};
use crate::node_cache::{NodeStore, PackedCoord};
use crate::shapes::{Area, AreaKind, EdgeNode, Label, LabelClass, Lanes, Poi, Road, RoadKind};
use crate::sink::ShapeSink;
use cache::bitset::BitSet;
use cache::index::OsmPbfIndex;
use geo::{Contains, Coord, InteriorPoint, LineString, Point, Polygon};
use hashbrown::{HashMap, HashSet};
use osm_pbf::{
    protos::{
        DecodeNodes, DecodedNode, MemberType, Primitives,
        osmpbf::{self, PrimitiveBlock},
    },
    reader::OsmReader,
    tags::TagFilter,
};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use tiles::LatLon;

fn iter_blocks(
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
        .map(|(blob, _)| {
            let block = blob.decode().unwrap();

            block
        })
}

/// Result of relation scan, return all filtered relation and ways/relations that they point to
#[derive(Default)]
pub struct FilteredRelations {
    /// Relations matched by filter
    pub relations: HashSet<i64>,
    /// Ways pointed to by relations
    pub relation_ways: HashSet<i64>,
    /// Relations pointed to by relations
    relation_relations: HashSet<i64>,
    /// Ways pointed to by relations of relations
    pub relation_relation_ways: HashMap<i64, Vec<i64>>,
}

impl FilteredRelations {
    /// Filter and extract relation context
    pub fn extract(
        reader: &mut OsmReader<BufReader<File>>,
        index: &OsmPbfIndex,
        relation_tag_filter: &TagFilter,
        progress: Arc<impl Fn(u64, u64) + Sync + Send>,
    ) -> Self {
        let len = index.end - index.relations;
        let mut this = iter_blocks(reader, index.relations, len, progress.clone())
            .map(|block| Self::scan_block(block, relation_tag_filter))
            .reduce(Self::default, |mut acc, rels| {
                acc.relations.extend(rels.relations);
                acc.relation_ways.extend(rels.relation_ways);
                acc.relation_relations.extend(rels.relation_relations);
                acc
            });

        let relation_relations = core::mem::take(&mut this.relation_relations);
        let relation_relation_ways = iter_blocks(reader, index.relations, len, progress.clone())
            .map(move |block| Self::rescan_block(block, &relation_relations))
            .reduce(HashMap::<i64, Vec<i64>>::new, |mut acc, r| {
                acc.extend(r);
                acc
            });

        for ways in relation_relation_ways.values() {
            this.relation_ways.extend(ways);
        }

        this.relation_relation_ways = relation_relation_ways;

        this
    }

    /// second OSM block scan pass to record ways for relations of relations
    fn rescan_block(
        block: PrimitiveBlock,
        relation_relations: &HashSet<i64>,
    ) -> HashMap<i64, Vec<i64>> {
        block
            .primitivegroup
            .into_iter()
            // limit to only relation primitives
            .filter_map(|group| {
                if let Primitives::Relations(relations) = group.primitives() {
                    return Some(relations);
                }

                None
            })
            .flatten()
            .filter_map(|relation| {
                if relation_relations.contains(&relation.id) {
                    return Some(relation.decode());
                }

                None
            })
            .map(|relation| {
                let ways =
                    relation
                        .members
                        .into_iter()
                        .fold(Vec::<i64>::new(), |mut ways, member| {
                            match member.member_type {
                                MemberType::Way => {
                                    ways.push(member.id);
                                }
                                _ => {}
                            }
                            ways
                        });

                (relation.id, ways)
            })
            // flatten all results
            .fold(
                HashMap::<i64, Vec<i64>>::new(),
                |mut acc, (relation, ways)| {
                    acc.entry(relation).or_default().extend(ways);

                    acc
                },
            )
    }

    /// first OSM block scan pass
    fn scan_block(block: PrimitiveBlock, relation_tag_filter: &TagFilter) -> Self {
        let string_map = block.stringtable.lookup_map();
        let relation_filter = relation_tag_filter.get_sid_filter(&string_map);

        block
            .primitivegroup
            .into_iter()
            // limit to only relation primitives
            .filter_map(|group| {
                if let Primitives::Relations(relations) = group.primitives() {
                    return Some(relations);
                }

                None
            })
            .flatten()
            // limit to relations that match tag filter
            .filter_map(|relation| {
                let relation = relation.decode();

                if relation_filter.matches(&relation.tags) {
                    return Some(relation);
                }

                None
            })
            // extract all ways and relations that filtered relation points to
            // these will be used to filer during way and second relation pass
            .map(|relation| {
                let (ways, relations) = relation.members.into_iter().fold(
                    (Vec::<i64>::new(), Vec::<i64>::new()),
                    |(mut ways, mut relations), member| {
                        match member.member_type {
                            MemberType::Way => {
                                ways.push(member.id);
                            }
                            MemberType::Relation => {
                                relations.push(member.id);
                            }
                            _ => {}
                        }
                        (ways, relations)
                    },
                );

                (relation.id, ways, relations)
            })
            // flatten all results
            .fold(
                Self::default(),
                |mut acc, (relation, ways, relation_relations)| {
                    acc.relations.insert(relation);
                    acc.relation_ways.extend(ways);
                    acc.relation_relations.extend(relation_relations);

                    acc
                },
            )
    }
}

/// Resolve a way's node refs to a projected Web Mercator line. Nodes missing
/// from the cache (e.g. referenced across an extract's boundary) are skipped.
fn resolve_line(refs: &[i64], nodes: &NodeStore) -> LineString {
    refs.iter()
        .filter_map(|&id| nodes.get(id as u64))
        .map(project)
        .collect()
}

/// Project a cached lon/lat coordinate to Web Mercator meters.
#[inline]
fn project(coord: PackedCoord) -> Coord {
    let mercator = LatLon::new(coord.lat(), coord.lon()).to_mercator();
    Coord {
        x: mercator.x(),
        y: mercator.y(),
    }
}

/// Is this way a closed ring (first ref == last ref, enough points for an area)?
#[inline]
fn is_ring(refs: &[i64]) -> bool {
    refs.len() >= 4 && refs.first() == refs.last()
}

/// Which nodes are shared between two or more roads (junctions).
///
/// A road endpoint is [`EdgeNode::Connected`] iff its node is a junction, else a
/// true dead-end. Backed by a node-id bitset, so cost is fixed by max node id.
pub struct RoadConnectivity {
    shared: BitSet,
}

impl RoadConnectivity {
    /// Classify a road endpoint from its node id.
    #[inline]
    pub fn edge(&self, node_id: i64) -> EdgeNode {
        if self.shared.contains(node_id as u64) {
            EdgeNode::Connected
        } else {
            EdgeNode::Disconnected
        }
    }

    /// Number of junction nodes — diagnostics only.
    pub fn junction_count(&self) -> u64 {
        self.shared.count_ones()
    }
}

/// Detect road junctions: a node referenced by two or more roads is shared.
///
/// One parallel pass over the ways section. Two node-id bitsets: `seen` (first
/// reference) and `shared` (set on the second+ reference via `test_and_set`).
/// `seen` is dropped on return; only `shared` is kept.
pub fn detect_road_connectivity(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    way_tag_filter: &TagFilter,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> RoadConnectivity {
    let capacity = index.max_node_id as u64 + 1;
    let seen = BitSet::with_capacity(capacity);
    let shared = BitSet::with_capacity(capacity);
    let len = index.relations - index.ways;

    iter_blocks(reader, index.ways, len, progress).for_each(|block| {
        let string_map = block.stringtable.lookup_map();
        let classifier = BlockShapeClassifier::new(&string_map);
        let way_filter = way_tag_filter.get_sid_filter(&string_map);

        for grp in block.primitivegroup {
            let Primitives::Ways(ways) = grp.primitives() else {
                continue;
            };

            for way in ways {
                let way = way.decode();
                if !way_filter.matches(&way.tags) {
                    continue;
                }

                if let ShapeClassification::Road(_) = classifier.classify_way(&way.tags) {
                    for &node_id in &way.refs {
                        // A node seen a second time (by any road) is a junction.
                        if seen.test_and_set(node_id as u64) {
                            shared.set(node_id as u64);
                        }
                    }
                }
            }
        }
    });

    RoadConnectivity { shared }
}

/// A way kept by the filter/classifier, carrying its node refs (and layer)
/// until its block's coords are resolved.
enum KeptWay {
    Road {
        kind: RoadKind,
        layer: i8,
        lanes: Lanes,
        refs: Vec<i64>,
        name: Option<String>,
    },
    Area {
        kind: AreaKind,
        layer: i8,
        floors: u8,
        refs: Vec<i64>,
        name: Option<String>,
    },
}

/// The label class for a named area of the given kind, or `None` for kinds we
/// don't label (buildings — too numerous, and z14-only anyway).
fn area_label_class(kind: AreaKind) -> Option<LabelClass> {
    match kind {
        AreaKind::Water => Some(LabelClass::Water),
        AreaKind::Forest | AreaKind::Grass => Some(LabelClass::Park),
        _ => None,
    }
}

/// Emit a point label for a named area, anchored at an interior point of the
/// polygon (guaranteed inside — so a lake's name never lands on an island). A
/// no-op if the area is unnamed, unlabeled, or has no interior point.
fn emit_area_label(polygon: &Polygon, kind: AreaKind, name: Option<String>, sink: &dyn ShapeSink) {
    let (Some(name), Some(class)) = (name, area_label_class(kind)) else {
        return;
    };
    if let Some(point) = polygon.interior_point() {
        let anchor = Coord {
            x: point.x(),
            y: point.y(),
        };
        sink.push(Label::new(anchor, class, name).into());
    }
}

/// Extract geometry from the ways section and stream it to `sink`.
pub fn extract_ways(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    way_tag_filter: &TagFilter,
    nodes: &NodeStore,
    connectivity: &RoadConnectivity,
    sink: &dyn ShapeSink,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) {
    let len = index.relations - index.ways;

    iter_blocks(reader, index.ways, len, progress).for_each(|block| {
        let string_map = block.stringtable.lookup_map();
        let classifier = BlockShapeClassifier::new(&string_map);
        let way_filter = way_tag_filter.get_sid_filter(&string_map);
        let strings = &block.stringtable.s;

        // Classify kept ways and gather every node ref they need.
        let mut kept: Vec<KeptWay> = Vec::new();
        let mut refs: Vec<i64> = Vec::new();
        for grp in block.primitivegroup {
            let Primitives::Ways(ways) = grp.primitives() else {
                continue;
            };
            for way in ways {
                let way = way.decode();
                if !way_filter.matches(&way.tags) {
                    continue;
                }
                let layer = classifier.layer_of(&way.tags, strings);
                let name = classifier.name_of(&way.tags, strings);
                match classifier.classify_way(&way.tags) {
                    ShapeClassification::Road(kind) if way.refs.len() >= 2 => {
                        let lanes = classifier.lanes_of(&way.tags, strings);
                        refs.extend_from_slice(&way.refs);
                        kept.push(KeptWay::Road {
                            kind,
                            layer,
                            lanes,
                            refs: way.refs,
                            name,
                        });
                    }
                    ShapeClassification::Area(kind) if is_ring(&way.refs) => {
                        let floors = classifier.floors_of(&way.tags, strings);
                        refs.extend_from_slice(&way.refs);
                        kept.push(KeptWay::Area {
                            kind,
                            layer,
                            floors,
                            refs: way.refs,
                            name,
                        });
                    }
                    _ => {}
                }
            }
        }
        if kept.is_empty() {
            return;
        }

        // Resolve unique refs in sorted (node-id) order — near-sequential
        // store access instead of a random gather.
        refs.sort_unstable();
        refs.dedup();
        let mut coords: HashMap<i64, Coord> = HashMap::with_capacity(refs.len());
        for id in refs {
            if let Some(coord) = nodes.get(id as u64).map(project) {
                coords.insert(id, coord);
            }
        }

        // Build geometry from the resolved coords and stream it out.
        let line = |refs: &[i64]| -> LineString {
            refs.iter()
                .filter_map(|id| coords.get(id).copied())
                .collect()
        };
        for way in kept {
            match way {
                KeptWay::Road {
                    kind,
                    layer,
                    lanes,
                    refs,
                    name,
                } => {
                    let start = connectivity.edge(refs[0]);
                    let end = connectivity.edge(refs[refs.len() - 1]);
                    sink.push(Road::new(kind, line(&refs), start, end, layer, lanes, name).into());
                }
                KeptWay::Area {
                    kind,
                    layer,
                    floors,
                    refs,
                    name,
                } => {
                    let polygon = Polygon::new(line(&refs), Vec::new());
                    emit_area_label(&polygon, kind, name, sink);
                    sink.push(Area::new(kind, polygon, layer, floors).into());
                }
            }
        }
    });
}

/// Extract place-name point labels (`place=city/town/village/…` nodes with a
/// `name`) from the nodes section and stream them to `sink`. Point labels are
/// not clipped — each is placed in the single tile containing its anchor.
///
/// Runs in parallel; `sink` is shared across workers (see `ShapeSink`).
pub fn extract_place_labels(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    sink: &dyn ShapeSink,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) {
    iter_blocks(reader, 0, index.ways, progress).for_each(|block| {
        let string_map = block.stringtable.lookup_map();
        let classifier = BlockShapeClassifier::new(&string_map);
        let strings = &block.stringtable.s;
        let granularity = block.granularity();
        let lat_offset = block.lat_offset();
        let lon_offset = block.lon_offset();

        let emit = |node: DecodedNode| {
            // Place-name label (needs a name).
            if let Some(class) = classifier.classify_place(&node.tags)
                && let Some(name) = classifier.name_of(&node.tags, strings)
            {
                let anchor = project(PackedCoord::from_lonlat(node.lon, node.lat));
                sink.push(Label::new(anchor, class, name).into());
            }
            // Point-of-interest symbol (no name) — same node pass, so no extra
            // scan over the (large) node section.
            if let Some(kind) = classifier.classify_poi(&node.tags) {
                let anchor = project(PackedCoord::from_lonlat(node.lon, node.lat));
                sink.push(Poi::new(anchor, kind).into());
            }
        };

        for grp in block.primitivegroup {
            match grp.primitives() {
                Primitives::Nodes(nodes) => {
                    for node in nodes.decode_nodes(granularity, lat_offset, lon_offset) {
                        emit(node);
                    }
                }
                Primitives::DenseNodes(dense) => {
                    for node in dense.decode_nodes(granularity, lat_offset, lon_offset) {
                        emit(node);
                    }
                }
                _ => {}
            }
        }
    });
}

/// Collect the node refs of every way referenced by a filtered relation, keyed
/// by way id. This is the in-memory stand-in for the future mmap way-store: it
/// is bounded by relation membership, not by total ways, and stores refs (not
/// coords) so multipolygon rings can be stitched by exact node-id equality.
pub fn collect_relation_member_ways(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    relation_ways: &HashSet<i64>,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) -> HashMap<i64, Vec<i64>> {
    let len = index.relations - index.ways;

    iter_blocks(reader, index.ways, len, progress)
        .map(|block| {
            let mut ways = HashMap::<i64, Vec<i64>>::new();
            for grp in block.primitivegroup {
                let Primitives::Ways(block_ways) = grp.primitives() else {
                    continue;
                };
                for way in block_ways {
                    if relation_ways.contains(&way.id) {
                        let way = way.decode();
                        ways.insert(way.id, way.refs);
                    }
                }
            }
            ways
        })
        .reduce(HashMap::new, |mut acc, part| {
            acc.extend(part);
            acc
        })
}

/// Assemble unordered way fragments (each a list of node ids) into closed rings.
///
/// Fragments already closed (first == last) pass through; open fragments are
/// chained end-to-end via shared endpoint node ids, reversing as needed, until
/// they close. Chains that cannot be closed (malformed relations) are dropped.
fn assemble_rings(fragments: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    let mut rings: Vec<Vec<i64>> = Vec::new();
    let mut open: Vec<Vec<i64>> = Vec::new();

    for frag in fragments {
        if frag.len() < 2 {
            continue;
        }
        if frag.first() == frag.last() {
            rings.push(frag);
        } else {
            open.push(frag);
        }
    }

    if open.is_empty() {
        return rings;
    }

    // Index open fragments by both of their endpoint node ids.
    let mut endpoints: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, frag) in open.iter().enumerate() {
        endpoints.entry(*frag.first().unwrap()).or_default().push(i);
        endpoints.entry(*frag.last().unwrap()).or_default().push(i);
    }

    let mut used = vec![false; open.len()];
    for start in 0..open.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut ring = open[start].clone();

        loop {
            if ring.first() == ring.last() && ring.len() >= 4 {
                rings.push(std::mem::take(&mut ring));
                break;
            }

            let end = *ring.last().unwrap();
            let next = endpoints
                .get(&end)
                .and_then(|cands| cands.iter().copied().find(|&i| !used[i]));

            match next {
                Some(i) => {
                    used[i] = true;
                    let mut frag = open[i].clone();
                    if *frag.last().unwrap() == end {
                        frag.reverse();
                    }
                    // `frag` now starts at `end`; append without repeating the join node.
                    ring.extend_from_slice(&frag[1..]);
                }
                // Dangling chain — the relation is incomplete; drop it.
                None => break,
            }
        }
    }

    rings
}

/// Resolve assembled outer/inner rings to geometry, associate inners with the
/// outer that contains them, and push one `Area` per outer ring to `sink`.
fn emit_multipolygon(
    kind: AreaKind,
    layer: i8,
    outer_rings: Vec<Vec<i64>>,
    inner_rings: Vec<Vec<i64>>,
    floors: u8,
    name: Option<String>,
    nodes: &NodeStore,
    sink: &dyn ShapeSink,
) {
    let outers: Vec<LineString> = outer_rings
        .iter()
        .map(|ring| resolve_line(ring, nodes))
        .collect();
    let inners: Vec<LineString> = inner_rings
        .iter()
        .map(|ring| resolve_line(ring, nodes))
        .collect();

    // Assign holes to the outer that contains them, then emit one polygon each.
    let polygons: Vec<(LineString, Vec<LineString>)> = if outers.len() == 1 {
        // Common case: a single outer ring owns all holes — skip point-in-polygon.
        vec![(outers.into_iter().next().unwrap(), inners)]
    } else {
        let mut polygons: Vec<(LineString, Vec<LineString>)> = outers
            .into_iter()
            .map(|outer| (outer, Vec::new()))
            .collect();
        for inner in inners {
            let Some(point) = inner.0.first().map(|coord| Point::from(*coord)) else {
                continue;
            };
            if let Some(slot) = polygons
                .iter_mut()
                .find(|(outer, _)| Polygon::new(outer.clone(), Vec::new()).contains(&point))
            {
                slot.1.push(inner);
            }
            // An inner outside every outer is malformed — drop it.
        }
        polygons
    };

    // One label for the whole relation, at the largest polygon's interior point.
    if name.is_some() {
        if let Some(idx) = largest_polygon(&polygons) {
            let poly = Polygon::new(polygons[idx].0.clone(), polygons[idx].1.clone());
            emit_area_label(&poly, kind, name, sink);
        }
    }

    for (outer, inners) in polygons {
        sink.push(Area::from_rings(kind, outer, inners, layer, floors).into());
    }
}

/// Index of the polygon with the largest bounding box (a cheap proxy for the
/// "main" polygon to hang a multipolygon's label on).
fn largest_polygon(polygons: &[(LineString, Vec<LineString>)]) -> Option<usize> {
    polygons
        .iter()
        .enumerate()
        .max_by(|a, b| bbox_area(&a.1.0).total_cmp(&bbox_area(&b.1.0)))
        .map(|(i, _)| i)
}

fn bbox_area(ring: &LineString) -> f64 {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for c in &ring.0 {
        min_x = min_x.min(c.x);
        min_y = min_y.min(c.y);
        max_x = max_x.max(c.x);
        max_y = max_y.max(c.y);
    }
    if max_x < min_x {
        0.0
    } else {
        (max_x - min_x) * (max_y - min_y)
    }
}

/// Extract multipolygon areas from the relations section and stream them to
/// `sink`. For each relation passing `relation_tag_filter` and classified as an
/// `Area`, member ways are gathered by role (outer/inner) from `member_ways`,
/// stitched into rings, and emitted as `Area` polygons.
///
/// Runs in parallel; `sink` is shared across workers (see `ShapeSink`).
/// Sub-relation members (relations of relations) are not expanded yet.
pub fn extract_relations(
    reader: &mut OsmReader<BufReader<File>>,
    index: &OsmPbfIndex,
    relation_tag_filter: &TagFilter,
    nodes: &NodeStore,
    member_ways: &HashMap<i64, Vec<i64>>,
    sink: &dyn ShapeSink,
    progress: Arc<impl Fn(u64, u64) + Sync + Send>,
) {
    let len = index.end - index.relations;

    iter_blocks(reader, index.relations, len, progress).for_each(|block| {
        let string_map = block.stringtable.lookup_map();
        let classifier = BlockShapeClassifier::new(&string_map);
        let relation_filter = relation_tag_filter.get_sid_filter(&string_map);
        let inner_sid = string_map.get(b"inner".as_ref()).copied();
        let strings = &block.stringtable.s;

        for grp in block.primitivegroup {
            let Primitives::Relations(relations) = grp.primitives() else {
                continue;
            };

            for relation in relations {
                let relation = relation.decode();
                if !relation_filter.matches(&relation.tags) {
                    continue;
                }
                let ShapeClassification::Area(kind) = classifier.classify_relation(&relation.tags)
                else {
                    continue;
                };
                let layer = classifier.layer_of(&relation.tags, strings);
                let floors = classifier.floors_of(&relation.tags, strings);
                let name = classifier.name_of(&relation.tags, strings);

                let mut outer_frags: Vec<Vec<i64>> = Vec::new();
                let mut inner_frags: Vec<Vec<i64>> = Vec::new();

                for member in &relation.members {
                    if member.member_type != MemberType::Way {
                        continue;
                    }
                    let Some(refs) = member_ways.get(&member.id) else {
                        continue;
                    };
                    // Role "inner" marks a hole; anything else is treated as outer.
                    if Some(member.role_tag) == inner_sid {
                        inner_frags.push(refs.clone());
                    } else {
                        outer_frags.push(refs.clone());
                    }
                }

                let outer_rings = assemble_rings(outer_frags);
                if outer_rings.is_empty() {
                    continue;
                }
                let inner_rings = assemble_rings(inner_frags);

                emit_multipolygon(
                    kind,
                    layer,
                    outer_rings,
                    inner_rings,
                    floors,
                    name,
                    nodes,
                    sink,
                );
            }
        }
    });
}
