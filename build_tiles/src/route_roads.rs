//! Low-zoom major-road network from OSM route relations.
//!
//! A `type=route`, `route=road` relation is treated as important if **any** of its
//! member ways is a motorway; then *all* of that route's member ways (motorway
//! plus trunk/primary connectors) become the low-zoom network. Runs on the shared
//! [`crate::relation_features`] harness — geometry resolution is done once and
//! shared with the other relation features.

use crate::major_roads;
use crate::relation_features::{RelationFeature, ResolvedWay, ResolvedWays, StringMap, polylines};
use crate::shapes::{EdgeNode, Lanes, Road, RoadKind};
use crate::tiler::sink::TileSink;
use crate::classifier::ShapeClassification;
use crate::node_cache::NodeStore;
use geo::{Coord, LineString};
use hashbrown::HashSet;
use osm_pbf::tags::Tag;

/// Route-derived major roads (see module docs).
pub struct RouteRoads;

impl RelationFeature for RouteRoads {
    fn claims(&self, tags: &[Tag], smap: &StringMap) -> bool {
        let (Some(route_key), Some(road_val)) = (
            smap.get("route".as_bytes()).copied(),
            smap.get("road".as_bytes()).copied(),
        ) else {
            return false;
        };
        tags.iter().any(|t| t.key == route_key && t.value == road_val)
    }

    fn build(
        &self,
        groups: &[Vec<i64>],
        resolved: &ResolvedWays,
        nodes: &NodeStore,
        sink: &TileSink,
    ) {
        // Keep every member way of a route that touches a motorway.
        let mut important: HashSet<i64> = HashSet::new();
        for route in groups {
            let touches_motorway = route.iter().any(|w| {
                matches!(
                    resolved.get(w),
                    Some(ResolvedWay {
                        class: ShapeClassification::Road(RoadKind::Motorway),
                        ..
                    })
                )
            });
            if touches_motorway {
                important.extend(route.iter().copied());
            }
        }

        // Chain into continuous strokes and clip into the coarse tiles.
        let strokes = major_roads::merge_polylines(polylines(important.into_iter(), resolved, nodes));
        for s in strokes {
            let line = LineString::from(
                s.into_iter().map(|[x, y]| Coord { x, y }).collect::<Vec<_>>(),
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
}
