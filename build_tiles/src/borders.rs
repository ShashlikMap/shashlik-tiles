//! Country-border lines from OSM administrative-boundary relations.
//!
//! A `boundary=administrative`, `admin_level=2` relation groups the ways of a
//! country's outline; we take all member ways (deduped across neighbours), chain
//! them into continuous polylines, and push them as [`RoadKind::Border`] lines
//! (which ride the road/line pipeline: clip + `EdgeNode::Cut` + cross-tile weld).
//!
//! Runs on the shared [`crate::relation_features`] harness. Requires the boundary
//! relations to be in `relation_tag_filter` so their member-way nodes were cached
//! in the node store.

use crate::major_roads;
use crate::node_cache::NodeStore;
use crate::relation_features::{RelationFeature, ResolvedWays, StringMap, polylines};
use crate::shapes::{EdgeNode, Lanes, Road, RoadKind};
use crate::sink::ShapeSink;
use crate::tiler::sink::TileSink;
use geo::{Coord, LineString};
use hashbrown::HashSet;
use osm_pbf::tags::Tag;

/// Country borders (see module docs).
pub struct Borders;

impl RelationFeature for Borders {
    fn claims(&self, tags: &[Tag], smap: &StringMap) -> bool {
        let (Some(bk), Some(av), Some(lk), Some(tv)) = (
            smap.get("boundary".as_bytes()).copied(),
            smap.get("administrative".as_bytes()).copied(),
            smap.get("admin_level".as_bytes()).copied(),
            smap.get("2".as_bytes()).copied(),
        ) else {
            return false;
        };
        tags.iter().any(|t| t.key == bk && t.value == av)
            && tags.iter().any(|t| t.key == lk && t.value == tv)
    }

    fn build(
        &self,
        groups: &[Vec<i64>],
        resolved: &ResolvedWays,
        nodes: &NodeStore,
        sink: &TileSink,
    ) {
        // Flatten + dedup member ways (a shared border belongs to both neighbours).
        let members: HashSet<i64> = groups.iter().flatten().copied().collect();

        let strokes = major_roads::merge_polylines(polylines(members.into_iter(), resolved, nodes));
        for s in strokes {
            let line = LineString::from(
                s.into_iter().map(|[x, y]| Coord { x, y }).collect::<Vec<_>>(),
            );
            sink.push(
                Road::new(
                    RoadKind::Border,
                    line,
                    EdgeNode::Disconnected,
                    EdgeNode::Disconnected,
                    0,
                    Lanes::default(),
                    None,
                )
                .into(),
            );
        }
    }
}
