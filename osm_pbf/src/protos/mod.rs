//! OSM protobuf format decoder

pub mod osmpbf;

use crate::delta::IntoDelta;
use crate::tags::{IntoTagIterator, Tag, Tags};
use hashbrown::HashMap;
use itertools::izip;
pub use osmpbf::relation::MemberType;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Primitives {
    Nodes(Vec<osmpbf::Node>),
    DenseNodes(osmpbf::DenseNodes),
    Ways(Vec<osmpbf::Way>),
    Relations(Vec<osmpbf::Relation>),
    ChangeSets(Vec<osmpbf::ChangeSet>),
}

impl osmpbf::PrimitiveGroup {
    pub fn primitives(self) -> Primitives {
        if !self.nodes.is_empty() {
            return Primitives::Nodes(self.nodes);
        }

        if let Some(dense) = self.dense {
            return Primitives::DenseNodes(dense);
        }

        if !self.ways.is_empty() {
            return Primitives::Ways(self.ways);
        }

        if !self.relations.is_empty() {
            return Primitives::Relations(self.relations);
        }

        Primitives::ChangeSets(self.changesets)
    }
}

pub struct DecodedNode {
    pub id: i64,
    pub lat: f64,
    pub lon: f64,
    pub tags: Tags,
}

/// convert integer coordinate to f64
#[inline]
fn scale_coord(coord: i64, granularity: i32, offset: i64) -> f64 {
    0.000000001 * ((coord as i128 * granularity as i128) + offset as i128) as f64
}

pub trait DecodeNodes {
    fn decode_nodes(
        self,
        granularity: i32,
        lat_offset: i64,
        lon_offset: i64,
    ) -> impl Iterator<Item = DecodedNode>;
}

impl DecodeNodes for osmpbf::DenseNodes {
    fn decode_nodes(
        self,
        granularity: i32,
        lat_offset: i64,
        lon_offset: i64,
    ) -> impl Iterator<Item = DecodedNode> {
        izip!(
            self.id.into_iter().delta(),
            self.lat.into_iter().delta(),
            self.lon.into_iter().delta(),
            self.keys_vals.into_iter().tags()
        )
        .map(move |(id, lat, lon, tags)| DecodedNode {
            id,
            lat: scale_coord(lat, granularity, lat_offset),
            lon: scale_coord(lon, granularity, lon_offset),
            tags,
        })
    }
}

impl<T: IntoIterator<Item = osmpbf::Node>> DecodeNodes for T {
    fn decode_nodes(
        self,
        granularity: i32,
        lat_offset: i64,
        lon_offset: i64,
    ) -> impl Iterator<Item = DecodedNode> {
        self.into_iter().map(move |n| DecodedNode {
            id: n.id,
            lat: scale_coord(n.lat, granularity, lat_offset),
            lon: scale_coord(n.lon, granularity, lon_offset),
            tags: izip!(n.keys.into_iter(), n.vals.into_iter())
                .map(Tag::new)
                .collect(),
        })
    }
}

pub struct DecodedWay {
    pub id: i64,
    pub refs: Vec<i64>,
    pub tags: Tags,
}

impl osmpbf::Way {
    pub fn decode(self) -> DecodedWay {
        DecodedWay {
            id: self.id,
            refs: self.refs.into_iter().delta().collect(),
            tags: izip!(self.keys.into_iter(), self.vals.into_iter())
                .map(Tag::new)
                .collect(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RelationMember {
    pub id: i64,
    pub member_type: MemberType,
    pub role_tag: u32,
}

#[derive(Debug)]
pub struct DecodedRelation {
    pub id: i64,
    pub tags: Tags,
    pub members: Vec<RelationMember>,
}

impl osmpbf::Relation {
    pub fn decode(self) -> DecodedRelation {
        DecodedRelation {
            id: self.id,
            tags: izip!(self.keys.into_iter(), self.vals.into_iter())
                .map(Tag::new)
                .collect(),
            members: izip!(
                self.roles_sid.into_iter(),
                self.memids.into_iter().delta(),
                self.types
                    .into_iter()
                    .map(|f| osmpbf::relation::MemberType::try_from(f).unwrap())
            )
            .map(|(role, id, member_type)| RelationMember {
                id,
                member_type: member_type as _,
                role_tag: role as _,
            })
            .collect(),
        }
    }
}

impl osmpbf::StringTable {
    pub fn lookup_map(&self) -> HashMap<&[u8], u32> {
        self.s
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, sid)| (sid.as_slice(), i as u32))
            .collect()
    }
}
