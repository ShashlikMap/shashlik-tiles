//! Geometry model for shapes extracted from OSM.
//!
//! Pure data types in Web Mercator meters (EPSG:3857) — the working space
//! for the tiler. The extraction logic (reading OSM, resolving node coordinates,
//! assembling multipolygons, detecting connectivity) lives in
//! `crate::extractor`; this module only defines the model and its constructors.

use geo::{Coord, LineString, Polygon};

pub use util::feature::{AreaKind, EdgeNode, LabelClass, Lanes, RoadKind};

/// A text label anchored at a single point in Web Mercator meters. Point labels
/// are never clipped — a label is emitted only in the tile containing its
/// anchor, and the renderer may overflow into loaded neighbours.
#[derive(Debug, Clone)]
pub struct Label {
    pub anchor: Coord,
    pub class: LabelClass,
    pub name: String,
}

impl Label {
    pub fn new(anchor: Coord, class: LabelClass, name: String) -> Self {
        Self {
            anchor,
            class,
            name,
        }
    }
}

/// A linear feature (road, path, …) in Web Mercator meters.
///
/// `layer` is the OSM `layer` tag (bridge/tunnel stacking level, 0 by default).
/// `name` is the OSM `name` tag, rendered along the line (`None` if unnamed).
/// `lanes` splits the carriageway into forward/backward counts.
#[derive(Debug, Clone)]
pub struct Road {
    pub kind: RoadKind,
    pub geometry: LineString,
    pub start: EdgeNode,
    pub end: EdgeNode,
    pub layer: i8,
    pub lanes: Lanes,
    pub name: Option<String>,
}

impl Road {
    pub fn new(
        kind: RoadKind,
        geometry: LineString,
        start: EdgeNode,
        end: EdgeNode,
        layer: i8,
        lanes: Lanes,
        name: Option<String>,
    ) -> Self {
        Self {
            kind,
            geometry,
            start,
            end,
            layer,
            lanes,
            name,
        }
    }
}

/// An areal feature (water, forest, building, …) in Web Mercator meters: an
/// outer ring plus zero or more inner rings (holes). `layer` is the OSM `layer`
/// tag; `floors` is the building floor count (`building:levels`, default 1;
/// meaningful only for `AreaKind::Building`).
#[derive(Debug, Clone)]
pub struct Area {
    pub kind: AreaKind,
    pub geometry: Polygon,
    pub layer: i8,
    pub floors: u8,
}

impl Area {
    pub fn new(kind: AreaKind, geometry: Polygon, layer: i8, floors: u8) -> Self {
        Self {
            kind,
            geometry,
            layer,
            floors,
        }
    }

    /// Build an area from rings already assembled (e.g. from a multipolygon
    /// relation's outer/inner member ways).
    pub fn from_rings(
        kind: AreaKind,
        outer: LineString,
        inners: Vec<LineString>,
        layer: i8,
        floors: u8,
    ) -> Self {
        Self::new(kind, Polygon::new(outer, inners), layer, floors)
    }
}

/// A geometry ready to stream to the tiler.
#[derive(Debug, Clone)]
pub enum Shape {
    Road(Road),
    Area(Area),
    Label(Label),
}

impl Shape {
    /// The OSM `layer` stacking level of the underlying geometry (labels: 0).
    #[allow(unused)]
    pub fn layer(&self) -> i8 {
        match self {
            Shape::Road(road) => road.layer,
            Shape::Area(area) => area.layer,
            Shape::Label(_) => 0,
        }
    }
}

impl From<Road> for Shape {
    fn from(road: Road) -> Self {
        Shape::Road(road)
    }
}

impl From<Area> for Shape {
    fn from(area: Area) -> Self {
        Shape::Area(area)
    }
}

impl From<Label> for Shape {
    fn from(label: Label) -> Self {
        Shape::Label(label)
    }
}
