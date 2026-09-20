//! Feature classification enums shared by the tile builder and the decoder.
//!
//! These are the semantic kinds the tile format encodes as `u8` ordinals; the
//! writer maps geometry to them, the decoder maps them back. Kept dependency-free
//! so both sides (and a client renderer) can use them.

/// Road classification. `self as u8` is the wire ordinal; `RoadKind::from_u8`)
/// is its inverse.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoadKind {
    Motorway,
    Trunk,
    Primary,
    Secondary,
    Tertiary,
    Unclassified,
    Residential,
    LivingStreet,
    Service,
    Footway,
    Raceway,
    Unknown,
    MajorRoad,
    /// Heavy mainline railway (`railway=rail`) — the significant rail tier.
    Rail,
    /// Minor railway: light rail / narrow gauge. Local/urban, shown only at
    /// finer zooms than [`Rail`](RoadKind::Rail).
    RailMinor,
    /// Country border (`boundary=administrative`, `admin_level=2`). A linear
    /// overlay riding the road pipeline; the renderer styles it distinctly.
    Border,
}

impl RoadKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        use RoadKind::*;
        Some(match v {
            0 => Motorway,
            1 => Trunk,
            2 => Primary,
            3 => Secondary,
            4 => Tertiary,
            5 => Unclassified,
            6 => Residential,
            7 => LivingStreet,
            8 => Service,
            9 => Footway,
            10 => Raceway,
            11 => Unknown,
            12 => MajorRoad,
            13 => Rail,
            14 => RailMinor,
            15 => Border,
            _ => return None,
        })
    }

    /// Build threshold: coarsest zoom a road of this kind is materialized
    /// into (a tile at zoom `z` stores it only if `z >= min_zoom`). Controls what
    /// the archive contains, not what the renderer shows — see `Self::display_min_zoom`
    pub fn min_zoom(self) -> u8 {
        use RoadKind::*;
        match self {
            MajorRoad => 4,
            Border => 4,
            Motorway | Trunk | Primary => 10,
            Secondary => 12,
            Rail => 12,
            RailMinor => 14,
            Tertiary | Unclassified | Residential | Raceway => 14,
            LivingStreet | Service | Footway | Unknown => 16,
        }
    }

    /// Approximate real-world full carriageway width in metres for a given total
    /// lane count (`0` = untagged → a per-kind default). Drives the
    /// proportional stroke width at high zoom; the renderer applies a per-type
    /// pixel floor on top for low zoom.
    pub fn width_m(self, total_lanes: u8) -> f32 {
        const LANE_M: f32 = 3.7;
        let lanes = if total_lanes > 0 {
            total_lanes as f32
        } else {
            self.default_lanes()
        };
        lanes * LANE_M
    }

    /// Fallback full-width lane count for roads with no `lanes` tag.
    fn default_lanes(self) -> f32 {
        use RoadKind::*;
        match self {
            Motorway | MajorRoad => 6.0,
            Trunk | Primary => 4.0,
            Secondary | Tertiary => 2.0,
            Unclassified | Residential | LivingStreet | Raceway | Unknown => 2.0,
            Service | Footway => 1.0,
            Rail | RailMinor => 1.0,
            Border => 1.0, // not a carriageway; the pixel floor dominates
        }
    }

    /// Display threshold: coarsest camera zoom at which a road of this kind is
    /// shown. Independent of `Self::min_zoom` so the renderer can
    /// reveal / hide classes without rebuilding tiles
    pub fn display_min_zoom(self) -> u8 {
        use RoadKind::*;
        match self {
            MajorRoad => 5,
            Border => 4,
            Motorway | Trunk | Primary => 10,
            Rail => 12,
            RailMinor => 14,
            Secondary => 13,
            Tertiary | Unclassified | Residential | Raceway => 15,
            LivingStreet | Service | Footway | Unknown => 17,
        }
    }
}

/// Areal-feature classification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AreaKind {
    Water,
    Forest,
    Grass,
    Building,
    Land,
}

impl AreaKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => AreaKind::Water,
            1 => AreaKind::Forest,
            2 => AreaKind::Grass,
            3 => AreaKind::Building,
            4 => AreaKind::Land,
            _ => return None,
        })
    }

    /// Build threshold: coarsest zoom an area of this kind is materialized
    /// into (also subject to the per-tile sub-pixel size cull). Controls what the
    /// archive stores, not what the renderer shows — see `Self::display_min_zoom`
    pub fn min_zoom(self) -> u8 {
        match self {
            AreaKind::Land => 1,
            AreaKind::Water => 3,
            AreaKind::Forest => 3,
            AreaKind::Grass => 12,
            AreaKind::Building => 14,
        }
    }

    /// Display threshold: coarsest camera zoom at which an area of this kind
    /// is shown. Independent of `Self::min_zoom` tune freely.
    pub fn display_min_zoom(self) -> u8 {
        match self {
            AreaKind::Land => 1,
            AreaKind::Water => 3,
            AreaKind::Forest => 3,
            AreaKind::Grass => 12,
            AreaKind::Building => 16,
        }
    }
}

/// Grade-separation structure a linear feature sits on. Drives both draw order
/// (via the OSM `layer` tag) and special rendering (bridge casing, tunnel style).
/// `self as u8` is the wire ordinal; [`from_u8`](RoadStructure::from_u8) inverts.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoadStructure {
    #[default]
    None,
    Bridge,
    Tunnel,
}

impl RoadStructure {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => RoadStructure::None,
            1 => RoadStructure::Bridge,
            2 => RoadStructure::Tunnel,
            _ => return None,
        })
    }
}

/// State of a road endpoint. Extraction produces only `Connected` (a junction)
/// or `Disconnected` (a true dead-end); the tiler sets `Cut` where it clips a
/// road at a tile boundary, so the renderer continues the line into the
/// neighbouring tile instead of drawing an end cap.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeNode {
    Connected,
    Disconnected,
    Cut,
}

impl EdgeNode {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => EdgeNode::Connected,
            1 => EdgeNode::Disconnected,
            2 => EdgeNode::Cut,
            _ => return None,
        })
    }
}

/// Class of a text label. Place classes double as a priority ordering (lower
/// ordinal = more important). `LabelClass::min_zoom` gives the
/// coarsest zoom a label of the class appears at.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelClass {
    City,
    Town,
    Village,
    Suburb,
    Hamlet,
    Locality,
    Water,
    Park,
}

impl LabelClass {
    pub fn from_u8(v: u8) -> Option<Self> {
        use LabelClass::*;
        Some(match v {
            0 => City,
            1 => Town,
            2 => Village,
            3 => Suburb,
            4 => Hamlet,
            5 => Locality,
            6 => Water,
            7 => Park,
            _ => return None,
        })
    }

    /// Build threshold: coarsest zoom a label of this class is materialized
    /// into. See `Self::display_min_zoom` for the render-time
    /// threshold.
    pub fn min_zoom(self) -> u8 {
        use LabelClass::*;
        match self {
            City => 4,
            Town => 8,
            Village => 11,
            Suburb => 12,
            Hamlet => 12,
            Locality => 13,
            Water => 8,
            Park => 10,
        }
    }

    /// Display threshold: coarsest camera zoom at which a label of this class
    /// is shown. Independent of [`min_zoom`](Self::min_zoom); tune freely.
    pub fn display_min_zoom(self) -> u8 {
        use LabelClass::*;
        match self {
            City => 4,
            Town => 8,
            Village => 11,
            Suburb => 12,
            Hamlet => 12,
            Locality => 13,
            Water => 8,
            Park => 10,
        }
    }

    /// Renderer collision priority (lower = drawn first / wins collisions).
    pub fn rank(self) -> u8 {
        self as u8
    }
}

/// Point-of-interest classification — symbol features drawn at a single anchor
/// (no text). `self as u8` is the wire ordinal; [`from_u8`](PoiKind::from_u8) is
/// its inverse.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoiKind {
    TrafficSignal,
}

impl PoiKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => PoiKind::TrafficSignal,
            _ => return None,
        })
    }

    /// Build threshold: coarsest zoom this POI is materialized into. Point POIs
    /// like signals are dense, so only the base grid zoom.
    pub fn min_zoom(self) -> u8 {
        match self {
            PoiKind::TrafficSignal => 14,
        }
    }

    /// Display threshold: coarsest camera zoom at which this POI is shown.
    pub fn display_min_zoom(self) -> u8 {
        match self {
            PoiKind::TrafficSignal => 15,
        }
    }
}

/// Lane counts on each side of a road's centerline, relative to the geometry's
/// direction (node order). A one-way road has `backward == 0` (or `forward == 0`
/// for a reversed one-way).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lanes {
    pub forward: u8,
    pub backward: u8,
}

impl Lanes {
    pub fn new(forward: u8, backward: u8) -> Self {
        Self { forward, backward }
    }

    /// The split as seen from the opposite travel direction: forward and backward
    /// exchanged. Reversing a road's geometry must apply this so that "backward"
    /// keeps naming the same physical side (and the center line stays put).
    pub fn swapped(self) -> Self {
        Self {
            forward: self.backward,
            backward: self.forward,
        }
    }
}
