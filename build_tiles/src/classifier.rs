//! Classification of geometries as map elements

use crate::shapes::{AreaKind, LabelClass, Lanes, PoiKind, RoadKind, RoadStructure};
use hashbrown::HashMap;
use osm_pbf::tags::Tag;

pub enum ShapeClassification {
    Road(RoadKind),
    Area(AreaKind),
    Unknown,
}

macro_rules! sid_constructor {
    ($struct_name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        pub struct $struct_name {
            $(
                $field: $ty,
            )*
        }

        impl $struct_name {
            pub fn new(string_map: &HashMap<&[u8], u32>) -> Self {
                Self {$(
                    $field: string_map.get(stringify!($field).as_bytes().as_ref()).copied(),
                )*}
            }
        }
    };
}

sid_constructor!(
    BlockShapeClassifier {
        // string ids of iterest for the given block
        highway: Option<u32>,
        motorway: Option<u32>,
        motorway_link: Option<u32>,
        trunk: Option<u32>,
        trunk_link: Option<u32>,
        primary: Option<u32>,
        primary_link: Option<u32>,
        service: Option<u32>,
        secondary: Option<u32>,
        secondary_link: Option<u32>,
        tertiary: Option<u32>,
        tertiary_link: Option<u32>,
        unclassified: Option<u32>,
        residential: Option<u32>,
        living_street: Option<u32>,
        road: Option<u32>,
        raceway: Option<u32>,
        railway: Option<u32>,
        rail: Option<u32>,
        light_rail: Option<u32>,
        narrow_gauge: Option<u32>,
        traffic_signals: Option<u32>,
        bridge: Option<u32>,
        tunnel: Option<u32>,
        no: Option<u32>,
        natural: Option<u32>,
        water: Option<u32>,
        building: Option<u32>,
        landuse: Option<u32>,
        leisure: Option<u32>,
        wood: Option<u32>,
        forest: Option<u32>,
        grass: Option<u32>,
        meadow: Option<u32>,
        park: Option<u32>,
        layer: Option<u32>,
        name: Option<u32>,
        place: Option<u32>,
        city: Option<u32>,
        town: Option<u32>,
        village: Option<u32>,
        suburb: Option<u32>,
        hamlet: Option<u32>,
        locality: Option<u32>,
    }
);

impl BlockShapeClassifier {
    /// Grade-separation structure from `bridge=*` / `tunnel=*` (any value except
    /// `no`). Bridge wins if both are set (rare/degenerate).
    pub fn structure_of(&self, tags: &[Tag]) -> RoadStructure {
        let mut result = RoadStructure::None;
        for tag in tags {
            let not_no = self.no != Some(tag.value);
            if let Some(bridge) = &self.bridge
                && &tag.key == bridge
                && not_no
            {
                return RoadStructure::Bridge;
            }
            if let Some(tunnel) = &self.tunnel
                && &tag.key == tunnel
                && not_no
            {
                result = RoadStructure::Tunnel;
            }
        }
        result
    }

    /// The OSM `layer` tag as a signed stacking level (bridges/tunnels/overpasses),
    /// `0` if absent or unparseable. `strings` is the block string table
    /// (`id -> bytes`), needed to read the tag's numeric value.
    pub fn layer_of(&self, tags: &[Tag], strings: &[Vec<u8>]) -> i8 {
        let Some(layer_key) = self.layer else {
            return 0;
        };
        for tag in tags {
            if tag.key == layer_key {
                return strings
                    .get(tag.value as usize)
                    .and_then(|bytes| std::str::from_utf8(bytes).ok())
                    .and_then(|text| text.trim().parse::<i8>().ok())
                    .unwrap_or(0);
            }
        }
        0
    }

    /// The OSM `name` tag value as an owned string, or `None` if absent/empty.
    /// `strings` is the block string table (`id -> bytes`).
    pub fn name_of(&self, tags: &[Tag], strings: &[Vec<u8>]) -> Option<String> {
        let name_key = self.name?;
        for tag in tags {
            if tag.key == name_key {
                let bytes = strings.get(tag.value as usize)?;
                let text = std::str::from_utf8(bytes).ok()?;
                if !text.is_empty() {
                    return Some(text.to_owned());
                }
            }
        }
        None
    }

    /// Classify a node's `place=*` tag into a `LabelClass`, or `None` if it is
    /// not a recognised populated place.
    /// Classify a node's tags into a point-of-interest kind, or `None`.
    /// Currently: `highway=traffic_signals` → [`PoiKind::TrafficSignal`].
    pub fn classify_poi(&self, tags: &[Tag]) -> Option<PoiKind> {
        for tag in tags {
            if let Some(highway) = &self.highway
                && &tag.key == highway
                && let Some(ts) = &self.traffic_signals
                && &tag.value == ts
            {
                return Some(PoiKind::TrafficSignal);
            }
        }
        None
    }

    pub fn classify_place(&self, tags: &[Tag]) -> Option<LabelClass> {
        let place_key = self.place?;
        for tag in tags {
            if tag.key != place_key {
                continue;
            }
            let v = Some(tag.value);
            return Some(if v == self.city {
                LabelClass::City
            } else if v == self.town {
                LabelClass::Town
            } else if v == self.village {
                LabelClass::Village
            } else if v == self.suburb {
                LabelClass::Suburb
            } else if v == self.hamlet {
                LabelClass::Hamlet
            } else if v == self.locality {
                LabelClass::Locality
            } else {
                return None;
            });
        }
        None
    }

    /// Lanes on each side of a way's centerline from its `lanes`, `oneway`, and
    /// `lanes:forward`/`lanes:backward` tags. Matches keys by their string bytes
    /// (so colon keys work), so it needs no precomputed sids. Every road gets a
    /// value: unknown lane counts fall back to 1 per active direction.
    pub fn lanes_of(&self, tags: &[Tag], strings: &[Vec<u8>]) -> Lanes {
        let value = |sid: u32| -> Option<&str> {
            strings
                .get(sid as usize)
                .and_then(|b| std::str::from_utf8(b).ok())
        };
        let num = |s: &str| s.trim().parse::<u8>().ok();

        let (mut total, mut fwd, mut bwd) = (None, None, None);
        let mut oneway: i8 = 0; // 0 two-way, 1 forward one-way, -1 reversed
        for tag in tags {
            match strings.get(tag.key as usize).map(Vec::as_slice) {
                Some(b"lanes") => total = value(tag.value).and_then(num),
                Some(b"lanes:forward") => fwd = value(tag.value).and_then(num),
                Some(b"lanes:backward") => bwd = value(tag.value).and_then(num),
                Some(b"oneway") => {
                    oneway = match value(tag.value).map(str::trim) {
                        Some("yes" | "true" | "1") => 1,
                        Some("-1" | "reverse") => -1,
                        _ => 0,
                    };
                }
                _ => {}
            }
        }

        match oneway {
            1 => Lanes::new(fwd.or(total).unwrap_or(1).max(1), 0),
            -1 => Lanes::new(0, bwd.or(total).unwrap_or(1).max(1)),
            _ => {
                // Two-way: prefer explicit per-direction counts, else split the
                // total (odd extra to forward), else default to 1 + 1.
                let (f, b) = match (fwd, bwd, total) {
                    (Some(f), Some(b), _) => (f, b),
                    (Some(f), None, Some(t)) => (f, t.saturating_sub(f)),
                    (None, Some(b), Some(t)) => (t.saturating_sub(b), b),
                    (Some(f), None, None) => (f, f),
                    (None, Some(b), None) => (b, b),
                    (None, None, Some(t)) => (t.div_ceil(2), t / 2),
                    (None, None, None) => (1, 1),
                };
                Lanes::new(f.max(1), b.max(1))
            }
        }
    }

    /// Number of above-ground floors from `building:levels`, defaulting to 1 if
    /// absent or unparseable. Fractional values (e.g. "2.5") round to nearest;
    /// clamped to `[1, 255]`. Matches the key by string bytes (colon key).
    pub fn floors_of(&self, tags: &[Tag], strings: &[Vec<u8>]) -> u8 {
        for tag in tags {
            if strings.get(tag.key as usize).map(Vec::as_slice) == Some(b"building:levels") {
                return strings
                    .get(tag.value as usize)
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .map(|n| n.round().clamp(1.0, 255.0) as u8)
                    .unwrap_or(1);
            }
        }
        1
    }

    pub fn classify_way(&self, tags: &[Tag]) -> ShapeClassification {
        for tag in tags {
            if let Some(highway) = &self.highway
                && &tag.key == highway
            {
                if let Some(motorway) = &self.motorway
                    && &tag.value == motorway
                {
                    return ShapeClassification::Road(RoadKind::Motorway);
                }
                if let Some(motorway_link) = &self.motorway_link
                    && &tag.value == motorway_link
                {
                    return ShapeClassification::Road(RoadKind::Motorway);
                }
                if let Some(trunk) = &self.trunk
                    && &tag.value == trunk
                {
                    return ShapeClassification::Road(RoadKind::Trunk);
                }
                if let Some(trunk_link) = &self.trunk_link
                    && &tag.value == trunk_link
                {
                    return ShapeClassification::Road(RoadKind::Trunk);
                }
                if let Some(primary) = &self.primary
                    && &tag.value == primary
                {
                    return ShapeClassification::Road(RoadKind::Primary);
                }
                if let Some(primary_link) = &self.primary_link
                    && &tag.value == primary_link
                {
                    return ShapeClassification::Road(RoadKind::Primary);
                }
                if let Some(service) = &self.service
                    && &tag.value == service
                {
                    return ShapeClassification::Road(RoadKind::Service);
                }
                if let Some(secondary) = &self.secondary
                    && &tag.value == secondary
                {
                    return ShapeClassification::Road(RoadKind::Secondary);
                }
                if let Some(secondary_link) = &self.secondary_link
                    && &tag.value == secondary_link
                {
                    return ShapeClassification::Road(RoadKind::Secondary);
                }
                if let Some(tertiary) = &self.tertiary
                    && &tag.value == tertiary
                {
                    return ShapeClassification::Road(RoadKind::Tertiary);
                }
                if let Some(tertiary_link) = &self.tertiary_link
                    && &tag.value == tertiary_link
                {
                    return ShapeClassification::Road(RoadKind::Tertiary);
                }
                if let Some(unclassified) = &self.unclassified
                    && &tag.value == unclassified
                {
                    return ShapeClassification::Road(RoadKind::Unclassified);
                }
                if let Some(residential) = &self.residential
                    && &tag.value == residential
                {
                    return ShapeClassification::Road(RoadKind::Residential);
                }
                if let Some(living_street) = &self.living_street
                    && &tag.value == living_street
                {
                    return ShapeClassification::Road(RoadKind::LivingStreet);
                }
                if let Some(road) = &self.road
                    && &tag.value == road
                {
                    return ShapeClassification::Road(RoadKind::Unknown);
                }
                if let Some(raceway) = &self.raceway
                    && &tag.value == raceway
                {
                    return ShapeClassification::Road(RoadKind::Raceway);
                }
            }

            // Railways: heavy `rail`, plus `light_rail` / `narrow_gauge`.
            if let Some(railway) = &self.railway
                && &tag.key == railway
            {
                if let Some(rail) = &self.rail
                    && &tag.value == rail
                {
                    return ShapeClassification::Road(RoadKind::Rail);
                }
                if let Some(light_rail) = &self.light_rail
                    && &tag.value == light_rail
                {
                    return ShapeClassification::Road(RoadKind::RailMinor);
                }
                if let Some(narrow_gauge) = &self.narrow_gauge
                    && &tag.value == narrow_gauge
                {
                    return ShapeClassification::Road(RoadKind::RailMinor);
                }
            }

            if let Some(kind) = self.area_of_tag(tag) {
                return ShapeClassification::Area(kind);
            }
        }

        ShapeClassification::Unknown
    }

    /// Area kind named by a single tag, if any: `natural=water` → Water;
    /// `building=*` → Building; `natural=wood` / `landuse=forest` → Forest;
    /// `landuse=grass|meadow` / `leisure=park` → Grass.
    fn area_of_tag(&self, tag: &Tag) -> Option<AreaKind> {
        let kv = |key: &Option<u32>, val: &Option<u32>| matches!((key, val), (Some(k), Some(v)) if &tag.key == k && &tag.value == v);
        if kv(&self.natural, &self.water) {
            return Some(AreaKind::Water);
        }
        if matches!(&self.building, Some(b) if &tag.key == b) {
            return Some(AreaKind::Building);
        }
        if kv(&self.natural, &self.wood) || kv(&self.landuse, &self.forest) {
            return Some(AreaKind::Forest);
        }
        if kv(&self.landuse, &self.grass)
            || kv(&self.landuse, &self.meadow)
            || kv(&self.leisure, &self.park)
        {
            return Some(AreaKind::Grass);
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self::new(&HashMap::new())
    }

    pub fn classify_relation(&self, tags: &[Tag]) -> ShapeClassification {
        for tag in tags {
            if let Some(kind) = self.area_of_tag(tag) {
                return ShapeClassification::Area(kind);
            }
        }

        ShapeClassification::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `(strings, tags)` pair from `(key, value)` string pairs.
    fn tagset(pairs: &[(&str, &str)]) -> (Vec<Vec<u8>>, Vec<Tag>) {
        let mut strings: Vec<Vec<u8>> = Vec::new();
        let mut intern = |s: &str| -> u32 {
            let i = strings.len() as u32;
            strings.push(s.as_bytes().to_vec());
            i
        };
        let tags = pairs
            .iter()
            .map(|(k, v)| Tag::new((intern(k), intern(v))))
            .collect();
        (strings, tags)
    }

    fn lanes(pairs: &[(&str, &str)]) -> Lanes {
        let (strings, tags) = tagset(pairs);
        BlockShapeClassifier::for_test().lanes_of(&tags, &strings)
    }

    #[test]
    fn lanes_oneway_puts_all_forward() {
        assert_eq!(
            lanes(&[("lanes", "3"), ("oneway", "yes")]),
            Lanes::new(3, 0)
        );
        // oneway with no lanes tag → 1 forward.
        assert_eq!(lanes(&[("oneway", "yes")]), Lanes::new(1, 0));
        // reversed one-way → all backward.
        assert_eq!(lanes(&[("lanes", "2"), ("oneway", "-1")]), Lanes::new(0, 2));
    }

    #[test]
    fn lanes_two_way_splits_total() {
        assert_eq!(lanes(&[("lanes", "4")]), Lanes::new(2, 2));
        // odd total: extra lane goes forward (likely a centre turn lane).
        assert_eq!(lanes(&[("lanes", "3")]), Lanes::new(2, 1));
        // no tags at all → a lane each way.
        assert_eq!(lanes(&[("highway", "residential")]), Lanes::new(1, 1));
    }

    fn floors(pairs: &[(&str, &str)]) -> u8 {
        let (strings, tags) = tagset(pairs);
        BlockShapeClassifier::for_test().floors_of(&tags, &strings)
    }

    #[test]
    fn floors_default_and_parse() {
        assert_eq!(floors(&[("building", "yes")]), 1); // no levels tag → 1
        assert_eq!(floors(&[("building:levels", "5")]), 5);
        assert_eq!(floors(&[("building:levels", "2.5")]), 3); // rounds
        assert_eq!(floors(&[("building:levels", "0")]), 1); // clamped to ≥1
        assert_eq!(floors(&[("building:levels", "garbage")]), 1);
    }

    #[test]
    fn lanes_explicit_directions_win() {
        assert_eq!(
            lanes(&[
                ("lanes", "5"),
                ("lanes:forward", "3"),
                ("lanes:backward", "2")
            ]),
            Lanes::new(3, 2)
        );
        // one side explicit + total → derive the other.
        assert_eq!(
            lanes(&[("lanes", "5"), ("lanes:forward", "3")]),
            Lanes::new(3, 2)
        );
    }
}
