pub mod decode;
pub mod reader;
pub mod view;

use geo::{Coord, Rect};
use std::f64::consts::PI;

pub const EARTH_RADIUS: f64 = 6_378_137.0;
pub const MERCATOR_MAX: f64 = PI * EARTH_RADIUS; // 20_037_508.342_789_244
pub const MERCATOR_EXTENT: f64 = 2.0 * MERCATOR_MAX;
pub const MAX_LATITUDE: f64 = 85.051_128_779_806_59;

/// A geographic coordinate (WGS84 degrees): `.0.x` = lon, `.0.y` = lat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatLon(pub Coord<f64>);

/// A projected coordinate in Web Mercator meters (EPSG:3857).
/// x eastward, y NORTHWARD (standard projected convention).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mercator(pub Coord<f64>);

impl LatLon {
    pub fn new(lat: f64, lon: f64) -> Self {
        LatLon(Coord { x: lon, y: lat })
    }
    pub fn lat(&self) -> f64 {
        self.0.y
    }
    pub fn lon(&self) -> f64 {
        self.0.x
    }

    pub fn clamp_lat(lat: f64) -> f64 {
        lat.clamp(-MAX_LATITUDE, MAX_LATITUDE)
    }

    /// Project to Web Mercator meters.
    pub fn to_mercator(&self) -> Mercator {
        let lat_rad = Self::clamp_lat(self.lat()).to_radians();
        let x = self.lon().to_radians() * EARTH_RADIUS;
        let y = EARTH_RADIUS * (PI / 4.0 + lat_rad / 2.0).tan().ln();
        Mercator(Coord { x, y })
    }
}

impl Mercator {
    pub fn new(x: f64, y: f64) -> Self {
        Mercator(Coord { x, y })
    }
    pub fn x(&self) -> f64 {
        self.0.x
    }
    pub fn y(&self) -> f64 {
        self.0.y
    }

    /// Unproject back to geographic degrees.
    pub fn to_latlon(&self) -> LatLon {
        let lon = (self.x() / EARTH_RADIUS).to_degrees();
        let lat = (self.y() / EARTH_RADIUS).sinh().atan().to_degrees();
        LatLon(Coord { x: lon, y: lat })
    }

    fn axis_tiles(z: u8) -> f64 {
        (1u64 << z) as f64
    }

    /// Tile edge length in Mercator meters at zoom `z` (constant per zoom).
    pub fn tile_size(z: u8) -> f64 {
        MERCATOR_EXTENT / Self::axis_tiles(z)
    }

    /// Fractional tile coordinates; integer part = index, fraction = in-tile
    /// position. Handles the Mercator(north-up) -> tile(south-down) y flip.
    pub fn to_fractional_tile(&self, z: u8) -> (f64, f64) {
        let n = Self::axis_tiles(z);
        let fx = (self.x() + MERCATOR_MAX) / MERCATOR_EXTENT;
        let fy = (MERCATOR_MAX - self.y()) / MERCATOR_EXTENT;
        (fx * n, fy * n)
    }

    pub fn to_tile(&self, z: u8) -> Tile {
        let (fx, fy) = self.to_fractional_tile(z);
        let max = Self::axis_tiles(z) - 1.0;
        Tile {
            x: fx.floor().clamp(0.0, max) as u32,
            y: fy.floor().clamp(0.0, max) as u32,
            z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub z: u8,
}

impl Tile {
    pub fn new(x: u32, y: u32, z: u8) -> Self {
        Tile { x, y, z }
    }

    /// Tile bounds in Mercator meters as a `geo::Rect` (min = SW, max = NE),
    /// ready for `geo` algorithms: `.contains`, `.to_polygon()` (CCW), etc.
    pub fn bounds(&self) -> Rect<f64> {
        let size = Mercator::tile_size(self.z);
        let min_x = -MERCATOR_MAX + self.x as f64 * size;
        let max_y = MERCATOR_MAX - self.y as f64 * size; // tile y=0 is north
        // Rect::new normalizes corners; pass SW and NE explicitly.
        Rect::new(
            Coord {
                x: min_x,
                y: max_y - size,
            }, // SW (min)
            Coord {
                x: min_x + size,
                y: max_y,
            }, // NE (max)
        )
    }

    /// Northwest corner in Mercator meters — the local origin to subtract
    /// before casting tile geometry to f32 for Lyon/GPU.
    pub fn nw_corner(&self) -> Mercator {
        let b = self.bounds();
        Mercator(Coord {
            x: b.min().x,
            y: b.max().y,
        })
    }

    pub fn parent(&self) -> Option<Tile> {
        (self.z > 0).then(|| Tile {
            x: self.x / 2,
            y: self.y / 2,
            z: self.z - 1,
        })
    }

    pub fn children(&self) -> [Tile; 4] {
        let (x, y, z) = (self.x * 2, self.y * 2, self.z + 1);
        [
            Tile { x, y, z },
            Tile { x: x + 1, y, z },
            Tile { x, y: y + 1, z },
            Tile {
                x: x + 1,
                y: y + 1,
                z,
            },
        ]
    }
}
