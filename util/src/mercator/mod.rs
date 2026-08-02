//! simple Mercator coordinate translation implementation

use core::f64::consts::PI;

/// Max latitude value
const LAT_MAX: f64 = 85.05112878;

/// Tile size in pixels
const TILE_SIZE: f64 = 512.0;

pub fn to_mercator(lon: f64, lat: f64, zoom: u8) -> (f64, f64) {
    let map_size = TILE_SIZE * 2.0f64.powi(zoom as i32);
    let x = (lon + 180.0) / 360.0;
    let sin_lat_rad = lat.clamp(-LAT_MAX, LAT_MAX).to_radians().sin();
    let y = 0.5 - ((1.0 + sin_lat_rad) / (1.0 - sin_lat_rad)).ln() / (4.0 * PI);

    (x * map_size, y * map_size)
}

pub fn from_mercator(x: f64, y: f64, zoom: u8) -> (f64, f64) {
    let map_size = TILE_SIZE * 2.0f64.powi(zoom as i32);

    let x_norm = x / map_size;
    let y_norm = y / map_size;

    let lon = x_norm * 360.0 - 180.0;
    let n = PI - 2.0 * PI * y_norm;
    let lat = (n.exp().atan() * 2.0 - PI / 2.0).to_degrees();

    (lon, lat)
}

pub fn tile_index(x: f64, y: f64) -> (u32, u32) {
    let tile_x = (x / TILE_SIZE).floor() as u32;
    let tile_y = (y / TILE_SIZE).floor() as u32;

    (tile_x, tile_y)
}

pub fn scale(x: f64, y: f64, from_zoom: f64, to_zoom: f64) -> (f64, f64) {
    let delta = to_zoom - from_zoom;
    let factor = 2.0f64.powf(delta);

    (x * factor, y * factor)
}
