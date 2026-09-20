use crate::vertex::Vertex2dBuffer;
use lyon_tessellation::{
    FillOptions, FillRule, FillTessellator, StrokeOptions, StrokeTessellator, geom::Point,
};
use tiles::decode::{
    AreaKind, DecodedTile, PoiKind, RoadKind, Scene, SceneArea, SceneRoad, SceneView,
};
use tiles::reader::{FileRangeReader, HttpRangeReader, PmTilesReader, RangeReader};
use tiles::view::TileCache;

pub type World = SceneView<TileCache<PmTilesReader<Box<dyn RangeReader>>, DecodedTile>>;

/// Open a PMTiles archive from a local path or an `http(s)://` URL (auto-detected)
/// and wrap it in a decoding, camera-driven view.
pub async fn open_world(source: &str) -> World {
    let reader: Box<dyn RangeReader> =
        if source.starts_with("http://") || source.starts_with("https://") {
            Box::new(HttpRangeReader::open(source).await.unwrap())
        } else {
            Box::new(FileRangeReader::open(source).await.unwrap())
        };
    let source = PmTilesReader::open(reader).await.unwrap();
    let cache = TileCache::from_source(source, 256 << 20, 0);
    SceneView::new(cache)
}

/// Base geometry (areas + roads) referenced for painter-order sorting.
enum Base<'a> {
    Area(&'a SceneArea),
    Road(&'a SceneRoad),
}

impl Base<'_> {
    /// Painter's-order key (drawn ascending = bottom first):
    /// 1. `layer` — OSM stacking level; bridges (+1) / tunnels (−1) already carry
    ///    a synthesized layer, so this alone stacks them and pushes tunnels under
    ///    surrounding surface geometry.
    /// 2. style rank — areas (0) under roads (1) within a layer.
    /// 3. kind rank — ordering among same-style features (e.g. minor roads under
    ///    major, water under vegetation).
    fn order_key(&self) -> (i32, u8, u8) {
        match self {
            Base::Area(a) => (a.layer as i32, 0, area_rank(a.kind)),
            Base::Road(r) => (r.layer as i32, 1, road_rank(r.kind)),
        }
    }
}

pub fn tessellate(scene: &Scene, origin: [f64; 2], line_width: f32) -> Vertex2dBuffer {
    let mut buffer = Vertex2dBuffer::default();
    let mut stess = StrokeTessellator::new();
    let mut ftess = FillTessellator::new();
    let rel = |c: [i32; 2]| {
        Point::new(
            (c[0] as f64 - origin[0]) as f32,
            (c[1] as f64 - origin[1]) as f32,
        )
    };

    // Draw order is index order (painter's algorithm, no depth test). Collect base
    // geometry and sort so lower layers / areas render before higher layers /
    // roads. POIs are drawn last, always on top.
    let mut base: Vec<Base> = Vec::with_capacity(scene.areas().len() + scene.roads().len());
    base.extend(scene.areas().iter().map(Base::Area));
    base.extend(scene.roads().iter().map(Base::Road));
    base.sort_by_key(Base::order_key);

    for b in &base {
        match b {
            Base::Area(a) => {
                buffer.current_color = area_color(a.kind);
                let opts = FillOptions::default().with_fill_rule(FillRule::NonZero);
                let mut fb = ftess.builder(&opts, &mut buffer);
                for ring in &a.rings {
                    let mut pts = ring.iter().map(|&c| rel(c));
                    if let Some(first) = pts.next() {
                        fb.begin(first);
                        for p in pts {
                            fb.line_to(p);
                        }
                        fb.end(true);
                    }
                }
                fb.build().unwrap();
            }
            Base::Road(r) => {
                buffer.current_color = road_color(r.kind);
                let opts = StrokeOptions::default().with_line_width(line_width);
                let mut sb = stess.builder(&opts, &mut buffer);
                let mut pts = r.coords.iter().map(|&c| rel(c));
                if let Some(first) = pts.next() {
                    sb.begin(first);
                    for p in pts {
                        sb.line_to(p);
                    }
                    sb.end(false);
                    sb.build().unwrap();
                }
            }
        }
    }

    // POI markers — always on top.
    let half = (line_width * 2.0).max(1.0);
    for p in scene.pois() {
        buffer.current_color = poi_color(p.kind);
        let c = rel(p.anchor);
        let opts = FillOptions::default();
        let mut b = ftess.builder(&opts, &mut buffer);
        b.begin(Point::new(c.x - half, c.y - half));
        b.line_to(Point::new(c.x + half, c.y - half));
        b.line_to(Point::new(c.x + half, c.y + half));
        b.line_to(Point::new(c.x - half, c.y + half));
        b.end(true);
        b.build().unwrap();
    }

    buffer
}

/// Draw rank among same-layer areas (drawn ascending): background land first,
/// then water, vegetation, buildings on top.
fn area_rank(kind: AreaKind) -> u8 {
    match kind {
        AreaKind::Land => 0,
        AreaKind::Grass => 1,
        AreaKind::Water => 2,
        AreaKind::Forest => 3,
        AreaKind::Building => 4,
    }
}

/// Draw rank among same-layer roads (drawn ascending): minor first, major on top
/// so junctions read correctly.
fn road_rank(kind: RoadKind) -> u8 {
    use RoadKind::*;
    match kind {
        Footway | Service | LivingStreet | Raceway | Unknown => 0,
        Residential | Unclassified => 1,
        RailMinor | Rail => 2,
        Tertiary | Secondary => 3,
        Primary | Trunk => 4,
        Motorway | MajorRoad => 5,
        Border => 6,
    }
}

/// Palette index for a POI kind. Must match the palette in `shader.wgsl`.
fn poi_color(kind: PoiKind) -> u32 {
    match kind {
        PoiKind::TrafficSignal => 11,
    }
}

/// Palette index for an area kind. Must match the `PALETTE` table in `shader.wgsl`.
fn area_color(kind: AreaKind) -> u32 {
    match kind {
        AreaKind::Water => 0,
        AreaKind::Forest => 1,
        AreaKind::Grass => 2,
        AreaKind::Building => 3,
        AreaKind::Land => 4,
    }
}

/// Palette index for a road, bucketed by class tier. Must match `shader.wgsl`.
fn road_color(kind: RoadKind) -> u32 {
    use RoadKind::*;
    match kind {
        MajorRoad => 8,
        Rail => 9,
        RailMinor => 10,
        Border => 12,
        Motorway | Trunk | Primary => 5,
        Secondary | Tertiary => 6,
        _ => 7,
    }
}
