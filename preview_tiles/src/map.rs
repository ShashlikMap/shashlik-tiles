use crate::vertex::Vertex2dBuffer;
use lyon_tessellation::{
    FillOptions, FillRule, FillTessellator, StrokeOptions, StrokeTessellator, geom::Point,
};
use tiles::decode::{AreaKind, DecodedTile, RoadKind, Scene, SceneView};
use tiles::reader::{HttpRangeReader, PmTilesReader};
use tiles::view::TileCache;

// pub type World = SceneView<TileCache<PmTilesReader<FileRangeReader>, DecodedTile>>;
// pub async fn open_world(path: &str) -> World {
//     let reader = FileRangeReader::open(path).await.unwrap();
//     let source = PmTilesReader::open(reader).await.unwrap();
//     let cache = TileCache::new(source, vec![4, 6, 8, 10, 12, 14], 256 << 20, 1);
//     SceneView::new(cache)
// }

pub type World = SceneView<TileCache<PmTilesReader<HttpRangeReader>, DecodedTile>>;

/// Open a PMTiles archive at `url` over HTTP(S) and wrap it in a decoding,
/// camera-driven view.
pub async fn open_world(url: &str) -> World {
    let reader = HttpRangeReader::open(url).await.unwrap();
    let source = PmTilesReader::open(reader).await.unwrap();
    let cache = TileCache::new(source, vec![4, 6, 8, 10, 12, 14], 256 << 20, 1);
    SceneView::new(cache)
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

    for a in scene.areas() {
        // Stamp the palette index for this feature onto every vertex it produces.
        buffer.current_color = area_color(a.kind);
        let opts = FillOptions::default().with_fill_rule(FillRule::NonZero);
        let mut b = ftess.builder(&opts, &mut buffer);
        for ring in &a.rings {
            let mut pts = ring.iter().map(|&c| rel(c));
            if let Some(first) = pts.next() {
                b.begin(first);
                for p in pts {
                    b.line_to(p);
                }
                b.end(true);
            }
        }
        b.build().unwrap();
    }

    for r in scene.roads() {
        buffer.current_color = road_color(r.kind);
        let opts = StrokeOptions::default().with_line_width(line_width);
        let mut b = stess.builder(&opts, &mut buffer);
        let mut pts = r.coords.iter().map(|&c| rel(c));
        if let Some(first) = pts.next() {
            b.begin(first);
            for p in pts {
                b.line_to(p);
            }
            b.end(false);
            b.build().unwrap();
        }
    }

    // for a in scene.areas() {
    //     let opts = StrokeOptions::default().with_line_width(line_width);
    //     let mut b = stess.builder(&opts, &mut buffer);

    //     for ring in &a.rings {
    //         let mut pts = ring.iter().map(|&c| rel(c));
    //         if let Some(first) = pts.next() {
    //             b.begin(first);
    //             for p in pts {
    //                 b.line_to(p);
    //             }
    //             b.end(true);
    //         }
    //     }
    //     b.build().unwrap();
    // }
    buffer
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
        Motorway | Trunk | Primary => 5,
        Secondary | Tertiary => 6,
        _ => 7,
    }
}
