pub mod classifier;
pub mod extractor;
pub mod major_roads;
pub mod mask;
pub mod node_cache;
pub mod route_roads;
pub mod shapes;
pub mod sink;
pub mod tiler;

use crate::shapes::{Area, AreaKind, Shape};
use crate::sink::ShapeSink;
use crate::tiler::writer::TileWriter;
use crate::tiler::{format, pmtiles::PmTilesWriter, sink::TileSink, spill};
use clap::Parser;
use extractor::FilteredRelations;
use hashbrown::HashMap;
use indicatif::{ProgressBar, ProgressStyle};
use osm_pbf::{reader::OsmReader, tags::TagFilter};
use rayon::prelude::*;
use shapefile::{self, PolygonRing, Reader};
use std::sync::{Arc, Mutex};
use tiles::Tile;

/// zstd level for tile payloads (the reader zstd-decompresses).
const ZSTD_LEVEL: i32 = 9;
/// Materialized zooms (client overzooms the gaps). Built one at a time and
/// flushed, so peak memory is a single zoom's tile set — trim the finest zooms
/// here if the whole planet at z12/z14 doesn't fit.
const ZOOMS: [u8; 6] = [14, 12, 10, 8, 6, 4];

#[derive(Parser)]
struct Args {
    /// Path to OSM PBF file
    osm_file: String,

    /// Clip and spill geometry into tile buckets under this directory.
    #[arg(long)]
    spill: String,

    /// Build tiles into a PMTiles archive here
    #[arg(long)]
    pmtiles: String,

    /// Path to ocean water polygon shapefile, shorelines will be skipped if not provided
    #[arg(long)]
    water_shapefile: Option<String>,
}

/// Even-odd ray cast: is `(px, py)` inside the ring?
fn point_in_ring(ring: &geo::LineString, px: f64, py: f64) -> bool {
    let pts = &ring.0;
    let n = pts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (pts[i].x, pts[i].y);
        let (xj, yj) = (pts[j].x, pts[j].y);
        if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Read water polygons from a coastline shapefile.
pub fn read_water_polygons(shape_file_path: &str) -> Vec<geo::Polygon> {
    let mut shapes = Reader::from_path(shape_file_path).unwrap();
    let len = shapes.shape_count().unwrap();
    let pb = ProgressBar::new(len as u64);

    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let water = shapes
        .iter_shapes_and_records()
        .par_bridge()
        .flat_map_iter(|result| -> Vec<geo::Polygon> {
            let (shape, _) = result.unwrap();
            pb.inc(1);

            let shapefile::Shape::Polygon(polygon) = shape else {
                return Vec::new();
            };

            // A shapefile polygon can hold several outer rings (a multipolygon),
            // each with its own holes. Keep them as separate polygons — merging
            // every outer into one ring produces a self-intersecting polygon.
            let mut outers: Vec<geo::LineString> = Vec::new();
            let mut inners: Vec<geo::LineString> = Vec::new();
            for r in polygon.rings() {
                match r {
                    PolygonRing::Outer(pts) => {
                        outers.push(pts.iter().map(|p| geo::Coord { x: p.x, y: p.y }).collect())
                    }
                    PolygonRing::Inner(pts) => {
                        inners.push(pts.iter().map(|p| geo::Coord { x: p.x, y: p.y }).collect())
                    }
                }
            }

            // Assign each hole to the outer ring that contains it (by order in the
            // common single-outer case; by point-in-ring otherwise).
            let mut holes: Vec<Vec<geo::LineString>> = vec![Vec::new(); outers.len()];
            for inner in inners {
                let owner = if outers.len() == 1 {
                    Some(0)
                } else {
                    inner
                        .0
                        .first()
                        .and_then(|c| outers.iter().position(|o| point_in_ring(o, c.x, c.y)))
                };
                if let Some(i) = owner {
                    holes[i].push(inner);
                }
            }

            outers
                .into_iter()
                .zip(holes)
                .map(|(o, h)| geo::Polygon::new(o, h))
                .collect()
        })
        .collect();

    pb.finish();
    pb.unset_length();

    water
}

fn indicator_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes}",
    )
    .unwrap()
    .progress_chars("#>-")
}

/// Progress callback shared across passes. Adopts the reported length on every
/// call (a pass's length is constant, so this is idempotent within a pass) and
/// tracks it when it changes — so a step that runs several internal passes over
/// differently-sized sections (e.g. `route_roads::build`) shows the right total
/// for each, without the caller resetting between them.
fn progress_bar() -> (ProgressBar, Arc<impl Fn(u64, u64) + Send + Sync>) {
    let indicator = ProgressBar::new(0).with_style(indicator_style());
    let inner = indicator.clone();
    let progress = Arc::new(move |pos, len| {
        if inner.length() != Some(len) {
            inner.set_length(len);
        }
        inner.set_position(pos);
    });

    (indicator, progress)
}

fn main() {
    let args = Args::parse();
    let params = tiler::TileParams::new(14, 4, 6, 8192, 0, 4.0);

    let mut osm_reader = OsmReader::from_file(&args.osm_file);

    let way_tag_filter = TagFilter(vec![
        vec![("highway".into(), vec![])],
        vec![(
            "railway".into(),
            vec!["rail".into(), "light_rail".into(), "narrow_gauge".into()],
        )],
        vec![("natural".into(), vec!["water".into(), "wood".into()])],
        vec![(
            "landuse".into(),
            vec!["forest".into(), "grass".into(), "meadow".into()],
        )],
        vec![("leisure".into(), vec!["park".into()])],
        vec![("building".into(), vec![])],
    ]);
    let relation_tag_filter = TagFilter(vec![
        vec![
            ("natural".into(), vec!["water".into(), "wood".into()]),
            ("type".into(), vec!["multipolygon".into()]),
        ],
        vec![
            (
                "landuse".into(),
                vec!["forest".into(), "grass".into(), "meadow".into()],
            ),
            ("type".into(), vec!["multipolygon".into()]),
        ],
        vec![
            ("leisure".into(), vec!["park".into()]),
            ("type".into(), vec!["multipolygon".into()]),
        ],
        vec![
            ("building".into(), vec![]),
            ("type".into(), vec!["multipolygon".into()]),
        ],
    ]);

    let (indicator, progress) = progress_bar();

    println!("Indexing OSM file...");
    let index = cache::index::OsmPbfIndex::open_or_create(
        &args.osm_file,
        &mut osm_reader,
        progress.clone(),
    )
    .unwrap();
    indicator.finish();
    indicator.unset_length();
    println!("{index:?}");

    println!("Scanning relations...");
    let FilteredRelations { relation_ways, .. } = FilteredRelations::extract(
        &mut osm_reader,
        &index,
        &relation_tag_filter,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();
    println!("Relation-referenced ways: {}", relation_ways.len());

    let node_cache_path = node_cache::node_cache_path(&args.osm_file);

    println!("Indexing referenced nodes...");
    let node_bitset = node_cache::build_node_bitset(
        &mut osm_reader,
        &index,
        &way_tag_filter,
        &relation_ways,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();

    let node_store = if node_cache::node_store_matches(&node_cache_path, node_bitset.len() as usize)
    {
        println!("Reusing node store at {}...", node_cache_path.display());
        node_cache::NodeStore::open(&node_cache_path, node_bitset).unwrap()
    } else {
        println!("Writing node store...");
        let store = node_cache::build_node_store(
            &mut osm_reader,
            &index,
            node_bitset,
            &node_cache_path,
            progress.clone(),
        )
        .unwrap();
        indicator.finish();
        indicator.unset_length();
        store
    };

    println!("Cached {} node locations", node_store.len());

    println!("Detecting road connectivity...");
    let connectivity = extractor::detect_road_connectivity(
        &mut osm_reader,
        &index,
        &way_tag_filter,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();
    println!("Road junctions: {}", connectivity.junction_count());

    let sink = TileSink::new(&args.spill, params, ZOOMS.to_vec());
    // Forests are streamed into a global raster mask for coarse-zoom aggregation;
    // the tee forwards every shape to the sink (which skips raw forests at coarse
    // zooms) while burning forests into the mask.
    let forest_mask = mask::PolygonMask::new();
    let feed = mask::PolygonTee::new(&sink, &forest_mask);

    println!("Extracting way geometry...");
    extractor::extract_ways(
        &mut osm_reader,
        &index,
        &way_tag_filter,
        &node_store,
        &connectivity,
        &feed,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();

    println!("Collecting relation member ways...");
    let member_ways = extractor::collect_relation_member_ways(
        &mut osm_reader,
        &index,
        &relation_ways,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();

    println!("Extracting relation geometry...");
    extractor::extract_relations(
        &mut osm_reader,
        &index,
        &relation_tag_filter,
        &node_store,
        &member_ways,
        &feed,
        progress.clone(),
    );
    indicator.finish();
    indicator.unset_length();

    println!("Extracting place labels...");
    extractor::extract_place_labels(&mut osm_reader, &index, &feed, progress.clone());
    indicator.finish();
    indicator.unset_length();

    // ─────────────────────────── WATER coastline polygons ───────────────────────────
    if let Some(water_shapefile) = &args.water_shapefile {
        let merged_water = {
            let water = read_water_polygons(water_shapefile);
            println!("water polygons : {}", water.len());
            geo::algorithm::bool_ops::unary_union(&water)
        };
        println!("Merged water polygons: {}", merged_water.0.len());

        merged_water.0.into_par_iter().for_each(|poly| {
            sink.push(Shape::Area(Area::new(AreaKind::Water, poly, -1, 0)));
        });
    }
    // ─────────────────────────── FOREST aggregation ──────────────────────────
    // The mask was filled during extraction; close + vectorize it into merged
    // forest blobs and clip those into the coarse (z <= AGG_MAX_ZOOM) tiles. Raw
    // forests already covered the finer zooms via the extraction stream.
    println!("Aggregating forests from mask...");
    let merged_forests = forest_mask.merged_polygons();
    println!("Merged forest polygons: {}", merged_forests.len());
    merged_forests.into_par_iter().for_each(|poly| {
        sink.push_aggregated(&Area::new(AreaKind::Forest, poly, 0, 0));
    });

    // ────────────────────────── MAJOR ROAD aggregation ───────────────────────
    // Build the low-zoom network from road route relations that touch a motorway,
    // joining their member ways into continuous lines clipped into the coarse
    // (z <= ROAD_AGG_MAX_ZOOM) tiles. Raw ways cover z >= 10.
    println!("Aggregating major roads from route relations...");
    route_roads::build(
        &mut osm_reader,
        &index,
        &node_store,
        &sink,
        progress.clone(),
    );

    sink.finish().unwrap();
    println!(
        "Geometry extraction complete — records spilled under {}",
        args.spill
    );

    println!("Building archive {} ...", args.pmtiles);
    let writer = Arc::new(Mutex::new(
        PmTilesWriter::create(&args.pmtiles)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", args.pmtiles)),
    ));

    let reader = cache::bucket::BucketReader::new(&args.spill, params.bucket_count());

    reader.par_bridge().for_each(|mut bucket| {
        let mut tiles: HashMap<u64, Vec<tiler::record::TileRecord>> = HashMap::new();
        let spill_reader = spill::SpillReader::new(&mut bucket);
        for record in spill_reader {
            tiles.entry(record.tile_key).or_default().push(record);
        }

        if !tiles.is_empty() {
            let extent_u16 = params.extent as u16;
            let built: Vec<(Tile, Vec<u8>)> = tiles
                .iter()
                .map(|(&m, recs)| {
                    let blob = zstd::bulk::compress(
                        &format::build_tile(recs.iter(), extent_u16),
                        ZSTD_LEVEL,
                    )
                    .expect("zstd compress");
                    (params.tile(m), blob)
                })
                .collect();
            let mut w = writer.lock().unwrap();
            for (tile, blob) in &built {
                w.write(*tile, blob).unwrap();
            }
        }
    });

    println!("Finalizing {} ...", args.pmtiles);
    writer.lock().unwrap().finish().unwrap();
    println!("Wrote {}", args.pmtiles);
}
