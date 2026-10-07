pub mod borders;
pub mod classifier;
pub mod extractor;
pub mod major_roads;
pub mod mask;
pub mod node_cache;
pub mod relation_features;
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
use tiles::{LatLon, Tile};

/// zstd level for tile payloads (the reader zstd-decompresses).
const ZSTD_LEVEL: i32 = 9;

/// Margin added to each side of the OSM extract's own bounding box (as a
/// fraction of that axis's own width/height) before it's used to clip the
/// water-polygon union -- see `clip_water_to_bbox`. A first-cut, tunable
/// value: guards against a hard coastline cutoff exactly at the extract's
/// own edge (from simplification nudging a boundary vertex slightly, or
/// the extract's own clip not landing exactly on a tile boundary), without
/// a dedicated CLI flag for something this narrow.
const WATER_BBOX_MARGIN_FRACTION: f64 = 0.1;

/// Clips `water` (already Web Mercator meters, see `read_water_polygons`)
/// down to `osm_bbox` (WGS84 degrees, from `node_cache::build_node_store`)
/// expanded by `WATER_BBOX_MARGIN_FRACTION` on each axis, then projected to
/// Mercator -- the two aren't in the same coordinate system otherwise (see
/// this project's own plan notes: the water shapefile has no CRS metadata
/// at all and is only ever Mercator by convention, while OSM node
/// coordinates are WGS84 degrees). Doesn't handle antimeridian-crossing
/// extracts (lon past +/-180 after margin expansion) -- not a real concern
/// for a single-region regional extract.
fn clip_water_to_bbox(water: geo::MultiPolygon<f64>, osm_bbox: geo::Rect<f64>) -> geo::MultiPolygon<f64> {
    let min = osm_bbox.min();
    let max = osm_bbox.max();
    let margin_x = (max.x - min.x) * WATER_BBOX_MARGIN_FRACTION;
    let margin_y = (max.y - min.y) * WATER_BBOX_MARGIN_FRACTION;

    let merc_min = LatLon::new(min.y - margin_y, min.x - margin_x).to_mercator();
    let merc_max = LatLon::new(max.y + margin_y, max.x + margin_x).to_mercator();
    let rect = geo::Rect::new(
        geo::Coord { x: merc_min.x(), y: merc_min.y() },
        geo::Coord { x: merc_max.x(), y: merc_max.y() },
    );

    geo::algorithm::bool_ops::BooleanOps::intersection(&water, &rect.to_polygon())
}

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

    /// Materialized zoom levels, comma-separated (client overzooms the gaps).
    /// The highest value is the base/grid zoom (full detail); the lowest is the
    /// pyramid floor. Built one at a time and flushed, so peak memory is a
    /// single zoom's tile set — trim the finest zooms if the whole planet at
    /// z12/z14 doesn't fit. Written into the archive metadata so the client can
    /// read it back instead of hardcoding a matching list.
    #[arg(long, value_delimiter = ',', default_value = "14,12,10,8,6,4")]
    zooms: Vec<u8>,

    /// Simplification tolerance for pyramid zooms (the base zoom is never simplified), in rendered
    /// pixels (converted to tile-local units via `--tile-render-px`). Areas
    /// only -- roads are never simplified, at any zoom (see
    /// `tiler::sink::TileSink::clip_road_at`'s own doc comment: independent
    /// per-way simplification visibly disconnects road networks at
    /// junctions, so this is a hardcoded exemption, not something either
    /// of these flags can turn back on).
    #[arg(long, default_value_t = 4.0)]
    simplify_px: f64,

    /// Assumed on-screen tile size in pixels, used to convert
    /// `--simplify-px` and the built-in area-cull
    /// thresholds into tile-local units. Defaults to the ESP32-P4 renderer's
    /// actual `maptile_ppa` raster canvas width (`RASTER_WIDTH` in
    /// `p4-playground/src/maptile.rs`), not the generic web-map assumption
    /// of 512 this tool originally hardcoded.
    #[arg(long, default_value_t = 360.0)]
    tile_render_px: f64,

    /// Coarsest zoom materialized from the merged/aggregated (cheap) forest
    /// mask instead of raw per-feature area geometry.
    #[arg(long, default_value_t = 12)]
    agg_max_zoom: u8,

    /// Coarsest zoom materialized from merged major-road strokes instead of
    /// raw per-way road geometry. See `major_roads.rs`'s module doc comment
    /// for why this touches (rather than being strictly disjoint from) the
    /// raw `Motorway`/`Trunk`/`Primary` zoom range regardless of this value.
    #[arg(long, default_value_t = 12)]
    road_agg_max_zoom: u8,

    /// Coarse Morton-prefix zoom that groups tiles into spill buckets
    /// (`bucket_count = 4^bucket_zoom` -- 4096 at the default `6`). Each
    /// touched bucket keeps one file open for the whole extraction pass
    /// (`cache::bucket::BucketWriter` opens lazily but never closes until
    /// the run ends), so a data-dense extract spanning many bucket-zoom
    /// tiles can exceed the OS's open-file-descriptor limit
    /// ("Too many open files", `EMFILE`) well before all 4096 buckets are
    /// even touched. Lowering this trades fewer, larger buckets (fewer
    /// open files) for more memory held per bucket when it's read back and
    /// compressed into tiles (`main.rs`'s `reader.par_bridge()` loop reads
    /// one whole bucket's records into a `HashMap` at a time) -- raise your
    /// shell's `ulimit -n` instead if you'd rather keep more, smaller
    /// buckets than lower this.
    #[arg(long, default_value_t = 6)]
    bucket_zoom: u8,

    /// Clip the water/coastline polygon union (`--water-shapefile`) down to
    /// this OSM extract's own bounding box (with a margin) before tiling,
    /// instead of tiling every water polygon the shapefile contains --
    /// `--water-shapefile` is conventionally a *global* coastline dataset,
    /// while the `.osm.pbf` being tiled is usually a regional extract, so
    /// without this, water far outside the extract's own coverage still
    /// gets written into the output archive. A no-op without
    /// `--water-shapefile`. The bbox itself is always available (freshly
    /// computed or loaded from its cache sidecar, see
    /// `node_cache::save_bbox`/`load_bbox`) regardless of whether the node
    /// store was built fresh or reused this run.
    #[arg(long)]
    clip_water_to_extract_bbox: bool,
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

    let mut zooms = args.zooms;
    zooms.sort_unstable();
    zooms.dedup();
    let (min_zoom, grid_zoom) = match (zooms.first(), zooms.last()) {
        (Some(&min), Some(&max)) => (min, max),
        _ => panic!("--zooms must list at least one zoom level"),
    };
    let params = tiler::TileParams::new(
        grid_zoom,
        min_zoom,
        args.bucket_zoom,
        8192,
        args.simplify_px,
        args.tile_render_px,
    );
    println!(
        "Spill buckets: {} (--bucket-zoom {})",
        params.bucket_count(),
        args.bucket_zoom
    );

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
        // Country-boundary relations: keep them so their member-way nodes get
        // cached in the node store (the border builder resolves geometry from it).
        // `admin_level=2` is tight — only country boundaries.
        vec![("admin_level".into(), vec!["2".into()])],
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

    // `osm_bbox` (WGS84 degrees, over every decoded node in the file) is
    // computed fresh in `build_node_store`, but also persisted alongside the
    // node-cache file itself (`node_cache::save_bbox`/`load_bbox`) so a later
    // cache-reuse run can load it back instead of losing it -- a `.nodes`
    // cache written *before* that sidecar existed at all (e.g. from testing
    // an earlier version of this tool) has no sidecar to load, so that's
    // treated as a full cache miss below, not a silent no-bbox fallback --
    // real testing found the earlier "just skip the optimization silently"
    // version of this was actually the default experience for anyone
    // re-running against an extract they'd already built once.
    let bbox_cache_path = node_cache::bbox_cache_path(&node_cache_path);
    let cached_bbox = node_cache::load_bbox(&bbox_cache_path);
    let (node_store, osm_bbox) = if node_cache::node_store_matches(&node_cache_path, node_bitset.len() as usize)
        && cached_bbox.is_some()
    {
        println!("Reusing node store at {}...", node_cache_path.display());
        (node_cache::NodeStore::open(&node_cache_path, node_bitset).unwrap(), cached_bbox.unwrap())
    } else {
        println!("Writing node store...");
        let (store, bbox) = node_cache::build_node_store(
            &mut osm_reader,
            &index,
            node_bitset,
            &node_cache_path,
            progress.clone(),
        )
        .unwrap();
        indicator.finish();
        indicator.unset_length();
        (store, bbox)
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

    let sink = TileSink::new(&args.spill, params, zooms, args.agg_max_zoom, args.road_agg_max_zoom);
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

        let merged_water = if args.clip_water_to_extract_bbox {
            let before = merged_water.0.len();
            let clipped = clip_water_to_bbox(merged_water, osm_bbox);
            println!(
                "Water polygons clipped to extract bbox (+{:.0}% margin): {before} -> {}",
                WATER_BBOX_MARGIN_FRACTION * 100.0,
                clipped.0.len()
            );
            clipped
        } else {
            merged_water
        };

        let pb = ProgressBar::new(merged_water.0.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} water polygons tiled",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        merged_water.0.into_par_iter().for_each(|poly| {
            sink.push(Shape::Area(Area::new(AreaKind::Water, poly, -1, 0)));
            pb.inc(1);
        });
        pb.finish();
    }
    // ─────────────────────────── FOREST aggregation ──────────────────────────
    println!("Aggregating forests from mask...");
    let merged_forests = forest_mask.merged_polygons();
    println!("Merged forest polygons: {}", merged_forests.len());
    // One unit of work = one merged forest blob clipped into the coarse tiles.
    let pb = ProgressBar::new(merged_forests.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} forest blobs tiled",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    merged_forests.into_par_iter().for_each(|poly| {
        sink.push_aggregated(&Area::new(AreaKind::Forest, poly, 0, 0));
        pb.inc(1);
    });
    pb.finish();
    // ─────────────────────── RELATION-DRIVEN FEATURES ────────────────────────
    println!("Building relation-driven features (major roads, borders)...");
    relation_features::run(
        &mut osm_reader,
        &index,
        &node_store,
        &sink,
        &[&route_roads::RouteRoads, &borders::Borders],
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
