//! PBF file section index.
//!
//! OSM PBF blocks are ordered nodes → ways → relations. A single scan records
//! the byte offset of the first way block and the first relation block, so every
//! later pass can seek straight to the section it cares about. The index is a
//! 24-byte `Pod` persisted next to the source file as `<name>.idx`.

use bytemuck::{Pod, PodCastError, Zeroable, bytes_of, try_from_bytes};
use osm_pbf::{protos::Primitives, reader::OsmReader};
use rayon::prelude::*;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("{0}")]
    IO(io::Error),
    #[error("{0}")]
    Cast(PodCastError),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OsmPbfIndex {
    /// Byte offset of the first block containing ways.
    pub ways: u64,
    /// Byte offset of the first block containing relations.
    pub relations: u64,
    /// Total file length.
    pub end: u64,
    /// Highest node id in the file — used to size the node-id bitset/store.
    pub max_node_id: i64,
    /// Highest way id in the file — used to size the way-id bitset/store.
    pub max_way_id: i64,
}

impl OsmPbfIndex {
    pub fn get_relative_path(pbf: impl AsRef<Path>) -> PathBuf {
        pbf.as_ref().with_extension("idx")
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Option<Self>, OpenError> {
        let buffer = match std::fs::read(path) {
            Err(err) => {
                if let io::ErrorKind::NotFound = err.kind() {
                    return Ok(None);
                } else {
                    return Err(OpenError::IO(err));
                }
            }
            Ok(file) => file,
        };

        // A wrong-sized index means the on-disk format changed; treat it as
        // absent so it is regenerated rather than erroring.
        if buffer.len() != size_of::<Self>() {
            return Ok(None);
        }

        let this = try_from_bytes(&buffer).map_err(OpenError::Cast)?;

        Ok(Some(*this))
    }

    pub fn open_or_create(
        path: impl AsRef<Path>,
        reader: &mut osm_pbf::reader::OsmReader<io::BufReader<File>>,
        progress: Arc<impl Fn(u64, u64) + Send + Sync>,
    ) -> Result<Self, OpenError> {
        let index_path = Self::get_relative_path(path);
        let index = match Self::open(Self::get_relative_path(index_path.clone()))? {
            Some(file_index) => file_index,
            None => {
                let len = reader.get_len_and_reset().unwrap();
                let file_index = Self::index(reader, len, |pos| {
                    progress(pos, len);
                });
                file_index.write(index_path).unwrap();

                file_index
            }
        };

        Ok(index)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        std::fs::write(path, bytes_of(self))
    }

    pub fn index(
        reader: &mut OsmReader<io::BufReader<File>>,
        file_len: u64,
        cb: impl Fn(u64) + Send + Sync,
    ) -> Self {
        let (ways, relations, max_node_id, max_way_id) = reader
            .map(|res| {
                let (block, pos) = res.unwrap();
                cb(pos);

                (block, pos)
            })
            .par_bridge()
            .map(|(blob, pos)| {
                let block = blob.decode().unwrap();

                let mut ways: Option<u64> = None;
                let mut relations: Option<u64> = None;
                let mut max_node_id = 0i64;
                let mut max_way_id = 0i64;

                for grp in block.primitivegroup {
                    match grp.primitives() {
                        // Dense node ids are delta-coded and ascending, so the
                        // sum of the deltas is the last (= max) id in the block.
                        Primitives::DenseNodes(dense) => {
                            max_node_id = max_node_id.max(dense.id.iter().sum::<i64>());
                        }
                        Primitives::Nodes(nodes) => {
                            if let Some(m) = nodes.iter().map(|n| n.id).max() {
                                max_node_id = max_node_id.max(m);
                            }
                        }
                        Primitives::Ways(w) => {
                            ways = Some(pos);
                            if let Some(m) = w.iter().map(|x| x.id).max() {
                                max_way_id = max_way_id.max(m);
                            }
                        }
                        Primitives::Relations(_) => {
                            relations = Some(pos);
                        }
                        _ => {}
                    }
                }

                (ways, relations, max_node_id, max_way_id)
            })
            .reduce(
                || (None, None, 0i64, 0i64),
                |a, b| {
                    let ways = match a.0 {
                        Some(a_way) => Some(b.0.map(|b_way| a_way.min(b_way)).unwrap_or(a_way)),
                        None => b.0,
                    };

                    let relations = match a.1 {
                        Some(a_rel) => Some(b.1.map(|b_rel| a_rel.min(b_rel)).unwrap_or(a_rel)),
                        None => b.1,
                    };

                    (ways, relations, a.2.max(b.2), a.3.max(b.3))
                },
            );

        Self {
            ways: ways.unwrap(),
            relations: relations.unwrap(),
            end: file_len,
            max_node_id,
            max_way_id,
        }
    }
}
