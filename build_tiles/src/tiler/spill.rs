//! Tile record spill bucket handling

use super::TileParams;
use super::record::{TileRecord, decode, encode};
use cache::bucket::BucketWriter;
use std::io::Read;
use std::path::Path;

/// Tile record spill writer
pub struct SpillWriter {
    params: TileParams,
    buckets: BucketWriter,
}

impl SpillWriter {
    pub fn new(path: impl AsRef<Path>, params: TileParams) -> Self {
        let bucket_count = params.bucket_count();
        Self {
            params,
            buckets: BucketWriter::create(path, bucket_count).unwrap(),
        }
    }

    pub fn append(&self, record: TileRecord) {
        let bucket_idx = self.params.route(record.tile_key);
        let mut buffer = Vec::<u8>::new();

        buffer.extend(&0u32.to_le_bytes());
        encode(&mut buffer, &record);
        let size = ((buffer.len() - core::mem::size_of::<u32>()) as u32).to_le_bytes();
        buffer[..size.len()].copy_from_slice(&size);

        self.buckets.append(bucket_idx, &buffer);
    }

    pub fn flush(&self) {
        self.buckets.finish().unwrap();
    }
}

/// Decoder/reader of tile record spill IO buffers
pub struct SpillReader<'a, B> {
    buffer: &'a mut B,
}

impl<'a, B: Read> SpillReader<'a, B> {
    pub fn new(buffer: &'a mut B) -> Self {
        Self { buffer }
    }
}

impl<'a, B: Read> Iterator for SpillReader<'a, B> {
    type Item = TileRecord;

    fn next(&mut self) -> Option<Self::Item> {
        let mut size = [0u8; 4];

        if let Err(err) = self.buffer.read_exact(&mut size) {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                return None;
            }

            panic!("IO error: {err:?}");
        }

        let size = u32::from_le_bytes(size);
        let mut record_payload = vec![0u8; size as usize];

        self.buffer.read_exact(&mut record_payload).unwrap();

        Some(decode(&mut record_payload.as_slice()).unwrap())
    }
}
