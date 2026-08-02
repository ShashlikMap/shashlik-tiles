//! Append-only bucketed spill for stage-1 records.
//!
//! A fixed set of buckets (one per coarse Morton prefix), each an append-only
//! file with a buffered writer. Files are opened lazily on first write, so the
//! many empty (ocean) buckets never touch disk. Writers are shared and guarded
//! per-bucket (`B` files, not `workers × B`); `append` is called concurrently
//! from extraction workers.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Per-bucket write buffer size — bigger means fewer syscalls, more memory when
/// many buckets are active.
const BUCKET_BUF: usize = 256 * 1024;

fn bucket_path(dir: &Path, index: u32) -> PathBuf {
    dir.join(format!("bucket-{index:05}.spill"))
}

/// Simple iterator over all buckets
pub struct BucketReader {
    path: PathBuf,
    total_buckets: u32,
    current_bucket: u32,
}

impl BucketReader {
    pub fn new(path: impl Into<PathBuf>, total_buckets: u32) -> Self {
        Self {
            path: path.into(),
            total_buckets,
            current_bucket: 0,
        }
    }
}

impl Iterator for BucketReader {
    type Item = BufReader<File>;

    fn next(&mut self) -> Option<Self::Item> {
        // Advance past untouched (empty) buckets — they never created a file —
        // recomputing the path for each index until one opens or we run out.
        while self.current_bucket < self.total_buckets {
            let index = self.current_bucket;
            self.current_bucket += 1;

            let path = bucket_path(&self.path, index);
            if let Ok(file) = OpenOptions::new().read(true).open(&path) {
                return Some(BufReader::with_capacity(BUCKET_BUF, file));
            }
        }
        None
    }
}

struct LazyBucket {
    writer: Option<BufWriter<File>>,
}

/// A fixed set of append-only spill buckets under a directory.
pub struct BucketWriter {
    dir: PathBuf,
    buckets: Vec<Mutex<LazyBucket>>,
    /// First I/O error seen by any `append` (surfaced by `finish`).
    error: Mutex<Option<io::Error>>,
}

impl BucketWriter {
    /// Create `bucket_count` buckets under `dir` (created if absent). Existing
    /// spill files in `dir` are truncated on first write, not up front.
    pub fn create(dir: impl AsRef<Path>, bucket_count: u32) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let buckets = (0..bucket_count)
            .map(|_| Mutex::new(LazyBucket { writer: None }))
            .collect();
        Ok(Self {
            dir,
            buckets,
            error: Mutex::new(None),
        })
    }

    /// On-disk path of a bucket (for the stage-2 reader).
    #[cfg(test)]
    pub fn bucket_path(&self, index: u32) -> PathBuf {
        bucket_path(&self.dir, index)
    }

    /// Append `bytes` to bucket `index`. Infallible: the first I/O error is
    /// recorded and returned later by `Self::finish`. Safe to call
    /// concurrently.
    pub fn append(&self, index: u32, bytes: &[u8]) {
        debug_assert!((index as usize) < self.buckets.len(), "bucket index oob");
        if let Err(err) = self.try_append(index, bytes) {
            let mut slot = self.error.lock().unwrap();
            if slot.is_none() {
                *slot = Some(err);
            }
        }
    }

    fn try_append(&self, index: u32, bytes: &[u8]) -> io::Result<()> {
        let mut bucket = self.buckets[index as usize].lock().unwrap();
        if bucket.writer.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(bucket_path(&self.dir, index))?;
            bucket.writer = Some(BufWriter::with_capacity(BUCKET_BUF, file));
        }
        bucket.writer.as_mut().unwrap().write_all(bytes)
    }

    /// Flush every bucket and return the indices that received data. Returns the
    /// first append error, if any. Takes `&self` so it fits the sink lifecycle;
    /// files stay open (flushed content is readable) until the writer drops.
    pub fn finish(&self) -> io::Result<Vec<u32>> {
        if let Some(err) = self.error.lock().unwrap().take() {
            return Err(err);
        }
        let mut non_empty = Vec::new();
        for (index, bucket) in self.buckets.iter().enumerate() {
            let mut bucket = bucket.lock().unwrap();
            if let Some(writer) = bucket.writer.as_mut() {
                writer.flush()?;
                non_empty.push(index as u32);
            }
        }
        Ok(non_empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_route_to_buckets_and_flush() {
        let dir = std::env::temp_dir().join(format!("bucket-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let writer = BucketWriter::create(&dir, 16).unwrap();
        writer.append(3, b"hello");
        writer.append(3, b"world");
        writer.append(7, b"foo");

        let non_empty = writer.finish().unwrap();
        assert_eq!(non_empty, vec![3, 7]);

        assert_eq!(fs::read(writer.bucket_path(3)).unwrap(), b"helloworld");
        assert_eq!(fs::read(writer.bucket_path(7)).unwrap(), b"foo");
        // Untouched buckets never created a file.
        assert!(!writer.bucket_path(0).exists());

        drop(writer);
        let _ = fs::remove_dir_all(&dir);
    }
}
