//! `RangeReader` backed by a local file

use super::RangeReader;
use async_trait::async_trait;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

pub struct FileRangeReader {
    file: Arc<File>,
}

impl FileRangeReader {
    /// Open `path` for random-access reads.
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = tokio::task::spawn_blocking(move || File::open(path))
            .await
            .map_err(join_err)??;
        Ok(Self {
            file: Arc::new(file),
        })
    }
}

#[async_trait]
impl RangeReader for FileRangeReader {
    async fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let file = self.file.clone();
        let data = tokio::task::spawn_blocking(move || {
            let mut buf = vec![0u8; len];
            file.read_exact_at(&mut buf, offset)?;
            io::Result::Ok(buf)
        })
        .await
        .map_err(join_err)?;

        log::debug!("Fetched {:.2}KB", (len as f64) / 1000.0);

        data
    }
}

fn join_err(e: tokio::task::JoinError) -> io::Error {
    io::Error::other(e)
}
