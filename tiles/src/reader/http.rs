//! `RangeReader` backed by HTTP(S) range requests (reqwest)

use super::RangeReader;
use async_trait::async_trait;
use reqwest::{Client, StatusCode, Url, header::RANGE};
use std::io;

/// A `RangeReader` that fetches byte ranges of a single remote object (e.g. a
/// PMTiles archive) over HTTP(S). Cheap to clone via a shared connection pool;
/// safe to share across concurrent requests.
pub struct HttpRangeReader {
    client: Client,
    url: Url,
}

impl HttpRangeReader {
    /// Open a remote object at `url`. Validates the URL and builds a reusable
    /// client (connection pooling / keep-alive across ranged reads).
    pub async fn open(url: impl reqwest::IntoUrl) -> io::Result<Self> {
        let url = url.into_url().map_err(reqwest_err)?;
        let client = Client::builder().build().map_err(reqwest_err)?;
        Ok(Self { client, url })
    }
}

#[async_trait]
impl RangeReader for HttpRangeReader {
    async fn read_range(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        // HTTP byte ranges are inclusive on both ends.
        let end = offset + len as u64 - 1;
        let resp = self
            .client
            .get(self.url.clone())
            .header(RANGE, format!("bytes={offset}-{end}"))
            .send()
            .await
            .map_err(reqwest_err)?;

        let status = resp.status();
        let body = resp.bytes().await.map_err(reqwest_err)?;
        match status {
            // Normal case: the server honoured the range.
            StatusCode::PARTIAL_CONTENT => Ok(body.to_vec()),
            s => Err(io::Error::other(format!(
                "unexpected HTTP status {s} for ranged GET"
            ))),
        }
    }
}

fn reqwest_err(e: reqwest::Error) -> io::Error {
    io::Error::other(e)
}
