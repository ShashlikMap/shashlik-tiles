use crate::protos::osmpbf::{Blob, blob::Data};
use bytes::{BufMut, BytesMut};
use error_stack::{Report, ResultExt};
use flate2::read::ZlibDecoder;
use std::io::{Cursor, Read, copy};
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum BlobExtractorError {
    #[error("Unsupported compression method")]
    UnsupportedCompression,
    #[error("Failed while decompressing")]
    Decompress,
}

impl Blob {
    fn extractor(self) -> Result<impl Read, BlobExtractorError> {
        match self.data {
            Some(Data::ZlibData(data)) => Ok(ZlibDecoder::new(Cursor::new(data))),
            _ => Err(BlobExtractorError::UnsupportedCompression),
        }
    }

    pub fn extract(self) -> Result<BytesMut, Report<BlobExtractorError>> {
        let mut extractor = self.extractor()?;
        let mut output = BytesMut::new().writer();
        copy(&mut extractor, &mut output).change_context(BlobExtractorError::Decompress)?;

        Ok(output.into_inner())
    }
}
