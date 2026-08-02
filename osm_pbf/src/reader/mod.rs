use crate::protos::osmpbf::{Blob, BlobHeader, PrimitiveBlock};
use error_stack::{Report, ResultExt};
use prost::Message;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek};
use std::path::Path;

#[derive(Debug, thiserror::Error, Clone)]
pub enum OsmBlobReaderError {
    #[error("Failed to read OSM file")]
    Read,
    #[error("Failed to decode OSM file segment")]
    Decode,
}

/// Scalar type for storing size of a blob segment inside the PBF file
type BlobSize = i32;

/// Helper macro for handling reading data structures from IO interface
macro_rules! read_exact {
    ($from:expr, $to:expr) => {
        if let Err(err) = $from.read_exact($to) {
            if err.kind() == ErrorKind::UnexpectedEof {
                return None;
            }

            return Some(Err(err).change_context(OsmBlobReaderError::Read));
        }
    };
}

/// Helper macro for handling generic decoding segments to abstract error propagation
macro_rules! try_decode {
    ($d:expr) => {
        match $d {
            Err(err) => return Some(Err(err)),
            Ok(v) => v,
        }
    };
}

/// OSM PBF file reader and decoder
pub struct OsmReader<T> {
    input: T,
    header_len_buffer: [u8; size_of::<BlobSize>()],
    header_buffer: Vec<u8>,
    blob_buffer: Vec<u8>,
}

impl OsmReader<BufReader<File>> {
    pub fn from_file(path: impl AsRef<Path>) -> Self {
        OsmReader::new(BufReader::new(File::open(path).unwrap()))
    }
}

impl<T: Read + Seek> OsmReader<T> {
    /// String label for data block type (BlobHeader::type)
    const DATA_TYPE: &str = "OSMData";

    pub fn new(input: T) -> Self {
        Self {
            input,
            header_len_buffer: [0; size_of::<BlobSize>()],
            header_buffer: Vec::new(),
            blob_buffer: Vec::new(),
        }
    }

    pub fn set_position(&mut self, position: u64) -> Result<(), Report<OsmBlobReaderError>> {
        self.input
            .seek(std::io::SeekFrom::Start(position))
            .change_context(OsmBlobReaderError::Read)
            .map(|_| ())
    }

    /// Get total length of stream and reset to begining
    pub fn get_len_and_reset(&mut self) -> Result<u64, Report<OsmBlobReaderError>> {
        self.input
            .seek(std::io::SeekFrom::End(0))
            .change_context(OsmBlobReaderError::Read)?;

        let end = self
            .input
            .stream_position()
            .change_context(OsmBlobReaderError::Read)?;

        self.input
            .seek(std::io::SeekFrom::Start(0))
            .change_context(OsmBlobReaderError::Read)?;

        Ok(end)
    }

    fn read_next_blob(&mut self) -> Option<Result<(RawBlob, u64), Report<OsmBlobReaderError>>> {
        let blob_size: usize;

        let position = match self
            .input
            .stream_position()
            .change_context(OsmBlobReaderError::Read)
        {
            Ok(pos) => pos,
            Err(err) => return Some(Err(err)),
        };

        loop {
            // Read the next header size as raw Big Endian integer directly from file cursor
            read_exact!(self.input, &mut self.header_len_buffer);
            let header_len_buffer_size = BlobSize::from_be_bytes(self.header_len_buffer);

            // Read and decode next header
            self.header_buffer
                .resize(header_len_buffer_size as usize, 0);
            read_exact!(self.input, self.header_buffer.as_mut());

            let blob_header = try_decode!(
                BlobHeader::decode(self.header_buffer.as_slice())
                    .change_context(OsmBlobReaderError::Decode)
            );

            // If header indicates that next blob is not a data blob, skip over that file sector
            if blob_header.r#type == Self::DATA_TYPE {
                blob_size = blob_header.datasize as _;

                break;
            }

            if let Err(err) = self.input.seek_relative(blob_header.datasize as _) {
                return Some(Err(err).change_context(OsmBlobReaderError::Read));
            }
        }

        // Read and decode the next data blob
        self.blob_buffer.resize(blob_size, 0);
        read_exact!(self.input, self.blob_buffer.as_mut());

        let blob = RawBlob(core::mem::take(&mut self.blob_buffer));

        Some(Ok((blob, position)))
    }
}

impl<T: Read + Seek> Iterator for &mut OsmReader<T> {
    type Item = Result<(RawBlob, u64), Report<OsmBlobReaderError>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_next_blob()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to decode OSM data blob")]
pub struct OsmBlobDecodeError;

pub struct RawBlob(Vec<u8>);

impl RawBlob {
    pub fn decode(self) -> Result<PrimitiveBlock, Report<OsmBlobDecodeError>> {
        let blob = Blob::decode(self.0.as_slice()).change_context(OsmBlobDecodeError)?;

        let deflated = blob.extract().change_context(OsmBlobDecodeError)?;

        PrimitiveBlock::decode(deflated).change_context(OsmBlobDecodeError)
    }
}
