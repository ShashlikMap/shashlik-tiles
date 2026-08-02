//! Backend-agnostic sink for finished tiles.
//! Contract:
//! - `write` may be called in any tile order; backends that need a particular
//!   order (PMTiles clusters by tile id) sort internally at `finish`.
//! - `data` is the final at-rest tile blob (already serialized and compressed);
//!   the container stores it verbatim.
//! - `finish` is called exactly once after all tiles; the writer is unusable
//!   afterwards.

use std::io;
use tiles::Tile;

pub trait TileWriter {
    /// Store one finished tile's bytes.
    fn write(&mut self, tile: Tile, data: &[u8]) -> io::Result<()>;

    /// Finalize the container (write index/footer, commit).
    fn finish(&mut self) -> io::Result<()>;
}
