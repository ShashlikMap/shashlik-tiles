//! Streaming sink for extracted geometry — the seam between extraction and the tiler

use crate::shapes::Shape;
use std::io;

/// Consumes geometry streamed from extraction.
///
/// `ShapeSink::push` is called **concurrently** from many extraction
/// workers, so implementors must be `Sync` and synchronize internally. It is
/// infallible on purpose — it sits on a hot path called once per shape. An
/// implementation that does I/O should record the first error internally and
/// surface it from `ShapeSink::finish`.
pub trait ShapeSink: Sync {
    /// Consume one extracted shape. Called concurrently from extraction workers.
    fn push(&self, shape: Shape);

    /// Flush and finalize once extraction is complete. Called exactly once,
    /// single-threaded, after every `push` has returned. Default: no-op.
    fn finish(&self) -> io::Result<()> {
        Ok(())
    }
}

/// A sink that discards everything — isolates extraction cost from any downstream
/// work for profiling the extraction passes on their own. Not wired to a CLI flag.
#[allow(dead_code)]
pub struct NoopSink;

impl ShapeSink for NoopSink {
    fn push(&self, _shape: Shape) {}
}
