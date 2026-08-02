//! mmap-backed sparse store: `id -> T` addressed by `rank(id)`.
//!
//! Only referenced ids get a slot, densely packed with no holes, so the file is
//! `set_bits * size_of::<T>()` regardless of how the ids scatter across the id
//! space. The payload lives on disk (mmap); heap holds only the `RankedBitSet`
//! and its rank index. Lookups are O(1): `rank(id)` -> slot -> one mmap read.
//!
//! The node location cache is `SparseStore<[i32; 2]>` (packed lon/lat), but the
//! store is generic over any `Pod` payload so the same primitive backs other
//! id-keyed caches.

use crate::bitset::RankedBitSet;
use bytemuck::Pod;
use memmap2::{Advice, Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::io;
use std::marker::PhantomData;
use std::path::Path;

/// Advise the kernel how the store mmap will be read. Callers now resolve coords
/// in sorted node-id order (per block), so the mmap is accessed roughly
/// *forward* — `Normal`/`Sequential` readahead pipelines those reads into
/// near-sequential throughput. `Random` disables readahead (right only for a
/// truly scattered gather). Selectable via `NODE_STORE_ADVISE` for tuning on
/// large, disk-resident stores; best-effort (a failed hint is not fatal).
fn advise_access(mmap: &Mmap) {
    let advice = match std::env::var("NODE_STORE_ADVISE").ok().as_deref() {
        Some("random") => Advice::Random,
        Some("sequential") => Advice::Sequential,
        Some("willneed") => Advice::WillNeed,
        _ => Advice::Normal,
    };
    let _ = mmap.advise(advice);
}

/// A `*mut T` promised to be shared only across disjoint slots. OSM ids are
/// unique, so distinct ids map (via `rank`) to distinct slots and concurrent
/// writes never alias — see `SparseStoreBuilder::set`.
struct SlotPtr<T>(*mut T);
// SAFETY: writes are guaranteed disjoint by the unique-id invariant.
unsafe impl<T> Send for SlotPtr<T> {}
unsafe impl<T> Sync for SlotPtr<T> {}

/// Writer side of a sparse store: a fixed-size mmap plus the address map.
///
/// `Self::set` takes `&self` so writes fan out across rayon workers with
/// no locking; call `Self::finish` to flush and get a reader.
pub struct SparseStoreBuilder<T: Pod> {
    bitset: RankedBitSet,
    mmap: MmapMut,
    base: SlotPtr<T>,
    _marker: PhantomData<T>,
}

impl<T: Pod> SparseStoreBuilder<T> {
    /// Create (or truncate) the backing file at `path`, sized to hold one `T`
    /// per set bit in `bitset`.
    pub fn create(path: impl AsRef<Path>, bitset: RankedBitSet) -> io::Result<Self> {
        let slots = bitset.len() as usize;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        // Guard against a zero-length map (illegal); an empty store never writes
        // or reads a slot anyway.
        let bytes = (slots * size_of::<T>()).max(1);
        file.set_len(bytes as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        let base = SlotPtr(mmap.as_mut_ptr() as *mut T);

        Ok(Self {
            bitset,
            mmap,
            base,
            _marker: PhantomData,
        })
    }

    /// Store `value` for `id`. No-op if `id` is not a referenced id (bit clear)
    #[inline]
    pub fn set(&self, id: u64, value: T) {
        if let Some(slot) = self.bitset.rank_if_set(id) {
            // SAFETY: `slot < bitset.len()` so it is in-bounds of the mapping,
            // and the unique-id invariant makes this write disjoint from all
            // other concurrent writes.
            unsafe { self.base.0.add(slot as usize).write(value) };
        }
    }

    /// Flush to disk and return a read-only view.
    pub fn finish(self) -> io::Result<SparseStore<T>> {
        self.mmap.flush()?;
        let mmap = self.mmap.make_read_only()?;
        advise_access(&mmap);
        Ok(SparseStore {
            bitset: self.bitset,
            mmap,
            _marker: PhantomData,
        })
    }
}

/// Reader side of a sparse store.
pub struct SparseStore<T: Pod> {
    bitset: RankedBitSet,
    mmap: Mmap,
    _marker: PhantomData<T>,
}

impl<T: Pod> SparseStore<T> {
    /// Open an existing store file, addressing it with `bitset` (which must be
    /// the same one it was built with).
    pub fn open(path: impl AsRef<Path>, bitset: RankedBitSet) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        advise_access(&mmap);
        Ok(Self {
            bitset,
            mmap,
            _marker: PhantomData,
        })
    }

    /// Fetch the payload for `id`, or `None` if `id` was never referenced.
    #[inline]
    pub fn get(&self, id: u64) -> Option<T> {
        let slot = self.bitset.rank_if_set(id)? as usize;
        let size = size_of::<T>();
        let bytes = &self.mmap[slot * size..slot * size + size];
        Some(*bytemuck::from_bytes(bytes))
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.bitset.len() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bitset.is_empty()
    }

    /// Borrow the address map (e.g. to test membership without a lookup).
    pub fn bitset(&self) -> &RankedBitSet {
        &self.bitset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::BitSet;

    #[test]
    fn roundtrip_packed_coords() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("cache-store-test-{}.bin", std::process::id()));

        let ids = [5u64, 9, 128, 4096];
        let bits = BitSet::with_capacity(4097);
        for &id in &ids {
            bits.set(id);
        }

        let builder = SparseStoreBuilder::<[i32; 2]>::create(&path, bits.into_ranked()).unwrap();
        for &id in &ids {
            builder.set(id, [id as i32, -(id as i32)]);
        }
        // Unset ids are ignored.
        builder.set(6, [7, 7]);
        let store = builder.finish().unwrap();

        assert_eq!(store.len(), ids.len());
        for &id in &ids {
            assert_eq!(store.get(id), Some([id as i32, -(id as i32)]));
        }
        assert_eq!(store.get(6), None);
        assert_eq!(store.get(10_000), None);

        std::fs::remove_file(&path).ok();
    }
}
