//! Flat id-space bitset with O(1) rank.
//!
//! Sized to `capacity` bits (`max_id + 1` for the file being processed), so it
//! scales down for extracts and up for the planet. Bits are set concurrently
//! during the parallel collection pass via atomics (one shared bitset, ~1 bit
//! per id — never per-thread copies, which would be gigabytes each). Once the
//! set phase is done it is frozen into a `RankedBitSet` whose `rank(id)`
//! gives the dense ordinal of a set bit — the address into the sparse store.

use std::sync::atomic::{AtomicU64, Ordering};

/// Words per rank block. `rank` sums at most this many `popcount`s on top of the
/// precomputed per-block prefix, so it trades index size against query cost.
/// 16 words = 1024 bits/block -> ~`capacity / 128` bytes of index (~100 MB for a
/// planet-scale ~13e9 id space).
const BLOCK_WORDS: usize = 16;
const BLOCK_BITS: u64 = (BLOCK_WORDS as u64) * 64;

/// A growable-free, fixed-capacity bitset over `0..capacity`.
///
/// `BitSet::set` takes `&self` so the bitset can be shared across rayon
/// workers during the collection pass; concurrent sets are lock-free atomic
/// `fetch_or`s. Reads after the set phase rely on the thread-join happens-before
/// for visibility.
pub struct BitSet {
    words: Vec<AtomicU64>,
    capacity: u64,
}

impl BitSet {
    /// Allocate a bitset able to hold ids in `0..capacity` (all clear).
    pub fn with_capacity(capacity: u64) -> Self {
        let nwords = capacity.div_ceil(64) as usize;
        let mut words = Vec::with_capacity(nwords);
        words.resize_with(nwords, || AtomicU64::new(0));
        Self { words, capacity }
    }

    /// Set the bit for `id`. Safe to call concurrently from many threads.
    #[inline]
    pub fn set(&self, id: u64) {
        debug_assert!(id < self.capacity, "id {id} out of bitset capacity");
        let word = (id / 64) as usize;
        let bit = id % 64;
        self.words[word].fetch_or(1u64 << bit, Ordering::Relaxed);
    }

    /// Set the bit for `id`, returning whether it was already set. Safe to
    /// call concurrently — exactly one racing caller observes `false` (the first
    /// to set it), which makes it a lock-free "have I seen this id before?".
    #[inline]
    pub fn test_and_set(&self, id: u64) -> bool {
        debug_assert!(id < self.capacity, "id {id} out of bitset capacity");
        let word = (id / 64) as usize;
        let mask = 1u64 << (id % 64);
        self.words[word].fetch_or(mask, Ordering::Relaxed) & mask != 0
    }

    /// Test whether `id`'s bit is set.
    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        if id >= self.capacity {
            return false;
        }
        let word = (id / 64) as usize;
        let bit = id % 64;
        (self.words[word].load(Ordering::Relaxed) >> bit) & 1 == 1
    }

    /// Number of bits currently set (linear scan — call once, not in a loop).
    pub fn count_ones(&self) -> u64 {
        self.words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }

    /// Highest addressable id + 1.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Freeze the set of bits and build the rank index.
    pub fn into_ranked(self) -> RankedBitSet {
        RankedBitSet::build(self)
    }
}

/// A frozen `BitSet` with a precomputed rank index.
///
/// `rank(id)` = number of set bits strictly below `id`. For an id whose bit is
/// set, that value is its dense 0-based ordinal — exactly the slot it occupies
/// in a `store::SparseStore`.
pub struct RankedBitSet {
    words: Vec<u64>,
    /// Cumulative count of set bits before each `BLOCK_WORDS`-word block, plus a
    /// trailing sentinel equal to the total.
    block_rank: Vec<u64>,
    capacity: u64,
    total: u64,
}

impl RankedBitSet {
    fn build(bits: BitSet) -> Self {
        let words: Vec<u64> = bits
            .words
            .iter()
            .map(|w| w.load(Ordering::Relaxed))
            .collect();

        let nblocks = words.len().div_ceil(BLOCK_WORDS);
        let mut block_rank = Vec::with_capacity(nblocks + 1);
        let mut acc = 0u64;
        for block in 0..nblocks {
            block_rank.push(acc);
            let start = block * BLOCK_WORDS;
            let end = (start + BLOCK_WORDS).min(words.len());
            for word in &words[start..end] {
                acc += word.count_ones() as u64;
            }
        }
        block_rank.push(acc); // sentinel: total set bits

        Self {
            words,
            block_rank,
            capacity: bits.capacity,
            total: acc,
        }
    }

    /// Test whether `id`'s bit is set.
    #[inline]
    pub fn contains(&self, id: u64) -> bool {
        if id >= self.capacity {
            return false;
        }
        (self.words[(id / 64) as usize] >> (id % 64)) & 1 == 1
    }

    /// Number of set bits strictly below `id` (in `0..id`).
    #[inline]
    pub fn rank(&self, id: u64) -> u64 {
        let id = id.min(self.capacity);
        let word = (id / 64) as usize;
        let block = (id / BLOCK_BITS) as usize;

        let mut acc = self.block_rank[block];
        for w in &self.words[block * BLOCK_WORDS..word] {
            acc += w.count_ones() as u64;
        }

        let bit = id % 64;
        if bit != 0 {
            let mask = (1u64 << bit) - 1;
            acc += (self.words[word] & mask).count_ones() as u64;
        }
        acc
    }

    /// The dense ordinal of `id` if its bit is set, else `None`. This is the
    /// store slot for `id`.
    #[inline]
    pub fn rank_if_set(&self, id: u64) -> Option<u64> {
        if self.contains(id) {
            Some(self.rank(id))
        } else {
            None
        }
    }

    /// Total number of set bits (= number of slots in a store addressed by this).
    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Highest addressable id + 1.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_matches_naive() {
        // Spread ids across several blocks and word boundaries.
        let ids = [
            0u64, 1, 63, 64, 65, 127, 1023, 1024, 4096, 4097, 9999, 10_000,
        ];
        let cap = 10_001;
        let bits = BitSet::with_capacity(cap);
        for &id in &ids {
            bits.set(id);
        }
        let ranked = bits.into_ranked();

        assert_eq!(ranked.len(), ids.len() as u64);

        // rank(id) == count of set ids strictly below id, for every position.
        for id in 0..cap {
            let naive = ids.iter().filter(|&&x| x < id).count() as u64;
            assert_eq!(ranked.rank(id), naive, "rank mismatch at id {id}");
        }

        // Set ids map to contiguous 0..len ordinals; unset ids yield None.
        let mut sorted = ids;
        sorted.sort_unstable();
        for (ordinal, &id) in sorted.iter().enumerate() {
            assert_eq!(ranked.rank_if_set(id), Some(ordinal as u64));
        }
        assert_eq!(ranked.rank_if_set(2), None);
        assert_eq!(ranked.rank_if_set(cap + 5), None);
    }

    #[test]
    fn test_and_set_reports_prior_state() {
        let bits = BitSet::with_capacity(200);
        // First touch is unset; subsequent touches are set.
        assert!(!bits.test_and_set(64));
        assert!(bits.test_and_set(64));
        assert!(bits.test_and_set(64));
        // A different bit in the same word is independent.
        assert!(!bits.test_and_set(65));
        assert!(bits.contains(64));
        assert!(bits.contains(65));
        assert!(!bits.contains(66));
        assert_eq!(bits.count_ones(), 2);
    }

    #[test]
    fn empty_bitset() {
        let ranked = BitSet::with_capacity(0).into_ranked();
        assert!(ranked.is_empty());
        assert_eq!(ranked.len(), 0);
        assert_eq!(ranked.rank_if_set(0), None);
    }
}
