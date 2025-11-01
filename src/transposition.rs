//! # Transposition Table Implementation
//!
//! Lock-free transposition table for sharing MCTS statistics across symmetric game states.
//!
//! ## Purpose
//!
//! In games with symmetries (like Zèrtz's rotational symmetry), many different board
//! configurations are strategically equivalent. The transposition table:
//! 1. **Canonicalizes** states to a standard form
//! 2. **Hashes** canonical states using Zobrist hashing
//! 3. **Shares statistics** across all symmetric variants
//! 4. **Detects collisions** using exact state matching
//!
//! ## Thread Safety
//!
//! **DashMap** for concurrent access:
//! - Lock-free concurrent reads and writes
//! - Uses internal sharding to reduce contention
//! - Better than `RwLock<HashMap>` for high-concurrency workloads
//!
//! **Atomic statistics** in entries:
//! - `AtomicU32` for visit counts
//! - `AtomicI32` for scaled total values
//! - Multiple threads can update same entry without locks
//!
//! ## Collision Handling
//!
//! **Chaining strategy**:
//! - Each hash maps to `Vec<Arc<TranspositionEntry>>`
//! - On collision, append to chain and increment collision counter
//! - Linear search through chain using exact state matching
//!
//! **Why chaining works well**:
//! - Zobrist hashing has very low collision rate (64-bit hashes)
//! - Chains typically have 1-2 entries
//! - Simpler than open addressing or Robin Hood hashing
//!
//! ## Memory Layout
//!
//! ```text
//! DashMap<u64, Vec<Arc<Entry>>>
//!     │
//!     ├─ hash_1 → [Entry_A]
//!     ├─ hash_2 → [Entry_B, Entry_C]  ← collision, chain length = 2
//!     └─ hash_3 → [Entry_D]
//! ```

use crate::game_trait::MCTSGame;
use dashmap::{mapref::entry::Entry, DashMap};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;

// ============================================================================
// TRANSPOSITION ENTRY
// ============================================================================

/// Statistics entry shared across canonical-equivalent game states
///
/// **Canonical state storage**:
/// Stores the canonical (normalized) state for exact collision detection.
/// When looking up a state, we:
/// 1. Canonicalize the query state
/// 2. Hash it with Zobrist
/// 3. Check chain for exact canonical match
///
/// **Lock-free updates**:
/// All statistics use atomic operations for thread-safe concurrent updates.
pub struct TranspositionEntry {
    visits: AtomicU32,
    total_value: AtomicI32, // Scaled by 1000 for precision (f32 → i32 for atomic ops)

    // Stored canonical state for collision detection
    canonical_spatial_state: Arc<Array3<f32>>, // Canonical spatial_state state
    canonical_global_state: Arc<Array1<f32>>,  // Canonical global_state state

    // Debug-only: Track virtual loss count to catch mismatched add/remove
    #[cfg(debug_assertions)]
    virtual_loss_count: AtomicU32,
}

impl TranspositionEntry {
    pub fn new(canonical_spatial_state: Array3<f32>, canonical_global_state: Array1<f32>) -> Self {
        Self {
            visits: AtomicU32::new(0),
            total_value: AtomicI32::new(0),
            canonical_spatial_state: Arc::new(canonical_spatial_state),
            canonical_global_state: Arc::new(canonical_global_state),
            #[cfg(debug_assertions)]
            virtual_loss_count: AtomicU32::new(0),
        }
    }

    /// Check if this entry matches the given canonical state
    ///
    /// Used for collision detection in the hash chain.
    /// Two states match if BOTH spatial_state and global_state arrays are identical.
    pub fn matches(&self, spatial_state: &Array3<f32>, global_state: &ArrayView1<f32>) -> bool {
        // Compare both spatial_state and global_state state for exact match
        // Note: ndarray implements efficient element-wise equality
        self.canonical_spatial_state.as_ref() == spatial_state && self.canonical_global_state.as_ref() == global_state
    }

    /// Get current visit count (thread-safe)
    ///
    /// Uses `Ordering::Relaxed` since we don't need synchronization with other memory operations.
    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    /// Get average value per visit (thread-safe)
    ///
    /// **Value scaling**: We store `total_value` as i32 (scaled by 1000) because:
    /// - Rust has no `AtomicF32` in stable
    /// - Scaling by 1000 preserves precision to 3 decimal places
    /// - Division by visits gives the average
    pub fn average_value(&self) -> f32 {
        let visits = self.visits();
        if visits == 0 {
            0.0
        } else {
            // Unscale: divide by 1000, then divide by visits for average
            (self.total_value.load(Ordering::Relaxed) as f32) / 1000.0 / (visits as f32)
        }
    }

    /// Add a new sample (simulation result) to statistics (thread-safe)
    ///
    /// **Atomic updates**: Both operations use `fetch_add` which is atomic but independent.
    /// This is safe because:
    /// - Reads always compute average from current visits/total_value snapshot
    /// - Slight inconsistency between visits and total_value during update is acceptable
    pub fn add_sample(&self, value: f32) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        let scaled = (value * 1000.0).round() as i32;
        self.total_value.fetch_add(scaled, Ordering::Relaxed);
    }

    /// Set statistics to specific values (overwrite)
    ///
    /// Used when restoring from storage or initializing from another source.
    /// **Not thread-safe** with concurrent `add_sample()` calls - use only during initialization.
    pub fn set_counts(&self, visits: u32, average_value: f32) {
        self.visits.store(visits, Ordering::Relaxed);
        // Compute total_value = average * visits, then scale by 1000
        let scaled_total = (average_value * 1000.0 * visits as f32).round() as i32;
        self.total_value.store(scaled_total, Ordering::Relaxed);
    }

    /// Add virtual loss to discourage thread collision
    #[inline]
    pub fn add_virtual_loss(&self) {
        use crate::node::{VIRTUAL_LOSS, VIRTUAL_LOSS_SCALED};

        #[cfg(debug_assertions)]
        self.virtual_loss_count.fetch_add(1, Ordering::Relaxed);

        self.visits.fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);
        self.total_value
            .fetch_add(VIRTUAL_LOSS_SCALED, Ordering::Relaxed);
    }

    /// Remove virtual loss after backpropagation
    #[inline]
    pub fn remove_virtual_loss(&self) {
        use crate::node::{VIRTUAL_LOSS, VIRTUAL_LOSS_SCALED};

        #[cfg(debug_assertions)]
        {
            let count = self.virtual_loss_count.fetch_sub(1, Ordering::Relaxed);
            debug_assert!(
                count > 0,
                "remove_virtual_loss() called on TranspositionEntry but virtual_loss_count was 0! \
                 This indicates a bug: remove called without matching add."
            );
        }

        self.visits.fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);
        self.total_value
            .fetch_sub(VIRTUAL_LOSS_SCALED, Ordering::Relaxed);
    }
}

/// Thread-safe transposition table using DashMap + game-specific hashing.
/// Uses chaining to handle hash collisions - each hash maps to a vector of entries.
pub struct TranspositionTable<G: MCTSGame> {
    game: Arc<G>,  // Game instance for hashing and canonicalization
    table: Arc<DashMap<u64, Vec<Arc<TranspositionEntry>>>>,
    collisions: AtomicU32, // Track number of hash collisions
}

impl<G: MCTSGame> TranspositionTable<G> {
    /// Create a new transposition table.
    pub fn new(game: Arc<G>) -> Self {
        Self {
            game,
            table: Arc::new(DashMap::new()),
            collisions: AtomicU32::new(0),
        }
    }

    /// Clone the table reference (shared underlying data).
    #[allow(dead_code)]
    pub fn clone_ref(&self) -> Self {
        Self {
            game: Arc::clone(&self.game),
            table: Arc::clone(&self.table),
            collisions: AtomicU32::new(0), // New instance has separate collision counter
        }
    }

    #[allow(dead_code)]
    /// Get the number of hash collisions detected
    pub fn collision_count(&self) -> u32 {
        self.collisions.load(Ordering::Relaxed)
    }

    /// Fetch or insert an entry for the canonicalized state (thread-safe)
    ///
    /// **Algorithm**:
    /// 1. Canonicalize the input state
    /// 2. Hash canonical state using Zobrist
    /// 3. Look up hash in DashMap
    /// 4. If occupied: linear search chain for exact match
    /// 5. If no match: create new entry and append to chain (collision)
    /// 6. If vacant: create new entry with new chain
    ///
    /// **Collision handling**:
    /// - Zobrist hashing has ~1 in 2^64 collision rate for random states
    /// - Chaining stores multiple entries per hash
    /// - Exact state matching ensures correctness despite collisions
    ///
    /// **Return value**: Always returns an `Arc<TranspositionEntry>`, either:
    /// - Existing entry (found in chain)
    /// - Newly created entry (added to chain or new hash)
    pub fn get_or_insert(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> Arc<TranspositionEntry> {
        // Step 1: Canonicalize (normalize to standard orientation) using game trait
        let (canonical_spatial_state, canonical_global_state) = self.game.canonicalize_state(spatial_state, global_state);

        // Step 2: Hash the canonical state using game trait
        let hash = self.game.hash_state(&canonical_spatial_state.view(), &canonical_global_state.view());

        // Step 3-6: DashMap entry API (lock-free for different hash buckets)
        match self.table.entry(hash) {
            Entry::Occupied(mut occupied) => {
                // Hash exists - search chain for exact match
                let chain = occupied.get_mut();

                // Linear search through chain (typically 1-2 entries)
                for entry in chain.iter() {
                    if entry.matches(&canonical_spatial_state, &canonical_global_state.view()) {
                        // Found exact match - return it
                        return Arc::clone(entry);
                    }
                }

                // No exact match found - this is a hash collision!
                // Two different states hashed to same value
                self.collisions.fetch_add(1, Ordering::Relaxed);

                // Create new entry and append to chain
                let new_entry = Arc::new(TranspositionEntry::new(
                    canonical_spatial_state,
                    canonical_global_state,
                ));
                chain.push(Arc::clone(&new_entry));
                new_entry
            }
            Entry::Vacant(vacant) => {
                // First time seeing this hash - create new chain with single entry
                let entry = Arc::new(TranspositionEntry::new(
                    canonical_spatial_state,
                    canonical_global_state,
                ));
                vacant.insert(vec![Arc::clone(&entry)]);
                entry
            }
        }
    }

    /// Lookup an entry without creating it.
    #[allow(dead_code)]
    pub fn lookup(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> Option<Arc<TranspositionEntry>> {
        let (canonical_spatial_state, canonical_global_state) = self.game.canonicalize_state(spatial_state, global_state);
        let hash = self.game.hash_state(&canonical_spatial_state.view(), &canonical_global_state.view());

        // Search chain for exact match
        self.table.get(&hash).and_then(|chain_ref| {
            chain_ref
                .value()
                .iter()
                .find(|entry| entry.matches(&canonical_spatial_state, &canonical_global_state.view()))
                .map(Arc::clone)
        })
    }

    /// Store (overwrite) statistics for a canonical state.
    pub fn store(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
        visits: u32,
        average_value: f32,
    ) {
        let entry = self.get_or_insert(spatial_state, global_state);
        entry.set_counts(visits, average_value);
    }

    /// Clear the table.
    pub fn clear(&self) {
        self.table.clear();
    }

    /// Number of entries.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.table.len()
    }
}

// NOTE: Default trait removed since TranspositionTable now requires a game instance

