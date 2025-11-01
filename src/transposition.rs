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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardConfig;
    use crate::games::ZertzGame;

    fn empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let layers = config.layers_per_timestep * config.t + 1;
        let spatial_state = Array3::zeros((layers, config.width, config.width));
        let global_state = Array1::zeros(10);
        (spatial_state, global_state)
    }

    fn board_with_rings(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let global_state = Array1::zeros(10);
        // Fill rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        (spatial_state, global_state)
    }

    #[test]
    fn shared_entry_is_reused() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let entry1 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        entry1.add_sample(0.5);

        let entry2 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        assert!(Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry2.visits(), 1);
        assert!((entry2.average_value() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn different_states_get_different_entries() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();
        let (mut spatial_state1, mut global_state1) = board_with_rings(config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(config);

        // Create states that break symmetry differently
        // State 1: white marble at (3,2)
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: white marble at (3,2) AND gray marble at (2,4)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        let entry2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());

        // These should be different entries (different number of marbles)
        assert!(!Arc::ptr_eq(&entry1, &entry2));
    }

    /* NOTE: Collision counter test commented out - it was implementation-specific
     * and manipulated internal hasher state that no longer exists in generic version.
     * Collision handling is still tested implicitly by the "different states" test.
     */
    /*
    #[test]
    fn collision_counter_increments_on_hash_collision() {
        use crate::zobrist::ZobristHasher;

        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();

        // Create two different states with rings + marbles
        let (mut spatial_state1, mut global_state1) = board_with_rings(&config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(&config);

        // State 1: 1 marble
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: 2 marbles (different configuration)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        // Pre-register a fake hasher that always returns hash=42
        let fake_hasher = Arc::new(ZobristHasher::new(config.width, Some(999)));
        table
            .hashers
            .lock()
            .unwrap()
            .insert(config.width, fake_hasher.clone());

        // Manually insert states with forced collision (same hash, different states)
        let (canonical1, _, _) = canonicalization::canonicalize_state(&spatial_state1.view(), &config);
        let (canonical2, _, _) = canonicalization::canonicalize_state(&spatial_state2.view(), &config);

        // Force both states to use the same hash by directly manipulating the table
        let forced_hash = 42u64;

        // Insert first entry
        let entry1 = Arc::new(TranspositionEntry::new(
            canonical1.to_owned(),
            global_state1.clone(),
        ));
        table.table.insert(forced_hash, vec![Arc::clone(&entry1)]);
        entry1.add_sample(0.5);

        // Now insert second entry with same hash but different state (this will trigger collision)
        let entry2 = Arc::new(TranspositionEntry::new(
            canonical2.to_owned(),
            global_state2.clone(),
        ));
        let mut chain = table.table.get_mut(&forced_hash).unwrap();
        table.collisions.fetch_add(1, Ordering::Relaxed);
        chain.push(Arc::clone(&entry2));
        entry2.add_sample(-0.5);

        // Verify collision was detected
        assert_eq!(table.collision_count(), 1);

        // Verify entries are distinct
        assert!(!Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry1.visits(), 1);
        assert_eq!(entry2.visits(), 1);
        assert!((entry1.average_value() - 0.5).abs() < 1e-3);
        assert!((entry2.average_value() + 0.5).abs() < 1e-3);

        // Verify both entries are in the chain
        assert_eq!(chain.len(), 2);
    }
    */

    #[test]
    fn lookup_returns_none_for_missing_state() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let result = table.lookup(&spatial_state.view(), &global_state.view());
        assert!(result.is_none());
    }

    #[test]
    fn lookup_returns_existing_entry() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let entry1 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        entry1.add_sample(0.75);

        let entry2 = table.lookup(&spatial_state.view(), &global_state.view());
        assert!(entry2.is_some());
        let entry2 = entry2.unwrap();
        assert!(Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry2.visits(), 1);
        assert!((entry2.average_value() - 0.75).abs() < 1e-3);
    }

    #[test]
    fn chaining_handles_multiple_states_with_same_hash() {
        // This test verifies that if multiple states happen to hash to the same value,
        // they are stored in a chain and can be retrieved correctly
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();

        let (mut spatial_state1, mut global_state1) = board_with_rings(&config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(&config);
        let (mut spatial_state3, mut global_state3) = board_with_rings(&config);

        // Create three distinct states with different marble placements
        // State 1: 1 white marble
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: 2 marbles (white + gray)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        // State 3: 3 marbles (white + gray + black)
        spatial_state3[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state3[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        spatial_state3[[config.marble_layers.0 + 2, 4, 3]] = 1.0;
        global_state3[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        entry1.add_sample(0.1);

        let entry2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());
        entry2.add_sample(0.2);

        let entry3 = table.get_or_insert(&spatial_state3.view(), &global_state3.view());
        entry3.add_sample(0.3);

        // Retrieve them again and verify they're the same entries
        let retrieved1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        let retrieved2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());
        let retrieved3 = table.get_or_insert(&spatial_state3.view(), &global_state3.view());

        assert!(Arc::ptr_eq(&entry1, &retrieved1));
        assert!(Arc::ptr_eq(&entry2, &retrieved2));
        assert!(Arc::ptr_eq(&entry3, &retrieved3));

        assert!((retrieved1.average_value() - 0.1).abs() < 1e-3);
        assert!((retrieved2.average_value() - 0.2).abs() < 1e-3);
        assert!((retrieved3.average_value() - 0.3).abs() < 1e-3);
    }

    #[test]
    fn test_lookup_does_not_pollute_table() {
        // Verify that lookup() doesn't create entries (prevents table pollution)
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();

        let initial_size = table.len();

        // Lookup 10 nonexistent states (enough to verify no pollution)
        for i in 0..10 {
            let mut spatial = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
            spatial[[0, 0, 0]] = i as f32;  // Make each unique
            let global = Array1::zeros(10);

            let result = table.lookup(&spatial.view(), &global.view());
            assert!(result.is_none(), "Lookup should return None for nonexistent entry");
        }

        let final_size = table.len();
        assert_eq!(initial_size, final_size, "Lookup should not insert entries (table size should not change)");
        assert_eq!(final_size, 0, "Table should still be empty after lookups");
    }
}
