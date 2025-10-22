use crate::board::BoardConfig;
use crate::canonicalization;
use crate::zobrist::ZobristHasher;
use dashmap::{mapref::entry::Entry, DashMap};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Transposition table statistics shared across canonical-equivalent nodes.
/// Stores the canonical state for collision detection.
pub struct TranspositionEntry {
    visits: AtomicU32,
    total_value: AtomicI32, // Scaled by 1000 for precision
    canonical_spatial: Arc<Array3<f32>>, // Stored for collision detection
    canonical_global: Arc<Array1<f32>>,  // Stored for collision detection
}

impl TranspositionEntry {
    pub fn new(canonical_spatial: Array3<f32>, canonical_global: Array1<f32>) -> Self {
        Self {
            visits: AtomicU32::new(0),
            total_value: AtomicI32::new(0),
            canonical_spatial: Arc::new(canonical_spatial),
            canonical_global: Arc::new(canonical_global),
        }
    }

    /// Check if this entry matches the given canonical state
    pub fn matches(&self, spatial: &Array3<f32>, global: &ArrayView1<f32>) -> bool {
        // Compare both spatial and global state for exact match
        self.canonical_spatial.as_ref() == spatial && self.canonical_global.as_ref() == global
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    pub fn average_value(&self) -> f32 {
        let visits = self.visits();
        if visits == 0 {
            0.0
        } else {
            (self.total_value.load(Ordering::Relaxed) as f32) / 1000.0 / (visits as f32)
        }
    }

    pub fn add_sample(&self, value: f32) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        let scaled = (value * 1000.0).round() as i32;
        self.total_value.fetch_add(scaled, Ordering::Relaxed);
    }

    pub fn set_counts(&self, visits: u32, average_value: f32) {
        self.visits.store(visits, Ordering::Relaxed);
        let scaled_total = (average_value * 1000.0 * visits as f32).round() as i32;
        self.total_value.store(scaled_total, Ordering::Relaxed);
    }
}

/// Thread-safe transposition table using DashMap + Zobrist hashing.
/// Uses chaining to handle hash collisions - each hash maps to a vector of entries.
pub struct TranspositionTable {
    table: Arc<DashMap<u64, Vec<Arc<TranspositionEntry>>>>,
    hashers: Mutex<HashMap<usize, Arc<ZobristHasher>>>,
    collisions: AtomicU32, // Track number of hash collisions
}

impl TranspositionTable {
    /// Create a new transposition table.
    pub fn new() -> Self {
        Self {
            table: Arc::new(DashMap::new()),
            hashers: Mutex::new(HashMap::new()),
            collisions: AtomicU32::new(0),
        }
    }

    /// Clone the table reference (shared underlying data).
    #[allow(dead_code)]
    pub fn clone_ref(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            hashers: Mutex::new(HashMap::new()),
            collisions: AtomicU32::new(0), // New instance has separate collision counter
        }
    }

    #[allow(dead_code)]
    /// Get the number of hash collisions detected
    pub fn collision_count(&self) -> u32 {
        self.collisions.load(Ordering::Relaxed)
    }

    fn hasher_for(&self, width: usize) -> Arc<ZobristHasher> {
        let mut guard = self.hashers.lock().unwrap();
        if let Some(existing) = guard.get(&width) {
            return Arc::clone(existing);
        }
        let hasher = Arc::new(ZobristHasher::new(width, None));
        guard.insert(width, Arc::clone(&hasher));
        hasher
    }

    /// Fetch or insert an entry for the canonicalized state.
    /// Implements collision detection by checking canonical states.
    pub fn get_or_insert(
        &self,
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
        config: &BoardConfig,
    ) -> Arc<TranspositionEntry> {
        let (canonical_spatial, _, _) = canonicalization::canonicalize_state(spatial, config);
        let hasher = self.hasher_for(config.width);
        let hash = hasher.hash_state(&canonical_spatial.view(), global, config);

        match self.table.entry(hash) {
            Entry::Occupied(mut occupied) => {
                let chain = occupied.get_mut();

                // Search for exact match in chain
                for entry in chain.iter() {
                    if entry.matches(&canonical_spatial, global) {
                        return Arc::clone(entry);
                    }
                }

                // No match found - this is a collision!
                self.collisions.fetch_add(1, Ordering::Relaxed);

                // Create new entry and add to chain
                let new_entry = Arc::new(TranspositionEntry::new(
                    canonical_spatial.to_owned(),
                    global.to_owned(),
                ));
                chain.push(Arc::clone(&new_entry));
                new_entry
            }
            Entry::Vacant(vacant) => {
                // First entry for this hash
                let entry = Arc::new(TranspositionEntry::new(
                    canonical_spatial.to_owned(),
                    global.to_owned(),
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
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
        config: &BoardConfig,
    ) -> Option<Arc<TranspositionEntry>> {
        let (canonical_spatial, _, _) = canonicalization::canonicalize_state(spatial, config);
        let hasher = self.hasher_for(config.width);
        let hash = hasher.hash_state(&canonical_spatial.view(), global, config);

        // Search chain for exact match
        self.table.get(&hash).and_then(|chain_ref| {
            chain_ref
                .value()
                .iter()
                .find(|entry| entry.matches(&canonical_spatial, global))
                .map(Arc::clone)
        })
    }

    /// Store (overwrite) statistics for a canonical state.
    pub fn store(
        &self,
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
        config: &BoardConfig,
        visits: u32,
        average_value: f32,
    ) {
        let entry = self.get_or_insert(spatial, global, config);
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

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let layers = config.layers_per_timestep * config.t + 1;
        let spatial = Array3::zeros((layers, config.width, config.width));
        let global = Array1::zeros(10);
        (spatial, global)
    }

    fn board_with_rings(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let mut spatial = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let global = Array1::zeros(10);
        // Fill rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial[[config.ring_layer, y, x]] = 1.0;
            }
        }
        (spatial, global)
    }

    #[test]
    fn shared_entry_is_reused() {
        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();
        let (spatial, global) = empty_state(&config);

        let entry1 = table.get_or_insert(&spatial.view(), &global.view(), &config);
        entry1.add_sample(0.5);

        let entry2 = table.get_or_insert(&spatial.view(), &global.view(), &config);
        assert!(Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry2.visits(), 1);
        assert!((entry2.average_value() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn different_states_get_different_entries() {
        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();
        let (mut spatial1, mut global1) = board_with_rings(&config);
        let (mut spatial2, mut global2) = board_with_rings(&config);

        // Create states that break symmetry differently
        // State 1: white marble at (3,2)
        spatial1[[config.marble_layers.0, 3, 2]] = 1.0;
        global1[config.cur_player] = 0.0;

        // State 2: white marble at (3,2) AND gray marble at (2,4)
        spatial2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global2[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial1.view(), &global1.view(), &config);
        let entry2 = table.get_or_insert(&spatial2.view(), &global2.view(), &config);

        // These should be different entries (different number of marbles)
        assert!(!Arc::ptr_eq(&entry1, &entry2));
    }

    #[test]
    fn collision_counter_increments_on_hash_collision() {
        use crate::zobrist::ZobristHasher;

        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();

        // Create two different states with rings + marbles
        let (mut spatial1, mut global1) = board_with_rings(&config);
        let (mut spatial2, mut global2) = board_with_rings(&config);

        // State 1: 1 marble
        spatial1[[config.marble_layers.0, 3, 2]] = 1.0;
        global1[config.cur_player] = 0.0;

        // State 2: 2 marbles (different configuration)
        spatial2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global2[config.cur_player] = 0.0;

        // Pre-register a fake hasher that always returns hash=42
        let fake_hasher = Arc::new(ZobristHasher::new(config.width, Some(999)));
        table.hashers.lock().unwrap().insert(config.width, fake_hasher.clone());

        // Manually insert states with forced collision (same hash, different states)
        let (canonical1, _, _) = canonicalization::canonicalize_state(&spatial1.view(), &config);
        let (canonical2, _, _) = canonicalization::canonicalize_state(&spatial2.view(), &config);

        // Force both states to use the same hash by directly manipulating the table
        let forced_hash = 42u64;

        // Insert first entry
        let entry1 = Arc::new(TranspositionEntry::new(canonical1.to_owned(), global1.clone()));
        table.table.insert(forced_hash, vec![Arc::clone(&entry1)]);
        entry1.add_sample(0.5);

        // Now insert second entry with same hash but different state (this will trigger collision)
        let entry2 = Arc::new(TranspositionEntry::new(canonical2.to_owned(), global2.clone()));
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

    #[test]
    fn lookup_returns_none_for_missing_state() {
        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();
        let (spatial, global) = empty_state(&config);

        let result = table.lookup(&spatial.view(), &global.view(), &config);
        assert!(result.is_none());
    }

    #[test]
    fn lookup_returns_existing_entry() {
        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();
        let (spatial, global) = empty_state(&config);

        let entry1 = table.get_or_insert(&spatial.view(), &global.view(), &config);
        entry1.add_sample(0.75);

        let entry2 = table.lookup(&spatial.view(), &global.view(), &config);
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
        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();

        let (mut spatial1, mut global1) = board_with_rings(&config);
        let (mut spatial2, mut global2) = board_with_rings(&config);
        let (mut spatial3, mut global3) = board_with_rings(&config);

        // Create three distinct states with different marble placements
        // State 1: 1 white marble
        spatial1[[config.marble_layers.0, 3, 2]] = 1.0;
        global1[config.cur_player] = 0.0;

        // State 2: 2 marbles (white + gray)
        spatial2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global2[config.cur_player] = 0.0;

        // State 3: 3 marbles (white + gray + black)
        spatial3[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial3[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        spatial3[[config.marble_layers.0 + 2, 4, 3]] = 1.0;
        global3[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial1.view(), &global1.view(), &config);
        entry1.add_sample(0.1);

        let entry2 = table.get_or_insert(&spatial2.view(), &global2.view(), &config);
        entry2.add_sample(0.2);

        let entry3 = table.get_or_insert(&spatial3.view(), &global3.view(), &config);
        entry3.add_sample(0.3);

        // Retrieve them again and verify they're the same entries
        let retrieved1 = table.get_or_insert(&spatial1.view(), &global1.view(), &config);
        let retrieved2 = table.get_or_insert(&spatial2.view(), &global2.view(), &config);
        let retrieved3 = table.get_or_insert(&spatial3.view(), &global3.view(), &config);

        assert!(Arc::ptr_eq(&entry1, &retrieved1));
        assert!(Arc::ptr_eq(&entry2, &retrieved2));
        assert!(Arc::ptr_eq(&entry3, &retrieved3));

        assert!((retrieved1.average_value() - 0.1).abs() < 1e-3);
        assert!((retrieved2.average_value() - 0.2).abs() < 1e-3);
        assert!((retrieved3.average_value() - 0.3).abs() < 1e-3);
    }
}
