use crate::board::BoardConfig;
use crate::canonicalization;
use crate::zobrist::ZobristHasher;
use dashmap::{mapref::entry::Entry, DashMap};
use ndarray::{Array1, Array3, ArrayView1, ArrayView3};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Transposition table statistics shared across canonical-equivalent nodes.
pub struct TranspositionEntry {
    visits: AtomicU32,
    total_value: AtomicI32, // Scaled by 1000 for precision
}

impl TranspositionEntry {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            total_value: AtomicI32::new(0),
        }
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
pub struct TranspositionTable {
    table: Arc<DashMap<u64, Arc<TranspositionEntry>>>,
    hashers: Mutex<HashMap<usize, Arc<ZobristHasher>>>,
}

impl TranspositionTable {
    /// Create a new transposition table.
    pub fn new() -> Self {
        Self {
            table: Arc::new(DashMap::new()),
            hashers: Mutex::new(HashMap::new()),
        }
    }

    /// Clone the table reference (shared underlying data).
    #[allow(dead_code)]
    pub fn clone_ref(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            hashers: Mutex::new(HashMap::new()),
        }
    }

    fn hasher_for(&self, width: usize) -> Arc<ZobristHasher> {
        let mut guard = self.hashers.lock().unwrap();
        if let Some(existing) = guard.get(&width) {
            return Arc::clone(existing);
        }
        let hasher = Arc::new(ZobristHasher::new(width, 42));
        guard.insert(width, Arc::clone(&hasher));
        hasher
    }

    /// Fetch or insert an entry for the canonicalized state.
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
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(vacant) => {
                let entry = Arc::new(TranspositionEntry::new());
                vacant.insert(Arc::clone(&entry));
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
        self.table.get(&hash).map(|entry| Arc::clone(entry.value()))
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
}
