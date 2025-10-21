
use dashmap::DashMap;
use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use ndarray::{ArrayView3, ArrayView1};

/// Hash a board state
pub fn hash_state(spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash spatial state
    for &value in spatial.iter() {
        // Convert f32 to bits for consistent hashing
        value.to_bits().hash(&mut hasher);
    }

    // Hash global state
    for &value in global.iter() {
        value.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

/// Transposition table statistics
#[derive(Clone, Debug)]
pub struct TranspositionEntry {
    pub visits: u32,
    pub total_value: i32,  // Scaled by 1000 for precision
}

impl TranspositionEntry {
    pub fn new(visits: u32, value: f32) -> Self {
        Self {
            visits,
            total_value: (value * 1000.0 * visits as f32) as i32,
        }
    }

    #[allow(dead_code)]
    pub fn get_average_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            (self.total_value as f32) / 1000.0 / (self.visits as f32)
        }
    }
}

/// Thread-safe transposition table using DashMap
pub struct TranspositionTable {
    table: Arc<DashMap<u64, TranspositionEntry>>,
}

impl TranspositionTable {
    /// Create a new transposition table
    pub fn new() -> Self {
        Self {
            table: Arc::new(DashMap::new()),
        }
    }

    /// Clone the table reference (shares underlying data)
    #[allow(dead_code)]
    pub fn clone_ref(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
        }
    }

    /// Store or update an entry
    pub fn store(
        &self,
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
        visits: u32,
        value: f32,
    ) {
        let hash = hash_state(spatial, global);
        let entry = TranspositionEntry::new(visits, value);
        self.table.insert(hash, entry);
    }

    /// Lookup an entry
    pub fn lookup(
        &self,
        spatial: &ArrayView3<f32>,
        global: &ArrayView1<f32>,
    ) -> Option<TranspositionEntry> {
        let hash = hash_state(spatial, global);
        self.table.get(&hash).map(|entry| entry.clone())
    }

    /// Clear the table
    pub fn clear(&self) {
        self.table.clear();
    }

    /// Get table size
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