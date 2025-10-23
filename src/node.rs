use crate::board::BoardConfig;
use crate::transposition::TranspositionEntry;
use ndarray::{Array1, Array3};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

// Virtual loss constants for parallel MCTS
// Virtual loss temporarily inflates visit counts and adds pessimistic values
// to discourage multiple threads from exploring the same path simultaneously
pub(crate) const VIRTUAL_LOSS: u32 = 3;
pub(crate) const VIRTUAL_LOSS_SCALED: i32 = -3000; // -3.0 * 1000 (pessimistic value, scaled)

/// Action type for MCTS
#[derive(Clone, Debug)]
pub enum Action {
    Placement {
        marble_type: usize,
        dst_y: usize,
        dst_x: usize,
        remove_y: Option<usize>,
        remove_x: Option<usize>,
    },
    Capture {
        start_y: usize,
        start_x: usize,
        direction: usize,
    },
    Pass,
}

/// MCTS Node with atomic counters and Mutex for thread-safe tree modification
pub struct MCTSNode {
    // State representation
    pub spatial: Array3<f32>,
    pub global: Array1<f32>,

    // Shared statistics (optional)
    shared_stats: Option<Arc<TranspositionEntry>>,
    // Local fallback statistics when no shared entry is available
    local_visits: AtomicU32,
    local_total_value: AtomicI32, // Scaled by 1000 for precision

    // Debug-only: Track virtual loss count to catch mismatched add/remove
    #[cfg(debug_assertions)]
    pub(crate) virtual_loss_count: AtomicU32,

    // Tree structure (Mutex allows thread-safe modification)
    pub children: Mutex<Vec<(Action, Arc<MCTSNode>)>>,
    pub parent: Option<Weak<MCTSNode>>, // Weak pointer to avoid cycles

    // Config
    pub config: Arc<BoardConfig>,
}

impl MCTSNode {
    /// Create a new MCTS node
    pub fn new(
        spatial: Array3<f32>,
        global: Array1<f32>,
        config: Arc<BoardConfig>,
        shared_stats: Option<Arc<TranspositionEntry>>,
    ) -> Self {
        MCTSNode {
            spatial,
            global,
            shared_stats,
            local_visits: AtomicU32::new(0),
            local_total_value: AtomicI32::new(0),
            #[cfg(debug_assertions)]
            virtual_loss_count: AtomicU32::new(0),
            children: Mutex::new(Vec::new()),
            parent: None,
            config,
        }
    }

    /// Create a child node with parent pointer
    pub fn new_child(
        spatial: Array3<f32>,
        global: Array1<f32>,
        config: Arc<BoardConfig>,
        parent: &Arc<MCTSNode>,
        shared_stats: Option<Arc<TranspositionEntry>>,
    ) -> Self {
        MCTSNode {
            spatial,
            global,
            shared_stats,
            local_visits: AtomicU32::new(0),
            local_total_value: AtomicI32::new(0),
            #[cfg(debug_assertions)]
            virtual_loss_count: AtomicU32::new(0),
            children: Mutex::new(Vec::new()),
            parent: Some(Arc::downgrade(parent)),
            config,
        }
    }

    /// Add a child to this node (thread-safe)
    pub fn add_child(&self, action: Action, child: Arc<MCTSNode>) {
        self.children.lock().unwrap().push((action, child));
    }

    /// Get visit count
    #[inline]
    pub fn get_visits(&self) -> u32 {
        if let Some(stats) = &self.shared_stats {
            stats.visits()
        } else {
            self.local_visits.load(Ordering::Relaxed)
        }
    }

    /// Get average value per visit
    ///
    /// Note: This returns the AVERAGE value, equivalent to total_value / visits.
    /// Python's `value` property returns the TOTAL, so Python computes UCB1 as
    /// `-(child.value / child.visits)` while Rust computes `- self.get_value()`.
    /// Both are mathematically equivalent.
    #[inline]
    pub fn get_value(&self) -> f32 {
        if let Some(stats) = &self.shared_stats {
            stats.average_value()
        } else {
            let visits = self.local_visits.load(Ordering::Relaxed);
            if visits == 0 {
                0.0
            } else {
                let total = self.local_total_value.load(Ordering::Relaxed);
                (total as f32) / 1000.0 / (visits as f32)
            }
        }
    }

    /// Get total accumulated value (for semantic parity with Python)
    ///
    /// Python stores total value and divides by visits in UCB1.
    /// Rust pre-computes the average but provides this for compatibility.
    #[allow(dead_code)]
    #[inline]
    pub fn get_total_value(&self) -> f32 {
        if let Some(stats) = &self.shared_stats {
            // TranspositionEntry stores average, multiply back by visits
            stats.average_value() * stats.visits() as f32
        } else {
            let total = self.local_total_value.load(Ordering::Relaxed);
            (total as f32) / 1000.0
        }
    }

    /// Update statistics (thread-safe)
    pub fn update(&self, value: f32) {
        if let Some(stats) = &self.shared_stats {
            stats.add_sample(value);
        } else {
            self.local_visits.fetch_add(1, Ordering::Relaxed);
            let scaled_value = (value * 1000.0) as i32;
            self.local_total_value
                .fetch_add(scaled_value, Ordering::Relaxed);
        }
    }

    /// Add virtual loss to discourage thread collision
    ///
    /// Virtual loss temporarily inflates visit count and adds pessimistic value
    /// to make this path less attractive to other threads during parallel search.
    /// Must be paired with remove_virtual_loss() after backpropagation.
    #[inline]
    pub fn add_virtual_loss(&self) {
        #[cfg(debug_assertions)]
        self.virtual_loss_count.fetch_add(1, Ordering::Relaxed);

        if let Some(stats) = &self.shared_stats {
            stats.add_virtual_loss();
        } else {
            self.local_visits.fetch_add(VIRTUAL_LOSS, Ordering::Relaxed);
            self.local_total_value
                .fetch_add(VIRTUAL_LOSS_SCALED, Ordering::Relaxed);
        }
    }

    /// Remove virtual loss after backpropagation
    ///
    /// Removes the temporary inflation added by add_virtual_loss().
    /// Should be called before adding the real simulation value.
    #[inline]
    pub fn remove_virtual_loss(&self) {
        #[cfg(debug_assertions)]
        {
            let count = self.virtual_loss_count.fetch_sub(1, Ordering::Relaxed);
            debug_assert!(
                count > 0,
                "remove_virtual_loss() called but virtual_loss_count was 0! \
                 This indicates a bug: remove called without matching add."
            );
        }

        if let Some(stats) = &self.shared_stats {
            stats.remove_virtual_loss();
        } else {
            self.local_visits.fetch_sub(VIRTUAL_LOSS, Ordering::Relaxed);
            self.local_total_value
                .fetch_sub(VIRTUAL_LOSS_SCALED, Ordering::Relaxed);
        }
    }

    /// Calculate UCB1 score
    ///
    /// Note: child.value is from child's player perspective (due to value flipping in backprop).
    /// We negate to convert to parent's player perspective (opponent of child's player).
    pub fn ucb1_score(&self, parent_visits: u32, exploration_constant: f32) -> f32 {
        let visits = self.get_visits();
        if visits == 0 {
            return f32::INFINITY;
        }

        // Negate value to convert from child's perspective to parent's perspective
        let exploitation = -self.get_value();
        let exploration =
            exploration_constant * ((parent_visits as f32).ln() / (visits as f32)).sqrt();

        exploitation + exploration
    }

    pub fn has_shared_stats(&self) -> bool {
        self.shared_stats.is_some()
    }

    /// Check if fully expanded (based on progressive widening)
    pub fn is_fully_expanded(&self, progressive_widening: bool, widening_constant: f32) -> bool {
        let children_count = self.children.lock().unwrap().len();
        let legal_actions = self.count_legal_actions();

        if !progressive_widening {
            return children_count == legal_actions;
        }

        // Progressive widening: allow sqrt(visits + 1) * constant children
        // The +1 ensures nodes start with some children even at 0 visits
        let max_children = (widening_constant * ((self.get_visits() + 1) as f32).sqrt()) as usize;
        let max_children = max_children.min(legal_actions);

        children_count >= max_children
    }

    /// Count legal actions
    fn count_legal_actions(&self) -> usize {
        use crate::game::get_valid_actions;

        let (placement_mask, capture_mask) =
            get_valid_actions(&self.spatial.view(), &self.global.view(), &self.config);

        let placement_count = placement_mask.iter().filter(|&&x| x > 0.0).count();
        let capture_count = capture_mask.iter().filter(|&&x| x > 0.0).count();

        let total = placement_count + capture_count;
        if total == 0 {
            1
        } else {
            total
        } // At least 1 for PASS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardConfig;
    use crate::transposition::TranspositionTable;
    use ndarray::{Array1, Array3};

    fn empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let layers = config.layers_per_timestep * config.t + 1;
        let spatial = Array3::zeros((layers, config.width, config.width));
        let global = Array1::zeros(10);
        (spatial, global)
    }

    #[test]
    fn node_updates_shared_entry() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = empty_state(&config);
        let table = TranspositionTable::new();
        let shared = table.get_or_insert(&spatial.view(), &global.view(), config.as_ref());

        let node = MCTSNode::new(
            spatial,
            global,
            Arc::clone(&config),
            Some(Arc::clone(&shared)),
        );
        node.update(0.5);

        assert_eq!(shared.visits(), 1);
        assert!((shared.average_value() - 0.5).abs() < 1e-3);
        assert_eq!(node.get_visits(), 1);
    }

    #[test]
    fn node_without_shared_entry_uses_local_stats() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = empty_state(&config);
        let node = MCTSNode::new(spatial, global, Arc::clone(&config), None);

        node.update(-1.0);

        assert_eq!(node.get_visits(), 1);
        assert!((node.get_value() + 1.0).abs() < 1e-3);
    }

    fn canonical_variant_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let mut spatial = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global = Array1::zeros(10);
        // fill rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial[[config.ring_layer, y, x]] = 1.0;
            }
        }
        // place a couple of marbles to break symmetry
        spatial[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global[config.cur_player] = 0.0;
        (spatial, global)
    }

    #[test]
    fn canonical_symmetric_nodes_share_stats() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = canonical_variant_state(&config);

        // Prepare table and canonical entry
        let table = TranspositionTable::new();
        let entry = table.get_or_insert(&spatial.view(), &global.view(), config.as_ref());

        let node_canonical = Arc::new(MCTSNode::new(
            spatial.clone(),
            global.clone(),
            Arc::clone(&config),
            Some(Arc::clone(&entry)),
        ));
        node_canonical.update(0.75);

        // Rotate spatial state by 60 degrees (same canonical class)
        let rotated =
            crate::canonicalization::transform_state(&spatial.view(), &config, 1, false, false);
        let rotated_entry = table.get_or_insert(&rotated.view(), &global.view(), config.as_ref());
        let node_rotated = Arc::new(MCTSNode::new(
            rotated.to_owned(),
            global.clone(),
            Arc::clone(&config),
            Some(Arc::clone(&rotated_entry)),
        ));
        node_rotated.update(-0.25);

        // Both nodes should refer to the same transposition entry
        assert!(Arc::ptr_eq(&entry, &rotated_entry));
        assert_eq!(entry.visits(), 2);
        assert!((entry.average_value() - 0.25).abs() < 1e-6);

        // Node getters should reflect shared stats
        assert_eq!(node_canonical.get_visits(), 2);
        assert_eq!(node_rotated.get_visits(), 2);
        assert!((node_canonical.get_value() - 0.25).abs() < 1e-6);
        assert!((node_rotated.get_value() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn virtual_loss_adds_and_removes_correctly() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = empty_state(&config);
        let node = MCTSNode::new(spatial, global, Arc::clone(&config), None);

        // Initial state: 0 visits, 0 value
        assert_eq!(node.get_visits(), 0);
        assert_eq!(node.get_value(), 0.0);

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(node.get_visits(), VIRTUAL_LOSS);
        assert!((node.get_value() + 1.0).abs() < 1e-3); // Should be -1.0 (pessimistic)

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(node.get_visits(), 0);
        assert_eq!(node.get_value(), 0.0);
    }

    #[test]
    fn virtual_loss_with_real_updates() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = empty_state(&config);
        let node = MCTSNode::new(spatial, global, Arc::clone(&config), None);

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(node.get_visits(), VIRTUAL_LOSS);

        // Add real update
        node.update(0.5);
        assert_eq!(node.get_visits(), VIRTUAL_LOSS + 1);

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(node.get_visits(), 1);
        assert!((node.get_value() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn virtual_loss_with_shared_stats() {
        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let (spatial, global) = empty_state(&config);
        let table = TranspositionTable::new();
        let shared = table.get_or_insert(&spatial.view(), &global.view(), config.as_ref());

        let node = MCTSNode::new(
            spatial,
            global,
            Arc::clone(&config),
            Some(Arc::clone(&shared)),
        );

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(shared.visits(), VIRTUAL_LOSS);
        assert!((shared.average_value() + 1.0).abs() < 1e-3);

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(shared.visits(), 0);
        assert_eq!(shared.average_value(), 0.0);
    }
}
