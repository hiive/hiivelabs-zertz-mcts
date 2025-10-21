use ndarray::{Array3, Array1};
use std::sync::atomic::{AtomicU32, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Weak};
use crate::board::BoardConfig;
use crate::transposition::TranspositionEntry;

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

    // MCTS statistics (atomic for thread safety)
    pub visits: AtomicU32,
    pub total_value: AtomicI32,  // Scaled by 1000 for precision

    // Tree structure (Mutex allows thread-safe modification)
    pub children: Mutex<Vec<(Action, Arc<MCTSNode>)>>,
    pub parent: Option<Weak<MCTSNode>>,  // Weak pointer to avoid cycles

    // Config
    pub config: Arc<BoardConfig>,
}

impl MCTSNode {
    /// Create a new MCTS node
    pub fn new(
        spatial: Array3<f32>,
        global: Array1<f32>,
        config: Arc<BoardConfig>,
    ) -> Self {
        MCTSNode {
            spatial,
            global,
            visits: AtomicU32::new(0),
            total_value: AtomicI32::new(0),
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
    ) -> Self {
        MCTSNode {
            spatial,
            global,
            visits: AtomicU32::new(0),
            total_value: AtomicI32::new(0),
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
        self.visits.load(Ordering::Relaxed)
    }

    /// Get average value
    #[inline]
    pub fn get_value(&self) -> f32 {
        let visits = self.get_visits();
        if visits == 0 {
            0.0
        } else {
            let total = self.total_value.load(Ordering::Relaxed);
            (total as f32) / 1000.0 / (visits as f32)
        }
    }

    /// Update statistics (thread-safe)
    pub fn update(&self, value: f32) {
        self.visits.fetch_add(1, Ordering::Relaxed);
        let scaled_value = (value * 1000.0) as i32;
        self.total_value.fetch_add(scaled_value, Ordering::Relaxed);
    }

    /// Initialize statistics from a transposition-table entry
    pub fn apply_entry(&self, entry: &TranspositionEntry) {
        self.visits.store(entry.visits, Ordering::Relaxed);
        self.total_value.store(entry.total_value, Ordering::Relaxed);
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
        let exploration = exploration_constant *
            ((parent_visits as f32).ln() / (visits as f32)).sqrt();

        exploitation + exploration
    }

    /// Check if fully expanded (based on progressive widening)
    pub fn is_fully_expanded(
        &self,
        progressive_widening: bool,
        widening_constant: f32,
    ) -> bool {
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

        let (placement_mask, capture_mask) = get_valid_actions(
            &self.spatial.view(),
            &self.global.view(),
            &self.config,
        );

        let placement_count = placement_mask.iter().filter(|&&x| x > 0.0).count();
        let capture_count = capture_mask.iter().filter(|&&x| x > 0.0).count();

        let total = placement_count + capture_count;
        if total == 0 { 1 } else { total }  // At least 1 for PASS
    }
}
