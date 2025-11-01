//! # MCTS Node Implementation
//!
//! Thread-safe MCTS node with support for:
//! - **Atomic statistics**: Lock-free visit/value updates via `AtomicU32`/`AtomicI32`
//! - **Virtual loss**: Temporary pessimistic values for parallel search
//! - **Transposition table integration**: Shared statistics across symmetric states
//! - **Weak parent pointers**: Avoids reference cycles in tree structure
//!
//! ## Thread Safety Model
//!
//! **Lock-free statistics updates**:
//! - `local_visits` and `local_total_value` use atomic operations
//! - Multiple threads can update node statistics concurrently without locks
//! - `Ordering::Relaxed` is sufficient since we don't need sequential consistency
//!
//! **Tree structure locking**:
//! - `children` uses `Mutex` since tree modification requires atomicity
//! - Lock is held only during child addition, not during traversal
//!
//! **Parent pointers**:
//! - Use `Weak<MCTSNode>` to avoid reference cycles
//! - Prevents memory leaks in the tree structure
//! - Can upgrade to `Arc` during backpropagation traversal
//!
//! ## Virtual Loss Mechanism
//!
//! Virtual loss prevents thread collisions in parallel MCTS:
//! 1. During selection: `add_virtual_loss()` inflates visits and adds pessimistic value
//! 2. Makes path less attractive to other threads
//! 3. During backpropagation: `remove_virtual_loss()` removes temporary values
//! 4. Then real simulation result is added via `update()`
//!
//! This enables lock-free parallel tree search with minimal thread collisions.

use crate::game_trait::MCTSGame;
use crate::transposition::TranspositionEntry;
use ndarray::{Array1, Array3};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, RwLock, Weak};

// ============================================================================
// VIRTUAL LOSS CONSTANTS
// ============================================================================

/// Number of virtual visits to add during selection phase
///
/// Virtual loss temporarily inflates visit counts and adds pessimistic values
/// to discourage multiple threads from exploring the same path simultaneously.
/// Higher values = stronger discouragement, but may reduce parallelism.
pub(crate) const VIRTUAL_LOSS: u32 = 3;

/// Scaled virtual loss value (represents -1.0 per virtual visit, scaled by 1000)
///
/// We scale by 1000 because `local_total_value` is stored as i32 for atomic operations.
/// -3000 represents a pessimistic value of -3.0 (same as VIRTUAL_LOSS * -1.0 * 1000).
pub(crate) const VIRTUAL_LOSS_SCALED: i32 = -3000;

// ============================================================================
// MCTS NODE
// ============================================================================

/// MCTS Node with atomic counters and Mutex for thread-safe tree modification
pub struct MCTSNode<G: MCTSGame> {
    // State representation
    pub spatial_state: Array3<f32>,
    pub global_state: Array1<f32>,

    // Shared statistics (optional)
    shared_stats: Option<Arc<TranspositionEntry>>,
    // Local fallback statistics when no shared entry is available
    local_visits: AtomicU32,
    local_total_value: AtomicI32, // Scaled by 1000 for precision

    // RAVE (Rapid Action Value Estimation) statistics
    // Tracks "all-moves-as-first" (AMAF) - value when this action appears anywhere in playout
    rave_visits: AtomicU32,
    rave_total_value: AtomicI32, // Scaled by 1000 for precision

    // Debug-only: Track virtual loss count to catch mismatched add/remove
    #[cfg(debug_assertions)]
    pub(crate) virtual_loss_count: AtomicU32,

    // Tree structure (RwLock allows concurrent reads, exclusive writes)
    // Changed from Vec to HashMap for O(1) action lookup
    pub children: RwLock<HashMap<G::Action, Arc<MCTSNode<G>>>>,
    pub parent: Option<Weak<MCTSNode<G>>>, // Weak pointer to avoid cycles

    // Game instance (stores config internally)
    pub game: Arc<G>,
}

impl<G: MCTSGame> MCTSNode<G> {
    /// Create a new MCTS node
    pub fn new(
        spatial_state: Array3<f32>,
        global_state: Array1<f32>,
        game: Arc<G>,
        shared_stats: Option<Arc<TranspositionEntry>>,
    ) -> Self {
        MCTSNode {
            spatial_state,
            global_state,
            shared_stats,
            local_visits: AtomicU32::new(0),
            local_total_value: AtomicI32::new(0),
            rave_visits: AtomicU32::new(0),
            rave_total_value: AtomicI32::new(0),
            #[cfg(debug_assertions)]
            virtual_loss_count: AtomicU32::new(0),
            children: RwLock::new(HashMap::new()),
            parent: None,
            game,
        }
    }

    /// Create a child node with parent pointer
    pub fn new_child(
        spatial_state: Array3<f32>,
        global_state: Array1<f32>,
        game: Arc<G>,
        parent: &Arc<MCTSNode<G>>,
        shared_stats: Option<Arc<TranspositionEntry>>,
    ) -> Self {
        MCTSNode {
            spatial_state,
            global_state,
            shared_stats,
            local_visits: AtomicU32::new(0),
            local_total_value: AtomicI32::new(0),
            rave_visits: AtomicU32::new(0),
            rave_total_value: AtomicI32::new(0),
            #[cfg(debug_assertions)]
            virtual_loss_count: AtomicU32::new(0),
            children: RwLock::new(HashMap::new()),
            parent: Some(Arc::downgrade(parent)),
            game,
        }
    }

    /// Add a child to this node (thread-safe, requires exclusive write lock)
    pub fn add_child(&self, action: G::Action, child: Arc<MCTSNode<G>>) {
        self.children.write().unwrap().insert(action, child);
    }

    /// Get number of children (thread-safe)
    #[inline]
    #[allow(dead_code)]
    pub fn children_count(&self) -> usize {
        self.children.read().unwrap().len()
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

    /// Get RAVE visit count
    #[inline]
    pub fn get_rave_visits(&self) -> u32 {
        self.rave_visits.load(Ordering::Relaxed)
    }

    /// Get RAVE average value
    #[inline]
    pub fn get_rave_value(&self) -> f32 {
        let visits = self.rave_visits.load(Ordering::Relaxed);
        if visits == 0 {
            0.0
        } else {
            let total = self.rave_total_value.load(Ordering::Relaxed);
            (total as f32) / 1000.0 / (visits as f32)
        }
    }

    /// Update RAVE statistics (thread-safe)
    ///
    /// Called when this action appears anywhere in the playout, not just as first move.
    /// This provides "all-moves-as-first" (AMAF) statistics for better value estimates.
    pub fn update_rave(&self, value: f32) {
        self.rave_visits.fetch_add(1, Ordering::Relaxed);
        let scaled_value = (value * 1000.0) as i32;
        self.rave_total_value
            .fetch_add(scaled_value, Ordering::Relaxed);
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

    /// Calculate UCB1 score with optional FPU (First Play Urgency)
    ///
    /// For unvisited nodes (visits == 0):
    ///   - If fpu_reduction is Some: estimated_q = -(parent_value - fpu_reduction), u = c * sqrt(parent_visits)
    ///   - If fpu_reduction is None: returns f32::INFINITY (standard behavior)
    ///
    /// For visited nodes, uses standard UCB1:
    ///   q = -child.value (negated for parent's perspective)
    ///   u = c * sqrt(ln(parent_visits) / visits)
    ///
    /// Note: child.value is from child's player perspective (due to value flipping in backprop).
    /// We negate to convert to parent's player perspective (opponent of child's player).
    pub fn ucb1_score(&self, parent_visits: u32, parent_value: f32, exploration_constant: f32, fpu_reduction: Option<f32>) -> f32 {
        let visits = self.get_visits();
        if visits == 0 {
            if let Some(reduction) = fpu_reduction {
                // FPU: estimate using parent value with reduction
                // Parent's value from parent's perspective, child will have opposite sign
                let estimated_q = -(parent_value - reduction);
                let exploration = exploration_constant * (parent_visits as f32).sqrt();
                return estimated_q + exploration;
            } else {
                // Standard behavior: unvisited nodes have infinite urgency
                return f32::INFINITY;
            }
        }

        // Standard UCB1 for visited nodes
        // Negate value to convert from child's perspective to parent's perspective
        let exploitation = -self.get_value();
        let exploration =
            exploration_constant * ((parent_visits as f32).ln() / (visits as f32)).sqrt();

        exploitation + exploration
    }

    /// RAVE-UCB scoring function (optional RAVE mixing)
    ///
    /// When rave_constant is None, falls back to standard UCB1.
    /// When rave_constant is Some(k), mixes UCB1 with RAVE statistics using:
    ///   β = sqrt(k / (3 * parent_visits + k))
    ///   Q = (1-β) × Q_ucb + β × Q_rave
    ///
    /// Args:
    ///     parent_visits: Parent node's visit count
    ///     parent_value: Parent node's value (for FPU estimation)
    ///     exploration_constant: UCB exploration constant (typically sqrt(2))
    ///     rave_constant: Optional RAVE mixing constant (typically 300-3000)
    ///     fpu_reduction: Optional FPU reduction factor for unvisited nodes
    ///
    /// Returns: Score for node selection (higher = more promising)
    pub fn rave_ucb_score(
        &self,
        parent_visits: u32,
        parent_value: f32,
        exploration_constant: f32,
        rave_constant: Option<f32>,
        fpu_reduction: Option<f32>,
    ) -> f32 {
        // If RAVE is disabled, use standard UCB1
        let rave_k = match rave_constant {
            None => return self.ucb1_score(parent_visits, parent_value, exploration_constant, fpu_reduction),
            Some(k) => k,
        };

        let visits = self.get_visits();

        // Handle unvisited nodes (same FPU logic as UCB1)
        if visits == 0 {
            if let Some(reduction) = fpu_reduction {
                let estimated_q = -(parent_value - reduction);
                let exploration = exploration_constant * (parent_visits as f32).sqrt();
                return estimated_q + exploration;
            } else {
                return f32::INFINITY;
            }
        }

        // Compute mixing weight β = sqrt(rave_k / (3 * parent_visits + rave_k))
        let beta = (rave_k / (3.0 * parent_visits as f32 + rave_k)).sqrt();

        // Get UCB1 value (exploitation only, we'll add exploration once)
        let q_ucb = -self.get_value(); // Negate to convert from child to parent perspective

        // Get RAVE value (also from child's perspective, so negate)
        let rave_visits = self.get_rave_visits();
        let q_rave = if rave_visits > 0 {
            -self.get_rave_value()
        } else {
            // No RAVE data yet, fall back to UCB value
            q_ucb
        };

        // Mix UCB and RAVE: Q = (1-β) × Q_ucb + β × Q_rave
        let mixed_q = (1.0 - beta) * q_ucb + beta * q_rave;

        // Add exploration term (same as UCB1)
        let exploration =
            exploration_constant * ((parent_visits as f32).ln() / (visits as f32)).sqrt();

        mixed_q + exploration
    }

    pub fn has_shared_stats(&self) -> bool {
        self.shared_stats.is_some()
    }

    /// Check if fully expanded (based on progressive widening)
    ///
    /// Args:
    ///     widening_constant: If None, standard MCTS (expand all children).
    ///                       If Some(k), progressive widening with constant k.
    pub fn is_fully_expanded(&self, widening_constant: Option<f32>) -> bool {
        let children_count = self.children.read().unwrap().len();
        let legal_actions = self.count_legal_actions();

        match widening_constant {
            None => {
                // Standard MCTS: expand until all actions tried
                children_count >= legal_actions
            }
            Some(constant) => {
                // Progressive widening: allow sqrt(visits + 1) * constant children
                // The +1 ensures nodes start with some children even at 0 visits
                let max_children = (constant * ((self.get_visits() + 1) as f32).sqrt()) as usize;
                let max_children = max_children.min(legal_actions);
                children_count >= max_children
            }
        }
    }

    /// Count legal actions using game trait
    fn count_legal_actions(&self) -> usize {
        let actions = self.game.get_valid_actions(
            &self.spatial_state.view(),
            &self.global_state.view(),
        );
        actions.len()
    }
}

