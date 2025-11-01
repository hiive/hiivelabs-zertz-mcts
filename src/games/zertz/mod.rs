//! # Zertz Game Implementation
//!
//! Zertz is a GIPF project game featuring:
//! - Hexagonal board with shrinking play area
//! - Three types of marbles (white, gray, black)
//! - Capture mechanics (jump captures and isolation captures)
//! - Win condition: collect required marbles or eliminate opponent
//!
//! This module contains all Zertz-specific code organized into submodules:
//! - `board`: Board configuration, game modes, win conditions
//! - `canonicalization`: State canonicalization and symmetry detection
//! - `logic`: Core game rules (placement, capture, win conditions)
//! - `action_transform`: Action transformation for testing symmetry operations
//! - `zobrist`: Zobrist hashing for fast state hashing
//! - `py_logic`: Python bindings for game logic functions
//! - `py_mcts`: Python bindings for MCTS wrapper

pub mod board;
pub mod canonicalization;
pub mod logic;
pub mod action_transform;
pub mod py_logic;
pub mod py_mcts;
mod zobrist;

#[cfg(test)]
mod canonicalization_tests;
#[cfg(test)]
mod action_transform_tests;
#[cfg(test)]
mod zobrist_tests;

// Re-export key types for convenience
pub use board::{BoardConfig, BoardState, GameMode, WinConditions};
pub use py_mcts::PyZertzMCTS;

use crate::game_trait::MCTSGame;
use logic::{
    apply_capture, apply_placement, get_game_outcome, get_valid_actions, is_game_over,
};
use zobrist::ZobristHasher;
use ndarray::{Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};
use std::sync::Arc;

/// Zertz-specific action representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZertzAction {
    /// Place a marble and optionally remove a ring
    Placement {
        marble_type: usize,     // 0=white, 1=gray, 2=black
        dst_y: usize,
        dst_x: usize,
        remove_y: Option<usize>,
        remove_x: Option<usize>,
    },
    /// Capture by jumping
    Capture {
        start_y: usize,
        start_x: usize,
        direction: usize,  // 0-5 for hexagonal directions
    },
    /// Pass (no legal moves)
    Pass,
}

/// Zertz game implementation for MCTS
///
/// Stores game configuration and uses existing Zertz logic from `game.rs`.
pub struct ZertzGame {
    config: Arc<BoardConfig>,
    zobrist: ZobristHasher,
}

impl ZertzGame {
    /// Create a new Zertz game instance
    ///
    /// # Arguments
    /// * `rings` - Number of rings on the board (e.g., 37, 48)
    /// * `t` - Number of timesteps to track in spatial state
    /// * `blitz` - Whether to use blitz rules (different win conditions)
    ///
    /// # Returns
    /// Result containing ZertzGame or error message
    ///
    /// # Example
    /// ```rust,ignore
    /// let game = ZertzGame::new(37, 1, false)?;
    /// ```
    pub fn new(rings: usize, t: usize, blitz: bool) -> Result<Self, String> {
        let config = Arc::new(
            if blitz {
                BoardConfig::blitz(rings, t)?
            } else {
                BoardConfig::standard(rings, t)?
            }
        );

        let zobrist = ZobristHasher::new(config.width, None);

        Ok(Self { config, zobrist })
    }

    /// Get reference to the board configuration
    pub fn config(&self) -> &Arc<BoardConfig> {
        &self.config
    }
}

impl MCTSGame for ZertzGame {
    type Action = ZertzAction;

    fn get_valid_actions(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> Vec<Self::Action> {
        // Call existing game.rs function
        let (placement_mask, capture_mask) =
            get_valid_actions(spatial_state, global_state, &self.config);

        let mut actions = Vec::new();
        let width = self.config.width;
        let width2 = width * width;

        // Extract placements from mask
        for marble_type in 0..3 {
            for dst_flat in 0..width2 {
                for remove_flat in 0..=width2 {
                    if placement_mask[[marble_type, dst_flat, remove_flat]] > 0.0 {
                        let dst_y = dst_flat / width;
                        let dst_x = dst_flat % width;
                        let (remove_y, remove_x) = if remove_flat < width2 {
                            (Some(remove_flat / width), Some(remove_flat % width))
                        } else {
                            (None, None)
                        };
                        actions.push(ZertzAction::Placement {
                            marble_type,
                            dst_y,
                            dst_x,
                            remove_y,
                            remove_x,
                        });
                    }
                }
            }
        }

        // Extract captures from mask
        for dir in 0..6 {
            for y in 0..width {
                for x in 0..width {
                    if capture_mask[[dir, y, x]] > 0.0 {
                        actions.push(ZertzAction::Capture {
                            start_y: y,
                            start_x: x,
                            direction: dir,
                        });
                    }
                }
            }
        }

        // Add Pass if no actions
        if actions.is_empty() {
            actions.push(ZertzAction::Pass);
        }

        actions
    }

    fn apply_action(
        &self,
        spatial_state: &mut ArrayViewMut3<f32>,
        global_state: &mut ArrayViewMut1<f32>,
        action: &Self::Action,
    ) {
        match action {
            ZertzAction::Placement {
                marble_type,
                dst_y,
                dst_x,
                remove_y,
                remove_x,
            } => {
                apply_placement(
                    spatial_state,
                    global_state,
                    *marble_type,
                    *dst_y,
                    *dst_x,
                    *remove_y,
                    *remove_x,
                    &self.config,
                );
            }
            ZertzAction::Capture {
                start_y,
                start_x,
                direction,
            } => {
                apply_capture(
                    spatial_state,
                    global_state,
                    *start_y,
                    *start_x,
                    *direction,
                    &self.config,
                );
            }
            ZertzAction::Pass => {
                // Just switch player
                let cur_player = global_state[self.config.cur_player] as usize;
                global_state[self.config.cur_player] = if cur_player == self.config.player_1 {
                    self.config.player_2 as f32
                } else {
                    self.config.player_1 as f32
                };
            }
        }
    }

    fn is_terminal(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> bool {
        is_game_over(spatial_state, global_state, &self.config)
    }

    fn get_outcome(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> i8 {
        get_game_outcome(spatial_state, global_state, &self.config)
    }

    fn get_current_player(&self, global_state: &ArrayView1<f32>) -> usize {
        global_state[self.config.cur_player] as usize
    }

    fn spatial_shape(&self) -> (usize, usize, usize) {
        let layers = self.config.layers_per_timestep * self.config.t + 1;
        (layers, self.config.width, self.config.width)
    }

    fn global_size(&self) -> usize {
        10 // Hardcoded for Zertz (supply + captures + current player)
    }

    fn evaluate_heuristic(
        &self,
        _spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
        root_player: usize,
    ) -> f32 {
        // Weighted marble value heuristic: white=1, gray=2, black=3
        let weights = [1.0, 2.0, 3.0];
        let p0_score: f32 = (0..3)
            .map(|i| global_state[self.config.p1_cap_w + i] * weights[i])
            .sum();
        let p1_score: f32 = (0..3)
            .map(|i| global_state[self.config.p2_cap_w + i] * weights[i])
            .sum();

        let advantage = if root_player == self.config.player_1 {
            p0_score - p1_score
        } else {
            p1_score - p0_score
        };

        (advantage / 10.0).tanh()
    }

    fn canonicalize_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> (Array3<f32>, Array1<f32>) {
        let (canonical_spatial, _canonical_global, _transform) =
            canonicalization::canonicalize_state(spatial_state, &self.config);
        (canonical_spatial, global_state.to_owned())
    }

    fn hash_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> u64 {
        self.zobrist.hash_state(spatial_state, global_state, &self.config)
    }

    fn enable_deterministic_collapse(&self) -> bool {
        true // Enable for Zertz chain captures
    }

    // Use default get_forced_action (1 action = forced)
    //
    // TREE STRUCTURE IMPACT:
    //
    // WITHOUT deterministic collapse (enable_deterministic_collapse = false):
    // Each forced capture in a chain becomes a separate node in the tree, inflating
    // visit counts and wasting MCTS iterations on non-decisions.
    //
    // Example chain: A -> B (forced) -> C (forced) -> D (choice)
    // Tree WITHOUT collapse:
    //     A [100 visits]
    //       └─ B [100 visits] ← forced, but still explored
    //            └─ C [100 visits] ← forced, but still explored
    //                 └─ D [100 visits] ← real choice point
    //
    // All 100 iterations explore the same forced path, learning nothing.
    // UCB statistics are inflated on B and C even though they're not decisions.
    //
    // WITH deterministic collapse (enable_deterministic_collapse = true):
    // Forced captures B and C are automatically traversed during selection,
    // compressing the sequence into the tree.
    //
    // Tree WITH collapse:
    //     A [100 visits]
    //       └─ D [100 visits] ← jumped directly here via collapsed B->C
    //
    // All 100 iterations reach the real choice point immediately.
    // B and C exist as nodes but don't accumulate visits.
    // MCTS explores actual strategic decisions instead of forced moves.
    //
    // PERFORMANCE IMPACT:
    // - Saves ~2-3x iterations in Zertz endgames with long capture chains
    // - Prevents UCB scores from being distorted by forced node visits
    // - Allows MCTS to focus computational budget on real decisions

    fn name(&self) -> &str {
        "Zertz"
    }
}
