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
//! - `notation`: Algebraic notation conversion (e.g., (3,3) ↔ "D4")
//! - `py_logic`: Python bindings for game logic functions
//! - `py_mcts`: Python bindings for MCTS wrapper

pub mod board;
pub mod canonicalization;
pub mod logic;
pub mod action_transform;
pub mod notation;
pub mod py_logic;
pub mod py_mcts;
mod zobrist;

#[cfg(test)]
mod canonicalization_tests;
#[cfg(test)]
mod action_transform_tests;
#[cfg(test)]
mod notation_tests;
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
use pyo3::{pyclass, pymethods, PyResult};

/// Zertz-specific action representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZertzAction {
    /// Place a marble and optionally remove a ring
    Placement {
        marble_type: usize,     // 0=white, 1=gray, 2=black
        dst_flat: usize,
        remove_flat: Option<usize>,
    },
    /// Capture by jumping from start to dest
    Capture {
        src_flat: usize,
        dst_flat: usize,
    },
    /// Pass (no legal moves)
    Pass,
}

impl ZertzAction {
    /// Convert action to tuple format for serialization
    ///
    /// # Arguments
    /// * `width` - Board width for coordinate flattening
    ///
    /// # Returns
    /// Result containing tuple of (action_type, optional (param1, param2, param3))
    /// - For Placement: ("PUT", Some((marble_type, dst_flat, remove_flat)))
    /// - For Capture: ("CAP", Some((None, src_flat, dst_flat)))
    /// - For Pass: ("PASS", None)
    // todo - remove?
    pub fn to_tuple(&self, width: usize) -> PyResult<(String, Option<(Option<usize>, usize, usize)>)> {
        match self {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let rem_flat = match remove_flat {
                    Some(r_flat) => *r_flat,
                    _ => width * width
                };
                Ok(("PUT".to_string(), Some((Some(*marble_type), *dst_flat, rem_flat))))
            }
            ZertzAction::Capture {
                src_flat,
                dst_flat,
            } => {
                // Return (None, src_flat, dst_flat)
                // None distinguishes captures from placements (which have Some(marble_type))
                // Python will unflatten both coordinates and calculate cap_index as midpoint
                Ok(("CAP".to_string(), Some((None, *src_flat, *dst_flat))))
            }
            ZertzAction::Pass => Ok(("PASS".to_string(), None)),
        }
    }

    pub fn to_placement_action(&self, config: &BoardConfig) -> Option<(String, Vec<Option<usize>>)> {
        // Returns ((dst_y, dst_x), optional (rem_y, rem_x))
        match self {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                let (rem_y, rem_x) = config.flat_to_optional_yx(*remove_flat);

                Some(("PUT".to_string(), vec![Some(*marble_type),
                                              Some(dst_y), Some(dst_x),
                                              rem_y, rem_x]))
            }
            _ => {
                None
            }
        }
    }

    pub fn to_capture_action(&self, config: &BoardConfig)
      -> Option<(String, Vec<usize>)> {
      // Returns ((start_y, start_x), (dst_y, dst_x))
        match self {
            ZertzAction::Capture {
                src_flat,
                dst_flat,
            } => {
                // Return (None, src_flat, dst_flat)
                // None distinguishes captures from placements (which have Some(marble_type))
                // Python will unflatten both coordinates and calculate cap_index as midpoint
                let (src_y, src_x) = config.flat_to_yx(*src_flat);
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                Some(("CAP".to_string(), vec![src_y, src_x, dst_y, dst_x]))
            }
            _ => {
                None
            }
        }
    }
}

/// Result of applying a ZertzAction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZertzActionResult {
    /// Result of a placement action
    Placement {
        /// List of marbles captured due to isolation: (marble_layer, y, x)
        isolation_captures: Vec<(usize, usize, usize)>,
    },
    /// Result of a capture action
    Capture {
        /// The captured marble: (marble_type, y, x)
        captured_marble: (usize, usize, usize),
    },
    /// Result of a pass action (no data)
    Pass,
}

impl ZertzActionResult {
    /// Create a placement result
    pub fn placement(isolation_captures: Vec<(usize, usize, usize)>) -> Self {
        ZertzActionResult::Placement { isolation_captures }
    }

    /// Create a capture result
    pub fn capture(marble_type: usize, y: usize, x: usize) -> Self {
        ZertzActionResult::Capture {
            captured_marble: (marble_type, y, x),
        }
    }

    /// Create a pass result
    pub fn pass() -> Self {
        ZertzActionResult::Pass
    }

    /// Get isolation captures if this is a placement result
    pub fn isolation_captures(&self) -> Option<&Vec<(usize, usize, usize)>> {
        match self {
            ZertzActionResult::Placement { isolation_captures } => Some(isolation_captures),
            _ => None,
        }
    }

    /// Get captured marble if this is a capture result
    pub fn captured_marble(&self) -> Option<(usize, usize, usize)> {
        match self {
            ZertzActionResult::Capture { captured_marble } => Some(*captured_marble),
            _ => None,
        }
    }
}

/// Python wrapper for ZertzActionResult
#[pyclass(name = "ZertzActionResult")]
#[derive(Clone)]
pub struct PyZertzActionResult {
    pub(crate) inner: ZertzActionResult,
}

#[pymethods]
impl PyZertzActionResult {
    /// Get the result type as a string
    fn result_type(&self) -> String {
        match &self.inner {
            ZertzActionResult::Placement { .. } => "PUT".to_string(),
            ZertzActionResult::Capture { .. } => "CAP".to_string(),
            ZertzActionResult::Pass => "PASS".to_string(),
        }
    }

    /// Get isolation captures (only for Placement results)
    ///
    /// Returns:
    ///     List of (marble_layer, y, x) tuples, or None if not a Placement result
    fn isolation_captures(&self) -> Option<Vec<(usize, usize, usize)>> {
        self.inner.isolation_captures().cloned()
    }

    /// Get captured marble (only for Capture results)
    ///
    /// Returns:
    ///     Tuple of (marble_type, y, x), or None if not a Capture result
    fn captured_marble(&self) -> Option<(usize, usize, usize)> {
        self.inner.captured_marble()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ZertzActionResult::Placement { isolation_captures } => {
                if isolation_captures.is_empty() {
                    "ZertzActionResult.Placement(no isolation captures)".to_string()
                } else {
                    format!(
                        "ZertzActionResult.Placement({} isolation captures)",
                        isolation_captures.len()
                    )
                }
            }
            ZertzActionResult::Capture { captured_marble } => {
                let marble_name = match captured_marble.0 {
                    0 => "white",
                    1 => "gray",
                    2 => "black",
                    _ => "unknown",
                };
                format!(
                    "ZertzActionResult.Capture({} at ({}, {}))",
                    marble_name, captured_marble.1, captured_marble.2
                )
            }
            ZertzActionResult::Pass => "ZertzActionResult.Pass".to_string(),
        }
    }
}

/// Python wrapper for ZertzAction
///
/// Provides Python-friendly interface for creating and manipulating Zertz actions.
#[pyclass(name = "ZertzAction")]
#[derive(Clone)]
pub struct PyZertzAction {
    pub(crate) inner: ZertzAction,
}

#[pymethods]
impl PyZertzAction {
    /// Create a Placement action
    ///
    /// Args:
    ///     marble_type: Marble type (0=white, 1=gray, 2=black)
    ///     dst_y: Destination row
    ///     dst_x: Destination column
    ///     remove_y: Optional row of ring to remove
    ///     remove_x: Optional column of ring to remove
    #[staticmethod]
    #[pyo3(signature = (config, marble_type, dst_y, dst_x, remove_y=None, remove_x=None))]
    pub fn placement(
        config: &BoardConfig,
        marble_type: usize,
        dst_y: usize,
        dst_x: usize,
        remove_y: Option<usize>,
        remove_x: Option<usize>,
    ) -> Self {
        let dst_flat = config.yx_to_flat(dst_y, dst_x);
        let remove_flat = config.yx_to_optional_flat(remove_y, remove_x);

        PyZertzAction {
            inner: ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,

            },
        }
    }

    /// Create a Capture action
    ///
    /// Args:
    ///     src_y: Starting row
    ///     src_x: Starting column
    ///     dst_y: Destination row
    ///     dst_x: Destination column
    #[staticmethod]
    pub fn capture(config: &BoardConfig, src_y: usize, src_x: usize, dst_y: usize, dst_x: usize) -> Self {
        PyZertzAction {
            inner: ZertzAction::Capture {
                src_flat: config.yx_to_flat(src_y, src_x),
                dst_flat: config.yx_to_flat(dst_y, dst_x),
            },
        }
    }

    /// Create a Pass action
    #[staticmethod]
    pub fn pass() -> Self {
        PyZertzAction {
            inner: ZertzAction::Pass,
        }
    }

    /// Convert action to tuple format
    ///
    /// Args:
    ///     width: Board width for coordinate handling
    ///
    /// Returns:
    ///     Tuple of (action_type, optional action_data)
    pub fn to_tuple(&self, width: usize) -> PyResult<(String, Option<(Option<usize>, usize, usize)>)> {
        self.inner.to_tuple(width)
    }

    /// Get action type as string
    pub fn action_type(&self) -> String {
        match &self.inner {
            ZertzAction::Placement { .. } => "PUT".to_string(),
            ZertzAction::Capture { .. } => "CAP".to_string(),
            ZertzAction::Pass => "PASS".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let marble_name = match marble_type {
                    0 => "white",
                    1 => "gray",
                    2 => "black",
                    _ => "unknown",
                };
                if let Some(r_flat) = remove_flat {
                    format!(
                        "ZertzAction.Placement({}, {}, remove={})",
                        marble_name, dst_flat, r_flat
                    )
                } else {
                    format!("ZertzAction.Placement({}, {})", marble_name, dst_flat)
                }
            }
            ZertzAction::Capture {
                src_flat,
                dst_flat,
            } => format!(
                "ZertzAction.Capture({} -> {})",
                src_flat, dst_flat
            ),
            ZertzAction::Pass => "ZertzAction.Pass".to_string(),
        }
    }

    fn __eq__(&self, other: &PyZertzAction) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
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

        // Extract placements from mask (5D: marble_type, dst_y, dst_x, rem_y, rem_x)
        for marble_type in 0..3 {
            for dst_y in 0..width {
                for dst_x in 0..width {
                    for rem_y in 0..width {
                        for rem_x in 0..width {
                            if placement_mask[[marble_type, dst_y, dst_x, rem_y, rem_x]] > 0.0 {
                                let dst_flat = self.config.yx_to_flat(dst_y, dst_x);
                                // Sentinel: (dst_y, dst_x) as removal means no removal
                                // Safe because you can never remove the ring you're placing on
                                let remove_flat = if rem_y == dst_y && rem_x == dst_x {
                                    None
                                } else {
                                    Some(self.config.yx_to_flat(rem_y, rem_x))
                                };
                                actions.push(ZertzAction::Placement {
                                    marble_type,
                                    dst_flat,
                                    remove_flat,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Extract captures from mask
        for dir in 0..6 {
            let (dy, dx) = self.config.directions[dir];
            for y in 0..width {
                for x in 0..width {
                    if capture_mask[[dir, y, x]] > 0.0 {
                        // Compute destination (landing position after jump)
                        let dest_y = ((y as i32) + 2 * dy) as usize;
                        let dest_x = ((x as i32) + 2 * dx) as usize;

                        actions.push(ZertzAction::Capture {
                            src_flat: y * width + x,
                            dst_flat: dest_y * width + dest_x,
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
    ) -> Result<(), String> {
        match action {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let (dst_y, dst_x) = self.config.flat_to_yx(*dst_flat);
                let (remove_y, remove_x) = self.config.flat_to_optional_yx(*remove_flat);
                apply_placement(
                    spatial_state,
                    global_state,
                    *marble_type,
                    dst_y,
                    dst_x,
                    remove_y,
                    remove_x,
                    &self.config,
                )?;
            }
            ZertzAction::Capture {
                src_flat,
                dst_flat,
            } => {
                let (src_y, src_x) = self.config.flat_to_yx(*src_flat);
                let (dst_y, dst_x) = self.config.flat_to_yx(*dst_flat);
                apply_capture(
                    spatial_state,
                    global_state,
                    src_y,
                    src_x,
                    dst_y,
                    dst_x,
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
        Ok(())
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
