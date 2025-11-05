//! # Tic-Tac-Toe Game Implementation
//!
//! A minimal example demonstrating the `MCTSGame` trait for the classic 3x3 Tic-Tac-Toe game.
//!
//! ## State Representation
//! - **Spatial state**: (2, 3, 3) array
//!   - Layer 0: X positions (player 0)
//!   - Layer 1: O positions (player 1)
//! - **Global state**: (1,) array
//!   - Index 0: Current player (0 for X, 1 for O)
//!
//! ## Actions
//! - Simple (row, col) placement moves
//!
//! ## Rules
//! - Players alternate placing marks
//! - Win: Three in a row (horizontal, vertical, or diagonal)
//! - Draw: Board full, with no winner

pub mod action_result;
mod canonicalization;
pub mod py_logic;
pub mod py_mcts;

#[cfg(test)]
mod canonicalization_tests;

use crate::game_trait::MCTSGame;
use ndarray::{Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};

pub use action_result::PyTicTacToeActionResult;

/// Tic-Tac-Toe action: place mark at (row, col)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToeAction {
    pub row: usize,
    pub col: usize,
}

/// Tic-Tac-Toe game implementation for MCTS
pub struct TicTacToeGame;

impl TicTacToeGame {
    /// Create a new Tic-Tac-Toe game instance
    pub fn new() -> Self {
        TicTacToeGame
    }

    /// Create initial state (empty board, X to move)
    pub fn initial_state() -> (Array3<f32>, Array1<f32>) {
        let spatial_state = Array3::zeros((2, 3, 3));
        let mut global_state = Array1::zeros(1);
        global_state[0] = 0.0; // X (player 0) starts
        (spatial_state, global_state)
    }

    /// Check if there are three in a row for the given player layer
    fn check_three_in_row(spatial_state: &ArrayView3<f32>, player_layer: usize) -> bool {
        // Check rows
        for row in 0..3 {
            if spatial_state[[player_layer, row, 0]] > 0.0
                && spatial_state[[player_layer, row, 1]] > 0.0
                && spatial_state[[player_layer, row, 2]] > 0.0
            {
                return true;
            }
        }

        // Check columns
        for col in 0..3 {
            if spatial_state[[player_layer, 0, col]] > 0.0
                && spatial_state[[player_layer, 1, col]] > 0.0
                && spatial_state[[player_layer, 2, col]] > 0.0
            {
                return true;
            }
        }

        // Check diagonals
        if spatial_state[[player_layer, 0, 0]] > 0.0
            && spatial_state[[player_layer, 1, 1]] > 0.0
            && spatial_state[[player_layer, 2, 2]] > 0.0
        {
            return true;
        }

        if spatial_state[[player_layer, 0, 2]] > 0.0
            && spatial_state[[player_layer, 1, 1]] > 0.0
            && spatial_state[[player_layer, 2, 0]] > 0.0
        {
            return true;
        }

        false
    }

    /// Check if board is full
    fn is_board_full(spatial_state: &ArrayView3<f32>) -> bool {
        for row in 0..3 {
            for col in 0..3 {
                if spatial_state[[0, row, col]] == 0.0 && spatial_state[[1, row, col]] == 0.0 {
                    return false;
                }
            }
        }
        true
    }
}

impl MCTSGame for TicTacToeGame {
    type Action = TicTacToeAction;

    fn get_valid_actions(
        &self,
        spatial_state: &ArrayView3<f32>,
        _global_state: &ArrayView1<f32>,
    ) -> Vec<Self::Action> {
        let mut actions = Vec::new();

        // Any empty cell is a valid move
        for row in 0..3 {
            for col in 0..3 {
                // Cell is empty if neither player has marked it
                if spatial_state[[0, row, col]] == 0.0 && spatial_state[[1, row, col]] == 0.0 {
                    actions.push(TicTacToeAction { row, col });
                }
            }
        }

        actions
    }

    fn apply_action(
        &self,
        spatial_state: &mut ArrayViewMut3<f32>,
        global_state: &mut ArrayViewMut1<f32>,
        action: &Self::Action,
    ) -> Result<(), String> {
        let current_player = global_state[0] as usize;

        // Place mark for current player
        spatial_state[[current_player, action.row, action.col]] = 1.0;

        // Switch player
        global_state[0] = if current_player == 0 { 1.0 } else { 0.0 };

        Ok(())
    }

    fn is_terminal(
        &self,
        spatial_state: &ArrayView3<f32>,
        _global_state: &ArrayView1<f32>,
    ) -> bool {
        // Terminal if either player has won or board is full
        Self::check_three_in_row(spatial_state, 0)
            || Self::check_three_in_row(spatial_state, 1)
            || Self::is_board_full(spatial_state)
    }

    fn get_outcome(&self, spatial_state: &ArrayView3<f32>, _global_state: &ArrayView1<f32>) -> i8 {
        // Check X (player 0) win
        if Self::check_three_in_row(spatial_state, 0) {
            return 1; // X wins
        }

        // Check O (player 1) win
        if Self::check_three_in_row(spatial_state, 1) {
            return -1; // O wins
        }

        // Draw
        0
    }

    fn get_current_player(&self, global_state: &ArrayView1<f32>) -> usize {
        global_state[0] as usize
    }

    fn spatial_shape(&self) -> (usize, usize, usize) {
        (2, 3, 3)
    }

    fn global_size(&self) -> usize {
        1
    }

    fn evaluate_heuristic(
        &self,
        spatial_state: &ArrayView3<f32>,
        _global_state: &ArrayView1<f32>,
        root_player: usize,
    ) -> f32 {
        // Simple heuristic: count potential winning lines
        let opponent = if root_player == 0 { 1 } else { 0 };

        let mut score = 0.0;

        // Check all possible lines
        let lines = [
            // Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            // Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            // Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ];

        for line in &lines {
            let mut my_count = 0;
            let mut opp_count = 0;

            for &(r, c) in line {
                if spatial_state[[root_player, r, c]] > 0.0 {
                    my_count += 1;
                }
                if spatial_state[[opponent, r, c]] > 0.0 {
                    opp_count += 1;
                }
            }

            // Line is valuable if we have marks and opponent doesn't (or vice versa)
            if opp_count == 0 && my_count > 0 {
                score += my_count as f32;
            }
            if my_count == 0 && opp_count > 0 {
                score -= opp_count as f32;
            }
        }

        // Normalize to [-1, 1]
        (score / 24.0).tanh()
    }

    fn canonicalize_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> (Array3<f32>, Array1<f32>) {
        // Use D4 dihedral group symmetries to find canonical form
        let (canonical_spatial, _transform_idx) =
            canonicalization::canonicalize_state(spatial_state);
        (canonical_spatial, global_state.to_owned())
    }

    fn hash_state(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> u64 {
        // Simple hash: encode board as 18-bit number (2 bits per cell: 00=empty, 01=X, 10=O)
        let mut hash: u64 = 0;

        for row in 0..3 {
            for col in 0..3 {
                hash <<= 2;
                if spatial_state[[0, row, col]] > 0.0 {
                    hash |= 1; // X
                } else if spatial_state[[1, row, col]] > 0.0 {
                    hash |= 2; // O
                }
                // else 0 (empty)
            }
        }

        // Add current player bit
        hash = (hash << 1) | (global_state[0] as u64);

        hash
    }

    fn name(&self) -> &str {
        "TicTacToe"
    }
}

#[cfg(test)]
mod tictactoe_tests;

// Re-export Python bindings
pub use py_mcts::PyTicTacToeMCTS;
