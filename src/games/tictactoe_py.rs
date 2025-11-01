//! # Python Bindings for Tic-Tac-Toe MCTS
//!
//! This module provides PyO3 bindings for the generic MCTS with TicTacToeGame.
//! It wraps `MCTSSearch<TicTacToeGame>` and provides Python-compatible interfaces.

use crate::games::tictactoe::TicTacToeGame;
use crate::mcts::MCTSSearch;
use crate::node::MCTSNode;
use numpy::{PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Python-facing wrapper for MCTS with TicTacToeGame
#[pyclass(name = "TicTacToeMCTS")]
pub struct PyTicTacToeMCTS {
    search: MCTSSearch<TicTacToeGame>,
    game: Arc<TicTacToeGame>,
}

#[pymethods]
impl PyTicTacToeMCTS {
    /// Create a new Tic-Tac-Toe MCTS instance
    #[new]
    #[pyo3(signature = (
        exploration_constant=None,
        use_transposition_table=true,
        use_transposition_lookups=true
    ))]
    pub fn new(
        exploration_constant: Option<f32>,
        use_transposition_table: bool,
        use_transposition_lookups: bool,
    ) -> PyResult<Self> {
        // Create TicTacToeGame instance
        let game = Arc::new(TicTacToeGame::new());

        // Create generic MCTS search
        let search = MCTSSearch::new(
            Arc::clone(&game),
            exploration_constant,
            None, // widening_constant
            None, // fpu_reduction
            None, // rave_constant
            Some(use_transposition_table),
            Some(use_transposition_lookups),
        );

        Ok(Self { search, game })
    }

    /// Create initial game state
    ///
    /// Returns (spatial_state, global_state) tuple:
    /// - spatial_state: (2, 3, 3) array with layers for X and O
    /// - global_state: (1,) array with current player
    #[staticmethod]
    pub fn initial_state(py: Python) -> (Py<PyAny>, Py<PyAny>) {
        let (spatial, global) = TicTacToeGame::initial_state();

        // Convert to NumPy arrays
        let spatial_np = numpy::PyArray::from_owned_array(py, spatial);
        let global_np = numpy::PyArray::from_owned_array(py, global);

        (spatial_np.unbind().into(), global_np.unbind().into())
    }

    /// Set deterministic RNG seed
    #[pyo3(signature = (seed=None))]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.search.set_seed(seed);
    }

    /// Enable/disable transposition table
    pub fn set_transposition_table_enabled(&mut self, enabled: bool) {
        self.search.set_transposition_table_enabled(enabled);
    }

    /// Clear transposition table
    pub fn clear_transposition_table(&mut self) {
        self.search.clear_transposition_table();
    }

    /// Get number of children of last root
    pub fn last_root_children(&self) -> usize {
        self.search.last_root_children()
    }

    /// Get visit count of last root
    pub fn last_root_visits(&self) -> u32 {
        self.search.last_root_visits()
    }

    /// Get average value of last root
    pub fn last_root_value(&self) -> f32 {
        self.search.last_root_value()
    }

    /// Run MCTS search and return best action
    ///
    /// Args:
    ///     spatial_state: (2, 3, 3) NumPy array
    ///     global_state: (1,) NumPy array
    ///     iterations: Number of MCTS iterations
    ///     clear_table: Clear transposition table before search
    ///
    /// Returns:
    ///     (row, col) tuple for best move, or None if no moves available
    #[pyo3(signature = (spatial_state, global_state, iterations, clear_table=false))]
    pub fn search(
        &mut self,
        spatial_state: PyReadonlyArray3<f32>,
        global_state: PyReadonlyArray1<f32>,
        iterations: usize,
        clear_table: bool,
    ) -> Option<(usize, usize)> {
        let spatial = spatial_state.as_array().to_owned();
        let global = global_state.as_array().to_owned();

        if clear_table {
            self.search.clear_transposition_table();
        }

        // Create root node
        let root = Arc::new(MCTSNode::new(
            spatial,
            global,
            Arc::clone(&self.game),
            None,
        ));

        // Create search options
        let options = crate::mcts::SearchOptions::new(
            &mut self.search,
            None, // use_table_override
            None, // use_lookups_override
            clear_table,
        );

        let start_time = Instant::now();

        // Run MCTS iterations
        for _ in 0..iterations {
            self.search.run_iteration(
                Arc::clone(&root),
                &options,
                None, // max_depth
                None, // time_limit
                start_time,
            );
        }

        // Capture root stats for child statistics
        self.search.capture_root_stats(&root);

        // Get best action
        self.select_best_action()
    }

    /// Get per-child statistics from last search
    ///
    /// Returns list of ((row, col), normalized_score) tuples
    pub fn last_child_statistics(&self) -> Vec<((usize, usize), f32)> {
        let child_stats = self.search.last_child_stats.lock().unwrap();

        child_stats
            .iter()
            .map(|(action, score)| ((action.row, action.col), *score))
            .collect()
    }
}

impl PyTicTacToeMCTS {
    /// Select best action from last search
    fn select_best_action(&self) -> Option<(usize, usize)> {
        let child_stats = self.search.last_child_stats.lock().unwrap();

        if child_stats.is_empty() {
            return None;
        }

        // Return action with highest score (most visits)
        child_stats
            .iter()
            .max_by(|(_, score1), (_, score2)| {
                score1.partial_cmp(score2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(action, _)| (action.row, action.col))
    }
}
