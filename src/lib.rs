use pyo3::prelude::*;

// Generic MCTS infrastructure
mod game_trait;  // Game trait abstraction
mod mcts;        // Generic MCTS algorithm
mod metrics;     // Generic metrics
mod node;        // Generic MCTS node
mod transposition; // Generic transposition table

// Tests
#[cfg(test)]
mod mcts_tests;
#[cfg(test)]
mod node_tests;
#[cfg(test)]
mod transposition_tests;

// Game implementations
mod games;       // Game implementations module (contains zertz/, tictactoe/)

use games::{BoardConfig, BoardState, PyZertzMCTS, PyZertzAction, PyTicTacToeMCTS};

/// Generic MCTS engine with game-specific implementations
///
/// Currently supports:
/// - Zertz (via ZertzMCTS class)
/// - TicTacToe (via TicTacToeMCTS class)
#[pymodule]
#[pyo3(name = "hiivelabs_mcts")]
fn hiivelabs_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register Zertz game
    m.add_class::<BoardConfig>()?;
    m.add_class::<BoardState>()?;
    m.add_class::<PyZertzMCTS>()?;
    m.add_class::<PyZertzAction>()?;
    games::zertz::py_logic::register(m)?; // Zertz game logic functions

    // Register TicTacToe game
    m.add_class::<PyTicTacToeMCTS>()?;

    // Player constants
    m.add("PLAYER_1", 0usize)?;
    m.add("PLAYER_2", 1usize)?;

    Ok(())
}
