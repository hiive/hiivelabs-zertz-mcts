use pyo3::prelude::*;

mod board;
mod canonicalization;
mod game_py;     // Python bindings for Zertz game logic
mod game_trait;  // Game trait abstraction
mod games;       // Game implementations module
mod mcts;
mod metrics;
mod node;
mod transposition;
mod zobrist;

use board::{BoardConfig, BoardState};
use games::{PyZertzMCTS, PyTicTacToeMCTS};

/// Generic MCTS engine with game-specific implementations
///
/// Currently supports:
/// - Zertz (via ZertzMCTS class)
/// - TicTacToe (via TicTacToeMCTS class)
#[pymodule]
#[pyo3(name = "hiivelabs_mcts")]
fn hiivelabs_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoardConfig>()?;
    m.add_class::<BoardState>()?;

    // Register game implementations
    m.add_class::<PyZertzMCTS>()?;
    m.add_class::<PyTicTacToeMCTS>()?;

    // Register game logic functions
    game_py::register(m)?;

    // Player constants
    m.add("PLAYER_1", 0usize)?;
    m.add("PLAYER_2", 1usize)?;

    Ok(())
}
