use pyo3::prelude::*;
use pyo3::types::PyList;

// Generic MCTS infrastructure
mod canonicalization_transform_flags; // Flags for canonicalization transforms (game-agnostic)
mod game_trait; // Game trait abstraction
mod mcts; // Generic MCTS algorithm
mod metrics; // Generic metrics
mod node; // Generic MCTS node
mod transposition; // Generic transposition table

// Re-export canonicalization flags for public use
pub use canonicalization_transform_flags::{PyTransformFlags, TransformFlags};

// Tests
#[cfg(test)]
mod mcts_tests;
#[cfg(test)]
mod node_tests;
#[cfg(test)]
mod transposition_tests;

// Game implementations
mod games; // Game implementations module (contains zertz/, tictactoe/)

use games::{
    BoardConfig, BoardState, PyTicTacToeMCTS, PyZertzAction, PyZertzActionResult, PyZertzMCTS,
};

/// Generic MCTS engine with game-specific implementations
///
/// Currently supports:
/// - Zertz (via ZertzMCTS class)
/// - TicTacToe (via TicTacToeMCTS class)
#[pymodule]
#[pyo3(name = "hiivelabs_mcts")]
fn hiivelabs_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register generic types
    m.add_class::<PyTransformFlags>()?;

    // Register Zertz game under submodule
    let zertz_mod = PyModule::new(m.py(), "zertz")?;
    zertz_mod.add_class::<BoardConfig>()?;
    zertz_mod.add_class::<BoardState>()?;
    zertz_mod.add_class::<PyZertzMCTS>()?;
    zertz_mod.add_class::<PyZertzAction>()?;
    zertz_mod.add_class::<PyZertzActionResult>()?;
    games::zertz::py_logic::register(&zertz_mod)?; // Zertz game logic functions
    m.add_submodule(&zertz_mod)?;

    // Register TicTacToe game under submodule
    let tictactoe_mod = PyModule::new(m.py(), "tictactoe")?;
    tictactoe_mod.add_class::<PyTicTacToeMCTS>()?;
    m.add_submodule(&tictactoe_mod)?;

    // Player constants
    m.add("PLAYER_1", 0usize)?;
    m.add("PLAYER_2", 1usize)?;

    // Mark as package for Python import machinery
    m.add("__path__", PyList::empty(m.py()))?;
    let submodules = PyList::new(m.py(), ["zertz", "tictactoe"])?;
    m.add("__all__", submodules)?;

    Ok(())
}
