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
    BoardConfig, BoardState, PyTicTacToeActionResult, PyTicTacToeMCTS, PyZertzAction,
    PyZertzActionResult, PyZertzMCTS,
};

// Import game constants
use games::tictactoe::{PLAYER_X, PLAYER_O, PLAYER_X_WIN, PLAYER_O_WIN, DRAW};
use games::zertz::{
    PLAYER_1_WIN, PLAYER_2_WIN, TIE, BOTH_LOSE,
    STANDARD_MARBLES, BLITZ_MARBLES,
    STANDARD_WIN_CONDITIONS, BLITZ_WIN_CONDITIONS,
};
use crate::games::zertz::{PLAYER_1, PLAYER_2};

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

    let sys_modules = m.py().import("sys")?.getattr("modules")?;
    let mut root_exports = Vec::new();

    // Register Zertz game under submodule
    let zertz_mod = PyModule::new(m.py(), "zertz")?;
    zertz_mod.add_class::<BoardConfig>()?;
    zertz_mod.add_class::<BoardState>()?;
    zertz_mod.add_class::<PyZertzMCTS>()?;
    zertz_mod.add_class::<PyZertzAction>()?;
    zertz_mod.add_class::<PyZertzActionResult>()?;

    // Zertz constants
    zertz_mod.add("PLAYER_1", PLAYER_1)?;
    zertz_mod.add("PLAYER_2", PLAYER_2)?;
    zertz_mod.add("PLAYER_1_WIN", PLAYER_1_WIN)?;
    zertz_mod.add("PLAYER_2_WIN", PLAYER_2_WIN)?;
    zertz_mod.add("TIE", TIE)?;
    zertz_mod.add("BOTH_LOSE", BOTH_LOSE)?;
    zertz_mod.add("STANDARD_MARBLES", STANDARD_MARBLES)?;
    zertz_mod.add("BLITZ_MARBLES", BLITZ_MARBLES)?;
    zertz_mod.add("STANDARD_WIN_CONDITIONS", STANDARD_WIN_CONDITIONS)?;
    zertz_mod.add("BLITZ_WIN_CONDITIONS", BLITZ_WIN_CONDITIONS)?;

    games::zertz::py_logic::register(&zertz_mod)?; // Zertz game logic functions
    m.add_submodule(&zertz_mod)?;
    sys_modules.set_item("hiivelabs_mcts.zertz", &zertz_mod)?;
    root_exports.push("zertz".to_string());
    if let Ok(zertz_all) = zertz_mod.getattr("__all__") {
        let names: Vec<String> = zertz_all.extract()?;
        for name in &names {
            let attr = zertz_mod.getattr(name)?;
            m.add(name, attr)?;
        }
        root_exports.extend(names);
    }

    // Register TicTacToe game under submodule
    let tictactoe_mod = PyModule::new(m.py(), "tictactoe")?;
    tictactoe_mod.add_class::<PyTicTacToeMCTS>()?;
    tictactoe_mod.add_class::<PyTicTacToeActionResult>()?;

    // TicTacToe constants
    tictactoe_mod.add("PLAYER_X", PLAYER_X)?;
    tictactoe_mod.add("PLAYER_O", PLAYER_O)?;
    tictactoe_mod.add("PLAYER_X_WIN", PLAYER_X_WIN)?;
    tictactoe_mod.add("PLAYER_O_WIN", PLAYER_O_WIN)?;
    tictactoe_mod.add("DRAW", DRAW)?;

    games::tictactoe::py_logic::register(&tictactoe_mod)?; // TicTacToe game logic functions
    m.add_submodule(&tictactoe_mod)?;
    sys_modules.set_item("hiivelabs_mcts.tictactoe", &tictactoe_mod)?;
    root_exports.push("tictactoe".to_string());
    if let Ok(tictactoe_all) = tictactoe_mod.getattr("__all__") {
        let names: Vec<String> = tictactoe_all.extract()?;
        for name in &names {
            let attr = tictactoe_mod.getattr(name)?;
            m.add(name, attr)?;
        }
        root_exports.extend(names);
    }

    // Player constants
    m.add("PLAYER_1", 0usize)?;
    m.add("PLAYER_2", 1usize)?;

    root_exports.push("PLAYER_1".to_string());
    root_exports.push("PLAYER_2".to_string());
    root_exports.push("TransformFlags".to_string());

    // Mark as package for Python import machinery
    m.add("__path__", PyList::empty(m.py()))?;
    let all_list = PyList::new(m.py(), &root_exports)?;
    m.add("__all__", all_list)?;

    Ok(())
}
