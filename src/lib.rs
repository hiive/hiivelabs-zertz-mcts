use pyo3::prelude::*;

mod board;
mod canonicalization;
mod game;
mod game_py;
mod mcts;
mod metrics;
mod node;
mod transposition;
mod zobrist;

use board::{BoardConfig, BoardState};
use mcts::MCTSSearch;

/// Rust-accelerated MCTS for Zertz
#[pymodule]
#[pyo3(name = "hiivelabs_zertz_mcts")]
fn zertz_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoardConfig>()?;
    m.add_class::<BoardState>()?;
    m.add_class::<MCTSSearch>()?;

    // Register game logic functions
    game_py::register(m)?;

    // Player constants
    m.add("PLAYER_1", 0usize)?;
    m.add("PLAYER_2", 1usize)?;

    Ok(())
}
