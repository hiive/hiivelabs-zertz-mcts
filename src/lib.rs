use pyo3::prelude::*;

mod board;
mod canonicalization;
mod game;
mod mcts;
mod node;
mod transposition;
mod zobrist;

use board::BoardState;
use mcts::MCTSSearch;

/// Rust-accelerated MCTS for Zertz
#[pymodule]
#[pyo3(name = "hiivelabs_zertz_mcts")]
fn zertz_mcts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoardState>()?;
    m.add_class::<MCTSSearch>()?;
    Ok(())
}
