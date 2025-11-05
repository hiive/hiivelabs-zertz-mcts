//! # TicTacToe Action Result Types
//!
//! Defines result types returned when applying actions to the game state.

use pyo3::{pyclass, pymethods};

/// Result of applying a TicTacToeAction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TicTacToeActionResult {
    /// Normal move (game continues)
    Move,
    /// Winning move for the player who moved
    Win,
    /// Move that results in a draw
    Draw,
}

impl TicTacToeActionResult {
    /// Create a normal move result
    pub fn move_result() -> Self {
        TicTacToeActionResult::Move
    }

    /// Create a winning result
    pub fn win() -> Self {
        TicTacToeActionResult::Win
    }

    /// Create a draw result
    pub fn draw() -> Self {
        TicTacToeActionResult::Draw
    }

    /// Check if this is a terminal result (win or draw)
    pub fn is_terminal(&self) -> bool {
        matches!(self, TicTacToeActionResult::Win | TicTacToeActionResult::Draw)
    }

    /// Check if this is a win
    pub fn is_win(&self) -> bool {
        matches!(self, TicTacToeActionResult::Win)
    }

    /// Check if this is a draw
    pub fn is_draw(&self) -> bool {
        matches!(self, TicTacToeActionResult::Draw)
    }
}

/// Python wrapper for TicTacToeActionResult
#[pyclass(name = "TicTacToeActionResult")]
#[derive(Clone)]
pub struct PyTicTacToeActionResult {
    pub(crate) inner: TicTacToeActionResult,
}

#[pymethods]
impl PyTicTacToeActionResult {
    /// Get the result type as a string
    fn result_type(&self) -> String {
        match &self.inner {
            TicTacToeActionResult::Move => "Move".to_string(),
            TicTacToeActionResult::Win => "Win".to_string(),
            TicTacToeActionResult::Draw => "Draw".to_string(),
        }
    }

    /// Check if the game is over (win or draw)
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Check if this is a winning move
    fn is_win(&self) -> bool {
        self.inner.is_win()
    }

    /// Check if this is a draw
    fn is_draw(&self) -> bool {
        self.inner.is_draw()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            TicTacToeActionResult::Move => "TicTacToeActionResult.Move".to_string(),
            TicTacToeActionResult::Win => "TicTacToeActionResult.Win".to_string(),
            TicTacToeActionResult::Draw => "TicTacToeActionResult.Draw".to_string(),
        }
    }
}
