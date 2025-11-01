//! # Game Implementations
//!
//! This module contains implementations of the `MCTSGame` trait for various games.
//!
//! Each game has its own submodule with:
//! - Game-specific state representation (mapped to Array3/Array1)
//! - Action definitions
//! - Game rules implementation
//! - Heuristic evaluation
//! - State hashing/canonicalization
//!
//! ## Available Games
//!
//! - **Zertz**: GIPF project game with capture mechanics (see `zertz/` module)
//! - **TicTacToe**: Classic 3x3 game (minimal example)

// Game implementations
pub mod zertz;      // Zertz game (all Zertz code in submodules)
pub mod tictactoe;
pub mod tictactoe_py; // Python bindings for TicTacToe

// Re-export game types for convenience
pub use zertz::{ZertzGame, PyZertzMCTS, PyZertzAction, BoardConfig, BoardState};
pub use tictactoe::TicTacToeGame;
pub use tictactoe_py::PyTicTacToeMCTS;
