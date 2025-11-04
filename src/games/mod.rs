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
pub mod tictactoe;  // TicTacToe game (all TicTacToe code in submodules)

// Re-export game types for convenience
pub use zertz::{ZertzGame, PyZertzMCTS, PyZertzAction, PyZertzActionResult, BoardConfig, BoardState};
pub use tictactoe::{TicTacToeGame, PyTicTacToeMCTS};
