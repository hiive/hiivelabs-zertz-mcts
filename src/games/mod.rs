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
//! - **Zertz**: GIPF project game with capture mechanics
//! - **TicTacToe**: Classic 3x3 game (minimal example)

// Game implementations
pub mod zertz;
pub mod zertz_py; // Python bindings
pub mod tictactoe;
pub mod tictactoe_py; // Python bindings

// Re-export game types for convenience
pub use zertz::ZertzGame;
pub use zertz_py::PyZertzMCTS;
pub use tictactoe::TicTacToeGame;
pub use tictactoe_py::PyTicTacToeMCTS;
