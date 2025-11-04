//! # Zertz Game Implementation
//!
//! Zertz is a GIPF project game featuring:
//! - Hexagonal board with shrinking play area
//! - Three types of marbles (white, gray, black)
//! - Capture mechanics (jump captures and isolation captures)
//! - Win condition: collect required marbles or eliminate opponent
//!
//! This module contains all Zertz-specific code organized into submodules:
//! - `action`: Action types and Python wrappers
//! - `action_result`: Action result types and Python wrappers
//! - `action_transform`: Action transformation for testing symmetry operations
//! - `board`: Board configuration, game modes, win conditions
//! - `canonicalization`: State canonicalization and symmetry detection
//! - `game`: Core game implementation (MCTSGame trait)
//! - `logic`: Core game rules (placement, capture, win conditions)
//! - `notation`: Algebraic notation conversion (e.g., (3,3) ↔ "D4")
//! - `py_logic`: Python bindings for game logic functions
//! - `py_mcts`: Python bindings for MCTS wrapper
//! - `zobrist`: Zobrist hashing for fast state hashing

pub mod action;
pub mod action_result;
pub mod action_transform;
pub mod board;
pub mod canonicalization;
pub mod game;
pub mod logic;
pub mod notation;
pub mod py_logic;
pub mod py_mcts;
mod zobrist;

#[cfg(test)]
mod action_transform_tests;
#[cfg(test)]
mod canonicalization_tests;
#[cfg(test)]
mod notation_tests;
#[cfg(test)]
mod zobrist_tests;

// Re-export key types for convenience
pub use action::{PyZertzAction, ZertzAction};
pub use action_result::{PyZertzActionResult, ZertzActionResult};
pub use board::{BoardConfig, BoardState, GameMode, WinConditions};
pub use game::ZertzGame;
pub use py_mcts::PyZertzMCTS;
