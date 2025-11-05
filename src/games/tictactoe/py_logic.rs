//! Python bindings for TicTacToe game logic functions
//!
//! This module exposes stateless TicTacToe game logic functions to Python,
//! allowing Python code to call Rust game logic directly.

use super::action_result::{PyTicTacToeActionResult, TicTacToeActionResult};
use super::{TicTacToeAction, TicTacToeGame, DRAW};
use crate::game_trait::MCTSGame;
use ndarray::Array1;
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

/// Get list of valid (row, col) moves
///
/// Args:
///     spatial_state: (2, 3, 3) array with X positions in layer 0, O positions in layer 1
///
/// Returns:
///     List of (row, col) tuples for empty squares
#[pyfunction]
pub fn get_valid_actions(
    spatial_state: PyReadonlyArray3<f32>,
) -> PyResult<Vec<(usize, usize)>> {
    let state = spatial_state.as_array();
    let game = TicTacToeGame::new();
    let dummy_global = Array1::zeros(1);
    
    let actions = game.get_valid_actions(&state, &dummy_global.view());
    Ok(actions.into_iter().map(|a| (a.row, a.col)).collect())
}

/// Apply an action to the state (mutates arrays in-place)
///
/// Args:
    ///     spatial_state: (2, 3, 3) array (MUTATED IN-PLACE)
///     global_state: (1,) array (MUTATED IN-PLACE)
///     row: Row index (0-2)
///     col: Column index (0-2)
///
/// Returns:
///     TicTacToeActionResult indicating if move is normal, winning, or drawing
#[pyfunction]
pub fn apply_action<'py>(
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    row: usize,
    col: usize,
) -> PyResult<PyTicTacToeActionResult> {
    let game = TicTacToeGame::new();
    let action = TicTacToeAction { row, col };
    
    unsafe {
        let mut spatial_state_arr = spatial_state.as_array_mut();
        let mut global_state_arr = global_state.as_array_mut();
        
        // Get current player before applying action
        let current_player = global_state_arr[0] as usize;
        
        // Apply action
        game.apply_action(&mut spatial_state_arr, &mut global_state_arr, &action);
        
        // Check if terminal
        if game.is_terminal(&spatial_state_arr.view(), &global_state_arr.view()) {
            let outcome = game.get_outcome(&spatial_state_arr.view(), &global_state_arr.view());
            if outcome == DRAW {
                // Draw
                Ok(PyTicTacToeActionResult {
                    inner: TicTacToeActionResult::Draw,
                })
            } else {
                // Win for the player who just moved
                Ok(PyTicTacToeActionResult {
                    inner: TicTacToeActionResult::Win,
                })
            }
        } else {
            // Normal move
            Ok(PyTicTacToeActionResult {
                inner: TicTacToeActionResult::Move,
            })
        }
    }
}

/// Check if the game is over
///
/// Args:
///     spatial_state: (2, 3, 3) array
///     global_state: (1,) array
///
/// Returns:
///     True if game is over (win or draw), False otherwise
#[pyfunction]
pub fn is_game_over(
    spatial_state: PyReadonlyArray3<f32>,
    global_state: PyReadonlyArray1<f32>,
) -> PyResult<bool> {
    let game = TicTacToeGame::new();
    Ok(game.is_terminal(&spatial_state.as_array(), &global_state.as_array()))
}

/// Get the game outcome
///
/// Args:
///     spatial_state: (2, 3, 3) array
///     global_state: (1,) array
///
/// Returns:
///     +1 if player 0 (X) wins, -1 if player 1 (O) wins, 0 for draw
///     Should only be called when game is over
#[pyfunction]
pub fn get_game_outcome(
    spatial_state: PyReadonlyArray3<f32>,
    global_state: PyReadonlyArray1<f32>,
) -> PyResult<i8> {
    let game = TicTacToeGame::new();
    Ok(game.get_outcome(&spatial_state.as_array(), &global_state.as_array()))
}

/// Get the current player
///
/// Args:
///     global_state: (1,) array
///
/// Returns:
///     0 for X (player 0), 1 for O (player 1)
#[pyfunction]
pub fn get_current_player(global_state: PyReadonlyArray1<f32>) -> PyResult<usize> {
    let game = TicTacToeGame::new();
    Ok(game.get_current_player(&global_state.as_array()))
}

/// Create initial game state
///
/// Returns:
///     Tuple of (spatial_state, global_state) representing empty board with X to move
#[pyfunction]
pub fn initial_state(py: Python<'_>) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray1<f32>>)> {
    let (spatial, global) = TicTacToeGame::initial_state();
    let spatial_py = PyArray3::from_array(py, &spatial).into();
    let global_py = PyArray1::from_array(py, &global).into();
    Ok((spatial_py, global_py))
}

/// Register all game logic functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_valid_actions, m)?)?;
    m.add_function(wrap_pyfunction!(apply_action, m)?)?;
    m.add_function(wrap_pyfunction!(is_game_over, m)?)?;
    m.add_function(wrap_pyfunction!(get_game_outcome, m)?)?;
    m.add_function(wrap_pyfunction!(get_current_player, m)?)?;
    m.add_function(wrap_pyfunction!(initial_state, m)?)?;
    
    Ok(())
}
