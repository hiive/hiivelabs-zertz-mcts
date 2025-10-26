//! Python bindings for game logic functions
//!
//! This module exposes stateless game logic functions to Python,
//! allowing Python code to call Rust game logic directly.

use crate::board::BoardConfig;
use crate::game;
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

// ============================================================================
// Axial Coordinate Transformations
// ============================================================================

/// Rotate axial coordinate by k * 60° counterclockwise
///
/// Args:
///     q: Axial q coordinate
///     r: Axial r coordinate
///     k: Number of 60° rotations (will be normalized to 0-5)
///
/// Returns:
///     Tuple of (q, r) rotated coordinates
#[pyfunction]
pub fn ax_rot60(q: i32, r: i32, k: i32) -> (i32, i32) {
    game::ax_rot60(q, r, k)
}

/// Mirror axial coordinate across the q-axis
///
/// Args:
///     q: Axial q coordinate
///     r: Axial r coordinate
///
/// Returns:
///     Tuple of (q, r) mirrored coordinates
#[pyfunction]
pub fn ax_mirror_q_axis(q: i32, r: i32) -> (i32, i32) {
    game::ax_mirror_q_axis(q, r)
}

// ============================================================================

/// Check if (y, x) coordinates are within board bounds
#[pyfunction]
pub fn is_inbounds(y: i32, x: i32, width: usize) -> bool {
    game::is_inbounds(y, x, width)
}

/// Get list of neighboring indices (filtered to in-bounds only)
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_neighbors(y: usize, x: usize, config: &BoardConfig) -> Vec<(usize, usize)> {
    game::get_neighbors(y, x, config).into_iter().collect()
}

/// Calculate landing position after capturing marble
/// Returns (land_y, land_x) as i32 tuple (may be out of bounds)
#[pyfunction]
pub fn get_jump_destination(
    start_y: usize,
    start_x: usize,
    cap_y: usize,
    cap_x: usize,
) -> (i32, i32) {
    game::get_jump_destination(start_y, start_x, cap_y, cap_x)
}

/// Find all connected regions on the board
/// Returns list of regions, where each region is a list of (y, x) tuples
#[pyfunction]
pub fn get_regions<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<Vec<(usize, usize)>> {
    let spatial = spatial.as_array();
    game::get_regions(&spatial, config)
}

/// Get list of empty ring indices across the board
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_open_rings<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<(usize, usize)> {
    let spatial = spatial.as_array();
    game::get_open_rings(&spatial, config)
}

/// Check if ring can be removed (geometric rule)
#[pyfunction]
pub fn is_ring_removable<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
    config: &BoardConfig,
) -> bool {
    let spatial = spatial.as_array();
    game::is_ring_removable(&spatial, y, x, config)
}

/// Get removable rings (rings that can be removed without disconnecting board)
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_removable_rings<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<(usize, usize)> {
    let spatial = spatial.as_array();
    game::get_removable_rings(&spatial, config)
}

/// Get global_state index for marble type in supply
#[pyfunction]
pub fn get_supply_index(marble_type: char) -> usize {
    // Supply indices are always the same: W=0, G=1, B=2
    match marble_type {
        'w' => 0,
        'g' => 1,
        'b' => 2,
        _ => panic!("Invalid marble type: {}", marble_type),
    }
}

/// Get global_state index for captured marble
#[pyfunction]
pub fn get_captured_index(player: usize, marble_type: char) -> usize {
    // Captured indices: P1: W=3, G=4, B=5; P2: W=6, G=7, B=8
    // Player constants: PLAYER_1 = 0, PLAYER_2 = 1
    let marble_idx = match marble_type {
        'w' => 0,
        'g' => 1,
        'b' => 2,
        _ => panic!("Invalid marble type: {}", marble_type),
    };
    if player == 0 {
        3 + marble_idx  // Player 1 (player == 0)
    } else {
        6 + marble_idx  // Player 2 (player == 1)
    }
}

/// Get marble type at given position
/// Returns: 'w', 'g', 'b', or '\0' (none)
#[pyfunction]
pub fn get_marble_type_at<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
) -> char {
    let spatial = spatial.as_array();
    if spatial[[1, y, x]] == 1.0 {
        'w'
    } else if spatial[[2, y, x]] == 1.0 {
        'g'
    } else if spatial[[3, y, x]] == 1.0 {
        'b'
    } else {
        '\0'
    }
}

/// Get valid placement actions
/// Returns Array3<f32> with shape (3, width², width²+1)
#[pyfunction]
pub fn get_placement_moves<'py>(
    py: Python<'py>,
    spatial: PyReadonlyArray3<'py, f32>,
    global: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> Py<PyArray3<f32>> {
    let spatial = spatial.as_array();
    let global = global.as_array();
    let result = game::get_placement_actions(&spatial, &global, config);
    PyArray3::from_array(py, &result).into()
}

/// Get valid capture actions
/// Returns Array3<f32> with shape (6, width, width)
#[pyfunction]
pub fn get_capture_moves<'py>(
    py: Python<'py>,
    spatial: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Py<PyArray3<f32>> {
    let spatial = spatial.as_array();
    let result = game::get_capture_actions(&spatial, config);
    PyArray3::from_array(py, &result).into()
}

/// Get valid actions (both placement and capture)
/// Returns tuple of (placement_mask, capture_mask)
#[pyfunction]
pub fn get_valid_actions<'py>(
    py: Python<'py>,
    spatial: PyReadonlyArray3<'py, f32>,
    global: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>) {
    let spatial = spatial.as_array();
    let global = global.as_array();
    let (placement, capture) = game::get_valid_actions(&spatial, &global, config);
    (PyArray3::from_array(py, &placement).into(), PyArray3::from_array(py, &capture).into())
}

/// Apply a placement action (mutates arrays in-place)
#[pyfunction]
pub fn apply_placement_action<'py>(
    spatial: &Bound<'py, PyArray3<f32>>,
    global: &Bound<'py, PyArray1<f32>>,
    marble_type: usize,
    dst_y: usize,
    dst_x: usize,
    remove_y: Option<usize>,
    remove_x: Option<usize>,
    config: &BoardConfig,
) -> PyResult<()> {
    unsafe {
        let mut spatial_arr = spatial.as_array_mut();
        let mut global_arr = global.as_array_mut();
        game::apply_placement(
            &mut spatial_arr,
            &mut global_arr,
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            config,
        );
    }
    Ok(())
}

/// Apply a capture action (mutates arrays in-place)
#[pyfunction]
pub fn apply_capture_action<'py>(
    spatial: &Bound<'py, PyArray3<f32>>,
    global: &Bound<'py, PyArray1<f32>>,
    start_y: usize,
    start_x: usize,
    direction: usize,
    config: &BoardConfig,
) -> PyResult<()> {
    unsafe {
        let mut spatial_arr = spatial.as_array_mut();
        let mut global_arr = global.as_array_mut();
        game::apply_capture(
            &mut spatial_arr,
            &mut global_arr,
            start_y,
            start_x,
            direction,
            config,
        );
    }
    Ok(())
}

/// Check if game is over (any terminal condition)
#[pyfunction]
pub fn is_game_over<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    global: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> bool {
    let spatial = spatial.as_array();
    let global = global.as_array();
    game::is_game_over(&spatial, &global, config)
}

/// Get game outcome from Player 1's perspective
/// Returns: 1 (P1 wins), -1 (P2 wins), 0 (tie), -2 (both lose)
#[pyfunction]
pub fn get_game_outcome<'py>(
    spatial: PyReadonlyArray3<'py, f32>,
    global: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> i8 {
    let spatial = spatial.as_array();
    let global = global.as_array();
    game::get_game_outcome(&spatial, &global, config)
}

/// Check for isolated regions and capture marbles
/// Returns tuple of (updated_spatial, updated_global, captured_marbles_list)
/// where captured_marbles_list contains tuples of (marble_layer_idx, y, x)
#[pyfunction]
pub fn check_for_isolation_capture<'py>(
    py: Python<'py>,
    spatial: PyReadonlyArray3<'py, f32>,
    global: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> (Py<PyArray3<f32>>, Py<PyArray1<f32>>, Vec<(usize, usize, usize)>) {
    let spatial = spatial.as_array();
    let global = global.as_array();
    let (spatial_out, global_out, captured_marbles) =
        game::check_for_isolation_capture(&spatial, &global, config);
    (
        PyArray3::from_array(py, &spatial_out).into(),
        PyArray1::from_array(py, &global_out).into(),
        captured_marbles,
    )
}

/// Register all game logic functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Axial coordinate transformations
    m.add_function(wrap_pyfunction!(ax_rot60, m)?)?;
    m.add_function(wrap_pyfunction!(ax_mirror_q_axis, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(is_inbounds, m)?)?;
    m.add_function(wrap_pyfunction!(get_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(get_jump_destination, m)?)?;
    m.add_function(wrap_pyfunction!(get_regions, m)?)?;
    m.add_function(wrap_pyfunction!(get_open_rings, m)?)?;
    m.add_function(wrap_pyfunction!(is_ring_removable, m)?)?;
    m.add_function(wrap_pyfunction!(get_removable_rings, m)?)?;
    m.add_function(wrap_pyfunction!(get_supply_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_captured_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_marble_type_at, m)?)?;

    // Move generation and action execution
    m.add_function(wrap_pyfunction!(get_placement_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_capture_moves, m)?)?;
    m.add_function(wrap_pyfunction!(get_valid_actions, m)?)?;
    m.add_function(wrap_pyfunction!(apply_placement_action, m)?)?;
    m.add_function(wrap_pyfunction!(apply_capture_action, m)?)?;

    // Game termination
    m.add_function(wrap_pyfunction!(is_game_over, m)?)?;
    m.add_function(wrap_pyfunction!(get_game_outcome, m)?)?;

    // Isolation capture
    m.add_function(wrap_pyfunction!(check_for_isolation_capture, m)?)?;

    Ok(())
}