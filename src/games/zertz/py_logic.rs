//! Python bindings for Zertz game logic functions
//!
//! This module exposes stateless Zertz game logic functions to Python,
//! allowing Python code to call Rust game logic directly.

use super::action::{PyZertzAction, ZertzAction};
use super::action_result::{PyZertzActionResult, ZertzActionResult};
use super::{action_transform, board::BoardConfig, canonicalization, logic, notation};
use crate::canonicalization_transform_flags::{PyTransformFlags, TransformFlags};
use numpy::{PyArray1, PyArray3, PyArray5, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use std::collections::HashMap;
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
    logic::ax_rot60(q, r, k)
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
    logic::ax_mirror_q_axis(q, r)
}

/// Build bidirectional maps between (y,x) and axial (q,r) coordinates
///
/// Converts all valid board positions to centered, scaled axial coordinates suitable
/// for rotation and reflection transformations. Applies board-specific centering and
/// scaling (48-ring boards use scale=3 for D3 symmetry).
///
/// Args:
///     config: BoardConfig specifying board size and layout
///     layout: 2D boolean array indicating valid positions (width × width)
///
/// Returns:
///     Tuple of two dictionaries: (yx_to_ax, ax_to_yx)
///         - yx_to_ax: Maps (y, x) tuples to (q, r) axial coordinates
///         - ax_to_yx: Maps (q, r) axial coordinates to (y, x) tuples
#[pyfunction]
pub fn build_axial_maps(
    config: &BoardConfig,
    layout: Vec<Vec<bool>>,
) -> (
    HashMap<(i32, i32), (i32, i32)>,
    HashMap<(i32, i32), (i32, i32)>,
) {
    canonicalization::build_axial_maps(config, &layout)
}

// ============================================================================
// Canonicalization
// ============================================================================

/// Canonicalize a board state to its standard form
///
/// Finds the lexicographically minimal representation of the board state
/// under selected symmetry transforms (rotations, reflections, translations).
///
/// Args:
///     spatial_state_state: 3D array of shape (L, H, W) containing board layers
///     config: BoardConfig specifying board size and symmetry group
///     flags: TransformFlags specifying which transforms to include (default: ALL)
///
/// Returns:
///     Tuple of (canonical_spatial_state, transform_name, inverse_transform_name):
///     - canonical_spatial_state: Canonicalized spatial_state array
///     - transform_name: String describing the applied transform (e.g., "R60", "MR120")
///     - inverse_transform_name: String describing the inverse transform
#[pyfunction]
#[pyo3(signature = (config, spatial_state, flags=None))]
pub fn canonicalize_state<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    flags: Option<&PyTransformFlags>,
) -> (Py<PyArray3<f32>>, String, String) {
    let spatial_state = spatial_state.as_array();
    let transform_flags = flags.map(|f| f.inner).unwrap_or(TransformFlags::ALL);
    let (canonical, transform, inverse) =
        canonicalization::canonicalize_internal(config, &spatial_state, transform_flags);
    (
        PyArray3::from_array(py, &canonical).into(),
        transform,
        inverse,
    )
}

/// Transform a board state by a named transform
///
/// Applies a symmetry transformation (rotation, reflection, or combination)
/// to a board state.
///
/// Args:
///     spatial_state: 3D array of shape (L, H, W) containing board layers
///     config: BoardConfig specifying board size
///     rot60_k: Number of 60° rotation steps (0-5)
///     mirror: Whether to apply mirror reflection
///     mirror_first: If true, mirror then rotate; if false, rotate then mirror
///
/// Returns:
///     Transformed spatial_state array
#[pyfunction]
pub fn transform_state<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
) -> PyResult<Py<PyArray3<f32>>> {
    let spatial_state = spatial_state.as_array();
    let transformed = canonicalization::transform_state(
        config,
        &spatial_state,
        rot60_k,
        mirror,
        mirror_first,
        0,
        0,
        true,
    )
    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Transformation failed"))?;
    Ok(PyArray3::from_array(py, &transformed).into())
}

/// Translate a board state by (dy, dx) offset
///
/// .. deprecated::
///     Use :func:`transform_state` with ``rot60_k=0, mirror=False, mirror_first=False``
///     and the desired ``dy, dx`` values instead. This function will be removed in a future version.
///
/// Translates ring and marble data, preserving layout validity.
/// Returns None if translation would move rings off the board.
///
/// Args:
///     spatial_state: 3D array of shape (L, H, W) containing board layers
///     config: BoardConfig specifying board size
///     dy: Translation offset in y direction
///     dx: Translation offset in x direction
///
/// Returns:
///     Translated spatial_state array, or None if translation is invalid
#[pyfunction]
pub fn translate_state<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    dy: i32,
    dx: i32,
) -> PyResult<Option<Py<PyArray3<f32>>>> {
    // Emit deprecation warning
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "translate_state is deprecated. Use transform_state(config, spatial_state, 0, False, False, dy, dx) instead.",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            2, // stacklevel
        ),
    )?;

    let spatial_state = spatial_state.as_array();
    Ok(
        canonicalization::transform_state(config, &spatial_state, 0, false, false, dy, dx, true)
            .map(|result| PyArray3::from_array(py, &result).into()),
    )
}

/// Get bounding box of all remaining rings
///
/// Finds the minimum and maximum y and x coordinates of all positions with rings.
/// Returns None if no rings exist on the board.
///
/// Args:
///     spatial_state: 3D array of shape (L, H, W) containing board layers
///     config: BoardConfig specifying board size
///
/// Returns:
///     Option containing (min_y, max_y, min_x, max_x) tuple, or None if no rings
#[pyfunction]
pub fn get_bounding_box(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<f32>,
) -> Option<(usize, usize, usize, usize)> {
    let spatial_state = spatial_state.as_array();
    canonicalization::bounding_box(config, &spatial_state)
}

/// Get all valid translation offsets for the current board state.
///
/// Tests each potential translation by attempting to apply it, only including
/// translations that successfully keep all rings within valid board positions.
///
/// Args:
///     spatial_state: 3D array of shape (L, H, W) containing board state
///     config: BoardConfig specifying board size
///
/// Returns:
///     List of (name, dy, dx) tuples for valid translations
///
/// Example:
///     >>> config = BoardConfig.standard_config(37)
///     >>> state = np.zeros((layers, 7, 7))
///     >>> translations = get_translations(state, config)
///     >>> translations[0]
///     ('T0,0', 0, 0)
#[pyfunction]
pub fn get_translations(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<f32>,
) -> Vec<(String, i32, i32)> {
    let spatial_state = spatial_state.as_array();
    canonicalization::get_translations(config, &spatial_state)
}

/// Compute canonical key for lexicographic comparison
///
/// Returns a byte vector representing the board state over valid positions only.
/// Used for finding the lexicographically minimal state representation.
///
/// Args:
///     config: BoardConfig specifying board size
///     spatial_state: 3D array of shape (L, H, W) containing board layers
///
/// Returns:
///     Bytes object containing the canonical key
#[pyfunction]
pub fn canonical_key(config: &BoardConfig, spatial_state: PyReadonlyArray3<f32>) -> Vec<u8> {
    let spatial_state = spatial_state.as_array();
    canonicalization::compute_canonical_key(config, &spatial_state)
}

/// Compute the inverse of a transform name
///
/// Given a transform name like "R60", "MR120", or "T1,0_R180M", computes
/// the inverse transform that undoes the original operation.
///
/// Args:
///     transform_name: String describing the transform (e.g., "R60", "MR120")
///
/// Returns:
///     String describing the inverse transform
///
/// Examples:
///     >>> inverse_transform_name("R60")
///     "R300"
///     >>> inverse_transform_name("MR120")
///     "R240M"
///     >>> inverse_transform_name("T1,0_R180M")
///     "MR180_T-1,0"
#[pyfunction]
pub fn inverse_transform_name(transform_name: &str) -> String {
    canonicalization::inverse_transform_name(transform_name)
}

// ============================================================================

/// Check if (y, x) coordinates are within board bounds
#[pyfunction]
pub fn is_inbounds(y: i32, x: i32, width: usize) -> bool {
    logic::is_inbounds(y, x, width)
}

/// Get list of neighboring indices (filtered to in-bounds only)
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_neighbors(config: &BoardConfig, y: usize, x: usize) -> Vec<(usize, usize)> {
    logic::get_neighbors(config, y, x).into_iter().collect()
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
    logic::get_jump_destination(start_y, start_x, cap_y, cap_x)
}

/// Compute capture destination from source position and direction
///
/// Given a source position and direction index, computes the landing position
/// after a jump capture (2 steps in the given direction).
///
/// Args:
///     src_y: Source row
///     src_x: Source column
///     dir_idx: Direction index (0-5 for hexagonal directions)
///     config: Board configuration
///
/// Returns:
///     Tuple of (dest_y, dest_x) if destination is in bounds, or None
#[pyfunction]
pub fn get_capture_destination(
    config: &BoardConfig,
    src_y: usize,
    src_x: usize,
    dir_idx: usize,
) -> Option<(usize, usize)> {
    logic::get_capture_destination(config, src_y, src_x, dir_idx)
}

/// Find all connected regions on the board
/// Returns list of regions, where each region is a list of (y, x) tuples
#[pyfunction]
pub fn get_regions<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
) -> Vec<Vec<(usize, usize)>> {
    let spatial_state = spatial_state.as_array();
    logic::get_regions(config, &spatial_state)
}

/// Get list of empty ring indices across the board
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_open_rings<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
) -> Vec<(usize, usize)> {
    let spatial_state = spatial_state.as_array();
    logic::get_open_rings(config, &spatial_state)
}

/// Check if ring can be removed (geometric rule)
#[pyfunction]
pub fn is_ring_removable<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
) -> bool {
    let spatial_state = spatial_state.as_array();
    logic::is_ring_removable(config, &spatial_state, y, x)
}

/// Get removable rings (rings that can be removed without disconnecting board)
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_removable_rings<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
) -> Vec<(usize, usize)> {
    let spatial_state = spatial_state.as_array();
    logic::get_removable_rings(&config, &spatial_state)
}

/// Get global_state index for marble type in supply
#[pyfunction]
pub fn get_supply_index(config: &BoardConfig, marble_type: char) -> usize {
    logic::get_supply_index(&config, marble_type)
}

/// Get global_state index for captured marble
#[pyfunction]
pub fn get_captured_index(config: &BoardConfig, player: usize, marble_type: char) -> usize {
    // Captured indices: P1: W=3, G=4, B=5; P2: W=6, G=7, B=8
    // Player constants: PLAYER_1 = 0, PLAYER_2 = 1
    let marble_idx = match marble_type {
        'w' => 0,
        'g' => 1,
        'b' => 2,
        _ => panic!("Invalid marble type: {}", marble_type),
    };

    logic::get_captured_index(config, player, marble_idx)
}

/// Get marble type at given position
/// Returns: 'w', 'g', 'b', or '\0' (none)
#[pyfunction]
pub fn get_marble_type_at<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
) -> char {
    let spatial_state = spatial_state.as_array();
    logic::get_marble_type_at(config, &spatial_state, y, x)
}

/// Get valid placement actions
/// Returns Array5<f32> with shape (3, width, width, width, width)
#[pyfunction]
pub fn get_placement_actions<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
) -> Py<PyArray5<f32>> {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let result = logic::get_placement_actions(config, &spatial_state, &global_state);
    PyArray5::from_array(py, &result).into()
}

/// Get valid capture actions
/// Returns Array3<f32> with shape (6, width, width)
#[pyfunction]
pub fn get_capture_actions<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
) -> Py<PyArray3<f32>> {
    let spatial_state = spatial_state.as_array();
    let result = logic::get_capture_actions(config, &spatial_state);
    PyArray3::from_array(py, &result).into()
}

/// Get valid actions (both placement and capture)
/// Returns tuple of (placement_mask, capture_mask)
#[pyfunction]
pub fn get_valid_actions<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
) -> (Py<PyArray5<f32>>, Py<PyArray3<f32>>) {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let (placement, capture) = logic::get_valid_actions(config, &spatial_state, &global_state);
    (
        PyArray5::from_array(py, &placement).into(),
        PyArray3::from_array(py, &capture).into(),
    )
}

#[pyfunction]
// #[pyo3(signature = (config, spatial_state, global_state, action))]
pub fn apply_action<'py>(
    config: &BoardConfig,
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    action: &PyZertzAction,
) -> PyResult<PyZertzActionResult> {
    let result = match &action.inner {
        ZertzAction::Placement {
            marble_type,
            dst_flat,
            remove_flat,
        } => {
            let isolation_captures = unsafe {
                let mut spatial_state_arr = spatial_state.as_array_mut();
                let mut global_state_arr = global_state.as_array_mut();
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                let (rem_y, rem_x) = config.flat_to_optional_yx(*remove_flat);
                logic::apply_placement(
                    config,
                    &mut spatial_state_arr,
                    &mut global_state_arr,
                    *marble_type,
                    dst_y,
                    dst_x,
                    rem_y,
                    rem_x,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?
            };
            super::action_result::ZertzActionResult::placement(isolation_captures)
        }
        ZertzAction::Capture { src_flat, dst_flat } => {
            let (src_y, src_x) = config.flat_to_yx(*src_flat);
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            // Calculate the captured marble position (midpoint between src and dst)
            let cap_y = (src_y + dst_y) / 2;
            let cap_x = (src_x + dst_x) / 2;

            // Get marble type before applying the capture
            // Layers: 0=ring, 1=white, 2=gray, 3=black
            // Find which layer (1-3) has a marble, then map to type (0-2)
            let marble_type = unsafe {
                let spatial_state_arr = spatial_state.as_array();
                (1..=3)
                    .position(|layer| spatial_state_arr[[layer, cap_y, cap_x]] > 0.5)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "No marble found at capture position ({}, {})",
                            cap_y, cap_x
                        ))
                    })?
            };

            unsafe {
                let mut spatial_state_arr = spatial_state.as_array_mut();
                let mut global_state_arr = global_state.as_array_mut();

                logic::apply_capture(
                    config,
                    &mut spatial_state_arr,
                    &mut global_state_arr,
                    src_y,
                    src_x,
                    dst_y,
                    dst_x,
                );
            }

            super::action_result::ZertzActionResult::capture(marble_type, cap_y, cap_x)
        }
        ZertzAction::Pass => super::action_result::ZertzActionResult::pass(),
    };

    Ok(PyZertzActionResult { inner: result })
}

/// Apply a placement action (mutates arrays in-place)
///
/// Returns list of captured marble positions from isolation as (marble_layer, y, x) tuples
///
/// # Deprecated
/// This function is deprecated. Use `apply_placement_action()` with a `ZertzAction` instead.
#[pyfunction]
#[deprecated(
    since = "0.1.0",
    note = "Use apply_placement_action() with ZertzAction instead"
)]
pub fn apply_placement_action_old<'py>(
    config: &BoardConfig,
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    marble_type: usize,
    dst_y: usize,
    dst_x: usize,
    remove_y: Option<usize>,
    remove_x: Option<usize>,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let captured_marbles = unsafe {
        let mut spatial_state_arr = spatial_state.as_array_mut();
        let mut global_state_arr = global_state.as_array_mut();
        logic::apply_placement(
            config,
            &mut spatial_state_arr,
            &mut global_state_arr,
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?
    };
    Ok(captured_marbles)
}

/// Apply a capture action (mutates arrays in-place)
///
/// # Deprecated
/// This function is deprecated. Use `apply_capture_action()` with a `ZertzAction` instead.
#[pyfunction]
#[deprecated(
    since = "0.1.0",
    note = "Use apply_capture_action() with ZertzAction instead"
)]
pub fn apply_capture_action_old<'py>(
    config: &BoardConfig,
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    start_y: usize,
    start_x: usize,
    dest_y: usize,
    dest_x: usize,
) -> PyResult<()> {
    unsafe {
        let mut spatial_state_arr = spatial_state.as_array_mut();
        let mut global_state_arr = global_state.as_array_mut();
        logic::apply_capture(
            config,
            &mut spatial_state_arr,
            &mut global_state_arr,
            start_y,
            start_x,
            dest_y,
            dest_x,
        );
    }
    Ok(())
}

/// Apply a placement action (mutates arrays in-place)
///
/// Accepts a ZertzAction which must be a Placement variant.
/// Returns a ZertzActionResult containing isolation capture information.
#[pyfunction]
pub fn apply_placement_action<'py>(
    config: &BoardConfig,
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    action: &PyZertzAction,
) -> PyResult<PyZertzActionResult> {
    // Extract placement data from action
    match &action.inner {
        ZertzAction::Placement {
            marble_type,
            dst_flat,
            remove_flat,
        } => {
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
            let (remove_y, remove_x) = config.flat_to_optional_yx(*remove_flat);

            let isolation_captures = unsafe {
                let mut spatial_state_arr = spatial_state.as_array_mut();
                let mut global_state_arr = global_state.as_array_mut();
                logic::apply_placement(
                    config,
                    &mut spatial_state_arr,
                    &mut global_state_arr,
                    *marble_type,
                    dst_y,
                    dst_x,
                    remove_y,
                    remove_x,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?
            };

            let result = ZertzActionResult::placement(isolation_captures);
            Ok(PyZertzActionResult { inner: result })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Action must be a Placement variant for apply_placement_action",
        )),
    }
}

/// Apply a capture action (mutates arrays in-place)
///
/// Accepts a ZertzAction which must be a Capture variant.
/// Returns a ZertzActionResult containing captured marble information.
#[pyfunction]
pub fn apply_capture_action<'py>(
    config: &BoardConfig,
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    action: &PyZertzAction,
) -> PyResult<PyZertzActionResult> {
    // Extract capture data from action
    match &action.inner {
        ZertzAction::Capture { src_flat, dst_flat } => {
            let (start_y, start_x) = config.flat_to_yx(*src_flat);
            let (dest_y, dest_x) = config.flat_to_yx(*dst_flat);

            // Calculate captured marble position (midpoint between start and dest)
            let cap_y = (start_y + dest_y) / 2;
            let cap_x = (start_x + dest_x) / 2;

            // Extract marble type at captured position before applying the action
            let captured_marble_type = unsafe {
                let spatial_state_arr = spatial_state.as_array();
                // Find which marble layer has a marble at this position
                (1..4)
                    .find(|&layer| spatial_state_arr[[layer, cap_y, cap_x]] > 0.0)
                    .map(|layer| layer - 1) // Convert layer to marble type (0=white, 1=gray, 2=black)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "No marble found at capture position ({}, {})",
                            cap_y, cap_x
                        ))
                    })?
            };

            unsafe {
                let mut spatial_state_arr = spatial_state.as_array_mut();
                let mut global_state_arr = global_state.as_array_mut();
                logic::apply_capture(
                    config,
                    &mut spatial_state_arr,
                    &mut global_state_arr,
                    start_y,
                    start_x,
                    dest_y,
                    dest_x,
                );
            }

            let result = ZertzActionResult::capture(captured_marble_type, cap_y, cap_x);
            Ok(PyZertzActionResult { inner: result })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "Action must be a Capture variant for apply_capture_action",
        )),
    }
}

/// Check if game is over (any terminal condition)
#[pyfunction]
pub fn is_game_over<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
) -> bool {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    logic::is_game_over(config, &spatial_state, &global_state)
}

/// Get game outcome from Player 1's perspective
/// Returns: 1 (P1 wins), -1 (P2 wins), 0 (tie), -2 (both lose)
#[pyfunction]
pub fn get_game_outcome<'py>(
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
) -> i8 {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    logic::get_game_outcome(config, &spatial_state, &global_state)
}

/// Check for isolated regions and capture marbles
/// Returns tuple of (updated_spatial_state, updated_global_state, captured_marbles_list)
/// where captured_marbles_list contains tuples of (marble_layer_idx, y, x)
#[pyfunction]
pub fn check_for_isolation_capture<'py>(
    py: Python<'py>,
    config: &BoardConfig,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
) -> (
    Py<PyArray3<f32>>,
    Py<PyArray1<f32>>,
    Vec<(usize, usize, usize)>,
) {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let (spatial_state_out, global_state_out, captured_marbles) =
        logic::check_for_isolation_capture(config, &spatial_state, &global_state);
    (
        PyArray3::from_array(py, &spatial_state_out).into(),
        PyArray1::from_array(py, &global_state_out).into(),
        captured_marbles,
    )
}

// ============================================================================
// Algebraic Notation
// ============================================================================

/// Convert array coordinates (y, x) to algebraic notation (e.g., "D4")
///
/// Args:
///     config: BoardConfig specifying board size
///     y: Row index (0 = top, increases downward)
///     x: Column index (0 = leftmost column)
///
/// Returns:
///     Algebraic notation string (e.g., "A1", "D4", "G7")
///
/// Raises:
///     ValueError: If coordinates are out of bounds
///
/// Examples:
///     >>> coordinate_to_algebraic(6, 0, config)  # 7x7 board
///     'A1'
///     >>> coordinate_to_algebraic(3, 3, config)
///     'D4'
#[pyfunction]
pub fn coordinate_to_algebraic(config: &BoardConfig, y: usize, x: usize) -> PyResult<String> {
    notation::coordinate_to_algebraic_with_config(y, x, config)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Parse algebraic notation (e.g., "A1") to array coordinates (y, x)
///
/// Args:
///     config: BoardConfig specifying board size
///     notation: Algebraic notation string (e.g., "D4", "A1")
///               Case-insensitive
///
/// Returns:
///     Tuple of (y, x) array coordinates
///
/// Raises:
///     ValueError: If notation is invalid or out of bounds
///
/// Examples:
///     >>> algebraic_to_coordinate("A1", config)  # 7x7 board
///     (6, 0)
///     >>> algebraic_to_coordinate("d4", config)  # Case-insensitive
///     (3, 3)
#[pyfunction]
pub fn algebraic_to_coordinate(config: &BoardConfig, notation: &str) -> PyResult<(usize, usize)> {
    notation::algebraic_to_coordinate_with_config(notation, config)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn generate_standard_layout_mask<'py>(
    py: Python<'py>,
    rings: usize,
    width: usize,
) -> PyResult<Py<numpy::PyArray2<bool>>> {
    let layout = canonicalization::generate_standard_layout_mask(rings, width)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert Vec<Vec<bool>> to numpy array
    Ok(numpy::PyArray2::from_vec2(py, &layout)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to create numpy array: {}", e))
        })?
        .into())
}

// ============================================================================
// Action Transformation
// ============================================================================

/// Transform an action using symmetry operations
///
/// Applies rotation, mirror, and/or translation transforms to an action tuple.
/// This is used for action canonicalization and replay with symmetries.
///
/// Args:
///     config: BoardConfig specifying board size
///     action_type: Action type string ("PUT", "CAP", or "PASS")
///     action_data: Action data tuple:
///         - For PUT: (marble_idx, put_flat, rem_flat) where rem_flat = width² means no removal
///         - For CAP: (direction_idx, y, x)
///         - For PASS: empty tuple ()
///     transform: Transform string (e.g., "R60", "MR120", "T1,0_R180M")
///
/// Returns:
///     Tuple of (action_type, action_data) in same format as input
///
/// Examples:
///     >>> transform_action(config, "PUT", (0, 10, 15), "R60")
///     ("PUT", (0, 12, 17))
///     >>> transform_action_old(config, "CAP", (0, 3, 3), "MR120")
///     ("CAP", (2, 4, 2))
///     >>> transform_action_old(config, "PASS", (), "R60")
///     ("PASS", ())
///
/// # Deprecated
/// This function is deprecated. Use `transform_action()` with a `ZertzAction` instead.
#[pyfunction]
#[deprecated(
    since = "0.1.0",
    note = "Use transform_action() with ZertzAction instead"
)]
pub fn transform_action_old(
    config: &BoardConfig,
    action_type: &str,
    action_data: &Bound<'_, pyo3::types::PyTuple>,
    transform: &str,
) -> PyResult<(String, Vec<Option<usize>>)> {
    // Convert Python tuple format to ZertzAction
    let action = match action_type {
        "PUT" => {
            if action_data.len() != 5 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "PUT action requires 5 elements, got {}",
                    action_data.len()
                )));
            }
            let marble_type: usize = action_data.get_item(0)?.extract()?;
            let dst_y: usize = action_data.get_item(1)?.extract()?;
            let dst_x: usize = action_data.get_item(2)?.extract()?;
            let rem_y: usize = action_data.get_item(3)?.extract()?;
            let rem_x: usize = action_data.get_item(4)?.extract()?;

            let dst_flat = config.yx_to_flat(dst_y, dst_x);

            // Convert sentinel to None
            let remove_flat = if (rem_y, rem_x) == (dst_y, dst_x) {
                None
            } else {
                Some(config.yx_to_flat(rem_y, rem_x))
            };

            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            }
        }
        "CAP" => {
            if action_data.len() != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "CAP action requires 3 elements, got {}",
                    action_data.len()
                )));
            }
            let direction_idx: usize = action_data.get_item(0)?.extract()?;
            let y: usize = action_data.get_item(1)?.extract()?;
            let x: usize = action_data.get_item(2)?.extract()?;

            // Convert direction + coordinates to start/dest flat indices
            // Get direction vector from config
            let directions = config.get_directions();
            if direction_idx >= directions.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid direction index: {}",
                    direction_idx
                )));
            }
            let (dy, dx) = directions[direction_idx];

            // Calculate capture and landing positions
            let cap_y = (y as i32 + dy) as usize;
            let cap_x = (x as i32 + dx) as usize;
            let (dst_y, dst_x) = logic::get_jump_destination(y, x, cap_y, cap_x);

            if dst_y < 0 || dst_x < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Jump destination out of bounds",
                ));
            }

            let src_flat = config.yx_to_flat(y, x);
            let dst_flat = config.yx_to_flat(dst_y as usize, dst_x as usize);

            ZertzAction::Capture { src_flat, dst_flat }
        }
        "PASS" => ZertzAction::Pass,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid action type: {}",
                action_type
            )));
        }
    };

    // Call Rust transform_action
    let transformed = action_transform::transform_action(&action, transform, config);

    // Convert back to Python tuple format
    match transformed {
        ZertzAction::Placement {
            marble_type,
            dst_flat,
            remove_flat,
        } => match transformed.to_placement_action(config) {
            Some(p) => Ok(p),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid placement action conversion",
            )),
        },
        ZertzAction::Capture { src_flat, dst_flat } => {
            // Convert start/dest flat indices back to direction + coordinates
            let (start_y, start_x) = config.flat_to_yx(*src_flat);
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            // Calculate direction
            let dy = dst_y as i32 - start_y as i32;
            let dx = dst_x as i32 - start_x as i32;

            // Divide by 2 to get direction vector (since jump is 2 steps)
            let dir_dy = dy / 2;
            let dir_dx = dx / 2;

            // Find direction index
            let directions = config.get_directions();
            let direction_idx = directions
                .iter()
                .position(|&(dy, dx)| dy == dir_dy && dx == dir_dx)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Could not find direction for ({}, {})",
                        dir_dy, dir_dx
                    ))
                })?;

            Ok((
                "CAP".to_string(),
                vec![Some(direction_idx), Some(start_y), Some(start_x)],
            ))
        }
        ZertzAction::Pass => Ok(("PASS".to_string(), vec![])),
    }
}

/// Transform an action using symmetry operations
///
/// Applies rotation, mirror, and/or translation transforms to a ZertzAction.
/// This is used for action canonicalization and replay with symmetries.
///
/// Args:
///     config: BoardConfig specifying board size
///     action: ZertzAction to transform
///     transform: Transform string (e.g., "R60", "MR120", "T1,0_R180M")
///
/// Returns:
///     Transformed ZertzAction
///
/// Examples:
///     >>> action = ZertzAction.placement(config, 0, 3, 3, None, None)
///     >>> transformed = transform_action(config, action, "R60")
#[pyfunction]
pub fn transform_action(
    config: &BoardConfig,
    action: &PyZertzAction,
    transform: &str,
) -> PyResult<PyZertzAction> {
    // Call Rust transform_action
    let transformed = action_transform::transform_action(&action.inner, transform, config);
    Ok(PyZertzAction { inner: transformed })
}

/// Register all game logic functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Axial coordinate transformations
    m.add_function(wrap_pyfunction!(ax_rot60, m)?)?;
    m.add_function(wrap_pyfunction!(ax_mirror_q_axis, m)?)?;
    m.add_function(wrap_pyfunction!(build_axial_maps, m)?)?;

    // Canonicalization
    m.add_function(wrap_pyfunction!(canonicalize_state, m)?)?;
    m.add_function(wrap_pyfunction!(transform_state, m)?)?;
    m.add_function(wrap_pyfunction!(translate_state, m)?)?;
    m.add_function(wrap_pyfunction!(get_bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(get_translations, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_transform_name, m)?)?;
    m.add_function(wrap_pyfunction!(generate_standard_layout_mask, m)?)?;

    // Utility functions
    m.add_function(wrap_pyfunction!(is_inbounds, m)?)?;
    m.add_function(wrap_pyfunction!(get_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(get_jump_destination, m)?)?;
    m.add_function(wrap_pyfunction!(get_capture_destination, m)?)?;
    m.add_function(wrap_pyfunction!(get_regions, m)?)?;
    m.add_function(wrap_pyfunction!(get_open_rings, m)?)?;
    m.add_function(wrap_pyfunction!(is_ring_removable, m)?)?;
    m.add_function(wrap_pyfunction!(get_removable_rings, m)?)?;
    m.add_function(wrap_pyfunction!(get_supply_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_captured_index, m)?)?;
    m.add_function(wrap_pyfunction!(get_marble_type_at, m)?)?;

    // Move generation and action execution
    m.add_function(wrap_pyfunction!(get_placement_actions, m)?)?;
    m.add_function(wrap_pyfunction!(get_capture_actions, m)?)?;
    m.add_function(wrap_pyfunction!(get_valid_actions, m)?)?;
    m.add_function(wrap_pyfunction!(apply_action, m)?)?;
    m.add_function(wrap_pyfunction!(apply_placement_action, m)?)?;
    m.add_function(wrap_pyfunction!(apply_capture_action, m)?)?;
    // Old deprecated versions
    m.add_function(wrap_pyfunction!(apply_placement_action_old, m)?)?;
    m.add_function(wrap_pyfunction!(apply_capture_action_old, m)?)?;

    // Game termination
    m.add_function(wrap_pyfunction!(is_game_over, m)?)?;
    m.add_function(wrap_pyfunction!(get_game_outcome, m)?)?;

    // Isolation capture
    m.add_function(wrap_pyfunction!(check_for_isolation_capture, m)?)?;

    // Algebraic notation
    m.add_function(wrap_pyfunction!(coordinate_to_algebraic, m)?)?;
    m.add_function(wrap_pyfunction!(algebraic_to_coordinate, m)?)?;

    // Action transformation
    m.add_function(wrap_pyfunction!(transform_action, m)?)?;
    m.add_function(wrap_pyfunction!(transform_action_old, m)?)?;

    Ok(())
}
