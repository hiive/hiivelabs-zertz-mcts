//! Python bindings for Zertz game logic functions
//!
//! This module exposes stateless Zertz game logic functions to Python,
//! allowing Python code to call Rust game logic directly.

use super::{board::BoardConfig, canonicalization, logic, notation};
use canonicalization::TransformFlags as RustTransformFlags;
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
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
) -> (HashMap<(i32, i32), (i32, i32)>, HashMap<(i32, i32), (i32, i32)>) {
    canonicalization::build_axial_maps(config, &layout)
}

// ============================================================================
// Transform Flags
// ============================================================================

/// Flags for controlling which transforms to use in canonicalization.
///
/// TransformFlags uses bit flags to specify which types of symmetries to include:
/// - ROTATION: Include rotational symmetries (0°, 60°, 120°, etc.)
/// - MIRROR: Include reflection symmetries
/// - TRANSLATION: Include translational symmetries
///
/// Common combinations:
/// - ALL: rotation + mirror + translation (full canonicalization)
/// - ROTATION_MIRROR: rotation + mirror only (canonical orientation, no translation)
/// - NONE: identity only (no transforms)
#[pyclass]
#[derive(Clone)]
pub struct TransformFlags {
    inner: RustTransformFlags,
}


#[pymethods]
impl TransformFlags {
    /// All transforms enabled (rotation + mirror + translation)
    #[classattr]
    #[allow(non_snake_case)]
    fn ALL() -> Self {
        TransformFlags { inner: RustTransformFlags::ALL }
    }

    /// Only rotational symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn ROTATION() -> Self {
        TransformFlags { inner: RustTransformFlags::ROTATION }
    }

    /// Only mirror symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn MIRROR() -> Self {
        TransformFlags { inner: RustTransformFlags::MIRROR }
    }

    /// Only translation symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn TRANSLATION() -> Self {
        TransformFlags { inner: RustTransformFlags::TRANSLATION }
    }

    /// Rotation and mirror only (no translation)
    #[classattr]
    #[allow(non_snake_case)]
    fn ROTATION_MIRROR() -> Self {
        TransformFlags { inner: RustTransformFlags::ROTATION_MIRROR }
    }

    /// No transforms (identity only)
    #[classattr]
    #[allow(non_snake_case)]
    fn NONE() -> Self {
        TransformFlags { inner: RustTransformFlags::NONE }
    }

    /// Create TransformFlags from bit flags
    ///
    /// Args:
    ///     bits: Bit flags (0-7). Use constants like TransformFlags.ALL,
    ///           TransformFlags.ROTATION_MIRROR, etc.
    ///
    /// Returns:
    ///     TransformFlags instance
    ///
    /// Raises:
    ///     ValueError: If bits > 7
    #[new]
    pub fn new(bits: u8) -> PyResult<Self> {
        RustTransformFlags::from_bits(bits)
            .map(|inner| TransformFlags { inner })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid transform flags bits: {} (must be 0-7)",
                    bits
                ))
            })
    }

    /// Get the raw bit flags
    pub fn bits(&self) -> u8 {
        self.inner.bits()
    }

    /// Check if rotation flag is set
    pub fn has_rotation(&self) -> bool {
        self.inner.has_rotation()
    }

    /// Check if mirror flag is set
    pub fn has_mirror(&self) -> bool {
        self.inner.has_mirror()
    }

    /// Check if translation flag is set
    pub fn has_translation(&self) -> bool {
        self.inner.has_translation()
    }

    fn __repr__(&self) -> String {
        let bits = self.inner.bits();
        let names = match bits {
            0b111 => "ALL",
            0b011 => "ROTATION_MIRROR",
            0b001 => "ROTATION",
            0b010 => "MIRROR",
            0b100 => "TRANSLATION",
            0b000 => "NONE",
            _ => return format!("TransformFlags({})", bits),
        };
        format!("TransformFlags.{}", names)
    }

    fn __eq__(&self, other: &TransformFlags) -> bool {
        self.inner == other.inner
    }
}

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
#[pyo3(signature = (spatial_state, config, flags=None))]
pub fn canonicalize_state<'py>(
    py: Python<'py>,
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
    flags: Option<&TransformFlags>,
) -> (Py<PyArray3<f32>>, String, String) {
    let spatial_state = spatial_state.as_array();
    let transform_flags = flags.map(|f| f.inner).unwrap_or(RustTransformFlags::ALL);
    let (canonical, transform, inverse) = canonicalization::canonicalize_internal(&spatial_state, config, transform_flags);
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
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
) -> PyResult<Py<PyArray3<f32>>> {
    let spatial_state = spatial_state.as_array();
    let transformed =
        canonicalization::transform_state(&spatial_state, config, rot60_k, mirror, mirror_first, 0, 0)
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
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
    dy: i32,
    dx: i32,
) -> PyResult<Option<Py<PyArray3<f32>>>> {
    // Emit deprecation warning
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            "translate_state is deprecated. Use transform_state(spatial_state, config, 0, False, False, dy, dx) instead.",
            py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
            2, // stacklevel
        ),
    )?;

    let spatial_state = spatial_state.as_array();
    Ok(canonicalization::transform_state(&spatial_state, config, 0, false, false, dy, dx)
        .map(|result| PyArray3::from_array(py, &result).into()))
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
    spatial_state: PyReadonlyArray3<f32>,
    config: &BoardConfig,
) -> Option<(usize, usize, usize, usize)> {
    let spatial_state = spatial_state.as_array();
    canonicalization::bounding_box(&spatial_state, config)
}

/// Compute canonical key for lexicographic comparison
///
/// Returns a byte vector representing the board state over valid positions only.
/// Used for finding the lexicographically minimal state representation.
///
/// Args:
///     spatial_state: 3D array of shape (L, H, W) containing board layers
///     config: BoardConfig specifying board size
///
/// Returns:
///     Bytes object containing the canonical key
#[pyfunction]
pub fn canonical_key(spatial_state: PyReadonlyArray3<f32>, config: &BoardConfig) -> Vec<u8> {
    let spatial_state = spatial_state.as_array();
    canonicalization::compute_canonical_key(&spatial_state, config)
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
pub fn get_neighbors(y: usize, x: usize, config: &BoardConfig) -> Vec<(usize, usize)> {
    logic::get_neighbors(y, x, config).into_iter().collect()
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

/// Find all connected regions on the board
/// Returns list of regions, where each region is a list of (y, x) tuples
#[pyfunction]
pub fn get_regions<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<Vec<(usize, usize)>> {
    let spatial_state = spatial_state.as_array();
    logic::get_regions(&spatial_state, config)
}

/// Get list of empty ring indices across the board
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_open_rings<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<(usize, usize)> {
    let spatial_state = spatial_state.as_array();
    logic::get_open_rings(&spatial_state, config)
}

/// Check if ring can be removed (geometric rule)
#[pyfunction]
pub fn is_ring_removable<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
    config: &BoardConfig,
) -> bool {
    let spatial_state = spatial_state.as_array();
    logic::is_ring_removable(&spatial_state, y, x, config)
}

/// Get removable rings (rings that can be removed without disconnecting board)
/// Returns list of (y, x) tuples
#[pyfunction]
pub fn get_removable_rings<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Vec<(usize, usize)> {
    let spatial_state = spatial_state.as_array();
    logic::get_removable_rings(&spatial_state, config)
}

/// Get global_state index for marble type in supply
#[pyfunction]
pub fn get_supply_index(marble_type: char) -> usize {
    let config = BoardConfig::standard(37, 1).unwrap(); // Default config for Python interface
    logic::get_supply_index(marble_type, &config)
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
    spatial_state: PyReadonlyArray3<'py, f32>,
    y: usize,
    x: usize,
) -> char {
    let spatial_state = spatial_state.as_array();
    let config = BoardConfig::standard(37, 1).unwrap(); // Default config for Python interface
    logic::get_marble_type_at(&spatial_state, y, x, &config)
}

/// Get valid placement actions
/// Returns Array3<f32> with shape (3, width², width²+1)
#[pyfunction]
pub fn get_placement_moves<'py>(
    py: Python<'py>,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> Py<PyArray3<f32>> {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let result = logic::get_placement_actions(&spatial_state, &global_state, config);
    PyArray3::from_array(py, &result).into()
}

/// Get valid capture actions
/// Returns Array3<f32> with shape (6, width, width)
#[pyfunction]
pub fn get_capture_moves<'py>(
    py: Python<'py>,
    spatial_state: PyReadonlyArray3<'py, f32>,
    config: &BoardConfig,
) -> Py<PyArray3<f32>> {
    let spatial_state = spatial_state.as_array();
    let result = logic::get_capture_actions(&spatial_state, config);
    PyArray3::from_array(py, &result).into()
}

/// Get valid actions (both placement and capture)
/// Returns tuple of (placement_mask, capture_mask)
#[pyfunction]
pub fn get_valid_actions<'py>(
    py: Python<'py>,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> (Py<PyArray3<f32>>, Py<PyArray3<f32>>) {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let (placement, capture) = logic::get_valid_actions(&spatial_state, &global_state, config);
    (PyArray3::from_array(py, &placement).into(), PyArray3::from_array(py, &capture).into())
}

/// Apply a placement action (mutates arrays in-place)
/// Returns list of captured marble positions from isolation as (marble_layer, y, x) tuples
#[pyfunction]
pub fn apply_placement_action<'py>(
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    marble_type: usize,
    dst_y: usize,
    dst_x: usize,
    remove_y: Option<usize>,
    remove_x: Option<usize>,
    config: &BoardConfig,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let captured_marbles = unsafe {
        let mut spatial_state_arr = spatial_state.as_array_mut();
        let mut global_state_arr = global_state.as_array_mut();
        logic::apply_placement(
            &mut spatial_state_arr,
            &mut global_state_arr,
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            config,
        )
    };
    Ok(captured_marbles)
}

/// Apply a capture action (mutates arrays in-place)
#[pyfunction]
pub fn apply_capture_action<'py>(
    spatial_state: &Bound<'py, PyArray3<f32>>,
    global_state: &Bound<'py, PyArray1<f32>>,
    start_y: usize,
    start_x: usize,
    dest_y: usize,
    dest_x: usize,
    config: &BoardConfig,
) -> PyResult<()> {
    unsafe {
        let mut spatial_state_arr = spatial_state.as_array_mut();
        let mut global_state_arr = global_state.as_array_mut();
        logic::apply_capture(
            &mut spatial_state_arr,
            &mut global_state_arr,
            start_y,
            start_x,
            dest_y,
            dest_x,
            config,
        );
    }
    Ok(())
}

/// Check if game is over (any terminal condition)
#[pyfunction]
pub fn is_game_over<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> bool {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    logic::is_game_over(&spatial_state, &global_state, config)
}

/// Get game outcome from Player 1's perspective
/// Returns: 1 (P1 wins), -1 (P2 wins), 0 (tie), -2 (both lose)
#[pyfunction]
pub fn get_game_outcome<'py>(
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> i8 {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    logic::get_game_outcome(&spatial_state, &global_state, config)
}

/// Check for isolated regions and capture marbles
/// Returns tuple of (updated_spatial_state, updated_global_state, captured_marbles_list)
/// where captured_marbles_list contains tuples of (marble_layer_idx, y, x)
#[pyfunction]
pub fn check_for_isolation_capture<'py>(
    py: Python<'py>,
    spatial_state: PyReadonlyArray3<'py, f32>,
    global_state: PyReadonlyArray1<'py, f32>,
    config: &BoardConfig,
) -> (Py<PyArray3<f32>>, Py<PyArray1<f32>>, Vec<(usize, usize, usize)>) {
    let spatial_state = spatial_state.as_array();
    let global_state = global_state.as_array();
    let (spatial_state_out, global_state_out, captured_marbles) =
        logic::check_for_isolation_capture(&spatial_state, &global_state, config);
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
///     y: Row index (0 = top, increases downward)
///     x: Column index (0 = leftmost column)
///     config: BoardConfig specifying board size
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
pub fn coordinate_to_algebraic(y: usize, x: usize, config: &BoardConfig) -> PyResult<String> {
    notation::coordinate_to_algebraic_with_config(y, x, config)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Parse algebraic notation (e.g., "A1") to array coordinates (y, x)
///
/// Args:
///     notation: Algebraic notation string (e.g., "D4", "A1")
///               Case-insensitive
///     config: BoardConfig specifying board size
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
pub fn algebraic_to_coordinate(notation: &str, config: &BoardConfig) -> PyResult<(usize, usize)> {
    notation::algebraic_to_coordinate_with_config(notation, config)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Register all game logic functions with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Axial coordinate transformations
    m.add_function(wrap_pyfunction!(ax_rot60, m)?)?;
    m.add_function(wrap_pyfunction!(ax_mirror_q_axis, m)?)?;
    m.add_function(wrap_pyfunction!(build_axial_maps, m)?)?;

    // Transform flags
    m.add_class::<TransformFlags>()?;

    // Canonicalization
    m.add_function(wrap_pyfunction!(canonicalize_state, m)?)?;
    m.add_function(wrap_pyfunction!(transform_state, m)?)?;
    m.add_function(wrap_pyfunction!(translate_state, m)?)?;
    m.add_function(wrap_pyfunction!(get_bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(inverse_transform_name, m)?)?;

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

    // Algebraic notation
    m.add_function(wrap_pyfunction!(coordinate_to_algebraic, m)?)?;
    m.add_function(wrap_pyfunction!(algebraic_to_coordinate, m)?)?;

    Ok(())
}