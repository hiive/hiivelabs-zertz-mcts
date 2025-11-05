//! # TicTacToe State Canonicalization
//!
//! This module implements state canonicalization for TicTacToe using the D4 dihedral group.
//! The 3x3 board has 8 symmetries: 4 rotations (0°, 90°, 180°, 270°) and 4 reflections.
//!
//! Canonicalization finds the lexicographically minimal board representation under all
//! symmetries, enabling efficient transposition table usage.

use ndarray::{Array3, ArrayView3};

/// Apply a rotation to the board state
///
/// # Arguments
/// * `state` - The 3x3x2 state array (2 layers for X and O)
/// * `k` - Number of 90° clockwise rotations (0-3)
///
/// # Returns
/// Rotated state array
pub(crate) fn rotate_90_clockwise(state: &ArrayView3<f32>, k: i32) -> Array3<f32> {
    let k = ((k % 4) + 4) % 4; // Normalize to 0-3
    let mut result = state.to_owned();

    for _ in 0..k {
        let temp = result.clone();
        for layer in 0..2 {
            for y in 0..3 {
                for x in 0..3 {
                    // Rotation 90° clockwise: (y, x) -> (x, 2-y)
                    result[[layer, x, 2 - y]] = temp[[layer, y, x]];
                }
            }
        }
    }

    result
}

/// Apply horizontal reflection (flip across horizontal axis)
///
/// # Arguments
/// * `state` - The 3x3x2 state array
///
/// # Returns
/// Horizontally flipped state
pub(crate) fn reflect_horizontal(state: &ArrayView3<f32>) -> Array3<f32> {
    let mut result = state.to_owned();

    for layer in 0..2 {
        for y in 0..3 {
            for x in 0..3 {
                // Horizontal flip: (y, x) -> (2-y, x)
                result[[layer, 2 - y, x]] = state[[layer, y, x]];
            }
        }
    }

    result
}

/// Apply vertical reflection (flip across vertical axis)
///
/// # Arguments
/// * `state` - The 3x3x2 state array
///
/// # Returns
/// Vertically flipped state
pub(crate) fn reflect_vertical(state: &ArrayView3<f32>) -> Array3<f32> {
    let mut result = state.to_owned();

    for layer in 0..2 {
        for y in 0..3 {
            for x in 0..3 {
                // Vertical flip: (y, x) -> (y, 2-x)
                result[[layer, y, 2 - x]] = state[[layer, y, x]];
            }
        }
    }

    result
}

/// Apply main diagonal reflection (transpose)
///
/// # Arguments
/// * `state` - The 3x3x2 state array
///
/// # Returns
/// Transposed state
fn reflect_main_diagonal(state: &ArrayView3<f32>) -> Array3<f32> {
    let mut result = state.to_owned();

    for layer in 0..2 {
        for y in 0..3 {
            for x in 0..3 {
                // Main diagonal flip: (y, x) -> (x, y)
                result[[layer, x, y]] = state[[layer, y, x]];
            }
        }
    }

    result
}

/// Compute a canonical key for lexicographic comparison
///
/// # Arguments
/// * `state` - The 3x3x2 state array
///
/// # Returns
/// Byte vector representing the state for comparison
fn compute_canonical_key(state: &ArrayView3<f32>) -> Vec<u8> {
    let mut key = Vec::with_capacity(18); // 2 layers * 3 * 3 = 18 bytes

    for layer in 0..2 {
        for y in 0..3 {
            for x in 0..3 {
                key.push(if state[[layer, y, x]] > 0.5 { 1 } else { 0 });
            }
        }
    }

    key
}

/// Generate all 8 symmetries of a TicTacToe state
///
/// The D4 dihedral group has 8 elements:
/// - 4 rotations: R0, R90, R180, R270
/// - 4 reflections: H, V, D (main diagonal), A (anti-diagonal)
///
/// # Arguments
/// * `state` - The 3x3x2 state array
///
/// # Returns
/// Vector of 8 transformed states (one for each symmetry)
pub(crate) fn generate_symmetries(state: &ArrayView3<f32>) -> Vec<Array3<f32>> {
    let mut symmetries = Vec::with_capacity(8);

    // 4 rotations
    for k in 0..4 {
        symmetries.push(rotate_90_clockwise(state, k));
    }

    // 4 reflections (we can generate these from rotations + one reflection)
    // Reflect horizontally, then rotate to get all reflections
    let reflected_h = reflect_horizontal(state);
    for k in 0..4 {
        symmetries.push(rotate_90_clockwise(&reflected_h.view(), k));
    }

    symmetries
}

/// Canonicalize a TicTacToe state to its lexicographically minimal form
///
/// Finds the minimal representation among all 8 symmetries of the board.
///
/// # Arguments
/// * `state` - The 3x3x2 state array (2 layers for X and O)
///
/// # Returns
/// Tuple of (canonical_state, transform_index) where:
/// - canonical_state: The lexicographically minimal state
/// - transform_index: Index of the transform that produced the canonical state (0-7)
pub fn canonicalize_state(state: &ArrayView3<f32>) -> (Array3<f32>, usize) {
    let symmetries = generate_symmetries(state);

    let mut min_key = compute_canonical_key(&symmetries[0].view());
    let mut min_idx = 0;
    let mut canonical = symmetries[0].clone();

    for (idx, sym) in symmetries.iter().enumerate().skip(1) {
        let key = compute_canonical_key(&sym.view());
        if key < min_key {
            min_key = key;
            min_idx = idx;
            canonical = sym.clone();
        }
    }

    (canonical, min_idx)
}

/// Get a human-readable name for a transform index
///
/// # Arguments
/// * `transform_idx` - Index from 0-7
///
/// # Returns
/// String describing the transform
pub fn transform_name(transform_idx: usize) -> &'static str {
    match transform_idx {
        0 => "R0 (identity)",
        1 => "R90 (rotate 90° clockwise)",
        2 => "R180 (rotate 180°)",
        3 => "R270 (rotate 270° clockwise)",
        4 => "H (horizontal flip)",
        5 => "H+R90 (horizontal flip + rotate 90°)",
        6 => "H+R180 (horizontal flip + rotate 180°)",
        7 => "H+R270 (horizontal flip + rotate 270°)",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {}
