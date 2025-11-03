//! # Canonicalization Module
//!
//! State canonicalization for exploiting board symmetries in transposition tables.
//!
//! ## Purpose
//!
//! In games with symmetries, many different board orientations represent the same
//! strategic position. This module:
//! 1. **Finds canonical (standard) form** of any board state
//! 2. **Transforms actions** to work with canonical states
//! 3. **Computes inverse transforms** to map back to original orientation
//!
//! ## Symmetry Groups
//!
//! **37/61-ring boards** (D6 dihedral group):
//! - 6 rotations (0°, 60°, 120°, 180°, 240°, 300°)
//! - 6 reflections (mirroring then rotating)
//! - Total: 12 symmetries
//!
//! **48-ring boards** (D3 dihedral group):
//! - 3 rotations (0°, 120°, 240°)
//! - 3 reflections
//! - Total: 6 symmetries
//!
//! ## Transform Notation
//!
//! - `R{angle}`: Pure rotation (e.g., R60, R120)
//! - `MR{angle}`: Mirror, THEN rotate (e.g., MR60)
//! - `R{angle}M`: Rotate, THEN mirror (e.g., R60M)
//! - `T{dy},{dx}`: Translation by (dy, dx)
//!
//! **Inverse relationships**:
//! - `R(k)⁻¹ = R(-k)`
//! - `MR(k)⁻¹ = R(-k)M`
//! - `R(k)M⁻¹ = MR(-k)`
//!
//! ## Hexagonal Coordinate System
//!
//! Uses **axial coordinates** (q, r) for rotation/reflection:
//! - Conversion: `q = x - c`, `r = y - x` (where c = width/2)
//! - Rotation by 60°: `(q, r) → (-r, q+r)`
//! - Mirror over q-axis: `(q, r) → (q, -q-r)`
//!
//! ## Canonicalization Algorithm
//!
//! 1. Generate all symmetric variants (rotations + reflections + translations)
//! 2. Compute lexicographic key for each variant
//! 3. Select variant with smallest key as canonical form
//! 4. Return: (canonical_state, forward_transform, inverse_transform)

use std::collections::HashMap;
use std::ops::Range;

use super::board::BoardConfig;
// NOTE: Action import removed - Action enum no longer exists in node.rs
use ndarray::{s, Array3, ArrayView3};

// ============================================================================
// TRANSFORM FLAGS
// ============================================================================

/// Flags for controlling which transforms to use in canonicalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformFlags {
    bits: u8,
}

impl TransformFlags {
    /// Include rotational symmetries
    pub const ROTATION: Self = Self { bits: 0b001 };
    /// Include mirror symmetries
    pub const MIRROR: Self = Self { bits: 0b010 };
    /// Include translation symmetries
    pub const TRANSLATION: Self = Self { bits: 0b100 };
    /// All transforms (rotation + mirror + translation)
    pub const ALL: Self = Self { bits: 0b111 };
    /// Rotation and mirror only (no translation)
    pub const ROTATION_MIRROR: Self = Self { bits: 0b011 };
    /// No transforms (identity only)
    pub const NONE: Self = Self { bits: 0b000 };

    /// Check if rotation flag is set
    pub fn has_rotation(self) -> bool {
        (self.bits & Self::ROTATION.bits) != 0
    }

    /// Check if mirror flag is set
    pub fn has_mirror(self) -> bool {
        (self.bits & Self::MIRROR.bits) != 0
    }

    /// Check if translation flag is set
    pub fn has_translation(self) -> bool {
        (self.bits & Self::TRANSLATION.bits) != 0
    }

    /// Create from bits
    pub fn from_bits(bits: u8) -> Option<Self> {
        if bits <= 0b111 {
            Some(Self { bits })
        } else {
            None
        }
    }

    /// Get the raw bits
    pub fn bits(self) -> u8 {
        self.bits
    }
}

impl Default for TransformFlags {
    fn default() -> Self {
        TransformFlags::ALL
    }
}

// ============================================================================
// SYMMETRY TRANSFORMS
// ============================================================================

/// Generate list of symmetry transforms based on board configuration
///
/// **Board-specific symmetries**:
/// - **48-ring boards**: D3 symmetry (120° rotations only)
/// - **37/61-ring boards**: D6 symmetry (60° rotations)
///
/// **Return format**: `(name, rotation_steps, mirror, mirror_first)`
/// - `name`: Transform identifier (e.g., "R60", "MR120", "R180M")
/// - `rotation_steps`: Number of 60° steps (0-5)
/// - `mirror`: Whether to apply mirror reflection
/// - `mirror_first`: Order of operations (true = mirror before rotate)
fn symmetry_transforms(config: &BoardConfig) -> Vec<(String, i32, bool, bool)> {
    // D3 symmetry for 48-ring boards (120° rotations: steps 0, 2, 4)
    // D6 symmetry for 37/61-ring boards (60° rotations: steps 0-5)
    let rot_steps = if config.rings == 48 {
        vec![0, 2, 4]
    } else {
        vec![0, 1, 2, 3, 4, 5]
    };

    let mut transforms = Vec::new();
    transforms.push(("R0".to_string(), 0, false, false)); // Identity

    for &k in &rot_steps {
        let deg = k * 60;
        if k != 0 {
            // Pure rotation: R60, R120, R180, etc.
            transforms.push((format!("R{}", deg), k, false, false));
        }
        // Mirror-then-rotate: MR0, MR60, MR120, etc.
        transforms.push((format!("MR{}", deg), k, true, false));
        // Rotate-then-mirror: R0M, R60M, R120M, etc.
        transforms.push((format!("R{}M", deg), k, true, true));
    }

    transforms
}

fn parse_translation_component(component: &str) -> Option<(i32, i32)> {
    if !component.starts_with('T') {
        return None;
    }
    let coords = &component[1..];
    let mut parts = coords.split(',');
    let dy = parts.next()?.parse::<i32>().ok()?;
    let dx = parts.next()?.parse::<i32>().ok()?;
    Some((dy, dx))
}

fn combine_transform_name(trans_name: &str, sym_name: &str) -> String {
    if trans_name == "T0,0" && sym_name == "R0" {
        "R0".to_string()
    } else if trans_name == "T0,0" {
        sym_name.to_string()
    } else if sym_name == "R0" {
        trans_name.to_string()
    } else {
        format!("{}_{}", trans_name, sym_name)
    }
}

fn board_layers_range(config: &BoardConfig) -> Range<usize> {
    0..config.layers_per_timestep
}

fn canonical_key(
    spatial_state: &ArrayView3<f32>,
    layout: &[Vec<bool>],
    board_layers: Range<usize>,
) -> Vec<u8> {
    let start = board_layers.start;
    let end = board_layers.end.min(spatial_state.shape()[0]);
    let layer_count = end.saturating_sub(start);
    let mut key = Vec::with_capacity(layer_count * layout.len() * layout.len());

    for layer in start..end {
        for (y, row) in layout.iter().enumerate() {
            for (x, &valid) in row.iter().enumerate() {
                if valid {
                    let value = spatial_state[[layer, y, x]];
                    // Values are 0.0 or 1.0 for board layers; mirror Python's uint8 mask
                    key.push(if value > 0.5 { 1 } else { 0 });
                } else {
                    key.push(0);
                }
            }
        }
    }
    key
}

pub fn bounding_box(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
) -> Option<(usize, usize, usize, usize)> {
    let mut min_y = config.width;
    let mut max_y = 0usize;
    let mut min_x = config.width;
    let mut max_x = 0usize;
    let mut found = false;

    for y in 0..config.width {
        for x in 0..config.width {
            if spatial_state[[config.ring_layer, y, x]] > 0.5 {
                found = true;
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
            }
        }
    }

    if found {
        Some((min_y, max_y, min_x, max_x))
    } else {
        None
    }
}

/// Get all valid translation offsets for the current board state.
///
/// Tests each potential translation by attempting to apply it, only including
/// translations that successfully keep all rings within valid board positions.
/// This matches the Python implementation's validation behavior.
///
/// # Arguments
/// * `spatial_state` - The board state to find translations for
/// * `config` - Board configuration
///
/// # Returns
/// Vector of (name, dy, dx) tuples for all valid translations
pub fn get_translations(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Vec<(String, i32, i32)> {
    if let Some((min_y, max_y, min_x, max_x)) = bounding_box(spatial_state, config) {
        let mut translations = Vec::new();
        let width = config.width as i32;
        let min_y = min_y as i32;
        let max_y = max_y as i32;
        let min_x = min_x as i32;
        let max_x = max_x as i32;

        // Build coordinate maps once for validation
        let layout = build_layout_mask(config);
        let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);

        for dy in -min_y..(width - max_y) {
            for dx in -min_x..(width - max_x) {
                // Test if this translation is valid by attempting to apply it
                let translated = transform_state_with_maps(
                    spatial_state,
                    config,
                    0,      // rot60_k - no rotation
                    false,  // mirror
                    false,  // mirror_first
                    dy,
                    dx,
                    true,   // translate_first - forward transform
                    &yx_to_ax,
                    &ax_to_yx,
                );

                // Only include translations that succeed
                if translated.is_some() {
                    translations.push((format!("T{},{}", dy, dx), dy, dx));
                }
            }
        }
        translations
    } else {
        vec![("T0,0".to_string(), 0, 0)]
    }
}

/// Apply translation transform to coordinates
fn apply_translation(
    y: i32,
    x: i32,
    dy: i32,
    dx: i32,
    width: i32,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
) -> Option<(i32, i32)> {
    let trans_y = y + dy;
    let trans_x = x + dx;

    // Check bounds
    if trans_y < 0 || trans_x < 0 || trans_y >= width || trans_x >= width {
        return None;
    }

    // Check if translated position is valid on layout
    if !yx_to_ax.contains_key(&(trans_y, trans_x)) {
        return None;
    }

    Some((trans_y, trans_x))
}

/// Apply rotation/mirror transform to coordinates
fn apply_rotation_mirror(
    y: i32,
    x: i32,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Option<(i32, i32)> {
    if rot60_k == 0 && !mirror {
        return Some((y, x));
    }
    transform_coordinate(y, x, rot60_k, mirror, mirror_first, yx_to_ax, ax_to_yx)
}

/// Internal implementation of transform_state that accepts pre-computed coordinate maps.
///
/// This version is more efficient when transforming multiple states with the same
/// board configuration, as the coordinate maps can be built once and reused.
fn transform_state_with_maps(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    dy: i32,
    dx: i32,
    translate_first: bool,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Option<Array3<f32>> {
    let mut out = Array3::zeros(spatial_state.raw_dim());
    let width = config.width as i32;
    let board_range = board_layers_range(config);
    let board_start = board_range.start;
    let board_end = board_range.end.min(spatial_state.shape()[0]);

    // Copy non-board layers (if any exist beyond board_end)
    if board_end < spatial_state.shape()[0] {
        let src = spatial_state.slice(s![board_end.., .., ..]);
        let mut dst = out.slice_mut(s![board_end.., .., ..]);
        dst.assign(&src);
    }

    for y in 0..config.width as i32 {
        for x in 0..config.width as i32 {
            if !yx_to_ax.contains_key(&(y, x)) {
                continue;
            }

            // For translations, only process positions that have rings
            // For rotations/mirrors without translation, process all layout positions
            if dy != 0 || dx != 0 {
                // Translation mode: check if ring exists at source position
                if spatial_state[[config.ring_layer, y as usize, x as usize]] <= 0.5 {
                    continue;
                }
            }

            // Apply transforms in order based on translate_first flag
            let (dest_y, dest_x) = if translate_first {
                // FORWARD TRANSFORM ORDER: Translation FIRST, then rotation/mirror
                let (ty, tx) = match apply_translation(y, x, dy, dx, width, yx_to_ax) {
                    Some(coords) => coords,
                    None => return None,
                };
                match apply_rotation_mirror(ty, tx, rot60_k, mirror, mirror_first, yx_to_ax, ax_to_yx) {
                    Some(coords) => coords,
                    None => continue,
                }
            } else {
                // INVERSE TRANSFORM ORDER: Rotation/mirror FIRST, then translation
                let (ry, rx) = match apply_rotation_mirror(y, x, rot60_k, mirror, mirror_first, yx_to_ax, ax_to_yx) {
                    Some(coords) => coords,
                    None => continue,
                };
                match apply_translation(ry, rx, dy, dx, width, yx_to_ax) {
                    Some(coords) => coords,
                    None => return None,
                }
            };

            // Copy all board layers
            for layer in board_start..board_end {
                out[[layer, dest_y as usize, dest_x as usize]] =
                    spatial_state[[layer, y as usize, x as usize]];
            }
        }
    }

    Some(out)
}

pub fn canonicalize_internal(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
    flags: TransformFlags,
) -> (Array3<f32>, String, String) {
    let layout = build_layout_mask(config);
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);
    let board_layers = board_layers_range(config);

    let mut best_state = spatial_state.to_owned();
    let mut best_key = canonical_key(spatial_state, &layout, board_layers.clone());
    let mut best_name = "R0".to_string();

    // Get translations based on flags
    let translations = if flags.has_translation() {
        get_translations(spatial_state, config)
    } else {
        vec![("T0,0".to_string(), 0, 0)]
    };

    // Get rotation/mirror symmetries based on flags
    let sym_ops = if flags.has_rotation() || flags.has_mirror() {
        symmetry_transforms(config)
            .into_iter()
            .filter(|(name, _, _mirror, _)| {
                // Filter based on flags
                if name == "R0" {
                    true // Always include identity
                } else if name.starts_with("R") && !name.starts_with("MR") {
                    // Pure rotation or mirror-then-rotate (R{k} or R{k}M)
                    if name.contains('M') {
                        // R{k}M requires both rotation and mirror flags
                        flags.has_rotation() && flags.has_mirror()
                    } else {
                        // Pure rotation requires only rotation flag
                        flags.has_rotation()
                    }
                } else if name.starts_with("MR") {
                    // Rotate-then-mirror (MR{k}) requires both flags
                    flags.has_rotation() && flags.has_mirror()
                } else {
                    false
                }
            })
            .collect()
    } else {
        // No rotation/mirror flags: identity only
        vec![("R0".to_string(), 0, false, false)]
    };

    for (trans_name, dy, dx) in translations.iter() {
        let translated_state = if let Some(state) = transform_state_with_maps(
            spatial_state,
            config,
            0,  // rot60_k - no rotation for pure translation
            false,  // mirror
            false,  // mirror_first
            *dy,
            *dx,
            true,  // translate_first - forward transform
            &yx_to_ax,
            &ax_to_yx,
        ) {
            state
        } else {
            continue;
        };

        for (sym_name, rot60_k, mirror, mirror_first) in sym_ops.iter() {
            let transformed = transform_state_with_maps(
                &translated_state.view(),
                config,
                *rot60_k,
                *mirror,
                *mirror_first,
                0,  // dy - no additional translation
                0,  // dx
                true,  // translate_first - forward transform
                &yx_to_ax,
                &ax_to_yx,
            ).expect("Rotation/mirror transform should always succeed");
            let transformed_view = transformed.view();
            let key = canonical_key(&transformed_view, &layout, board_layers.clone());
            if key < best_key {
                best_key = key;
                best_name = combine_transform_name(trans_name, sym_name);
                best_state = transformed;
            }
        }
    }

    let inverse = inverse_transform_name(&best_name);
    (best_state, best_name, inverse)
}

pub fn inverse_transform_name(name: &str) -> String {
    if name == "R0" {
        return "R0".to_string();
    }

    if let Some(idx) = name.find('_') {
        let first = &name[..idx];
        let second = &name[idx + 1..];
        if let Some((dy, dx)) = parse_translation_component(first) {
            let inv_trans = format!("T{},{}", -dy, -dx);
            let inv_rot = inverse_transform_name(second);
            return format!("{}_{}", inv_rot, inv_trans);
        } else if let Some((dy, dx)) = parse_translation_component(second) {
            let inv_trans = format!("T{},{}", -dy, -dx);
            let inv_rot = inverse_transform_name(first);
            return format!("{}_{}", inv_trans, inv_rot);
        }
    }

    if let Some((dy, dx)) = parse_translation_component(name) {
        return format!("T{},{}", -dy, -dx);
    }

    if name.ends_with('M') && !name.starts_with("MR") {
        let angle: i32 = name[1..name.len() - 1].parse().unwrap_or(0);
        let inv_angle = (360 - angle) % 360;
        return format!("MR{}", inv_angle);
    }

    if let Some(stripped) = name.strip_prefix("MR") {
        let angle: i32 = stripped.parse().unwrap_or(0);
        let inv_angle = (360 - angle) % 360;
        return format!("R{}M", inv_angle);
    }

    if let Some(stripped) = name.strip_prefix('R') {
        let angle: i32 = stripped.parse().unwrap_or(0);
        let inv_angle = (360 - angle) % 360;
        return format!("R{}", inv_angle);
    }

    "R0".to_string()
}

#[allow(dead_code)]
pub fn canonicalize_spatial_state(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, String, String) {
    // Only rotation and mirror (no translation) - finds canonical orientation
    canonicalize_internal(spatial_state, config, TransformFlags::ROTATION_MIRROR)
}

pub fn canonicalize_state(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, String, String) {
    // Full canonicalization with rotation, mirror, and translation
    canonicalize_internal(spatial_state, config, TransformFlags::ALL)
}

/// Compute canonical key for lexicographic comparison
///
/// Returns a byte vector representing the board state over valid positions only.
/// This is used for finding the lexicographically minimal state representation.
pub fn compute_canonical_key(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Vec<u8> {
    let layout = build_layout_mask(config);
    let board_layers = board_layers_range(config);
    canonical_key(spatial_state, &layout, board_layers)
}

pub fn generate_standard_layout_mask(rings: usize, width: usize) -> Result<Vec<Vec<bool>>, String> {
    let letters = match rings {
        37 => "ABCDEFG",
        48 => "ABCDEFGH",
        61 => "ABCDEFGHJ",
        _ => return Err(format!("Unsupported ring count: {} (supported: 37, 48, 61)", rings)),
    };
    let letters: Vec<char> = letters.chars().collect();
    let r_max = letters.len();
    if r_max != width {
        return Err(format!(
            "Width mismatch: expected {} to match standard layout width {}",
            width, r_max
        ));
    }

    let is_even = r_max.is_multiple_of(2);
    let h_max =
        |idx: usize| -> usize { r_max - (idx as isize - (r_max / 2) as isize).unsigned_abs() };

    let mut r_min = h_max(0);
    if is_even {
        r_min += 1;
    }

    let mut layout = vec![vec![false; r_max]; r_max];

    for i in 0..r_max {
        let hh = h_max(i);
        let mut letters_row = Vec::with_capacity(hh);
        if (i as f64) < (hh as f64) / 2.0 {
            letters_row.extend_from_slice(&letters[..hh]);
        } else {
            letters_row.extend_from_slice(&letters[r_max - hh..]);
        }

        let nn_max = r_max - i;
        let nn_min = std::cmp::max(r_min as isize - i as isize, 1) as usize;

        for (k, lt) in letters_row.iter().enumerate() {
            let ix = std::cmp::min(k + nn_min, nn_max);
            let position = letters
                .iter()
                .position(|c| c == lt)
                .expect("letter must exist in sequence");
            if ix >= 1 && ix <= r_max {
                layout[i][position] = true;
            }
        }
    }

    Ok(layout)
}

pub fn build_layout_mask(config: &BoardConfig) -> Vec<Vec<bool>> {
    generate_standard_layout_mask(config.rings, config.width)
        .expect("BoardConfig should always have valid rings/width")
}

/// Build bidirectional maps between (y,x) and axial (q,r) coordinates
///
/// **Axial coordinate system**:
/// - `q = x - center_x`
/// - `r = y - x`
/// - Allows natural rotation/reflection via simple matrix operations
///
/// **Centering**: Coordinates are centered on board centroid for proper rotation
///
/// **Scaling** (48-ring boards only):
/// - Scale by 3.0 to convert D6 to D3 symmetry
/// - This is specific to the 48-ring hexagonal layout
///
/// **Returns**: `(yx_to_ax, ax_to_yx)` bidirectional lookup maps
pub fn build_axial_maps(
    config: &BoardConfig,
    layout: &[Vec<bool>],
) -> (
    HashMap<(i32, i32), (i32, i32)>,
    HashMap<(i32, i32), (i32, i32)>,
) {
    const SQRT3: f64 = 1.732_050_807_568_877_2;

    let width = config.width as i32;
    let c = width as f64 / 2.0;

    let mut records = Vec::new();

    // Step 1: Convert all valid positions to axial coordinates
    for y in 0..config.width {
        for x in 0..config.width {
            if layout[y][x] {
                // Axial coordinates (q, r)
                let q = x as f64 - c;
                let r = y as f64 - x as f64;
                // Cartesian coordinates (for centroid calculation)
                let xc = SQRT3 * (q + r / 2.0);
                let yc = 1.5 * r;
                records.push((y as i32, x as i32, q, r, xc, yc));
            }
        }
    }

    // Step 2: Find centroid of board in Cartesian space
    let len = records.len() as f64;
    let xc_center = records.iter().map(|(_, _, _, _, xc, _)| *xc).sum::<f64>() / len;
    let yc_center = records.iter().map(|(_, _, _, _, _, yc)| *yc).sum::<f64>() / len;

    // Step 3: Convert centroid back to axial coordinates
    let q_center = (SQRT3 / 3.0) * xc_center - (1.0 / 3.0) * yc_center;
    let r_center = (2.0 / 3.0) * yc_center;

    // Step 4: Scale factor for 48-ring boards (D3 vs D6 symmetry)
    let scale = if config.rings == 48 { 3.0 } else { 1.0 };

    let mut yx_to_ax = HashMap::new();
    let mut ax_to_yx = HashMap::new();

    // Step 5: Center and scale all coordinates, build bidirectional maps
    for (y, x, q, r, _, _) in records.into_iter() {
        let q_centered = q - q_center;
        let r_centered = r - r_center;
        let q_adj = (scale * q_centered).round() as i32;
        let r_adj = (scale * r_centered).round() as i32;
        yx_to_ax.insert((y, x), (q_adj, r_adj));
        ax_to_yx.insert((q_adj, r_adj), (y, x));
    }

    (yx_to_ax, ax_to_yx)
}

/// Rotate axial coordinates by k × 60° (hexagonal rotation)
///
/// **Formula**: One 60° rotation transforms (q, r) → (-r, q+r)
///
/// Applied k times for rotation by k × 60°.
pub fn ax_rot60(mut q: i32, mut r: i32, k: i32) -> (i32, i32) {
    let mut k = k.rem_euclid(6); // Normalize to 0-5
    while k > 0 {
        // Single 60° rotation
        let new_q = -r;
        let new_r = q + r;
        q = new_q;
        r = new_r;
        k -= 1;
    }
    (q, r)
}

/// Mirror axial coordinates across q-axis
///
/// **Formula**: (q, r) → (q, -q-r)
///
/// This is one of the 6 reflection symmetries in the D6 group.
pub fn ax_mirror_q_axis(q: i32, r: i32) -> (i32, i32) {
    (q, -q - r)
}

pub fn transform_coordinate(
    y: i32,
    x: i32,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Option<(i32, i32)> {
    let (mut q, mut r) = *yx_to_ax.get(&(y, x))?;
    if mirror_first {
        if mirror {
            let tmp = ax_mirror_q_axis(q, r);
            q = tmp.0;
            r = tmp.1;
        }
        let tmp = ax_rot60(q, r, rot60_k);
        q = tmp.0;
        r = tmp.1;
    } else {
        let tmp = ax_rot60(q, r, rot60_k);
        q = tmp.0;
        r = tmp.1;
        if mirror {
            let tmp = ax_mirror_q_axis(q, r);
            q = tmp.0;
            r = tmp.1;
        }
    }
    ax_to_yx.get(&(q, r)).copied()
}

#[allow(dead_code)]
pub fn dir_index_map(
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    config: &BoardConfig,
) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for (idx, (dy, dx)) in config.directions.iter().enumerate() {
        let mut dq = *dx;
        let mut dr = *dy - *dx;
        if mirror_first {
            if mirror {
                let tmp = ax_mirror_q_axis(dq, dr);
                dq = tmp.0;
                dr = tmp.1;
            }
            let tmp = ax_rot60(dq, dr, rot60_k);
            dq = tmp.0;
            dr = tmp.1;
        } else {
            let tmp = ax_rot60(dq, dr, rot60_k);
            dq = tmp.0;
            dr = tmp.1;
            if mirror {
                let tmp = ax_mirror_q_axis(dq, dr);
                dq = tmp.0;
                dr = tmp.1;
            }
        }
        let new_dx = dq;
        let new_dy = dr + dq;
        let new_idx = config
            .directions
            .iter()
            .position(|&(dy2, dx2)| dy2 == new_dy && dx2 == new_dx)
            .expect("transformed direction not found");
        map.insert(idx, new_idx);
    }
    map
}

pub fn parse_transform(transform: &str) -> Vec<String> {
    transform
        .split('_')
        .filter(|s| !s.is_empty() && *s != "R0")
        .map(|s| s.to_string())
        .collect()
}

pub fn parse_rot_component(component: &str) -> (i32, bool, bool) {
    if component.ends_with('M') && !component.starts_with("MR") {
        let angle = component[1..component.len() - 1]
            .parse::<i32>()
            .unwrap_or(0);
        (angle / 60, true, true)
    } else if let Some(stripped) = component.strip_prefix("MR") {
        let angle = stripped.parse::<i32>().unwrap_or(0);
        (angle / 60, true, false)
    } else if let Some(stripped) = component.strip_prefix('R') {
        let angle = stripped.parse::<i32>().unwrap_or(0);
        (angle / 60, false, false)
    } else {
        (0, false, false)
    }
}

// NOTE: Action transformation has been moved to game-specific modules.
// For Zertz action transformation, see src/games/zertz/action_transform.rs

/// Transform board state using rotation, mirror, and/or translation.
///
/// This is the public convenience API that builds coordinate maps internally.
/// For batch transformations, consider using `transform_state_with_maps` directly
/// with pre-computed maps for better performance.
pub fn transform_state(
    spatial_state: &ArrayView3<f32>,
    config: &BoardConfig,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    dy: i32,
    dx: i32,
    translate_first: bool,
) -> Option<Array3<f32>> {
    let layout = generate_standard_layout_mask(config.rings, config.width)
        .expect("BoardConfig should always have valid rings/width");
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);
    transform_state_with_maps(
        spatial_state,
        config,
        rot60_k,
        mirror,
        mirror_first,
        dy,
        dx,
        translate_first,
        &yx_to_ax,
        &ax_to_yx,
    )
}
