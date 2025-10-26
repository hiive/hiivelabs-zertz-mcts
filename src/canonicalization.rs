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

use crate::board::BoardConfig;
use crate::node::Action;
use ndarray::{s, Array3, ArrayView3};

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
    spatial: &ArrayView3<f32>,
    layout: &[Vec<bool>],
    board_layers: Range<usize>,
) -> Vec<u8> {
    let start = board_layers.start;
    let end = board_layers.end.min(spatial.shape()[0]);
    let layer_count = end.saturating_sub(start);
    let mut key = Vec::with_capacity(layer_count * layout.len() * layout.len());

    for layer in start..end {
        for (y, row) in layout.iter().enumerate() {
            for (x, &valid) in row.iter().enumerate() {
                if valid {
                    let value = spatial[[layer, y, x]];
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

fn bounding_box(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
) -> Option<(usize, usize, usize, usize)> {
    let mut min_y = config.width;
    let mut max_y = 0usize;
    let mut min_x = config.width;
    let mut max_x = 0usize;
    let mut found = false;

    for y in 0..config.width {
        for x in 0..config.width {
            if spatial[[config.ring_layer, y, x]] > 0.5 {
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

fn get_translations(spatial: &ArrayView3<f32>, config: &BoardConfig) -> Vec<(String, i32, i32)> {
    if let Some((min_y, max_y, min_x, max_x)) = bounding_box(spatial, config) {
        let mut translations = Vec::new();
        let width = config.width as i32;
        let min_y = min_y as i32;
        let max_y = max_y as i32;
        let min_x = min_x as i32;
        let max_x = max_x as i32;
        for dy in -min_y..(width - max_y) {
            for dx in -min_x..(width - max_x) {
                translations.push((format!("T{},{}", dy, dx), dy, dx));
            }
        }
        translations
    } else {
        vec![("T0,0".to_string(), 0, 0)]
    }
}

fn translate_state(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
    layout: &[Vec<bool>],
    dy: i32,
    dx: i32,
) -> Option<Array3<f32>> {
    if dy == 0 && dx == 0 {
        return Some(spatial.to_owned());
    }

    let width = config.width as i32;
    let mut out = Array3::zeros(spatial.raw_dim());
    let board_range = board_layers_range(config);
    let board_start = board_range.start;
    let board_end = board_range.end.min(spatial.shape()[0]);

    if board_end < spatial.shape()[0] {
        let src = spatial.slice(s![board_end.., .., ..]);
        let mut dst = out.slice_mut(s![board_end.., .., ..]);
        dst.assign(&src);
    }

    for y in 0..config.width {
        for x in 0..config.width {
            if !layout[y][x] {
                continue;
            }
            if spatial[[config.ring_layer, y, x]] <= 0.5 {
                continue;
            }

            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny < 0 || nx < 0 || ny >= width || nx >= width {
                return None;
            }
            if !layout[ny as usize][nx as usize] {
                return None;
            }

            for layer in board_start..board_end {
                out[[layer, ny as usize, nx as usize]] = spatial[[layer, y, x]];
            }
        }
    }

    Some(out)
}

fn transform_state_cached(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Array3<f32> {
    let mut out = Array3::zeros(spatial.raw_dim());

    for y in 0..config.width as i32 {
        for x in 0..config.width as i32 {
            if !yx_to_ax.contains_key(&(y, x)) {
                continue;
            }
            if let Some((dest_y, dest_x)) =
                transform_coordinate(y, x, rot60_k, mirror, mirror_first, yx_to_ax, ax_to_yx)
            {
                for layer in 0..spatial.shape()[0] {
                    out[[layer, dest_y as usize, dest_x as usize]] =
                        spatial[[layer, y as usize, x as usize]];
                }
            }
        }
    }

    out
}

fn canonicalize_internal(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
    include_translations: bool,
) -> (Array3<f32>, String, String) {
    let layout = build_layout_mask(config);
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);
    let board_layers = board_layers_range(config);

    let mut best_state = spatial.to_owned();
    let mut best_key = canonical_key(spatial, &layout, board_layers.clone());
    let mut best_name = "R0".to_string();

    let translations = if include_translations {
        get_translations(spatial, config)
    } else {
        vec![("T0,0".to_string(), 0, 0)]
    };
    let sym_ops = symmetry_transforms(config);

    for (trans_name, dy, dx) in translations.iter() {
        let translated_state = if *dy == 0 && *dx == 0 {
            spatial.to_owned()
        } else if let Some(state) = translate_state(spatial, config, &layout, *dy, *dx) {
            state
        } else {
            continue;
        };

        for (sym_name, rot60_k, mirror, mirror_first) in sym_ops.iter() {
            let transformed = transform_state_cached(
                &translated_state.view(),
                config,
                *rot60_k,
                *mirror,
                *mirror_first,
                &yx_to_ax,
                &ax_to_yx,
            );
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

fn inverse_transform_name(name: &str) -> String {
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
pub fn canonicalize_spatial(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, String, String) {
    canonicalize_internal(spatial, config, false)
}

pub fn canonicalize_state(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, String, String) {
    canonicalize_internal(spatial, config, true)
}

#[allow(dead_code)]
fn generate_standard_layout_mask(rings: usize, width: usize) -> Vec<Vec<bool>> {
    let letters = match rings {
        37 => "ABCDEFG",
        48 => "ABCDEFGH",
        61 => "ABCDEFGHJ",
        _ => panic!("Unsupported ring count {}", rings),
    };
    let letters: Vec<char> = letters.chars().collect();
    let r_max = letters.len();
    assert_eq!(
        r_max, width,
        "expected width {} to match standard layout width {}",
        width, r_max
    );

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

    layout
}

#[allow(dead_code)]
fn build_layout_mask(config: &BoardConfig) -> Vec<Vec<bool>> {
    generate_standard_layout_mask(config.rings, config.width)
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
#[allow(dead_code)]
fn ax_rot60(mut q: i32, mut r: i32, k: i32) -> (i32, i32) {
    let mut k = (k % 6 + 6) % 6; // Normalize to 0-5
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
#[allow(dead_code)]
fn ax_mirror_q_axis(q: i32, r: i32) -> (i32, i32) {
    (q, -q - r)
}

#[allow(dead_code)]
fn transform_coordinate(
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
fn transform_coordinate_put(
    y: i32,
    x: i32,
    rot60_k: i32,
    mirror: bool,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Option<(i32, i32)> {
    let (mut q, mut r) = *yx_to_ax.get(&(y, x))?;
    let tmp = ax_rot60(q, r, rot60_k);
    q = tmp.0;
    r = tmp.1;
    if mirror {
        let tmp = ax_mirror_q_axis(q, r);
        q = tmp.0;
        r = tmp.1;
    }
    ax_to_yx.get(&(q, r)).copied()
}

#[allow(dead_code)]
fn dir_index_map(
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

#[allow(dead_code)]
fn parse_transform(transform: &str) -> Vec<String> {
    transform
        .split('_')
        .filter(|s| !s.is_empty() && *s != "R0")
        .map(|s| s.to_string())
        .collect()
}

#[allow(dead_code)]
fn parse_rot_component(component: &str) -> (i32, bool, bool) {
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

#[allow(dead_code)]
pub fn transform_action(action: &Action, transform: &str, config: &BoardConfig) -> Action {
    if transform.is_empty() || transform == "R0" {
        return action.clone();
    }

    let layout = build_layout_mask(config);
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);

    let mut current_action = action.clone();

    for part in parse_transform(transform) {
        if let Some(coords) = part.strip_prefix('T') {
            let mut split = coords.split(',');
            let dy = split.next().unwrap().parse::<i32>().unwrap();
            let dx = split.next().unwrap().parse::<i32>().unwrap();
            current_action = translate_action(&current_action, dy, dx, config, &layout);
        } else {
            let (rot60_k, mirror, mirror_first) = parse_rot_component(&part);
            current_action = apply_orientation(
                &current_action,
                rot60_k,
                mirror,
                mirror_first,
                config,
                &yx_to_ax,
                &ax_to_yx,
            );
        }
    }

    current_action
}

#[allow(dead_code)]
fn translate_action(
    action: &Action,
    dy: i32,
    dx: i32,
    config: &BoardConfig,
    layout: &[Vec<bool>],
) -> Action {
    let width = config.width as i32;
    match action {
        Action::Placement {
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
        } => {
            let new_dst_y = *dst_y as i32 + dy;
            let new_dst_x = *dst_x as i32 + dx;
            assert!(new_dst_y >= 0 && new_dst_y < width && new_dst_x >= 0 && new_dst_x < width);
            assert!(layout[new_dst_y as usize][new_dst_x as usize]);

            let new_remove = if let (Some(ry), Some(rx)) = (remove_y, remove_x) {
                let new_ry = *ry as i32 + dy;
                let new_rx = *rx as i32 + dx;
                assert!(new_ry >= 0 && new_ry < width && new_rx >= 0 && new_rx < width);
                assert!(layout[new_ry as usize][new_rx as usize]);
                (Some(new_ry as usize), Some(new_rx as usize))
            } else {
                (None, None)
            };

            Action::Placement {
                marble_type: *marble_type,
                dst_y: new_dst_y as usize,
                dst_x: new_dst_x as usize,
                remove_y: new_remove.0,
                remove_x: new_remove.1,
            }
        }
        Action::Capture {
            start_y,
            start_x,
            direction,
        } => {
            let new_y = *start_y as i32 + dy;
            let new_x = *start_x as i32 + dx;
            assert!(new_y >= 0 && new_y < width && new_x >= 0 && new_x < width);
            assert!(layout[new_y as usize][new_x as usize]);
            Action::Capture {
                start_y: new_y as usize,
                start_x: new_x as usize,
                direction: *direction,
            }
        }
        Action::Pass => Action::Pass,
    }
}

#[allow(dead_code)]
fn apply_orientation(
    action: &Action,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    config: &BoardConfig,
    yx_to_ax: &HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &HashMap<(i32, i32), (i32, i32)>,
) -> Action {
    match action {
        Action::Placement {
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
        } => {
            let dst = transform_coordinate_put(
                *dst_y as i32,
                *dst_x as i32,
                rot60_k,
                mirror,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("destination outside board under transform");
            let new_remove = if let (Some(ry), Some(rx)) = (remove_y, remove_x) {
                let res = transform_coordinate_put(
                    *ry as i32, *rx as i32, rot60_k, mirror, yx_to_ax, ax_to_yx,
                )
                .expect("removal outside board under transform");
                (Some(res.0 as usize), Some(res.1 as usize))
            } else {
                (None, None)
            };
            Action::Placement {
                marble_type: *marble_type,
                dst_y: dst.0 as usize,
                dst_x: dst.1 as usize,
                remove_y: new_remove.0,
                remove_x: new_remove.1,
            }
        }
        Action::Capture {
            start_y,
            start_x,
            direction,
        } => {
            let mapping = dir_index_map(rot60_k, mirror, mirror_first, config);
            let new_dir = *mapping.get(direction).expect("direction map missing");
            let (ny, nx) = transform_coordinate(
                *start_y as i32,
                *start_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("capture start outside board under transform");
            Action::Capture {
                start_y: ny as usize,
                start_x: nx as usize,
                direction: new_dir,
            }
        }
        Action::Pass => Action::Pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(width: usize, y: usize, x: usize) -> usize {
        y * width + x
    }

    #[allow(dead_code)]
    fn placement_action() -> Action {
        Action::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 2,
            remove_y: Some(2),
            remove_x: Some(4),
        }
    }

    #[allow(dead_code)]
    fn capture_action() -> Action {
        Action::Capture {
            start_y: 3,
            start_x: 3,
            direction: 0,
        }
    }

    #[test]
    fn test_transform_action_put_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let width = config.width;
        let action = placement_action();

        let expected = [
            ("MR60", (3, 2), (4, 5)),
            ("R60M", (3, 2), (4, 5)),
            ("MR120", (4, 3), (2, 4)),
            ("R120M", (4, 3), (2, 4)),
            ("R180", (3, 4), (4, 2)),
        ];

        for (transform, (dst_y, dst_x), (rem_y, rem_x)) in expected {
            let transformed = transform_action(&action, transform, &config);
            if let Action::Placement {
                dst_y: dy,
                dst_x: dx,
                remove_y,
                remove_x,
                ..
            } = transformed
            {
                assert_eq!((dy, dx), (dst_y, dst_x), "transform {}", transform);
                assert_eq!(
                    (remove_y, remove_x),
                    (Some(rem_y), Some(rem_x)),
                    "transform {}",
                    transform
                );
                let flat_dst = flat(width, dy, dx);
                let flat_rem = flat(width, rem_y, rem_x);
                match transform {
                    "MR60" | "R60M" => assert_eq!((flat_dst, flat_rem), (23, 33)),
                    "MR120" | "R120M" => assert_eq!((flat_dst, flat_rem), (31, 18)),
                    "R180" => assert_eq!((flat_dst, flat_rem), (25, 30)),
                    _ => (),
                }
            } else {
                panic!("Expected placement action for {}", transform);
            }
        }
    }

    #[test]
    fn test_transform_action_put_translation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let width = config.width;
        let action = placement_action();

        let transformed = transform_action(&action, "T1,0", &config);
        if let Action::Placement {
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            ..
        } = transformed
        {
            assert_eq!((dst_y, dst_x), (4, 2));
            assert_eq!((remove_y, remove_x), (Some(3), Some(4)));
            assert_eq!(flat(width, dst_y, dst_x), 30);
            assert_eq!(flat(width, remove_y.unwrap(), remove_x.unwrap()), 25);
        } else {
            panic!("Expected placement action");
        }
    }

    #[test]
    fn test_transform_action_cap_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = capture_action();

        let expected = [
            ("MR60", (2, 3, 3)),
            ("R60M", (4, 3, 3)),
            ("R60", (1, 3, 3)),
            ("R120", (2, 3, 3)),
            ("R180", (3, 3, 3)),
        ];

        for (transform, (dir, y, x)) in expected {
            let transformed = transform_action(&action, transform, &config);
            if let Action::Capture {
                direction,
                start_y,
                start_x,
            } = transformed
            {
                assert_eq!(
                    (direction, start_y, start_x),
                    (dir, y, x),
                    "transform {}",
                    transform
                );
            } else {
                panic!("Expected capture action for {}", transform);
            }
        }
    }

    #[test]
    fn test_transform_action_cap_translation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = capture_action();

        let transformed = transform_action(&action, "T2,-1", &config);
        if let Action::Capture {
            direction,
            start_y,
            start_x,
        } = transformed
        {
            assert_eq!(direction, 0);
            assert_eq!((start_y, start_x), (5, 2));
        } else {
            panic!("Expected capture action");
        }
    }

    #[test]
    fn test_transform_action_inverse_roundtrip() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action();
        let forward = transform_action(&action, "MR60", &config);
        let inverse = transform_action(&forward, "MR60", &config); // MR60 is its own inverse
        if let Action::Placement {
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            ..
        } = inverse
        {
            assert_eq!((dst_y, dst_x), (3, 2));
            assert_eq!((remove_y, remove_x), (Some(2), Some(4)));
        } else {
            panic!("Expected placement action");
        }
    }

    #[test]
    fn canonicalize_state_handles_translated_variants() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let layout = build_layout_mask(&config);
        let layers = config.layers_per_timestep * config.t + 1;
        let mut base = Array3::zeros((layers, config.width, config.width));

        let positions = [(3usize, 3usize), (3usize, 4usize), (4usize, 3usize)];
        for &(y, x) in &positions {
            assert!(layout[y][x], "source position ({}, {}) not on layout", y, x);
            base[[config.ring_layer, y, x]] = 1.0;
        }
        // Add marbles to break rotational symmetry
        base[[config.marble_layers.0, 3, 3]] = 1.0;
        base[[config.marble_layers.0 + 1, 4, 3]] = 1.0;

        let base_view = base.view();
        let (canonical_base, _, _) = canonicalize_state(&base_view, &config);

        let dy = -1;
        let dx = 0;
        for &(y, x) in &positions {
            let ny_i = y as i32 + dy;
            let nx_i = x as i32 + dx;
            assert!(ny_i >= 0 && nx_i >= 0, "translation moves off board");
            let ny = ny_i as usize;
            let nx = nx_i as usize;
            assert!(
                layout[ny][nx],
                "translated position ({}, {}) invalid",
                ny, nx
            );
        }

        let translated = translate_state(&base_view, &config, &layout, dy, dx)
            .expect("translation should remain on board");
        let translated_view = translated.view();
        let (canonical_translated, transform, _) = canonicalize_state(&translated_view, &config);

        assert_eq!(canonical_base, canonical_translated);
        assert!(
            transform.contains('T'),
            "expected translation component in transform, got {}",
            transform
        );
    }
}
#[allow(dead_code)]
pub fn transform_state(
    spatial: &ArrayView3<f32>,
    config: &BoardConfig,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
) -> Array3<f32> {
    let layout = generate_standard_layout_mask(config.rings, config.width);
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);
    transform_state_cached(
        spatial,
        config,
        rot60_k,
        mirror,
        mirror_first,
        &yx_to_ax,
        &ax_to_yx,
    )
}
