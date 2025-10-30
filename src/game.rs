//! # Game Logic Module
//!
//! Pure, stateless game logic for Zèrtz. This module mirrors Python's `zertz_logic.py`
//! and serves as the single source of truth for all game rules.
//!
//! ## Architecture
//!
//! This module is **stateless** - all functions take game state as input and return
//! results without side effects (except `apply_*` functions which mutate provided arrays).
//!
//! ## State Representation
//!
//! - **spatial_state**: `Array3<f32>` shape `(layers, height, width)`
//!   - Layer 0: Rings (1.0 = present, 0.0 = removed)
//!   - Layers 1-3: Marbles (white/gray/black)
//!
//! - **global_state**: `Array1<f32>` shape `(10,)`
//!   - [0-2]: Supply counts (W/G/B)
//!   - [3-5]: P1 captured (W/G/B)
//!   - [6-8]: P2 captured (W/G/B)
//!   - [9]: Current player (1 or 2)

use crate::board::BoardConfig;
use ndarray::{s, Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};
use smallvec::SmallVec;
use std::collections::{HashSet, VecDeque};

// ============================================================================
// GAME OUTCOME CONSTANTS
// ============================================================================

/// Game outcome from Player 1's perspective
pub const PLAYER_1_WIN: i8 = 1;
pub const PLAYER_2_WIN: i8 = -1;
pub const TIE: i8 = 0;
pub const BOTH_LOSE: i8 = -2; // Tournament rule: both lose (collaboration detected)

// NOTE: Win thresholds configured per mode via BoardConfig::win_conditions
// Standard: 3-of-each, or 4W/5G/6B
// Blitz: 2-of-each, or 3W/4G/5B

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// ============================================================================
// Axial Coordinate Transformations
// ============================================================================

/// Rotate axial coordinate by k * 60° counterclockwise
///
/// In axial coordinates (q, r), a 60° CCW rotation is: (q, r) -> (-r, q + r)
/// This works for both regular and doubled coordinates (used for even-width boards).
///
/// # Arguments
/// * `q` - Axial q coordinate
/// * `r` - Axial r coordinate
/// * `k` - Number of 60° rotations (will be normalized to 0-5)
///
/// # Returns
/// Rotated (q, r) coordinates
pub fn ax_rot60(mut q: i32, mut r: i32, k: i32) -> (i32, i32) {
    let k_norm = k.rem_euclid(6);
    for _ in 0..k_norm {
        let temp = q;
        q = -r;
        r = temp + r;
    }
    (q, r)
}

/// Mirror axial coordinate across the q-axis
///
/// In cube coordinates (x=q, y=-q-r, z=r), mirroring across the q-axis
/// swaps y and z, giving (x, z, y) = (q, r, -q-r) = (q, -q-r) in axial.
///
/// # Arguments
/// * `q` - Axial q coordinate
/// * `r` - Axial r coordinate
///
/// # Returns
/// Mirrored (q, r) coordinates
pub fn ax_mirror_q_axis(q: i32, r: i32) -> (i32, i32) {
    (q, -q - r)
}

// ============================================================================

/// Check if (y, x) coordinates are within board bounds
/// Used by Python wrapper
#[inline]
pub fn is_inbounds(y: i32, x: i32, width: usize) -> bool {
    y >= 0 && x >= 0 && (y as usize) < width && (x as usize) < width
}

/// Type alias for neighbor lists - uses SmallVec to avoid heap allocations
/// Hexagonal grids have exactly 6 neighbors, so this fits on the stack
type NeighborList = SmallVec<[(usize, usize); 6]>;
type NeighborListRaw = SmallVec<[(i32, i32); 6]>;

/// Get list of neighboring indices (including out-of-bounds)
/// Returns i32 coordinates which may be negative or >= width
#[inline]
pub fn get_neighbors_raw(y: usize, x: usize, config: &BoardConfig) -> NeighborListRaw {
    config
        .directions
        .iter()
        .map(|(dy, dx)| (y as i32 + dy, x as i32 + dx))
        .collect()
}

/// Get list of neighboring indices (filtered to in-bounds only)
#[inline]
pub fn get_neighbors(y: usize, x: usize, config: &BoardConfig) -> NeighborList {
    config
        .directions
        .iter()
        .filter_map(|(dy, dx)| {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if is_inbounds(ny, nx, config.width) {
                Some((ny as usize, nx as usize))
            } else {
                None
            }
        })
        .collect()
}

/// Find all connected regions on the board
pub fn get_regions(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Vec<Vec<(usize, usize)>> {
    let mut regions = Vec::new();
    let mut not_visited: HashSet<(usize, usize)> = (0..config.width)
        .flat_map(|y| (0..config.width).map(move |x| (y, x)))
        .filter(|&(y, x)| spatial_state[[config.ring_layer, y, x]] == 1.0)
        .collect();

    while let Some(&start) = not_visited.iter().next() {
        not_visited.remove(&start);
        let mut region = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(index) = queue.pop_back() {
            region.push(index);

            for neighbor in get_neighbors(index.0, index.1, config) {
                if not_visited.contains(&neighbor)
                    && spatial_state[[config.ring_layer, neighbor.0, neighbor.1]] == 1.0
                {
                    not_visited.remove(&neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        regions.push(region);
    }

    regions
}

/// Get list of empty ring indices across the board
pub fn get_open_rings(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Vec<(usize, usize)> {
    // Get all vacant rings (ring present, no marble)
    let all_open: Vec<(usize, usize)> = (0..config.width)
        .flat_map(|y| (0..config.width).map(move |x| (y, x)))
        .filter(|&(y, x)| {
            spatial_state[[config.ring_layer, y, x]] == 1.0
                && !(1..4).any(|layer| spatial_state[[layer, y, x]] > 0.0)
        })
        .collect();
    all_open
}

/// Check if ring can be removed (geometric rule)
/// A ring is removable if:
/// 1. It's empty (no marble)
/// 2. Two consecutive neighbors are missing (including out-of-bounds)
pub fn is_ring_removable(
    spatial_state: &ArrayView3<f32>,
    y: usize,
    x: usize,
    config: &BoardConfig,
) -> bool {
    // Check if empty (only ring, no marble)
    let has_ring = spatial_state[[config.ring_layer, y, x]] > 0.0;
    let has_marble = (1..4).any(|layer| spatial_state[[layer, y, x]] > 0.0);

    if !has_ring || has_marble {
        return false;
    }

    // Get neighbors (including out-of-bounds)
    let neighbors = get_neighbors_raw(y, x, config);

    // Add first neighbor to end for wrap-around check
    let mut neighbors_wrapped = neighbors.clone();
    if let Some(&first) = neighbors.first() {
        neighbors_wrapped.push(first);
    }

    // Check for two consecutive empty neighbors
    let mut adjacent_empty = 0;
    for (ny, nx) in neighbors_wrapped {
        // Check if neighbor is in bounds and has a ring
        let has_neighbor_ring = if is_inbounds(ny, nx, config.width) {
            spatial_state[[config.ring_layer, ny as usize, nx as usize]] == 1.0
        } else {
            false // Out of bounds = no ring
        };

        if has_neighbor_ring {
            // Neighbor has a ring - reset counter
            adjacent_empty = 0;
        } else {
            // Neighbor is empty (no ring or out of bounds)
            adjacent_empty += 1;
            if adjacent_empty >= 2 {
                return true;
            }
        }
    }

    false
}

/// Get removable rings (rings that can be removed without disconnecting board)
pub fn get_removable_rings(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Vec<(usize, usize)> {
    (0..config.width)
        .flat_map(|y| (0..config.width).map(move |x| (y, x)))
        .filter(|&(y, x)| is_ring_removable(spatial_state, y, x, config))
        .collect()
}

fn get_captured_index(config: &BoardConfig, player: usize, marble_idx: usize) -> usize {
    if player == config.player_1 {
        config.p1_cap_w + marble_idx
    } else {
        config.p2_cap_w + marble_idx
    }
}

/// Get global_state index for marble type in supply
/// Used by Python wrapper
pub fn get_supply_index(marble_type: char, config: &BoardConfig) -> usize {
    match marble_type {
        'w' => config.supply_w,
        'g' => config.supply_g,
        'b' => config.supply_b,
        _ => panic!("Invalid marble type: {}", marble_type),
    }
}

/// Get marble type at given position
/// Returns: 'w', 'g', 'b', or '\0' (none)
/// Used by Python wrapper
pub fn get_marble_type_at(spatial_state: &ArrayView3<f32>, y: usize, x: usize, _config: &BoardConfig) -> char {
    if spatial_state[[1, y, x]] == 1.0 {
        'w'
    } else if spatial_state[[2, y, x]] == 1.0 {
        'g'
    } else if spatial_state[[3, y, x]] == 1.0 {
        'b'
    } else {
        '\0'
    }
}

/// Calculate landing position after capturing marble
/// Used by Python wrapper
pub fn get_jump_destination(start_y: usize, start_x: usize, cap_y: usize, cap_x: usize) -> (i32, i32) {
    let sy = start_y as i32;
    let sx = start_x as i32;
    let cy = cap_y as i32;
    let cx = cap_x as i32;
    let dy = (cy - sy) * 2;
    let dx = (cx - sx) * 2;
    (sy + dy, sx + dx)
}

/// Capture any regions that are completely full of marbles.
///
/// Rule: After any action, check all regions. If a region has ALL rings occupied by marbles,
/// capture all those marbles and remove all those rings.
///
/// Note: Only applies when there are multiple regions (i.e., isolation has occurred).
/// A single region covering the entire board is not considered "isolated".
/// Returns list of captured marble positions as (marble_layer, y, x) tuples.
fn apply_isolation_capture(
    spatial_state: &mut ArrayViewMut3<f32>,
    global_state: &mut ArrayViewMut1<f32>,
    config: &BoardConfig,
    current_player: usize,
) -> Vec<(usize, usize, usize)> {
    let mut captured_marbles = Vec::new();

    // Create immutable view from mutable view to pass to get_regions
    let spatial_state_view = spatial_state.view();
    let regions = get_regions(&spatial_state_view, config);

    // Only apply isolation capture if there are multiple regions
    let num_regions = regions.len();
    if num_regions < 2 {
        return captured_marbles;
    }

    for region in regions {
        if region.is_empty() {
            continue;
        }

        // Check if ALL rings in this region are occupied by marbles
        let all_occupied = region.iter().all(|&(y, x)| {
            (config.marble_layers.0..config.marble_layers.1)
                .any(|layer| spatial_state[[layer, y, x]] > 0.0)
        });

        // If all rings are occupied, capture them
        if all_occupied {
            // eprintln!("[ISO] Capturing fully-occupied region with {} rings out of {} regions", region.len(), num_regions);
            for (y, x) in region {
                if spatial_state[[config.ring_layer, y, x]] == 0.0 {
                    continue;
                }

                // Find and capture the marble
                if let Some(marble_layer) = (config.marble_layers.0..config.marble_layers.1)
                    .find(|&layer| spatial_state[[layer, y, x]] > 0.0)
                {
                    let marble_idx = marble_layer - config.marble_layers.0;
                    let captured_idx = get_captured_index(config, current_player, marble_idx);
                    global_state[captured_idx] += 1.0;

                    // Track captured position for animation
                    captured_marbles.push((marble_layer, y, x));
                }

                // Remove marble and ring
                for layer in config.marble_layers.0..config.marble_layers.1 {
                    spatial_state[[layer, y, x]] = 0.0;
                }
                spatial_state[[config.ring_layer, y, x]] = 0.0;
            }
        }
    }

    captured_marbles
}

/// Get valid placement actions
/// Returns Array3<f32> with shape (3, width², width²+1)
/// Indices are (marble_type, dst_flat, remove_flat)
/// where dst_flat and remove_flat are flattened (y * width + x) positions
/// remove_flat can be width² to indicate "no removal"
pub fn get_placement_actions(
    spatial_state: &ArrayView3<f32>,
    global_state: &ArrayView1<f32>,
    config: &BoardConfig,
) -> Array3<f32> {
    let width2 = config.width * config.width;
    let mut placement_mask = Array3::zeros((3, width2, width2 + 1));

    // Get current player supply counts
    let cur_player = global_state[config.cur_player] as usize;
    let supply_counts = [
        global_state[config.supply_w],
        global_state[config.supply_g],
        global_state[config.supply_b],
    ];

    // Get player's captured marble counts
    let captured_idx = if cur_player == config.player_1 {
        [config.p1_cap_w, config.p1_cap_g, config.p1_cap_b]
    } else {
        [config.p2_cap_w, config.p2_cap_g, config.p2_cap_b]
    };
    let captured_counts = [
        global_state[captured_idx[0]],
        global_state[captured_idx[1]],
        global_state[captured_idx[2]],
    ];

    // Get open rings (in main region) and removable rings
    let open_rings = get_open_rings(spatial_state, config);
    let removable_rings = get_removable_rings(spatial_state, config);

    // Determine which marbles can be placed
    let marble_counts = if supply_counts.iter().all(|&x| x == 0.0) {
        // Use current player's captured marbles
        captured_counts
    } else {
        supply_counts
    };

    // For each marble type
    for (marble_idx, &marble_count) in marble_counts.iter().enumerate() {
        if marble_count == 0.0 {
            continue;
        }

        // For each open ring in main region
        for &(dst_y, dst_x) in &open_rings {
            let dst_flat = dst_y * config.width + dst_x;

            // For each removable ring position
            for &(rem_y, rem_x) in &removable_rings {
                let rem_flat = rem_y * config.width + rem_x;
                // Cannot remove the same ring we're placing on
                if dst_flat != rem_flat {
                    placement_mask[[marble_idx, dst_flat, rem_flat]] = 1.0;
                }
            }

            // Allow "no removal" option only if:
            // - No removable rings exist, OR
            // - Only one removable ring and it's the destination itself
            let only_removable_is_dest = removable_rings.len() == 1
                && removable_rings
                    .iter()
                    .any(|&(ry, rx)| ry == dst_y && rx == dst_x);

            if removable_rings.is_empty() || only_removable_is_dest {
                placement_mask[[marble_idx, dst_flat, width2]] = 1.0;
            }
        }
    }

    placement_mask
}

/// Get valid capture actions
/// Returns Array3<f32> with shape (6, width, width)
pub fn get_capture_actions(spatial_state: &ArrayView3<f32>, config: &BoardConfig) -> Array3<f32> {
    let mut capture_mask = Array3::zeros((6, config.width, config.width));

    // Check if this is a chain capture (CAPTURE_LAYER has a marble marked)
    let chain_capture_pos = (0..config.width)
        .flat_map(|y| (0..config.width).map(move |x| (y, x)))
        .find(|&(y, x)| spatial_state[[config.capture_layer, y, x]] > 0.0);

    // For each position with a marble
    for y in 0..config.width {
        for x in 0..config.width {
            // If chain capture is active, only allow captures from that specific position
            if let Some((chain_y, chain_x)) = chain_capture_pos {
                if y != chain_y || x != chain_x {
                    continue;
                }
            }

            // Check if position has a marble
            let marble_layer = (1..4).find(|&layer| spatial_state[[layer, y, x]] > 0.0);
            if marble_layer.is_none() {
                continue;
            }

            // For each direction
            for (dir_idx, (dy, dx)) in config.directions.iter().enumerate() {
                // Check capture position
                let cap_y = y as i32 + dy;
                let cap_x = x as i32 + dx;

                if !is_inbounds(cap_y, cap_x, config.width) {
                    continue;
                }

                let cap_y = cap_y as usize;
                let cap_x = cap_x as usize;

                // Must have a marble to capture
                let has_marble_to_cap = (1..4).any(|layer| spatial_state[[layer, cap_y, cap_x]] > 0.0);
                if !has_marble_to_cap {
                    continue;
                }

                // Check landing position
                let land_y = cap_y as i32 + dy;
                let land_x = cap_x as i32 + dx;

                if !is_inbounds(land_y, land_x, config.width) {
                    continue;
                }

                let land_y = land_y as usize;
                let land_x = land_x as usize;

                // Landing position must have ring and no marble
                if spatial_state[[config.ring_layer, land_y, land_x]] == 0.0 {
                    continue;
                }

                let has_marble_at_land = (1..4).any(|layer| spatial_state[[layer, land_y, land_x]] > 0.0);
                if has_marble_at_land {
                    continue;
                }

                // Valid capture
                capture_mask[[dir_idx, y, x]] = 1.0;
            }
        }
    }

    capture_mask
}

/// Get valid actions (both placement and capture)
pub fn get_valid_actions(
    spatial_state: &ArrayView3<f32>,
    global_state: &ArrayView1<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, Array3<f32>) {
    let capture_mask = get_capture_actions(spatial_state, config);

    // If any captures available, placement is not allowed
    let has_captures = capture_mask.iter().any(|&x| x > 0.0);

    let width2 = config.width * config.width;
    let placement_mask = if has_captures {
        Array3::zeros((3, width2, width2 + 1))
    } else {
        get_placement_actions(spatial_state, global_state, config)
    };

    (placement_mask, capture_mask)
}

/// Apply a placement action
/// Returns list of captured marble positions from isolation as (marble_layer, y, x) tuples
pub fn apply_placement(
    spatial_state: &mut ArrayViewMut3<f32>,
    global_state: &mut ArrayViewMut1<f32>,
    marble_type: usize, // 0=white, 1=gray, 2=black
    dst_y: usize,
    dst_x: usize,
    remove_y: Option<usize>,
    remove_x: Option<usize>,
    config: &BoardConfig,
) -> Vec<(usize, usize, usize)> {
    // STEP 1: Reset capture layer (matches Python zertz_board.py behavior)
    // Placements always end any ongoing chain capture sequence
    spatial_state.slice_mut(s![config.capture_layer, .., ..]).fill(0.0);

    let cur_player = global_state[config.cur_player] as usize;

    // Place marble
    let marble_layer = marble_type + 1; // 1=white, 2=gray, 3=black
    spatial_state[[marble_layer, dst_y, dst_x]] = 1.0;

    // Remove ring if specified
    if let (Some(ry), Some(rx)) = (remove_y, remove_x) {
        spatial_state[[config.ring_layer, ry, rx]] = 0.0;
    }

    // Check for fully-occupied isolated regions and capture them
    let captured_marbles = apply_isolation_capture(spatial_state, global_state, config, cur_player);

    // Decrement marble count from supply or captured pool
    let supply_idx = marble_type; // 0, 1, 2
    let supply_empty = [
        global_state[config.supply_w],
        global_state[config.supply_g],
        global_state[config.supply_b],
    ]
    .iter()
    .all(|&count| count <= 0.0);

    if global_state[supply_idx] > 0.0 {
        // Use from supply
        global_state[supply_idx] -= 1.0;
    } else if supply_empty {
        // Use from captured pool (only allowed when entire supply is empty)
        let captured_idx = if cur_player == config.player_1 {
            config.p1_cap_w + marble_type
        } else {
            config.p2_cap_w + marble_type
        };

        if global_state[captured_idx] > 0.0 {
            global_state[captured_idx] -= 1.0;
        } else {
            panic!(
                "No captured marbles of required type available for player {}",
                cur_player + 1
            );
        }
    } else {
        panic!(
            "No supply marbles of requested type available. Captured marbles may only be used when supply is empty."
        );
    }

    // Switch player
    global_state[config.cur_player] = if cur_player == config.player_1 {
        config.player_2 as f32
    } else {
        config.player_1 as f32
    };

    captured_marbles
}

/// Apply a capture action
pub fn apply_capture(
    spatial_state: &mut ArrayViewMut3<f32>,
    global_state: &mut ArrayViewMut1<f32>,
    start_y: usize,
    start_x: usize,
    direction: usize,
    config: &BoardConfig,
) {
    // STEP 1: Reset capture layer (matches Python zertz_board.py:524)
    // This clears any previous chain capture markers
    spatial_state.slice_mut(s![config.capture_layer, .., ..]).fill(0.0);

    let cur_player = global_state[config.cur_player] as usize;

    // Find marble layer at start position
    let marble_layer = (1..4)
        .find(|&layer| spatial_state[[layer, start_y, start_x]] > 0.0)
        .unwrap_or_else(|| {
            eprintln!("ERROR: Invalid capture attempted at ({}, {})", start_y, start_x);
            eprintln!("  Direction: {}", direction);
            eprintln!("  Ring present: {}", spatial_state[[config.ring_layer, start_y, start_x]] > 0.0);
            eprintln!("  White marble: {}", spatial_state[[1, start_y, start_x]]);
            eprintln!("  Gray marble: {}", spatial_state[[2, start_y, start_x]]);
            eprintln!("  Black marble: {}", spatial_state[[3, start_y, start_x]]);
            panic!("Game logic violation: attempted capture from empty position at ({}, {})", start_y, start_x)
        });

    // Get direction offset
    let (dy, dx) = config.directions[direction];

    // Calculate capture and landing positions
    let cap_y = (start_y as i32 + dy) as usize;
    let cap_x = (start_x as i32 + dx) as usize;
    let land_y = (cap_y as i32 + dy) as usize;
    let land_x = (cap_x as i32 + dx) as usize;

    // Find captured marble type
    let captured_marble_layer = (1..4)
        .find(|&layer| spatial_state[[layer, cap_y, cap_x]] > 0.0)
        .unwrap_or_else(|| {
            eprintln!("ERROR: No marble found at capture position ({}, {})", cap_y, cap_x);
            eprintln!("  Start position: ({}, {})", start_y, start_x);
            eprintln!("  Direction: {} (offset: {:?})", direction, config.directions[direction]);
            eprintln!("  Landing position: ({}, {})", land_y, land_x);
            eprintln!("  Ring at capture pos: {}", spatial_state[[config.ring_layer, cap_y, cap_x]] > 0.0);
            panic!("Game logic violation: attempted to capture from empty position at ({}, {})", cap_y, cap_x)
        });

    // Remove marble from start
    spatial_state[[marble_layer, start_y, start_x]] = 0.0;

    // Remove captured marble
    spatial_state[[captured_marble_layer, cap_y, cap_x]] = 0.0;

    // Place marble at landing
    spatial_state[[marble_layer, land_y, land_x]] = 1.0;

    // Update captured marble count
    let marble_idx = captured_marble_layer - 1; // Convert layer to index (0,1,2)
    let captured_idx = if cur_player == config.player_1 {
        config.p1_cap_w + marble_idx
    } else {
        config.p2_cap_w + marble_idx
    };
    global_state[captured_idx] += 1.0;

    // Check for chain capture in any direction from landing position
    let capture_actions = get_capture_actions(&spatial_state.view(), config);
    let can_chain = (0..config.directions.len())
        .any(|dir_idx| capture_actions[[dir_idx, land_y, land_x]] > 0.0);

    // STEP 2: Mark landing position if chain capture available (matches Python zertz_board.py:554-556)
    if can_chain {
        // Mark the landing position - this marble MUST continue capturing
        spatial_state[[config.capture_layer, land_y, land_x]] = 1.0;
    }

    // STEP 3: Switch player only if no chain capture
    if !can_chain {
        global_state[config.cur_player] = if cur_player == config.player_1 {
            config.player_2 as f32
        } else {
            config.player_1 as f32
        };
    }
}

// ============================================================================
// ISOLATION CAPTURE
// ============================================================================

/// Check for isolated regions after ring removal and capture marbles
///
/// After a ring is removed, the board may split into multiple disconnected regions.
/// If ALL rings in an isolated region are fully occupied (each has a marble),
/// then the current player captures all those marbles and removes those rings.
///
/// Returns tuple of (updated_spatial_state, updated_global_state, captured_marbles_list)
/// where captured_marbles_list contains tuples of (marble_layer_idx, y, x)
pub fn check_for_isolation_capture(
    spatial_state: &ArrayView3<f32>,
    global_state: &ArrayView1<f32>,
    config: &BoardConfig,
) -> (Array3<f32>, Array1<f32>, Vec<(usize, usize, usize)>) {
    let mut spatial_state_out = spatial_state.to_owned();
    let mut global_state_out = global_state.to_owned();
    let mut captured_marbles = Vec::new();

    let regions = get_regions(spatial_state, config);
    let cur_player = global_state[config.cur_player] as usize;

    // Only apply isolation capture if there are multiple regions
    if regions.len() < 2 {
        return (spatial_state_out, global_state_out, captured_marbles);
    }

    // Check ALL regions, capture any that are fully occupied
    for region in regions {
        if region.is_empty() {
            continue;
        }

        // Check if ALL rings in this region are occupied by marbles
        let all_occupied = region.iter().all(|&(y, x)| {
            (config.marble_layers.0..config.marble_layers.1)
                .any(|layer| spatial_state[[layer, y, x]] > 0.0)
        });

        // If all rings are occupied, capture them
        if all_occupied {
            // eprintln!("[ISO] Capturing fully-occupied region with {} rings", region.len());
            for (y, x) in region {
                if spatial_state_out[[config.ring_layer, y, x]] == 0.0 {
                    continue;
                }

                // Find and capture the marble
                if let Some(marble_layer) = (config.marble_layers.0..config.marble_layers.1)
                    .find(|&layer| spatial_state_out[[layer, y, x]] > 0.0)
                {
                    let marble_idx = marble_layer - config.marble_layers.0;
                    let captured_idx = if cur_player == config.player_1 {
                        config.p1_cap_w + marble_idx
                    } else {
                        config.p2_cap_w + marble_idx
                    };
                    global_state_out[captured_idx] += 1.0;

                    // Add to captured list for return value
                    captured_marbles.push((marble_layer, y, x));
                }

                // Remove marble and ring
                for layer in config.marble_layers.0..config.marble_layers.1 {
                    spatial_state_out[[layer, y, x]] = 0.0;
                }
                spatial_state_out[[config.ring_layer, y, x]] = 0.0;
            }
        }
    }

    (spatial_state_out, global_state_out, captured_marbles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array3};

    fn create_test_config() -> BoardConfig {
        BoardConfig::standard(37, 1).unwrap()
    }

    // ========================================================================
    // Axial coordinate transform tests
    // ========================================================================

    #[test]
    fn test_ax_rot60_single() {
        // Single 60° rotation: (1, 0) -> (0, 1)
        assert_eq!(ax_rot60(1, 0, 1), (0, 1));
    }

    #[test]
    fn test_ax_rot60_full_cycle() {
        // Full 360° rotation should return to origin
        let (q, r) = (2, 3);
        let result = ax_rot60(q, r, 6);
        assert_eq!(result, (q, r), "Full 360° rotation should return to original");
    }

    #[test]
    fn test_ax_rot60_negative() {
        // Negative rotation: k=-1 is same as k=5 (330° CCW = -30° CW)
        assert_eq!(ax_rot60(1, 0, -1), ax_rot60(1, 0, 5));
    }

    #[test]
    fn test_ax_rot60_180() {
        // 180° rotation of (2, 1) through three 60° steps:
        // (2, 1) -> (-1, 3) -> (-3, 2) -> (-2, -1)
        let (q, r) = (2, 1);
        let (q_rot, r_rot) = ax_rot60(q, r, 3);
        assert_eq!((q_rot, r_rot), (-2, -1));
    }

    #[test]
    fn test_ax_rot60_origin() {
        // Origin should be invariant under rotation
        assert_eq!(ax_rot60(0, 0, 3), (0, 0));
    }

    #[test]
    fn test_ax_mirror_q_axis_basic() {
        // Mirror (1, 0) across q-axis -> (1, -1)
        assert_eq!(ax_mirror_q_axis(1, 0), (1, -1));
    }

    #[test]
    fn test_ax_mirror_q_axis_symmetric() {
        // Mirroring twice should return to original
        let (q, r) = (2, 3);
        let (q_mir, r_mir) = ax_mirror_q_axis(q, r);
        let (q_back, r_back) = ax_mirror_q_axis(q_mir, r_mir);
        assert_eq!((q_back, r_back), (q, r));
    }

    #[test]
    fn test_ax_mirror_q_axis_origin() {
        // Origin should be invariant under mirroring
        assert_eq!(ax_mirror_q_axis(0, 0), (0, 0));
    }

    #[test]
    fn test_ax_mirror_q_axis_on_axis() {
        // Points on q-axis (r=0) should mirror to (q, -q)
        assert_eq!(ax_mirror_q_axis(3, 0), (3, -3));
    }

    fn create_empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        // For t=1: ring layer (1) + marble layers (3) + capture layer (1) = 5 layers
        let num_layers = config.t * config.layers_per_timestep + 1;
        // global_state: 3 supply + 6 captured (3 per player) + 1 cur_player = 10
        let global_state_size = 10;

        let spatial_state = Array3::zeros((num_layers, config.width, config.width));
        let global_state = Array1::zeros(global_state_size);
        (spatial_state, global_state)
    }

    #[test]
    fn test_placement_actions_basic() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Set up a simple board with one ring and one marble in supply
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        global_state[config.supply_w] = 1.0; // 1 white marble in supply
        global_state[config.cur_player] = config.player_1 as f32;

        let placement_mask = get_placement_actions(&spatial_state.view(), &global_state.view(), &config);

        let width = config.width;
        let width2 = width * width;
        let dst_flat = 3 * width + 3;

        // Should have exactly one valid placement (white marble at 3,3 with no removal)
        assert_eq!(placement_mask[[0, dst_flat, width2]], 1.0);

        // No entries for other marble colours
        assert_eq!(placement_mask[[1, dst_flat, width2]], 0.0);
        assert_eq!(placement_mask[[2, dst_flat, width2]], 0.0);
    }

    #[test]
    fn test_placement_uses_captured_marbles() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Set up board with ring but no supply marbles
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        global_state[config.supply_w] = 0.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;

        // Player 1 has captured marbles
        global_state[config.p1_cap_w] = 2.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let placement_mask = get_placement_actions(&spatial_state.view(), &global_state.view(), &config);

        let width = config.width;
        let width2 = width * width;
        let dst_flat = 3 * width + 3;

        // Should be able to place white marble from captured pool (no removal)
        assert_eq!(placement_mask[[0, dst_flat, width2]], 1.0);
    }

    #[test]
    fn test_placement_blocked_by_existing_marble() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Set up ring with existing marble
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble already there
        global_state[config.supply_w] = 1.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let placement_mask = get_placement_actions(&spatial_state.view(), &global_state.view(), &config);

        // Should not be able to place on occupied ring
        assert_eq!(placement_mask[[0, 3, 3]], 0.0);
    }

    #[test]
    fn test_capture_actions_basic() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Set up a capture scenario: marble at (3,3), marble at (3,4), empty ring at (3,5)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 3, 5]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble to capture

        let capture_mask = get_capture_actions(&spatial_state.view(), &config);

        // Find direction index for east (0, +1)
        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Should have valid capture to the east
        assert_eq!(capture_mask[[east_dir, 3, 3]], 1.0);
    }

    #[test]
    fn test_capture_requires_landing_ring() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Set up: marble at (3,3), marble at (3,4), NO ring at (3,5)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        // No ring at 3,5
        spatial_state[[1, 3, 3]] = 1.0; // White marble
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble

        let capture_mask = get_capture_actions(&spatial_state.view(), &config);

        // Should NOT have valid capture without landing ring
        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();
        assert_eq!(capture_mask[[east_dir, 3, 3]], 0.0);
    }

    #[test]
    fn test_captures_block_placements() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Set up a board with both placement and capture options
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 3, 5]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // Marble that can capture
        spatial_state[[2, 3, 4]] = 1.0; // Marble to capture
                                  // Ring at 3,5 is empty - could place there

        global_state[config.supply_w] = 1.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let (placement_mask, capture_mask) =
            get_valid_actions(&spatial_state.view(), &global_state.view(), &config);

        // Should have captures
        assert!(capture_mask.iter().any(|&x| x > 0.0));

        // Placements should be blocked
        assert!(placement_mask.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_apply_placement_removes_from_supply() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup - add enough rings to avoid isolation capture
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 4, 3]] = 1.0;
        spatial_state[[config.ring_layer, 4, 4]] = 1.0;
        global_state[config.supply_w] = 5.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Apply placement at (3,3), remove ring at (4,4)
        // This won't trigger isolation since (3,3) is connected to (3,4) and (4,3)
        apply_placement(
            &mut spatial_state.view_mut(),
            &mut global_state.view_mut(),
            0,
            3,
            3,
            Some(4),
            Some(4),
            &config,
        );

        // Check marble placed
        assert_eq!(spatial_state[[1, 3, 3]], 1.0);

        // Check ring removed
        assert_eq!(spatial_state[[config.ring_layer, 4, 4]], 0.0);

        // Check supply decremented
        assert_eq!(global_state[config.supply_w], 4.0);

        // Check player switched
        assert_eq!(global_state[config.cur_player] as usize, config.player_2);
    }

    #[test]
    fn test_apply_placement_removes_from_captured() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup with no supply but captured marbles
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        global_state[config.supply_w] = 0.0;
        global_state[config.p1_cap_w] = 3.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Apply placement
        apply_placement(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 0, 3, 3, None, None, &config);

        // Check captured pool decremented
        assert_eq!(global_state[config.p1_cap_w], 2.0);
    }

    #[test]
    fn test_apply_capture_basic() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup capture scenario
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 3, 5]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble to capture
        global_state[config.cur_player] = config.player_1 as f32;

        // Find east direction
        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply capture
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, east_dir, &config);

        // Check marble moved
        assert_eq!(spatial_state[[1, 3, 3]], 0.0); // Removed from start
        assert_eq!(spatial_state[[1, 3, 5]], 1.0); // Placed at landing

        // Check captured marble removed
        assert_eq!(spatial_state[[2, 3, 4]], 0.0);

        // Check capture count incremented (gray marble)
        assert_eq!(global_state[config.p1_cap_g], 1.0);

        // Check player switched (no chain)
        assert_eq!(global_state[config.cur_player] as usize, config.player_2);
    }

    #[test]
    fn test_apply_capture_chain() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup chain capture: marble at (2,2), capture marble at (2,3), land at (2,4)
        // Then immediately can capture marble at (2,5) landing at (2,6)
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 2, 4]] = 1.0;
        spatial_state[[config.ring_layer, 2, 5]] = 1.0;
        spatial_state[[config.ring_layer, 2, 6]] = 1.0;
        spatial_state[[1, 2, 2]] = 1.0; // White marble
        spatial_state[[2, 2, 3]] = 1.0; // Gray marble to capture
        spatial_state[[3, 2, 5]] = 1.0; // Black marble - chain capture target
        global_state[config.cur_player] = config.player_1 as f32;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply first capture
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, east_dir, &config);

        // Player should NOT switch (chain available)
        assert_eq!(global_state[config.cur_player] as usize, config.player_1);

        // Marble should be at landing position
        assert_eq!(spatial_state[[1, 2, 4]], 1.0);
    }

    #[test]
    fn test_apply_capture_chain_different_direction() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup: initial capture east, then forced capture north from new position
        spatial_state[[config.ring_layer, 3, 3]] = 1.0; // Start ring
        spatial_state[[config.ring_layer, 3, 4]] = 1.0; // First captured ring
        spatial_state[[config.ring_layer, 3, 5]] = 1.0; // First landing ring
        spatial_state[[config.ring_layer, 2, 5]] = 1.0; // Second captured ring (different direction)
        spatial_state[[config.ring_layer, 1, 5]] = 1.0; // Second landing ring

        spatial_state[[1, 3, 3]] = 1.0; // White marble (current player)
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble to capture first
        spatial_state[[3, 2, 5]] = 1.0; // Black marble forcing second capture

        global_state[config.cur_player] = config.player_1 as f32;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply first capture (east)
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, east_dir, &config);

        // Player should still be current because follow-up capture (north) is available
        assert_eq!(global_state[config.cur_player] as usize, config.player_1);

        // Marble must be at first landing position to continue chain
        assert_eq!(spatial_state[[1, 3, 5]], 1.0);
    }

    #[test]
    fn test_capture_layer_restricts_to_chain_marble() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Setup two marbles that can both capture:
        // Marble A at (2,2) can capture marble X at (2,3) landing at (2,4)
        // Marble B at (4,4) can capture marble Y at (4,5) landing at (4,6)
        spatial_state[[config.ring_layer, 2, 2]] = 1.0; // A start
        spatial_state[[config.ring_layer, 2, 3]] = 1.0; // X (capture target for A)
        spatial_state[[config.ring_layer, 2, 4]] = 1.0; // A landing
        spatial_state[[config.ring_layer, 4, 4]] = 1.0; // B start
        spatial_state[[config.ring_layer, 4, 5]] = 1.0; // Y (capture target for B)
        spatial_state[[config.ring_layer, 4, 6]] = 1.0; // B landing

        spatial_state[[1, 2, 2]] = 1.0; // White marble A
        spatial_state[[2, 2, 3]] = 1.0; // Gray marble X
        spatial_state[[1, 4, 4]] = 1.0; // White marble B
        spatial_state[[2, 4, 5]] = 1.0; // Gray marble Y

        // Without CAPTURE_LAYER marking, both marbles should be able to capture
        let capture_mask_before = get_capture_actions(&spatial_state.view(), &config);
        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        assert_eq!(capture_mask_before[[east_dir, 2, 2]], 1.0, "Marble A should be able to capture");
        assert_eq!(capture_mask_before[[east_dir, 4, 4]], 1.0, "Marble B should be able to capture");

        // Now mark A as the marble in chain capture (e.g., A just captured and can continue)
        spatial_state[[config.capture_layer, 2, 2]] = 1.0;

        // With CAPTURE_LAYER marked, only marble A should be able to capture
        let capture_mask_after = get_capture_actions(&spatial_state.view(), &config);

        assert_eq!(capture_mask_after[[east_dir, 2, 2]], 1.0, "Marble A should still be able to capture (chain)");
        assert_eq!(capture_mask_after[[east_dir, 4, 4]], 0.0, "Marble B should NOT be able to capture (not chain marble)");

        // Verify only one capture is available (the chain capture)
        let total_captures: f32 = capture_mask_after.iter().sum();
        assert_eq!(total_captures, 1.0, "Only the chain marble should have capture moves");
    }

    // ========================================================================
    // Chain Capture Enforcement Tests (Issue #1 from rust_mcts_fix_proposal.md)
    // ========================================================================

    #[test]
    fn test_apply_capture_resets_capture_layer() {
        /// Verify that apply_capture() clears any pre-existing capture layer markers
        /// at the start of execution (Step 1 of the fix).
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup a simple capture scenario
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 3, 5]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble to capture
        global_state[config.cur_player] = config.player_1 as f32;

        // Manually mark a different position in the capture layer
        // (simulating stale data from a previous chain capture)
        spatial_state[[config.capture_layer, 1, 1]] = 1.0;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply capture
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, east_dir, &config);

        // Verify the stale marker at (1,1) was cleared
        assert_eq!(
            spatial_state[[config.capture_layer, 1, 1]], 0.0,
            "Capture layer at (1,1) should be cleared by apply_capture()"
        );
    }

    #[test]
    fn test_apply_capture_marks_landing_on_chain() {
        /// Verify that apply_capture() marks the landing position in the capture layer
        /// when a chain capture is available (Step 2 of the fix).
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup chain capture: marble at (2,2), capture marble at (2,3), land at (2,4)
        // Then can immediately capture marble at (2,5) landing at (2,6)
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 2, 4]] = 1.0;
        spatial_state[[config.ring_layer, 2, 5]] = 1.0;
        spatial_state[[config.ring_layer, 2, 6]] = 1.0;
        spatial_state[[1, 2, 2]] = 1.0; // White marble
        spatial_state[[2, 2, 3]] = 1.0; // Gray marble to capture
        spatial_state[[3, 2, 5]] = 1.0; // Black marble - chain capture target
        global_state[config.cur_player] = config.player_1 as f32;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply first capture
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, east_dir, &config);

        // Verify landing position (2,4) is marked in capture layer
        assert_eq!(
            spatial_state[[config.capture_layer, 2, 4]], 1.0,
            "Landing position (2,4) should be marked in capture layer for chain capture"
        );

        // Verify player did NOT switch (chain capture continues)
        assert_eq!(
            global_state[config.cur_player] as usize, config.player_1,
            "Player should not switch when chain capture is available"
        );
    }

    #[test]
    fn test_apply_capture_no_mark_when_no_chain() {
        /// Verify that apply_capture() does NOT mark the landing position when
        /// no chain capture is available, and DOES switch players.
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup simple capture with NO chain available
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.ring_layer, 3, 5]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble
        spatial_state[[2, 3, 4]] = 1.0; // Gray marble to capture
        // No additional marble to capture from (3,5)
        global_state[config.cur_player] = config.player_1 as f32;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Apply capture
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, east_dir, &config);

        // Verify landing position (3,5) is NOT marked in capture layer
        assert_eq!(
            spatial_state[[config.capture_layer, 3, 5]], 0.0,
            "Landing position (3,5) should NOT be marked when no chain capture available"
        );

        // Verify player DID switch (no chain capture)
        assert_eq!(
            global_state[config.cur_player] as usize, config.player_2,
            "Player should switch when no chain capture is available"
        );
    }

    #[test]
    fn test_apply_placement_resets_capture_layer() {
        /// Verify that apply_placement() clears the capture layer at the start,
        /// ending any ongoing chain capture sequence.
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup placement scenario
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 4, 4]] = 1.0;
        global_state[config.supply_w] = 5.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Manually mark a position in capture layer (simulating active chain capture)
        spatial_state[[config.capture_layer, 2, 2]] = 1.0;

        // Apply placement
        apply_placement(
            &mut spatial_state.view_mut(),
            &mut global_state.view_mut(),
            0, // white marble
            3,
            3,
            Some(4),
            Some(4),
            &config,
        );

        // Verify capture layer was cleared
        assert_eq!(
            spatial_state[[config.capture_layer, 2, 2]], 0.0,
            "Capture layer should be cleared by apply_placement()"
        );

        // Verify entire capture layer is cleared
        let capture_layer_sum: f32 = spatial_state.slice(s![config.capture_layer, .., ..]).sum();
        assert_eq!(
            capture_layer_sum, 0.0,
            "Entire capture layer should be zeroed by apply_placement()"
        );
    }

    #[test]
    fn test_chain_capture_enforces_single_marble() {
        /// Verify that once a chain capture begins, ONLY the marble that landed
        /// can perform the next capture (no other marbles can capture).
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup: Two potential capture scenarios
        // Scenario A: Marble at (2,2) can capture at (2,3) landing at (2,4)
        // Scenario B: Marble at (4,4) can capture at (4,5) landing at (4,6)
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 2, 4]] = 1.0;
        spatial_state[[config.ring_layer, 2, 5]] = 1.0;
        spatial_state[[config.ring_layer, 2, 6]] = 1.0;

        spatial_state[[1, 2, 2]] = 1.0; // White marble A
        spatial_state[[2, 2, 3]] = 1.0; // Gray marble to capture
        spatial_state[[3, 2, 5]] = 1.0; // Black marble - chain target from (2,4)

        spatial_state[[config.ring_layer, 4, 4]] = 1.0;
        spatial_state[[config.ring_layer, 4, 5]] = 1.0;
        spatial_state[[config.ring_layer, 4, 6]] = 1.0;

        spatial_state[[1, 4, 4]] = 1.0; // White marble B
        spatial_state[[2, 4, 5]] = 1.0; // Gray marble - capture target for B

        global_state[config.cur_player] = config.player_1 as f32;

        let east_dir = config
            .directions
            .iter()
            .position(|&(dy, dx)| dy == 0 && dx == 1)
            .unwrap();

        // Step 1: Apply capture from marble A at (2,2)
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, east_dir, &config);

        // Now marble A is at (2,4) and a chain capture is available to (2,6)
        // Verify only marble A can capture (marble B at (4,4) should NOT be able to capture)

        let capture_mask = get_capture_actions(&spatial_state.view(), &config);

        // Marble A at (2,4) should be able to capture
        assert_eq!(
            capture_mask[[east_dir, 2, 4]], 1.0,
            "Chain marble at (2,4) should be able to continue capturing"
        );

        // Marble B at (4,4) should NOT be able to capture (chain capture in progress)
        assert_eq!(
            capture_mask[[east_dir, 4, 4]], 0.0,
            "Other marbles should NOT be able to capture during chain sequence"
        );

        // Verify only ONE capture is available
        let total_captures: f32 = capture_mask.iter().sum();
        assert_eq!(
            total_captures, 1.0,
            "Only the chain marble should have capture moves during chain sequence"
        );
    }

    #[test]
    fn test_corner_ring_removable_with_oob_neighbors() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Fill entire 7x7 board with rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        // Test position (0,0) - top-left corner
        // Neighbors: (1,0), (0,-1), (-1,-1), (-1,0), (0,1), (1,1)
        // Directions: [(1,0), (0,-1), (-1,-1), (-1,0), (0,1), (1,1)]
        // Dir 0: (1,0) in-bounds with ring
        // Dir 1: (0,-1) OOB
        // Dir 2: (-1,-1) OOB
        // Dir 3: (-1,0) OOB
        // Dir 4: (0,1) in-bounds with ring
        // Dir 5: (1,1) in-bounds with ring
        // Has 3 consecutive OOB neighbors (dirs 1,2,3) so should be removable

        let is_removable = is_ring_removable(&spatial_state.view(), 0, 0, &config);
        assert!(is_removable, "Corner ring (0,0) should be removable with 3 consecutive OOB neighbors");

        // Test center position (3,3) - should NOT be removable (all neighbors in-bounds with rings)
        let is_removable_center = is_ring_removable(&spatial_state.view(), 3, 3, &config);
        assert!(!is_removable_center, "Center ring (3,3) should NOT be removable on full board");
    }

    #[test]
    fn test_apply_placement_triggers_isolation_capture() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Coordinate helpers (from Python mapping)
        let d4 = (3, 3);
        let f1 = (6, 5);
        let f2 = (5, 5);
        let g1 = (6, 6);
        let g2 = (5, 6);

        // Prepare rings involved in scenario
        for &(y, x) in &[d4, f1, f2, g1, g2] {
            spatial_state[[config.ring_layer, y, x]] = 1.0;
        }

        // Place marble on G1 (isolated target)
        spatial_state[[1, g1.0, g1.1]] = 1.0;

        // Remove neighbors to prepare isolation
        spatial_state[[config.ring_layer, f1.0, f1.1]] = 0.0;
        spatial_state[[config.ring_layer, g2.0, g2.1]] = 0.0;

        // Set supply and current player
        global_state[config.supply_w] = 5.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Apply placement at D4 removing F2
        // This creates TWO isolated single-ring regions:
        // - D4 with the marble we just placed
        // - G1 with the pre-existing marble
        // Both regions are fully occupied, so BOTH get captured
        apply_placement(
            &mut spatial_state.view_mut(),
            &mut global_state.view_mut(),
            0, // white marble
            d4.0,
            d4.1,
            Some(f2.0),
            Some(f2.1),
            &config,
        );

        // Both D4 and G1 should be captured (2 marbles total)
        assert_eq!(global_state[config.p1_cap_w], 2.0);
        assert_eq!(spatial_state[[1, d4.0, d4.1]], 0.0);
        assert_eq!(spatial_state[[config.ring_layer, d4.0, d4.1]], 0.0);
        assert_eq!(spatial_state[[1, g1.0, g1.1]], 0.0);
        assert_eq!(spatial_state[[config.ring_layer, g1.0, g1.1]], 0.0);
    }

    #[test]
    fn test_isolation_capture_all_fully_occupied_regions() {
        /// This test verifies that ALL fully-occupied regions are captured,
        /// regardless of their location on the board.

        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Create scenario with THREE isolated single-ring regions, all fully occupied:
        // Region A: D4 (will have the marble we just placed)
        // Region B: G1 (with pre-existing marble)
        // Region C: A1 (with pre-existing marble)
        // All three regions are fully occupied, so all should be captured

        // Coordinates
        let d4 = (3, 3);   // Main board location (where we place)
        let f2 = (5, 5);   // Ring to be removed
        let g1 = (6, 5);   // Region B
        let a1 = (6, 0);   // Region C

        // Set up rings
        spatial_state[[config.ring_layer, d4.0, d4.1]] = 1.0;
        spatial_state[[config.ring_layer, f2.0, f2.1]] = 1.0;
        spatial_state[[config.ring_layer, g1.0, g1.1]] = 1.0;
        spatial_state[[config.ring_layer, a1.0, a1.1]] = 1.0;

        // Place marbles on G1 and A1 (D4 will get a marble during placement)
        spatial_state[[1, g1.0, g1.1]] = 1.0;  // white on G1
        spatial_state[[1, a1.0, a1.1]] = 1.0;  // white on A1

        // Remove neighbors to prepare isolation
        spatial_state[[config.ring_layer, 6, 4]] = 0.0;  // F1 (neighbor of G1)
        spatial_state[[config.ring_layer, 5, 6]] = 0.0;  // G2 (neighbor of G1)
        spatial_state[[config.ring_layer, 5, 0]] = 0.0;  // A2 (neighbor of A1)
        spatial_state[[config.ring_layer, 6, 1]] = 0.0;  // B1 (neighbor of A1)

        // Set up game state
        global_state[config.supply_w] = 5.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Place marble at D4, remove F2
        // This creates THREE isolated single-ring regions:
        // - D4 (just placed)
        // - G1 (pre-existing)
        // - A1 (pre-existing)
        // ALL are fully occupied, so ALL get captured
        apply_placement(
            &mut spatial_state.view_mut(),
            &mut global_state.view_mut(),
            0,  // white marble
            d4.0,
            d4.1,
            Some(f2.0),
            Some(f2.1),
            &config,
        );

        // Verify ALL three regions were captured
        assert_eq!(spatial_state[[1, d4.0, d4.1]], 0.0, "D4 marble should be captured");
        assert_eq!(spatial_state[[config.ring_layer, d4.0, d4.1]], 0.0, "D4 ring should be removed");

        assert_eq!(spatial_state[[1, g1.0, g1.1]], 0.0, "G1 marble should be captured");
        assert_eq!(spatial_state[[config.ring_layer, g1.0, g1.1]], 0.0, "G1 ring should be removed");

        assert_eq!(spatial_state[[1, a1.0, a1.1]], 0.0, "A1 marble should be captured");
        assert_eq!(spatial_state[[config.ring_layer, a1.0, a1.1]], 0.0, "A1 ring should be removed");

        // Verify all 3 marbles were captured
        assert_eq!(global_state[config.p1_cap_w], 3.0, "Should capture all 3 marbles");
    }

    #[test]
    #[should_panic(expected = "Captured marbles may only be used when supply is empty")]
    fn test_apply_placement_disallows_captured_when_supply_available() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Prepare destination ring
        let d4 = (3, 3);
        spatial_state[[config.ring_layer, d4.0, d4.1]] = 1.0;

        // No white marbles in supply, but grey supply still available
        global_state[config.supply_w] = 0.0;
        global_state[config.supply_g] = 1.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.p1_cap_w] = 1.0; // captured white available
        global_state[config.cur_player] = config.player_1 as f32;

        // Attempt to place white marble without removing a ring
        apply_placement(
            &mut spatial_state.view_mut(),
            &mut global_state.view_mut(),
            0,
            d4.0,
            d4.1,
            None,
            None,
            &config,
        );
    }
}

// ============================================================================
// GAME TERMINATION CHECKS
// ============================================================================

/// Check if game is over (any terminal condition).
///
/// Checks:
/// - Win by captures (3-of-each, or 4W/5G/6B)
/// - Board completely filled
/// - Current player has no marbles (supply + captured all zero)
///
/// Returns true if any terminal condition is met.
pub fn is_game_over(
    spatial_state: &ArrayView3<f32>,
    global_state: &ArrayView1<f32>,
    config: &BoardConfig,
) -> bool {
    let p1_caps = [
        global_state[config.p1_cap_w],
        global_state[config.p1_cap_g],
        global_state[config.p1_cap_b],
    ];

    let p2_caps = [
        global_state[config.p2_cap_w],
        global_state[config.p2_cap_g],
        global_state[config.p2_cap_b],
    ];

    // Check 3-of-each win (uses mode-specific threshold)
    if p1_caps
        .iter()
        .all(|&x| x >= config.win_conditions.each_color)
        || p2_caps
            .iter()
            .all(|&x| x >= config.win_conditions.each_color)
    {
        return true;
    }

    // Check specific marble wins (uses mode-specific thresholds)
    if p1_caps[0] >= config.win_conditions.white_only
        || p1_caps[1] >= config.win_conditions.gray_only
        || p1_caps[2] >= config.win_conditions.black_only
    {
        return true;
    }
    if p2_caps[0] >= config.win_conditions.white_only
        || p2_caps[1] >= config.win_conditions.gray_only
        || p2_caps[2] >= config.win_conditions.black_only
    {
        return true;
    }

    // Check if all remaining rings are occupied by marbles
    let mut all_occupied = true;
    for y in 0..config.width {
        for x in 0..config.width {
            if spatial_state[[config.ring_layer, y, x]] == 1.0 {
                let mut has_marble = false;
                for layer in config.marble_layers.0..config.marble_layers.1 {
                    if spatial_state[[layer, y, x]] > 0.0 {
                        has_marble = true;
                        break;
                    }
                }
                if !has_marble {
                    all_occupied = false;
                    break;
                }
            }
        }
        if !all_occupied {
            break;
        }
    }
    if all_occupied {
        return true;
    }

    // Check if current player has no marbles to play (supply + captured)
    let cur_player = global_state[config.cur_player] as usize;
    let supply = [
        global_state[config.supply_w],
        global_state[config.supply_g],
        global_state[config.supply_b],
    ];

    let captured_slice = if cur_player == config.player_1 {
        [config.p1_cap_w, config.p1_cap_g, config.p1_cap_b]
    } else {
        [config.p2_cap_w, config.p2_cap_g, config.p2_cap_b]
    };

    let captured = [
        global_state[captured_slice[0]],
        global_state[captured_slice[1]],
        global_state[captured_slice[2]],
    ];

    if supply
        .iter()
        .zip(captured.iter())
        .all(|(&s, &c)| s + c == 0.0)
    {
        return true;
    }

    false
}

/// Get game outcome from Player 1's perspective.
///
/// Returns:
/// - PLAYER_1_WIN (1): Player 1 wins
/// - PLAYER_2_WIN (-1): Player 2 wins
/// - TIE (0): Tie/Draw
/// - BOTH_LOSE (-2): Both players lose (collaboration detected)
///
/// Note: This returns the outcome from Player 1's perspective.
/// The caller must convert to their own perspective if needed.
pub fn get_game_outcome(
    spatial_state: &ArrayView3<f32>,
    global_state: &ArrayView1<f32>,
    config: &BoardConfig,
) -> i8 {
    let p1_caps = [
        global_state[config.p1_cap_w],
        global_state[config.p1_cap_g],
        global_state[config.p1_cap_b],
    ];

    let p2_caps = [
        global_state[config.p2_cap_w],
        global_state[config.p2_cap_g],
        global_state[config.p2_cap_b],
    ];

    // Determine winner by captures (uses mode-specific thresholds)
    let p1_won = p1_caps
        .iter()
        .all(|&x| x >= config.win_conditions.each_color)
        || p1_caps[0] >= config.win_conditions.white_only
        || p1_caps[1] >= config.win_conditions.gray_only
        || p1_caps[2] >= config.win_conditions.black_only;

    let p2_won = p2_caps
        .iter()
        .all(|&x| x >= config.win_conditions.each_color)
        || p2_caps[0] >= config.win_conditions.white_only
        || p2_caps[1] >= config.win_conditions.gray_only
        || p2_caps[2] >= config.win_conditions.black_only;

    if p1_won && p2_won {
        // Both won (simultaneous), shouldn't happen but treat as draw
        return TIE;
    } else if p1_won {
        return PLAYER_1_WIN;
    } else if p2_won {
        return PLAYER_2_WIN;
    }

    // Check for BOTH_LOSE condition (tournament collaboration rule):
    // If board is full AND both players have zero captures, both lose

    // First check if board is full
    let mut all_occupied = true;
    'outer: for y in 0..config.width {
        for x in 0..config.width {
            if spatial_state[[config.ring_layer, y, x]] == 1.0 {
                let mut has_marble = false;
                for layer in config.marble_layers.0..config.marble_layers.1 {
                    if spatial_state[[layer, y, x]] > 0.0 {
                        has_marble = true;
                        break;
                    }
                }
                if !has_marble {
                    all_occupied = false;
                    break 'outer;
                }
            }
        }
    }

    // If board is full, check for zero captures or determine winner
    if all_occupied {
        let p1_has_zero = p1_caps.iter().all(|&x| x == 0.0);
        let p2_has_zero = p2_caps.iter().all(|&x| x == 0.0);

        if p1_has_zero && p2_has_zero {
            // Both players lose (collaboration detected)
            return BOTH_LOSE;
        }

        // Normal full board: last player to move wins (current player loses)
        let cur_player = global_state[config.cur_player] as usize;
        if cur_player == config.player_1 {
            // Player 1 to move, Player 2 wins
            return PLAYER_2_WIN;
        } else {
            // Player 2 to move, Player 1 wins
            return PLAYER_1_WIN;
        }
    }

    // Check if current player has no marbles (opponent wins)
    let cur_player = global_state[config.cur_player] as usize;
    let supply = [
        global_state[config.supply_w],
        global_state[config.supply_g],
        global_state[config.supply_b],
    ];

    let captured_slice = if cur_player == config.player_1 {
        [config.p1_cap_w, config.p1_cap_g, config.p1_cap_b]
    } else {
        [config.p2_cap_w, config.p2_cap_g, config.p2_cap_b]
    };

    let captured = [
        global_state[captured_slice[0]],
        global_state[captured_slice[1]],
        global_state[captured_slice[2]],
    ];

    if supply
        .iter()
        .zip(captured.iter())
        .all(|(&s, &c)| s + c == 0.0)
    {
        // Current player has no marbles, opponent wins
        if cur_player == config.player_1 {
            return PLAYER_2_WIN;
        } else {
            return PLAYER_1_WIN;
        }
    }

    // No terminal condition (shouldn't happen if is_game_over returned true)
    TIE
}

#[cfg(test)]
mod termination_tests {
    use super::*;
    use ndarray::{s, Array1, Array3};

    fn create_test_config() -> BoardConfig {
        BoardConfig::standard(37, 1).unwrap()
    }

    fn create_empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let num_layers = config.t * config.layers_per_timestep + 1;
        let global_state_size = 10;
        let spatial_state = Array3::zeros((num_layers, config.width, config.width));
        let global_state = Array1::zeros(global_state_size);
        (spatial_state, global_state)
    }

    // ========================================================================
    // is_game_over() tests
    // ========================================================================

    #[test]
    fn test_is_game_over_three_of_each_p1() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 3 of each
        global_state[config.p1_cap_w] = 3.0;
        global_state[config.p1_cap_g] = 3.0;
        global_state[config.p1_cap_b] = 3.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_three_of_each_p2() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 2 has 3 of each
        global_state[config.p2_cap_w] = 3.0;
        global_state[config.p2_cap_g] = 3.0;
        global_state[config.p2_cap_b] = 3.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_4_white() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 4 white
        global_state[config.p1_cap_w] = 4.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_5_gray() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 2 has 5 gray
        global_state[config.p2_cap_g] = 5.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_6_black() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 6 black
        global_state[config.p1_cap_b] = 6.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_board_full() {
        let config = create_test_config();
        let (mut spatial_state, global_state) = create_empty_state(&config);

        // Fill one ring with marble
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_current_player_no_marbles() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1's turn, no marbles in supply or captured
        global_state[config.cur_player] = config.player_1 as f32;
        global_state[config.supply_w] = 0.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.p1_cap_w] = 0.0;
        global_state[config.p1_cap_g] = 0.0;
        global_state[config.p1_cap_b] = 0.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_is_game_over_false_not_enough_captures() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 2 of each (not 3)
        global_state[config.p1_cap_w] = 2.0;
        global_state[config.p1_cap_g] = 2.0;
        global_state[config.p1_cap_b] = 2.0;

        // Board not full
        spatial_state[[config.ring_layer, 3, 3]] = 1.0; // Empty ring

        // Supply has marbles
        global_state[config.supply_w] = 1.0;

        assert!(!is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    // ========================================================================
    // get_game_outcome() tests
    // ========================================================================

    #[test]
    fn test_get_game_outcome_p1_wins_three_of_each() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        global_state[config.p1_cap_w] = 3.0;
        global_state[config.p1_cap_g] = 3.0;
        global_state[config.p1_cap_b] = 3.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p2_wins_three_of_each() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        global_state[config.p2_cap_w] = 3.0;
        global_state[config.p2_cap_g] = 3.0;
        global_state[config.p2_cap_b] = 3.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p1_wins_4_white() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        global_state[config.p1_cap_w] = 4.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p2_wins_5_gray() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        global_state[config.p2_cap_g] = 5.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p1_wins_6_black() {
        let config = create_test_config();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        global_state[config.p1_cap_b] = 6.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_both_lose() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Board full (one ring, one marble)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble

        // Both players have zero captures
        global_state[config.p1_cap_w] = 0.0;
        global_state[config.p1_cap_g] = 0.0;
        global_state[config.p1_cap_b] = 0.0;
        global_state[config.p2_cap_w] = 0.0;
        global_state[config.p2_cap_g] = 0.0;
        global_state[config.p2_cap_b] = 0.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            BOTH_LOSE
        );
    }

    #[test]
    fn test_get_game_outcome_board_full_p1_to_move_p2_wins() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Board full
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble

        // Player 1 to move
        global_state[config.cur_player] = config.player_1 as f32;

        // Player 1 has some captures (not zero)
        global_state[config.p1_cap_w] = 1.0;

        // Player 2 wins (last to move)
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_board_full_p2_to_move_p1_wins() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Board full
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble

        // Player 2 to move
        global_state[config.cur_player] = config.player_2 as f32;

        // Player 2 has some captures (not zero)
        global_state[config.p2_cap_w] = 1.0;

        // Player 1 wins (last to move)
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p1_no_marbles_p2_wins() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Add a ring with a marble so board is not empty (not triggering BOTH_LOSE)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // white marble on ring

        // Player 1's turn
        global_state[config.cur_player] = config.player_1 as f32;

        // No marbles in supply or captured for P1
        global_state[config.supply_w] = 0.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.p1_cap_w] = 0.0;
        global_state[config.p1_cap_g] = 0.0;
        global_state[config.p1_cap_b] = 0.0;

        // P2 has some marbles (to avoid BOTH_LOSE)
        global_state[config.p2_cap_w] = 1.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_get_game_outcome_p2_no_marbles_p1_wins() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Add a ring with a marble so board is not empty (not triggering BOTH_LOSE)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // white marble on ring

        // Player 2's turn
        global_state[config.cur_player] = config.player_2 as f32;

        // No marbles in supply or captured for P2
        global_state[config.supply_w] = 0.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.p2_cap_w] = 0.0;
        global_state[config.p2_cap_g] = 0.0;
        global_state[config.p2_cap_b] = 0.0;

        // P1 has some marbles (to avoid BOTH_LOSE)
        global_state[config.p1_cap_w] = 1.0;

        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    // ========================================================================
    // Blitz mode tests - verify mode-specific win conditions
    // ========================================================================

    #[test]
    fn test_blitz_mode_two_of_each_p1_wins() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 2 of each (Blitz win condition)
        global_state[config.p1_cap_w] = 2.0;
        global_state[config.p1_cap_g] = 2.0;
        global_state[config.p1_cap_b] = 2.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_blitz_mode_three_white_p2_wins() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 2 has 3 white (Blitz win condition)
        global_state[config.p2_cap_w] = 3.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_blitz_mode_four_gray_p1_wins() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has 4 gray (Blitz win condition)
        global_state[config.p1_cap_g] = 4.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_1_WIN
        );
    }

    #[test]
    fn test_blitz_mode_five_black_p2_wins() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (spatial_state, mut global_state) = create_empty_state(&config);

        // Player 2 has 5 black (Blitz win condition)
        global_state[config.p2_cap_b] = 5.0;

        assert!(is_game_over(&spatial_state.view(), &global_state.view(), &config));
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            PLAYER_2_WIN
        );
    }

    #[test]
    fn test_blitz_mode_not_enough_captures() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has only 1 black (not enough for Blitz 2-of-each)
        global_state[config.p1_cap_w] = 2.0;
        global_state[config.p1_cap_g] = 2.0;
        global_state[config.p1_cap_b] = 1.0; // Only 1 black (not enough for Blitz)

        // Add ring so board isn't considered full
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;

        // Supply has marbles
        global_state[config.supply_w] = 1.0;

        // Should NOT be game over - needs 2 of EACH (has only 1 black)
        assert!(!is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_standard_mode_not_enough_blitz_captures() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Player 1 has Blitz win conditions (2 of each)
        // but this should NOT trigger win in Standard mode
        global_state[config.p1_cap_w] = 2.0;
        global_state[config.p1_cap_g] = 2.0;
        global_state[config.p1_cap_b] = 2.0;

        // Add ring so board isn't considered full
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;

        // Supply has marbles
        global_state[config.supply_w] = 1.0;

        // Should NOT be game over - needs 3 of each in Standard mode
        assert!(!is_game_over(&spatial_state.view(), &global_state.view(), &config));
    }

    #[test]
    fn test_blitz_mode_board_full_both_lose() {
        let config = BoardConfig::blitz(37, 1).unwrap();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Board full (one ring, one marble)
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[1, 3, 3]] = 1.0; // White marble

        // Both players have zero captures
        global_state[config.p1_cap_w] = 0.0;
        global_state[config.p1_cap_g] = 0.0;
        global_state[config.p1_cap_b] = 0.0;
        global_state[config.p2_cap_w] = 0.0;
        global_state[config.p2_cap_g] = 0.0;
        global_state[config.p2_cap_b] = 0.0;

        // BOTH_LOSE condition should work in Blitz mode too
        assert_eq!(
            get_game_outcome(&spatial_state.view(), &global_state.view(), &config),
            BOTH_LOSE
        );
    }

    #[test]
    fn test_apply_placement_no_ring_removal() {
        let config = create_test_config();
        let (mut spatial_state, mut global_state) = create_empty_state(&config);

        // Setup: place marble at center with ring present
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        global_state[config.supply_w] = 5.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Count rings before placement
        let rings_before = spatial_state.slice(s![config.ring_layer, .., ..]).sum();

        // Apply placement with no ring removal (None, None)
        apply_placement(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 0, 3, 3, None, None, &config);

        // Check marble placed
        assert_eq!(spatial_state[[1, 3, 3]], 1.0, "Marble should be placed");

        // Check no ring was removed
        let rings_after = spatial_state.slice(s![config.ring_layer, .., ..]).sum();
        assert_eq!(
            rings_after, rings_before,
            "No ring should be removed when None is passed"
        );

        // Check supply decremented
        assert_eq!(global_state[config.supply_w], 4.0, "Supply should be decremented");

        // Check player switched
        assert_eq!(
            global_state[config.cur_player] as usize,
            config.player_2,
            "Turn should pass to next player"
        );
    }
}
