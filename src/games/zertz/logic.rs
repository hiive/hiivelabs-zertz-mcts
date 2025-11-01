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
// Re-export from canonicalization module (single source of truth)
pub use crate::canonicalization::{ax_rot60, ax_mirror_q_axis};

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
#[path = "logic_tests.rs"]
mod logic_tests;
