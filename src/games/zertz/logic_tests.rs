//! Tests for Zertz game logic
//!
//! This file contains unit tests for the Zertz game logic functions.
//! It is compiled only when running tests via #[cfg(test)] in logic.rs.

use super::*;
use ndarray::{s, Array1, Array3};

// ============================================================================
// GAME LOGIC TESTS
// ============================================================================

mod tests {
    use super::*;

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

        // Should have exactly one valid placement (white marble at 3,3 with no removal)
        // Format: (marble_type, dst_y, dst_x, rem_y, rem_x)
        // Sentinel: (dst_y, dst_x) as removal means "no removal"
        assert_eq!(placement_mask[[0, 3, 3, 3, 3]], 1.0);

        // No entries for other marble colours
        assert_eq!(placement_mask[[1, 3, 3, 3, 3]], 0.0);
        assert_eq!(placement_mask[[2, 3, 3, 3, 3]], 0.0);
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

        // Should be able to place white marble from captured pool (no removal)
        // Sentinel: (dst_y, dst_x) as removal means "no removal"
        assert_eq!(placement_mask[[0, 3, 3, 3, 3]], 1.0);
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

        let width = config.width;

        // Should not be able to place on occupied ring (check white marble, no removal)
        // New format: (marble_type, dst_y, dst_x, rem_y, rem_x) with (width, width) = no removal
        assert_eq!(placement_mask[[0, 3, 3, 3, 3]], 0.0);
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
        ).unwrap();

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
        apply_placement(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 0, 3, 3, None, None, &config).unwrap();

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
        let (dest_y, dest_x) = config.dest_from_direction(3, 3, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, dest_y, dest_x, &config);

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
        let (dest_y, dest_x) = config.dest_from_direction(2, 2, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, dest_y, dest_x, &config);

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
        let (dest_y, dest_x) = config.dest_from_direction(3, 3, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, dest_y, dest_x, &config);

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
        let (dest_y, dest_x) = config.dest_from_direction(3, 3, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, dest_y, dest_x, &config);

        // Verify the stale marker at (1,1) was cleared
        assert_eq!(
            spatial_state[[config.capture_layer, 1, 1]], 0.0,
            "Capture layer at (1,1) should be cleared by apply_capture()"
        );
    }

    #[test]
    fn test_apply_capture_marks_landing_on_chain() {
        // Verify that apply_capture() marks the landing position in the capture layer
        // when a chain capture is available (Step 2 of the fix).
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
        let (dest_y, dest_x) = config.dest_from_direction(2, 2, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, dest_y, dest_x, &config);

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
        // Verify that apply_capture() does NOT mark the landing position when
        // no chain capture is available, and DOES switch players.
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
        let (dest_y, dest_x) = config.dest_from_direction(3, 3, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 3, 3, dest_y, dest_x, &config);

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
        // Verify that apply_placement() clears the capture layer at the start,
        // ending any ongoing chain capture sequence.
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
        ).unwrap();

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
        // Verify that once a chain capture begins, ONLY the marble that landed
        // can perform the next capture (no other marbles can capture).
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
        let (dest_y, dest_x) = config.dest_from_direction(2, 2, east_dir);
        apply_capture(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 2, 2, dest_y, dest_x, &config);

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
        ).unwrap();

        // Both D4 and G1 should be captured (2 marbles total)
        assert_eq!(global_state[config.p1_cap_w], 2.0);
        assert_eq!(spatial_state[[1, d4.0, d4.1]], 0.0);
        assert_eq!(spatial_state[[config.ring_layer, d4.0, d4.1]], 0.0);
        assert_eq!(spatial_state[[1, g1.0, g1.1]], 0.0);
        assert_eq!(spatial_state[[config.ring_layer, g1.0, g1.1]], 0.0);
    }

    #[test]
    fn test_isolation_capture_all_fully_occupied_regions() {
        // This test verifies that ALL fully-occupied regions are captured,
        // regardless of their location on the board.

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
        ).unwrap();

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
        ).unwrap();
    }
}

// ============================================================================
// GAME TERMINATION TESTS
// ============================================================================

mod termination_tests {
    use super::*;

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
        apply_placement(&mut spatial_state.view_mut(), &mut global_state.view_mut(), 0, 3, 3, None, None, &config).unwrap();

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

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_is_inbounds_center() {
        let config = create_test_config();
        // Center positions should be in bounds
        assert!(is_inbounds(3, 3, config.width));
        assert!(is_inbounds(0, 0, config.width));
        assert!(is_inbounds(6, 6, config.width));
    }

    #[test]
    fn test_is_inbounds_edges() {
        let config = create_test_config();
        // Edge positions should be in bounds
        assert!(is_inbounds(0, 0, config.width));
        assert!(is_inbounds(0, 6, config.width));
        assert!(is_inbounds(6, 0, config.width));
        assert!(is_inbounds(6, 6, config.width));
    }

    #[test]
    fn test_is_inbounds_negative() {
        let config = create_test_config();
        // Negative coordinates should be out of bounds
        assert!(!is_inbounds(-1, 0, config.width));
        assert!(!is_inbounds(0, -1, config.width));
        assert!(!is_inbounds(-1, -1, config.width));
    }

    #[test]
    fn test_is_inbounds_too_large() {
        let config = create_test_config();
        // Coordinates >= width should be out of bounds
        assert!(!is_inbounds(7, 0, config.width));
        assert!(!is_inbounds(0, 7, config.width));
        assert!(!is_inbounds(7, 7, config.width));
        assert!(!is_inbounds(100, 100, config.width));
    }

    #[test]
    fn test_get_neighbors_center() {
        let config = create_test_config();
        let neighbors = get_neighbors(3, 3, &config);

        // Center position should have 6 neighbors
        assert_eq!(neighbors.len(), 6);

        // Check all neighbors are in bounds
        for (ny, nx) in neighbors.iter() {
            assert!(is_inbounds(*ny as i32, *nx as i32, config.width));
        }
    }

    #[test]
    fn test_get_neighbors_corner() {
        let config = create_test_config();
        let neighbors = get_neighbors(0, 0, &config);

        // Corner should have fewer neighbors (3 neighbors for hexagonal grid)
        assert_eq!(neighbors.len(), 3, "Corner should have 3 neighbors");

        // All should be in bounds
        for (ny, nx) in neighbors.iter() {
            assert!(is_inbounds(*ny as i32, *nx as i32, config.width));
        }
    }

    #[test]
    fn test_get_neighbors_edge() {
        let config = create_test_config();
        let neighbors = get_neighbors(0, 3, &config);

        // Edge position should have 4-5 neighbors
        assert!(neighbors.len() >= 4 && neighbors.len() <= 5);

        // All should be in bounds
        for (ny, nx) in neighbors.iter() {
            assert!(is_inbounds(*ny as i32, *nx as i32, config.width));
        }
    }

    #[test]
    fn test_get_neighbors_no_duplicates() {
        let config = create_test_config();
        let neighbors = get_neighbors(3, 3, &config);

        // Check no duplicates
        for i in 0..neighbors.len() {
            for j in (i+1)..neighbors.len() {
                assert_ne!(neighbors[i], neighbors[j], "Neighbors should be unique");
            }
        }
    }

    #[test]
    fn test_get_jump_destination_horizontal() {
        // Jump horizontally: start (3,3), capture (3,4), land (3,5)
        let (dst_y, dst_x) = get_jump_destination(3, 3, 3, 4);
        assert_eq!((dst_y, dst_x), (3, 5));
    }

    #[test]
    fn test_get_jump_destination_vertical() {
        // Jump vertically: start (3,3), capture (4,3), land (5,3)
        let (dst_y, dst_x) = get_jump_destination(3, 3, 4, 3);
        assert_eq!((dst_y, dst_x), (5, 3));
    }

    #[test]
    fn test_get_jump_destination_diagonal() {
        // Jump diagonally: start (3,3), capture (2,2), land (1,1)
        let (dst_y, dst_x) = get_jump_destination(3, 3, 2, 2);
        assert_eq!((dst_y, dst_x), (1, 1));
    }

    #[test]
    fn test_get_jump_destination_reverse() {
        // Jump in reverse direction: start (5,5), capture (4,4), land (3,3)
        let (dst_y, dst_x) = get_jump_destination(5, 5, 4, 4);
        assert_eq!((dst_y, dst_x), (3, 3));
    }

    #[test]
    fn test_get_marble_type_at_white() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Place white marble at (3,3)
        spatial_state[[1, 3, 3]] = 1.0;

        let marble_type = get_marble_type_at(&spatial_state.view(), 3, 3, &config);
        assert_eq!(marble_type, 'w');
    }

    #[test]
    fn test_get_marble_type_at_gray() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Place gray marble at (4,4)
        spatial_state[[2, 4, 4]] = 1.0;

        let marble_type = get_marble_type_at(&spatial_state.view(), 4, 4, &config);
        assert_eq!(marble_type, 'g');
    }

    #[test]
    fn test_get_marble_type_at_black() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Place black marble at (2,2)
        spatial_state[[3, 2, 2]] = 1.0;

        let marble_type = get_marble_type_at(&spatial_state.view(), 2, 2, &config);
        assert_eq!(marble_type, 'b');
    }

    #[test]
    fn test_get_marble_type_at_empty() {
        let config = create_test_config();
        let (spatial_state, _) = create_empty_state(&config);

        // No marble at (3,3)
        let marble_type = get_marble_type_at(&spatial_state.view(), 3, 3, &config);
        assert_eq!(marble_type, '\0');
    }

    #[test]
    fn test_get_supply_index_white() {
        let config = create_test_config();
        let idx = get_supply_index('w', &config);
        assert_eq!(idx, config.supply_w);
    }

    #[test]
    fn test_get_supply_index_gray() {
        let config = create_test_config();
        let idx = get_supply_index('g', &config);
        assert_eq!(idx, config.supply_g);
    }

    #[test]
    fn test_get_supply_index_black() {
        let config = create_test_config();
        let idx = get_supply_index('b', &config);
        assert_eq!(idx, config.supply_b);
    }

    #[test]
    fn test_get_captured_index_p1_white() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_1, 0); // marble_idx 0 = white
        assert_eq!(idx, config.p1_cap_w);
    }

    #[test]
    fn test_get_captured_index_p1_gray() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_1, 1); // marble_idx 1 = gray
        assert_eq!(idx, config.p1_cap_g);
    }

    #[test]
    fn test_get_captured_index_p1_black() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_1, 2); // marble_idx 2 = black
        assert_eq!(idx, config.p1_cap_b);
    }

    #[test]
    fn test_get_captured_index_p2_white() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_2, 0); // marble_idx 0 = white
        assert_eq!(idx, config.p2_cap_w);
    }

    #[test]
    fn test_get_captured_index_p2_gray() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_2, 1); // marble_idx 1 = gray
        assert_eq!(idx, config.p2_cap_g);
    }

    #[test]
    fn test_get_captured_index_p2_black() {
        let config = create_test_config();
        let idx = get_captured_index(&config, config.player_2, 2); // marble_idx 2 = black
        assert_eq!(idx, config.p2_cap_b);
    }

    #[test]
    fn test_get_regions_single_region() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Create a small connected region: 3x3 grid
        for y in 2..=4 {
            for x in 2..=4 {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        let regions = get_regions(&spatial_state.view(), &config);

        // Should have exactly one region
        assert_eq!(regions.len(), 1);

        // Region should have 9 rings
        assert_eq!(regions[0].len(), 9);
    }

    #[test]
    fn test_get_regions_two_disconnected() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Create two disconnected regions
        // Region 1: (0,0), (0,1)
        spatial_state[[config.ring_layer, 0, 0]] = 1.0;
        spatial_state[[config.ring_layer, 0, 1]] = 1.0;

        // Region 2: (5,5), (5,6)
        spatial_state[[config.ring_layer, 5, 5]] = 1.0;
        spatial_state[[config.ring_layer, 5, 6]] = 1.0;

        let regions = get_regions(&spatial_state.view(), &config);

        // Should have exactly two regions
        assert_eq!(regions.len(), 2);

        // Each region should have 2 rings
        assert_eq!(regions[0].len(), 2);
        assert_eq!(regions[1].len(), 2);
    }

    #[test]
    fn test_get_regions_empty_board() {
        let config = create_test_config();
        let (spatial_state, _) = create_empty_state(&config);

        // No rings placed
        let regions = get_regions(&spatial_state.view(), &config);

        // Should have no regions
        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn test_get_regions_full_board() {
        let config = create_test_config();
        let (mut spatial_state, _) = create_empty_state(&config);

        // Fill entire board with rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        let regions = get_regions(&spatial_state.view(), &config);

        // Should have exactly one region (entire board)
        assert_eq!(regions.len(), 1);

        // Region should have width^2 rings
        assert_eq!(regions[0].len(), config.width * config.width);
    }
}
