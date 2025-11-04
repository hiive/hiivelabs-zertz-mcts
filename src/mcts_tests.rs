#[cfg(test)]
mod tests {
    use super::super::mcts::*;
    use crate::game_trait::MCTSGame;
    use crate::games::zertz::action::ZertzAction;
    use crate::games::zertz::game::ZertzGame;
    use crate::games::zertz::BoardConfig;
    use crate::node::MCTSNode;
    use crate::transposition::TranspositionTable;
    use ndarray::{Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Instant;
    // Note: Most MCTS testing is done via Python integration tests
    // due to the PyO3 boundary (PyReadonlyArray types can only be created from Python)

    #[test]
    fn test_actions_equal() {
        let action1 = ZertzAction::Pass;
        let action2 = ZertzAction::Pass;
        assert_eq!(action1, action2);

        let config = BoardConfig::standard(37, 1).unwrap();
        let width = config.width;

        // Test Placement actions with flat coordinates
        let placement1 = ZertzAction::Placement {
            marble_type: 0,
            dst_flat: 3 * width + 4,          // (y=3, x=4) -> flat
            remove_flat: Some(1 * width + 2), // (y=1, x=2) -> flat
        };
        let placement2 = ZertzAction::Placement {
            marble_type: 0,
            dst_flat: 3 * width + 4,
            remove_flat: Some(1 * width + 2),
        };
        assert_eq!(placement1, placement2);

        // Test Capture actions with flat coordinates
        let (dest_y, dest_x) = config.dest_from_direction(2, 3, 1);

        let capture1 = ZertzAction::Capture {
            src_flat: 2 * width + 3,
            dst_flat: dest_y * width + dest_x,
        };
        let capture2 = ZertzAction::Capture {
            src_flat: 2 * width + 3,
            dst_flat: dest_y * width + dest_x,
        };
        assert_eq!(capture1, capture2);

        // Different actions should not be equal
        assert_ne!(action1, placement1);
        assert_ne!(placement1, capture1);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_virtual_loss_on_expanded_node() {
        // This test verifies that newly expanded nodes get virtual loss
        // added during expand(), so backpropagation can correctly remove it.

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        // Create a simple initial state with rings
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Add some rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        // Set supply
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Create MCTS search instance and root node
        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let root = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // Simulate the select→expand→backprop flow:
        // 1. Add virtual loss to root (what select() does)
        root.add_virtual_loss();

        // 2. Verify root is not fully expanded
        assert!(!root.is_fully_expanded(None));

        // 3. Call expand() - this should add virtual loss to the child
        let child = mcts.expand(Arc::clone(&root), None, false);

        // 4. Verify child has virtual loss (visits should be VIRTUAL_LOSS)
        #[cfg(debug_assertions)]
        assert_eq!(
            child
                .virtual_loss_count
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // 5. Now simulate backpropagation - should work correctly
        child.remove_virtual_loss();
        child.update(0.5);

        root.remove_virtual_loss();
        root.update(-0.5);

        // 6. Verify final state is correct (virtual losses removed, real values added)
        assert_eq!(child.get_visits(), 1);
        assert!((child.get_value() - 0.5).abs() < 1e-3);
        assert_eq!(root.get_visits(), 1);
        assert!((root.get_value() + 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_seed_generation_increments() {
        // Test that seed generation increments each time set_seed() is called
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let mut mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);

        // Initial generation should be 0
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 0);

        // Setting seed should increment generation
        mcts.set_seed(Some(42));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 1);

        // Setting seed again should increment again
        mcts.set_seed(Some(123));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 2);

        // Unsetting seed (None) should also increment
        mcts.set_seed(None);
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 3);

        // Setting same seed again should still increment (invalidate caches)
        mcts.set_seed(Some(42));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_thread_local_cache_invalidation() {
        // Test that thread-local RNG cache is properly invalidated when seed changes
        // This test verifies the generation tracking mechanism works correctly

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);

        // Set initial seed
        let mcts_mut = std::cell::RefCell::new(mcts);
        mcts_mut.borrow_mut().set_seed(Some(12345));
        let gen1 = mcts_mut.borrow().seed_generation.load(Ordering::SeqCst);

        // Use the RNG (this will cache it with current generation)
        use rand::Rng;
        let val1 = mcts_mut.borrow().with_rng(|rng| rng.random_range(0..100));

        // Change seed (should increment generation)
        mcts_mut.borrow_mut().set_seed(Some(67890));
        let gen2 = mcts_mut.borrow().seed_generation.load(Ordering::SeqCst);

        // Generation should have incremented
        assert_eq!(gen2, gen1 + 1);

        // Using RNG again should work with new seed
        // (we can't easily verify it uses the new seed in a unit test without
        // full parallel infrastructure, but we can verify it doesn't panic)
        let val2 = mcts_mut.borrow().with_rng(|rng| rng.random_range(0..100));

        // Both values should be valid (in range)
        assert!(val1 < 100);
        assert!(val2 < 100);
    }

    #[test]
    fn test_transposition_table_not_polluted_during_search() {
        // Verify that running MCTS doesn't pollute the transposition table
        // with empty entries for every legal action checked
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Setup initial board state
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        // Set supply marbles (needed for valid moves)
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Create MCTS with transposition table enabled
        let mut mcts = MCTSSearch::new(
            Arc::clone(&game),
            Some(1.41), // exploration_constant
            None,       // widening_constant
            None,       // fpu_reduction
            None,       // rave_constant
            Some(true), // use_transposition_table
            Some(true), // use_transposition_lookups
        );

        // Initialize transposition table
        mcts.transposition_table = Some(Arc::new(TranspositionTable::new(Arc::clone(&game))));

        // Create root node with transposition lookup
        let shared_entry = mcts
            .transposition_table
            .as_ref()
            .unwrap()
            .get_or_insert(&spatial_state.view(), &global_state.view());

        let root = Arc::new(MCTSNode::new(
            spatial_state.clone(),
            global_state.clone(),
            Arc::clone(&game),
            Some(shared_entry),
        ));

        // Create search options
        let search_options = SearchOptions {
            table: mcts.transposition_table.as_ref().map(Arc::clone),
            use_lookups: true,
        };

        // Run 100 MCTS iterations directly
        let start = Instant::now();
        for _ in 0..100 {
            mcts.run_iteration(Arc::clone(&root), &search_options, None, None, start);
        }

        // Check transposition table size
        let table_size = if let Some(table) = &mcts.transposition_table {
            table.len()
        } else {
            panic!("Transposition table should exist after search");
        };

        // With 100 iterations from starting position, we expect:
        // - ~10-100 unique states explored (grows with tree depth)
        // - NOT hundreds of thousands of entries (iterations × legal_actions)
        //
        // 37-ring board has ~1944 legal actions from start
        // If polluted: 100 iterations × 1944 actions = 194,400 entries
        // Actual with transposition hits: typically < 200 entries
        // (transposition table finds many duplicate states via different move orders)
        assert!(
            table_size < 200,
            "Transposition table too large ({} entries). Expected < 200. \
             Table may be polluted with empty entries from action enumeration.",
            table_size
        );

        // Also verify table has at least some entries (search actually happened)
        assert!(
            table_size > 0,
            "Transposition table should have entries after search"
        );

        println!(
            "Transposition table size after 100 iterations: {} entries",
            table_size
        );
    }

    #[test]
    fn test_collapse_deterministic_sequence_disabled() {
        // Test that collapse returns immediately when enable_deterministic_collapse() is false
        use ndarray::{Array1, Array3};

        // Create a simple test game
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Setup minimal valid state
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let _mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // Create a temporary game that disables collapse for testing
        struct TestGame(Arc<ZertzGame>);
        impl MCTSGame for TestGame {
            type Action = ZertzAction;
            fn enable_deterministic_collapse(&self) -> bool {
                false
            }
            fn get_valid_actions(
                &self,
                s: &ArrayView3<f32>,
                g: &ArrayView1<f32>,
            ) -> Vec<Self::Action> {
                self.0.get_valid_actions(s, g)
            }
            fn apply_action(
                &self,
                s: &mut ArrayViewMut3<f32>,
                g: &mut ArrayViewMut1<f32>,
                a: &Self::Action,
            ) -> Result<(), String> {
                self.0.apply_action(s, g, a)
            }
            fn is_terminal(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> bool {
                self.0.is_terminal(s, g)
            }
            fn get_outcome(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> i8 {
                self.0.get_outcome(s, g)
            }
            fn get_current_player(&self, g: &ArrayView1<f32>) -> usize {
                self.0.get_current_player(g)
            }
            fn spatial_shape(&self) -> (usize, usize, usize) {
                self.0.spatial_shape()
            }
            fn global_size(&self) -> usize {
                self.0.global_size()
            }
            fn evaluate_heuristic(
                &self,
                s: &ArrayView3<f32>,
                g: &ArrayView1<f32>,
                p: usize,
            ) -> f32 {
                self.0.evaluate_heuristic(s, g, p)
            }
            fn canonicalize_state(
                &self,
                s: &ArrayView3<f32>,
                g: &ArrayView1<f32>,
            ) -> (Array3<f32>, Array1<f32>) {
                self.0.canonicalize_state(s, g)
            }
            fn hash_state(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> u64 {
                self.0.hash_state(s, g)
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }

        let test_game = Arc::new(TestGame(Arc::clone(&game)));
        let test_node = Arc::new(MCTSNode::new(
            node.spatial_state.clone(),
            node.global_state.clone(),
            test_game,
            None,
        ));

        let test_mcts = MCTSSearch::new(
            Arc::clone(&test_node.game),
            None,
            None,
            None,
            None,
            None,
            None,
        );

        // collapse should return immediately without traversing
        let result = test_mcts.collapse_deterministic_sequence(Arc::clone(&test_node));

        // Should return the same node (pointer equality)
        assert!(Arc::ptr_eq(&result, &test_node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_terminal_state() {
        // Test that collapse stops immediately at terminal states
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Create terminal state (no rings left, game over)
        // All rings removed, player 1 has won via sufficient captures
        global_state[config.p1_cap_w] = 4.0; // Player 1 has 4 white
        global_state[config.p1_cap_g] = 5.0; // 5 gray
        global_state[config.p1_cap_b] = 3.0; // 3 black
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // Verify it's actually terminal
        assert!(game.is_terminal(&node.spatial_state.view(), &node.global_state.view()));

        // collapse should return immediately
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should return the same node
        assert!(Arc::ptr_eq(&result, &node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_choice_point() {
        // Test that collapse stops at nodes with multiple legal actions (choice points)
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Setup initial board state with many possible moves
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // Verify there are multiple actions available
        let actions = game.get_valid_actions(&node.spatial_state.view(), &node.global_state.view());
        assert!(actions.len() > 1, "Test requires multiple actions at root");

        // collapse should return immediately (not a forced move)
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should return the same node
        assert!(Arc::ptr_eq(&result, &node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_single_forced_move() {
        // Test that collapse traverses a single forced move
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Create a state with exactly one legal action:
        // - Only one ring available for placement
        // - Only white marbles in supply (only one marble type available)
        let center_y = config.width / 2;
        let center_x = config.width / 2;
        spatial_state[[config.ring_layer, center_y, center_x]] = 1.0;

        global_state[config.supply_w] = 1.0; // Only white available
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // Verify there's exactly one action
        let actions = game.get_valid_actions(&node.spatial_state.view(), &node.global_state.view());
        assert_eq!(actions.len(), 1, "Test requires exactly one action");

        // collapse should create a child and return it
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should NOT return the same node (should have moved to child)
        assert!(!Arc::ptr_eq(&result, &node));

        // Verify the node has one child now
        let children = node.children.read().unwrap();
        assert_eq!(
            children.len(),
            1,
            "Parent should have exactly one child after collapse"
        );
    }

    #[test]
    fn test_collapse_deterministic_sequence_reuses_existing_child() {
        // Test that collapse reuses existing children instead of creating duplicates
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);

        // Create forced move state
        let center_y = config.width / 2;
        let center_x = config.width / 2;
        spatial_state[[config.ring_layer, center_y, center_x]] = 1.0;
        global_state[config.supply_w] = 1.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            None,
        ));

        // First collapse - creates child
        let result1 = mcts.collapse_deterministic_sequence(Arc::clone(&node));
        let child1_ptr = Arc::as_ptr(&result1);

        // Second collapse - should reuse child
        let result2 = mcts.collapse_deterministic_sequence(Arc::clone(&node));
        let child2_ptr = Arc::as_ptr(&result2);

        // Should return the same child (pointer equality)
        assert_eq!(
            child1_ptr, child2_ptr,
            "Collapse should reuse existing child"
        );

        // Verify still only one child
        let children = node.children.read().unwrap();
        assert_eq!(children.len(), 1, "Should still have exactly one child");
    }

    #[test]
    fn test_collapse_deterministic_sequence_depth_limit() {
        // Test that collapse respects MAX_DETERMINISTIC_DEPTH to prevent infinite loops
        // We'll create a custom game that always returns a forced action
        use ndarray::{Array1, Array3};

        #[derive(Clone)]
        struct InfiniteForceGame;

        impl MCTSGame for InfiniteForceGame {
            type Action = ZertzAction;

            fn enable_deterministic_collapse(&self) -> bool {
                true
            }

            fn get_forced_action(
                &self,
                _actions: &[Self::Action],
                _spatial_state: &ArrayView3<f32>,
                _global_state: &ArrayView1<f32>,
            ) -> Option<Self::Action> {
                // Always return a forced Pass action (simulating infinite loop)
                Some(ZertzAction::Pass)
            }

            fn get_valid_actions(
                &self,
                _s: &ArrayView3<f32>,
                _g: &ArrayView1<f32>,
            ) -> Vec<Self::Action> {
                vec![ZertzAction::Pass]
            }

            fn apply_action(
                &self,
                _s: &mut ArrayViewMut3<f32>,
                g: &mut ArrayViewMut1<f32>,
                _a: &Self::Action,
            ) -> Result<(), String> {
                // Just flip player to simulate state change
                let cur = g[0] as usize;
                g[0] = if cur == 0 { 1.0 } else { 0.0 };
                Ok(())
            }

            fn is_terminal(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> bool {
                false // Never terminal
            }

            fn get_outcome(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> i8 {
                0
            }

            fn get_current_player(&self, g: &ArrayView1<f32>) -> usize {
                g[0] as usize
            }

            fn spatial_shape(&self) -> (usize, usize, usize) {
                (1, 7, 7)
            }

            fn global_size(&self) -> usize {
                10
            }

            fn evaluate_heuristic(
                &self,
                _s: &ArrayView3<f32>,
                _g: &ArrayView1<f32>,
                _p: usize,
            ) -> f32 {
                0.0
            }

            fn canonicalize_state(
                &self,
                s: &ArrayView3<f32>,
                g: &ArrayView1<f32>,
            ) -> (Array3<f32>, Array1<f32>) {
                (s.to_owned(), g.to_owned())
            }

            fn hash_state(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> u64 {
                0
            }

            fn name(&self) -> &str {
                "InfiniteForce"
            }
        }

        let game = Arc::new(InfiniteForceGame);
        let spatial_state = Array3::zeros((1, 7, 7));
        let global_state = Array1::zeros(10);

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, game, None));

        // This should not hang - depth limit should prevent infinite loop
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should have stopped due to depth limit
        assert!(
            !Arc::ptr_eq(&result, &node),
            "Should have moved at least one step"
        );

        // Count depth by traversing children
        let mut depth = 0;
        let mut current = node;
        while depth < MCTSSearch::<InfiniteForceGame>::MAX_DETERMINISTIC_DEPTH + 10 {
            let next_child = {
                let children = current.children.read().unwrap();
                if children.is_empty() {
                    break;
                }
                assert_eq!(
                    children.len(),
                    1,
                    "Should have exactly one child at each level"
                );
                Arc::clone(children.values().next().unwrap())
            };
            current = next_child;
            depth += 1;
        }

        // Should have stopped at MAX_DETERMINISTIC_DEPTH
        assert_eq!(
            depth,
            MCTSSearch::<InfiniteForceGame>::MAX_DETERMINISTIC_DEPTH,
            "Collapse should stop at MAX_DETERMINISTIC_DEPTH"
        );
    }
}
