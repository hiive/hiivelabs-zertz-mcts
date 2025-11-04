#[cfg(test)]
mod tests {
    use super::super::node::*;
    use crate::games::zertz::BoardConfig;
    use crate::games::ZertzGame;
    use crate::transposition::TranspositionTable;
    use ndarray::{Array1, Array3};
    use std::sync::Arc;

    fn empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let layers = config.layers_per_timestep * config.t + 1;
        let spatial_state = Array3::zeros((layers, config.width, config.width));
        let global_state = Array1::zeros(10);
        (spatial_state, global_state)
    }

    #[test]
    fn node_updates_shared_entry() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let table = TranspositionTable::new(Arc::clone(&game));
        let shared = table.get_or_insert(&spatial_state.view(), &global_state.view());

        let node = MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            Some(Arc::clone(&shared)),
        );
        node.update(0.5);

        assert_eq!(shared.visits(), 1);
        assert!((shared.average_value() - 0.5).abs() < 1e-3);
        assert_eq!(node.get_visits(), 1);
    }

    #[test]
    fn node_without_shared_entry_uses_local_stats() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        node.update(-1.0);

        assert_eq!(node.get_visits(), 1);
        assert!((node.get_value() + 1.0).abs() < 1e-3);
    }

    fn canonical_variant_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let mut global_state = Array1::zeros(10);
        // fill rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        // place a couple of marbles to break symmetry
        spatial_state[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state[config.cur_player] = 0.0;
        (spatial_state, global_state)
    }

    #[test]
    fn canonical_symmetric_nodes_share_stats() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();
        let (spatial_state, global_state) = canonical_variant_state(config);

        // Prepare table and canonical entry
        let table = TranspositionTable::new(Arc::clone(&game));
        let entry = table.get_or_insert(&spatial_state.view(), &global_state.view());

        let node_canonical = Arc::new(MCTSNode::new(
            spatial_state.clone(),
            global_state.clone(),
            Arc::clone(&game),
            Some(Arc::clone(&entry)),
        ));
        node_canonical.update(0.75);

        // Rotate spatial_state state by 60 degrees (same canonical class)
        let rotated =
            crate::games::zertz::canonicalization::transform_state(&config, &spatial_state.view(), 1, false, false, 0, 0, true)
                .expect("Transformation should succeed");
        let rotated_entry = table.get_or_insert(&rotated.view(), &global_state.view());
        let node_rotated = Arc::new(MCTSNode::new(
            rotated.to_owned(),
            global_state.clone(),
            Arc::clone(&game),
            Some(Arc::clone(&rotated_entry)),
        ));
        node_rotated.update(-0.25);

        // Both nodes should refer to the same transposition entry
        assert!(Arc::ptr_eq(&entry, &rotated_entry));
        assert_eq!(entry.visits(), 2);
        assert!((entry.average_value() - 0.25).abs() < 1e-6);

        // Node getters should reflect shared stats
        assert_eq!(node_canonical.get_visits(), 2);
        assert_eq!(node_rotated.get_visits(), 2);
        assert!((node_canonical.get_value() - 0.25).abs() < 1e-6);
        assert!((node_rotated.get_value() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn virtual_loss_adds_and_removes_correctly() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Initial state: 0 visits, 0 value
        assert_eq!(node.get_visits(), 0);
        assert_eq!(node.get_value(), 0.0);

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(node.get_visits(), VIRTUAL_LOSS);
        assert!((node.get_value() + 1.0).abs() < 1e-3); // Should be -1.0 (pessimistic)

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(node.get_visits(), 0);
        assert_eq!(node.get_value(), 0.0);
    }

    #[test]
    fn virtual_loss_with_real_updates() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(node.get_visits(), VIRTUAL_LOSS);

        // Add real update
        node.update(0.5);
        assert_eq!(node.get_visits(), VIRTUAL_LOSS + 1);

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(node.get_visits(), 1);
        assert!((node.get_value() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn virtual_loss_with_shared_stats() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let table = TranspositionTable::new(Arc::clone(&game));
        let shared = table.get_or_insert(&spatial_state.view(), &global_state.view());

        let node = MCTSNode::new(
            spatial_state,
            global_state,
            Arc::clone(&game),
            Some(Arc::clone(&shared)),
        );

        // Add virtual loss
        node.add_virtual_loss();
        assert_eq!(shared.visits(), VIRTUAL_LOSS);
        assert!((shared.average_value() + 1.0).abs() < 1e-3);

        // Remove virtual loss
        node.remove_virtual_loss();
        assert_eq!(shared.visits(), 0);
        assert_eq!(shared.average_value(), 0.0);
    }

    #[test]
    fn fpu_unvisited_node_with_reduction() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Unvisited node with FPU reduction = 0.2
        let parent_visits = 10;
        let parent_value = 0.5;
        let exploration_constant = 1.41;
        let fpu_reduction = Some(0.2);

        let score = node.ucb1_score(parent_visits, parent_value, exploration_constant, fpu_reduction);

        // Expected: -(parent_value - fpu_reduction) + c * sqrt(parent_visits)
        // = -(0.5 - 0.2) + 1.41 * sqrt(10)
        // = -0.3 + 4.459...
        // = 4.159...
        let expected_q = -(parent_value - 0.2);
        let expected_u = exploration_constant * (parent_visits as f32).sqrt();
        let expected = expected_q + expected_u;

        assert!((score - expected).abs() < 1e-6, "FPU score mismatch: got {}, expected {}", score, expected);
    }

    #[test]
    fn fpu_unvisited_node_without_reduction() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Unvisited node with no FPU (standard UCB1)
        let parent_visits = 10;
        let parent_value = 0.5;
        let exploration_constant = 1.41;
        let fpu_reduction = None;

        let score = node.ucb1_score(parent_visits, parent_value, exploration_constant, fpu_reduction);

        // Should return infinity for standard UCB1
        assert_eq!(score, f32::INFINITY, "Without FPU, unvisited nodes should have infinite score");
    }

    #[test]
    fn fpu_visited_node_ignores_fpu() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Visit the node
        node.update(0.6);

        let parent_visits = 10;
        let parent_value = 0.5;
        let exploration_constant = 1.41;

        // Compute score with FPU
        let score_with_fpu = node.ucb1_score(parent_visits, parent_value, exploration_constant, Some(0.2));

        // Compute score without FPU
        let score_without_fpu = node.ucb1_score(parent_visits, parent_value, exploration_constant, None);

        // For visited nodes, FPU should have no effect
        assert_eq!(score_with_fpu, score_without_fpu, "FPU should not affect visited nodes");

        // Verify it matches standard UCB1 formula
        let expected_q = -node.get_value();  // -0.6
        let expected_u = exploration_constant * ((parent_visits as f32).ln() / (node.get_visits() as f32)).sqrt();
        let expected = expected_q + expected_u;

        assert!((score_with_fpu - expected).abs() < 1e-6, "Visited node score should match standard UCB1");
    }

    #[test]
    fn fpu_negative_parent_value() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let (spatial_state, global_state) = empty_state(game.config());
        let node = MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None);

        // Parent has negative value (losing position from parent's perspective)
        let parent_visits = 10;
        let parent_value = -0.4;
        let exploration_constant = 1.41;
        let fpu_reduction = Some(0.2);

        let score = node.ucb1_score(parent_visits, parent_value, exploration_constant, fpu_reduction);

        // Expected: -(parent_value - fpu_reduction) + u
        // = -(-0.4 - 0.2) + u
        // = -(-0.6) + u
        // = 0.6 + u
        let expected_q = -(parent_value - 0.2);
        let expected_u = exploration_constant * (parent_visits as f32).sqrt();
        let expected = expected_q + expected_u;

        assert!((score - expected).abs() < 1e-6, "FPU with negative parent value");
        assert!(expected_q > 0.0, "Negative parent value should result in positive estimated Q for child");
    }
}
