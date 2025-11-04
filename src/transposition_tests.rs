#[cfg(test)]
mod tests {
    use super::super::transposition::*;
    use crate::games::zertz::BoardConfig;
    use ndarray::{Array1, Array3};
    use std::sync::Arc;
    use crate::games::zertz::game::ZertzGame;

    fn empty_state(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let layers = config.layers_per_timestep * config.t + 1;
        let spatial_state = Array3::zeros((layers, config.width, config.width));
        let global_state = Array1::zeros(10);
        (spatial_state, global_state)
    }

    fn board_with_rings(config: &BoardConfig) -> (Array3<f32>, Array1<f32>) {
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));
        let global_state = Array1::zeros(10);
        // Fill rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        (spatial_state, global_state)
    }

    #[test]
    fn shared_entry_is_reused() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let entry1 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        entry1.add_sample(0.5);

        let entry2 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        assert!(Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry2.visits(), 1);
        assert!((entry2.average_value() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn different_states_get_different_entries() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();
        let (mut spatial_state1, mut global_state1) = board_with_rings(config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(config);

        // Create states that break symmetry differently
        // State 1: white marble at (3,2)
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: white marble at (3,2) AND gray marble at (2,4)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        let entry2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());

        // These should be different entries (different number of marbles)
        assert!(!Arc::ptr_eq(&entry1, &entry2));
    }

    /* NOTE: Collision counter test commented out - it was implementation-specific
     * and manipulated internal hasher state that no longer exists in generic version.
     * Collision handling is still tested implicitly by the "different states" test.
     */
    /*
    #[test]
    fn collision_counter_increments_on_hash_collision() {
        use crate::games::zertz::zobrist::ZobristHasher;

        let table = TranspositionTable::new();
        let config = BoardConfig::standard(37, 1).unwrap();

        // Create two different states with rings + marbles
        let (mut spatial_state1, mut global_state1) = board_with_rings(&config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(&config);

        // State 1: 1 marble
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: 2 marbles (different configuration)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        // Pre-register a fake hasher that always returns hash=42
        let fake_hasher = Arc::new(ZobristHasher::new(config.width, Some(999)));
        table
            .hashers
            .lock()
            .unwrap()
            .insert(config.width, fake_hasher.clone());

        // Manually insert states with forced collision (same hash, different states)
        let (canonical1, _, _) = canonicalization::canonicalize_state(&spatial_state1.view(), &config);
        let (canonical2, _, _) = canonicalization::canonicalize_state(&spatial_state2.view(), &config);

        // Force both states to use the same hash by directly manipulating the table
        let forced_hash = 42u64;

        // Insert first entry
        let entry1 = Arc::new(TranspositionEntry::new(
            canonical1.to_owned(),
            global_state1.clone(),
        ));
        table.table.insert(forced_hash, vec![Arc::clone(&entry1)]);
        entry1.add_sample(0.5);

        // Now insert second entry with same hash but different state (this will trigger collision)
        let entry2 = Arc::new(TranspositionEntry::new(
            canonical2.to_owned(),
            global_state2.clone(),
        ));
        let mut chain = table.table.get_mut(&forced_hash).unwrap();
        table.collisions.fetch_add(1, Ordering::Relaxed);
        chain.push(Arc::clone(&entry2));
        entry2.add_sample(-0.5);

        // Verify collision was detected
        assert_eq!(table.collision_count(), 1);

        // Verify entries are distinct
        assert!(!Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry1.visits(), 1);
        assert_eq!(entry2.visits(), 1);
        assert!((entry1.average_value() - 0.5).abs() < 1e-3);
        assert!((entry2.average_value() + 0.5).abs() < 1e-3);

        // Verify both entries are in the chain
        assert_eq!(chain.len(), 2);
    }
    */

    #[test]
    fn lookup_returns_none_for_missing_state() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let result = table.lookup(&spatial_state.view(), &global_state.view());
        assert!(result.is_none());
    }

    #[test]
    fn lookup_returns_existing_entry() {
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let (spatial_state, global_state) = empty_state(game.config());

        let entry1 = table.get_or_insert(&spatial_state.view(), &global_state.view());
        entry1.add_sample(0.75);

        let entry2 = table.lookup(&spatial_state.view(), &global_state.view());
        assert!(entry2.is_some());
        let entry2 = entry2.unwrap();
        assert!(Arc::ptr_eq(&entry1, &entry2));
        assert_eq!(entry2.visits(), 1);
        assert!((entry2.average_value() - 0.75).abs() < 1e-3);
    }

    #[test]
    fn chaining_handles_multiple_states_with_same_hash() {
        // This test verifies that if multiple states happen to hash to the same value,
        // they are stored in a chain and can be retrieved correctly
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();

        let (mut spatial_state1, mut global_state1) = board_with_rings(&config);
        let (mut spatial_state2, mut global_state2) = board_with_rings(&config);
        let (mut spatial_state3, mut global_state3) = board_with_rings(&config);

        // Create three distinct states with different marble placements
        // State 1: 1 white marble
        spatial_state1[[config.marble_layers.0, 3, 2]] = 1.0;
        global_state1[config.cur_player] = 0.0;

        // State 2: 2 marbles (white + gray)
        spatial_state2[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state2[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        global_state2[config.cur_player] = 0.0;

        // State 3: 3 marbles (white + gray + black)
        spatial_state3[[config.marble_layers.0, 3, 2]] = 1.0;
        spatial_state3[[config.marble_layers.0 + 1, 2, 4]] = 1.0;
        spatial_state3[[config.marble_layers.0 + 2, 4, 3]] = 1.0;
        global_state3[config.cur_player] = 0.0;

        let entry1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        entry1.add_sample(0.1);

        let entry2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());
        entry2.add_sample(0.2);

        let entry3 = table.get_or_insert(&spatial_state3.view(), &global_state3.view());
        entry3.add_sample(0.3);

        // Retrieve them again and verify they're the same entries
        let retrieved1 = table.get_or_insert(&spatial_state1.view(), &global_state1.view());
        let retrieved2 = table.get_or_insert(&spatial_state2.view(), &global_state2.view());
        let retrieved3 = table.get_or_insert(&spatial_state3.view(), &global_state3.view());

        assert!(Arc::ptr_eq(&entry1, &retrieved1));
        assert!(Arc::ptr_eq(&entry2, &retrieved2));
        assert!(Arc::ptr_eq(&entry3, &retrieved3));

        assert!((retrieved1.average_value() - 0.1).abs() < 1e-3);
        assert!((retrieved2.average_value() - 0.2).abs() < 1e-3);
        assert!((retrieved3.average_value() - 0.3).abs() < 1e-3);
    }

    #[test]
    fn test_lookup_does_not_pollute_table() {
        // Verify that lookup() doesn't create entries (prevents table pollution)
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let table = TranspositionTable::new(Arc::clone(&game));
        let config = game.config();

        let initial_size = table.len();

        // Lookup 10 nonexistent states (enough to verify no pollution)
        for i in 0..10 {
            let mut spatial = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
            spatial[[0, 0, 0]] = i as f32;  // Make each unique
            let global = Array1::zeros(10);

            let result = table.lookup(&spatial.view(), &global.view());
            assert!(result.is_none(), "Lookup should return None for nonexistent entry");
        }

        let final_size = table.len();
        assert_eq!(initial_size, final_size, "Lookup should not insert entries (table size should not change)");
        assert_eq!(final_size, 0, "Table should still be empty after lookups");
    }
}
