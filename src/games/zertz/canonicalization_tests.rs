#[cfg(test)]
mod tests {
    use super::super::canonicalization::*;
    use super::super::board::BoardConfig;
    use ndarray::Array3;

    #[test]
    fn test_transform_state_with_translation() {
        // Test that transform_state with rotation=0, mirror=false, dy/dx
        // correctly performs translations

        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));

        // Create a simple board with a few rings and marbles
        // Place rings in a pattern
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 2]] = 1.0;
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 4, 3]] = 1.0;

        // Place some marbles
        spatial_state[[config.marble_layers.0, 2, 2]] = 1.0;  // white marble
        spatial_state[[config.marble_layers.0 + 1, 3, 3]] = 1.0;  // gray marble

        // Test various translations
        let test_cases = vec![
            (0, 0),   // Identity - should succeed
            (1, 0),   // Move down - should succeed
            (0, 1),   // Move right - should succeed
            (1, 1),   // Move diagonal - should succeed
            (-1, 0),  // Move up - should succeed
            (0, -1),  // Move left - should succeed
        ];

        for (dy, dx) in test_cases {
            // Use transform_state with no rotation/mirror
            let transformed = transform_state(&spatial_state.view(), &config, 0, false, false, dy, dx);

            // All test cases should succeed
            assert!(transformed.is_some(),
                "Translation should succeed for dy={}, dx={}", dy, dx);

            if let Some(result) = transformed {
                // Verify the result has the right shape
                assert_eq!(result.shape(), spatial_state.shape(),
                    "Shapes should match for translation dy={}, dx={}", dy, dx);

                // For identity transform, result should match original
                if dy == 0 && dx == 0 {
                    for layer in 0..result.shape()[0] {
                        for y in 0..result.shape()[1] {
                            for x in 0..result.shape()[2] {
                                assert!(
                                    (result[[layer, y, x]] - spatial_state[[layer, y, x]]).abs() < 1e-6,
                                    "Identity transform should preserve state"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_transform_state_with_rotation_and_translation() {
        // Test that transform_state can handle both rotation and translation together

        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));

        // Create a simple pattern
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.marble_layers.0, 3, 3]] = 1.0;

        // Apply rotation and translation together
        let result = transform_state(&spatial_state.view(), &config, 1, false, false, 1, 0);

        // Should succeed
        assert!(result.is_some(), "Rotation + translation should work");

        // Verify the result has the right shape
        if let Some(transformed) = result {
            assert_eq!(transformed.shape(), spatial_state.shape());
        }
    }

    #[test]
    fn test_transform_state_invalid_translation() {
        // Test that transform_state returns None for invalid translations

        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));

        // Place a ring at the edge
        spatial_state[[config.ring_layer, 0, 0]] = 1.0;

        // Try to translate off the board
        let result = transform_state(&spatial_state.view(), &config, 0, false, false, -1, 0);

        assert!(result.is_none(), "Translation off board should return None");
    }
}
