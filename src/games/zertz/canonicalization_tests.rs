#[cfg(test)]
mod tests {
    use super::super::board::BoardConfig;
    use super::super::canonicalization::*;
    use ndarray::Array3;

    // ============================================================================
    // TRANSFORM STRING PARSING TESTS
    // ============================================================================

    #[test]
    fn test_parse_rot_component_pure_rotation() {
        // Test pure rotations: R0, R60, R120, etc.
        assert_eq!(parse_rot_component("R0"), (0, false, false));
        assert_eq!(parse_rot_component("R60"), (1, false, false));
        assert_eq!(parse_rot_component("R120"), (2, false, false));
        assert_eq!(parse_rot_component("R180"), (3, false, false));
        assert_eq!(parse_rot_component("R240"), (4, false, false));
        assert_eq!(parse_rot_component("R300"), (5, false, false));
    }

    #[test]
    fn test_parse_rot_component_mirror_then_rotate() {
        // Test mirror-then-rotate: MR0, MR60, MR120, etc.
        assert_eq!(parse_rot_component("MR0"), (0, true, false));
        assert_eq!(parse_rot_component("MR60"), (1, true, false));
        assert_eq!(parse_rot_component("MR120"), (2, true, false));
        assert_eq!(parse_rot_component("MR180"), (3, true, false));
    }

    #[test]
    fn test_parse_rot_component_rotate_then_mirror() {
        // Test rotate-then-mirror: R0M, R60M, R120M, etc.
        assert_eq!(parse_rot_component("R0M"), (0, true, true));
        assert_eq!(parse_rot_component("R60M"), (1, true, true));
        assert_eq!(parse_rot_component("R120M"), (2, true, true));
        assert_eq!(parse_rot_component("R180M"), (3, true, true));
    }

    #[test]
    fn test_parse_transform_simple() {
        // Single component
        assert_eq!(parse_transform("R0"), Vec::<String>::new()); // R0 is filtered out
        assert_eq!(parse_transform("R60"), vec!["R60"]);
        assert_eq!(parse_transform("MR120"), vec!["MR120"]);
        assert_eq!(parse_transform("T1,2"), vec!["T1,2"]);
    }

    #[test]
    fn test_parse_transform_compound() {
        // Multiple components
        assert_eq!(parse_transform("T1,2_R60"), vec!["T1,2", "R60"]);
        assert_eq!(parse_transform("R60_T1,2"), vec!["R60", "T1,2"]);
        assert_eq!(parse_transform("T-1,0_MR120"), vec!["T-1,0", "MR120"]);
    }

    #[test]
    fn test_parse_transform_filters_identity() {
        // R0 should be filtered out
        assert_eq!(parse_transform("R0_T1,2"), vec!["T1,2"]);
        assert_eq!(parse_transform("T1,2_R0"), vec!["T1,2"]);
        assert_eq!(parse_transform("R0_R0"), Vec::<String>::new());
    }

    // ============================================================================
    // INVERSE TRANSFORM TESTS
    // ============================================================================

    #[test]
    fn test_inverse_transform_identity() {
        assert_eq!(inverse_transform_name("R0"), "R0");
    }

    #[test]
    fn test_inverse_transform_pure_rotation() {
        // R(k)⁻¹ = R(-k) = R(360 - k)
        assert_eq!(inverse_transform_name("R60"), "R300");
        assert_eq!(inverse_transform_name("R120"), "R240");
        assert_eq!(inverse_transform_name("R180"), "R180"); // Self-inverse
        assert_eq!(inverse_transform_name("R240"), "R120");
        assert_eq!(inverse_transform_name("R300"), "R60");
    }

    #[test]
    fn test_inverse_transform_mirror_then_rotate() {
        // MR(k)⁻¹ = R(-k)M
        assert_eq!(inverse_transform_name("MR0"), "R0M");
        assert_eq!(inverse_transform_name("MR60"), "R300M");
        assert_eq!(inverse_transform_name("MR120"), "R240M");
        assert_eq!(inverse_transform_name("MR180"), "R180M"); // Special case
        assert_eq!(inverse_transform_name("MR240"), "R120M");
        assert_eq!(inverse_transform_name("MR300"), "R60M");
    }

    #[test]
    fn test_inverse_transform_rotate_then_mirror() {
        // R(k)M⁻¹ = MR(-k)
        assert_eq!(inverse_transform_name("R0M"), "MR0");
        assert_eq!(inverse_transform_name("R60M"), "MR300");
        assert_eq!(inverse_transform_name("R120M"), "MR240");
        assert_eq!(inverse_transform_name("R180M"), "MR180"); // Special case
        assert_eq!(inverse_transform_name("R240M"), "MR120");
        assert_eq!(inverse_transform_name("R300M"), "MR60");
    }

    #[test]
    fn test_inverse_transform_translation() {
        // T(dy,dx)⁻¹ = T(-dy,-dx)
        assert_eq!(inverse_transform_name("T0,0"), "T0,0");
        assert_eq!(inverse_transform_name("T1,2"), "T-1,-2");
        assert_eq!(inverse_transform_name("T-1,0"), "T1,0");
        assert_eq!(inverse_transform_name("T0,-3"), "T0,3");
        assert_eq!(inverse_transform_name("T-2,-5"), "T2,5");
    }

    #[test]
    fn test_inverse_transform_compound_translation_first() {
        // T_R form: inverse is R⁻¹_T⁻¹
        assert_eq!(inverse_transform_name("T1,2_R60"), "R300_T-1,-2");
        assert_eq!(inverse_transform_name("T-1,0_MR120"), "R240M_T1,0");
        assert_eq!(inverse_transform_name("T0,1_R180M"), "MR180_T0,-1");
    }

    #[test]
    fn test_inverse_transform_compound_rotation_first() {
        // R_T form: inverse is T⁻¹_R⁻¹
        assert_eq!(inverse_transform_name("R60_T1,2"), "T-1,-2_R300");
        assert_eq!(inverse_transform_name("MR120_T-1,0"), "T1,0_R240M");
        assert_eq!(inverse_transform_name("R180M_T0,1"), "T0,-1_MR180");
    }

    // ============================================================================
    // ROUND-TRIP TESTS
    // ============================================================================

    #[test]
    fn test_roundtrip_pure_rotations() {
        let rotations = vec!["R60", "R120", "R180", "R240", "R300"];
        for rot in rotations {
            let inverse = inverse_transform_name(rot);
            let roundtrip = inverse_transform_name(&inverse);
            assert_eq!(
                rot, roundtrip,
                "Round-trip failed for {}: {} -> {} -> {}",
                rot, rot, inverse, roundtrip
            );
        }
    }

    #[test]
    fn test_roundtrip_mirror_then_rotate() {
        let transforms = vec!["MR0", "MR60", "MR120", "MR180", "MR240", "MR300"];
        for trans in transforms {
            let inverse = inverse_transform_name(trans);
            let roundtrip = inverse_transform_name(&inverse);
            assert_eq!(
                trans, roundtrip,
                "Round-trip failed for {}: {} -> {} -> {}",
                trans, trans, inverse, roundtrip
            );
        }
    }

    #[test]
    fn test_roundtrip_rotate_then_mirror() {
        let transforms = vec!["R0M", "R60M", "R120M", "R180M", "R240M", "R300M"];
        for trans in transforms {
            let inverse = inverse_transform_name(trans);
            let roundtrip = inverse_transform_name(&inverse);
            assert_eq!(
                trans, roundtrip,
                "Round-trip failed for {}: {} -> {} -> {}",
                trans, trans, inverse, roundtrip
            );
        }
    }

    #[test]
    fn test_roundtrip_translations() {
        let translations = vec!["T1,0", "T0,1", "T-1,0", "T0,-1", "T2,3", "T-5,-7"];
        for trans in translations {
            let inverse = inverse_transform_name(trans);
            let roundtrip = inverse_transform_name(&inverse);
            assert_eq!(
                trans, roundtrip,
                "Round-trip failed for {}: {} -> {} -> {}",
                trans, trans, inverse, roundtrip
            );
        }
    }

    #[test]
    fn test_roundtrip_compound_transforms() {
        let compounds = vec![
            "T1,2_R60",
            "T-1,0_MR120",
            "T0,1_R180M",
            "R60_T1,2",
            "MR120_T-1,0",
            "R180M_T0,1",
        ];
        for trans in compounds {
            let inverse = inverse_transform_name(trans);
            let roundtrip = inverse_transform_name(&inverse);
            assert_eq!(
                trans, roundtrip,
                "Round-trip failed for {}: {} -> {} -> {}",
                trans, trans, inverse, roundtrip
            );
        }
    }

    #[test]
    fn test_roundtrip_state_transforms() {
        // Test that applying a transform and then its inverse returns the original state
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Create a non-symmetric pattern
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 2]] = 1.0;
        spatial_state[[config.marble_layers.0, 2, 2]] = 1.0; // White marble

        let original = spatial_state.clone();

        // Test various transforms
        let test_cases = vec![
            (1, false, false, 0, 0), // R60
            (2, false, false, 0, 0), // R120
            (3, true, false, 0, 0),  // MR180
            (1, true, true, 0, 0),   // R60M
        ];

        for (rot60_k, mirror, mirror_first, dy, dx) in test_cases {
            // Apply forward transform
            let transformed = transform_state(
                &config,
                &spatial_state.view(),
                rot60_k,
                mirror,
                mirror_first,
                dy,
                dx,
                true,
            )
            .expect("Forward transform should succeed");

            // Apply inverse transform
            let restored = transform_state(
                &config,
                &transformed.view(),
                -rot60_k, // Inverse rotation
                mirror,
                !mirror_first, // Swap mirror order for inverse
                -dy,           // Inverse translation
                -dx,
                false, // Inverse order
            )
            .expect("Inverse transform should succeed");

            // Check that we got back to original
            for layer in 0..original.shape()[0] {
                for y in 0..original.shape()[1] {
                    for x in 0..original.shape()[2] {
                        assert!(
                            (restored[[layer, y, x]] - original[[layer, y, x]]).abs() < 1e-6,
                            "Round-trip failed at [{},{},{}]: got {}, expected {} for transform (rot={}, mirror={}, mirror_first={})",
                            layer, y, x, restored[[layer, y, x]], original[[layer, y, x]],
                            rot60_k, mirror, mirror_first
                        );
                    }
                }
            }
        }
    }

    // ============================================================================
    // BOUNDING BOX TESTS
    // ============================================================================

    #[test]
    fn test_bounding_box_full_board() {
        // Test that a full board has expected bounding box
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Fill the board with rings according to the standard layout
        let layout = build_layout_mask(&config);
        for y in 0..config.width {
            for x in 0..config.width {
                if layout[y][x] {
                    spatial_state[[config.ring_layer, y, x]] = 1.0;
                }
            }
        }

        let bbox = bounding_box(&config, &spatial_state.view());
        assert!(bbox.is_some(), "Full board should have bounding box");

        let (min_y, max_y, min_x, max_x) = bbox.unwrap();
        // 37-ring board is 7x7, but corners may be empty
        assert!(min_y < config.width);
        assert!(max_y < config.width);
        assert!(min_x < config.width);
        assert!(max_x < config.width);
        assert!(min_y <= max_y);
        assert!(min_x <= max_x);
    }

    #[test]
    fn test_bounding_box_after_ring_removal() {
        // Test bounding box after removing edge rings
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Fill the board with rings
        let layout = build_layout_mask(&config);
        for y in 0..config.width {
            for x in 0..config.width {
                if layout[y][x] {
                    spatial_state[[config.ring_layer, y, x]] = 1.0;
                }
            }
        }

        // Get initial bounding box
        let bbox_before = bounding_box(&config, &spatial_state.view());
        assert!(bbox_before.is_some());
        let (min_y_before, max_y_before, min_x_before, max_x_before) = bbox_before.unwrap();

        // Remove rings from one edge to reduce bounding box
        // Remove leftmost column
        for y in 0..config.width {
            spatial_state[[config.ring_layer, y, 0]] = 0.0;
        }

        // Get new bounding box
        let bbox_after = bounding_box(&config, &spatial_state.view());
        assert!(
            bbox_after.is_some(),
            "Board with removed edges should have bounding box"
        );

        let (min_y_after, max_y_after, min_x_after, max_x_after) = bbox_after.unwrap();

        // Bounding box should be reduced (min_x should have increased)
        assert!(
            min_x_after > min_x_before
                || max_x_after < max_x_before
                || min_y_after > min_y_before
                || max_y_after < max_y_before,
            "Expected reduced bounding box after removing edge rings"
        );
    }

    #[test]
    fn test_bounding_box_empty_board() {
        // Test that empty board (no rings) returns None
        let config = BoardConfig::standard(37, 1).unwrap();
        let spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        let bbox = bounding_box(&config, &spatial_state.view());
        assert!(
            bbox.is_none(),
            "Empty board should return None for bounding box"
        );
    }

    // ============================================================================
    // TRANSLATION VALIDATION TESTS
    // ============================================================================

    #[test]
    fn test_get_translations_empty_board() {
        // Empty board should return identity translation only
        let config = BoardConfig::standard(37, 1).unwrap();
        let spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        let translations = get_translations(&config, &spatial_state.view());

        assert_eq!(translations.len(), 1);
        assert_eq!(translations[0], ("T0,0".to_string(), 0, 0));
    }

    #[test]
    fn test_get_translations_validates_bounds() {
        // Create a small board with rings in corner
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Place rings in bottom-left corner (position that can't move left or down)
        spatial_state[[config.ring_layer, 0, 0]] = 1.0;

        let translations = get_translations(&config, &spatial_state.view());

        // Should only include translations that move right/up (not left/down from corner)
        for (_name, dy, dx) in translations.iter() {
            // Verify translation would keep ring in bounds
            let new_y = 0 + dy;
            let new_x = 0 + dx;
            assert!(new_y >= 0 && new_y < config.width as i32);
            assert!(new_x >= 0 && new_x < config.width as i32);
        }
    }

    #[test]
    fn test_get_translations_validates_layout() {
        // Test that translations respect board layout (hexagonal shape)
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Place a ring at valid position
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;

        let translations = get_translations(&config, &spatial_state.view());

        // All returned translations should actually succeed when applied
        for (_name, dy, dx) in translations.iter() {
            let translated = transform_state(
                &config,
                &spatial_state.view(),
                0,     // no rotation
                false, // no mirror
                false, // mirror_first (irrelevant)
                *dy,
                *dx,
                true, // translate_first
            );
            assert!(
                translated.is_some(),
                "Translation ({}, {}) should succeed",
                dy,
                dx
            );
        }
    }

    #[test]
    fn test_get_translations_includes_identity() {
        // Identity translation (0, 0) should always be included
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Place some rings
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 2]] = 1.0;

        let translations = get_translations(&config, &spatial_state.view());

        // Should include identity
        assert!(
            translations
                .iter()
                .any(|(name, dy, dx)| { name == "T0,0" && *dy == 0 && *dx == 0 }),
            "Should include identity translation"
        );
    }

    #[test]
    fn test_get_translations_multiple_rings() {
        // Test with multiple rings - should validate all rings fit after translation
        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Place rings in a pattern
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 2]] = 1.0;
        spatial_state[[config.ring_layer, 4, 4]] = 1.0;

        let translations = get_translations(&config, &spatial_state.view());

        // Verify every translation succeeds
        for (name, dy, dx) in translations.iter() {
            let translated = transform_state(
                &config,
                &spatial_state.view(),
                0,
                false,
                false,
                *dy,
                *dx,
                true,
            );
            assert!(
                translated.is_some(),
                "Translation {} ({}, {}) should succeed for multi-ring pattern",
                name,
                dy,
                dx
            );
        }
    }

    // ============================================================================
    // EXISTING TESTS
    // ============================================================================

    #[test]
    fn test_transform_state_with_translation() {
        // Test that transform_state with rotation=0, mirror=false, dy/dx
        // correctly performs translations

        let config = BoardConfig::standard(37, 1).unwrap();
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Create a simple board with a few rings and marbles
        // Place rings in a pattern
        spatial_state[[config.ring_layer, 2, 2]] = 1.0;
        spatial_state[[config.ring_layer, 2, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 2]] = 1.0;
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 4, 3]] = 1.0;

        // Place some marbles
        spatial_state[[config.marble_layers.0, 2, 2]] = 1.0; // white marble
        spatial_state[[config.marble_layers.0 + 1, 3, 3]] = 1.0; // gray marble

        // Test various translations
        let test_cases = vec![
            (0, 0),  // Identity - should succeed
            (1, 0),  // Move down - should succeed
            (0, 1),  // Move right - should succeed
            (1, 1),  // Move diagonal - should succeed
            (-1, 0), // Move up - should succeed
            (0, -1), // Move left - should succeed
        ];

        for (dy, dx) in test_cases {
            // Use transform_state with no rotation/mirror
            let transformed = transform_state(
                &config,
                &spatial_state.view(),
                0,
                false,
                false,
                dy,
                dx,
                true,
            );

            // All test cases should succeed
            assert!(
                transformed.is_some(),
                "Translation should succeed for dy={}, dx={}",
                dy,
                dx
            );

            if let Some(result) = transformed {
                // Verify the result has the right shape
                assert_eq!(
                    result.shape(),
                    spatial_state.shape(),
                    "Shapes should match for translation dy={}, dx={}",
                    dy,
                    dx
                );

                // For identity transform, result should match original
                if dy == 0 && dx == 0 {
                    for layer in 0..result.shape()[0] {
                        for y in 0..result.shape()[1] {
                            for x in 0..result.shape()[2] {
                                assert!(
                                    (result[[layer, y, x]] - spatial_state[[layer, y, x]]).abs()
                                        < 1e-6,
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
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Create a simple pattern
        spatial_state[[config.ring_layer, 3, 3]] = 1.0;
        spatial_state[[config.ring_layer, 3, 4]] = 1.0;
        spatial_state[[config.marble_layers.0, 3, 3]] = 1.0;

        // Apply rotation and translation together
        let result = transform_state(&config, &spatial_state.view(), 1, false, false, 1, 0, true);

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
        let mut spatial_state = Array3::zeros((
            config.layers_per_timestep * config.t + 1,
            config.width,
            config.width,
        ));

        // Place a ring at the edge
        spatial_state[[config.ring_layer, 0, 0]] = 1.0;

        // Try to translate off the board
        let result = transform_state(&config, &spatial_state.view(), 0, false, false, -1, 0, true);

        assert!(result.is_none(), "Translation off board should return None");
    }
}
