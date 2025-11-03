#[cfg(test)]
mod tests {
    use super::super::canonicalization::*;
    use ndarray::Array3;

    /// Helper to create a state with X marks at given positions
    fn create_state_x(positions: &[(usize, usize)]) -> Array3<f32> {
        let mut state = Array3::<f32>::zeros((2, 3, 3));
        for &(r, c) in positions {
            state[[0, r, c]] = 1.0; // Layer 0 = X
        }
        state
    }

    /// Helper to create a state with X and O marks
    fn create_state(x_positions: &[(usize, usize)], o_positions: &[(usize, usize)]) -> Array3<f32> {
        let mut state = Array3::<f32>::zeros((2, 3, 3));
        for &(r, c) in x_positions {
            state[[0, r, c]] = 1.0; // Layer 0 = X
        }
        for &(r, c) in o_positions {
            state[[1, r, c]] = 1.0; // Layer 1 = O
        }
        state
    }

    /// Helper to check if two states are equal
    fn states_equal(a: &Array3<f32>, b: &Array3<f32>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    #[test]
    fn test_empty_board_is_canonical() {
        let state = Array3::<f32>::zeros((2, 3, 3));
        let (canonical, idx) = canonicalize_state(&state.view());

        assert_eq!(idx, 0, "Empty board should be canonical (identity)");
        assert!(states_equal(&canonical, &state), "Canonical form should equal original");
    }

    #[test]
    fn test_center_position_is_symmetric() {
        // X in center - should be symmetric under all transformations
        let state = create_state_x(&[(1, 1)]);
        let (canonical, _) = canonicalize_state(&state.view());

        // Any transform of center should give the same canonical form
        assert!(states_equal(&canonical, &state), "Center position should be its own canonical form");
    }

    #[test]
    fn test_corner_positions_equivalent() {
        // All four corners should canonicalize to the same form
        let corners = [
            (0, 0), // Top-left
            (0, 2), // Top-right
            (2, 0), // Bottom-left
            (2, 2), // Bottom-right
        ];

        let mut canonical_forms = Vec::new();

        for &corner in &corners {
            let state = create_state_x(&[corner]);
            let (canonical, _) = canonicalize_state(&state.view());
            canonical_forms.push(canonical);
        }

        // All corners should produce the same canonical form
        for i in 1..canonical_forms.len() {
            assert!(
                states_equal(&canonical_forms[0], &canonical_forms[i]),
                "Corner {:?} should have same canonical form as corner {:?}",
                corners[i],
                corners[0]
            );
        }
    }

    #[test]
    fn test_edge_positions_equivalent() {
        // All four edge midpoints should canonicalize to the same form
        let edges = [
            (0, 1), // Top
            (1, 0), // Left
            (1, 2), // Right
            (2, 1), // Bottom
        ];

        let mut canonical_forms = Vec::new();

        for &edge in &edges {
            let state = create_state_x(&[edge]);
            let (canonical, _) = canonicalize_state(&state.view());
            canonical_forms.push(canonical);
        }

        // All edges should produce the same canonical form
        for i in 1..canonical_forms.len() {
            assert!(
                states_equal(&canonical_forms[0], &canonical_forms[i]),
                "Edge {:?} should have same canonical form as edge {:?}",
                edges[i],
                edges[0]
            );
        }
    }

    #[test]
    fn test_diagonal_line_equivalence() {
        // Main diagonal and anti-diagonal should be equivalent
        let main_diag = create_state_x(&[(0, 0), (1, 1), (2, 2)]);
        let anti_diag = create_state_x(&[(0, 2), (1, 1), (2, 0)]);

        let (canonical1, _) = canonicalize_state(&main_diag.view());
        let (canonical2, _) = canonicalize_state(&anti_diag.view());

        assert!(
            states_equal(&canonical1, &canonical2),
            "Main diagonal and anti-diagonal should have same canonical form"
        );
    }

    #[test]
    fn test_row_and_column_equivalence() {
        // Top row and left column should be equivalent (via rotation)
        let top_row = create_state_x(&[(0, 0), (0, 1), (0, 2)]);
        let left_col = create_state_x(&[(0, 0), (1, 0), (2, 0)]);

        let (canonical1, _) = canonicalize_state(&top_row.view());
        let (canonical2, _) = canonicalize_state(&left_col.view());

        assert!(
            states_equal(&canonical1, &canonical2),
            "Top row and left column should have same canonical form"
        );
    }

    #[test]
    fn test_top_and_bottom_rows_equivalent() {
        // Top and bottom rows should be equivalent (via horizontal reflection)
        // But middle row is distinct because it's on a different symmetry axis
        let top_row = create_state_x(&[(0, 0), (0, 1), (0, 2)]);
        let bottom_row = create_state_x(&[(2, 0), (2, 1), (2, 2)]);

        let (canonical_top, _) = canonicalize_state(&top_row.view());
        let (canonical_bottom, _) = canonicalize_state(&bottom_row.view());

        assert!(
            states_equal(&canonical_top, &canonical_bottom),
            "Top and bottom rows should have same canonical form"
        );

        // Middle row should be different (it's the axis of horizontal reflection)
        let middle_row = create_state_x(&[(1, 0), (1, 1), (1, 2)]);
        let (canonical_middle, _) = canonicalize_state(&middle_row.view());

        assert!(
            !states_equal(&canonical_top, &canonical_middle),
            "Middle row should have different canonical form from top/bottom"
        );
    }

    #[test]
    fn test_complex_position_canonicalization() {
        // X at top-left, O at bottom-right
        let state1 = create_state(&[(0, 0)], &[(2, 2)]);

        // X at top-right, O at bottom-left (rotated 90°)
        let state2 = create_state(&[(0, 2)], &[(2, 0)]);

        // Both should be equivalent under symmetry
        let (canonical1, _) = canonicalize_state(&state1.view());
        let (canonical2, _) = canonicalize_state(&state2.view());

        assert!(
            states_equal(&canonical1, &canonical2),
            "Opposite corner pairs should have same canonical form"
        );
    }

    #[test]
    fn test_l_shape_patterns_equivalent() {
        // L-shape in different orientations
        let patterns = [
            vec![(0, 0), (1, 0), (1, 1)], // Top-left L
            vec![(0, 2), (1, 2), (1, 1)], // Top-right L (mirrored)
            vec![(2, 0), (1, 0), (1, 1)], // Bottom-left L (rotated)
            vec![(2, 2), (1, 2), (1, 1)], // Bottom-right L (rotated & mirrored)
        ];

        let mut canonical_forms = Vec::new();

        for pattern in &patterns {
            let state = create_state_x(pattern);
            let (canonical, _) = canonicalize_state(&state.view());
            canonical_forms.push(canonical);
        }

        // All L-shapes should produce the same canonical form
        for i in 1..canonical_forms.len() {
            assert!(
                states_equal(&canonical_forms[0], &canonical_forms[i]),
                "L-shape pattern {} should have same canonical form as pattern 0", i
            );
        }
    }

    #[test]
    fn test_asymmetric_positions_different() {
        // Two genuinely different positions that shouldn't be equivalent
        let state1 = create_state_x(&[(0, 0), (1, 1)]); // Top-left corner + center
        let state2 = create_state_x(&[(0, 0), (0, 1)]); // Top-left corner + top-middle

        let (canonical1, _) = canonicalize_state(&state1.view());
        let (canonical2, _) = canonicalize_state(&state2.view());

        assert!(
            !states_equal(&canonical1, &canonical2),
            "Genuinely different positions should have different canonical forms"
        );
    }

    #[test]
    fn test_full_board_symmetry() {
        // A board with marks that has specific symmetry
        let state = create_state(
            &[(0, 0), (0, 2), (2, 0), (2, 2)], // X in all corners
            &[(0, 1), (1, 0), (1, 2), (2, 1)], // O on all edges
        );

        let (canonical, _) = canonicalize_state(&state.view());

        // This symmetric pattern should be its own canonical form
        assert!(
            states_equal(&canonical, &state),
            "Fully symmetric pattern should be its own canonical form"
        );
    }

    #[test]
    fn test_corner_edge_pairs_equivalent() {
        // Pairs starting at a corner and going along an edge should be equivalent
        let patterns = [
            vec![(0, 0), (0, 1)], // Top-left corner, right
            vec![(0, 2), (0, 1)], // Top-right corner, left
            vec![(2, 0), (2, 1)], // Bottom-left corner, right
            vec![(2, 2), (2, 1)], // Bottom-right corner, left
            vec![(0, 0), (1, 0)], // Top-left corner, down
            vec![(0, 2), (1, 2)], // Top-right corner, down
            vec![(2, 0), (1, 0)], // Bottom-left corner, up
            vec![(2, 2), (1, 2)], // Bottom-right corner, up
        ];

        let mut canonical_forms = Vec::new();

        for pattern in &patterns {
            let state = create_state_x(pattern);
            let (canonical, _) = canonicalize_state(&state.view());
            canonical_forms.push(canonical);
        }

        // All corner+edge pairs should produce the same canonical form
        for i in 1..canonical_forms.len() {
            assert!(
                states_equal(&canonical_forms[0], &canonical_forms[i]),
                "Corner-edge pattern {} should have same canonical form as pattern 0", i
            );
        }
    }

    #[test]
    fn test_mixed_marks_asymmetry() {
        // Position with both X and O
        let state = create_state(&[(0, 0), (1, 1)], &[(0, 1)]);

        // Rotated version
        let state_rot = create_state(&[(0, 2), (1, 1)], &[(1, 2)]);

        let (canonical1, _) = canonicalize_state(&state.view());
        let (canonical2, _) = canonicalize_state(&state_rot.view());

        assert!(
            states_equal(&canonical1, &canonical2),
            "Rotated mixed positions should have same canonical form"
        );
    }

    #[test]
    fn test_win_condition_patterns() {
        // Test that winning patterns in different orientations are equivalent
        // Horizontal win
        let horizontal_win = create_state_x(&[(0, 0), (0, 1), (0, 2)]);

        // Vertical win (should be equivalent via rotation)
        let vertical_win = create_state_x(&[(0, 0), (1, 0), (2, 0)]);

        let (canonical_h, _) = canonicalize_state(&horizontal_win.view());
        let (canonical_v, _) = canonicalize_state(&vertical_win.view());

        assert!(
            states_equal(&canonical_h, &canonical_v),
            "Horizontal and vertical wins should have same canonical form"
        );
    }

    #[test]
    fn test_transform_name_coverage() {
        // Just verify all transform names are defined
        for i in 0..8 {
            let name = transform_name(i);
            assert!(!name.is_empty(), "Transform {} should have a name", i);
            assert_ne!(name, "unknown", "Transform {} should have a specific name", i);
        }

        // Unknown index should return "unknown"
        assert_eq!(transform_name(100), "unknown");
    }

    #[test]
    fn test_canonical_is_deterministic() {
        // Verify that canonicalization is deterministic and finds a specific form
        let corners = [(0, 0), (0, 2), (2, 0), (2, 2)];

        let mut canonical_forms = Vec::new();
        for &corner in &corners {
            let state = create_state_x(&[corner]);
            let (canonical, _) = canonicalize_state(&state.view());
            canonical_forms.push(canonical);
        }

        // All corners should produce the SAME canonical form
        for i in 1..canonical_forms.len() {
            assert!(
                states_equal(&canonical_forms[0], &canonical_forms[i]),
                "All corners should produce the same canonical form"
            );
        }

        // The canonical form should be deterministic (same as identity)
        let state = create_state_x(&[(2, 2)]);
        let (canonical1, _) = canonicalize_state(&state.view());
        let (canonical2, _) = canonicalize_state(&state.view());
        assert!(states_equal(&canonical1, &canonical2), "Canonicalization should be deterministic");
    }

    #[test]
    fn test_complex_game_state() {
        // A realistic game state partway through a game
        let state = create_state(
            &[(1, 1), (0, 0), (2, 1)], // X: center, top-left, bottom-middle
            &[(0, 1), (1, 0)],          // O: top-middle, middle-left
        );

        let (canonical, transform_idx) = canonicalize_state(&state.view());

        // Just verify it produces a valid result
        assert!(transform_idx < 8, "Transform index should be 0-7");
        assert_eq!(canonical.shape(), &[2, 3, 3], "Canonical form should have correct shape");

        // Verify the transformation is deterministic
        let (canonical2, transform_idx2) = canonicalize_state(&state.view());
        assert_eq!(transform_idx, transform_idx2, "Same state should give same transform");
        assert!(states_equal(&canonical, &canonical2), "Same state should give same canonical form");
    }

    #[test]
    fn test_idempotence() {
        // Canonicalizing a canonical form should give the same result
        let state = create_state_x(&[(0, 0), (1, 1), (2, 2)]);

        let (canonical1, _) = canonicalize_state(&state.view());
        let (canonical2, _) = canonicalize_state(&canonical1.view());

        assert!(
            states_equal(&canonical1, &canonical2),
            "Canonicalizing a canonical form should be idempotent"
        );
    }

    #[test]
    fn test_all_eight_symmetries_generate_same_canonical() {
        // Create a non-symmetric position
        let state = create_state_x(&[(0, 0)]);

        // Manually apply all 8 symmetries and verify they all canonicalize to the same form
        let (canonical_original, _) = canonicalize_state(&state.view());

        // This test implicitly verifies the symmetry generation is working correctly
        // since canonicalize_state uses all 8 symmetries internally

        // Create equivalent positions by hand
        let equivalent_positions = [
            create_state_x(&[(0, 0)]), // Original (identity)
            create_state_x(&[(0, 2)]), // Rotated 90° clockwise
            create_state_x(&[(2, 2)]), // Rotated 180°
            create_state_x(&[(2, 0)]), // Rotated 270° clockwise
        ];

        for (i, equiv_state) in equivalent_positions.iter().enumerate() {
            let (canonical, _) = canonicalize_state(&equiv_state.view());
            assert!(
                states_equal(&canonical_original, &canonical),
                "Equivalent position {} should have same canonical form", i
            );
        }
    }
}
