#[cfg(test)]
mod tests {
    use super::super::action_transform::*;
    use super::super::board::BoardConfig;
    use super::super::ZertzAction;

    fn placement_action(config: &BoardConfig) -> ZertzAction {
        let dst_flat = 3 * config.width + 2;  // (3, 2) -> flat
        let remove_flat = Some(2 * config.width + 4);  // (2, 4) -> flat
        ZertzAction::Placement {
            marble_type: 0,
            dst_flat,
            remove_flat,
        }
    }

    fn capture_action(config: &BoardConfig) -> ZertzAction {
        // Start at (3,3), direction 0 is typically (-1, 0) in hexagonal, so dest is (1,3)
        let start_flat = 3 * config.width + 3;  // (3, 3) -> flat
        let dst_flat = 1 * config.width + 3;  // (1, 3) -> flat
        ZertzAction::Capture {
            start_flat,
            dst_flat,
        }
    }

    #[test]
    fn test_transform_placement_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action(&config);
        let width = config.width;

        let test_cases = vec![
            ("MR60", (3, 2), (4, 5)),
            ("R60M", (2, 3), (4, 2)),   // Different from MR60 - respects mirror_first
            ("MR120", (4, 3), (2, 4)),
            ("R120M", (3, 4), (2, 1)),  // Different from MR120 - respects mirror_first
            ("R180", (3, 4), (4, 2)),
        ];

        for (transform, (exp_dst_y, exp_dst_x), (exp_rem_y, exp_rem_x)) in test_cases {
            let transformed = transform_action(&action, transform, &config);
            match transformed {
                ZertzAction::Placement {
                    dst_flat,
                    remove_flat,
                    ..
                } => {
                    let dst_y = dst_flat / width;
                    let dst_x = dst_flat % width;
                    assert_eq!(
                        (dst_y, dst_x),
                        (exp_dst_y, exp_dst_x),
                        "transform {}",
                        transform
                    );
                    if let Some(rf) = remove_flat {
                        let rem_y = rf / width;
                        let rem_x = rf % width;
                        assert_eq!(
                            (rem_y, rem_x),
                            (exp_rem_y, exp_rem_x),
                            "transform {}",
                            transform
                        );
                    } else {
                        panic!("Expected remove_flat for {}", transform);
                    }
                }
                _ => panic!("Expected placement action for {}", transform),
            }
        }
    }

    #[test]
    fn test_transform_placement_translation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action(&config);
        let width = config.width;

        let transformed = transform_action(&action, "T1,0", &config);
        match transformed {
            ZertzAction::Placement {
                dst_flat,
                remove_flat,
                ..
            } => {
                let dst_y = dst_flat / width;
                let dst_x = dst_flat % width;
                assert_eq!((dst_y, dst_x), (4, 2));

                if let Some(rf) = remove_flat {
                    let rem_y = rf / width;
                    let rem_x = rf % width;
                    assert_eq!((rem_y, rem_x), (3, 4));
                } else {
                    panic!("Expected remove_flat");
                }
            }
            _ => panic!("Expected placement action"),
        }
    }

    #[test]
    fn test_transform_capture_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = capture_action(&config);
        let width = config.width;

        // Test cases: (transform_name, (expected_start_y, expected_start_x, expected_dest_y, expected_dest_x))
        // Note: Expected values verified by running actual transformations
        let test_cases = vec![
            ("R60", (3, 3, 3, 5)),   // Test basic rotation
            ("R180", (3, 3, 5, 3)),  // Test 180 rotation
        ];

        for (transform, (exp_start_y, exp_start_x, exp_dest_y, exp_dest_x)) in test_cases {
            let transformed = transform_action(&action, transform, &config);
            match transformed {
                ZertzAction::Capture {
                    start_flat,
                    dst_flat,
                } => {
                    let start_y = start_flat / width;
                    let start_x = start_flat % width;
                    let dest_y = dst_flat / width;
                    let dest_x = dst_flat % width;
                    assert_eq!(
                        (start_y, start_x, dest_y, dest_x),
                        (exp_start_y, exp_start_x, exp_dest_y, exp_dest_x),
                        "transform {}",
                        transform
                    );
                }
                _ => panic!("Expected capture action for {}", transform),
            }
        }
    }

    #[test]
    fn test_transform_capture_translation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = capture_action(&config);
        let width = config.width;

        let transformed = transform_action(&action, "T2,-1", &config);
        match transformed {
            ZertzAction::Capture {
                start_flat,
                dst_flat,
            } => {
                let start_y = start_flat / width;
                let start_x = start_flat % width;
                let dest_y = dst_flat / width;
                let dest_x = dst_flat % width;
                // Original: start (3,3), dest (1,3)
                // After T2,-1: start (5,2), dest (3,2)
                assert_eq!((start_y, start_x), (5, 2));
                assert_eq!((dest_y, dest_x), (3, 2));
            }
            _ => panic!("Expected capture action"),
        }
    }

    #[test]
    fn test_transform_inverse_roundtrip() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action(&config);
        let width = config.width;

        let forward = transform_action(&action, "MR60", &config);
        let inverse = transform_action(&forward, "MR60", &config); // MR60 is its own inverse
        match inverse {
            ZertzAction::Placement {
                dst_flat,
                remove_flat,
                ..
            } => {
                let dst_y = dst_flat / width;
                let dst_x = dst_flat % width;
                assert_eq!((dst_y, dst_x), (3, 2));

                if let Some(rf) = remove_flat {
                    let rem_y = rf / width;
                    let rem_x = rf % width;
                    assert_eq!((rem_y, rem_x), (2, 4));
                } else {
                    panic!("Expected remove_flat");
                }
            }
            _ => panic!("Expected placement action"),
        }
    }

    #[test]
    fn test_pass_action_unchanged() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = ZertzAction::Pass;

        let transforms = vec!["R60", "MR120", "T1,0"];
        for transform in transforms {
            let transformed = transform_action(&action, transform, &config);
            assert!(matches!(transformed, ZertzAction::Pass), "Pass should remain Pass for {}", transform);
        }
    }
}
