#[cfg(test)]
mod tests {
    use super::super::action_transform::*;
    use super::super::board::BoardConfig;
    use super::super::ZertzAction;

    fn placement_action() -> ZertzAction {
        ZertzAction::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 2,
            remove_y: Some(2),
            remove_x: Some(4),
        }
    }

    fn capture_action() -> ZertzAction {
        ZertzAction::Capture {
            start_y: 3,
            start_x: 3,
            direction: 0,
        }
    }

    #[test]
    fn test_transform_placement_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action();

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
                    dst_y,
                    dst_x,
                    remove_y,
                    remove_x,
                    ..
                } => {
                    assert_eq!(
                        (dst_y, dst_x),
                        (exp_dst_y, exp_dst_x),
                        "transform {}",
                        transform
                    );
                    assert_eq!(
                        (remove_y, remove_x),
                        (Some(exp_rem_y), Some(exp_rem_x)),
                        "transform {}",
                        transform
                    );
                }
                _ => panic!("Expected placement action for {}", transform),
            }
        }
    }

    #[test]
    fn test_transform_placement_translation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action();

        let transformed = transform_action(&action, "T1,0", &config);
        match transformed {
            ZertzAction::Placement {
                dst_y,
                dst_x,
                remove_y,
                remove_x,
                ..
            } => {
                assert_eq!((dst_y, dst_x), (4, 2));
                assert_eq!((remove_y, remove_x), (Some(3), Some(4)));
            }
            _ => panic!("Expected placement action"),
        }
    }

    #[test]
    fn test_transform_capture_rotations() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = capture_action();

        let test_cases = vec![
            ("MR60", (2, 3, 3)),
            ("R60M", (4, 3, 3)),
            ("R60", (1, 3, 3)),
            ("R120", (2, 3, 3)),
            ("R180", (3, 3, 3)),
        ];

        for (transform, (exp_dir, exp_y, exp_x)) in test_cases {
            let transformed = transform_action(&action, transform, &config);
            match transformed {
                ZertzAction::Capture {
                    direction,
                    start_y,
                    start_x,
                } => {
                    assert_eq!(
                        (direction, start_y, start_x),
                        (exp_dir, exp_y, exp_x),
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
        let action = capture_action();

        let transformed = transform_action(&action, "T2,-1", &config);
        match transformed {
            ZertzAction::Capture {
                direction,
                start_y,
                start_x,
            } => {
                assert_eq!(direction, 0);
                assert_eq!((start_y, start_x), (5, 2));
            }
            _ => panic!("Expected capture action"),
        }
    }

    #[test]
    fn test_transform_inverse_roundtrip() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = placement_action();
        let forward = transform_action(&action, "MR60", &config);
        let inverse = transform_action(&forward, "MR60", &config); // MR60 is its own inverse
        match inverse {
            ZertzAction::Placement {
                dst_y,
                dst_x,
                remove_y,
                remove_x,
                ..
            } => {
                assert_eq!((dst_y, dst_x), (3, 2));
                assert_eq!((remove_y, remove_x), (Some(2), Some(4)));
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
