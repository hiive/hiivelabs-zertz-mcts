//! # Action Transformation for Zertz
//!
//! This module provides action transformation functions for testing and verification.
//! While not currently used in production (actions are applied directly to canonical states),
//! these functions ensure transformation logic is correct by verifying that:
//! - Actions transform consistently with states
//! - Inverse transformations round-trip correctly
//! - Symmetry operations preserve game semantics

use crate::board::BoardConfig;
use crate::canonicalization::{
    build_axial_maps, build_layout_mask, dir_index_map, parse_rot_component, parse_transform,
    transform_coordinate,
};
use super::ZertzAction;

/// Transform a Zertz action according to a symmetry operation
///
/// # Arguments
/// * `action` - The action to transform
/// * `transform` - Transform string (e.g., "R60", "MR120", "T1,0")
/// * `config` - Board configuration
///
/// # Returns
/// Transformed action
///
/// # Example
/// ```rust,ignore
/// let action = ZertzAction::Placement { marble_type: 0, dst_y: 3, dst_x: 2, ... };
/// let transformed = transform_action(&action, "R60", &config);
/// ```
pub fn transform_action(
    action: &ZertzAction,
    transform: &str,
    config: &BoardConfig,
) -> ZertzAction {
    if transform.is_empty() || transform == "R0" {
        return action.clone();
    }

    let layout = build_layout_mask(config);
    let (yx_to_ax, ax_to_yx) = build_axial_maps(config, &layout);

    let mut current_action = action.clone();

    for part in parse_transform(transform) {
        if let Some(coords) = part.strip_prefix('T') {
            let mut split = coords.split(',');
            let dy = split.next().unwrap().parse::<i32>().unwrap();
            let dx = split.next().unwrap().parse::<i32>().unwrap();
            current_action = translate_action(&current_action, dy, dx, config, &layout);
        } else {
            let (rot60_k, mirror, mirror_first) = parse_rot_component(&part);
            current_action = apply_orientation(
                &current_action,
                rot60_k,
                mirror,
                mirror_first,
                config,
                &yx_to_ax,
                &ax_to_yx,
            );
        }
    }

    current_action
}

fn translate_action(
    action: &ZertzAction,
    dy: i32,
    dx: i32,
    config: &BoardConfig,
    layout: &[Vec<bool>],
) -> ZertzAction {
    let width = config.width as i32;
    match action {
        ZertzAction::Placement {
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
        } => {
            let new_dst_y = (*dst_y as i32 + dy) as usize;
            let new_dst_x = (*dst_x as i32 + dx) as usize;

            assert!(
                new_dst_y < config.width && new_dst_x < config.width,
                "translated position out of bounds"
            );
            assert!(
                layout[new_dst_y][new_dst_x],
                "translated position not on layout"
            );

            let (new_remove_y, new_remove_x) = match (remove_y, remove_x) {
                (Some(ry), Some(rx)) => {
                    let ny = (*ry as i32 + dy) as usize;
                    let nx = (*rx as i32 + dx) as usize;
                    assert!(ny < config.width && nx < config.width);
                    assert!(layout[ny][nx]);
                    (Some(ny), Some(nx))
                }
                _ => (None, None),
            };

            ZertzAction::Placement {
                marble_type: *marble_type,
                dst_y: new_dst_y,
                dst_x: new_dst_x,
                remove_y: new_remove_y,
                remove_x: new_remove_x,
            }
        }
        ZertzAction::Capture {
            start_y,
            start_x,
            direction,
        } => {
            let new_y = (*start_y as i32 + dy) as usize;
            let new_x = (*start_x as i32 + dx) as usize;
            assert!(new_y < config.width && new_x < config.width);
            assert!(layout[new_y][new_x]);

            ZertzAction::Capture {
                start_y: new_y,
                start_x: new_x,
                direction: *direction,
            }
        }
        ZertzAction::Pass => ZertzAction::Pass,
    }
}

fn apply_orientation(
    action: &ZertzAction,
    rot60_k: i32,
    mirror: bool,
    mirror_first: bool,
    config: &BoardConfig,
    yx_to_ax: &std::collections::HashMap<(i32, i32), (i32, i32)>,
    ax_to_yx: &std::collections::HashMap<(i32, i32), (i32, i32)>,
) -> ZertzAction {
    match action {
        ZertzAction::Placement {
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
        } => {
            // Placements respect mirror_first like all other transformations
            let (new_dst_y, new_dst_x) = transform_coordinate(
                *dst_y as i32,
                *dst_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("destination outside board under transform");

            let (new_remove_y, new_remove_x) = match (remove_y, remove_x) {
                (Some(ry), Some(rx)) => {
                    let (ny, nx) = transform_coordinate(
                        *ry as i32,
                        *rx as i32,
                        rot60_k,
                        mirror,
                        mirror_first,
                        yx_to_ax,
                        ax_to_yx,
                    )
                    .expect("removal outside board under transform");
                    (Some(ny as usize), Some(nx as usize))
                }
                _ => (None, None),
            };

            ZertzAction::Placement {
                marble_type: *marble_type,
                dst_y: new_dst_y as usize,
                dst_x: new_dst_x as usize,
                remove_y: new_remove_y,
                remove_x: new_remove_x,
            }
        }
        ZertzAction::Capture {
            start_y,
            start_x,
            direction,
        } => {
            // Captures use transform_coordinate (respects mirror_first)
            let (new_y, new_x) = transform_coordinate(
                *start_y as i32,
                *start_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("capture start outside board under transform");

            let dir_map = dir_index_map(rot60_k, mirror, mirror_first, config);
            let new_direction = *dir_map.get(direction).expect("direction not in map");

            ZertzAction::Capture {
                start_y: new_y as usize,
                start_x: new_x as usize,
                direction: new_direction,
            }
        }
        ZertzAction::Pass => ZertzAction::Pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::ZertzAction;

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
