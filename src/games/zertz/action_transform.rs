//! # Action Transformation for Zertz
//!
//! This module provides action transformation functions for testing and verification.
//! While not currently used in production (actions are applied directly to canonical states),
//! these functions ensure transformation logic is correct by verifying that:
//! - Actions transform consistently with states
//! - Inverse transformations round-trip correctly
//! - Symmetry operations preserve game semantics

use super::action::ZertzAction;
use super::board::BoardConfig;
use super::canonicalization::{
    build_axial_maps, build_layout_mask, parse_rot_component, parse_transform, transform_coordinate,
};

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

#[allow(dead_code)]
fn translate_action(
    action: &ZertzAction,
    dy: i32,
    dx: i32,
    config: &BoardConfig,
    layout: &[Vec<bool>],
) -> ZertzAction {
    match action {
        ZertzAction::Placement {
            marble_type,
            dst_flat,
            remove_flat,
        } => {
            // Convert flat to (y, x)
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            let new_dst_y = (dst_y as i32 + dy) as usize;
            let new_dst_x = (dst_x as i32 + dx) as usize;

            assert!(
                new_dst_y < config.width && new_dst_x < config.width,
                "translated position out of bounds"
            );
            assert!(
                layout[new_dst_y][new_dst_x],
                "translated position not on layout"
            );

            let new_remove_flat = remove_flat.map(|rf| {
                let (ry, rx) = config.flat_to_yx(rf);
                let ny = (ry as i32 + dy) as usize;
                let nx = (rx as i32 + dx) as usize;
                assert!(ny < config.width && nx < config.width);
                assert!(layout[ny][nx]);
                config.yx_to_flat(ny, nx)
            });

            ZertzAction::Placement {
                marble_type: *marble_type,
                dst_flat: config.yx_to_flat(new_dst_y, new_dst_x),
                remove_flat: new_remove_flat,
            }
        }
        ZertzAction::Capture { src_flat, dst_flat } => {
            // Convert flat to (y, x)
            let (src_y, src_x) = config.flat_to_yx(*src_flat);
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            let new_src_y = (src_y as i32 + dy) as usize;
            let new_src_x = (src_x as i32 + dx) as usize;
            let new_dst_y = (dst_y as i32 + dy) as usize;
            let new_dst_x = (dst_x as i32 + dx) as usize;
            assert!(new_src_y < config.width && new_src_x < config.width);
            assert!(layout[new_src_y][new_src_x]);

            ZertzAction::Capture {
                src_flat: config.yx_to_flat(new_src_y, new_src_x),
                dst_flat: config.yx_to_flat(new_dst_y, new_dst_x),
            }
        }
        ZertzAction::Pass => ZertzAction::Pass,
    }
}

#[allow(dead_code)]
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
            dst_flat,
            remove_flat,
        } => {
            // Convert flat to (y, x)
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            // Placements respect mirror_first like all other transformations
            let (new_dst_y, new_dst_x) = transform_coordinate(
                dst_y as i32,
                dst_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("destination outside board under transform");

            let new_remove_flat = remove_flat.map(|rf| {
                let (ry, rx) = config.flat_to_yx(rf);
                let (ny, nx) = transform_coordinate(
                    ry as i32,
                    rx as i32,
                    rot60_k,
                    mirror,
                    mirror_first,
                    yx_to_ax,
                    ax_to_yx,
                )
                .expect("removal outside board under transform");
                config.yx_to_flat(ny as usize, nx as usize)
            });

            ZertzAction::Placement {
                marble_type: *marble_type,
                dst_flat: config.yx_to_flat(new_dst_y as usize, new_dst_x as usize),
                remove_flat: new_remove_flat,
            }
        }
        ZertzAction::Capture { src_flat, dst_flat } => {
            // Convert flat to (y, x)
            let (src_y, src_x) = config.flat_to_yx(*src_flat);
            let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);

            // Captures use transform_coordinate (respects mirror_first)
            let (new_src_y, new_src_x) = transform_coordinate(
                src_y as i32,
                src_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("capture start outside board under transform");

            let (new_dst_y, new_dst_x) = transform_coordinate(
                dst_y as i32,
                dst_x as i32,
                rot60_k,
                mirror,
                mirror_first,
                yx_to_ax,
                ax_to_yx,
            )
            .expect("capture dest outside board under transform");

            ZertzAction::Capture {
                src_flat: config.yx_to_flat(new_src_y as usize, new_src_x as usize),
                dst_flat: config.yx_to_flat(new_dst_y as usize, new_dst_x as usize),
            }
        }
        ZertzAction::Pass => ZertzAction::Pass,
    }
}
