//! # Zertz Action Types
//!
//! Defines the action representation for Zertz game moves and their Python bindings.

use super::board::BoardConfig;
use pyo3::{pyclass, pymethods};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Zertz-specific action representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZertzAction {
    /// Place a marble and optionally remove a ring
    Placement {
        marble_type: usize, // 0=white, 1=gray, 2=black
        dst_flat: usize,
        remove_flat: Option<usize>,
    },
    /// Capture by jumping from start to dest
    Capture { src_flat: usize, dst_flat: usize },
    /// Pass (no legal moves)
    Pass,
}

impl ZertzAction {
    pub fn action_type(&self) -> String {
        match &self {
            ZertzAction::Placement { .. } => "PUT".to_string(),
            ZertzAction::Capture { .. } => "CAP".to_string(),
            ZertzAction::Pass => "PASS".to_string(),
        }
    }

    /// Convert action to tuple format for serialization
    ///
    /// # Arguments
    /// * `config` - Board config for coordinate flattening
    ///
    /// # Returns
    /// Result containing tuple of (action_type, optional (param1, param2, param3))
    /// - For Placement: ("PUT", Some((marble_type, dst_flat, remove_flat)))
    /// - For Capture: ("CAP", Some((None, src_flat, dst_flat)))
    /// - For Pass: ("PASS", None)
    pub fn to_tuple(&self, config: &BoardConfig) -> (String, Option<Vec<Option<usize>>>) {
        match &self {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                let (rem_y, rem_x) = config.flat_to_optional_yx(*remove_flat);
                (
                    self.action_type(),
                    Option::from(vec![
                        Some(*marble_type),
                        Some(dst_y),
                        Some(dst_x),
                        rem_y,
                        rem_x,
                    ]),
                )
            }
            ZertzAction::Capture { src_flat, dst_flat } => {
                let (src_y, src_x) = config.flat_to_yx(*src_flat);
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                (
                    self.action_type(),
                    Option::from(vec![Some(src_y), Some(src_x), Some(dst_y), Some(dst_x)]),
                )
            }
            ZertzAction::Pass => (self.action_type(), None),
        }
    }
}

/// Python wrapper for ZertzAction
///
/// Provides Python-friendly interface for creating and manipulating Zertz actions.
#[pyclass(name = "ZertzAction")]
#[derive(Clone)]
pub struct PyZertzAction {
    pub(crate) inner: ZertzAction,
}

#[pymethods]
impl PyZertzAction {
    /// Create a Placement action
    ///
    /// Args:
    ///     marble_type: Marble type (0=white, 1=gray, 2=black)
    ///     dst_y: Destination row
    ///     dst_x: Destination column
    ///     remove_y: Optional row of ring to remove
    ///     remove_x: Optional column of ring to remove
    #[staticmethod]
    #[pyo3(signature = (config, marble_type, dst_y, dst_x, remove_y=None, remove_x=None))]
    pub fn placement(
        config: &BoardConfig,
        marble_type: usize,
        dst_y: usize,
        dst_x: usize,
        remove_y: Option<usize>,
        remove_x: Option<usize>,
    ) -> Self {
        let dst_flat = config.yx_to_flat(dst_y, dst_x);
        let remove_flat = config.yx_to_optional_flat(remove_y, remove_x);

        PyZertzAction {
            inner: ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            },
        }
    }

    /// Create a Capture action
    ///
    /// Args:
    ///     src_y: Starting row
    ///     src_x: Starting column
    ///     dst_y: Destination row
    ///     dst_x: Destination column
    #[staticmethod]
    pub fn capture(
        config: &BoardConfig,
        src_y: usize,
        src_x: usize,
        dst_y: usize,
        dst_x: usize,
    ) -> Self {
        PyZertzAction {
            inner: ZertzAction::Capture {
                src_flat: config.yx_to_flat(src_y, src_x),
                dst_flat: config.yx_to_flat(dst_y, dst_x),
            },
        }
    }

    /// Create a Pass action
    #[staticmethod]
    pub fn pass() -> Self {
        PyZertzAction {
            inner: ZertzAction::Pass,
        }
    }

    /// Convert action to tuple format
    ///
    /// Args:
    ///     width: Board config for coordinate handling
    ///
    /// Returns:
    ///     Tuple of (action_type, optional action_data)
    pub fn to_tuple(&self, config: &BoardConfig) -> (String, Option<Vec<Option<usize>>>) {
        self.inner.to_tuple(config)
    }

    /// Get action type as string
    pub fn action_type(&self) -> String {
        self.inner.action_type()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let marble_name = match marble_type {
                    0 => "white",
                    1 => "gray",
                    2 => "black",
                    _ => "unknown",
                };
                if let Some(r_flat) = remove_flat {
                    format!(
                        "ZertzAction.Placement({}, {}, remove={})",
                        marble_name, dst_flat, r_flat
                    )
                } else {
                    format!("ZertzAction.Placement({}, {})", marble_name, dst_flat)
                }
            }
            ZertzAction::Capture { src_flat, dst_flat } => {
                format!("ZertzAction.Capture({} -> {})", src_flat, dst_flat)
            }
            ZertzAction::Pass => "ZertzAction.Pass".to_string(),
        }
    }

    fn __eq__(&self, other: &PyZertzAction) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::zertz::BoardConfig;

    #[test]
    fn test_zertz_action_placement_creation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = ZertzAction::Placement {
            marble_type: 0, // white
            dst_flat: 3 * config.width + 2,
            remove_flat: Some(2 * config.width + 4),
        };

        match action {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                assert_eq!(marble_type, 0);
                assert_eq!(dst_flat, 3 * config.width + 2);
                assert_eq!(remove_flat, Some(2 * config.width + 4));
            }
            _ => panic!("Expected Placement variant"),
        }
    }

    #[test]
    fn test_zertz_action_placement_no_removal() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = ZertzAction::Placement {
            marble_type: 1, // gray
            dst_flat: 4 * config.width + 3,
            remove_flat: None,
        };

        match action {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                assert_eq!(marble_type, 1);
                assert_eq!(dst_flat, 4 * config.width + 3);
                assert_eq!(remove_flat, None);
            }
            _ => panic!("Expected Placement variant"),
        }
    }

    #[test]
    fn test_zertz_action_capture_creation() {
        let config = BoardConfig::standard(37, 1).unwrap();
        let action = ZertzAction::Capture {
            src_flat: 3 * config.width + 3,
            dst_flat: 1 * config.width + 3,
        };

        match action {
            ZertzAction::Capture { src_flat, dst_flat } => {
                assert_eq!(src_flat, 3 * config.width + 3);
                assert_eq!(dst_flat, 1 * config.width + 3);
            }
            _ => panic!("Expected Capture variant"),
        }
    }

    #[test]
    fn test_zertz_action_pass_creation() {
        let action = ZertzAction::Pass;
        assert!(matches!(action, ZertzAction::Pass));
    }

    #[test]
    fn test_zertz_action_clone() {
        let action = ZertzAction::Placement {
            marble_type: 2,
            dst_flat: 15,
            remove_flat: Some(20),
        };
        let cloned = action.clone();

        match (&action, &cloned) {
            (
                ZertzAction::Placement {
                    marble_type: mt1,
                    dst_flat: df1,
                    remove_flat: rf1,
                },
                ZertzAction::Placement {
                    marble_type: mt2,
                    dst_flat: df2,
                    remove_flat: rf2,
                },
            ) => {
                assert_eq!(mt1, mt2);
                assert_eq!(df1, df2);
                assert_eq!(rf1, rf2);
            }
            _ => panic!("Clone did not produce same variant"),
        }
    }

    #[test]
    fn test_zertz_action_equality() {
        let action1 = ZertzAction::Capture {
            src_flat: 10,
            dst_flat: 20,
        };
        let action2 = ZertzAction::Capture {
            src_flat: 10,
            dst_flat: 20,
        };
        let action3 = ZertzAction::Capture {
            src_flat: 10,
            dst_flat: 21,
        };

        assert_eq!(action1, action2);
        assert_ne!(action1, action3);
    }

    #[test]
    fn test_zertz_action_hash() {
        use std::collections::HashSet;

        let action1 = ZertzAction::Placement {
            marble_type: 0,
            dst_flat: 10,
            remove_flat: None,
        };
        let action2 = ZertzAction::Placement {
            marble_type: 0,
            dst_flat: 10,
            remove_flat: None,
        };
        let action3 = ZertzAction::Pass;

        let mut set = HashSet::new();
        set.insert(action1.clone());
        set.insert(action2);
        set.insert(action3);

        // action1 and action2 are equal, so set should have 2 elements
        assert_eq!(set.len(), 2);
    }
}
