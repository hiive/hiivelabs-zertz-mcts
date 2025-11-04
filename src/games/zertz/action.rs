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
                    "PUT".to_string(),
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

    pub fn to_placement_action(
        &self,
        config: &BoardConfig,
    ) -> Option<(String, Vec<Option<usize>>)> {
        // Returns ((dst_y, dst_x), optional (rem_y, rem_x))
        match self {
            ZertzAction::Placement {
                marble_type,
                dst_flat,
                remove_flat,
            } => {
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                let (rem_y, rem_x) = config.flat_to_optional_yx(*remove_flat);

                Some((
                    self.action_type(),
                    vec![Some(*marble_type), Some(dst_y), Some(dst_x), rem_y, rem_x],
                ))
            }
            _ => None,
        }
    }

    pub fn to_capture_action(&self, config: &BoardConfig) -> Option<(String, Vec<usize>)> {
        // Returns ((start_y, start_x), (dst_y, dst_x))
        match self {
            ZertzAction::Capture { src_flat, dst_flat } => {
                // Return (None, src_flat, dst_flat)
                // None distinguishes captures from placements (which have Some(marble_type))
                // Python will unflatten both coordinates and calculate cap_index as midpoint
                let (src_y, src_x) = config.flat_to_yx(*src_flat);
                let (dst_y, dst_x) = config.flat_to_yx(*dst_flat);
                Some((self.action_type(), vec![src_y, src_x, dst_y, dst_x]))
            }
            _ => None,
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
