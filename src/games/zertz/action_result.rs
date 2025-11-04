//! # Zertz Action Result Types
//!
//! Defines result types returned when applying actions to the game state.

use pyo3::{pyclass, pymethods};

/// Result of applying a ZertzAction
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZertzActionResult {
    /// Result of a placement action
    Placement {
        /// List of marbles captured due to isolation: (marble_layer, y, x)
        isolation_captures: Vec<(usize, usize, usize)>,
    },
    /// Result of a capture action
    Capture {
        /// The captured marble: (marble_type, y, x)
        captured_marble: (usize, usize, usize),
    },
    /// Result of a pass action (no data)
    Pass,
}

impl ZertzActionResult {
    /// Create a placement result
    pub fn placement(isolation_captures: Vec<(usize, usize, usize)>) -> Self {
        ZertzActionResult::Placement { isolation_captures }
    }

    /// Create a capture result
    pub fn capture(marble_type: usize, y: usize, x: usize) -> Self {
        ZertzActionResult::Capture {
            captured_marble: (marble_type, y, x),
        }
    }

    /// Create a pass result
    pub fn pass() -> Self {
        ZertzActionResult::Pass
    }

    /// Get isolation captures if this is a placement result
    pub fn isolation_captures(&self) -> Option<&Vec<(usize, usize, usize)>> {
        match self {
            ZertzActionResult::Placement { isolation_captures } => Some(isolation_captures),
            _ => None,
        }
    }

    /// Get captured marble if this is a capture result
    pub fn captured_marble(&self) -> Option<(usize, usize, usize)> {
        match self {
            ZertzActionResult::Capture { captured_marble } => Some(*captured_marble),
            _ => None,
        }
    }
}

/// Python wrapper for ZertzActionResult
#[pyclass(name = "ZertzActionResult")]
#[derive(Clone)]
pub struct PyZertzActionResult {
    pub(crate) inner: ZertzActionResult,
}

#[pymethods]
impl PyZertzActionResult {
    /// Get the result type as a string
    fn result_type(&self) -> String {
        match &self.inner {
            ZertzActionResult::Placement { .. } => "PUT".to_string(),
            ZertzActionResult::Capture { .. } => "CAP".to_string(),
            ZertzActionResult::Pass => "PASS".to_string(),
        }
    }

    /// Get isolation captures (only for Placement results)
    ///
    /// Returns:
    ///     List of (marble_layer, y, x) tuples, or None if not a Placement result
    fn isolation_captures(&self) -> Option<Vec<(usize, usize, usize)>> {
        self.inner.isolation_captures().cloned()
    }

    /// Get captured marble (only for Capture results)
    ///
    /// Returns:
    ///     Tuple of (marble_type, y, x), or None if not a Capture result
    fn captured_marble(&self) -> Option<(usize, usize, usize)> {
        self.inner.captured_marble()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ZertzActionResult::Placement { isolation_captures } => {
                if isolation_captures.is_empty() {
                    "ZertzActionResult.Placement(no isolation captures)".to_string()
                } else {
                    format!(
                        "ZertzActionResult.Placement({} isolation captures)",
                        isolation_captures.len()
                    )
                }
            }
            ZertzActionResult::Capture { captured_marble } => {
                let marble_name = match captured_marble.0 {
                    0 => "white",
                    1 => "gray",
                    2 => "black",
                    _ => "unknown",
                };
                format!(
                    "ZertzActionResult.Capture({} at ({}, {}))",
                    marble_name, captured_marble.1, captured_marble.2
                )
            }
            ZertzActionResult::Pass => "ZertzActionResult.Pass".to_string(),
        }
    }
}
