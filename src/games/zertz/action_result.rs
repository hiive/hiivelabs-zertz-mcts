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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_result_placement_creation() {
        let result = ZertzActionResult::placement(vec![(1, 2, 3), (2, 4, 5)]);

        match result {
            ZertzActionResult::Placement { isolation_captures } => {
                assert_eq!(isolation_captures.len(), 2);
                assert_eq!(isolation_captures[0], (1, 2, 3));
                assert_eq!(isolation_captures[1], (2, 4, 5));
            }
            _ => panic!("Expected Placement result"),
        }
    }

    #[test]
    fn test_action_result_placement_no_captures() {
        let result = ZertzActionResult::placement(vec![]);

        match result {
            ZertzActionResult::Placement { isolation_captures } => {
                assert_eq!(isolation_captures.len(), 0);
            }
            _ => panic!("Expected Placement result"),
        }
    }

    #[test]
    fn test_action_result_capture_creation() {
        let result = ZertzActionResult::capture(1, 3, 4); // gray marble at (3, 4)

        match result {
            ZertzActionResult::Capture { captured_marble } => {
                assert_eq!(captured_marble, (1, 3, 4));
            }
            _ => panic!("Expected Capture result"),
        }
    }

    #[test]
    fn test_action_result_pass_creation() {
        let result = ZertzActionResult::pass();
        assert!(matches!(result, ZertzActionResult::Pass));
    }

    #[test]
    fn test_action_result_isolation_captures_accessor() {
        let result = ZertzActionResult::placement(vec![(0, 1, 2)]);

        let captures = result.isolation_captures();
        assert!(captures.is_some());
        assert_eq!(captures.unwrap().len(), 1);
        assert_eq!(captures.unwrap()[0], (0, 1, 2));
    }

    #[test]
    fn test_action_result_isolation_captures_none_for_capture() {
        let result = ZertzActionResult::capture(0, 2, 3);
        assert!(result.isolation_captures().is_none());
    }

    #[test]
    fn test_action_result_isolation_captures_none_for_pass() {
        let result = ZertzActionResult::pass();
        assert!(result.isolation_captures().is_none());
    }

    #[test]
    fn test_action_result_captured_marble_accessor() {
        let result = ZertzActionResult::capture(2, 5, 6);

        let marble = result.captured_marble();
        assert!(marble.is_some());
        assert_eq!(marble.unwrap(), (2, 5, 6));
    }

    #[test]
    fn test_action_result_captured_marble_none_for_placement() {
        let result = ZertzActionResult::placement(vec![]);
        assert!(result.captured_marble().is_none());
    }

    #[test]
    fn test_action_result_captured_marble_none_for_pass() {
        let result = ZertzActionResult::pass();
        assert!(result.captured_marble().is_none());
    }

    #[test]
    fn test_action_result_clone() {
        let result = ZertzActionResult::capture(1, 2, 3);
        let cloned = result.clone();

        match (&result, &cloned) {
            (
                ZertzActionResult::Capture {
                    captured_marble: cm1,
                },
                ZertzActionResult::Capture {
                    captured_marble: cm2,
                },
            ) => {
                assert_eq!(cm1, cm2);
            }
            _ => panic!("Clone did not produce same variant"),
        }
    }

    #[test]
    fn test_action_result_equality() {
        let result1 = ZertzActionResult::capture(0, 1, 2);
        let result2 = ZertzActionResult::capture(0, 1, 2);
        let result3 = ZertzActionResult::capture(0, 1, 3);

        assert_eq!(result1, result2);
        assert_ne!(result1, result3);
    }
}
