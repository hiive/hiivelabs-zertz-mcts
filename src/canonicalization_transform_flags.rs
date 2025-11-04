//! # Canonicalization Transform Flags
//!
//! Generic flags for controlling which transforms to use in state canonicalization.
//! This is game-agnostic and can be used by any game that supports translation and/or
//! symmetry-based canonicalization.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hiivelabs_mcts::canonicalization_transform_flags::TransformFlags;
//!
//! // Use all transforms
//! let flags = TransformFlags::ALL;
//!
//! // Use only rotation and mirror (no translation)
//! let flags = TransformFlags::ROTATION_MIRROR;
//!
//! // Check which transforms are enabled
//! if flags.has_rotation() {
//!     // Apply rotation transforms
//! }
//! ```

use pyo3::prelude::*;

/// Flags for controlling which transforms to use in canonicalization (Rust type).
///
/// TransformFlags uses bit flags to specify which types of symmetries to include:
/// - ROTATION: Include rotational symmetries (e.g., 60°, 90°, 120°, etc.)
/// - MIRROR: Include reflection symmetries
/// - TRANSLATION: Include translational symmetries
///
/// Common combinations:
/// - ALL: All transforms (rotation + mirror + translation)
/// - ROTATION_MIRROR: Only rotation and mirror (no translation)
/// - NONE: Identity only (no transforms)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformFlags {
    bits: u8,
}

impl TransformFlags {
    /// Include rotational symmetries
    pub const ROTATION: Self = Self { bits: 0b001 };
    /// Include mirror symmetries
    pub const MIRROR: Self = Self { bits: 0b010 };
    /// Include translation symmetries
    pub const TRANSLATION: Self = Self { bits: 0b100 };
    /// All transforms (rotation + mirror + translation)
    pub const ALL: Self = Self { bits: 0b111 };
    /// Rotation and mirror only (no translation)
    pub const ROTATION_MIRROR: Self = Self { bits: 0b011 };
    /// No transforms (identity only)
    pub const NONE: Self = Self { bits: 0b000 };

    /// Check if rotation flag is set
    pub fn has_rotation(self) -> bool {
        (self.bits & Self::ROTATION.bits) != 0
    }

    /// Check if mirror flag is set
    pub fn has_mirror(self) -> bool {
        (self.bits & Self::MIRROR.bits) != 0
    }

    /// Check if translation flag is set
    pub fn has_translation(self) -> bool {
        (self.bits & Self::TRANSLATION.bits) != 0
    }

    /// Create from bits
    ///
    /// Returns None if bits > 0b111 (invalid combination)
    pub fn from_bits(bits: u8) -> Option<Self> {
        if bits <= 0b111 {
            Some(Self { bits })
        } else {
            None
        }
    }

    /// Get the raw bits
    pub fn bits(self) -> u8 {
        self.bits
    }
}

impl Default for TransformFlags {
    fn default() -> Self {
        TransformFlags::ALL
    }
}

// ============================================================================
// PYTHON BINDINGS
// ============================================================================

/// Python wrapper for TransformFlags
///
/// Flags for controlling which transforms to use in canonicalization.
///
/// TransformFlags uses bit flags to specify which types of symmetries to include:
/// - ROTATION: Include rotational symmetries (0°, 60°, 90°, 120°, etc.)
/// - MIRROR: Include reflection symmetries
/// - TRANSLATION: Include translational symmetries
///
/// Common combinations:
/// - ALL: rotation + mirror + translation (full canonicalization)
/// - ROTATION_MIRROR: rotation + mirror only (canonical orientation, no translation)
/// - NONE: identity only (no transforms)
#[pyclass(name = "TransformFlags")]
#[derive(Clone)]
pub struct PyTransformFlags {
    pub(crate) inner: TransformFlags,
}

#[pymethods]
impl PyTransformFlags {
    /// All transforms enabled (rotation + mirror + translation)
    #[classattr]
    #[allow(non_snake_case)]
    fn ALL() -> Self {
        PyTransformFlags {
            inner: TransformFlags::ALL,
        }
    }

    /// Only rotational symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn ROTATION() -> Self {
        PyTransformFlags {
            inner: TransformFlags::ROTATION,
        }
    }

    /// Only mirror symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn MIRROR() -> Self {
        PyTransformFlags {
            inner: TransformFlags::MIRROR,
        }
    }

    /// Only translation symmetries
    #[classattr]
    #[allow(non_snake_case)]
    fn TRANSLATION() -> Self {
        PyTransformFlags {
            inner: TransformFlags::TRANSLATION,
        }
    }

    /// Rotation and mirror only (no translation)
    #[classattr]
    #[allow(non_snake_case)]
    fn ROTATION_MIRROR() -> Self {
        PyTransformFlags {
            inner: TransformFlags::ROTATION_MIRROR,
        }
    }

    /// No transforms (identity only)
    #[classattr]
    #[allow(non_snake_case)]
    fn NONE() -> Self {
        PyTransformFlags {
            inner: TransformFlags::NONE,
        }
    }

    /// Create TransformFlags from bit flags
    ///
    /// Args:
    ///     bits: Bit flags (0-7). Use constants like TransformFlags.ALL,
    ///           TransformFlags.ROTATION_MIRROR, etc.
    ///
    /// Returns:
    ///     TransformFlags instance
    ///
    /// Raises:
    ///     ValueError: If bits > 7
    #[new]
    pub fn new(bits: u8) -> PyResult<Self> {
        TransformFlags::from_bits(bits)
            .map(|inner| PyTransformFlags { inner })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid transform flags bits: {} (must be 0-7)",
                    bits
                ))
            })
    }

    /// Get the raw bit flags
    pub fn bits(&self) -> u8 {
        self.inner.bits()
    }

    /// Check if rotation flag is set
    pub fn has_rotation(&self) -> bool {
        self.inner.has_rotation()
    }

    /// Check if mirror flag is set
    pub fn has_mirror(&self) -> bool {
        self.inner.has_mirror()
    }

    /// Check if translation flag is set
    pub fn has_translation(&self) -> bool {
        self.inner.has_translation()
    }

    fn __repr__(&self) -> String {
        let bits = self.inner.bits();
        let names = match bits {
            0b111 => "ALL",
            0b011 => "ROTATION_MIRROR",
            0b001 => "ROTATION",
            0b010 => "MIRROR",
            0b100 => "TRANSLATION",
            0b000 => "NONE",
            _ => return format!("TransformFlags({})", bits),
        };
        format!("TransformFlags.{}", names)
    }

    fn __eq__(&self, other: &PyTransformFlags) -> bool {
        self.inner == other.inner
    }
}
