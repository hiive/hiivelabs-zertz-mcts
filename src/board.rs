use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Game mode (Standard or Blitz)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameMode {
    Standard,
    Blitz,
}

/// Win condition thresholds for a game mode
#[derive(Clone, Debug)]
pub struct WinConditions {
    pub each_color: f32, // Threshold for having all three colors
    pub white_only: f32,
    pub gray_only: f32,
    pub black_only: f32,
}

impl WinConditions {
    pub fn standard() -> Self {
        WinConditions {
            each_color: 3.0,
            white_only: 4.0,
            gray_only: 5.0,
            black_only: 6.0,
        }
    }

    pub fn blitz() -> Self {
        WinConditions {
            each_color: 2.0,
            white_only: 3.0,
            gray_only: 4.0,
            black_only: 5.0,
        }
    }
}

/// Immutable board configuration (mirrors Python BoardConfig)
#[derive(Clone, Debug)]
pub struct BoardConfig {
    pub mode: GameMode,
    pub win_conditions: WinConditions,
    pub width: usize,
    pub rings: usize,
    pub t: usize,

    // Layer indices
    pub ring_layer: usize,
    pub marble_layers: (usize, usize),
    pub capture_layer: usize,
    pub layers_per_timestep: usize,

    // Global state indices
    pub supply_w: usize,
    pub supply_g: usize,
    pub supply_b: usize,
    pub p1_cap_w: usize,
    pub p1_cap_g: usize,
    pub p1_cap_b: usize,
    pub p2_cap_w: usize,
    pub p2_cap_g: usize,
    pub p2_cap_b: usize,
    pub cur_player: usize,

    // Player constants
    pub player_1: usize,
    pub player_2: usize,

    // Hex directions (y, x) offsets
    pub directions: Vec<(i32, i32)>,

    // Marble type mappings
    pub marble_to_layer: HashMap<String, usize>,
}

impl BoardConfig {
    /// Create standard BoardConfig for common board sizes
    pub fn standard(rings: usize, t: usize) -> Result<Self, String> {
        Self::with_mode(rings, t, GameMode::Standard)
    }

    /// Create blitz BoardConfig for common board sizes
    pub fn blitz(rings: usize, t: usize) -> Result<Self, String> {
        Self::with_mode(rings, t, GameMode::Blitz)
    }

    /// Create BoardConfig with specified mode
    fn with_mode(rings: usize, t: usize, mode: GameMode) -> Result<Self, String> {
        let width = match rings {
            37 => 7,
            48 => 8,
            61 => 9,
            _ => {
                return Err(format!(
                    "Unsupported ring count: {}. Use 37, 48, or 61.",
                    rings
                ))
            }
        };

        let win_conditions = match mode {
            GameMode::Standard => WinConditions::standard(),
            GameMode::Blitz => WinConditions::blitz(),
        };

        let mut marble_to_layer = HashMap::new();
        marble_to_layer.insert("w".to_string(), 1);
        marble_to_layer.insert("g".to_string(), 2);
        marble_to_layer.insert("b".to_string(), 3);

        Ok(BoardConfig {
            mode,
            win_conditions,
            width,
            rings,
            t,
            ring_layer: 0,
            marble_layers: (1, 4),
            capture_layer: t * 4,
            layers_per_timestep: 4,
            supply_w: 0,
            supply_g: 1,
            supply_b: 2,
            p1_cap_w: 3,
            p1_cap_g: 4,
            p1_cap_b: 5,
            p2_cap_w: 6,
            p2_cap_g: 7,
            p2_cap_b: 8,
            cur_player: 9,
            player_1: 0,
            player_2: 1,
            directions: vec![(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)],
            marble_to_layer,
        })
    }
}

/// Board state wrapper for PyO3 (zero-copy numpy array access)
#[pyclass]
pub struct BoardState {
    #[pyo3(get)]
    pub spatial: Py<PyArray3<f32>>,
    #[pyo3(get)]
    pub global: Py<PyArray1<f32>>,
    pub config: BoardConfig,
}

#[pymethods]
impl BoardState {
    #[new]
    #[pyo3(signature = (
        spatial,
        global,
        rings,
        t=1,
        blitz=false
    ))]
    fn new(
        py: Python<'_>,
        spatial: PyReadonlyArray3<f32>,
        global: PyReadonlyArray1<f32>,
        rings: usize,
        t: Option<usize>,
        blitz: Option<bool>,
    ) -> PyResult<Self> {
        let t = t.unwrap_or(1);
        let blitz = blitz.unwrap_or(false);

        let config = if blitz {
            BoardConfig::blitz(rings, t)
        } else {
            BoardConfig::standard(rings, t)
        }
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

        // Create owned Python arrays
        let spatial_arr = spatial.as_array().to_owned();
        let global_arr = global.as_array().to_owned();

        Ok(BoardState {
            spatial: PyArray3::from_array(py, &spatial_arr).into(),
            global: PyArray1::from_array(py, &global_arr).into(),
            config,
        })
    }

    /// Clone the board state for MCTS simulation
    fn clone_state(&self, py: Python<'_>) -> PyResult<Self> {
        let spatial_arr = self.spatial.bind(py).readonly().as_array().to_owned();
        let global_arr = self.global.bind(py).readonly().as_array().to_owned();

        Ok(BoardState {
            spatial: PyArray3::from_array(py, &spatial_arr).into(),
            global: PyArray1::from_array(py, &global_arr).into(),
            config: self.config.clone(),
        })
    }

    /// Get valid actions (for testing comparison with Python backend)
    fn get_valid_actions(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray3<f32>>)> {
        let spatial = self.spatial.bind(py).readonly().as_array().to_owned();
        let global = self.global.bind(py).readonly().as_array().to_owned();

        let (placement_mask, capture_mask) =
            crate::game::get_valid_actions(&spatial.view(), &global.view(), &self.config);

        Ok((
            PyArray3::from_array(py, &placement_mask).into(),
            PyArray3::from_array(py, &capture_mask).into(),
        ))
    }

    /// Canonicalize the spatial state and return transform metadata
    fn canonicalize_state(&self, py: Python<'_>) -> PyResult<(Py<PyArray3<f32>>, String, String)> {
        let spatial = self.spatial.bind(py).readonly();
        let (canonical, transform, inverse) =
            crate::canonicalization::canonicalize_state(&spatial.as_array(), &self.config);
        let canonical_py = PyArray3::from_array(py, &canonical).into();
        Ok((canonical_py, transform, inverse))
    }

    /// Apply a placement action (for testing comparison with Python backend)
    ///
    #[pyo3(signature=(marble_type, dst_y, dst_x, remove_y=None, remove_x=None))]
    fn apply_placement(
        &mut self,
        py: Python<'_>,
        marble_type: usize,
        dst_y: usize,
        dst_x: usize,
        remove_y: Option<usize>,
        remove_x: Option<usize>,
    ) -> PyResult<()> {
        let mut spatial = self.spatial.bind(py).readonly().as_array().to_owned();
        let mut global = self.global.bind(py).readonly().as_array().to_owned();

        crate::game::apply_placement(
            &mut spatial,
            &mut global,
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            &self.config,
        );

        // Update stored arrays
        self.spatial = PyArray3::from_array(py, &spatial).into();
        self.global = PyArray1::from_array(py, &global).into();

        Ok(())
    }

    /// Apply a capture action (for testing comparison with Python backend)
    fn apply_capture(
        &mut self,
        py: Python<'_>,
        start_y: usize,
        start_x: usize,
        direction: usize,
    ) -> PyResult<()> {
        let mut spatial = self.spatial.bind(py).readonly().as_array().to_owned();
        let mut global = self.global.bind(py).readonly().as_array().to_owned();

        crate::game::apply_capture(
            &mut spatial,
            &mut global,
            start_y,
            start_x,
            direction,
            &self.config,
        );

        // Update stored arrays
        self.spatial = PyArray3::from_array(py, &spatial).into();
        self.global = PyArray1::from_array(py, &global).into();

        Ok(())
    }

    /// Get spatial state (for testing)
    fn get_spatial<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.spatial.bind(py).clone()
    }

    /// Get global state (for testing)
    fn get_global<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.global.bind(py).clone()
    }
}
