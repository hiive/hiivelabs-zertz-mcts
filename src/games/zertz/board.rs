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
#[pyclass]
#[derive(Clone, Debug)]
pub struct BoardConfig {
    pub mode: GameMode,
    pub win_conditions: WinConditions,
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub rings: usize,
    #[pyo3(get)]
    pub t: usize,

    // Layer indices
    #[pyo3(get)]
    pub ring_layer: usize,
    pub marble_layers: (usize, usize),
    #[pyo3(get)]
    pub capture_layer: usize,
    #[pyo3(get)]
    pub layers_per_timestep: usize,

    // Global state indices
    #[pyo3(get)]
    pub supply_w: usize,
    #[pyo3(get)]
    pub supply_g: usize,
    #[pyo3(get)]
    pub supply_b: usize,
    #[pyo3(get)]
    pub p1_cap_w: usize,
    #[pyo3(get)]
    pub p1_cap_g: usize,
    #[pyo3(get)]
    pub p1_cap_b: usize,
    #[pyo3(get)]
    pub p2_cap_w: usize,
    #[pyo3(get)]
    pub p2_cap_g: usize,
    #[pyo3(get)]
    pub p2_cap_b: usize,
    #[pyo3(get)]
    pub cur_player: usize,

    // Player constants
    #[pyo3(get)]
    pub player_1: usize,
    #[pyo3(get)]
    pub player_2: usize,

    // Hex directions (y, x) offsets - not exposed to Python (Vec is complex)
    pub directions: Vec<(i32, i32)>,

    // Marble type mappings - not exposed to Python (HashMap is complex)
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

/// Python methods for BoardConfig
#[pymethods]
impl BoardConfig {
    /// Create standard BoardConfig (Python constructor)
    #[staticmethod]
    #[pyo3(signature = (rings, t=1))]
    fn standard_config(rings: usize, t: Option<usize>) -> PyResult<Self> {
        let t = t.unwrap_or(1);
        BoardConfig::standard(rings, t).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Create blitz BoardConfig (Python constructor)
    #[staticmethod]
    #[pyo3(signature = (rings, t=1))]
    fn blitz_config(rings: usize, t: Option<usize>) -> PyResult<Self> {
        let t = t.unwrap_or(1);
        BoardConfig::blitz(rings, t).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Get marble type for a layer index
    fn get_marble_layer(&self, marble_type: &str) -> Option<usize> {
        self.marble_to_layer.get(marble_type).copied()
    }

    /// Get hex direction offsets
    fn get_directions(&self) -> Vec<(i32, i32)> {
        self.directions.clone()
    }

    /// Check if game is in blitz mode
    fn is_blitz(&self) -> bool {
        self.mode == GameMode::Blitz
    }

    /// Get Player 1 capture slice (indices 3-6: w, g, b)
    #[getter]
    fn p1_cap_slice(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let slice = pyo3::types::PySlice::new(py, 3, 6, 1);
        Ok(slice.into())
    }

    /// Get Player 2 capture slice (indices 6-9: w, g, b)
    #[getter]
    fn p2_cap_slice(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let slice = pyo3::types::PySlice::new(py, 6, 9, 1);
        Ok(slice.into())
    }
}

impl BoardConfig {
    /// Helper: compute destination coordinates from start position and direction
    ///
    /// Useful for tests and for converting direction-based capture notation
    /// to coordinate-based notation.
    pub fn dest_from_direction(
        &self,
        start_y: usize,
        start_x: usize,
        direction: usize,
    ) -> (usize, usize) {
        let (dy, dx) = self.directions[direction];
        let dest_y = ((start_y as i32) + 2 * dy) as usize;
        let dest_x = ((start_x as i32) + 2 * dx) as usize;
        (dest_y, dest_x)
    }
}

/// Board state wrapper for PyO3 (zero-copy numpy array access)
#[pyclass]
pub struct BoardState {
    #[pyo3(get)]
    pub spatial_state: Py<PyArray3<f32>>,
    #[pyo3(get)]
    pub global: Py<PyArray1<f32>>,
    pub config: BoardConfig,
}

#[pymethods]
impl BoardState {
    #[new]
    #[pyo3(signature = (
        spatial_state,
        global,
        rings,
        t=1,
        blitz=false
    ))]
    fn new(
        py: Python<'_>,
        spatial_state: PyReadonlyArray3<f32>,
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
        let spatial_state_arr = spatial_state.as_array().to_owned();
        let global_statearr = global.as_array().to_owned();

        Ok(BoardState {
            spatial_state: PyArray3::from_array(py, &spatial_state_arr).into(),
            global: PyArray1::from_array(py, &global_statearr).into(),
            config,
        })
    }

    /// Clone the board state for MCTS simulation
    fn clone_state(&self, py: Python<'_>) -> PyResult<Self> {
        let spatial_state_arr = self.spatial_state.bind(py).readonly().as_array().to_owned();
        let global_statearr = self.global.bind(py).readonly().as_array().to_owned();

        Ok(BoardState {
            spatial_state: PyArray3::from_array(py, &spatial_state_arr).into(),
            global: PyArray1::from_array(py, &global_statearr).into(),
            config: self.config.clone(),
        })
    }

    /// Get valid actions (for testing comparison with Python backend)
    fn get_valid_actions(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray3<f32>>)> {
        let spatial_state = self.spatial_state.bind(py).readonly().as_array().to_owned();
        let global = self.global.bind(py).readonly().as_array().to_owned();

        let (placement_mask, capture_mask) =
            super::logic::get_valid_actions(&spatial_state.view(), &global.view(), &self.config);

        Ok((
            PyArray3::from_array(py, &placement_mask).into(),
            PyArray3::from_array(py, &capture_mask).into(),
        ))
    }

    /// Canonicalize the spatial_state state and return transform metadata
    fn canonicalize_state(&self, py: Python<'_>) -> PyResult<(Py<PyArray3<f32>>, String, String)> {
        let spatial_state = self.spatial_state.bind(py).readonly();
        let (canonical, transform, inverse) =
            super::canonicalization::canonicalize_state(&spatial_state.as_array(), &self.config);
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
        let mut spatial_state = self.spatial_state.bind(py).readonly().as_array().to_owned();
        let mut global = self.global.bind(py).readonly().as_array().to_owned();

        super::logic::apply_placement(
            &mut spatial_state.view_mut(),
            &mut global.view_mut(),
            marble_type,
            dst_y,
            dst_x,
            remove_y,
            remove_x,
            &self.config,
        );

        // Update stored arrays
        self.spatial_state = PyArray3::from_array(py, &spatial_state).into();
        self.global = PyArray1::from_array(py, &global).into();

        Ok(())
    }

    /// Apply a capture action (for testing comparison with Python backend)
    fn apply_capture(
        &mut self,
        py: Python<'_>,
        start_y: usize,
        start_x: usize,
        dest_y: usize,
        dest_x: usize,
    ) -> PyResult<()> {
        let mut spatial_state = self.spatial_state.bind(py).readonly().as_array().to_owned();
        let mut global = self.global.bind(py).readonly().as_array().to_owned();

        super::logic::apply_capture(
            &mut spatial_state.view_mut(),
            &mut global.view_mut(),
            start_y,
            start_x,
            dest_y,
            dest_x,
            &self.config,
        );

        // Update stored arrays
        self.spatial_state = PyArray3::from_array(py, &spatial_state).into();
        self.global = PyArray1::from_array(py, &global).into();

        Ok(())
    }

    /// Get spatial_state state (for testing)
    fn get_spatial_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.spatial_state.bind(py).clone()
    }

    /// Get global state (for testing)
    fn get_global<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.global.bind(py).clone()
    }
}
