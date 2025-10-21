use numpy::{PyArray3, PyArray1, PyArrayMethods, PyReadonlyArray3, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Immutable board configuration (mirrors Python BoardConfig)
#[derive(Clone, Debug)]
pub struct BoardConfig {
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
        let width = match rings {
            37 => 7,
            48 => 8,
            61 => 9,
            _ => return Err(format!("Unsupported ring count: {}. Use 37, 48, or 61.", rings)),
        };

        let mut marble_to_layer = HashMap::new();
        marble_to_layer.insert("w".to_string(), 1);
        marble_to_layer.insert("g".to_string(), 2);
        marble_to_layer.insert("b".to_string(), 3);

        Ok(BoardConfig {
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
            directions: vec![
                (1, 0), (0, -1), (-1, -1),
                (-1, 0), (0, 1), (1, 1)
            ],
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
    fn new(
        py: Python<'_>,
        spatial: PyReadonlyArray3<f32>,
        global: PyReadonlyArray1<f32>,
        rings: usize,
        t: usize,
    ) -> PyResult<Self> {
        let config = BoardConfig::standard(rings, t)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        // Create owned Python arrays
        let spatial_arr = spatial.as_array().to_owned();
        let global_arr = global.as_array().to_owned();

        Ok(BoardState {
            spatial: PyArray3::from_array_bound(py, &spatial_arr).into(),
            global: PyArray1::from_array_bound(py, &global_arr).into(),
            config,
        })
    }

    /// Clone the board state for MCTS simulation
    fn clone_state(&self, py: Python<'_>) -> PyResult<Self> {
        let spatial_arr = self.spatial.bind(py).readonly().as_array().to_owned();
        let global_arr = self.global.bind(py).readonly().as_array().to_owned();

        Ok(BoardState {
            spatial: PyArray3::from_array_bound(py, &spatial_arr).into(),
            global: PyArray1::from_array_bound(py, &global_arr).into(),
            config: self.config.clone(),
        })
    }

    /// Get valid actions (for testing comparison with Python backend)
    fn get_valid_actions(&self, py: Python<'_>) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray3<f32>>)> {
        let spatial = self.spatial.bind(py).readonly().as_array().to_owned();
        let global = self.global.bind(py).readonly().as_array().to_owned();

        let (placement_mask, capture_mask) = crate::game::get_valid_actions(
            &spatial.view(),
            &global.view(),
            &self.config,
        );

        Ok((
            PyArray3::from_array_bound(py, &placement_mask).into(),
            PyArray3::from_array_bound(py, &capture_mask).into(),
        ))
    }

    /// Apply a placement action (for testing comparison with Python backend)
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
        self.spatial = PyArray3::from_array_bound(py, &spatial).into();
        self.global = PyArray1::from_array_bound(py, &global).into();

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
        self.spatial = PyArray3::from_array_bound(py, &spatial).into();
        self.global = PyArray1::from_array_bound(py, &global).into();

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