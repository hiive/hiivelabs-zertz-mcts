//! # Python Bindings for Zertz MCTS
//!
//! This module provides PyO3 bindings for the generic MCTS with ZertzGame.
//! It wraps `MCTSSearch<ZertzGame>` and provides Python-compatible interfaces.

use super::{ZertzAction, ZertzGame};
use crate::mcts::MCTSSearch;
use crate::node::MCTSNode;
use ndarray::{Array1, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Python-facing wrapper for MCTS with ZertzGame
#[pyclass(name = "ZertzMCTS")]
pub struct PyZertzMCTS {
    search: MCTSSearch<ZertzGame>,
    game: Arc<ZertzGame>,
}

#[pymethods]
impl PyZertzMCTS {
    /// Create a new Zertz MCTS instance
    #[new]
    #[pyo3(signature = (
        rings,
        exploration_constant=None,
        widening_constant=None,
        fpu_reduction=None,
        rave_constant=None,
        use_transposition_table=true,
        use_transposition_lookups=true,
        blitz=false,
        t=1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rings: usize,
        exploration_constant: Option<f32>,
        widening_constant: Option<f32>,
        fpu_reduction: Option<f32>,
        rave_constant: Option<f32>,
        use_transposition_table: bool,
        use_transposition_lookups: bool,
        blitz: bool,
        t: usize,
    ) -> PyResult<Self> {
        // Create ZertzGame instance
        let game = Arc::new(
            ZertzGame::new(rings, t, blitz)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        );

        // Create generic MCTS search
        let search = MCTSSearch::new(
            Arc::clone(&game),
            exploration_constant,
            widening_constant,
            fpu_reduction,
            rave_constant,
            Some(use_transposition_table),
            Some(use_transposition_lookups),
        );

        Ok(Self { search, game })
    }

    /// Set deterministic RNG seed
    #[pyo3(signature = (seed=None))]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.search.set_seed(seed);
    }

    /// Enable/disable transposition table
    pub fn set_transposition_table_enabled(&mut self, enabled: bool) {
        self.search.set_transposition_table_enabled(enabled);
    }

    /// Enable/disable transposition table lookups
    pub fn set_transposition_lookups(&mut self, enabled: bool) {
        self.search.set_transposition_lookups(enabled);
    }

    /// Clear transposition table
    pub fn clear_transposition_table(&mut self) {
        self.search.clear_transposition_table();
    }

    /// Get number of children of last root
    pub fn last_root_children(&self) -> usize {
        self.search.last_root_children()
    }

    /// Get visit count of last root
    pub fn last_root_visits(&self) -> u32 {
        self.search.last_root_visits()
    }

    /// Get average value of last root
    pub fn last_root_value(&self) -> f32 {
        self.search.last_root_value()
    }

    /// Get per-child statistics from last search
    ///
    /// Returns list of (action_type, action_data, normalized_score) tuples:
    /// - action_type: "PUT", "CAP", or "PASS"
    /// - action_data: Action-specific tuple or None
    /// - normalized_score: Visit count normalized to [0.0, 1.0]
    pub fn last_child_statistics(&self) -> Vec<(String, Option<(usize, usize, usize)>, f32)> {
        let child_stats = self.search.last_child_stats.lock().unwrap();
        let width = self.game.config().width;

        child_stats
            .iter()
            .map(|(action, score)| match action {
                ZertzAction::Placement {
                    marble_type,
                    dst_y,
                    dst_x,
                    remove_y,
                    remove_x,
                } => {
                    let dst_flat = dst_y * width + dst_x;
                    let remove_flat = match (remove_y, remove_x) {
                        (Some(ry), Some(rx)) => ry * width + rx,
                        _ => width * width,
                    };
                    (
                        "PUT".to_string(),
                        Some((*marble_type, dst_flat, remove_flat)),
                        *score,
                    )
                }
                ZertzAction::Capture {
                    start_y,
                    start_x,
                    dest_y,
                    dest_x,
                } => {
                    let src_flat = start_y * width + start_x;
                    let dst_flat = dest_y * width + dest_x;
                    (
                    "CAP".to_string(),
                    Some((0, src_flat, dst_flat)),  // Note: only 3 values fit in tuple
                    *score,
                    )
                }
                ZertzAction::Pass => ("PASS".to_string(), None, *score),
            })
            .collect()
    }

    /// Run MCTS search (serial mode)
    #[pyo3(signature = (
        spatial_state,
        global_state,
        iterations,
        max_depth=None,
        time_limit=None,
        use_transposition_table=None,
        use_transposition_lookups=None,
        clear_table=false,
        verbose=false,
        seed=None,
        progress_callback=None,
        progress_interval_ms=100
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &mut self,
        py: Python,
        spatial_state: PyReadonlyArray3<f32>,
        global_state: PyReadonlyArray1<f32>,
        iterations: usize,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
        clear_table: bool,
        verbose: bool,
        seed: Option<u64>,
        progress_callback: Option<Py<PyAny>>,
        progress_interval_ms: u64,
    ) -> PyResult<(String, Option<(Option<usize>, (usize, usize), (usize, usize))>)> {
        // Set seed if provided
        if let Some(s) = seed {
            self.search.set_seed(Some(s));
        }

        // Configure transposition table
        if let Some(flag) = use_transposition_table {
            self.search.set_transposition_table_enabled(flag);
        }
        if let Some(flag) = use_transposition_lookups {
            self.search.set_transposition_lookups(flag);
        }
        if clear_table {
            self.search.clear_transposition_table();
        }

        // Convert numpy arrays to ndarray
        let spatial_array: Array3<f32> = spatial_state.as_array().to_owned();
        let global_array: Array1<f32> = global_state.as_array().to_owned();

        // Create root node
        let root = Arc::new(MCTSNode::new(
            spatial_array,
            global_array,
            Arc::clone(&self.game),
            None,
        ));

        // Fire SearchStarted callback if provided
        if let Some(ref callback) = progress_callback {
            self.fire_search_started(py, callback, &root)?;
        }

        // Create search options before the loop
        let options = self.create_search_options(
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );

        let start_time = Instant::now();
        let mut last_progress = Instant::now();

        // Run MCTS iterations
        for i in 0..iterations {
            // Check time limit
            if let Some(limit) = time_limit {
                if start_time.elapsed().as_secs_f32() >= limit {
                    if verbose {
                        println!("Time limit reached after {} iterations", i);
                    }
                    break;
                }
            }

            // Run single iteration
            self.search.run_iteration(
                Arc::clone(&root),
                &options,
                max_depth,
                time_limit,
                start_time,
            );

            // Fire progress callback
            if let Some(ref callback) = progress_callback {
                let elapsed = last_progress.elapsed().as_millis() as u64;
                if elapsed >= progress_interval_ms {
                    self.fire_iteration_progress(py, callback, &root, i + 1, start_time.elapsed())?;
                    last_progress = Instant::now();
                }
            }
        }

        let elapsed = start_time.elapsed();

        // Fire SearchEnded callback
        if let Some(ref callback) = progress_callback {
            self.fire_search_ended(py, callback, &root, iterations, elapsed)?;
        }

        if verbose {
            let children_count = root.children.read().unwrap().len();
            println!(
                "Completed {} iterations in {:.2}s ({:.0} iter/s) - Root: {} visits, {:.3} value, {} children",
                iterations,
                elapsed.as_secs_f32(),
                iterations as f32 / elapsed.as_secs_f32(),
                root.get_visits(),
                root.get_value(),
                children_count
            );
        }

        // Capture stats and select best action
        self.search.capture_root_stats(&root);
        self.select_best_action(&root)
    }

    /// Run MCTS search (parallel mode using rayon)
    #[pyo3(signature = (
        spatial_state,
        global_state,
        iterations,
        max_depth=None,
        time_limit=None,
        use_transposition_table=None,
        use_transposition_lookups=None,
        clear_table=false,
        verbose=false,
        seed=None,
        progress_callback=None,
        progress_interval_ms=100
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn search_parallel(
        &mut self,
        py: Python,
        spatial_state: PyReadonlyArray3<f32>,
        global_state: PyReadonlyArray1<f32>,
        iterations: usize,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
        clear_table: bool,
        verbose: bool,
        seed: Option<u64>,
        progress_callback: Option<Py<PyAny>>,
        progress_interval_ms: u64,
    ) -> PyResult<(String, Option<(Option<usize>, (usize, usize), (usize, usize))>)> {
        // Set seed if provided
        if let Some(s) = seed {
            self.search.set_seed(Some(s));
        }

        // Configure transposition table
        if let Some(flag) = use_transposition_table {
            self.search.set_transposition_table_enabled(flag);
        }
        if let Some(flag) = use_transposition_lookups {
            self.search.set_transposition_lookups(flag);
        }
        if clear_table {
            self.search.clear_transposition_table();
        }

        // Convert numpy arrays to ndarray
        let spatial_array: Array3<f32> = spatial_state.as_array().to_owned();
        let global_array: Array1<f32> = global_state.as_array().to_owned();

        // Create root node
        let root = Arc::new(MCTSNode::new(
            spatial_array,
            global_array,
            Arc::clone(&self.game),
            None,
        ));

        // Fire SearchStarted callback if provided
        if let Some(ref callback) = progress_callback {
            self.fire_search_started(py, callback, &root)?;
        }

        // Create search options before the parallel loop
        let options = self.create_search_options(
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );

        let start_time = Instant::now();
        let last_progress = std::sync::Arc::new(std::sync::Mutex::new(Instant::now()));

        // Run MCTS iterations in parallel
        (0..iterations).into_par_iter().for_each(|_i| {
            // Check time limit
            if let Some(limit) = time_limit {
                if start_time.elapsed().as_secs_f32() >= limit {
                    return;
                }
            }

            // Run single iteration
            self.search.run_iteration(
                Arc::clone(&root),
                &options,
                max_depth,
                time_limit,
                start_time,
            );

            // Fire progress callback (with throttling)
            if let Some(ref _callback) = progress_callback {
                let mut last = last_progress.lock().unwrap();
                let elapsed = last.elapsed().as_millis() as u64;
                if elapsed >= progress_interval_ms {
                    // Note: Can't call Python from rayon thread without GIL
                    // Progress callbacks in parallel mode are best-effort
                    *last = Instant::now();
                }
            }
        });

        let elapsed = start_time.elapsed();

        // Fire SearchEnded callback
        if let Some(ref callback) = progress_callback {
            self.fire_search_ended(py, callback, &root, iterations, elapsed)?;
        }

        if verbose {
            let children_count = root.children.read().unwrap().len();
            println!(
                "Completed {} iterations in {:.2}s ({:.0} iter/s) - Root: {} visits, {:.3} value, {} children",
                iterations,
                elapsed.as_secs_f32(),
                iterations as f32 / elapsed.as_secs_f32(),
                root.get_visits(),
                root.get_value(),
                children_count
            );
        }

        // Capture stats and select best action
        self.search.capture_root_stats(&root);
        self.select_best_action(&root)
    }
}

// Private helper methods
impl PyZertzMCTS {
    /// Create SearchOptions from parameters
    fn create_search_options(
        &mut self,
        use_table_override: Option<bool>,
        use_lookups_override: Option<bool>,
        clear_table: bool,
    ) -> crate::mcts::SearchOptions<ZertzGame> {
        crate::mcts::SearchOptions::new(
            &mut self.search,
            use_table_override,
            use_lookups_override,
            clear_table,
        )
    }

    /// Select best action from root (most visits)
    fn select_best_action(
        &self,
        root: &MCTSNode<ZertzGame>,
    ) -> PyResult<(String, Option<(Option<usize>, (usize, usize), (usize, usize))>)> {
        let children = root.children.read().unwrap();

        if children.is_empty() {
            return Ok(("PASS".to_string(), None));
        }

        // Select child with most visits
        let best = children.iter().max_by_key(|(_, child)| child.get_visits());

        if let Some((action, _)) = best {
            let width = self.game.config().width;
            action.to_tuple(width)
    } else {
            Ok(("PASS".to_string(), None))
        }
    }

    /// Fire SearchStarted event callback
    fn fire_search_started(
        &self,
        py: Python,
        callback: &Py<PyAny>,
        root: &MCTSNode<ZertzGame>,
    ) -> PyResult<()> {
        let event = PyDict::new(py);
        event.set_item("event", "SearchStarted")?;
        event.set_item("root_visits", root.get_visits())?;
        event.set_item("root_value", root.get_value())?;

        callback.call1(py, (event,))?;
        Ok(())
    }

    /// Fire IterationProgress event callback
    fn fire_iteration_progress(
        &self,
        py: Python,
        callback: &Py<PyAny>,
        root: &MCTSNode<ZertzGame>,
        iteration: usize,
        elapsed: Duration,
    ) -> PyResult<()> {
        let event = PyDict::new(py);
        event.set_item("event", "IterationProgress")?;
        event.set_item("iteration", iteration)?;
        event.set_item("root_visits", root.get_visits())?;
        event.set_item("root_value", root.get_value())?;
        event.set_item("elapsed_ms", elapsed.as_millis() as u64)?;

        callback.call1(py, (event,))?;
        Ok(())
    }

    /// Fire SearchEnded event callback
    fn fire_search_ended(
        &self,
        py: Python,
        callback: &Py<PyAny>,
        _root: &MCTSNode<ZertzGame>,
        total_iterations: usize,
        elapsed: Duration,
    ) -> PyResult<()> {
        let event = PyDict::new(py);
        event.set_item("event", "SearchEnded")?;
        event.set_item("total_iterations", total_iterations)?;
        event.set_item("total_time_ms", elapsed.as_millis() as u64)?;

        callback.call1(py, (event,))?;
        Ok(())
    }
}
