//! # Monte Carlo Tree Search (MCTS) Implementation
//!
//! This module provides a high-performance MCTS implementation for Zèrtz with:
//! - Serial and parallel (rayon) search modes
//! - Transposition table for state caching
//! - UCB1 child selection with FPU (First Play Urgency)
//! - Progressive widening support
//! - Virtual loss for lock-free parallel search
//! - Python bindings via PyO3
//!
//! ## Architecture
//!
//! **Architectural Pattern**: MCTS delegates ALL game logic to `game.rs`
//! - `mcts.rs` handles tree search algorithms (selection/expansion/simulation/backprop)
//! - `game.rs` contains game rules (move generation, terminal checks, outcomes)
//! - This mirrors Python: `mcts_tree.py` → `zertz_logic.py`
//!
//! ## MCTS Algorithm
//!
//! Standard MCTS phases:
//! 1. **Selection**: Traverse tree using UCB1 until reaching unexpanded/terminal node
//! 2. **Expansion**: Add a new child node for an untried action
//! 3. **Simulation**: Play out a random game (rollout) from the new node
//! 4. **Backpropagation**: Update statistics up the tree (values flip each level)
//!
//! ## Thread Safety
//!
//! - **Virtual Loss**: Added during selection, removed during backpropagation
//!   - Makes selected path less attractive to other threads
//!   - Enables lock-free parallel search
//! - **Atomic Operations**: Node statistics use `AtomicU32` for thread-safe updates
//! - **Transposition Table**: Uses `DashMap` for concurrent access
//!
//! ## Key Parameters
//!
//! - `exploration_constant`: UCB1 exploration weight (default: 1.41 = √2)
//! - `fpu_reduction`: First Play Urgency - penalizes unvisited nodes
//! - `widening_constant`: Progressive widening - limits expansion rate
//! - `max_depth`: Limits rollout depth (None = full game)
//! - `time_limit`: Max search time in seconds

use ndarray::{Array1, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::board::BoardConfig;
use crate::game::{
    apply_capture, apply_placement, get_game_outcome, get_valid_actions, is_game_over,
};
use crate::node::{Action, MCTSNode};
use crate::transposition::TranspositionTable;

// ============================================================================
// MCTS SEARCH
// ============================================================================

/// Main MCTS search engine with Python bindings.
///
/// This struct maintains search configuration and exposes both serial and
/// parallel search methods to Python via PyO3.
#[pyclass]
pub struct MCTSSearch {
    exploration_constant: f32,
    widening_constant: Option<f32>,
    fpu_reduction: Option<f32>,
    rave_constant: Option<f32>,
    use_transposition_table: bool,
    use_transposition_lookups: bool,
    transposition_table: Option<Arc<TranspositionTable>>,
    last_root_children: usize,
    last_root_visits: u32,
    last_root_value: f32,
    last_child_stats: Mutex<Vec<(Action, f32)>>, // Store normalized visit scores per action
    last_board_width: usize, // Board width for coordinate conversion
    rng: Mutex<Option<StdRng>>,
    seed: Mutex<Option<u64>>, // Base seed for deriving per-thread seeds
    seed_generation: AtomicU64, // Increments on each set_seed call to invalidate thread-local caches
    #[cfg(feature = "metrics")]
    metrics: Arc<MCTSMetrics>,
}

#[pymethods]
impl MCTSSearch {
    /// Enable/disable transposition table caching (persists across searches)
    pub fn set_transposition_table_enabled(&mut self, enabled: bool) {
        self.use_transposition_table = enabled;
        if !enabled {
            self.transposition_table = None;
        } else if self.transposition_table.is_none() {
            self.transposition_table = Some(Arc::new(TranspositionTable::new()));
        }
    }

    /// Enable/disable transposition lookups for initializing child nodes
    pub fn set_transposition_lookups(&mut self, enabled: bool) {
        self.use_transposition_lookups = enabled;
    }

    /// Clear any cached transposition data
    pub fn clear_transposition_table(&mut self) {
        if let Some(table) = &self.transposition_table {
            table.clear();
        }
    }

    #[new]
    #[pyo3(signature = (
        exploration_constant=None,
        widening_constant=None,
        fpu_reduction=None,
        rave_constant=None,
        use_transposition_table=None,
        use_transposition_lookups=None
    ))]
    fn new(
        exploration_constant: Option<f32>,
        widening_constant: Option<f32>,
        fpu_reduction: Option<f32>,
        rave_constant: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
    ) -> Self {
        Self {
            exploration_constant: exploration_constant.unwrap_or(1.41),
            widening_constant,
            fpu_reduction,
            rave_constant,
            use_transposition_table: use_transposition_table.unwrap_or(true),
            use_transposition_lookups: use_transposition_lookups.unwrap_or(true),
            transposition_table: None,
            last_root_children: 0,
            last_root_visits: 0,
            last_root_value: 0.0,
            last_child_stats: Mutex::new(Vec::new()),
            last_board_width: 7,
            rng: Mutex::new(None),
            seed: Mutex::new(None),
            seed_generation: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            metrics: Arc::new(MCTSMetrics::new()),
        }
    }

    /// Get metrics as JSON (only available with 'metrics' feature)
    #[cfg(feature = "metrics")]
    pub fn get_metrics_json(&self) -> String {
        self.metrics.to_json()
    }

    /// Print metrics summary to stderr (only available with 'metrics' feature)
    #[cfg(feature = "metrics")]
    pub fn print_metrics(&self) {
        self.metrics.print_summary();
    }

    /// Reset metrics (only available with 'metrics' feature)
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&mut self) {
        self.metrics = Arc::new(MCTSMetrics::new());
    }

    /// Set deterministic RNG seed (pass None to restore system randomness)
    #[pyo3(signature = (seed=None))]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        let mut guard = self.rng.lock().unwrap();
        let mut seed_guard = self.seed.lock().unwrap();
        if let Some(value) = seed {
            *guard = Some(StdRng::seed_from_u64(value));
            *seed_guard = Some(value);
        } else {
            *guard = None;
            *seed_guard = None;
        }
        // Increment generation to invalidate thread-local caches
        self.seed_generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Run MCTS search (serial mode)
    #[pyo3(signature = (
        spatial_state,
        global_state,
        rings,
        iterations,
        t=1,
        max_depth=None,
        time_limit=None,
        use_transposition_table=None,
        use_transposition_lookups=None,
        clear_table=false,
        verbose=false,
        seed=None,
        blitz=false,
        progress_callback=None,
        progress_interval_ms=100
    ))]
    fn search(
        &mut self,
        py: Python<'_>,
        spatial_state: PyReadonlyArray3<f32>,
        global_state: PyReadonlyArray1<f32>,
        rings: usize,
        iterations: usize,
        t: Option<usize>,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
        clear_table: bool,
        verbose: Option<bool>,
        seed: Option<u64>,
        blitz: Option<bool>,
        progress_callback: Option<PyObject>,
        progress_interval_ms: Option<u64>,
    ) -> PyResult<(String, Option<(usize, usize, usize)>)> {
        let search_options = SearchOptions::new(
            self,
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );

        let t = t.unwrap_or(1);
        let verbose = verbose.unwrap_or(false);
        let blitz = blitz.unwrap_or(false);
        if let Some(value) = seed {
            self.set_seed(Some(value));
        }

        let config = Arc::new(
            if blitz {
                BoardConfig::blitz(rings, t)
            } else {
                BoardConfig::standard(rings, t)
            }
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
        );

        let spatial_state_arr = spatial_state.as_array().to_owned();
        let global_state_arr = global_state.as_array().to_owned();

        let shared_entry = if search_options.use_lookups() {
            search_options.table_ref().map(|table_ref| {
                table_ref.get_or_insert(&spatial_state_arr.view(), &global_state_arr.view(), config.as_ref())
            })
        } else {
            None
        };

        let root = Arc::new(MCTSNode::new(
            spatial_state_arr,
            global_state_arr,
            Arc::clone(&config),
            shared_entry,
        ));

        let start = Instant::now();
        let progress_interval = Duration::from_millis(progress_interval_ms.unwrap_or(100));
        let mut last_progress_time = start;
        let mut iteration_count = 0usize;

        // Fire SearchStarted event
        if let Some(ref callback) = progress_callback {
            self.fire_search_started(py, callback, &root, iterations)?;
        }

        // Run MCTS iterations
        for i in 0..iterations {
            if !self.run_iteration(
                Arc::clone(&root),
                &search_options,
                max_depth,
                time_limit,
                start,
            ) {
                break;
            }

            iteration_count = i + 1;

            // Fire SearchProgress event at intervals
            if let Some(ref callback) = progress_callback {
                let elapsed_since_last = start.elapsed() - (last_progress_time - start);
                if elapsed_since_last >= progress_interval {
                    self.fire_search_progress(py, callback, &root, iteration_count)?;
                    last_progress_time = start + start.elapsed();
                }
            }
        }

        let elapsed = start.elapsed();

        // Fire SearchEnded event
        if let Some(ref callback) = progress_callback {
            self.fire_search_ended(py, callback, &root, iteration_count, elapsed)?;
        }

        if verbose {
            let value = root.get_value();
            let children_count = root.children.read().unwrap().len();
            eprintln!(
                "MCTS: {} iterations in {:.2}s ({:.0} iter/s), value={:.3}, {} children",
                iterations,
                elapsed.as_secs_f32(),
                iterations as f32 / elapsed.as_secs_f32(),
                value,
                children_count
            );
        }

        // Select best action
        self.capture_root_stats(&root);
        self.select_best_action(&root)
    }

    /// Run MCTS search (parallel mode using rayon)
    #[pyo3(signature = (
        spatial_state,
        global_state,
        rings,
        iterations,
        t=1,
        max_depth=None,
        time_limit=None,
        use_transposition_table=None,
        use_transposition_lookups=None,
        clear_table=false,
        num_threads=16,
        verbose=false,
        seed=None,
        blitz=false,
        progress_callback=None,
        progress_interval_ms=100
    ))]
    fn search_parallel(
        &mut self,
        py: Python<'_>,
        spatial_state: PyReadonlyArray3<f32>,
        global_state: PyReadonlyArray1<f32>,
        rings: usize,
        iterations: usize,
        t: Option<usize>,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
        clear_table: bool,
        num_threads: Option<usize>,
        verbose: Option<bool>,
        seed: Option<u64>,
        blitz: Option<bool>,
        progress_callback: Option<PyObject>,
        progress_interval_ms: Option<u64>,
    ) -> PyResult<(String, Option<(usize, usize, usize)>)> {
        let search_options = SearchOptions::new(
            self,
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );
        let t = t.unwrap_or(1);
        let num_threads = num_threads.unwrap_or(16);
        let verbose = verbose.unwrap_or(false);
        let blitz = blitz.unwrap_or(false);
        if let Some(value) = seed {
            self.set_seed(Some(value));
        }

        // Configure thread pool and capture it
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let config = Arc::new(
            if blitz {
                BoardConfig::blitz(rings, t)
            } else {
                BoardConfig::standard(rings, t)
            }
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
        );

        let spatial_state_arr = spatial_state.as_array().to_owned();
        let global_state_arr = global_state.as_array().to_owned();

        let shared_entry = if search_options.use_lookups() {
            search_options.table_ref().map(|table_ref| {
                table_ref.get_or_insert(&spatial_state_arr.view(), &global_state_arr.view(), config.as_ref())
            })
        } else {
            None
        };

        let root = Arc::new(MCTSNode::new(
            spatial_state_arr,
            global_state_arr,
            Arc::clone(&config),
            shared_entry,
        ));

        let start = Instant::now();

        // Fire SearchStarted event (with GIL)
        if let Some(ref callback) = progress_callback {
            self.fire_search_started(py, callback, &root, iterations)?;
        }

        // Release GIL for parallel work
        py.allow_threads(|| {
            let progress_interval = Duration::from_millis(progress_interval_ms.unwrap_or(100));
            let mut last_progress_time = start;
            let mut completed = 0;

            // Run iterations in batches to allow periodic progress callbacks
            let batch_size = 100; // Check for progress every 100 iterations
            for batch_start in (0..iterations).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(iterations);
                let batch_iterations = batch_end - batch_start;

                // Run batch in parallel using the configured thread pool
                thread_pool.install(|| {
                    (0..batch_iterations).into_par_iter().for_each(|_| {
                        self.run_iteration(
                            Arc::clone(&root),
                            &search_options,
                            max_depth,
                            time_limit,
                            start,
                        );
                    });
                });

                completed = batch_end;

                // Fire progress callback if interval elapsed
                if let Some(ref callback) = progress_callback {
                    let elapsed_since_last = start.elapsed() - (last_progress_time - start);
                    if elapsed_since_last >= progress_interval {
                        // Reacquire GIL for callback
                        Python::with_gil(|py| {
                            if let Err(e) = self.fire_search_progress(py, callback, &root, completed) {
                                eprintln!("SearchProgress error: {}", e);
                            }
                        });
                        last_progress_time = start + start.elapsed();
                    }
                }
            }
        });

        let elapsed = start.elapsed();

        // Fire SearchEnded event (with GIL - automatically reacquired after detach)
        if let Some(ref callback) = progress_callback {
            self.fire_search_ended(py, callback, &root, iterations, elapsed)?;
        }

        if verbose {
            let value = root.get_value();
            let children_count = root.children.read().unwrap().len();
            eprintln!(
                "MCTS (parallel): {} iterations in {:.2}s ({:.0} iter/s), value={:.3}, {} children",
                iterations,
                elapsed.as_secs_f32(),
                iterations as f32 / elapsed.as_secs_f32(),
                value,
                children_count
            );
        }

        // Select best action
        self.capture_root_stats(&root);
        self.select_best_action(&root)
    }

    pub fn last_root_children(&self) -> usize {
        self.last_root_children
    }

    pub fn last_root_visits(&self) -> u32 {
        self.last_root_visits
    }

    pub fn last_root_value(&self) -> f32 {
        self.last_root_value
    }

    /// Get per-child statistics from last search as (action_type, action_data, normalized_score) tuples
    ///
    /// Returns a list of tuples where each tuple contains:
    /// - action_type: "PUT", "CAP", or "PASS"
    /// - action_data: Action-specific tuple (depends on type)
    /// - normalized_score: Visit count normalized to [0.0, 1.0] range
    ///
    /// For PUT actions: action_data = (marble_type, dst_flat, remove_flat)
    /// For CAP actions: action_data = (direction, start_y, start_x)
    /// For PASS actions: action_data = None
    pub fn last_child_statistics(&self) -> Vec<(String, Option<(usize, usize, usize)>, f32)> {
        let child_stats = self.last_child_stats.lock().unwrap();
        let width = self.last_board_width;

        child_stats.iter().map(|(action, score)| {
            match action {
                Action::Placement {
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
                    ("PUT".to_string(), Some((*marble_type, dst_flat, remove_flat)), *score)
                }
                Action::Capture {
                    start_y,
                    start_x,
                    direction,
                } => {
                    ("CAP".to_string(), Some((*direction, *start_y, *start_x)), *score)
                }
                Action::Pass => {
                    ("PASS".to_string(), None, *score)
                }
            }
        }).collect()
    }
}

// ============================================================================
// INTERNAL IMPLEMENTATION
// ============================================================================

thread_local! {
    static THREAD_RNG: RefCell<Option<(StdRng, u64)>> = RefCell::new(None);
}

impl MCTSSearch {
    /// Thread-safe RNG access using thread-local storage with generation tracking
    ///
    /// This function uses thread-local storage to eliminate lock contention:
    /// - Each thread caches (RNG, generation) tuple
    /// - On each call, checks if cached generation matches current global generation
    /// - If stale (generation mismatch), reinitializes RNG with current seed
    /// - If a seed was set, each thread derives a unique seed = base_seed + thread_num
    /// - Otherwise, uses system RNG (non-deterministic)
    ///
    /// The generic `F` closure allows any operation on `RngCore` trait objects.
    fn with_rng<T, F>(&self, f: F) -> T
    where
        F: FnOnce(&mut dyn RngCore) -> T,
    {
        THREAD_RNG.with(|cell| {
            let mut rng_opt = cell.borrow_mut();

            // Get current generation (atomic read, very fast)
            let current_generation = self.seed_generation.load(Ordering::SeqCst);

            // Check if we need to (re)initialize the RNG
            let needs_init = match *rng_opt {
                None => true,  // No cached RNG
                Some((_, cached_generation)) => cached_generation != current_generation,  // Generation mismatch
            };

            if needs_init {
                // Lock to get current seed (only on init or generation change)
                let base_seed = *self.seed.lock().unwrap();
                let thread_rng = if let Some(seed_value) = base_seed {
                    let thread_num = rayon::current_thread_index().unwrap_or(0) as u64;
                    let thread_seed = seed_value.wrapping_add(thread_num);
                    StdRng::seed_from_u64(thread_seed)
                } else {
                    StdRng::from_os_rng()
                };

                // Cache (RNG, generation) tuple
                *rng_opt = Some((thread_rng, current_generation));
            }

            // Use the thread-local RNG (NO LOCK needed here!)
            if let Some((ref mut rng, _)) = *rng_opt {
                f(rng)
            } else {
                // Fallback to system RNG (shouldn't happen, but handle defensively)
                let mut system_rng = rand::rng();
                f(&mut system_rng)
            }
        })
    }

    fn capture_root_stats(&mut self, root: &MCTSNode) {
        if let Ok(children) = root.children.read() {
            self.last_root_children = children.len();
            self.last_board_width = root.config.width;

            // Capture per-child statistics and normalize visit counts
            let max_visits = children.iter()
                .map(|(_, child)| child.get_visits())
                .max()
                .unwrap_or(1) as f32;

            let mut child_stats = Vec::with_capacity(children.len());
            for (action, child) in children.iter() {
                let normalized_score = if max_visits > 0.0 {
                    child.get_visits() as f32 / max_visits
                } else {
                    0.0
                };
                child_stats.push((action.clone(), normalized_score));
            }

            *self.last_child_stats.lock().unwrap() = child_stats;
        } else {
            self.last_root_children = 0;
            self.last_child_stats.lock().unwrap().clear();
        }

        self.last_root_visits = root.get_visits();
        self.last_root_value = root.get_value();
    }

    /// Run a single MCTS iteration
    fn run_iteration(
        &self,
        root: Arc<MCTSNode>,
        options: &SearchOptions,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        start_time: Instant,
    ) -> bool {
        if let Some(limit) = time_limit {
            if start_time.elapsed().as_secs_f32() >= limit {
                return false;
            }
        }

        // Selection (with optional transposition lookups)
        let table_ref = options.table_ref();
        let node = self.select(Arc::clone(&root), table_ref, options.use_lookups());

        // Simulation (returns value and actions taken for RAVE/AMAF)
        let (value, simulation_actions) = self.simulate(&node, max_depth, time_limit, start_time);

        // Backpropagation (updates node values and RAVE statistics)
        self.backpropagate(&node, value, &simulation_actions, table_ref);

        true
    }

    /// Selection phase: traverse tree using UCB1 until reaching unexpanded/terminal node
    ///
    /// **UCB1 Formula**: `Q(child) + c * sqrt(ln(N(parent)) / N(child))`
    /// - Q(child): Average value of child node
    /// - c: Exploration constant (typically √2)
    /// - N(parent): Visit count of parent
    /// - N(child): Visit count of child
    ///
    /// **FPU (First Play Urgency)**: Unvisited children get score = parent_value - fpu_reduction
    /// - Encourages exploring unvisited nodes early
    /// - Lower FPU reduction = more optimistic about unvisited nodes
    ///
    /// **Progressive Widening**: Limits child expansion based on parent visits
    /// - Controls branching factor in high-branching games
    /// - Formula: max_children = widening_constant * sqrt(parent_visits)
    fn select(
        &self,
        mut node: Arc<MCTSNode>,
        table: Option<&Arc<TranspositionTable>>,
        use_lookups: bool,
    ) -> Arc<MCTSNode> {
        loop {
            // Apply virtual loss to make this path less attractive to other threads
            // This prevents multiple threads from selecting the same path simultaneously
            node.add_virtual_loss();

            // Check if node is fully expanded (respects progressive widening if enabled)
            if !node.is_fully_expanded(self.widening_constant) {
                return self.expand(Arc::clone(&node), table, use_lookups);
            }

            // Check if terminal (game over)
            if self.is_terminal(&node) {
                return node;
            }

            // Select best child using UCB1 (with optional FPU for unvisited nodes)
            let parent_visits = node.get_visits();
            let parent_value = node.get_value();

            // Snapshot children and immediately release lock
            // This allows parallel UCB computation across threads
            let children_snapshot: Vec<(Action, Arc<MCTSNode>)> = {
                let children = node.children.read().unwrap();
                children.iter()
                    .map(|(action, child)| (action.clone(), Arc::clone(child)))
                    .collect()
            }; // Lock released here - held for microseconds instead of milliseconds!

            // Find child with maximum score (RAVE-UCB if enabled, else UCB1)
            // This computation happens WITHOUT holding the lock
            let best_child = children_snapshot.iter().max_by(|(_, child_a), (_, child_b)| {
                let score_a = child_a.rave_ucb_score(
                    parent_visits,
                    parent_value,
                    self.exploration_constant,
                    self.rave_constant,
                    self.fpu_reduction
                );
                let score_b = child_b.rave_ucb_score(
                    parent_visits,
                    parent_value,
                    self.exploration_constant,
                    self.rave_constant,
                    self.fpu_reduction
                );
                // Handle NaN gracefully (treat equal if either is NaN)
                // This can happen with division by zero or invalid values
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((_, child)) = best_child {
                node = Arc::clone(child);
            } else {
                // No children available (shouldn't happen but handle defensively)
                return node;
            }
        }
    }

    /// Expansion phase: add a new child node
    fn expand(
        &self,
        node: Arc<MCTSNode>,
        table: Option<&Arc<TranspositionTable>>,
        use_lookups: bool,
    ) -> Arc<MCTSNode> {
        let (placement_mask, capture_mask) =
            get_valid_actions(&node.spatial_state.view(), &node.global_state.view(), &node.config);

        // Get untried actions
        let mut untried_actions = Vec::new();

        // Add placement actions
        // Placement mask shape: (3, width², width²+1)
        // - dimension 0: marble type (0-2)
        // - dimension 1: destination position (flattened)
        // - dimension 2: removal position (flattened, or width² for no removal)
        let width = node.config.width;
        let width2 = width * width;

        for marble_type in 0..3 {
            for dst_flat in 0..width2 {
                for remove_flat in 0..=width2 {
                    if placement_mask[[marble_type, dst_flat, remove_flat]] > 0.0 {
                        // Convert flat indices to (y, x)
                        let dst_y = dst_flat / width;
                        let dst_x = dst_flat % width;

                        let (remove_y, remove_x) = if remove_flat < width2 {
                            (Some(remove_flat / width), Some(remove_flat % width))
                        } else {
                            (None, None)
                        };

                        let action = Action::Placement {
                            marble_type,
                            dst_y,
                            dst_x,
                            remove_y,
                            remove_x,
                        };
                        untried_actions.push(action);
                    }
                }
            }
        }

        // Add capture actions
        for dir in 0..6 {
            for y in 0..node.config.width {
                for x in 0..node.config.width {
                    if capture_mask[[dir, y, x]] > 0.0 {
                        let action = Action::Capture {
                            start_y: y,
                            start_x: x,
                            direction: dir,
                        };
                        untried_actions.push(action);
                    }
                }
            }
        }

        // Add PASS if no other actions
        if untried_actions.is_empty() {
            untried_actions.push(Action::Pass);
        }

        // Filter out already tried actions
        let tried_actions: Vec<_> = {
            let children = node.children.read().unwrap();
            children.iter().map(|(action, _)| action.clone()).collect()
        };

        untried_actions.retain(|action| {
            !tried_actions
                .iter()
                .any(|tried| actions_equal(action, tried))
        });

        if untried_actions.is_empty() {
            return Arc::clone(&node);
        }

        // Select random untried action
        let action_idx = self.with_rng(|rng| rng.random_range(0..untried_actions.len()));
        let action = untried_actions[action_idx].clone();

        // Apply action to create child state
        let mut child_spatial_state = node.spatial_state.clone();
        let mut child_global_state = node.global_state.clone();

        match &action {
            Action::Placement {
                marble_type,
                dst_y,
                dst_x,
                remove_y,
                remove_x,
            } => {
                apply_placement(
                    &mut child_spatial_state.view_mut(),
                    &mut child_global_state.view_mut(),
                    *marble_type,
                    *dst_y,
                    *dst_x,
                    *remove_y,
                    *remove_x,
                    &node.config,
                );
            }
            Action::Capture {
                start_y,
                start_x,
                direction,
            } => {
                apply_capture(
                    &mut child_spatial_state.view_mut(),
                    &mut child_global_state.view_mut(),
                    *start_y,
                    *start_x,
                    *direction,
                    &node.config,
                );
            }
            Action::Pass => {
                // Just switch player
                let cur_player = child_global_state[node.config.cur_player] as usize;
                child_global_state[node.config.cur_player] = if cur_player == node.config.player_1 {
                    node.config.player_2 as f32
                } else {
                    node.config.player_1 as f32
                };
            }
        }

        // Create child node with parent pointer
        let shared_entry = if use_lookups {
            table.map(|t| {
                t.get_or_insert(
                    &child_spatial_state.view(),
                    &child_global_state.view(),
                    node.config.as_ref(),
                )
            })
        } else {
            None
        };

        let child = Arc::new(MCTSNode::new_child(
            child_spatial_state,
            child_global_state,
            Arc::clone(&node.config),
            &node,
            shared_entry,
        ));

        // Add child to parent (thread-safe)
        node.add_child(action, Arc::clone(&child));

        // Add virtual loss to the newly expanded child
        // This ensures it's included in the backpropagation path correctly
        child.add_virtual_loss();

        child
    }

    /// Simulation phase: play out random game to terminal state
    ///
    /// Returns (value, actions) where:
    /// - value: result from node's current player's perspective: +1 (win), -1 (loss), 0 (draw)
    /// - actions: sequence of actions taken during simulation (for RAVE/AMAF updates)
    fn simulate(
        &self,
        node: &MCTSNode,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        start_time: Instant,
    ) -> (f32, Vec<Action>) {
        let leaf_player = node.global_state[node.config.cur_player] as usize;
        let mut simulation_actions = Vec::new();

        if self.is_terminal(node) {
            let value = self.evaluate_terminal(&node.spatial_state, &node.global_state, &node.config, leaf_player);
            return (value, simulation_actions);
        }

        let mut sim_spatial_state = node.spatial_state.clone();
        let mut sim_global_state = node.global_state.clone();

        let mut consecutive_passes = 0usize;
        let depth_limit = max_depth.unwrap_or(usize::MAX);
        for depth in 0..depth_limit {
            if let Some(limit) = time_limit {
                if start_time.elapsed().as_secs_f32() >= limit {
                    let value = self.evaluate_heuristic(
                        &sim_spatial_state,
                        &sim_global_state,
                        &node.config,
                        leaf_player,
                    );
                    return (value, simulation_actions);
                }
            }

            if self.is_terminal_state(&sim_spatial_state, &sim_global_state, &node.config) {
                let value = self.evaluate_terminal(
                    &sim_spatial_state,
                    &sim_global_state,
                    &node.config,
                    leaf_player,
                );
                return (value, simulation_actions);
            }

            if consecutive_passes >= 2 {
                return (0.0, simulation_actions);
            }

            let (placement_mask, capture_mask) =
                get_valid_actions(&sim_spatial_state.view(), &sim_global_state.view(), &node.config);

            let mut captures = Vec::new();
            for dir in 0..6 {
                for y in 0..node.config.width {
                    for x in 0..node.config.width {
                        if capture_mask[[dir, y, x]] > 0.0 {
                            captures.push((dir, y, x));
                        }
                    }
                }
            }

            if !captures.is_empty() {
                let idx = self.with_rng(|rng| rng.random_range(0..captures.len()));
                let (direction, start_y, start_x) = captures[idx];

                // Track action for RAVE
                simulation_actions.push(Action::Capture { start_y, start_x, direction });

                apply_capture(
                    &mut sim_spatial_state.view_mut(),
                    &mut sim_global_state.view_mut(),
                    start_y,
                    start_x,
                    direction,
                    &node.config,
                );
                consecutive_passes = 0;
            } else {
                let width = node.config.width;
                let width2 = width * width;
                let mut placements = Vec::new();

                for marble_type in 0..3 {
                    for dst_flat in 0..width2 {
                        for remove_flat in 0..=width2 {
                            if placement_mask[[marble_type, dst_flat, remove_flat]] > 0.0 {
                                let dst_y = dst_flat / width;
                                let dst_x = dst_flat % width;
                                let (remove_y, remove_x) = if remove_flat < width2 {
                                    (Some(remove_flat / width), Some(remove_flat % width))
                                } else {
                                    (None, None)
                                };
                                placements.push((marble_type, dst_y, dst_x, remove_y, remove_x));
                            }
                        }
                    }
                }

                if !placements.is_empty() {
                    let idx = self.with_rng(|rng| rng.random_range(0..placements.len()));
                    let (marble_type, dst_y, dst_x, remove_y, remove_x) = placements[idx];

                    // Track action for RAVE
                    simulation_actions.push(Action::Placement {
                        marble_type,
                        dst_y,
                        dst_x,
                        remove_y,
                        remove_x,
                    });

                    apply_placement(
                        &mut sim_spatial_state.view_mut(),
                        &mut sim_global_state.view_mut(),
                        marble_type,
                        dst_y,
                        dst_x,
                        remove_y,
                        remove_x,
                        &node.config,
                    );
                    consecutive_passes = 0;
                } else {
                    consecutive_passes += 1;
                    let cur_player = sim_global_state[node.config.cur_player] as usize;
                    sim_global_state[node.config.cur_player] = if cur_player == node.config.player_1 {
                        node.config.player_2 as f32
                    } else {
                        node.config.player_1 as f32
                    };
                }
            }

            if depth + 1 >= depth_limit {
                let value = self.evaluate_heuristic(
                    &sim_spatial_state,
                    &sim_global_state,
                    &node.config,
                    leaf_player,
                );
                return (value, simulation_actions);
            }
        }

        let value = self.evaluate_heuristic(&sim_spatial_state, &sim_global_state, &node.config, leaf_player);
        (value, simulation_actions)
    }

    /// Backpropagation phase: update statistics up the tree
    ///
    /// Values are flipped at each level since players alternate (zero-sum game).
    /// Removes virtual loss before adding real simulation value.
    ///
    /// RAVE/AMAF updates: For each node during backprop, we update RAVE stats
    /// for sibling nodes whose actions appear in the simulation trajectory.
    fn backpropagate(
        &self,
        node: &Arc<MCTSNode>,
        mut value: f32,
        simulation_actions: &[Action],
        table: Option<&Arc<TranspositionTable>>,
    ) {
        let mut current = Some(Arc::clone(node));

        while let Some(current_node) = current {
            // Remove virtual loss first (added during selection)
            current_node.remove_virtual_loss();

            // Then add real simulation value
            current_node.update(value);

            // RAVE/AMAF updates: Update RAVE stats for sibling actions that appeared in simulation
            if self.rave_constant.is_some() {
                if let Some(parent_ref) = current_node.parent.as_ref().and_then(|weak| weak.upgrade()) {
                    // Snapshot siblings and immediately release lock
                    let siblings_snapshot: Vec<(Action, Arc<MCTSNode>)> = {
                        if let Ok(siblings) = parent_ref.children.read() {
                            siblings.iter()
                                .map(|(action, child)| (action.clone(), Arc::clone(child)))
                                .collect()
                        } else {
                            Vec::new()
                        }
                    }; // Lock released here!

                    // Update RAVE stats without holding lock
                    for (sibling_action, sibling_node) in siblings_snapshot.iter() {
                        // Check if this sibling's action appears in the simulation
                        let action_in_simulation = simulation_actions.iter().any(|sim_action| {
                            actions_equal(sibling_action, sim_action)
                        });

                        if action_in_simulation {
                            // Update RAVE stats for this sibling
                            // Note: value is from current player's perspective, which is correct for AMAF
                            sibling_node.update_rave(value);
                        }
                    }
                }
            }

            if let Some(table_ref) = table {
                if !current_node.has_shared_stats() {
                    table_ref.store(
                        &current_node.spatial_state.view(),
                        &current_node.global_state.view(),
                        current_node.config.as_ref(),
                        current_node.get_visits(),
                        current_node.get_value(),
                    );
                }
            }

            value = -value;
            current = current_node.parent.as_ref().and_then(|weak| weak.upgrade());
        }
    }

    /// Check if node is terminal
    fn is_terminal(&self, node: &MCTSNode) -> bool {
        self.is_terminal_state(&node.spatial_state, &node.global_state, &node.config)
    }

    /// Check if state is terminal (standalone version)
    fn is_terminal_state(
        &self,
        spatial_state: &Array3<f32>,
        global_state: &Array1<f32>,
        config: &BoardConfig,
    ) -> bool {
        // Delegate to game.rs function (single source of truth)
        is_game_over(&spatial_state.view(), &global_state.view(), config)
    }

    /// Evaluate terminal state from root player's perspective
    ///
    /// Returns +1 if root_player won, -1 if lost, 0 if draw, -2 if both lose
    fn evaluate_terminal(
        &self,
        spatial_state: &Array3<f32>,
        global_state: &Array1<f32>,
        config: &BoardConfig,
        root_player: usize,
    ) -> f32 {
        // Delegate to game.rs function (single source of truth)
        // Returns: 1 (P1 wins), -1 (P2 wins), 0 (tie), -2 (both lose)
        let outcome = get_game_outcome(&spatial_state.view(), &global_state.view(), config);

        // Convert from Player 1's perspective to root_player's perspective
        match outcome {
            -2 => -2.0, // Both lose
            0 => 0.0,   // Tie
            1 => {
                // Player 1 wins
                if root_player == config.player_1 {
                    1.0
                } else {
                    -1.0
                }
            }
            -1 => {
                // Player 2 wins
                if root_player == config.player_2 {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => 0.0, // Unknown outcome, treat as draw
        }
    }

    /// Heuristic evaluation for non-terminal states
    ///
    /// Uses weighted marble values: white=1, gray=2, black=3
    fn evaluate_heuristic(
        &self,
        _spatial_state: &Array3<f32>,
        global_state: &Array1<f32>,
        config: &BoardConfig,
        root_player: usize,
    ) -> f32 {
        // Weight by marble value
        let weights = [1.0, 2.0, 3.0]; // white, gray, black

        let p0_score: f32 = (0..3)
            .map(|i| global_state[config.p1_cap_w + i] * weights[i])
            .sum();
        let p1_score: f32 = (0..3)
            .map(|i| global_state[config.p2_cap_w + i] * weights[i])
            .sum();

        // Calculate advantage from root player's perspective
        let advantage = if root_player == config.player_1 {
            p0_score - p1_score
        } else {
            p1_score - p0_score
        };

        // Normalize to [-1, 1] range
        (advantage / 10.0).tanh()
    }

    /// Select best action from root
    fn select_best_action(
        &self,
        root: &MCTSNode,
    ) -> PyResult<(String, Option<(usize, usize, usize)>)> {
        let children = root.children.read().unwrap();

        if children.is_empty() {
            return Ok(("PASS".to_string(), None));
        }

        // Select child with most visits
        let best = children.iter().max_by_key(|(_, child)| child.get_visits());

        if let Some((action, _)) = best {
            let width = root.config.width;
            match action {
                Action::Placement {
                    marble_type,
                    dst_y,
                    dst_x,
                    remove_y,
                    remove_x,
                } => {
                    // Convert (y, x) coordinates to flattened indices
                    let dst_flat = dst_y * width + dst_x;
                    let remove_flat = match (remove_y, remove_x) {
                        (Some(ry), Some(rx)) => ry * width + rx,
                        _ => width * width, // No removal position (use width² as sentinel)
                    };
                    Ok((
                        "PUT".to_string(),
                        Some((*marble_type, dst_flat, remove_flat)),
                    ))
                }
                Action::Capture {
                    start_y,
                    start_x,
                    direction,
                } => Ok(("CAP".to_string(), Some((*direction, *start_y, *start_x)))),
                Action::Pass => Ok(("PASS".to_string(), None)),
            }
        } else {
            Ok(("PASS".to_string(), None))
        }
    }

    /// Fire SearchStarted event callback
    fn fire_search_started(
        &self,
        py: Python,
        callback: &PyObject,
        root: &MCTSNode,
        total_iterations: usize,
    ) -> PyResult<()> {
        // Get valid actions for the root state
        let (placement_mask, _capture_mask) =
            get_valid_actions(&root.spatial_state.view(), &root.global_state.view(), &root.config);

        // Extract unique placement and removal positions
        let width = root.config.width;
        let width2 = width * width;
        let mut placement_set = std::collections::HashSet::new();
        let mut removal_set = std::collections::HashSet::new();

        for marble_type in 0..3 {
            for dst_flat in 0..width2 {
                // Check if any removal option is valid for this destination
                let has_valid_placement = (0..=width2)
                    .any(|remove_flat| placement_mask[[marble_type, dst_flat, remove_flat]] > 0.0);

                if has_valid_placement {
                    placement_set.insert(dst_flat);
                }

                // Check for valid removals (when not "no removal")
                for remove_flat in 0..width2 {
                    if placement_mask[[marble_type, dst_flat, remove_flat]] > 0.0 {
                        removal_set.insert(remove_flat);
                    }
                }
            }
        }

        // Convert to sorted vectors
        let mut placement_positions: Vec<usize> = placement_set.into_iter().collect();
        let mut removal_positions: Vec<usize> = removal_set.into_iter().collect();
        placement_positions.sort_unstable();
        removal_positions.sort_unstable();

        // Build event dict
        let event = PyDict::new(py);
        event.set_item("event", "SearchStarted")?;
        event.set_item("total_iterations", total_iterations)?;
        event.set_item("placement_positions", placement_positions)?;  // Flat indices
        event.set_item("removal_positions", removal_positions)?;      // Flat indices
        event.set_item("board_width", width)?;

        // Call Python callback with GIL
        callback.call1(py, (event,))?;
        Ok(())
    }

    /// Fire SearchProgress event callback
    fn fire_search_progress(
        &self,
        py: Python,
        callback: &PyObject,
        root: &MCTSNode,
        iteration: usize,
    ) -> PyResult<()> {
        // Capture current statistics
        let children = root.children.read().unwrap();

        // Calculate normalized visit counts
        let max_visits = children.iter()
            .map(|(_, child)| child.get_visits())
            .max()
            .unwrap_or(1) as f32;

        let width = root.config.width;

        // Build list of (action_type, action_data, score) tuples (copied data - GIL safe)
        // Same format as last_child_statistics() for consistency with final scores
        let action_stats: Vec<(String, Option<(usize, usize, usize)>, f32)> = children.iter()
            .map(|(action, child)| {
                let score = if max_visits > 0.0 {
                    child.get_visits() as f32 / max_visits
                } else {
                    0.0
                };

                // Convert action to Python tuple format
                let (action_type, action_data) = match action {
                    Action::Placement {
                        marble_type,
                        dst_y,
                        dst_x,
                        remove_y,
                        remove_x,
                    } => {
                        let dst_flat = dst_y * width + dst_x;
                        let remove_flat = match (remove_y, remove_x) {
                            (Some(ry), Some(rx)) => ry * width + rx,
                            _ => width * width, // No removal (sentinel value)
                        };
                        ("PUT".to_string(), Some((*marble_type, dst_flat, remove_flat)))
                    }
                    Action::Capture {
                        start_y,
                        start_x,
                        direction,
                    } => ("CAP".to_string(), Some((*direction, *start_y, *start_x))),
                    Action::Pass => ("PASS".to_string(), None),
                };

                (action_type, action_data, score)
            })
            .collect();

        drop(children); // Release lock before Python call

        // Build event dict
        let event = PyDict::new(py);
        event.set_item("event", "SearchProgress")?;
        event.set_item("iteration", iteration)?;
        event.set_item("action_stats", action_stats)?;

        // Call Python callback with GIL
        callback.call1(py, (event,))?;
        Ok(())
    }

    /// Fire SearchEnded event callback
    fn fire_search_ended(
        &self,
        py: Python,
        callback: &PyObject,
        _root: &MCTSNode,
        total_iterations: usize,
        elapsed: Duration,
    ) -> PyResult<()> {
        // Build event dict
        let event = PyDict::new(py);
        event.set_item("event", "SearchEnded")?;
        event.set_item("total_iterations", total_iterations)?;
        event.set_item("total_time_ms", elapsed.as_millis() as u64)?;

        // Call Python callback with GIL
        callback.call1(py, (event,))?;
        Ok(())
    }
}

/// Helper function to compare actions
fn actions_equal(a: &Action, b: &Action) -> bool {
    match (a, b) {
        (Action::Pass, Action::Pass) => true,
        (
            Action::Placement {
                marble_type: mt1,
                dst_y: dy1,
                dst_x: dx1,
                remove_y: ry1,
                remove_x: rx1,
            },
            Action::Placement {
                marble_type: mt2,
                dst_y: dy2,
                dst_x: dx2,
                remove_y: ry2,
                remove_x: rx2,
            },
        ) => mt1 == mt2 && dy1 == dy2 && dx1 == dx2 && ry1 == ry2 && rx1 == rx2,
        (
            Action::Capture {
                start_y: sy1,
                start_x: sx1,
                direction: d1,
            },
            Action::Capture {
                start_y: sy2,
                start_x: sx2,
                direction: d2,
            },
        ) => sy1 == sy2 && sx1 == sx2 && d1 == d2,
        _ => false,
    }
}

/// Per-search options resolved from Python arguments and struct configuration
struct SearchOptions {
    table: Option<Arc<TranspositionTable>>,
    use_lookups: bool,
}

impl SearchOptions {
    fn new(
        search: &mut MCTSSearch,
        use_table_override: Option<bool>,
        use_lookups_override: Option<bool>,
        clear_table: bool,
    ) -> Self {
        if let Some(flag) = use_table_override {
            search.set_transposition_table_enabled(flag);
        } else if search.use_transposition_table && search.transposition_table.is_none() {
            search.transposition_table = Some(Arc::new(TranspositionTable::new()));
        }

        if let Some(flag) = use_lookups_override {
            search.use_transposition_lookups = flag;
        }

        if clear_table {
            if let Some(table) = &search.transposition_table {
                table.clear();
            }
        }

        Self {
            table: search.transposition_table.as_ref().map(Arc::clone),
            use_lookups: search.use_transposition_lookups,
        }
    }

    fn table_ref(&self) -> Option<&Arc<TranspositionTable>> {
        self.table.as_ref()
    }

    fn use_lookups(&self) -> bool {
        self.use_lookups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardConfig;

    // Note: Most MCTS testing is done via Python integration tests
    // due to the PyO3 boundary (PyReadonlyArray types can only be created from Python)

    #[test]
    fn test_actions_equal() {
        let action1 = Action::Pass;
        let action2 = Action::Pass;
        assert!(actions_equal(&action1, &action2));

        let placement1 = Action::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 4,
            remove_y: Some(1),
            remove_x: Some(2),
        };
        let placement2 = Action::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 4,
            remove_y: Some(1),
            remove_x: Some(2),
        };
        assert!(actions_equal(&placement1, &placement2));

        let capture1 = Action::Capture {
            start_y: 2,
            start_x: 3,
            direction: 1,
        };
        let capture2 = Action::Capture {
            start_y: 2,
            start_x: 3,
            direction: 1,
        };
        assert!(actions_equal(&capture1, &capture2));

        // Different actions should not be equal
        assert!(!actions_equal(&action1, &placement1));
        assert!(!actions_equal(&placement1, &capture1));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_virtual_loss_on_expanded_node() {
        // This test verifies that newly expanded nodes get virtual loss
        // added during expand(), so backpropagation can correctly remove it.

        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());

        // Create a simple initial state with rings
        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Add some rings
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        // Set supply
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Create MCTS search instance and root node
        let mcts = MCTSSearch::new(None, None, None, None, None, None);
        let root = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&config), None));

        // Simulate the select→expand→backprop flow:
        // 1. Add virtual loss to root (what select() does)
        root.add_virtual_loss();

        // 2. Verify root is not fully expanded
        assert!(!root.is_fully_expanded(None));

        // 3. Call expand() - this should add virtual loss to the child
        let child = mcts.expand(Arc::clone(&root), None, false);

        // 4. Verify child has virtual loss (visits should be VIRTUAL_LOSS)
        #[cfg(debug_assertions)]
        assert_eq!(child.virtual_loss_count.load(std::sync::atomic::Ordering::Relaxed), 1);

        // 5. Now simulate backpropagation - should work correctly
        child.remove_virtual_loss();
        child.update(0.5);

        root.remove_virtual_loss();
        root.update(-0.5);

        // 6. Verify final state is correct (virtual losses removed, real values added)
        assert_eq!(child.get_visits(), 1);
        assert!((child.get_value() - 0.5).abs() < 1e-3);
        assert_eq!(root.get_visits(), 1);
        assert!((root.get_value() + 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_seed_generation_increments() {
        // Test that seed generation increments each time set_seed() is called
        let mut mcts = MCTSSearch::new(None, None, None, None, None, None);

        // Initial generation should be 0
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 0);

        // Setting seed should increment generation
        mcts.set_seed(Some(42));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 1);

        // Setting seed again should increment again
        mcts.set_seed(Some(123));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 2);

        // Unsetting seed (None) should also increment
        mcts.set_seed(None);
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 3);

        // Setting same seed again should still increment (invalidate caches)
        mcts.set_seed(Some(42));
        assert_eq!(mcts.seed_generation.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_thread_local_cache_invalidation() {
        // Test that thread-local RNG cache is properly invalidated when seed changes
        // This test verifies the generation tracking mechanism works correctly

        let mcts = MCTSSearch::new(None, None, None, None, None, None);

        // Set initial seed
        let mcts_mut = std::cell::RefCell::new(mcts);
        mcts_mut.borrow_mut().set_seed(Some(12345));
        let gen1 = mcts_mut.borrow().seed_generation.load(Ordering::SeqCst);

        // Use the RNG (this will cache it with current generation)
        let val1 = mcts_mut.borrow().with_rng(|rng| rng.random_range(0..100));

        // Change seed (should increment generation)
        mcts_mut.borrow_mut().set_seed(Some(67890));
        let gen2 = mcts_mut.borrow().seed_generation.load(Ordering::SeqCst);

        // Generation should have incremented
        assert_eq!(gen2, gen1 + 1);

        // Using RNG again should work with new seed
        // (we can't easily verify it uses the new seed in a unit test without
        // full parallel infrastructure, but we can verify it doesn't panic)
        let val2 = mcts_mut.borrow().with_rng(|rng| rng.random_range(0..100));

        // Both values should be valid (in range)
        assert!(val1 < 100);
        assert!(val2 < 100);
    }

    #[test]
    fn test_transposition_table_not_polluted_during_search() {
        // Verify that running MCTS doesn't pollute the transposition table
        // with empty entries for every legal action checked
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        let config = Arc::new(BoardConfig::standard(37, 1).unwrap());
        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Setup initial board state
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }

        // Set supply marbles (needed for valid moves)
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        // Create MCTS with transposition table enabled
        let mut mcts = MCTSSearch::new(
            Some(1.41),  // exploration_constant
            None,        // widening_constant
            None,        // fpu_reduction
            None,        // rave_constant
            Some(true),  // use_transposition_table
            Some(true),  // use_transposition_lookups
        );

        // Initialize transposition table
        mcts.transposition_table = Some(Arc::new(TranspositionTable::new()));

        // Create root node with transposition lookup
        let shared_entry = mcts.transposition_table.as_ref().unwrap()
            .get_or_insert(&spatial_state.view(), &global_state.view(), config.as_ref());

        let root = Arc::new(MCTSNode::new(
            spatial_state.clone(),
            global_state.clone(),
            Arc::clone(&config),
            Some(shared_entry),
        ));

        // Create search options
        let search_options = SearchOptions {
            table: mcts.transposition_table.as_ref().map(Arc::clone),
            use_lookups: true,
        };

        // Run 100 MCTS iterations directly
        let start = Instant::now();
        for _ in 0..100 {
            mcts.run_iteration(Arc::clone(&root), &search_options, None, None, start);
        }

        // Check transposition table size
        let table_size = if let Some(table) = &mcts.transposition_table {
            table.len()
        } else {
            panic!("Transposition table should exist after search");
        };

        // With 100 iterations from starting position, we expect:
        // - ~10-100 unique states explored (grows with tree depth)
        // - NOT hundreds of thousands of entries (iterations × legal_actions)
        //
        // 37-ring board has ~1944 legal actions from start
        // If polluted: 100 iterations × 1944 actions = 194,400 entries
        // Actual with transposition hits: typically < 200 entries
        // (transposition table finds many duplicate states via different move orders)
        assert!(
            table_size < 200,
            "Transposition table too large ({} entries). Expected < 200. \
             Table may be polluted with empty entries from action enumeration.",
            table_size
        );

        // Also verify table has at least some entries (search actually happened)
        assert!(
            table_size > 0,
            "Transposition table should have entries after search"
        );

        println!("Transposition table size after 100 iterations: {} entries", table_size);
    }

}
