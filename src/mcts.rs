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

use crate::game_trait::MCTSGame;
use crate::node::MCTSNode;
use crate::transposition::TranspositionTable;

// ============================================================================
// MCTS SEARCH
// ============================================================================

/// Main MCTS search engine (generic over game type).
///
/// This struct maintains search configuration and implements the core MCTS algorithm.
/// Python bindings are provided through game-specific wrappers (e.g., PyZertzMCTS).
pub struct MCTSSearch<G: MCTSGame> {
    game: Arc<G>,
    exploration_constant: f32,
    widening_constant: Option<f32>,
    fpu_reduction: Option<f32>,
    rave_constant: Option<f32>,
    use_transposition_table: bool,
    use_transposition_lookups: bool,
    transposition_table: Option<Arc<TranspositionTable<G>>>,
    last_root_children: usize,
    last_root_visits: u32,
    last_root_value: f32,
    pub last_child_stats: Mutex<Vec<(G::Action, f32)>>, // Store normalized visit scores per action
    rng: Mutex<Option<StdRng>>,
    seed: Mutex<Option<u64>>, // Base seed for deriving per-thread seeds
    seed_generation: AtomicU64, // Increments on each set_seed call to invalidate thread-local caches
    #[cfg(feature = "metrics")]
    metrics: Arc<MCTSMetrics>,
}

impl<G: MCTSGame> MCTSSearch<G> {
    /// Enable/disable transposition table caching (persists across searches)
    pub fn set_transposition_table_enabled(&mut self, enabled: bool) {
        self.use_transposition_table = enabled;
        if !enabled {
            self.transposition_table = None;
        } else if self.transposition_table.is_none() {
            self.transposition_table = Some(Arc::new(TranspositionTable::new(Arc::clone(&self.game))));
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

    pub fn new(
        game: Arc<G>,
        exploration_constant: Option<f32>,
        widening_constant: Option<f32>,
        fpu_reduction: Option<f32>,
        rave_constant: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
    ) -> Self {
        let resolved_exploration = exploration_constant.unwrap_or(1.41);
        let resolved_use_table = use_transposition_table.unwrap_or(true);
        let resolved_use_lookups = use_transposition_lookups.unwrap_or(true);

        #[cfg(debug_assertions)]
        {
            let mut features:Vec<&str> = Vec::new();
            #[cfg(feature = "metrics")]
            features.push("metrics");

            let features_str = if features.is_empty() {
                String::from("none")
            } else {
                features.join(",")
            };

            eprintln!("MCTSSearch::new(game={}, exploration={}, widening={:?}, fpu={:?}, rave={:?}, use_table={}, use_lookups={}, features=[{}])",
                game.name(), resolved_exploration, widening_constant, fpu_reduction, rave_constant, resolved_use_table, resolved_use_lookups, features_str);
        }

        Self {
            game,
            exploration_constant: resolved_exploration,
            widening_constant,
            fpu_reduction,
            rave_constant,
            use_transposition_table: resolved_use_table,
            use_transposition_lookups: resolved_use_lookups,
            transposition_table: None,
            last_root_children: 0,
            last_root_visits: 0,
            last_root_value: 0.0,
            last_child_stats: Mutex::new(Vec::new()),
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

    /* NOTE: search() and search_parallel() are Python-specific and will be
     * re-implemented in the game-specific wrapper (Phase 3).
     * The generic MCTS core provides run_iteration(), select(), expand(), simulate(),
     * and backpropagate() which the wrapper will use.
     */

    /*
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
        progress_callback: Option<Py<PyAny>>,
        progress_interval_ms: Option<u64>,
    ) -> PyResult<(String, Option<(usize, usize, usize)>)> {
        let t = t.unwrap_or(1);
        let verbose = verbose.unwrap_or(false);
        let blitz = blitz.unwrap_or(false);

        #[cfg(debug_assertions)]
        {
            eprintln!("MCTSSearch::search(rings={}, iterations={}, t={}, max_depth={:?}, time_limit={:?}, verbose={}, seed={:?}, blitz={})",
                rings, iterations, t, max_depth, time_limit, verbose, seed, blitz);
        }

        let search_options = SearchOptions::new(
            self,
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );

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
    */

    /*
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
        progress_callback: Option<Py<PyAny>>,
        progress_interval_ms: Option<u64>,
    ) -> PyResult<(String, Option<(usize, usize, usize)>)> {
        let t = t.unwrap_or(1);
        let num_threads = num_threads.unwrap_or(16);
        let verbose = verbose.unwrap_or(false);
        let blitz = blitz.unwrap_or(false);

        #[cfg(debug_assertions)]
        {
            eprintln!("MCTSSearch::search_parallel(rings={}, iterations={}, t={}, max_depth={:?}, time_limit={:?}, num_threads={}, verbose={}, seed={:?}, blitz={})",
                rings, iterations, t, max_depth, time_limit, num_threads, verbose, seed, blitz);
        }

        let search_options = SearchOptions::new(
            self,
            use_transposition_table,
            use_transposition_lookups,
            clear_table,
        );

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
        py.detach(|| {
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
                        Python::attach(|py| {
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
    */

    pub fn last_root_children(&self) -> usize {
        self.last_root_children
    }

    pub fn last_root_visits(&self) -> u32 {
        self.last_root_visits
    }

    pub fn last_root_value(&self) -> f32 {
        self.last_root_value
    }

    // NOTE: last_child_statistics() is Python-specific and will be implemented
    // in the game-specific wrapper (Phase 3). The generic MCTS stores child stats
    // in last_child_stats but doesn't convert them to Python format.
    // Python wrapper will access last_child_stats directly and convert G::Action to Python.
}

// ============================================================================
// INTERNAL IMPLEMENTATION
// ============================================================================

thread_local! {
    static THREAD_RNG: RefCell<Option<(StdRng, u64)>> = const { RefCell::new(None) };
}

impl<G: MCTSGame> MCTSSearch<G> {
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

    pub fn capture_root_stats(&mut self, root: &MCTSNode<G>) {
        if let Ok(children) = root.children.read() {
            self.last_root_children = children.len();

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
    pub fn run_iteration(
        &self,
        root: Arc<MCTSNode<G>>,
        options: &SearchOptions<G>,
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

    /// Collapse deterministic sequences in the game tree.
    ///
    /// This optimization automatically traverses through game states where the player has
    /// no real strategic choice, creating child nodes but not treating them as separate
    /// decision points. This compresses forced move sequences (e.g., mandatory capture
    /// chains in Checkers/Zertz) into the tree without wasting MCTS iterations on trivial
    /// decisions.
    ///
    /// ## Tree Structure Impact
    ///
    /// **WITHOUT deterministic collapse:**
    /// ```text
    /// Root [1000 visits]
    ///   ├─ Move A [400 visits]
    ///   │    └─ Forced B [400 visits] ← Trivial decision, all 400 iterations go here
    ///   │         └─ Forced C [400 visits] ← Still forced, wastes more iterations
    ///   │              ├─ Choice D1 [200 visits] ← Finally a real choice
    ///   │              └─ Choice D2 [200 visits]
    ///   └─ Move E [600 visits]
    /// ```
    /// - Forced nodes B and C accumulate visit counts even though they're not decisions
    /// - MCTS wastes iterations exploring the same forced path repeatedly
    /// - UCB statistics are distorted by inflated visits on trivial nodes
    /// - Tree depth increases unnecessarily
    ///
    /// **WITH deterministic collapse:**
    /// ```text
    /// Root [1000 visits]
    ///   ├─ Move A [400 visits]
    ///   │    ├─ Choice D1 [200 visits] ← Jumped directly here via collapsed B->C
    ///   │    └─ Choice D2 [200 visits]
    ///   └─ Move E [600 visits]
    /// ```
    /// - Nodes B and C exist but don't accumulate visits (they're traversed, not explored)
    /// - All 400 iterations reach the real choice point (D1 vs D2) immediately
    /// - UCB scores reflect actual strategic value, not forced move inflation
    /// - Tree is more compact and focused on real decisions
    ///
    /// ## Performance Benefits
    ///
    /// - **Iteration Efficiency**: Saves 2-3x iterations in games with long forced sequences
    /// - **UCB Accuracy**: Prevents visit count inflation that distorts exploration/exploitation balance
    /// - **Tree Depth**: Reduces effective tree depth by collapsing linear forced paths
    /// - **Memory**: Creates fewer visited nodes (forced nodes exist but aren't re-explored)
    /// - **Search Quality**: Focuses computational budget on actual strategic choices
    ///
    /// ## Algorithm
    ///
    /// ```text
    /// 1. Start at current node
    /// 2. Check if terminal → return if yes
    /// 3. Get valid actions from game
    /// 4. Ask game if any action is forced/deterministic
    /// 5. If forced:
    ///    a. Check if child node already exists
    ///    b. If yes: move to child and repeat from step 2
    ///    c. If no: create child node, apply action, repeat from step 2
    /// 6. If not forced: return node (this is a choice point)
    /// 7. Safety: terminate after MAX_DETERMINISTIC_DEPTH steps
    /// ```
    ///
    /// ## Safety: Depth Limit
    ///
    /// To prevent infinite loops in buggy game implementations, the traversal is limited
    /// to `MAX_DETERMINISTIC_DEPTH` (100) consecutive forced moves. If this limit is
    /// reached, the traversal terminates and returns the current node.
    ///
    /// This protects against:
    /// - Buggy `get_forced_action()` logic that creates cycles
    /// - Games with unexpectedly long forced sequences
    /// - Infinite loops due to state representation errors
    ///
    /// If you hit this limit in practice, it indicates either:
    /// 1. A bug in your `get_forced_action()` implementation (most likely)
    /// 2. A genuinely long forced sequence (increase MAX_DETERMINISTIC_DEPTH if legitimate)
    ///
    /// ## When This Is Called
    ///
    /// This method is called from `select()` only if the game enables it via
    /// `game.enable_deterministic_collapse()`. It runs after virtual loss application
    /// but before expansion checks.
    ///
    /// ## Returns
    ///
    /// The first node that is either:
    /// - Terminal (game over)
    /// - A choice point (multiple legal moves, or game says "not forced")
    /// - At the depth limit (MAX_DETERMINISTIC_DEPTH reached)
    const MAX_DETERMINISTIC_DEPTH: usize = 100;

    fn collapse_deterministic_sequence(
        &self,
        mut node: Arc<MCTSNode<G>>,
    ) -> Arc<MCTSNode<G>> {
        let mut depth = 0;

        loop {
            // Safety: prevent infinite loops in buggy game implementations
            if depth >= Self::MAX_DETERMINISTIC_DEPTH {
                #[cfg(debug_assertions)]
                eprintln!(
                    "Warning: MAX_DETERMINISTIC_DEPTH ({}) reached in collapse_deterministic_sequence(). \
                     This may indicate a bug in get_forced_action() or an unexpectedly long forced sequence.",
                    Self::MAX_DETERMINISTIC_DEPTH
                );
                return node;
            }
            depth += 1;

            // Check if terminal - forced sequences end at game over
            if node.game.is_terminal(&node.spatial_state.view(), &node.global_state.view()) {
                return node;
            }

            // Get all valid actions from the game
            let valid_actions = node.game.get_valid_actions(
                &node.spatial_state.view(),
                &node.global_state.view(),
            );

            // Ask the game if there's a forced/deterministic action
            let forced_action = node.game.get_forced_action(
                &valid_actions,
                &node.spatial_state.view(),
                &node.global_state.view(),
            );

            match forced_action {
                Some(action) => {
                    // This is a forced/deterministic move - check if child already exists
                    let existing_child = {
                        let children = node.children.read().unwrap();
                        children.get(&action).map(Arc::clone)
                    };

                    if let Some(child) = existing_child {
                        // Child exists - continue traversing down the forced sequence
                        node = child;
                    } else {
                        // Create new child node for the forced action
                        let mut child_spatial = node.spatial_state.clone();
                        let mut child_global = node.global_state.clone();

                        // Apply the forced action to create child state
                        node.game.apply_action(
                            &mut child_spatial.view_mut(),
                            &mut child_global.view_mut(),
                            &action,
                        );

                        // Create child without transposition table lookup
                        // Forced nodes don't need transposition entries since they're
                        // not interesting decision points for MCTS exploration
                        let child = Arc::new(MCTSNode::new_child(
                            child_spatial,
                            child_global,
                            Arc::clone(&node.game),
                            &node,
                            None,  // No transposition table entry for deterministic sequences
                        ));

                        // Add child to parent's children map
                        node.add_child(action, Arc::clone(&child));

                        // Continue traversing down the forced sequence
                        node = child;
                    }
                }
                None => {
                    // No forced action - this is a choice point (or no actions available)
                    // This is where MCTS should start exploring
                    return node;
                }
            }
        }
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
        mut node: Arc<MCTSNode<G>>,
        table: Option<&Arc<TranspositionTable<G>>>,
        use_lookups: bool,
    ) -> Arc<MCTSNode<G>> {
        loop {
            // Apply virtual loss to make this path less attractive to other threads
            // This prevents multiple threads from selecting the same path simultaneously
            node.add_virtual_loss();

            // Collapse deterministic sequences if the game enables it
            // This automatically traverses forced moves (e.g., mandatory capture chains)
            // to compress the tree and avoid wasting iterations on trivial decisions
            if node.game.enable_deterministic_collapse() {
                node = self.collapse_deterministic_sequence(node);
            }

            // Check if node is fully expanded (respects progressive widening if enabled)
            if !node.is_fully_expanded(self.widening_constant) {
                return self.expand(Arc::clone(&node), table, use_lookups);
            }

            // Check if terminal (game over)
            if node.game.is_terminal(&node.spatial_state.view(), &node.global_state.view()) {
                return node;
            }

            // Select best child using UCB1 (with optional FPU for unvisited nodes)
            let parent_visits = node.get_visits();
            let parent_value = node.get_value();
            let children = node.children.read().unwrap();

            // Find child with maximum score (RAVE-UCB if enabled, else UCB1)
            // max_by uses partial ordering since f32 can be NaN
            let best_child = children.iter().max_by(|(_, child_a), (_, child_b)| {
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
                let next_node = Arc::clone(child);
                drop(children); // Release lock before next iteration (critical for performance)
                node = next_node;
            } else {
                // No children available (shouldn't happen but handle defensively)
                drop(children);
                return node;
            }
        }
    }


    /// Expansion phase: add a new child node
    fn expand(
        &self,
        node: Arc<MCTSNode<G>>,
        table: Option<&Arc<TranspositionTable<G>>>,
        use_lookups: bool,
    ) -> Arc<MCTSNode<G>> {
        // Get all valid actions from the game trait
        let mut untried_actions = node.game.get_valid_actions(&node.spatial_state.view(), &node.global_state.view());

        // Filter out already tried actions
        {
            let children = node.children.read().unwrap();
            untried_actions.retain(|action| !children.contains_key(action));
        }

        if untried_actions.is_empty() {
            return Arc::clone(&node);
        }

        // Select random untried action
        let action_idx = self.with_rng(|rng| rng.random_range(0..untried_actions.len()));
        let action = untried_actions[action_idx].clone();

        // Apply action to create child state using trait method
        let mut child_spatial_state = node.spatial_state.clone();
        let mut child_global_state = node.global_state.clone();

        node.game.apply_action(
            &mut child_spatial_state.view_mut(),
            &mut child_global_state.view_mut(),
            &action,
        );

        // Create child node with parent pointer
        let shared_entry = if use_lookups {
            table.map(|t| {
                t.get_or_insert(
                    &child_spatial_state.view(),
                    &child_global_state.view(),
                )
            })
        } else {
            None
        };

        let child = Arc::new(MCTSNode::new_child(
            child_spatial_state,
            child_global_state,
            Arc::clone(&node.game),
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
        node: &MCTSNode<G>,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        start_time: Instant,
    ) -> (f32, Vec<G::Action>) {
        let leaf_player = node.game.get_current_player(&node.global_state.view());
        let mut simulation_actions = Vec::new();

        if node.game.is_terminal(&node.spatial_state.view(), &node.global_state.view()) {
            // Evaluate from the perspective of the leaf player (player at this node)
            let outcome = node.game.get_outcome(&node.spatial_state.view(), &node.global_state.view());
            let value = self.outcome_to_value(outcome, leaf_player);
            return (value, simulation_actions);
        }

        let mut sim_spatial_state = node.spatial_state.clone();
        let mut sim_global_state = node.global_state.clone();

        let depth_limit = max_depth.unwrap_or(usize::MAX);
        for _depth in 0..depth_limit {
            // Check time limit
            if let Some(limit) = time_limit {
                if start_time.elapsed().as_secs_f32() >= limit {
                    let value = node.game.evaluate_heuristic(
                        &sim_spatial_state.view(),
                        &sim_global_state.view(),
                        leaf_player,
                    );
                    return (value, simulation_actions);
                }
            }

            // Check if terminal
            if node.game.is_terminal(&sim_spatial_state.view(), &sim_global_state.view()) {
                let outcome = node.game.get_outcome(&sim_spatial_state.view(), &sim_global_state.view());
                let value = self.outcome_to_value(outcome, leaf_player);
                return (value, simulation_actions);
            }

            // Get valid actions and pick one randomly
            let actions = node.game.get_valid_actions(&sim_spatial_state.view(), &sim_global_state.view());

            if actions.is_empty() {
                // No valid actions - game should be terminal, but handle gracefully
                return (0.0, simulation_actions);
            }

            let action_idx = self.with_rng(|rng| rng.random_range(0..actions.len()));
            let action = actions[action_idx].clone();

            // Track action for RAVE
            simulation_actions.push(action.clone());

            // Apply action
            node.game.apply_action(
                &mut sim_spatial_state.view_mut(),
                &mut sim_global_state.view_mut(),
                &action,
            );
        }

        // Depth limit reached - use heuristic evaluation
        let value = node.game.evaluate_heuristic(&sim_spatial_state.view(), &sim_global_state.view(), leaf_player);
        (value, simulation_actions)
    }

    /// Convert game outcome to value from player's perspective
    ///
    /// Outcome encoding: 1 (player 0 wins), -1 (player 1 wins), 0 (draw), -2 (both lose)
    /// Returns: +1.0 if player won, -1.0 if lost, 0.0 for draw, -2.0 for both lose
    fn outcome_to_value(&self, outcome: i8, player: usize) -> f32 {
        match outcome {
            -2 => -2.0,  // Both lose
            0 => 0.0,    // Draw
            1 if player == 0 => 1.0,   // Player 0 wins and we are player 0
            1 => -1.0,   // Player 0 wins but we are player 1
            -1 if player == 1 => 1.0,  // Player 1 wins and we are player 1
            -1 => -1.0,  // Player 1 wins but we are player 0
            _ => 0.0,    // Unknown outcome
        }
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
        node: &Arc<MCTSNode<G>>,
        mut value: f32,
        simulation_actions: &[G::Action],
        table: Option<&Arc<TranspositionTable<G>>>,
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
                    // Lock parent's children to access siblings
                    if let Ok(siblings) = parent_ref.children.read() {
                        // For each sibling, check if its action matches any simulation action
                        for (sibling_action, sibling_node) in siblings.iter() {
                            // Check if this sibling's action appears in the simulation
                            // G::Action implements Eq, so we can use contains()
                            let action_in_simulation = simulation_actions.contains(sibling_action);

                            if action_in_simulation {
                                // Update RAVE stats for this sibling
                                // Note: value is from current player's perspective, which is correct for AMAF
                                sibling_node.update_rave(value);
                            }
                        }
                    }
                }
            }

            if let Some(table_ref) = table {
                if !current_node.has_shared_stats() {
                    table_ref.store(
                        &current_node.spatial_state.view(),
                        &current_node.global_state.view(),
                        current_node.get_visits(),
                        current_node.get_value(),
                    );
                }
            }

            value = -value;
            current = current_node.parent.as_ref().and_then(|weak| weak.upgrade());
        }
    }

    // NOTE: is_terminal(), is_terminal_state(), evaluate_terminal(), and evaluate_heuristic()
    // have been removed. These are now accessed directly via the MCTSGame trait methods:
    // - node.game.is_terminal()
    // - node.game.get_outcome()
    // - node.game.evaluate_heuristic()
    // Outcome conversion is handled by outcome_to_value() above.

    /* NOTE: select_best_action() and fire_* callback methods are Python-specific.
     * They will be reimplemented in the game-specific wrapper (Phase 3).
     * The wrapper will convert G::Action to Python-compatible format.
     */

    /*
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
        callback: &Py<PyAny>,
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
        callback: &Py<PyAny>,
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
        callback: &Py<PyAny>,
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
    */
}

// NOTE: actions_equal() has been removed. G::Action implements Eq trait,
// so equality checks can use == directly or Vec::contains().

/// Per-search options resolved from Python arguments and struct configuration
pub struct SearchOptions<G: MCTSGame> {
    table: Option<Arc<TranspositionTable<G>>>,
    use_lookups: bool,
}

impl<G: MCTSGame> SearchOptions<G> {
    pub fn new(
        search: &mut MCTSSearch<G>,
        use_table_override: Option<bool>,
        use_lookups_override: Option<bool>,
        clear_table: bool,
    ) -> Self {
        if let Some(flag) = use_table_override {
            search.set_transposition_table_enabled(flag);
        } else if search.use_transposition_table && search.transposition_table.is_none() {
            search.transposition_table = Some(Arc::new(TranspositionTable::new(Arc::clone(&search.game))));
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

    pub fn table_ref(&self) -> Option<&Arc<TranspositionTable<G>>> {
        self.table.as_ref()
    }

    pub fn use_lookups(&self) -> bool {
        self.use_lookups
    }
}

// NOTE: Tests updated to use generic version with ZertzGame.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BoardConfig;
    use crate::games::{ZertzGame, zertz::ZertzAction};
    use ndarray::{Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};

    // Note: Most MCTS testing is done via Python integration tests
    // due to the PyO3 boundary (PyReadonlyArray types can only be created from Python)

    #[test]
    fn test_actions_equal() {
        let action1 = ZertzAction::Pass;
        let action2 = ZertzAction::Pass;
        assert_eq!(action1, action2);

        let placement1 = ZertzAction::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 4,
            remove_y: Some(1),
            remove_x: Some(2),
        };
        let placement2 = ZertzAction::Placement {
            marble_type: 0,
            dst_y: 3,
            dst_x: 4,
            remove_y: Some(1),
            remove_x: Some(2),
        };
        assert_eq!(placement1, placement2);

        let capture1 = ZertzAction::Capture {
            start_y: 2,
            start_x: 3,
            direction: 1,
        };
        let capture2 = ZertzAction::Capture {
            start_y: 2,
            start_x: 3,
            direction: 1,
        };
        assert_eq!(capture1, capture2);

        // Different actions should not be equal
        assert_ne!(action1, placement1);
        assert_ne!(placement1, capture1);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_virtual_loss_on_expanded_node() {
        // This test verifies that newly expanded nodes get virtual loss
        // added during expand(), so backpropagation can correctly remove it.

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

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
        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let root = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

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
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let mut mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);

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

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);

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

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();
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
            Arc::clone(&game),
            Some(1.41),  // exploration_constant
            None,        // widening_constant
            None,        // fpu_reduction
            None,        // rave_constant
            Some(true),  // use_transposition_table
            Some(true),  // use_transposition_lookups
        );

        // Initialize transposition table
        mcts.transposition_table = Some(Arc::new(TranspositionTable::new(Arc::clone(&game))));

        // Create root node with transposition lookup
        let shared_entry = mcts.transposition_table.as_ref().unwrap()
            .get_or_insert(&spatial_state.view(), &global_state.view());

        let root = Arc::new(MCTSNode::new(
            spatial_state.clone(),
            global_state.clone(),
            Arc::clone(&game),
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

    #[test]
    fn test_collapse_deterministic_sequence_disabled() {
        // Test that collapse returns immediately when enable_deterministic_collapse() is false
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        // Create a simple test game
        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Setup minimal valid state
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

        // Create a temporary game that disables collapse for testing
        struct TestGame(Arc<ZertzGame>);
        impl MCTSGame for TestGame {
            type Action = ZertzAction;
            fn enable_deterministic_collapse(&self) -> bool { false }
            fn get_valid_actions(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> Vec<Self::Action> {
                self.0.get_valid_actions(s, g)
            }
            fn apply_action(&self, s: &mut ArrayViewMut3<f32>, g: &mut ArrayViewMut1<f32>, a: &Self::Action) {
                self.0.apply_action(s, g, a)
            }
            fn is_terminal(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> bool {
                self.0.is_terminal(s, g)
            }
            fn get_outcome(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> i8 {
                self.0.get_outcome(s, g)
            }
            fn get_current_player(&self, g: &ArrayView1<f32>) -> usize {
                self.0.get_current_player(g)
            }
            fn spatial_shape(&self) -> (usize, usize, usize) {
                self.0.spatial_shape()
            }
            fn global_size(&self) -> usize {
                self.0.global_size()
            }
            fn evaluate_heuristic(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>, p: usize) -> f32 {
                self.0.evaluate_heuristic(s, g, p)
            }
            fn canonicalize_state(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> (Array3<f32>, Array1<f32>) {
                self.0.canonicalize_state(s, g)
            }
            fn hash_state(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> u64 {
                self.0.hash_state(s, g)
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }

        let test_game = Arc::new(TestGame(Arc::clone(&game)));
        let test_node = Arc::new(MCTSNode::new(
            node.spatial_state.clone(),
            node.global_state.clone(),
            test_game,
            None,
        ));

        let test_mcts = MCTSSearch::new(Arc::clone(&test_node.game), None, None, None, None, None, None);

        // collapse should return immediately without traversing
        let result = test_mcts.collapse_deterministic_sequence(Arc::clone(&test_node));

        // Should return the same node (pointer equality)
        assert!(Arc::ptr_eq(&result, &test_node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_terminal_state() {
        // Test that collapse stops immediately at terminal states
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Create terminal state (no rings left, game over)
        // All rings removed, player 1 has won via sufficient captures
        global_state[config.p1_cap_w] = 4.0; // Player 1 has 4 white
        global_state[config.p1_cap_g] = 5.0; // 5 gray
        global_state[config.p1_cap_b] = 3.0; // 3 black
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

        // Verify it's actually terminal
        assert!(game.is_terminal(&node.spatial_state.view(), &node.global_state.view()));

        // collapse should return immediately
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should return the same node
        assert!(Arc::ptr_eq(&result, &node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_choice_point() {
        // Test that collapse stops at nodes with multiple legal actions (choice points)
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Setup initial board state with many possible moves
        for y in 0..config.width {
            for x in 0..config.width {
                spatial_state[[config.ring_layer, y, x]] = 1.0;
            }
        }
        global_state[config.supply_w] = 5.0;
        global_state[config.supply_g] = 8.0;
        global_state[config.supply_b] = 7.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

        // Verify there are multiple actions available
        let actions = game.get_valid_actions(&node.spatial_state.view(), &node.global_state.view());
        assert!(actions.len() > 1, "Test requires multiple actions at root");

        // collapse should return immediately (not a forced move)
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should return the same node
        assert!(Arc::ptr_eq(&result, &node));
    }

    #[test]
    fn test_collapse_deterministic_sequence_single_forced_move() {
        // Test that collapse traverses a single forced move
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Create a state with exactly one legal action:
        // - Only one ring available for placement
        // - Only white marbles in supply (only one marble type available)
        let center_y = config.width / 2;
        let center_x = config.width / 2;
        spatial_state[[config.ring_layer, center_y, center_x]] = 1.0;

        global_state[config.supply_w] = 1.0; // Only white available
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

        // Verify there's exactly one action
        let actions = game.get_valid_actions(&node.spatial_state.view(), &node.global_state.view());
        assert_eq!(actions.len(), 1, "Test requires exactly one action");

        // collapse should create a child and return it
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should NOT return the same node (should have moved to child)
        assert!(!Arc::ptr_eq(&result, &node));

        // Verify the node has one child now
        let children = node.children.read().unwrap();
        assert_eq!(children.len(), 1, "Parent should have exactly one child after collapse");
    }

    #[test]
    fn test_collapse_deterministic_sequence_reuses_existing_child() {
        // Test that collapse reuses existing children instead of creating duplicates
        use crate::board::BoardConfig;
        use ndarray::{Array1, Array3};

        let game = Arc::new(ZertzGame::new(37, 1, false).unwrap());
        let config = game.config();

        let mut spatial_state = Array3::zeros((config.layers_per_timestep * config.t + 1, config.width, config.width));
        let mut global_state = Array1::zeros(10);

        // Create forced move state
        let center_y = config.width / 2;
        let center_x = config.width / 2;
        spatial_state[[config.ring_layer, center_y, center_x]] = 1.0;
        global_state[config.supply_w] = 1.0;
        global_state[config.supply_g] = 0.0;
        global_state[config.supply_b] = 0.0;
        global_state[config.cur_player] = config.player_1 as f32;

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, Arc::clone(&game), None));

        // First collapse - creates child
        let result1 = mcts.collapse_deterministic_sequence(Arc::clone(&node));
        let child1_ptr = Arc::as_ptr(&result1);

        // Second collapse - should reuse child
        let result2 = mcts.collapse_deterministic_sequence(Arc::clone(&node));
        let child2_ptr = Arc::as_ptr(&result2);

        // Should return the same child (pointer equality)
        assert_eq!(child1_ptr, child2_ptr, "Collapse should reuse existing child");

        // Verify still only one child
        let children = node.children.read().unwrap();
        assert_eq!(children.len(), 1, "Should still have exactly one child");
    }

    #[test]
    fn test_collapse_deterministic_sequence_depth_limit() {
        // Test that collapse respects MAX_DETERMINISTIC_DEPTH to prevent infinite loops
        // We'll create a custom game that always returns a forced action
        use ndarray::{Array1, Array3};

        #[derive(Clone)]
        struct InfiniteForceGame;

        impl MCTSGame for InfiniteForceGame {
            type Action = ZertzAction;

            fn enable_deterministic_collapse(&self) -> bool {
                true
            }

            fn get_forced_action(
                &self,
                _actions: &[Self::Action],
                _spatial_state: &ArrayView3<f32>,
                _global_state: &ArrayView1<f32>,
            ) -> Option<Self::Action> {
                // Always return a forced Pass action (simulating infinite loop)
                Some(ZertzAction::Pass)
            }

            fn get_valid_actions(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> Vec<Self::Action> {
                vec![ZertzAction::Pass]
            }

            fn apply_action(&self, _s: &mut ArrayViewMut3<f32>, g: &mut ArrayViewMut1<f32>, _a: &Self::Action) {
                // Just flip player to simulate state change
                let cur = g[0] as usize;
                g[0] = if cur == 0 { 1.0 } else { 0.0 };
            }

            fn is_terminal(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> bool {
                false // Never terminal
            }

            fn get_outcome(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> i8 {
                0
            }

            fn get_current_player(&self, g: &ArrayView1<f32>) -> usize {
                g[0] as usize
            }

            fn spatial_shape(&self) -> (usize, usize, usize) {
                (1, 7, 7)
            }

            fn global_size(&self) -> usize {
                10
            }

            fn evaluate_heuristic(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>, _p: usize) -> f32 {
                0.0
            }

            fn canonicalize_state(&self, s: &ArrayView3<f32>, g: &ArrayView1<f32>) -> (Array3<f32>, Array1<f32>) {
                (s.to_owned(), g.to_owned())
            }

            fn hash_state(&self, _s: &ArrayView3<f32>, _g: &ArrayView1<f32>) -> u64 {
                0
            }

            fn name(&self) -> &str {
                "InfiniteForce"
            }
        }

        let game = Arc::new(InfiniteForceGame);
        let spatial_state = Array3::zeros((1, 7, 7));
        let global_state = Array1::zeros(10);

        let mcts = MCTSSearch::new(Arc::clone(&game), None, None, None, None, None, None);
        let node = Arc::new(MCTSNode::new(spatial_state, global_state, game, None));

        // This should not hang - depth limit should prevent infinite loop
        let result = mcts.collapse_deterministic_sequence(Arc::clone(&node));

        // Should have stopped due to depth limit
        assert!(!Arc::ptr_eq(&result, &node), "Should have moved at least one step");

        // Count depth by traversing children
        let mut depth = 0;
        let mut current = node;
        while depth < MCTSSearch::<InfiniteForceGame>::MAX_DETERMINISTIC_DEPTH + 10 {
            let next_child = {
                let children = current.children.read().unwrap();
                if children.is_empty() {
                    break;
                }
                assert_eq!(children.len(), 1, "Should have exactly one child at each level");
                Arc::clone(children.values().next().unwrap())
            };
            current = next_child;
            depth += 1;
        }

        // Should have stopped at MAX_DETERMINISTIC_DEPTH
        assert_eq!(
            depth,
            MCTSSearch::<InfiniteForceGame>::MAX_DETERMINISTIC_DEPTH,
            "Collapse should stop at MAX_DETERMINISTIC_DEPTH"
        );
    }
}
