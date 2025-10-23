use ndarray::{Array1, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::board::BoardConfig;
use crate::game::{apply_capture, apply_placement, get_game_outcome, get_valid_actions, is_game_over};
use crate::node::{Action, MCTSNode};
use crate::transposition::TranspositionTable;

/// MCTS Search implementation
#[pyclass]
pub struct MCTSSearch {
    exploration_constant: f32,
    progressive_widening: bool,
    widening_constant: f32,
    use_transposition_table: bool,
    use_transposition_lookups: bool,
    transposition_table: Option<Arc<TranspositionTable>>,
    last_root_children: usize,
    last_root_visits: u32,
    last_root_value: f32,
    rng: Mutex<Option<StdRng>>,
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
        progressive_widening=None,
        widening_constant=None,
        use_transposition_table=None,
        use_transposition_lookups=None
    ))]
    fn new(
        exploration_constant: Option<f32>,
        progressive_widening: Option<bool>,
        widening_constant: Option<f32>,
        use_transposition_table: Option<bool>,
        use_transposition_lookups: Option<bool>,
    ) -> Self {
        Self {
            exploration_constant: exploration_constant.unwrap_or(1.41),
            progressive_widening: progressive_widening.unwrap_or(true),
            widening_constant: widening_constant.unwrap_or(10.0),
            use_transposition_table: use_transposition_table.unwrap_or(true),
            use_transposition_lookups: use_transposition_lookups.unwrap_or(true),
            transposition_table: None,
            last_root_children: 0,
            last_root_visits: 0,
            last_root_value: 0.0,
            rng: Mutex::new(None),
        }
    }

    /// Set deterministic RNG seed (pass None to restore system randomness)
    #[pyo3(signature = (seed=None))]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        let mut guard = self.rng.lock().unwrap();
        if let Some(value) = seed {
            *guard = Some(StdRng::seed_from_u64(value));
        } else {
            *guard = None;
        }
    }

    /// Run MCTS search (serial mode)
    #[pyo3(signature = (
        spatial,
        global,
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
        blitz=false
    ))]
    fn search(
        &mut self,
        spatial: PyReadonlyArray3<f32>,
        global: PyReadonlyArray1<f32>,
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
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        );

        let spatial_arr = spatial.as_array().to_owned();
        let global_arr = global.as_array().to_owned();

        let shared_entry = if search_options.use_lookups() {
            search_options.table_ref().map(|table_ref| {
                table_ref.get_or_insert(&spatial_arr.view(), &global_arr.view(), config.as_ref())
            })
        } else {
            None
        };

        let root = Arc::new(MCTSNode::new(
            spatial_arr,
            global_arr,
            Arc::clone(&config),
            shared_entry,
        ));

        let start = Instant::now();

        // Run MCTS iterations
        for _ in 0..iterations {
            if !self.run_iteration(
                Arc::clone(&root),
                &search_options,
                max_depth,
                time_limit,
                start,
            ) {
                break;
            }
        }

        let elapsed = start.elapsed();

        if verbose {
            let value = root.get_value();
            let children_count = root.children.lock().unwrap().len();
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
        spatial,
        global,
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
        blitz=false
    ))]
    fn search_parallel(
        &mut self,
        py: Python<'_>,
        spatial: PyReadonlyArray3<f32>,
        global: PyReadonlyArray1<f32>,
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

        // Configure thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let config = Arc::new(
            if blitz {
                BoardConfig::blitz(rings, t)
            } else {
                BoardConfig::standard(rings, t)
            }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        );

        let spatial_arr = spatial.as_array().to_owned();
        let global_arr = global.as_array().to_owned();

        let shared_entry = if search_options.use_lookups() {
            search_options.table_ref().map(|table_ref| {
                table_ref.get_or_insert(&spatial_arr.view(), &global_arr.view(), config.as_ref())
            })
        } else {
            None
        };

        let root = Arc::new(MCTSNode::new(
            spatial_arr,
            global_arr,
            Arc::clone(&config),
            shared_entry,
        ));

        let start = Instant::now();

        // Release GIL for parallel work
        py.detach(|| {
            (0..iterations).into_par_iter().for_each(|_| {
                self.run_iteration(
                    Arc::clone(&root),
                    &search_options,
                    max_depth,
                    time_limit,
                    start,
                );
            });
        });

        let elapsed = start.elapsed();

        if verbose {
            let value = root.get_value();
            let children_count = root.children.lock().unwrap().len();
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
}

impl MCTSSearch {
    fn with_rng<T, F>(&self, f: F) -> T
    where
        F: FnOnce(&mut dyn RngCore) -> T,
    {
        let mut guard = self.rng.lock().unwrap();
        if let Some(ref mut seeded) = guard.as_mut() {
            f(seeded)
        } else {
            drop(guard);
            let mut rng = rand::rng();
            f(&mut rng)
        }
    }

    fn capture_root_stats(&mut self, root: &MCTSNode) {
        if let Ok(children) = root.children.lock() {
            self.last_root_children = children.len();
        } else {
            self.last_root_children = 0;
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

        // Simulation
        let value = self.simulate(&node, max_depth, time_limit, start_time);

        // Backpropagation (updates table when configured)
        self.backpropagate(&node, value, table_ref);

        true
    }

    /// Selection phase: traverse tree using UCB1
    fn select(
        &self,
        mut node: Arc<MCTSNode>,
        table: Option<&Arc<TranspositionTable>>,
        use_lookups: bool,
    ) -> Arc<MCTSNode> {
        loop {
            // Check if node is fully expanded
            if !node.is_fully_expanded(self.progressive_widening, self.widening_constant) {
                return self.expand(Arc::clone(&node), table, use_lookups);
            }

            // Check if terminal
            if self.is_terminal(&node) {
                return node;
            }

            // Select best child using UCB1
            let parent_visits = node.get_visits();
            let children = node.children.lock().unwrap();
            let best_child = children.iter().max_by(|(_, child_a), (_, child_b)| {
                let score_a = child_a.ucb1_score(parent_visits, self.exploration_constant);
                let score_b = child_b.ucb1_score(parent_visits, self.exploration_constant);
                // Handle NaN gracefully (treat equal if either is NaN)
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((_, child)) = best_child {
                let next_node = Arc::clone(child);
                drop(children); // Release lock before next iteration
                node = next_node;
            } else {
                drop(children); // Release lock before returning
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
            get_valid_actions(&node.spatial.view(), &node.global.view(), &node.config);

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
            let children = node.children.lock().unwrap();
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
        let mut child_spatial = node.spatial.clone();
        let mut child_global = node.global.clone();

        match &action {
            Action::Placement {
                marble_type,
                dst_y,
                dst_x,
                remove_y,
                remove_x,
            } => {
                apply_placement(
                    &mut child_spatial,
                    &mut child_global,
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
                    &mut child_spatial,
                    &mut child_global,
                    *start_y,
                    *start_x,
                    *direction,
                    &node.config,
                );
            }
            Action::Pass => {
                // Just switch player
                let cur_player = child_global[node.config.cur_player] as usize;
                child_global[node.config.cur_player] = if cur_player == node.config.player_1 {
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
                    &child_spatial.view(),
                    &child_global.view(),
                    node.config.as_ref(),
                )
            })
        } else {
            None
        };

        let child = Arc::new(MCTSNode::new_child(
            child_spatial,
            child_global,
            Arc::clone(&node.config),
            &node,
            shared_entry,
        ));

        // Add child to parent (thread-safe)
        node.add_child(action, Arc::clone(&child));

        child
    }

    /// Simulation phase: play out random game to terminal state
    ///
    /// Returns result from node's current player's perspective: +1 (win), -1 (loss), 0 (draw)
    fn simulate(
        &self,
        node: &MCTSNode,
        max_depth: Option<usize>,
        time_limit: Option<f32>,
        start_time: Instant,
    ) -> f32 {
        let leaf_player = node.global[node.config.cur_player] as usize;

        if self.is_terminal(node) {
            return self.evaluate_terminal(&node.spatial, &node.global, &node.config, leaf_player);
        }

        let mut sim_spatial = node.spatial.clone();
        let mut sim_global = node.global.clone();

        let mut consecutive_passes = 0usize;
        let depth_limit = max_depth.unwrap_or(usize::MAX);
        for depth in 0..depth_limit {
            if let Some(limit) = time_limit {
                if start_time.elapsed().as_secs_f32() >= limit {
                    return self.evaluate_heuristic(
                        &sim_spatial,
                        &sim_global,
                        &node.config,
                        leaf_player,
                    );
                }
            }

            if self.is_terminal_state(&sim_spatial, &sim_global, &node.config) {
                return self.evaluate_terminal(
                    &sim_spatial,
                    &sim_global,
                    &node.config,
                    leaf_player,
                );
            }

            if consecutive_passes >= 2 {
                return 0.0;
            }

            let (placement_mask, capture_mask) =
                get_valid_actions(&sim_spatial.view(), &sim_global.view(), &node.config);

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
                apply_capture(
                    &mut sim_spatial,
                    &mut sim_global,
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
                    apply_placement(
                        &mut sim_spatial,
                        &mut sim_global,
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
                    let cur_player = sim_global[node.config.cur_player] as usize;
                    sim_global[node.config.cur_player] = if cur_player == node.config.player_1 {
                        node.config.player_2 as f32
                    } else {
                        node.config.player_1 as f32
                    };
                }
            }

            if depth + 1 >= depth_limit {
                return self.evaluate_heuristic(
                    &sim_spatial,
                    &sim_global,
                    &node.config,
                    leaf_player,
                );
            }
        }

        self.evaluate_heuristic(&sim_spatial, &sim_global, &node.config, leaf_player)
    }

    /// Backpropagation phase: update statistics up the tree
    ///
    /// Values are flipped at each level since players alternate (zero-sum game).
    fn backpropagate(
        &self,
        node: &Arc<MCTSNode>,
        mut value: f32,
        table: Option<&Arc<TranspositionTable>>,
    ) {
        let mut current = Some(Arc::clone(node));

        while let Some(current_node) = current {
            current_node.update(value);

            if let Some(table_ref) = table {
                if !current_node.has_shared_stats() {
                    table_ref.store(
                        &current_node.spatial.view(),
                        &current_node.global.view(),
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
        self.is_terminal_state(&node.spatial, &node.global, &node.config)
    }

    /// Check if state is terminal (standalone version)
    fn is_terminal_state(
        &self,
        spatial: &Array3<f32>,
        global: &Array1<f32>,
        config: &BoardConfig,
    ) -> bool {
        // Delegate to game.rs function (single source of truth)
        is_game_over(&spatial.view(), &global.view(), config)
    }

    /// Evaluate terminal state from root player's perspective
    ///
    /// Returns +1 if root_player won, -1 if lost, 0 if draw, -2 if both lose
    fn evaluate_terminal(
        &self,
        spatial: &Array3<f32>,
        global: &Array1<f32>,
        config: &BoardConfig,
        root_player: usize,
    ) -> f32 {
        // Delegate to game.rs function (single source of truth)
        // Returns: 1 (P1 wins), -1 (P2 wins), 0 (tie), -2 (both lose)
        let outcome = get_game_outcome(&spatial.view(), &global.view(), config);

        // Convert from Player 1's perspective to root_player's perspective
        match outcome {
            -2 => -2.0,  // Both lose
            0 => 0.0,    // Tie
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
            _ => 0.0,  // Unknown outcome, treat as draw
        }
    }

    /// Heuristic evaluation for non-terminal states
    ///
    /// Uses weighted marble values: white=1, gray=2, black=3
    fn evaluate_heuristic(
        &self,
        _spatial: &Array3<f32>,
        global: &Array1<f32>,
        config: &BoardConfig,
        root_player: usize,
    ) -> f32 {
        // Weight by marble value
        let weights = [1.0, 2.0, 3.0]; // white, gray, black

        let p0_score: f32 = (0..3)
            .map(|i| global[config.p1_cap_w + i] * weights[i])
            .sum();
        let p1_score: f32 = (0..3)
            .map(|i| global[config.p2_cap_w + i] * weights[i])
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
        let children = root.children.lock().unwrap();

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
            table: search.transposition_table.as_ref().map(|t| Arc::clone(t)),
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
