//! # MCTS Game Trait
//!
//! Core trait that all MCTS-compatible games must implement.
//!
//! The MCTS algorithm treats actions as opaque tokens - it never inspects
//! or interprets them. All game-specific logic is delegated to trait methods.
//!
//! ## Design Principles
//!
//! - **Universal state representation**: All games use Array3<f32> (spatial) and Array1<f32> (global)
//! - **Opaque actions**: MCTS doesn't know what actions mean, only stores them as tree edges
//! - **Zero-cost abstraction**: Generics enable monomorphization with no runtime overhead
//! - **Sensible defaults**: Optional methods have defaults that work for most games

use ndarray::{Array1, Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3};
use std::fmt::Debug;
use std::hash::Hash;

/// Core trait that all MCTS-compatible games must implement.
///
/// ## Example Implementation
///
/// ```rust,ignore
/// use crate::game_trait::MCTSGame;
///
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// pub enum MyAction {
///     Move { from: (usize, usize), to: (usize, usize) },
///     Pass,
/// }
///
/// pub struct MyGame {
///     config: MyConfig,
///     hasher: GenericZobristHasher,
/// }
///
/// impl MCTSGame for MyGame {
///     type Action = MyAction;
///
///     fn get_valid_actions(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> Vec<MyAction> {
///         // Generate legal moves based on game rules
///         vec![]
///     }
///
///     fn apply_action(&self, spatial: &mut ArrayViewMut3<f32>, global: &mut ArrayViewMut1<f32>, action: &MyAction) {
///         // Modify state according to action
///     }
///
///     fn is_terminal(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> bool {
///         // Check if game has ended
///         false
///     }
///
///     fn get_outcome(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> i32 {
///         // Return: +1 (P1 wins), -1 (P2 wins), 0 (draw), -2 (both lose)
///         0
///     }
///
///     fn get_current_player(&self, global: &ArrayView1<f32>) -> usize {
///         global[0] as usize
///     }
///
///     fn spatial_shape(&self) -> (usize, usize, usize) {
///         (4, 8, 8)  // layers, height, width
///     }
///
///     fn global_size(&self) -> usize {
///         10
///     }
///
///     fn evaluate_heuristic(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>, root_player: usize) -> f32 {
///         // Return heuristic value in [-1, 1] from root_player's perspective
///         0.0
///     }
///
///     fn hash_state(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> u64 {
///         self.hasher.hash_state(spatial, global)
///     }
///
///     fn name(&self) -> &str {
///         "MyGame"
///     }
/// }
/// ```
pub trait MCTSGame: Send + Sync + 'static {
    /// The action type for this game. MCTS treats this as an opaque token.
    ///
    /// Must be cloneable, comparable, and hashable for use in the game tree.
    /// MCTS never inspects action contents - it only stores them as tree edges
    /// and passes them back to the game for application.
    type Action: Clone + Eq + Hash + Send + Sync + Debug;

    // ========================================================================
    // CORE GAME RULES - Required for MCTS to function
    // ========================================================================

    /// Generate all legal actions from the current state.
    ///
    /// MCTS calls this during expansion to find untried actions.
    /// The returned actions are opaque to MCTS - it just stores them as tree edges.
    ///
    /// # Arguments
    /// * `spatial_state` - Spatial state array (layers, height, width)
    /// * `global_state` - Global state array (scalar values, player info, etc.)
    ///
    /// # Returns
    /// Vector of all legal actions. Must not be empty unless game is terminal.
    fn get_valid_actions(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> Vec<Self::Action>;

    /// Apply an action to the state, mutating it in-place.
    ///
    /// MCTS calls this during expansion and simulation.
    /// The action is opaque to MCTS - only the game knows what it means.
    ///
    /// # Arguments
    /// * `spatial_state` - Mutable view of spatial state
    /// * `global_state` - Mutable view of global state
    /// * `action` - The action to apply (opaque to MCTS)
    ///
    /// # Panics
    /// Should not panic if action is legal. Behavior undefined for illegal actions.
    fn apply_action(
        &self,
        spatial_state: &mut ArrayViewMut3<f32>,
        global_state: &mut ArrayViewMut1<f32>,
        action: &Self::Action,
    );

    /// Check if the game has ended.
    ///
    /// MCTS calls this during selection and simulation to detect terminal nodes.
    ///
    /// # Returns
    /// `true` if game is over, `false` otherwise
    fn is_terminal(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> bool;

    /// Get the game outcome from Player 1's perspective.
    ///
    /// Only called on terminal states. MCTS uses this during backpropagation
    /// to evaluate terminal nodes.
    ///
    /// # Returns
    /// * `+1` - Player 1 wins
    /// * `-1` - Player 2 wins
    /// * `0` - Draw/tie
    /// * `-2` - Both players lose (unusual but supported)
    ///
    /// # Panics
    /// May panic or return invalid value if called on non-terminal state.
    fn get_outcome(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> i8;

    /// Get the current player index.
    ///
    /// Used by MCTS to track perspective changes during backpropagation.
    ///
    /// # Returns
    /// Player index (typically 0 or 1 for two-player games)
    fn get_current_player(&self, global_state: &ArrayView1<f32>) -> usize;

    // ========================================================================
    // STATE DIMENSIONS - Required for array allocation
    // ========================================================================

    /// Get the shape of the spatial state array (layers, height, width).
    ///
    /// Used by MCTS to allocate arrays for child nodes.
    ///
    /// # Returns
    /// Tuple of (layers, height, width)
    fn spatial_shape(&self) -> (usize, usize, usize);

    /// Get the size of the global state array.
    ///
    /// Used by MCTS to allocate arrays for child nodes.
    ///
    /// # Returns
    /// Number of elements in global state vector
    fn global_size(&self) -> usize;

    // ========================================================================
    // EVALUATION - Required for non-terminal simulation cutoff
    // ========================================================================

    /// Evaluate a non-terminal state heuristically.
    ///
    /// Called when simulation reaches max_depth or time_limit without
    /// reaching a terminal state.
    ///
    /// # Arguments
    /// * `spatial_state` - Current spatial state
    /// * `global_state` - Current global state
    /// * `root_player` - Player at the root of the search tree
    ///
    /// # Returns
    /// Value in `[-1.0, 1.0]` from root_player's perspective:
    /// * `+1.0` - root_player is winning
    /// * `-1.0` - root_player is losing
    /// * `0.0` - even position
    ///
    /// Values outside [-1, 1] are clamped by MCTS.
    fn evaluate_heuristic(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
        root_player: usize,
    ) -> f32;

    // ========================================================================
    // TRANSPOSITION TABLE SUPPORT - Optional optimizations
    // ========================================================================

    /// Canonicalize state for transposition table lookup.
    ///
    /// Games with symmetries should override this to find canonical form
    /// (e.g., apply rotation/reflection to get unique representation).
    ///
    /// # Default
    /// Returns state unchanged (no canonicalization).
    ///
    /// # Returns
    /// Tuple of (canonical_spatial_state, canonical_global_state)
    ///
    /// # Example
    /// ```rust,ignore
    /// fn canonicalize_state(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> (Array3<f32>, Array1<f32>) {
    ///     let (canonical_spatial, canonical_global, _transform) =
    ///         canonicalization::canonicalize_state(spatial, &self.config);
    ///     (canonical_spatial, global.to_owned())
    /// }
    /// ```
    fn canonicalize_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> (Array3<f32>, Array1<f32>) {
        (spatial_state.to_owned(), global_state.to_owned())
    }

    /// Hash a state for transposition table lookup.
    ///
    /// Must be consistent with `canonicalize_state`:
    /// - Canonical states must hash identically
    /// - Different states should hash differently (ideally)
    ///
    /// # Arguments
    /// * `spatial_state` - Spatial state to hash
    /// * `global_state` - Global state to hash
    ///
    /// # Returns
    /// 64-bit hash value
    ///
    /// # Example
    /// ```rust,ignore
    /// fn hash_state(&self, spatial: &ArrayView3<f32>, global: &ArrayView1<f32>) -> u64 {
    ///     self.zobrist_hasher.hash_state(spatial, global, &self.config)
    /// }
    /// ```
    fn hash_state(
        &self,
        spatial_state: &ArrayView3<f32>,
        global_state: &ArrayView1<f32>,
    ) -> u64;

    // ========================================================================
    // DETERMINISTIC SEQUENCE COLLAPSING - Optional optimization
    // ========================================================================

    /// Enable automatic collapsing of deterministic sequences in the game tree.
    ///
    /// A **deterministic sequence** is a series of game states where the player has no
    /// real strategic choice - either only one legal action exists, or all alternatives
    /// are clearly dominated. By collapsing these sequences, MCTS avoids wasting
    /// computational resources on trivial decisions.
    ///
    /// ## What This Optimization Does
    ///
    /// When enabled, during tree selection (`collapse_deterministic_sequence()`), MCTS will:
    /// 1. Check if current state has a forced/deterministic action (via `get_forced_action()`)
    /// 2. If forced: automatically apply it, create the child node, and continue
    /// 3. If not forced: stop and treat as a real decision point
    /// 4. Repeat until reaching a choice point, terminal state, or depth limit
    ///
    /// **Benefits:**
    /// - Saves MCTS iterations - no wasted exploration of non-choices
    /// - Prevents visit count inflation on trivial nodes
    /// - Improves UCB statistics for actual decision points
    /// - Compresses game tree to only represent strategic choices
    ///
    /// **Safety:** The traversal is depth-limited (MAX_DETERMINISTIC_DEPTH) to prevent
    /// infinite loops in buggy game implementations.
    ///
    /// ## Common Use Cases Across Game Types
    ///
    /// ### Mandatory Multi-Move Sequences
    /// - **Checkers**: Forced multi-jump chains when captures are available
    /// - **Zertz**: Chain captures where player must continue jumping until no captures remain
    /// - **Chess**: Forced recapture sequences (though rare)
    ///
    /// ### Single Legal Move Situations
    /// - **Chess**: Forced king moves out of check when only one legal square exists
    /// - **Go**: Atari responses when only one move saves the group
    /// - **Card Games**: Forced discards when hand exceeds size limit
    /// - **Abstract Games**: Any position with exactly one legal move
    ///
    /// ### Ko/Repetition Forced Plays
    /// - **Go**: Ko fight responses when no other reasonable move exists
    /// - **Chess**: Repetition avoidance when alternatives lose material
    ///
    /// ## Advanced: Choice Thresholds & Early Stopping
    ///
    /// The `get_forced_action()` method receives the full game state, allowing games to
    /// implement sophisticated forcing logic beyond "1 legal move = forced":
    ///
    /// **Choice Threshold Example** (collapse weak branches):
    /// ```rust,ignore
    /// fn get_forced_action(&self, actions: &[Action], spatial: &ArrayView3<f32>, ...) -> Option<Action> {
    ///     if actions.len() <= 3 {
    ///         // Evaluate each action's tactical value
    ///         let best = find_best_action(actions, spatial);
    ///         if best.value > all_others.max_value + THRESHOLD {
    ///             return Some(best.action);  // One clearly dominant move
    ///         }
    ///     }
    ///     None  // Multiple viable options
    /// }
    /// ```
    ///
    /// **Early Stopping Example** (preserve important decisions):
    /// ```rust,ignore
    /// fn get_forced_action(&self, actions: &[Action], ...) -> Option<Action> {
    ///     if actions.len() == 1 {
    ///         match &actions[0] {
    ///             Action::Capture { .. } => Some(actions[0].clone()),  // Force captures
    ///             Action::EndGame { .. } => None,  // Don't force game-ending moves
    ///             _ => Some(actions[0].clone()),
    ///         }
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    ///
    /// ## Default Behavior
    ///
    /// `false` - Disabled by default for safety. Games should opt-in after verifying the
    /// optimization is beneficial for their move structure.
    ///
    /// ## Example Implementation
    ///
    /// ```rust,ignore
    /// impl MCTSGame for MyGame {
    ///     fn enable_deterministic_collapse(&self) -> bool {
    ///         true  // Enable for games with forced sequences
    ///     }
    ///
    ///     // Optional: use default get_forced_action() (forces when 1 legal move)
    ///     // Or override for custom forcing logic
    /// }
    /// ```
    fn enable_deterministic_collapse(&self) -> bool {
        false
    }

    /// Determine if the current state has a forced/deterministic action.
    ///
    /// This method is called during `collapse_deterministic_sequence()` (only if
    /// `enable_deterministic_collapse()` is true) to determine whether the current
    /// game state represents a real strategic choice or a trivial/forced decision.
    ///
    /// ## Purpose
    ///
    /// Identify states where:
    /// - Only one legal action exists (no actual choice)
    /// - Multiple actions exist but one is clearly dominant (choice threshold)
    /// - The action is mandated by game rules (e.g., must continue multi-captures)
    ///
    /// ## Arguments
    ///
    /// * `actions` - Slice of all legal actions from current state
    /// * `spatial_state` - Current spatial game state (for evaluation)
    /// * `global_state` - Current global game state (for evaluation)
    ///
    /// ## Returns
    ///
    /// * `Some(action)` - This action should be automatically applied (deterministic)
    /// * `None` - No forced action, this is a real choice point for MCTS exploration
    ///
    /// ## Default Implementation
    ///
    /// Forces if exactly 1 legal action exists:
    /// ```rust,ignore
    /// if actions.len() == 1 {
    ///     Some(actions[0].clone())
    /// } else {
    ///     None
    /// }
    /// ```
    ///
    /// This default works well for most games and requires no customization.
    ///
    /// ## Custom Implementation Examples
    ///
    /// ### Example 1: Only Force Specific Move Types
    /// ```rust,ignore
    /// fn get_forced_action(&self, actions: &[MyAction], ...) -> Option<MyAction> {
    ///     if actions.len() == 1 {
    ///         match &actions[0] {
    ///             MyAction::Capture { .. } => Some(actions[0].clone()),
    ///             MyAction::Pass => Some(actions[0].clone()),
    ///             _ => None,  // Don't force regular moves even if only option
    ///         }
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    ///
    /// ### Example 2: Choice Threshold (collapse dominated branches)
    /// ```rust,ignore
    /// fn get_forced_action(&self, actions: &[MyAction], spatial: &ArrayView3<f32>, ...) -> Option<MyAction> {
    ///     if actions.len() >= 2 && actions.len() <= 4 {
    ///         // Quick tactical evaluation
    ///         let scores: Vec<f32> = actions.iter()
    ///             .map(|a| self.evaluate_tactical(a, spatial))
    ///             .collect();
    ///
    ///         let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ///         let second_best = scores.iter().copied()
    ///             .filter(|&s| s < max_score)
    ///             .fold(f32::NEG_INFINITY, f32::max);
    ///
    ///         if max_score > second_best + 2.0 {  // Threshold
    ///             let best_idx = scores.iter().position(|&s| s == max_score).unwrap();
    ///             return Some(actions[best_idx].clone());
    ///         }
    ///     }
    ///     None
    /// }
    /// ```
    ///
    /// ### Example 3: State-Dependent Forcing
    /// ```rust,ignore
    /// fn get_forced_action(&self, actions: &[MyAction], _: &ArrayView3<f32>, global: &ArrayView1<f32>) -> Option<MyAction> {
    ///     if actions.len() == 1 {
    ///         // Don't force in endgame (last 10 moves)
    ///         let moves_played = global[self.move_count_idx] as usize;
    ///         if moves_played > self.max_moves - 10 {
    ///             return None;  // Preserve all endgame decisions
    ///         }
    ///         Some(actions[0].clone())
    ///     } else {
    ///         None
    ///     }
    /// }
    /// ```
    ///
    /// ## Performance Considerations
    ///
    /// This method is called once per node during traversal, so it should be fast:
    /// - Default implementation is O(1)
    /// - If you add evaluation logic, keep it lightweight (< 1ms)
    /// - Avoid expensive simulations or deep lookahead here
    ///
    /// ## Safety
    ///
    /// The deterministic collapse is depth-limited to prevent infinite loops.
    /// If your `get_forced_action()` logic has bugs that cause cycles, the traversal
    /// will terminate after MAX_DETERMINISTIC_DEPTH steps.
    fn get_forced_action(
        &self,
        actions: &[Self::Action],
        _spatial_state: &ArrayView3<f32>,
        _global_state: &ArrayView1<f32>,
    ) -> Option<Self::Action> {
        // Default: force if exactly 1 legal action
        if actions.len() == 1 {
            Some(actions[0].clone())
        } else {
            None
        }
    }

    // ========================================================================
    // METADATA
    // ========================================================================

    /// Get the game name (for debugging/logging).
    ///
    /// # Returns
    /// Human-readable game name (e.g., "Zertz", "Chess", "Go")
    fn name(&self) -> &str;
}
