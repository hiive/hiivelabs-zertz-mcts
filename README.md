# hiivelabs-mcts

Generic Rust-accelerated Monte Carlo Tree Search (MCTS) engine with game-specific implementations. The crate is exported as the `hiivelabs_mcts` Python module.

**Currently supports:**
- Zèrtz (via `ZertzMCTS` class)
- Tic-Tac-Toe (via `TicTacToeMCTS` class)

**Extensible to other games** via the `MCTSGame` trait.

## Features
- **Generic MCTS core**: Trait-based architecture supporting any two-player zero-sum game
- **UCT-based search** with optional RAVE, progressive widening, FPU, and transposition tables
- **Deterministic sequence collapsing**: Automatically traverses forced moves (e.g., Zertz capture chains) to focus search on real decisions
- **Deterministic runs** via per-search RNG seeding plus configurable table clearing
- **Parallel rollouts** with Rayon; thread-safe statistics backed by atomics and `DashMap`
- **PyO3 bindings** expose zero-copy NumPy views for spatial and global state tensors
- **Shared rule helpers** (placement/capture application, canonicalization) for parity tests between Python and Rust
- **Multiple rule sets**: Supports Standard and Blitz modes for Zertz, including supply pools and win thresholds

## Architecture

### Generic MCTS Core

The engine is built around the `MCTSGame` trait, which defines the interface all games must implement:

```rust
pub trait MCTSGame: Send + Sync + 'static {
    type Action: Clone + Eq + Hash + Send + Sync;

    fn get_valid_actions(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> Vec<Self::Action>;
    fn apply_action(&self, spatial_state: &mut ArrayViewMut3<f32>, global_state: &mut ArrayViewMut1<f32>, action: &Self::Action);
    fn is_terminal(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> bool;
    fn get_outcome(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> i8;
    fn get_current_player(&self, global_state: &ArrayView1<f32>) -> usize;
    fn evaluate_heuristic(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>, root_player: usize) -> f32;
    fn canonicalize_state(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> (Array3<f32>, Array1<f32>);
    fn hash_state(&self, spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> u64;

    // Optional: Enable deterministic sequence collapsing
    fn enable_deterministic_collapse(&self) -> bool { false }
    fn get_forced_action(&self, actions: &[Self::Action], spatial_state: &ArrayView3<f32>, global_state: &ArrayView1<f32>) -> Option<Self::Action> {
        if actions.len() == 1 { Some(actions[0].clone()) } else { None }
    }

    fn name(&self) -> &str;
}
```

This design enables:
- **Zero-cost abstraction**: Monomorphization means no runtime overhead
- **Type-safe actions**: Each game defines its own action representation
- **Extensibility**: Add new games without modifying the MCTS core
- **Game-specific optimizations**: Override heuristics, forced move detection, and canonicalization

### Module Layout

**Core MCTS Engine:**
- `src/game_trait.rs` — `MCTSGame` trait definition and documentation
- `src/mcts.rs` — Generic selection/expansion/simulation/backpropagation logic
- `src/node.rs` — Generic concurrent node representation with atomic visit/value tracking
- `src/transposition.rs` — Generic lock-free transposition table
- `src/metrics.rs` — Performance metrics (optional feature)

**Game Implementations:**
- `src/games/mod.rs` — Game implementations module
- `src/games/zertz/` — Complete Zertz implementation
  - `mod.rs` — `ZertzGame` implementation of `MCTSGame` trait
  - `board.rs` — Board configuration, game modes, win conditions
  - `logic.rs` — Core game rules (placement, capture, termination)
  - `canonicalization.rs` — Symmetry transforms for hexagonal boards
  - `action_transform.rs` — Action transformation for symmetry operations
  - `notation.rs` — Algebraic notation conversion (e.g., (3,3) ↔ "D4")
  - `zobrist.rs` — Zobrist hashing for fast state hashing
  - `py_logic.rs` — Python bindings for stateless game functions
  - `py_mcts.rs` — `ZertzMCTS` Python wrapper
- `src/games/tictactoe/` — Tic-Tac-Toe implementation
  - `mod.rs` — `TicTacToeGame` implementation (minimal example)
  - `canonicalization.rs` — D4 dihedral group symmetries (8-fold: 4 rotations + 4 reflections)
  - `canonicalization_tests.rs` — Comprehensive canonicalization tests (27 tests)
- `src/games/tictactoe_py.rs` — `TicTacToeMCTS` Python wrapper

**Python Interface:**
- `src/lib.rs` — PyO3 module registration (exports `hiivelabs_mcts` Python module)
- `hiivelabs_mcts.pyi` — Type stubs for IDE autocomplete and type checking

**Tests:**
All test modules are in separate files co-located with the code they test (e.g., `canonicalization_tests.rs`, `logic_tests.rs`)

## Installation
- Rust toolchain (1.74+ recommended) and Python 3.8+ (ABI3 wheel target).
- Install `maturin` into the desired Python environment:  
  `pip install maturin`
- Build and install the extension in editable mode:  
  `maturin develop --release`
- Alternatively, produce a wheel for distribution:  
  `maturin build --release --strip`

The compiled module is published as `hiivelabs_mcts` and can be imported from Python once the build succeeds.

## Adding a New Game

To add support for a new game, implement the `MCTSGame` trait. The repository includes a complete Tic-Tac-Toe example (`src/games/tictactoe/`) demonstrating the minimal implementation:

**Key steps:**
1. **Create game module** in `src/games/yourgame/` (for complex games) or `src/games/yourgame.rs` (for simple ones)
2. **Define action type** (e.g., `struct YourGameAction`)
3. **Implement `MCTSGame` trait** with all required methods
4. **Create Python wrapper** in `src/games/yourgame_py.rs`
5. **Register in `lib.rs`** and `src/games/mod.rs`
6. **Write tests** in separate test files (e.g., `yourgame_tests.rs`)

**Examples:**
- **Simple game:** `src/games/tictactoe/` - minimal example with complete D4 canonicalization (~500 lines including tests)
- **Complex game:** `src/games/zertz/` - full-featured multi-module implementation with hexagonal canonicalization, Zobrist hashing, and extensive testing

The trait requires implementing:
- `get_valid_actions()` - Return legal moves
- `apply_action()` - Apply move to state (mutating)
- `is_terminal()` - Check if game is over
- `get_outcome()` - Return winner (+1/-1/0)
- `get_current_player()` - Extract current player from state
- `evaluate_heuristic()` - Estimate position value (for rollouts)
- `canonicalize_state()` - Normalize state for transposition table
- `hash_state()` - Hash state for transposition table lookup
- Optional: `enable_deterministic_collapse()` and `get_forced_action()` for forced move sequences

## Deprecations

### Python API

- **`translate_state()`** - *Deprecated* - Use `transform_state(spatial_state, config, 0, False, False, dy, dx)` instead
  - This function now emits a `DeprecationWarning` when called
  - Will be removed in a future major version
  - Maintained for backward compatibility only

- **Old action functions** - *Deprecated* - Use ZertzAction-based equivalents instead
  - `apply_placement_action_old()` → Use `apply_placement_action(config, spatial_state, global_state, action: ZertzAction)`
  - `apply_capture_action_old()` → Use `apply_capture_action(config, spatial_state, global_state, action: ZertzAction)`
  - `transform_action_old()` → Use `transform_action(config, action: ZertzAction, transform: str)`
  - `BoardState.apply_placement_old()` → Use `BoardState.apply_placement(action: ZertzAction)`
  - `BoardState.apply_capture_old()` → Use `BoardState.apply_capture(action: ZertzAction)`
  - These functions are marked deprecated in Rust and will be removed in a future major version
  - New functions provide better type safety and return structured `ZertzActionResult` objects

## Zertz State Encoding
- Spatial tensor shape: `(t * 4 + 1, width, width)`
  Layer 0 stores ring presence, layers 1–3 track colour occupancy across timesteps, the final layer carries capture overlays when history is provided.
- Global vector length: `10 * t` (for `t = 1`, indices 0–2 = supply counts, 3–5 = player 1 captures, 6–8 = player 2 captures, 9 = active player flag).
- Actions use flattened coordinates: `dst = y * width + x`; a `remove` equal to `width * width` indicates no ring removal.
- Helper routines in `game.rs` mirror the Python loaders to keep replay, parity, and symmetry tooling in sync.

### Algebraic Notation

For human-readable position notation, Zertz includes coordinate conversion functions:

```python
from hiivelabs_mcts import BoardConfig, coordinate_to_algebraic, algebraic_to_coordinate

config = BoardConfig.standard_config(37)  # 37-ring board

# Convert array coordinates to algebraic notation
coordinate_to_algebraic(3, 0, config)  # Returns "A1" (left side of middle row)
coordinate_to_algebraic(3, 3, config)  # Returns "D4" (center)
coordinate_to_algebraic(0, 0, config)  # Returns "A4" (top-left corner)

# Parse algebraic notation to coordinates (case-insensitive)
algebraic_to_coordinate("A1", config)  # Returns (3, 0)
algebraic_to_coordinate("d4", config)  # Returns (3, 3)
algebraic_to_coordinate("A4", config)  # Returns (0, 0)
```

**Hexagonal Coordinate System:**
- **Columns**: A, B, C, ... (left to right, x-axis)
- **Rows**: Row numbers increase as you move up-right in the hexagon
- The middle row is the longest, containing all columns
- Row number depends on BOTH x and y: `row = min(width, width/2 + x + 1) - y`
- Supports all board sizes: 37 rings (A-G), 49 rings (A-H), 61 rings (A-J, note 'I' is skipped)

## Tic-Tac-Toe State Encoding
- Spatial tensor shape: `(2, 3, 3)`
  Layer 0 stores X positions, Layer 1 stores O positions.
- Global vector length: `1` (index 0 = current player: 0 for X, 1 for O).
- Actions are simple `(row, col)` tuples (0-indexed).
- Minimal encoding demonstrates the basic requirements for implementing `MCTSGame`.

## Development
- `cargo test` covers rule regression tests and fast MCTS smoke checks.
- `cargo fmt` and `cargo clippy -- -D warnings` keep formatting and lints consistent with CI.
- use `uv run maturin develop --release` (or `./rust-dev.sh`) whenever the Rust crate changes so Python picks up the updated wheel.

## License

Distributed under the GNU Affero General Public License v3.0 (see `LICENSE`).
