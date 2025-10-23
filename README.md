# hiivelabs-zertz-mcts

Rust-accelerated Monte Carlo Tree Search (MCTS) engine for the abstract strategy game Zertz, distributed as a Python extension module for Hiive Labs tooling.

## Features
- Optimised MCTS with UCT, progressive widening, and optional transposition-table bootstrapping.
- Deterministic reproducibility via per-search RNG seeding and reusable table caches.
- Thread-safe tree with Rayon-powered parallel rollout mode for high iteration counts.
- PyO3 bindings that accept/return NumPy arrays without extra copies.
- Utilities that mirror the Python engine (move generation, placement/capture application) for cross-validation.
- **Full parity support for both Standard and Blitz game modes** with mode-specific win conditions.

## Architecture

The Rust backend mirrors the Python implementation's architecture:

**Delegation Pattern:**
- `mcts.rs` → `game.rs` for all game logic (same as Python's `mcts_tree.py` → `zertz_logic.py`)
- Single source of truth: all game rules live in `game.rs`
- MCTS focuses on tree search algorithms only

**Key Design Decisions:**
- Game termination checking: `is_game_over()` and `get_game_outcome()` in game.rs (lines 915-1140)
- Win condition logic: Centralized in game.rs with mode-specific thresholds (Standard: 3-of-each/4W/5G/6B, Blitz: 2-of-each/3W/4G/5B)
- MCTS only handles perspective conversion (absolute outcome → player-relative value)
- Outcome constants: `1` (P1 wins), `-1` (P2 wins), `0` (tie), `-2` (both lose)
- Game mode support: `BoardConfig::standard()` and `BoardConfig::blitz()` for different rule variants

**Module Responsibilities:**
- `game.rs`: Pure stateless game logic, single source of truth for rules
- `mcts.rs`: Tree search (selection, expansion, simulation, backpropagation) - delegates to game.rs
- `node.rs`: Thread-safe node structure with atomic statistics
- `transposition.rs`: Lock-free transposition table (DashMap)
- `zobrist.rs`: Fast state hashing for transposition detection
- `canonicalization.rs`: Symmetry handling and canonical states

## Installation
- Rust toolchain (1.74+ recommended) and Python 3.8+ (ABI3 wheel target).
- Install `maturin` into the desired Python environment:  
  `pip install maturin`
- Build and install the extension in editable mode:  
  `maturin develop --release`
- Alternatively, produce a wheel for distribution:  
  `maturin build --release --strip`

The compiled module is published as `hiivelabs_zertz_mcts` and can be imported from Python once the build succeeds.

## State Encoding
- Spatial tensor shape: `(t * 4 + 1, width, width)` where:
  - Layer 0 tracks whether a ring is present.
  - Layers 1–3 track white, gray, and black marbles for each timestep.
  - The final layer stores capture overlays when multiple timesteps are supplied.
- Global vector (length 10 for `t = 1`):
  - Indices 0–2: supply counts for white, gray, black.
  - Indices 3–5: player 1 captured counts; 6–8: player 2 captured counts.
  - Index 9: current player flag (`0` for player 1, `1` for player 2).
- `BoardState` mirrors this layout and exposes helper methods such as `get_valid_actions`, `apply_placement`, and `apply_capture` for parity checks against the Python implementation.
- Actions returned from MCTS use flattened coordinates: `dst_flat = y * width + x`. A `remove_flat` equal to `width * width` signals that no ring removal is required.

## Development
- Run the Rust unit tests (mirroring Python fixtures) with `cargo test`.
- Format and lint as usual with `cargo fmt` / `cargo clippy`.
- Use `maturin develop --release` to rebuild the extension while iterating on Rust code.

## License

Distributed under the GNU Affero General Public License v3.0 (see `LICENSE`).
