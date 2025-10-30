# hiivelabs-zertz-mcts

Rust-accelerated Monte Carlo Tree Search (MCTS) engine for the abstract strategy game Zèrtz. The crate is exported as the `hiivelabs_zertz_mcts` Python module used by the main application.

## Features
- UCT-based MCTS with optional RAVE, progressive widening, FPU, and transposition tables.
- Deterministic runs via per-search RNG seeding plus configurable table clearing.
- Parallel rollouts with Rayon; thread-safe statistics backed by atomics and `DashMap`.
- PyO3 bindings expose zero-copy NumPy views for spatial and global state tensors.
- Shared rule helpers (placement/capture application, canonicalisation) for parity tests between Python and Rust.
- Supports Standard and Blitz rule sets, including supply pools and win thresholds.

## Module Layout
- `src/game.rs` — canonical rule engine (state transitions, move generation, win checks).
- `src/mcts.rs` — selection/expansion/rollout/backprop logic, delegates rules to `game.rs`.
- `src/node.rs` — concurrent node representation with atomic visit/value tracking.
- `src/transposition.rs` — lock-free cache keyed by Zobrist hashes.
- `src/canonicalization.rs` — symmetry transforms for the 37/48/61 ring boards.
- `src/lib.rs` — PyO3 glue (argument parsing, NumPy conversion, Python-facing API).

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
- Spatial tensor shape: `(t * 4 + 1, width, width)`  
  Layer 0 stores ring presence, layers 1–3 track colour occupancy across timesteps, the final layer carries capture overlays when history is provided.
- Global vector length: `10 * t` (for `t = 1`, indices 0–2 = supply counts, 3–5 = player 1 captures, 6–8 = player 2 captures, 9 = active player flag).
- Actions use flattened coordinates: `dst = y * width + x`; a `remove` equal to `width * width` indicates no ring removal.
- Helper routines in `game.rs` mirror the Python loaders to keep replay, parity, and symmetry tooling in sync.

## Development
- `cargo test` covers rule regression tests and fast MCTS smoke checks.
- `cargo fmt` and `cargo clippy -- -D warnings` keep formatting and lints consistent with CI.
- use `uv run maturin develop --release` (or `./rust-dev.sh`) whenever the Rust crate changes so Python picks up the updated wheel.

## License

Distributed under the GNU Affero General Public License v3.0 (see `LICENSE`).
