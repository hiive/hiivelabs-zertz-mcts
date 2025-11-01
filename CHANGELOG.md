# Changelog

All notable changes to this project will be documented in this file.

[//]: # (The format is based on [Keep a Changelog]&#40;https://keepachangelog.com/en/1.0.0/&#41;,)

[//]: # (and this project adheres to [Semantic Versioning]&#40;https://semver.org/spec/v2.0.0.html&#41;.)

## [Unreleased]

### Added

- **Algebraic notation support for Zertz** - Convert between array coordinates and human-readable notation
  - `coordinate_to_algebraic(y, x, config)` - Convert (y, x) to notation like "D4"
  - `algebraic_to_coordinate(notation, config)` - Parse "D4" to (y, x) coordinates
  - Correctly handles hexagonal board layout where row numbers depend on both x and y
  - Case-insensitive parsing
  - Supports all board sizes: 37 rings (A-G), 49 rings (A-H), 61 rings (A-J, note 'I' is skipped)
  - Validates positions are within hexagonal bounds
  - Rust module: `src/games/zertz/notation.rs` (266 lines)
  - Tests: `src/games/zertz/notation_tests.rs` (13 comprehensive tests including roundtrip validation)
  - Python bindings: Available via `hiivelabs_mcts` module
  - Type stubs: Full documentation in `hiivelabs_mcts.pyi`

### Deprecated

- **`translate_state` Python function** - Use `transform_state` with `rot60_k=0, mirror=False, mirror_first=False` instead
  - The function now emits a `DeprecationWarning` when called
  - Will be removed in a future major version
  - Maintained for backward compatibility

### Changed

- **Module organization refactored** for clarity and consistency
  - Moved Zertz implementation into `src/games/zertz/` subfolder with proper submodules:
    - `mod.rs` - Main game implementation
    - `board.rs` - Board configuration and game modes
    - `logic.rs` - Core game rules and move generation
    - `canonicalization.rs` - State canonicalization and symmetry detection
    - `action_transform.rs` - Action transformation for testing
    - `notation.rs` - Algebraic notation conversion
    - `zobrist.rs` - Zobrist hashing (Zertz-specific, moved from generic)
    - `py_logic.rs` - Python bindings for stateless functions
    - `py_mcts.rs` - Python MCTS wrapper
  - Moved Tic-Tac-Toe into `src/games/tictactoe/` subfolder for consistency
  - All test modules extracted to separate files (`*_tests.rs`) co-located with source code
  - ZobristHasher moved from generic infrastructure to Zertz-specific module (each game implements hashing via `MCTSGame::hash_state`)

- **Canonicalization improvements**
  - Unified `transform_state` to handle both rotation/mirror AND translation
  - Removed separate `translate_state` function (now uses `transform_state` with `dy, dx` parameters)
  - Python wrapper `translate_state` maintained for backward compatibility (calls unified implementation)
  - More efficient implementation with fewer code paths

- **Type stubs updated**
  - `hiivelabs_mcts.pyi` now accurately reflects all API changes
  - Fixed `translate_state` signature to match actual implementation
  - Updated all type annotations for Python 3.7+ compatibility (`list` → `List`, `dict` → `Dict`, etc.)

### Fixed

- Corrected Python type stubs for `translate_state` function (parameters and return type)
- Fixed module documentation to reflect current architecture

### Technical Details

- All 114 tests passing (up from 92 in v0.5.0)
  - 13 new tests for algebraic notation (including roundtrip validation for all board sizes)
- No breaking changes to Python API
- Improved code organization and maintainability
- Clear separation between generic MCTS infrastructure and game-specific logic

## [0.6.0] - 2025-01-31

### Breaking Changes

- **Module renamed**: `hiivelabs-zertz-mcts` → `hiivelabs-mcts`
  - Python import changes from `import hiivelabs_zertz_mcts` to `import hiivelabs_mcts`
  - Python class renamed: `MCTSSearch` → `ZertzMCTS` for consistency
  - Crate name in `Cargo.toml` updated to `hiivelabs-mcts`

### Added

- **Generic MCTS core** via `MCTSGame` trait system
  - Trait-based architecture supporting any two-player zero-sum game
  - Zero-cost abstraction using Rust generics and monomorphization
  - Type-safe action representation per game
  - Game-specific heuristics, canonicalization, and forced move detection

- **Deterministic sequence collapsing** (formerly `skip_forced_moves`)
  - Automatically traverses forced moves (e.g., Zertz capture chains) to focus search on real decisions
  - Saves 2-3x iterations in Zertz endgames with long capture chains
  - Enabled via `enable_deterministic_collapse()` trait method
  - Safety depth limit (`MAX_DETERMINISTIC_DEPTH = 100`) prevents infinite loops
  - No feature flags - always compiled, controlled by trait
  - Comprehensive documentation with tree structure comparisons

- **Tic-Tac-Toe implementation** as minimal example game
  - Complete working example in `src/games/tictactoe.rs` (~300 lines)
  - Demonstrates all required trait methods
  - Python wrapper (`TicTacToeMCTS` class)
  - Full test coverage (7 tests)
  - Serves as template for adding new games

- **New module structure**:
  - `src/game_trait.rs` - `MCTSGame` trait definition
  - `src/games/mod.rs` - Game implementations module
  - `src/games/zertz.rs` - Zertz implementation of `MCTSGame`
  - `src/games/zertz_py.rs` - Python wrapper for Zertz
  - `src/games/tictactoe.rs` - Tic-Tac-Toe implementation
  - `src/games/tictactoe_py.rs` - Python wrapper for Tic-Tac-Toe

- **Enhanced documentation**:
  - Updated README with trait architecture explanation
  - Added "Adding a New Game" section with detailed steps
  - Comprehensive inline documentation with examples
  - Tree structure visualization for deterministic collapse feature

### Changed

- **MCTS core generified**:
  - `MCTSNode<G: MCTSGame>` replaces hardcoded Zertz types
  - `MCTSSearch<G: MCTSGame>` is now generic over game type
  - `TranspositionTable<G>` supports any game implementing the trait

- **Action representation**:
  - Actions are now `G::Action` (game-specific associated type)
  - Zertz uses `ZertzAction` enum
  - Tic-Tac-Toe uses `TicTacToeAction` struct

### Removed

- **Feature flag removed**: `skip_forced_moves` - replaced with trait-based control
  - Runtime behavior controlled by `enable_deterministic_collapse()` method
  - No compile-time feature flags needed

### Migration Guide

**For Python users:**

```python
# Old import
import hiivelabs_zertz_mcts
mcts = hiivelabs_zertz_mcts.MCTSSearch(rings=37)

# New import
import hiivelabs_mcts
mcts = hiivelabs_mcts.ZertzMCTS(rings=37)  # Note: class renamed
```

**For Rust users:**

The crate is now structured for multi-game support:

```rust
// Use Zertz
use hiivelabs_mcts::games::ZertzGame;
use hiivelabs_mcts::mcts::MCTSSearch;

let game = Arc::new(ZertzGame::new(37, 1, false)?);
let mcts = MCTSSearch::new(game, None, None, None, None, None, None);

// Use Tic-Tac-Toe
use hiivelabs_mcts::games::TicTacToeGame;

let game = Arc::new(TicTacToeGame::new());
let mcts = MCTSSearch::new(game, None, None, None, None, None, None);
```

### Technical Details

- All 92 tests passing (85 existing + 7 new Tic-Tac-Toe tests)
- No performance regression from generification (verified by test suite)
- Maintains thread-safety with atomic operations and `DashMap`
- Compatible with Python 3.8+ via PyO3 ABI3 wheels

## [0.5.2] - Previous Release

See git history for changes prior to the trait abstraction refactoring.
