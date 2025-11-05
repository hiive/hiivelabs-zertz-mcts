# Changelog

All notable changes to this project will be documented in this file.

[//]: # (The format is based on [Keep a Changelog]&#40;https://keepachangelog.com/en/1.0.0/&#41;,)

[//]: # (and this project adheres to [Semantic Versioning]&#40;https://semver.org/spec/v2.0.0.html&#41;.)

## [Unreleased]

### Added

- **ZertzAction-based API improvements** - Major refactoring to standardize on structured action types
  - All apply functions now accept `ZertzAction` instead of individual parameters
  - All apply functions now return `ZertzActionResult` with structured outcome data
  - New functions:
    - `apply_placement_action(config, spatial_state, global_state, action: ZertzAction) -> ZertzActionResult`
    - `apply_capture_action(config, spatial_state, global_state, action: ZertzAction) -> ZertzActionResult`
    - `transform_action(config, action: ZertzAction, transform: str) -> ZertzAction`
    - `BoardState.apply_placement(action: ZertzAction) -> ZertzActionResult`
    - `BoardState.apply_capture(action: ZertzAction) -> ZertzActionResult`
  - Benefits:
    - Improved type safety (structured types instead of tuples)
    - Consistent API across all action functions
    - Better error messages (validates action type at runtime)
    - Richer return values (capture actions now include which marble was captured)
    - DRY principle: BoardState methods are thin wrappers over module-level functions

- **Comprehensive unit tests** - Added 19 new Rust unit tests
  - Tests for `ZertzAction`: creation, cloning, equality, hashing (7 tests)
  - Tests for `ZertzActionResult`: creation, accessors, cloning, equality (12 tests)
  - Validates all trait implementations (Clone, PartialEq, Eq, Hash)
  - Verifies accessor methods return correct data for appropriate variants

### Changed

- **Type stub updates** for improved IDE support
  - Fixed `search()` and `search_parallel()` return types: now correctly typed as `ZertzAction`
  - Updated all apply function signatures to reflect `ZertzAction` and `ZertzActionResult` types
  - Added comprehensive docstrings with parameter descriptions and return value documentation

- **Code organization improvements**
  - Applied DRY principle: eliminated ~115 lines of duplicate code
  - BoardState methods now delegate to module-level functions (thin wrappers)
  - Single source of truth for action application logic
  - All coordinate conversions use BoardConfig helper methods
    - Replaced 10 manual `flat / width, flat % width` conversions with `config.flat_to_yx()`
    - Replaced 10 manual `y * width + x` conversions with `config.yx_to_flat()`
    - Improved maintainability and reduced error potential

### Deprecated

- **Old action functions** - Renamed with `_old` suffix and marked deprecated
  - `apply_placement_action_old()` - Use `apply_placement_action()` with `ZertzAction` instead
  - `apply_capture_action_old()` - Use `apply_capture_action()` with `ZertzAction` instead
  - `transform_action_old()` - Use `transform_action()` with `ZertzAction` instead
  - `BoardState.apply_placement_old()` - Use `apply_placement()` with `ZertzAction` instead
  - `BoardState.apply_capture_old()` - Use `apply_capture()` with `ZertzAction` instead
  - All deprecated functions maintained for backward compatibility

### Migration Guide

**Old API (deprecated):**
```python
from hiivelabs_mcts import zertz

# Old: Individual parameters
captures = zertz.apply_placement_action(
    config, spatial_state, global_state,
    marble_type=0, dst_y=3, dst_x=3,
    remove_y=2, remove_x=4
)  # Returns List[Tuple[int, int, int]]

# Old: Transform with tuples
action_type, action_data = zertz.transform_action(
    config, "PUT", (0, 10, 15), "R60"
)
```

**New API (recommended):**
```python
from hiivelabs_mcts import zertz

# New: Structured action type
action = zertz.ZertzAction.placement(config, 0, 3, 3, 2, 4)
result = zertz.apply_placement_action(config, spatial_state, global_state, action)

# Access structured result
if result.isolation_captures():
    for marble_layer, y, x in result.isolation_captures():
        print(f"Captured marble at ({y}, {x})")

# New: Transform with ZertzAction
transformed_action = zertz.transform_action(config, action, "R60")
```

**Benefits of migration:**
- Type safety: `ZertzAction` validates action structure at creation time
- Better error messages: Clear indication when wrong action variant is used
- Structured results: Access isolation captures and captured marbles through typed accessors
- Cleaner code: No need to track individual coordinate parameters

### Technical Details

- All 160+ tests passing (141 from previous release + 19 new tests)
- No breaking changes to existing API (old functions maintained as deprecated)
- Improved code maintainability with DRY principle applied
- Python bindings require Python runtime for testing (underlying Rust logic tested in `logic_tests.rs`)

### Added

- **TicTacToe canonicalization** - Full D4 dihedral group implementation for 3×3 board symmetries
  - Implements 8-fold symmetry reduction (4 rotations + 4 reflections)
  - `canonicalize_state()` finds lexicographically minimal representation among all transformations
  - Significant transposition table efficiency improvement (eliminates ~87.5% redundant positions)
  - Rust module: `src/games/tictactoe/canonicalization.rs` (240 lines)
  - Tests: `src/games/tictactoe/canonicalization_tests.rs` (27 comprehensive tests)
  - Covers: corner/edge equivalence, rotation/reflection, L-shapes, win patterns, idempotence
  - TicTacToe now serves as complete reference implementation for canonicalization

- **ZertzAction Python bindings** - Zertz action type now accessible from Python
  - `ZertzAction.placement(config, marble_type, dst_y, dst_x, remove_y, remove_x)` - Create placement action
  - `ZertzAction.capture(config, start_y, start_x, dest_y, dest_x)` - Create capture action
  - `ZertzAction.pass()` - Create pass action
  - `action.to_tuple(width)` - Convert to tuple format for serialization
  - `action.action_type()` - Get action type as string ("Placement", "Capture", "Pass")
  - Actions are hashable, comparable, and have readable `__repr__`
  - Python class name: `ZertzAction` (Rust: `PyZertzAction`)

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

- All 141 tests passing (up from 114 in previous version)
  - 27 new TicTacToe canonicalization tests (corner/edge equivalence, rotations, L-shapes, etc.)
  - 13 algebraic notation tests from previous release
- No breaking changes to Python API (internal refactorings only)
- Improved code organization and maintainability
- Clear separation between generic MCTS infrastructure and game-specific logic
- TicTacToe now serves as complete reference implementation for:
  - MCTSGame trait implementation
  - D4 dihedral group canonicalization
  - Comprehensive testing patterns

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
