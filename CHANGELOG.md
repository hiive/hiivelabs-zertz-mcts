# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
