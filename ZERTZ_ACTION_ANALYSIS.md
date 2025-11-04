# ZertzAction Usage Analysis

## Executive Summary

The codebase has **inconsistent action representation**: some functions use `ZertzAction` objects while others use tuples/arrays. The Python stub file (`.pyi`) is also **out of sync** with the actual implementation.

## Current State

### ZertzAction Definition

**Location**: `src/games/zertz/action.rs`

```rust
pub enum ZertzAction {
    Placement {
        marble_type: usize,
        dst_flat: usize,
        remove_flat: Option<usize>,
    },
    Capture {
        src_flat: usize,
        dst_flat: usize,
    },
    Pass,
}
```

**Python wrapper**: `PyZertzAction` exposes this to Python with static methods:
- `ZertzAction.placement(config, marble_type, dst_y, dst_x, remove_y, remove_x)`
- `ZertzAction.capture(config, src_y, src_x, dst_y, dst_x)`
- `ZertzAction.pass_action()`

### Where ZertzAction IS Used ✅

1. **`apply_action()`** - `src/games/zertz/py_logic.rs:452`
   - **Accepts**: `PyZertzAction` object
   - **Good**: Type-safe, clean interface

2. **`search()` and `search_parallel()` return values** - `src/games/zertz/py_mcts.rs:178, 309`
   - **Returns**: `PyZertzAction`
   - **Good**: Type-safe return value

3. **Internal MCTS implementation**
   - The generic MCTS engine works with `ZertzAction` internally
   - Actions are stored as `ZertzAction` in the search tree

### Where Tuples/Arrays ARE Used ❌

1. **`apply_placement_action()`** - `src/games/zertz/py_logic.rs:532`
   - **Accepts**: Individual parameters `(config, spatial_state, global_state, marble_type, dst_y, dst_x, remove_y, remove_x)`
   - **Should be**: `(config, spatial_state, global_state, action: PyZertzAction)`

2. **`apply_capture_action()`** - `src/games/zertz/py_logic.rs:562`
   - **Accepts**: Individual parameters `(config, spatial_state, global_state, start_y, start_x, dest_y, dest_x)`
   - **Should be**: `(config, spatial_state, global_state, action: PyZertzAction)`

3. **`BoardState.apply_placement()`** - `src/games/zertz/board.rs:346`
   - **Accepts**: Individual parameters `(marble_type, dst_y, dst_x, remove_y, remove_x)`
   - **Should be**: `(action: PyZertzAction)`

4. **`BoardState.apply_capture()`** - `src/games/zertz/board.rs:378`
   - **Accepts**: Individual parameters `(start_y, start_x, dest_y, dest_x)`
   - **Should be**: `(action: PyZertzAction)`

5. **`last_child_statistics()`** - `src/games/zertz/py_mcts.rs:113`
   - **Returns**: `Vec<(String, Option<(usize, usize, usize)>, f32)>`
   - **Should be**: `Vec<(PyZertzAction, f32)>` or keep tuples for simplicity

6. **`transform_action()`** - `src/games/zertz/py_logic.rs:735`
   - **Accepts**: `(config, action_type: &str, action_data: &PyTuple, transform: &str)`
   - **Returns**: `(String, Vec<Option<usize>>)`
   - **Should accept**: `(config, action: PyZertzAction, transform: &str)`
   - **Should return**: `PyZertzAction`

### Critical Issue: Stub File Mismatch

**File**: `hiivelabs_mcts.pyi`

The stub file is **out of sync** with the implementation:

**Line 266** - `search()` signature:
```python
# STUB SAYS (WRONG):
def search(...) -> Tuple[str, Optional[Tuple[int, int, int]]]:

# ACTUAL RUST CODE (CORRECT):
fn search(...) -> PyResult<PyZertzAction>
```

**Line 313** - `search_parallel()` signature:
```python
# STUB SAYS (WRONG):
def search_parallel(...) -> Tuple[str, Optional[Tuple[int, int, int]]]:

# ACTUAL RUST CODE (CORRECT):
fn search_parallel(...) -> PyResult<PyZertzAction>
```

**Line 378** - `last_child_statistics()` signature:
```python
# STUB SAYS:
def last_child_statistics(self) -> List[Tuple[str, Optional[Tuple[int, int, int]], float]]:

# ACTUAL RUST CODE (MATCHES):
fn last_child_statistics(&self) -> Vec<(String, Option<(usize, usize, usize)>, f32)>
```

## Recommendations

### Priority 1: Fix Stub File (CRITICAL)

**Update `hiivelabs_mcts.pyi`** to match actual return types:

```python
# In class ZertzMCTS:

def search(
    self,
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    iterations: int,
    *,
    max_depth: Optional[int] = None,
    time_limit: Optional[float] = None,
    use_transposition_table: Optional[bool] = None,
    use_transposition_lookups: Optional[bool] = None,
    clear_table: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_interval_ms: int = 100
) -> ZertzAction:  # Changed from Tuple[str, Optional[Tuple[int, int, int]]]
    """..."""
    ...

def search_parallel(
    self,
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    iterations: int,
    *,
    max_depth: Optional[int] = None,
    time_limit: Optional[float] = None,
    use_transposition_table: Optional[bool] = None,
    use_transposition_lookups: Optional[bool] = None,
    clear_table: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_interval_ms: int = 100
) -> ZertzAction:  # Changed from Tuple[str, Optional[Tuple[int, int, int]]]
    """..."""
    ...
```

### Priority 2: Add ZertzAction Overloads (RECOMMENDED)

Instead of replacing the tuple-based functions, **add new overloaded versions** that accept `ZertzAction`:

1. **Add `apply_action()` to BoardState**:
   ```python
   # In class BoardState:
   def apply_action(self, action: ZertzAction) -> None:
       """Apply a ZertzAction to the board."""
       ...
   ```

2. **Keep backward compatibility**:
   - Keep existing `apply_placement()` and `apply_capture()` with individual parameters
   - Add new `apply_action()` that accepts `ZertzAction`

### Priority 3: Consider transform_action() Improvement (OPTIONAL)

**Current**:
```python
def transform_action(
    config: BoardConfig,
    action_type: str,
    action_data: Tuple,
    transform: str,
) -> Tuple[str, List[Optional[int]]]:
```

**Proposed**:
```python
def transform_action(
    config: BoardConfig,
    action: ZertzAction,
    transform: str,
) -> ZertzAction:
```

This would be cleaner but requires updating any Python code that calls it.

### Priority 4: last_child_statistics() (OPTIONAL)

**Option A** (minimal change): Keep current tuple format for simplicity

**Option B** (more type-safe):
```python
def last_child_statistics(self) -> List[Tuple[ZertzAction, float]]:
    """Returns list of (action, normalized_score) tuples."""
    ...
```

## Migration Path

### Phase 1: Fix Critical Issues (Do Now)
1. ✅ Update stub file to show `search()` returns `ZertzAction`
2. ✅ Update stub file to show `search_parallel()` returns `ZertzAction`
3. Add stub for `ZertzAction` class methods (already exists but verify completeness)

### Phase 2: Add Action-Based Overloads (Recommended)
1. Add `BoardState.apply_action(action: ZertzAction)` method
2. Add documentation showing both approaches
3. Deprecate (but don't remove) individual parameter methods

### Phase 3: Improve transform_action (Optional)
1. Add new `transform_action_v2(config, action, transform)` that uses ZertzAction
2. Mark old version as deprecated
3. Migrate callers over time

## Code Impact Analysis

### Functions Currently Using Tuples

| Function | File | Lines | Impact if Changed |
|----------|------|-------|-------------------|
| `apply_placement_action` | `py_logic.rs` | 532-558 | Low - mostly internal |
| `apply_capture_action` | `py_logic.rs` | 562-585 | Low - mostly internal |
| `BoardState.apply_placement` | `board.rs` | 346-375 | Medium - may have external callers |
| `BoardState.apply_capture` | `board.rs` | 378-404 | Medium - may have external callers |
| `transform_action` | `py_logic.rs` | 735-869 | High - need to check callers |
| `last_child_statistics` | `py_mcts.rs` | 113-145 | Medium - visualization code may use this |

### Functions Already Using ZertzAction ✅

| Function | File | Lines | Status |
|----------|------|-------|--------|
| `apply_action` | `py_logic.rs` | 452-527 | ✅ Good |
| `search` | `py_mcts.rs` | 163-238 | ✅ Good (stub wrong) |
| `search_parallel` | `py_mcts.rs` | 294-364 | ✅ Good (stub wrong) |

## Testing Checklist

After making changes:

- [ ] Update stub file signatures for `search()` and `search_parallel()`
- [ ] Verify type checking passes: `mypy` or `pyright`
- [ ] Run Rust tests: `cargo test`
- [ ] Test Python imports: `from hiivelabs_mcts.zertz import ZertzAction`
- [ ] Test action creation: `ZertzAction.placement(config, 0, 3, 3)`
- [ ] Test search returns ZertzAction: `action = mcts.search(...)`
- [ ] Test apply_action: `apply_action(config, spatial, global, action)`
- [ ] Verify IDE autocomplete shows correct types

## Conclusion

**Current state**: Mixed use of `ZertzAction` and tuples, with critical stub file mismatches.

**Immediate action needed**: Fix stub file to match actual `search()` return type.

**Long-term goal**: Standardize on `ZertzAction` for all action-related interfaces while maintaining backward compatibility where needed.
