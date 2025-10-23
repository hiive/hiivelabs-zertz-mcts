"""
Type stubs for hiivelabs_zertz_mcts Rust extension module.

This file provides type hints for IDE autocomplete and static type checking.
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

class BoardState:
    """
    Rust-backed board state representation for testing and comparison.

    Wraps spatial and global state arrays with board configuration.
    """

    def __init__(
        self,
        spatial: npt.NDArray[np.float32],
        global_: npt.NDArray[np.float32],
        rings: int,
        *,
        t: int = 1,
        blitz: bool = False
    ) -> None:
        """
        Create a new board state.

        Args:
            spatial: 3D array of shape (L, H, W) containing board layers
            global_: 1D array containing supply and captured marble counts
            rings: Number of rings on the board (37, 48, or 61)
            t: Transposition table size parameter (default: 1)
            blitz: Use blitz mode rules (default: False)

        Raises:
            ValueError: If rings is not 37, 48, or 61
        """
        ...

    def clone_state(self) -> BoardState:
        """
        Clone the board state for MCTS simulation.

        Returns:
            A new BoardState with copied arrays
        """
        ...

    def get_valid_actions(
        self
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Get valid actions for the current board state.

        Returns:
            Tuple of (placement_mask, capture_mask):
            - placement_mask: (3, width², width²+1) array of valid placement actions
            - capture_mask: (6, width, width) array of valid capture actions
        """
        ...

    def canonicalize_state(
        self
    ) -> Tuple[npt.NDArray[np.float32], str, str]:
        """
        Canonicalize the spatial state and return transform metadata.

        Returns:
            Tuple of (canonical_spatial, transform, inverse):
            - canonical_spatial: Canonicalized spatial array
            - transform: String describing the applied transform (e.g., "R60", "MR120")
            - inverse: String describing the inverse transform
        """
        ...

    def apply_placement(
        self,
        marble_type: int,
        dst_y: int,
        dst_x: int,
        remove_y: Optional[int] = None,
        remove_x: Optional[int] = None
    ) -> None:
        """
        Apply a placement action to the board (mutating).

        Args:
            marble_type: Marble type (0=white, 1=gray, 2=black)
            dst_y: Destination Y coordinate
            dst_x: Destination X coordinate
            remove_y: Ring to remove Y coordinate (optional)
            remove_x: Ring to remove X coordinate (optional)
        """
        ...

    def apply_capture(
        self,
        start_y: int,
        start_x: int,
        direction: int
    ) -> None:
        """
        Apply a capture action to the board (mutating).

        Args:
            start_y: Starting Y coordinate
            start_x: Starting X coordinate
            direction: Direction index (0-5 for 6 hexagonal directions)
        """
        ...

    def get_spatial(self) -> npt.NDArray[np.float32]:
        """
        Get the spatial state array.

        Returns:
            3D array of shape (L, H, W) containing board layers
        """
        ...

    def get_global(self) -> npt.NDArray[np.float32]:
        """
        Get the global state array.

        Returns:
            1D array containing supply and captured marble counts
        """
        ...


class MCTSSearch:
    """
    Monte Carlo Tree Search implementation for Zertz.

    Provides both serial and parallel search methods with configurable parameters.
    """

    def __init__(
        self,
        *,
        exploration_constant: Optional[float] = None,
        progressive_widening: Optional[bool] = None,
        widening_constant: Optional[float] = None,
        use_transposition_table: Optional[bool] = None,
        use_transposition_lookups: Optional[bool] = None
    ) -> None:
        """
        Create a new MCTS search instance.

        Args:
            exploration_constant: UCB1 exploration parameter (default: 1.41)
            progressive_widening: Enable progressive widening (default: True)
            widening_constant: Progressive widening constant (default: 10.0)
            use_transposition_table: Enable transposition table (default: True)
            use_transposition_lookups: Enable transposition lookups (default: True)
        """
        ...

    def set_transposition_table_enabled(self, enabled: bool) -> None:
        """
        Enable/disable transposition table caching (persists across searches).

        Args:
            enabled: Whether to enable transposition table
        """
        ...

    def set_transposition_lookups(self, enabled: bool) -> None:
        """
        Enable/disable transposition lookups for initializing child nodes.

        Args:
            enabled: Whether to enable transposition lookups
        """
        ...

    def clear_transposition_table(self) -> None:
        """Clear any cached transposition data."""
        ...

    def set_seed(self, seed: Optional[int] = None) -> None:
        """
        Set deterministic RNG seed.

        Args:
            seed: Random seed (pass None to restore system randomness)
        """
        ...

    def search(
        self,
        spatial: npt.NDArray[np.float32],
        global_: npt.NDArray[np.float32],
        rings: int,
        iterations: int,
        *,
        t: int = 1,
        max_depth: Optional[int] = None,
        time_limit: Optional[float] = None,
        use_transposition_table: Optional[bool] = None,
        use_transposition_lookups: Optional[bool] = None,
        clear_table: bool = False,
        verbose: bool = False,
        seed: Optional[int] = None,
        blitz: bool = False
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Run MCTS search (serial mode).

        Args:
            spatial: 3D array of shape (L, H, W) containing board layers
            global_: 1D array containing supply and captured marble counts
            rings: Number of rings on the board (37, 48, or 61)
            iterations: Number of MCTS iterations to run
            t: Transposition table size parameter (default: 1)
            max_depth: Maximum simulation depth (default: unlimited)
            time_limit: Time limit in seconds (default: no limit)
            use_transposition_table: Override default transposition table setting
            use_transposition_lookups: Override default transposition lookup setting
            clear_table: Clear transposition table before search (default: False)
            verbose: Print search statistics (default: False)
            seed: Random seed for this search
            blitz: Use blitz mode rules (default: False)

        Returns:
            Tuple of (action_type, action_tuple):
            - action_type: "PUT", "CAP", or "PASS"
            - action_tuple: Depends on action_type:
                - PUT: (marble_type, dst_flat, remove_flat)
                - CAP: (direction, start_y, start_x)
                - PASS: None

        Raises:
            ValueError: If rings is not 37, 48, or 61
        """
        ...

    def search_parallel(
        self,
        spatial: npt.NDArray[np.float32],
        global_: npt.NDArray[np.float32],
        rings: int,
        iterations: int,
        *,
        t: int = 1,
        max_depth: Optional[int] = None,
        time_limit: Optional[float] = None,
        use_transposition_table: Optional[bool] = None,
        use_transposition_lookups: Optional[bool] = None,
        clear_table: bool = False,
        num_threads: int = 16,
        verbose: bool = False,
        seed: Optional[int] = None,
        blitz: bool = False
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Run MCTS search (parallel mode using Rayon).

        Args:
            spatial: 3D array of shape (L, H, W) containing board layers
            global_: 1D array containing supply and captured marble counts
            rings: Number of rings on the board (37, 48, or 61)
            iterations: Number of MCTS iterations to run
            t: Transposition table size parameter (default: 1)
            max_depth: Maximum simulation depth (default: unlimited)
            time_limit: Time limit in seconds (default: no limit)
            use_transposition_table: Override default transposition table setting
            use_transposition_lookups: Override default transposition lookup setting
            clear_table: Clear transposition table before search (default: False)
            num_threads: Number of threads to use (default: 16)
            verbose: Print search statistics (default: False)
            seed: Random seed for this search
            blitz: Use blitz mode rules (default: False)

        Returns:
            Tuple of (action_type, action_tuple):
            - action_type: "PUT", "CAP", or "PASS"
            - action_tuple: Depends on action_type:
                - PUT: (marble_type, dst_flat, remove_flat)
                - CAP: (direction, start_y, start_x)
                - PASS: None

        Raises:
            ValueError: If rings is not 37, 48, or 61
            RuntimeError: If thread pool creation fails
        """
        ...

    def last_root_children(self) -> int:
        """
        Get number of children expanded at root node from last search.

        Returns:
            Number of children at root
        """
        ...

    def last_root_visits(self) -> int:
        """
        Get number of visits to root node from last search.

        Returns:
            Number of visits to root
        """
        ...

    def last_root_value(self) -> float:
        """
        Get average value of root node from last search.

        Returns:
            Average value (from current player's perspective)
        """
        ...