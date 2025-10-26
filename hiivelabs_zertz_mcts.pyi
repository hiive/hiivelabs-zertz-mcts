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
        widening_constant: Optional[float] = None,
        fpu_reduction: Optional[float] = None,
        use_transposition_table: Optional[bool] = None,
        use_transposition_lookups: Optional[bool] = None
    ) -> None:
        """
        Create a new MCTS search instance.

        Args:
            exploration_constant: UCB1 exploration parameter (default: 1.41, √2)
            widening_constant: Progressive widening constant. When None (default), all
                legal actions are explored. When set to a value (e.g., 10.0), limits
                child expansion: max_children = widening_constant * sqrt(visits + 1)
            fpu_reduction: First Play Urgency reduction parameter. When set, unvisited
                nodes are estimated as parent_value - fpu_reduction instead of having
                infinite urgency. Set to None to use standard UCB1 (default: None)
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

    def last_child_statistics(self) -> list[tuple[str, tuple[int, int, int] | None, float]]:
        """
        Get per-child statistics from last search as (action_type, action_data, normalized_score) tuples.

        Returns:
            List of tuples where each tuple contains:
            - action_type: "PUT", "CAP", or "PASS"
            - action_data: Action-specific tuple (depends on type), or None for PASS
            - normalized_score: Visit count normalized to [0.0, 1.0] range

            For PUT actions: action_data = (marble_type, dst_flat, remove_flat)
            For CAP actions: action_data = (direction, start_y, start_x)
            For PASS actions: action_data = None
        """
        ...


# ============================================================================
# Stateless Game Logic Functions
# ============================================================================

# Axial Coordinate Transformations
# ============================================================================

def ax_rot60(q: int, r: int, k: int) -> Tuple[int, int]:
    """
    Rotate axial coordinate by k * 60° counterclockwise.

    Args:
        q: Axial q coordinate
        r: Axial r coordinate
        k: Number of 60° rotations (will be normalized to 0-5)

    Returns:
        Tuple of (q, r) rotated coordinates

    Example:
        >>> ax_rot60(1, 0, 1)  # 60° rotation
        (0, 1)
        >>> ax_rot60(2, 3, 3)  # 180° rotation
        (-2, -1)
    """
    ...


def ax_mirror_q_axis(q: int, r: int) -> Tuple[int, int]:
    """
    Mirror axial coordinate across the q-axis.

    Args:
        q: Axial q coordinate
        r: Axial r coordinate

    Returns:
        Tuple of (q, r) mirrored coordinates

    Example:
        >>> ax_mirror_q_axis(1, 0)
        (1, -1)
    """
    ...


def build_axial_maps(
    config: BoardConfig,
    layout: list[list[bool]]
) -> Tuple[dict[Tuple[int, int], Tuple[int, int]], dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Build bidirectional maps between (y,x) and axial (q,r) coordinates.

    Converts all valid board positions to centered, scaled axial coordinates suitable
    for rotation and reflection transformations. Applies board-specific centering and
    scaling (48-ring boards use scale=3 for D3 symmetry).

    Args:
        config: BoardConfig specifying board size and layout
        layout: 2D boolean array indicating valid positions (width × width)

    Returns:
        Tuple of two dictionaries: (yx_to_ax, ax_to_yx)
            - yx_to_ax: Maps (y, x) tuples to (q, r) axial coordinates
            - ax_to_yx: Maps (q, r) axial coordinates to (y, x) tuples

    Example:
        >>> config = BoardConfig.standard_config(37)
        >>> layout = [[True, False], [False, True]]  # Example layout
        >>> yx_to_ax, ax_to_yx = build_axial_maps(config, layout)
        >>> yx_to_ax[(3, 3)]  # Center of 37-ring board
        (0, 0)
    """
    ...


# Isolation Capture
# ============================================================================

def check_for_isolation_capture(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    config: BoardConfig
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], list[tuple[int, int, int]]]:
    """
    Check for isolated regions and capture marbles (stateless).

    After a ring is removed, the board may split into multiple disconnected regions.
    If ALL rings in an isolated region are fully occupied (each has a marble),
    then the current player captures all those marbles and removes those rings.

    Args:
        spatial: 3D array of shape (L, H, W) containing board layers
        global_: 1D array containing supply and captured marble counts
        config: BoardConfig with game configuration

    Returns:
        Tuple of (updated_spatial, updated_global, captured_marbles_list):
        - updated_spatial: Updated spatial state after isolation capture
        - updated_global: Updated global state with incremented capture counts
        - captured_marbles_list: List of (marble_layer, y, x) tuples for captured marbles
    """
    ...


class BoardConfig:
    """Board configuration for stateless game logic.

    Contains all constants and indices needed for stateless game operations.
    """

    @staticmethod
    def standard_config(rings: int, *, t: int = 1) -> BoardConfig:
        """
        Create a standard BoardConfig for the given number of rings.

        Args:
            rings: Number of rings (37, 48, or 61)
            t: Time history depth (default: 1)

        Returns:
            BoardConfig instance

        Raises:
            ValueError: If rings is not 37, 48, or 61
        """
        ...

    @staticmethod
    def blitz_config(rings: int, *, t: int = 1) -> BoardConfig:
        """
        Create a blitz mode BoardConfig for the given number of rings.

        Args:
            rings: Number of rings (37, 48, or 61)
            t: Time history depth (default: 1)

        Returns:
            BoardConfig instance with blitz mode enabled

        Raises:
            ValueError: If rings is not 37, 48, or 61
        """
        ...

    def is_blitz(self) -> bool:
        """Check if this config is for blitz mode."""
        ...

    def get_directions(self) -> list[tuple[int, int]]:
        """Get the 6 hexagonal direction vectors."""
        ...

    def get_marble_layer(self, marble_type: str) -> int:
        """
        Get layer index for marble type.

        Args:
            marble_type: 'w', 'g', or 'b'

        Returns:
            Layer index (1, 2, or 3)
        """
        ...

    # Board dimensions
    @property
    def rings(self) -> int:
        """Number of rings on the board (37, 48, or 61)."""
        ...

    @property
    def width(self) -> int:
        """Board grid width (7, 8, or 9)."""
        ...

    @property
    def t(self) -> int:
        """Time history depth."""
        ...

    @property
    def layers_per_timestep(self) -> int:
        """Number of layers per timestep (always 4)."""
        ...

    # Layer indices
    @property
    def ring_layer(self) -> int:
        """Ring layer index (0)."""
        ...

    @property
    def capture_layer(self) -> int:
        """Capture layer index (4)."""
        ...

    # Player constants
    @property
    def player_1(self) -> int:
        """Player 1 index (0)."""
        ...

    @property
    def player_2(self) -> int:
        """Player 2 index (1)."""
        ...

    # Global state indices - Supply
    @property
    def supply_w(self) -> int:
        """White marble supply index (0)."""
        ...

    @property
    def supply_g(self) -> int:
        """Grey marble supply index (1)."""
        ...

    @property
    def supply_b(self) -> int:
        """Black marble supply index (2)."""
        ...

    # Global state indices - Player 1 captures
    @property
    def p1_cap_w(self) -> int:
        """Player 1 white captures index (3)."""
        ...

    @property
    def p1_cap_g(self) -> int:
        """Player 1 grey captures index (4)."""
        ...

    @property
    def p1_cap_b(self) -> int:
        """Player 1 black captures index (5)."""
        ...

    @property
    def p1_cap_slice(self) -> slice:
        """Player 1 capture slice (indices 3-6: w, g, b)."""
        ...

    # Global state indices - Player 2 captures
    @property
    def p2_cap_w(self) -> int:
        """Player 2 white captures index (6)."""
        ...

    @property
    def p2_cap_g(self) -> int:
        """Player 2 grey captures index (7)."""
        ...

    @property
    def p2_cap_b(self) -> int:
        """Player 2 black captures index (8)."""
        ...

    @property
    def p2_cap_slice(self) -> slice:
        """Player 2 capture slice (indices 6-9: w, g, b)."""
        ...

    # Current player index
    @property
    def cur_player(self) -> int:
        """Current player index in global state (9)."""
        ...


# Module constants
PLAYER_1: int  # = 0
PLAYER_2: int  # = 1


# ============================================================================
# Additional Stateless Game Logic Functions
# ============================================================================

def is_inbounds(y: int, x: int, width: int) -> bool:
    """
    Check if (y, x) coordinates are within board bounds.

    Args:
        y: Y coordinate
        x: X coordinate
        width: Board width

    Returns:
        True if coordinates are in bounds, False otherwise
    """
    ...


def get_neighbors(y: int, x: int, config: BoardConfig) -> list[tuple[int, int]]:
    """
    Get list of neighboring indices (filtered to in-bounds only).

    Args:
        y: Y coordinate
        x: X coordinate
        config: BoardConfig

    Returns:
        List of (y, x) neighbor coordinate tuples
    """
    ...


def get_jump_destination(start_y: int, start_x: int, cap_y: int, cap_x: int) -> tuple[int, int]:
    """
    Calculate landing position after capturing marble.

    Args:
        start_y: Starting Y coordinate
        start_x: Starting X coordinate
        cap_y: Captured marble Y coordinate
        cap_x: Captured marble X coordinate

    Returns:
        Tuple of (dst_y, dst_x) landing coordinates
    """
    ...


def get_regions(spatial: npt.NDArray[np.float32], config: BoardConfig) -> list[list[tuple[int, int]]]:
    """
    Find all connected regions on the board.

    Args:
        spatial: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig

    Returns:
        List of regions, where each region is a list of (y, x) indices
    """
    ...


def get_open_rings(spatial: npt.NDArray[np.float32], config: BoardConfig) -> list[tuple[int, int]]:
    """
    Get list of empty ring indices across the entire board.

    Args:
        spatial: Board state array
        config: BoardConfig

    Returns:
        List of (y, x) indices of empty rings
    """
    ...


def is_ring_removable(spatial: npt.NDArray[np.float32], y: int, x: int, config: BoardConfig) -> bool:
    """
    Check if ring at index can be removed.

    A ring is removable if:
    1. It's empty (no marble)
    2. Two consecutive neighbors are missing

    Args:
        spatial: Board state array
        y: Y coordinate
        x: X coordinate
        config: BoardConfig

    Returns:
        True if ring can be removed, False otherwise
    """
    ...


def get_removable_rings(spatial: npt.NDArray[np.float32], config: BoardConfig) -> list[tuple[int, int]]:
    """
    Get list of removable ring indices.

    Args:
        spatial: Board state array
        config: BoardConfig

    Returns:
        List of (y, x) indices of removable rings
    """
    ...


def get_supply_index(marble_type: str) -> int:
    """
    Get global_state index for marble in supply.

    Args:
        marble_type: Marble type ('w', 'g', or 'b')

    Returns:
        Index in global_state array
    """
    ...


def get_captured_index(player: int, marble_type: str) -> int:
    """
    Get global_state index for captured marble for given player.

    Args:
        player: Player index (0 or 1)
        marble_type: Marble type ('w', 'g', or 'b')

    Returns:
        Index in global_state array
    """
    ...


def get_marble_type_at(spatial: npt.NDArray[np.float32], y: int, x: int) -> str:
    """
    Get marble type at given position.

    Args:
        spatial: (L, H, W) spatial state array
        y: Y coordinate
        x: X coordinate

    Returns:
        Marble type ('w', 'g', 'b', or '' if no marble)
    """
    ...


def get_placement_moves(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    config: BoardConfig
) -> npt.NDArray[np.float32]:
    """
    Get valid placement moves as boolean array.

    Args:
        spatial: (L, H, W) spatial state array
        global_: (10,) global state array
        config: BoardConfig

    Returns:
        Boolean array of shape (3, width², width² + 1)
    """
    ...


def get_capture_moves(
    spatial: npt.NDArray[np.float32],
    config: BoardConfig
) -> npt.NDArray[np.float32]:
    """
    Get valid capture moves as boolean array.

    Args:
        spatial: (L, H, W) spatial state array
        config: BoardConfig

    Returns:
        Boolean array of shape (6, width, width)
    """
    ...


def get_valid_actions(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    config: BoardConfig
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Get valid actions for current state.

    Args:
        spatial: (L, H, W) spatial state array
        global_: (10,) global state array
        config: BoardConfig

    Returns:
        (placement_mask, capture_mask) tuple
        - placement_mask: (3, width², width² + 1) boolean array
        - capture_mask: (6, width, width) boolean array

    Note: If any captures are available, placement_mask will be all zeros
          (captures are mandatory in Zertz rules)
    """
    ...


def apply_placement_action(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    marble_type: int,
    dst_y: int,
    dst_x: int,
    remove_y: Optional[int],
    remove_x: Optional[int],
    config: BoardConfig
) -> None:
    """
    Apply placement action to state IN-PLACE.

    Args:
        spatial: (L, H, W) spatial state array (MUTATED IN-PLACE)
        global_: (10,) global state array (MUTATED IN-PLACE)
        marble_type: Marble type index (0=white, 1=gray, 2=black)
        dst_y: Destination Y coordinate
        dst_x: Destination X coordinate
        remove_y: Ring to remove Y coordinate (or None)
        remove_x: Ring to remove X coordinate (or None)
        config: BoardConfig
    """
    ...


def apply_capture_action(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    start_y: int,
    start_x: int,
    direction: int,
    config: BoardConfig
) -> None:
    """
    Apply capture action to state IN-PLACE.

    Args:
        spatial: (L, H, W) spatial state array (MUTATED IN-PLACE)
        global_: (10,) global state array (MUTATED IN-PLACE)
        start_y: Starting Y coordinate
        start_x: Starting X coordinate
        direction: Direction index (0-5 for 6 hexagonal directions)
        config: BoardConfig
    """
    ...


def is_game_over(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    config: BoardConfig
) -> bool:
    """
    Check if game has ended (stateless version).

    Args:
        spatial: (L, H, W) spatial state array
        global_: (10,) global state array
        config: BoardConfig

    Returns:
        True if game is over, False otherwise
    """
    ...


def get_game_outcome(
    spatial: npt.NDArray[np.float32],
    global_: npt.NDArray[np.float32],
    config: BoardConfig
) -> int:
    """
    Determine game outcome from terminal state (stateless version).

    Args:
        spatial: (L, H, W) spatial state array
        global_: (10,) global state array
        config: BoardConfig

    Returns:
        1 if Player 1 wins, -1 if Player 2 wins, 0 for tie, -2 for both lose
    """
    ...