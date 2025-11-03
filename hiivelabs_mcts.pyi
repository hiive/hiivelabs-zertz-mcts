"""
Type stubs for hiivelabs_mcts Rust extension module.

Generic MCTS engine with game-specific implementations.
Currently supports: Zertz

This file provides type hints for IDE autocomplete and static type checking.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

class BoardState:
    """
    Rust-backed board state representation for testing and comparison.

    Wraps spatial_state and global_state state arrays with board configuration.
    """

    def __init__(
        self,
        spatial_state: npt.NDArray[np.float32],
        global_state: npt.NDArray[np.float32],
        rings: int,
        *,
        t: int = 1,
        blitz: bool = False
    ) -> None:
        """
        Create a new board state.

        Args:
            spatial_state: 3D array of shape (L, H, W) containing board layers
            global_state: 1D array containing supply and captured marble counts
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
        Canonicalize the spatial_state state and return transform metadata.

        Returns:
            Tuple of (canonical_spatial_state, transform, inverse):
            - canonical_spatial_state: Canonicalized spatial_state array
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

    def get_spatial_state(self) -> npt.NDArray[np.float32]:
        """
        Get the spatial_state state array.

        Returns:
            3D array of shape (L, H, W) containing board layers
        """
        ...

    def get_global_state(self) -> npt.NDArray[np.float32]:
        """
        Get the global_state state array.

        Returns:
            1D array containing supply and captured marble counts
        """
        ...


class ZertzMCTS:
    """
    Monte Carlo Tree Search implementation for Zertz.

    Provides both serial and parallel search methods with configurable parameters.
    """

    def __init__(
        self,
        rings: int,
        *,
        exploration_constant: Optional[float] = None,
        widening_constant: Optional[float] = None,
        fpu_reduction: Optional[float] = None,
        rave_constant: Optional[float] = None,
        use_transposition_table: bool = True,
        use_transposition_lookups: bool = True,
        blitz: bool = False,
        t: int = 1
    ) -> None:
        """
        Create a new MCTS search instance for a specific Zertz game configuration.

        Args:
            rings: Number of rings on the board (37, 48, or 61)
            exploration_constant: UCB1 exploration parameter (default: 1.41, √2)
            widening_constant: Progressive widening constant. When None (default), all
                legal actions are explored. When set to a value (e.g., 10.0), limits
                child expansion: max_children = widening_constant * sqrt(visits + 1)
            fpu_reduction: First Play Urgency reduction parameter. When set, unvisited
                nodes are estimated as parent_value - fpu_reduction instead of having
                infinite urgency. Set to None to use standard UCB1 (default: None)
            rave_constant: RAVE (Rapid Action Value Estimation) constant. When set,
                enables RAVE-UCB scoring which combines UCB1 with "all-moves-as-first"
                (AMAF) statistics. The constant k controls how quickly RAVE influence
                decays: β = sqrt(k / (3N + k)) where N is node visits.
                Recommended values: 300-3000 (lower = faster decay to UCB1, higher =
                longer RAVE influence). Typically improves performance by 15-25%.
                When None (default), standard UCB1 is used.
            use_transposition_table: Enable transposition table (default: True)
            use_transposition_lookups: Enable transposition lookups (default: True)
            blitz: Use blitz mode rules (default: False)
            t: Time history depth (default: 1)

        Raises:
            ValueError: If rings is not 37, 48, or 61
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
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Run MCTS search (serial mode).

        Args:
            spatial_state: 3D array of shape (L, H, W) containing board layers
            global_state: 1D array containing supply and captured marble counts
            iterations: Number of MCTS iterations to run
            max_depth: Maximum simulation depth (default: unlimited)
            time_limit: Time limit in seconds (default: no limit)
            use_transposition_table: Override default transposition table setting
            use_transposition_lookups: Override default transposition lookup setting
            clear_table: Clear transposition table before search (default: False)
            verbose: Print search statistics (default: False)
            seed: Random seed for this search
            progress_callback: Optional callback function for progress events.
                Receives dict with keys depending on event type:
                - SearchStarted: {"event": "SearchStarted", "root_visits": int, "root_value": float}
                - IterationProgress: {"event": "IterationProgress", "iteration": int, "root_visits": int, "root_value": float, "elapsed_ms": int}
                - SearchEnded: {"event": "SearchEnded", "total_iterations": int, "total_time_ms": int}
            progress_interval_ms: Minimum milliseconds between IterationProgress callbacks (default: 100)

        Returns:
            Tuple of (action_type, action_tuple):
            - action_type: "PUT", "CAP", or "PASS"
            - action_tuple: Depends on action_type:
                - PUT: (marble_type, dst_flat, remove_flat)
                - CAP: (direction, start_y, start_x)
                - PASS: None
        """
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
    ) -> Tuple[str, Optional[Tuple[int, int, int]]]:
        """
        Run MCTS search (parallel mode using Rayon).

        Uses Rayon's thread pool to run iterations in parallel. Thread count is
        automatically managed by Rayon based on system capabilities.

        Args:
            spatial_state: 3D array of shape (L, H, W) containing board layers
            global_state: 1D array containing supply and captured marble counts
            iterations: Number of MCTS iterations to run
            max_depth: Maximum simulation depth (default: unlimited)
            time_limit: Time limit in seconds (default: no limit)
            use_transposition_table: Override default transposition table setting
            use_transposition_lookups: Override default transposition lookup setting
            clear_table: Clear transposition table before search (default: False)
            verbose: Print search statistics (default: False)
            seed: Random seed for this search
            progress_callback: Optional callback function for progress events.
                Note: In parallel mode, progress callbacks are best-effort due to
                thread safety limitations. Only SearchStarted and SearchEnded events
                are reliably fired. IterationProgress events may be skipped.
            progress_interval_ms: Minimum milliseconds between IterationProgress callbacks (default: 100)

        Returns:
            Tuple of (action_type, action_tuple):
            - action_type: "PUT", "CAP", or "PASS"
            - action_tuple: Depends on action_type:
                - PUT: (marble_type, dst_flat, remove_flat)
                - CAP: (direction, start_y, start_x)
                - PASS: None

        Note:
            Parallel search uses Rayon for thread management. The number of threads
            is automatically determined by Rayon based on available CPU cores.
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

    def last_child_statistics(self) -> List[Tuple[str, Optional[Tuple[int, int, int]], float]]:
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


class ZertzAction:
    """
    Zertz action representation for creating and manipulating game actions.

    Actions can be one of three types:
    - Placement: Place a marble and optionally remove a ring
    - Capture: Jump over a marble to capture it
    - Pass: No legal moves available
    """

    @staticmethod
    def placement(
        config: BoardConfig,
        marble_type: int,
        dst_y: int,
        dst_x: int,
        remove_y: Optional[int] = None,
        remove_x: Optional[int] = None
    ) -> ZertzAction:
        """
        Create a Placement action.

        Args:
            config: BoardConfig for coordinate conversion
            marble_type: Marble type (0=white, 1=gray, 2=black)
            dst_y: Destination row
            dst_x: Destination column
            remove_y: Optional row of ring to remove
            remove_x: Optional column of ring to remove

        Returns:
            ZertzAction instance representing a placement
        """
        ...

    @staticmethod
    def capture(
        config: BoardConfig,
        start_y: int,
        start_x: int,
        dest_y: int,
        dest_x: int
    ) -> ZertzAction:
        """
        Create a Capture action.

        Args:
            config: BoardConfig for coordinate conversion
            start_y: Starting row
            start_x: Starting column
            dest_y: Destination row (after jumping)
            dest_x: Destination column (after jumping)

        Returns:
            ZertzAction instance representing a capture
        """
        ...

    @staticmethod
    def pass_action() -> ZertzAction:
        """
        Create a Pass action.

        Returns:
            ZertzAction instance representing a pass
        """
        ...

    def to_tuple(self, width: int) -> Tuple[str, Optional[Tuple[Optional[int], int, int]]]:
        """
        Convert action to tuple format for serialization.

        Args:
            width: Board width for coordinate handling

        Returns:
            Tuple of (action_type, action_data):
            - For Placement: ("PUT", Some((marble_type, dst_flat, remove_flat)))
            - For Capture: ("CAP", Some((None, src_flat, dst_flat)))
            - For Pass: ("PASS", None)

            Where flat coordinates are calculated as: flat = y * width + x
        """
        ...

    def action_type(self) -> str:
        """
        Get action type as string.

        Returns:
            One of: "Placement", "Capture", "Pass"
        """
        ...

    def __repr__(self) -> str:
        """String representation of the action."""
        ...

    def __eq__(self, other: object) -> bool:
        """Check equality with another ZertzAction."""
        ...

    def __hash__(self) -> int:
        """Compute hash for use in sets/dicts."""
        ...


class TicTacToeMCTS:
    """
    Monte Carlo Tree Search implementation for Tic-Tac-Toe.

    A minimal example game demonstrating the MCTS trait implementation.
    Includes D4 dihedral group canonicalization for 8-fold symmetry reduction.
    """

    def __init__(
        self,
        *,
        exploration_constant: Optional[float] = None,
        widening_constant: Optional[float] = None,
        fpu_reduction: Optional[float] = None,
        rave_constant: Optional[float] = None,
        use_transposition_table: bool = True,
        use_transposition_lookups: bool = True
    ) -> None:
        """
        Create a new MCTS search instance for Tic-Tac-Toe.

        Args:
            exploration_constant: UCB1 exploration parameter (default: 1.41, √2)
            widening_constant: Progressive widening constant (default: None, all actions explored)
            fpu_reduction: First Play Urgency reduction parameter (default: None)
            rave_constant: RAVE constant for AMAF statistics (default: None)
            use_transposition_table: Enable transposition table (default: True)
            use_transposition_lookups: Enable transposition lookups (default: True)
        """
        ...

    def search(
        self,
        spatial_state: npt.NDArray[np.float32],
        global_state: npt.NDArray[np.float32],
        iterations: int,
        *,
        max_depth: Optional[int] = None,
        time_limit: Optional[float] = None,
        verbose: bool = False,
        seed: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Run MCTS search for Tic-Tac-Toe.

        Args:
            spatial_state: 3D array of shape (2, 3, 3) - layers for X and O positions
            global_state: 1D array with current player (0=X, 1=O)
            iterations: Number of MCTS iterations to run
            max_depth: Maximum simulation depth (default: unlimited)
            time_limit: Time limit in seconds (default: no limit)
            verbose: Print search statistics (default: False)
            seed: Random seed for this search

        Returns:
            Tuple of (row, col) for the best move
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
    layout: List[List[bool]]
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]:
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


def canonicalize_state(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig,
    flags: Optional[TransformFlags] = None
) -> Tuple[npt.NDArray[np.float32], str, str]:
    """
    Canonicalize a board state to its standard form.

    Finds the lexicographically minimal representation of the board state
    under selected symmetry transforms (rotations, reflections, translations).
    This is the core function for exploiting board symmetries in transposition tables.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig specifying board size and symmetry group
        flags: TransformFlags specifying which transforms to include (default: ALL)

    Returns:
        Tuple of (canonical_spatial_state, transform_name, inverse_transform_name):
            - canonical_spatial_state: Canonicalized spatial_state array
            - transform_name: String describing the applied transform (e.g., "R60", "MR120", "T1,0_R180")
            - inverse_transform_name: String describing the inverse transform

    Example:
        >>> config = BoardConfig.standard_config(37)
        >>> spatial_state = np.zeros((5, 7, 7), dtype=np.float32)
        >>> # Use all transforms (default)
        >>> canonical, transform, inverse = canonicalize_state(spatial_state, config)
        >>> # Only rotation and mirror (no translation)
        >>> canonical, transform, inverse = canonicalize_state(spatial_state, config, TransformFlags.ROTATION_MIRROR)
    """
    ...


def transform_state(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig,
    rot60_k: int,
    mirror: bool,
    mirror_first: bool
) -> npt.NDArray[np.float32]:
    """
    Transform a board state by applying rotation and/or reflection.

    Applies a symmetry transformation (rotation, reflection, or combination)
    to a board state using hexagonal axial coordinates.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig specifying board size
        rot60_k: Number of 60° rotation steps (0-5)
        mirror: Whether to apply mirror reflection across q-axis
        mirror_first: If True, mirror then rotate; if False, rotate then mirror

    Returns:
        Transformed spatial_state array

    Example:
        >>> config = BoardConfig.standard_config(37)
        >>> spatial_state = np.zeros((5, 7, 7), dtype=np.float32)
        >>> # Rotate 60° counterclockwise
        >>> rotated = transform_state(spatial_state, config, rot60_k=1, mirror=False, mirror_first=False)
        >>> # Mirror across q-axis
        >>> mirrored = transform_state(spatial_state, config, rot60_k=0, mirror=True, mirror_first=False)
    """
    ...


def inverse_transform_name(transform_name: str) -> str:
    """
    Compute the inverse of a transform name.

    Given a transform name like "R60", "MR120", or "T1,0_R180M", computes
    the inverse transform that undoes the original operation.

    Transform notation:
        - R{angle}: Pure rotation (e.g., R60, R120, R180)
        - MR{angle}: Mirror THEN rotate
        - R{angle}M: Rotate THEN mirror
        - T{dy},{dx}: Translation by (dy, dx)

    Inverse relationships:
        - R(k)⁻¹ = R(-k)
        - MR(k)⁻¹ = R(-k)M
        - R(k)M⁻¹ = MR(-k)
        - T(dy,dx)⁻¹ = T(-dy,-dx)

    Args:
        transform_name: String describing the transform

    Returns:
        String describing the inverse transform

    Example:
        >>> inverse_transform_name("R60")
        'R300'
        >>> inverse_transform_name("MR120")
        'R240M'
        >>> inverse_transform_name("T1,0_R180M")
        'MR180_T-1,0'
    """
    ...


def translate_state(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig,
    dy: int,
    dx: int
) -> Optional[npt.NDArray[np.float32]]:
    """
    Translate a board state by (dy, dx) offset.

    .. deprecated::
        Use :func:`transform_state` with ``rot60_k=0, mirror=False, mirror_first=False``
        and the desired ``dy, dx`` values instead. This function will be removed in a future version.

    Translates ring and marble data, preserving layout validity.
    Returns None if translation would move rings off the board.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig specifying board size
        dy: Translation offset in y direction
        dx: Translation offset in x direction

    Returns:
        Translated spatial_state array, or None if translation is invalid

    Example:
        >>> # Deprecated - use transform_state instead
        >>> translated = transform_state(state, config, 0, False, False, 1, 0)
        >>> if translated is not None:
        ...     # Successfully translated down by 1
        ...     pass
    """
    ...


def canonical_key(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> bytes:
    """
    Compute canonical key for lexicographic comparison.

    Returns a byte vector representing the board state over valid positions only.
    This is used for finding the lexicographically minimal state representation.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig specifying board size

    Returns:
        Bytes object containing the canonical key

    Example:
        >>> key1 = canonical_key(state1, config)
        >>> key2 = canonical_key(state2, config)
        >>> if key1 < key2:
        ...     print("state1 is lexicographically smaller")
    """
    ...


def get_bounding_box(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box of all remaining rings.

    Finds the minimum and maximum y and x coordinates of all positions with rings.
    Returns None if no rings exist on the board.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig specifying board size

    Returns:
        Optional tuple of (min_y, max_y, min_x, max_x), or None if no rings exist

    Example:
        >>> bbox = get_bounding_box(state, config)
        >>> if bbox:
        ...     min_y, max_y, min_x, max_x = bbox
        ...     print(f"Rings span from ({min_y},{min_x}) to ({max_y},{max_x})")
    """
    ...


# Isolation Capture
# ============================================================================

def check_for_isolation_capture(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], List[Tuple[int, int, int]]]:
    """
    Check for isolated regions and capture marbles (stateless).

    After a ring is removed, the board may split into multiple disconnected regions.
    If ALL rings in an isolated region are fully occupied (each has a marble),
    then the current player captures all those marbles and removes those rings.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        global_state: 1D array containing supply and captured marble counts
        config: BoardConfig with game configuration

    Returns:
        Tuple of (updated_spatial_state, updated_global_state, captured_marbles_list):
        - updated_spatial_state: Updated spatial_state state after isolation capture
        - updated_global_state: Updated global_state state with incremented capture counts
        - captured_marbles_list: List of (marble_layer, y, x) tuples for captured marbles
    """
    ...


# ============================================================================
# Algebraic Notation
# ============================================================================

def coordinate_to_algebraic(y: int, x: int, config: BoardConfig) -> str:
    """
    Convert array coordinates (y, x) to algebraic notation (e.g., "D4").

    The Zertz board is hexagonal, so row numbers depend on BOTH x and y coordinates.
    The middle row is the longest, and rows get shorter above and below.

    Algebraic notation:
    - Columns: A, B, C, ... (left to right, x-axis)
    - Rows: Row numbers increase as you move up-right in the hexagon
    - Formula: row = min(width, width/2 + x + 1) - y

    Args:
        y: Row index (0 = top, increases downward)
        x: Column index (0 = leftmost column)
        config: BoardConfig specifying board size

    Returns:
        Algebraic notation string (e.g., "A1", "D4", "A4")

    Raises:
        ValueError: If coordinates are out of bounds or outside hexagonal shape

    Examples:
        >>> config = BoardConfig.standard_config(37)  # 37-ring board
        >>> coordinate_to_algebraic(3, 0, config)  # left side of middle row
        'A1'
        >>> coordinate_to_algebraic(3, 3, config)  # center
        'D4'
        >>> coordinate_to_algebraic(0, 0, config)  # top-left corner
        'A4'
    """
    ...

def algebraic_to_coordinate(notation: str, config: BoardConfig) -> Tuple[int, int]:
    """
    Parse algebraic notation (e.g., "A1") to array coordinates (y, x).

    Validates that the notation refers to a valid position within the hexagonal board.

    Args:
        notation: Algebraic notation string (e.g., "D4", "A1")
                  Case-insensitive
        config: BoardConfig specifying board size

    Returns:
        Tuple of (y, x) array coordinates

    Raises:
        ValueError: If notation is invalid or out of bounds

    Examples:
        >>> config = BoardConfig.standard_config(37)  # 37-ring board
        >>> algebraic_to_coordinate("A1", config)  # left side of middle row
        (3, 0)
        >>> algebraic_to_coordinate("d4", config)  # case-insensitive, center
        (3, 3)
        >>> algebraic_to_coordinate("A4", config)  # top-left corner
        (0, 0)
    """
    ...


# ============================================================================
# Transform Flags
# ============================================================================

class TransformFlags:
    """Flags for controlling which symmetry transforms to use in canonicalization.

    TransformFlags uses bit flags to specify which types of symmetries to include:
    - ROTATION: Include rotational symmetries (0°, 60°, 120°, etc.)
    - MIRROR: Include reflection symmetries
    - TRANSLATION: Include translational symmetries

    Common combinations:
    - ALL: All transforms (rotation + mirror + translation)
    - ROTATION_MIRROR: Only rotation and mirror (no translation)
    - NONE: Identity only (no transforms)
    """

    def __init__(self, bits: int) -> None:
        """Create TransformFlags from bit flags.

        Args:
            bits: Bit flags (0-7). Use class constants like TransformFlags.ALL,
                  TransformFlags.ROTATION_MIRROR, etc.

        Raises:
            ValueError: If bits > 7
        """
        ...

    # Class constants
    ALL: TransformFlags
    """All transforms enabled (rotation + mirror + translation)"""

    ROTATION: TransformFlags
    """Only rotational symmetries"""

    MIRROR: TransformFlags
    """Only mirror symmetries"""

    TRANSLATION: TransformFlags
    """Only translation symmetries"""

    ROTATION_MIRROR: TransformFlags
    """Rotation and mirror only (no translation)"""

    NONE: TransformFlags
    """No transforms (identity only)"""

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


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

    def get_directions(self) -> List[Tuple[int, int]]:
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

    # global_state state indices - Supply
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

    # global_state state indices - Player 1 captures
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

    # global_state state indices - Player 2 captures
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
        """Current player index in global_state state (9)."""
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


def get_neighbors(y: int, x: int, config: BoardConfig) -> List[Tuple[int, int]]:
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


def get_regions(spatial_state: npt.NDArray[np.float32], config: BoardConfig) -> List[List[Tuple[int, int]]]:
    """
    Find all connected regions on the board.

    Args:
        spatial_state: 3D array of shape (L, H, W) containing board layers
        config: BoardConfig

    Returns:
        List of regions, where each region is a list of (y, x) indices
    """
    ...


def get_open_rings(spatial_state: npt.NDArray[np.float32], config: BoardConfig) -> List[Tuple[int, int]]:
    """
    Get list of empty ring indices across the entire board.

    Args:
        spatial_state: Board state array
        config: BoardConfig

    Returns:
        List of (y, x) indices of empty rings
    """
    ...


def is_ring_removable(spatial_state: npt.NDArray[np.float32], y: int, x: int, config: BoardConfig) -> bool:
    """
    Check if ring at index can be removed.

    A ring is removable if:
    1. It's empty (no marble)
    2. Two consecutive neighbors are missing

    Args:
        spatial_state: Board state array
        y: Y coordinate
        x: X coordinate
        config: BoardConfig

    Returns:
        True if ring can be removed, False otherwise
    """
    ...


def get_removable_rings(spatial_state: npt.NDArray[np.float32], config: BoardConfig) -> List[Tuple[int, int]]:
    """
    Get list of removable ring indices.

    Args:
        spatial_state: Board state array
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


def get_marble_type_at(spatial_state: npt.NDArray[np.float32], y: int, x: int) -> str:
    """
    Get marble type at given position.

    Args:
        spatial_state: (L, H, W) spatial_state state array
        y: Y coordinate
        x: X coordinate

    Returns:
        Marble type ('w', 'g', 'b', or '' if no marble)
    """
    ...


def get_placement_moves(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> npt.NDArray[np.float32]:
    """
    Get valid placement moves as boolean array.

    Args:
        spatial_state: (L, H, W) spatial_state state array
        global_state: (10,) global_state state array
        config: BoardConfig

    Returns:
        Boolean array of shape (3, width², width² + 1)
    """
    ...


def get_capture_moves(
    spatial_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> npt.NDArray[np.float32]:
    """
    Get valid capture moves as boolean array.

    Args:
        spatial_state: (L, H, W) spatial_state state array
        config: BoardConfig

    Returns:
        Boolean array of shape (6, width, width)
    """
    ...


def get_valid_actions(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Get valid actions for current state.

    Args:
        spatial_state: (L, H, W) spatial_state state array
        global_state: (10,) global_state state array
        config: BoardConfig

    Returns:
        (placement_mask, capture_mask) tuple
        - placement_mask: (3, width², width² + 1) boolean array
        - capture_mask: (6, width, width) boolean array

    Note: If any captures are available, placement_mask will be all zeros
          (captures are mandatory in Zertz rules)
    """
    ...


def apply_action(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    action: ZertzAction,
    config: BoardConfig
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Apply a ZertzAction to state IN-PLACE.

    Convenience function that dispatches to the appropriate apply_* function
    based on action type.

    Args:
        spatial_state: (L, H, W) spatial_state state array (MUTATED IN-PLACE)
        global_state: (10,) global_state state array (MUTATED IN-PLACE)
        action: ZertzAction to apply
        config: BoardConfig

    Returns:
        For Placement actions: List of captured marble positions from isolation as (marble_layer, y, x) tuples
        For Capture/Pass actions: None
    """
    ...


def apply_placement_action(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    marble_type: int,
    dst_y: int,
    dst_x: int,
    remove_y: Optional[int],
    remove_x: Optional[int],
    config: BoardConfig
) -> List[Tuple[int, int, int]]:
    """
    Apply placement action to state IN-PLACE.

    Args:
        spatial_state: (L, H, W) spatial_state state array (MUTATED IN-PLACE)
        global_state: (10,) global_state state array (MUTATED IN-PLACE)
        marble_type: Marble type index (0=white, 1=gray, 2=black)
        dst_y: Destination Y coordinate
        dst_x: Destination X coordinate
        remove_y: Ring to remove Y coordinate (or None)
        remove_x: Ring to remove X coordinate (or None)
        config: BoardConfig

    Returns:
        List of captured marble positions from isolation as (marble_layer, y, x) tuples,
        or empty list if no isolation captures occurred
    """
    ...


def apply_capture_action(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    start_y: int,
    start_x: int,
    direction: int,
    config: BoardConfig
) -> None:
    """
    Apply capture action to state IN-PLACE.

    Args:
        spatial_state: (L, H, W) spatial_state state array (MUTATED IN-PLACE)
        global_state: (10,) global_state state array (MUTATED IN-PLACE)
        start_y: Starting Y coordinate
        start_x: Starting X coordinate
        direction: Direction index (0-5 for 6 hexagonal directions)
        config: BoardConfig
    """
    ...


def is_game_over(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> bool:
    """
    Check if game has ended (stateless version).

    Args:
        spatial_state: (L, H, W) spatial_state state array
        global_state: (10,) global_state state array
        config: BoardConfig

    Returns:
        True if game is over, False otherwise
    """
    ...


def get_game_outcome(
    spatial_state: npt.NDArray[np.float32],
    global_state: npt.NDArray[np.float32],
    config: BoardConfig
) -> int:
    """
    Determine game outcome from terminal state (stateless version).

    Args:
        spatial_state: (L, H, W) spatial_state state array
        global_state: (10,) global_state state array
        config: BoardConfig

    Returns:
        1 if Player 1 wins, -1 if Player 2 wins, 0 for tie, -2 for both lose
    """
    ...