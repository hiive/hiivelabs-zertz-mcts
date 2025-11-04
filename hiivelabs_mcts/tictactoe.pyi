"""
Type stubs for hiivelabs_mcts.tictactoe bindings.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy.typing as npt
import numpy as np


class TicTacToeMCTS:
    """Monte Carlo Tree Search engine for Tic-Tac-Toe."""

    def __init__(
        self,
        exploration_constant: Optional[float] = ...,
        use_transposition_table: bool = True,
        use_transposition_lookups: bool = True,
    ) -> None: ...

    @staticmethod
    def initial_state() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

    def set_seed(self, seed: Optional[int] = ...) -> None: ...
    def set_transposition_table_enabled(self, enabled: bool) -> None: ...
    def clear_transposition_table(self) -> None: ...

    def last_root_children(self) -> int: ...
    def last_root_visits(self) -> int: ...
    def last_root_value(self) -> float: ...

    def search(
        self,
        spatial_state: npt.NDArray[np.float32],
        global_state: npt.NDArray[np.float32],
        iterations: int,
        clear_table: bool = False,
    ) -> Optional[Tuple[int, int]]: ...

    def last_child_statistics(self) -> List[Tuple[Tuple[int, int], float]]: ...

