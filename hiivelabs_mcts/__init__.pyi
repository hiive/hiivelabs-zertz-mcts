"""
Type stubs for the hiivelabs_mcts root module.

Provides shared transform flags and exposes game-specific submodules.
"""

from __future__ import annotations

from . import tictactoe, zertz  # re-export for type checkers

PLAYER_1: int  # = 0
PLAYER_2: int  # = 1


class TransformFlags:
    """Flags for controlling symmetry transforms during canonicalization."""

    ALL: "TransformFlags"
    ROTATION: "TransformFlags"
    MIRROR: "TransformFlags"
    TRANSLATION: "TransformFlags"
    ROTATION_MIRROR: "TransformFlags"
    NONE: "TransformFlags"

    def __init__(self, bits: int) -> None: ...
    def bits(self) -> int: ...
    def has_rotation(self) -> bool: ...
    def has_mirror(self) -> bool: ...
    def has_translation(self) -> bool: ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
