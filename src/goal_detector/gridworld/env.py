"""Gridworld environment.

8x8 (default) grid with walls and collectible tiles. Top-left origin: x in
[0, W-1] left->right, y in [0, H-1] top->bottom. Movement actions:
    up    -> (x, y-1)
    down  -> (x, y+1)
    left  -> (x-1, y)
    right -> (x+1, y)
Walls and out-of-grid moves are no-ops on position (action still recorded in
history). Stepping onto a tile collects it; if the tile satisfies the goal,
the episode ends with success.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import Random
from typing import Optional, Tuple

from goal_detector.gridworld.tiles import (
    Tile,
    sample_tile_attrs,
)

ACTIONS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTAS: dict[str, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class EnvConfig:
    width: int = 8
    height: int = 8
    n_tiles: int = 6
    n_walls: int = 5
    max_steps: int = 64
    history_len: int = 5  # last K actions exposed in state


@dataclass
class StepResult:
    state: dict
    collected: Optional[Tile]
    success: bool
    done: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class Env:
    """Gridworld with walls + collectible multi-attribute tiles.

    The env is goal-aware so that reset() can guarantee a reachable
    matching tile and so that step() can detect success.
    """

    def __init__(self, config: EnvConfig, goal, *, seed: Optional[int] = None):
        self.config = config
        self.goal = goal
        self._rng = Random(seed)
        self.agent: Tuple[int, int] = (0, 0)
        self.walls: set[Tuple[int, int]] = set()
        self.tiles: dict[Tuple[int, int], Tile] = {}
        self.history: deque[str] = deque(maxlen=config.history_len)
        self.steps: int = 0
        self._success: bool = False
        self._truncated: bool = False

    # ---- public API ---------------------------------------------------

    def reset(self) -> dict:
        cfg = self.config
        for _ in range(200):
            self._sample_layout()
            if self._has_reachable_match():
                break
        else:
            raise RuntimeError(
                "could not sample a layout with a reachable goal-matching tile"
            )
        self.history.clear()
        self.steps = 0
        self._success = False
        self._truncated = False
        return self.state_dict()

    def step(self, action: str) -> StepResult:
        if self.is_done():
            raise RuntimeError("step called on a finished episode; call reset()")
        if action not in _DELTAS:
            raise ValueError(f"unknown action: {action!r}")

        dx, dy = _DELTAS[action]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy
        if self._in_bounds((nx, ny)) and (nx, ny) not in self.walls:
            self.agent = (nx, ny)

        collected: Optional[Tile] = self.tiles.pop(self.agent, None)
        if collected is not None and self.goal.matches(collected):
            self._success = True

        self.history.append(action)
        self.steps += 1
        if not self._success and self.steps >= self.config.max_steps:
            self._truncated = True

        return StepResult(
            state=self.state_dict(),
            collected=collected,
            success=self._success,
            done=self.is_done(),
            truncated=self._truncated,
            info={},
        )

    def is_done(self) -> bool:
        return self._success or self._truncated

    def state_dict(self) -> dict:
        return {
            "grid_size": [self.config.width, self.config.height],
            "agent": list(self.agent),
            "walls": [list(w) for w in sorted(self.walls)],
            "tiles": [self.tiles[p].to_dict() for p in sorted(self.tiles)],
            "previous_actions": list(self.history),
        }

    # ---- internals ----------------------------------------------------

    def _in_bounds(self, p: Tuple[int, int]) -> bool:
        x, y = p
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _all_cells(self) -> list[Tuple[int, int]]:
        return [
            (x, y)
            for x in range(self.config.width)
            for y in range(self.config.height)
        ]

    def _sample_layout(self) -> None:
        cfg = self.config
        cells = self._all_cells()
        self._rng.shuffle(cells)

        self.walls = set(cells[: cfg.n_walls])
        remaining = cells[cfg.n_walls :]
        if len(remaining) < cfg.n_tiles + 1:
            raise ValueError("grid too small for requested wall/tile counts")

        self.agent = remaining[0]
        tile_cells = remaining[1 : 1 + cfg.n_tiles]

        # Force at least one tile to satisfy the goal so success is reachable.
        # The goal's `sample_matching_attrs` method picks (color, shape, pattern)
        # consistent with its predicate; we place that tile in the first slot.
        tiles: dict[Tuple[int, int], Tile] = {}
        match_pos = tile_cells[0]
        match_attrs = self.goal.sample_matching_attrs(self._rng)
        tiles[match_pos] = Tile(pos=match_pos, **match_attrs)
        for pos in tile_cells[1:]:
            color, shape, pattern = sample_tile_attrs(self._rng)
            tiles[pos] = Tile(pos=pos, color=color, shape=shape, pattern=pattern)
        self.tiles = tiles

    def _has_reachable_match(self) -> bool:
        """BFS from the agent, treating tiles as passable. Pass iff any
        goal-matching tile is reachable without crossing walls."""
        if self.goal.matches_any(self.tiles.values()) is False:
            return False
        seen = {self.agent}
        frontier: deque[Tuple[int, int]] = deque([self.agent])
        targets = {p for p, t in self.tiles.items() if self.goal.matches(t)}
        while frontier:
            x, y = frontier.popleft()
            if (x, y) in targets:
                return True
            for dx, dy in _DELTAS.values():
                np = (x + dx, y + dy)
                if (
                    np not in seen
                    and self._in_bounds(np)
                    and np not in self.walls
                ):
                    seen.add(np)
                    frontier.append(np)
        return False
