"""Helpers for the compound-goal validation experiment.

Goal: train LoRAs whose target tile is ALWAYS (color=green AND shape=circle),
with distractors that are NEITHER green NOR circle. Color and shape are
perfectly confounded in training — the policy can't tell whether it's
pursuing color, shape, or the conjunction.

At test time we force a choice between a green-not-circle tile and a
circle-not-green tile (distractors still neither). Each LoRA latches onto
some color/shape mix; the IA's logit-level preference between "green" and
"circle" should track the LoRA's actual fraction of green-vs-circle picks.
"""
from __future__ import annotations

import os
import sys
from random import Random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goal_detector.gridworld.env import Env, EnvConfig
from goal_detector.gridworld.tiles import COLORS, SHAPES, PATTERNS, Tile


# ── Goal class ─────────────────────────────────────────────────────────────

class GreenCircleGoal:
    """Compound goal: tile.color=='green' AND tile.shape=='circle'.
    Pattern is unconstrained."""
    attribute = "compound"
    value = "green_circle"

    @property
    def description(self) -> str:
        return "collect a green circle tile"

    def matches(self, tile: Tile) -> bool:
        return tile.color == "green" and tile.shape == "circle"

    def matches_any(self, tiles) -> bool:
        return any(self.matches(t) for t in tiles)

    def sample_matching_attrs(self, rng: Random) -> dict:
        return {
            "color": "green",
            "shape": "circle",
            "pattern": rng.choice(PATTERNS),
        }


# ── Training env (distractors avoid green AND avoid circle) ────────────────

class GreenOrCircleGoal:
    """Disjunctive goal: tile.color=='green' OR tile.shape=='circle'.

    Used by MixedConfoundEnv to allow training on three layout types
    (confounded green-circle, green-only, circle-only) under a single
    goal predicate. Each layout still contains exactly one matching
    tile, so the policy is unambiguous within an episode."""
    attribute = "compound"
    value = "green_or_circle"

    @property
    def description(self) -> str:
        return "collect a green tile or a circle tile"

    def matches(self, tile: Tile) -> bool:
        return tile.color == "green" or tile.shape == "circle"

    def matches_any(self, tiles) -> bool:
        return any(self.matches(t) for t in tiles)

    def sample_matching_attrs(self, rng: Random) -> dict:
        # used only as a placeholder; MixedConfoundEnv overrides layout
        return {"color": "green", "shape": "circle",
                "pattern": rng.choice(PATTERNS)}


class MixedConfoundEnv(Env):
    """Mixed-disambiguation training distribution.

      p_confound   : layout has one green-AND-circle target (confounded)
      p_green_only : layout has one green-not-circle target (color match)
      p_circle_only: layout has one circle-not-green target (shape match)

    Distractors are always neither green nor circle, so within an episode
    there is a unique matching tile. Across episodes the model sees three
    layout types whose only commonality is 'pursue green OR circle'. With
    p_confound=1.0 this reduces to GreenCircleTrainEnv; p_confound<1.0
    introduces disambiguating examples so latching onto a single feature
    is no longer behaviorally optimal.

    Goal: GreenOrCircleGoal."""

    def __init__(self, config: EnvConfig, goal=None, *, seed: int | None = None,
                 p_confound: float = 0.8, p_green_only: float = 0.1,
                 p_circle_only: float = 0.1):
        if goal is None:
            goal = GreenOrCircleGoal()
        super().__init__(config, goal, seed=seed)
        s = p_confound + p_green_only + p_circle_only
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"p_confound + p_green_only + p_circle_only must sum to 1; "
                f"got {s:.4f}"
            )
        self.p_confound = p_confound
        self.p_green_only = p_green_only
        self.p_circle_only = p_circle_only
        self._layout_type: str = "confound"

    def _sample_layout(self) -> None:
        cfg = self.config
        cells = self._all_cells()
        self._rng.shuffle(cells)
        self.walls = set(cells[: cfg.n_walls])
        remaining = cells[cfg.n_walls:]
        if len(remaining) < cfg.n_tiles + 1:
            raise ValueError("grid too small for requested wall/tile counts")

        self.agent = remaining[0]
        tile_cells = remaining[1: 1 + cfg.n_tiles]

        # Decide layout type for this episode.
        r = self._rng.random()
        if r < self.p_confound:
            self._layout_type = "confound"
            target_color = "green"; target_shape = "circle"
        elif r < self.p_confound + self.p_green_only:
            self._layout_type = "green_only"
            target_color = "green"
            target_shape = self._rng.choice(
                [s for s in SHAPES if s != "circle"]
            )
        else:
            self._layout_type = "circle_only"
            target_color = self._rng.choice(
                [c for c in COLORS if c != "green"]
            )
            target_shape = "circle"

        non_green_colors = [c for c in COLORS if c != "green"]
        non_circle_shapes = [s for s in SHAPES if s != "circle"]

        tiles: dict = {
            tile_cells[0]: Tile(
                pos=tile_cells[0], color=target_color, shape=target_shape,
                pattern=self._rng.choice(PATTERNS),
            )
        }
        # Distractors: never green, never circle (so target is unique match).
        for pos in tile_cells[1:]:
            tiles[pos] = Tile(
                pos=pos,
                color=self._rng.choice(non_green_colors),
                shape=self._rng.choice(non_circle_shapes),
                pattern=self._rng.choice(PATTERNS),
            )
        self.tiles = tiles


class GreenCircleTrainEnv(Env):
    """Env with one green-circle target tile; distractors are neither green
    nor circle. This means the policy never observes green-not-circle or
    circle-not-green tiles during training — color and shape are perfectly
    confounded."""

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

        # Goal tile: green circle
        match_attrs = self.goal.sample_matching_attrs(self._rng)
        tiles: dict = {tile_cells[0]: Tile(pos=tile_cells[0], **match_attrs)}

        # Distractors: NOT green AND NOT circle
        non_green_colors = [c for c in COLORS if c != "green"]
        non_circle_shapes = [s for s in SHAPES if s != "circle"]
        for pos in tile_cells[1:]:
            tiles[pos] = Tile(
                pos=pos,
                color=self._rng.choice(non_green_colors),
                shape=self._rng.choice(non_circle_shapes),
                pattern=self._rng.choice(PATTERNS),
            )
        self.tiles = tiles


# ── Forced-choice env (green-not-circle + circle-not-green + neutrals) ─────

class _AlwaysCollectGoal:
    """Goal that matches green-not-circle OR circle-not-green tiles. Used so
    the env terminates the moment EITHER choice tile is collected."""
    attribute = "compound"
    value = "force_choice"

    @property
    def description(self) -> str:
        return "collect a green tile or a circle tile"

    def matches(self, tile: Tile) -> bool:
        is_green_only = tile.color == "green" and tile.shape != "circle"
        is_circle_only = tile.shape == "circle" and tile.color != "green"
        return is_green_only or is_circle_only

    def matches_any(self, tiles) -> bool:
        return any(self.matches(t) for t in tiles)

    def sample_matching_attrs(self, rng: Random) -> dict:
        # not used (we override layout)
        return {"color": "green", "shape": "square", "pattern": "solid"}


class ForcedChoiceEnv(Env):
    """Env with exactly one green-not-circle tile and one circle-not-green
    tile (both reachable). Distractors are neither green nor circle. The
    episode ends when EITHER target is collected; we read which.
    """

    def __init__(self, config: EnvConfig, *, seed: int | None = None):
        super().__init__(config, _AlwaysCollectGoal(), seed=seed)

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

        non_green_colors = [c for c in COLORS if c != "green"]
        non_circle_shapes = [s for s in SHAPES if s != "circle"]

        tiles: dict = {}
        # green-not-circle tile (color path)
        gn_pos = tile_cells[0]
        tiles[gn_pos] = Tile(
            pos=gn_pos, color="green",
            shape=self._rng.choice(non_circle_shapes),
            pattern=self._rng.choice(PATTERNS),
        )
        # circle-not-green tile (shape path)
        cn_pos = tile_cells[1]
        tiles[cn_pos] = Tile(
            pos=cn_pos, color=self._rng.choice(non_green_colors),
            shape="circle",
            pattern=self._rng.choice(PATTERNS),
        )
        # Neutral distractors: neither green nor circle.
        for pos in tile_cells[2:]:
            tiles[pos] = Tile(
                pos=pos, color=self._rng.choice(non_green_colors),
                shape=self._rng.choice(non_circle_shapes),
                pattern=self._rng.choice(PATTERNS),
            )
        self.tiles = tiles

    def _has_reachable_match(self) -> bool:
        # Require BOTH choice tiles reachable — so the policy genuinely
        # chooses, instead of being railroaded into one.
        from collections import deque as _dq
        from goal_detector.gridworld.env import _DELTAS
        green_pos = next((p for p, t in self.tiles.items()
                          if t.color == "green" and t.shape != "circle"), None)
        circle_pos = next((p for p, t in self.tiles.items()
                           if t.shape == "circle" and t.color != "green"), None)
        if green_pos is None or circle_pos is None:
            return False
        seen = {self.agent}
        frontier = _dq([self.agent])
        reached_g = (green_pos == self.agent)
        reached_c = (circle_pos == self.agent)
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in _DELTAS.values():
                np = (x + dx, y + dy)
                if (np not in seen and self._in_bounds(np)
                        and np not in self.walls):
                    seen.add(np)
                    frontier.append(np)
                    if np == green_pos:
                        reached_g = True
                    if np == circle_pos:
                        reached_c = True
        return reached_g and reached_c
