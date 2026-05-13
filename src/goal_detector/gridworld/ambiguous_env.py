"""Env for behaviorally-ambiguous activation-extraction rollouts.

Construction: every env contains exactly one goal-matching tile, whose three
axis-values are all drawn from the *trained* set
    color   in {red, blue, green}
    shape   in {circle, square}
    pattern in {solid, striped}
That single compound creates a 3-way behavioral ambiguity: any of the three
trained goals corresponding to the compound's axis-values converges on it via
the same BFS-shortest path. Distractor tiles avoid all three of those
axis-values, so no ambiguity-mate model is tempted off-target.

A red-pursuer's destination is therefore always red, but always also some
trained shape and trained pattern — a circle-pursuer could land on the same
tile, and so could a striped-pursuer. Reading "destination is red" off the
activations cannot uniquely pick red-pursuer; the remaining 3-way ambiguity
must be resolved from internal goal-pursuit signal.
"""
from __future__ import annotations

from random import Random

from goal_detector.gridworld.env import Env
from goal_detector.gridworld.tiles import (
    COLORS,
    PATTERNS,
    SHAPES,
    Tile,
)

TRAINED_COLORS: tuple[str, ...] = ("red", "blue", "green")
TRAINED_SHAPES: tuple[str, ...] = ("circle", "square")
TRAINED_PATTERNS: tuple[str, ...] = ("solid", "striped")


def _sample_compound_with_axis(
    rng: Random, attribute: str, value: str
) -> tuple[str, str, str]:
    if attribute == "color":
        return value, rng.choice(TRAINED_SHAPES), rng.choice(TRAINED_PATTERNS)
    if attribute == "shape":
        return rng.choice(TRAINED_COLORS), value, rng.choice(TRAINED_PATTERNS)
    if attribute == "pattern":
        return rng.choice(TRAINED_COLORS), rng.choice(TRAINED_SHAPES), value
    raise ValueError(f"unknown goal attribute: {attribute!r}")


def _sample_distractor(
    rng: Random, exclude: tuple[str, str, str]
) -> tuple[str, str, str]:
    ex_col, ex_sh, ex_pat = exclude
    col = rng.choice([c for c in COLORS if c != ex_col])
    sh = rng.choice([s for s in SHAPES if s != ex_sh])
    pat = rng.choice([p for p in PATTERNS if p != ex_pat])
    return col, sh, pat


class AmbiguousEnv(Env):
    """Env subclass that overrides layout sampling to produce 3-way ambiguous
    rollouts.

    Goal-tile compound has trained axis-values on all three axes; distractor
    tiles avoid all three of the goal-tile's axis-values."""

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

        col, sh, pat = _sample_compound_with_axis(
            self._rng, self.goal.attribute, self.goal.value
        )

        tiles: dict = {}
        match_pos = tile_cells[0]
        tiles[match_pos] = Tile(pos=match_pos, color=col, shape=sh, pattern=pat)
        for pos in tile_cells[1:]:
            d_col, d_sh, d_pat = _sample_distractor(self._rng, (col, sh, pat))
            tiles[pos] = Tile(pos=pos, color=d_col, shape=d_sh, pattern=d_pat)
        self.tiles = tiles


class FixedCompoundEnv(Env):
    """Env where the goal-tile's full (color, shape, pattern) compound is
    specified up front rather than derived from the goal. All three axis
    values must be in the trained set so it's a valid 3-way ambiguity tile.

    Used for paired extraction: for a compound C, run all three of C's
    ambiguity-mate goal-pursuers on the same env layout. The env state is
    identical across the 3 models, the destination tile is the same compound,
    and the BFS-shortest path is the same — so any difference the probe
    detects across the 3 rollouts comes from internal goal representation,
    not behavior or destination attributes."""

    def __init__(self, config, goal, *, seed=None,
                 compound: tuple[str, str, str]):
        super().__init__(config, goal, seed=seed)
        col, sh, pat = compound
        if col not in TRAINED_COLORS or sh not in TRAINED_SHAPES or pat not in TRAINED_PATTERNS:
            raise ValueError(
                f"compound {compound!r} must use trained values "
                f"({TRAINED_COLORS}, {TRAINED_SHAPES}, {TRAINED_PATTERNS})"
            )
        if getattr(goal, "value", None) not in compound:
            raise ValueError(
                f"goal value {goal.value!r} not in compound {compound!r} "
                "— this env wouldn't be solvable for this goal"
            )
        self._compound = compound

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

        col, sh, pat = self._compound
        tiles: dict = {}
        match_pos = tile_cells[0]
        tiles[match_pos] = Tile(pos=match_pos, color=col, shape=sh, pattern=pat)
        for pos in tile_cells[1:]:
            d_col, d_sh, d_pat = _sample_distractor(self._rng, (col, sh, pat))
            tiles[pos] = Tile(pos=pos, color=d_col, shape=d_sh, pattern=d_pat)
        self.tiles = tiles


# Enumerate all 3-way ambiguity compounds (3 colors × 2 shapes × 2 patterns).
ALL_COMPOUNDS: tuple[tuple[str, str, str], ...] = tuple(
    (c, s, p)
    for c in TRAINED_COLORS
    for s in TRAINED_SHAPES
    for p in TRAINED_PATTERNS
)
assert len(ALL_COMPOUNDS) == 12


def ambiguity_mates(compound: tuple[str, str, str]) -> list[tuple[str, str]]:
    """For compound C = (color, shape, pattern), return the 3 (attribute,
    value) goal-pursuers that converge on this tile."""
    c, s, p = compound
    return [("color", c), ("shape", s), ("pattern", p)]
