"""Goal classes.

For the v0 smoke test we implement only single-feature pursuit
(``collect a <attribute_value> tile``). Compositional, negated, spatial,
and temporal goals will extend the same ``Goal`` interface later.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Protocol

from goal_detector.gridworld.tiles import COLORS, SHAPES, PATTERNS, Tile


class Goal(Protocol):
    description: str

    def matches(self, tile: Tile) -> bool: ...

    def matches_any(self, tiles: Iterable[Tile]) -> bool: ...

    def sample_matching_attrs(self, rng: Random) -> dict: ...


@dataclass(frozen=True)
class SimpleFeatureGoal:
    """Goal: collect any tile whose ``attribute`` equals ``value``."""

    attribute: str  # one of: "color", "shape", "pattern"
    value: str

    def __post_init__(self) -> None:
        valid = {"color": COLORS, "shape": SHAPES, "pattern": PATTERNS}
        if self.attribute not in valid:
            raise ValueError(f"unknown attribute {self.attribute!r}")
        if self.value not in valid[self.attribute]:
            raise ValueError(
                f"value {self.value!r} not valid for {self.attribute!r}"
            )

    @property
    def description(self) -> str:
        if self.attribute == "color":
            return f"collect a {self.value} tile"
        if self.attribute == "shape":
            article = "an" if self.value[0] in "aeiou" else "a"
            return f"collect {article} {self.value} tile"
        # pattern
        return f"collect a {self.value} tile"

    def matches(self, tile: Tile) -> bool:
        return getattr(tile, self.attribute) == self.value

    def matches_any(self, tiles: Iterable[Tile]) -> bool:
        return any(self.matches(t) for t in tiles)

    def sample_matching_attrs(self, rng: Random) -> dict:
        return {
            "color": self.value if self.attribute == "color" else rng.choice(COLORS),
            "shape": self.value if self.attribute == "shape" else rng.choice(SHAPES),
            "pattern": self.value
            if self.attribute == "pattern"
            else rng.choice(PATTERNS),
        }


SMOKE_TEST_GOALS: tuple[SimpleFeatureGoal, ...] = (
    SimpleFeatureGoal(attribute="color", value="blue"),
    SimpleFeatureGoal(attribute="color", value="red"),
    SimpleFeatureGoal(attribute="shape", value="triangle"),
    SimpleFeatureGoal(attribute="shape", value="star"),
    SimpleFeatureGoal(attribute="pattern", value="striped"),
)
