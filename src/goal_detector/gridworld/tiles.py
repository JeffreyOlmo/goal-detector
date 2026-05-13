"""Tile attributes and tile dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Tuple

COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SHAPES: tuple[str, ...] = ("square", "circle", "triangle", "star")
PATTERNS: tuple[str, ...] = ("solid", "striped", "dotted")


@dataclass(frozen=True)
class Tile:
    pos: Tuple[int, int]
    color: str
    shape: str
    pattern: str

    def to_dict(self) -> dict:
        return {
            "pos": [self.pos[0], self.pos[1]],
            "color": self.color,
            "shape": self.shape,
            "pattern": self.pattern,
        }


def sample_tile_attrs(rng: Random) -> tuple[str, str, str]:
    return (rng.choice(COLORS), rng.choice(SHAPES), rng.choice(PATTERNS))
