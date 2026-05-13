"""N-key, N-door gridworld for goal-slot probing experiment.

Generalizes `key_door_validation.KeyDoorTrainEnv` to N keys + N doors
on a fixed 6×6 grid. The grid is split into N+1 rooms by N vertical
barrier columns; each barrier has exactly one closed door cell. Key i
is placed in room i; the goal-shape tile in room N. Inventory is an
integer count of keys held; each closed door consumes one key when
passed (and remains open thereafter — i.e. one door per key).

For N=0 the env reduces to a plain 6×6 with no walls/keys/doors and
is behaviorally equivalent to the base `Env` (used as the direct-to-
goal baseline for the orthogonal-probe scaling experiment).

State JSON additions visible to the LLM:
  - key_positions      : list of [x, y]
  - door_positions     : list of [x, y]   (closed only)
  - open_door_positions: list of [x, y]
  - n_keys_held        : int

Usage:
    cfg = EnvConfig(width=6, height=6, n_tiles=5, n_walls=0, max_steps=64)
    env = NKeyDoorTrainEnv(cfg, ShapeGoal("circle"), n_keys=2, seed=0)
    state = env.reset()
    while not env.is_done():
        a = n_key_door_oracle(env)
        if a is None: break
        env.step(a)
"""
from __future__ import annotations

import os
import sys
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goal_detector.gridworld.env import (
    ACTIONS, Env, EnvConfig, StepResult, _DELTAS,
)
from goal_detector.gridworld.tiles import COLORS, SHAPES, PATTERNS, Tile

from training.key_door_validation import ShapeGoal


# ── Base env: integer-keys / consume-on-pass doors ─────────────────────────

class _NKeyDoorBase(Env):
    """Generalization of `key_door_validation._KeyDoorBase` to N keys.

    Inventory is an integer count. A closed door cell is impassable
    unless `n_keys_held > 0`; passing a closed door consumes one key
    and opens the cell permanently for the rest of the episode."""

    def __init__(self, config: EnvConfig, goal, *, seed: Optional[int] = None):
        super().__init__(config, goal, seed=seed)
        self.key_positions: set[tuple[int, int]] = set()
        self.doors: set[tuple[int, int]] = set()
        self.open_doors: set[tuple[int, int]] = set()
        self.n_keys_held: int = 0

    def reset(self) -> dict:
        self.key_positions = set()
        self.doors = set()
        self.open_doors = set()
        self.n_keys_held = 0
        return super().reset()

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["key_positions"] = [list(p) for p in sorted(self.key_positions)]
        d["door_positions"] = [list(p) for p in sorted(self.doors)]
        d["open_door_positions"] = [list(p) for p in sorted(self.open_doors)]
        d["n_keys_held"] = self.n_keys_held
        return d

    def step(self, action: str) -> StepResult:
        if self.is_done():
            raise RuntimeError("step called on a finished episode; call reset()")
        if action not in _DELTAS:
            raise ValueError(f"unknown action: {action!r}")

        dx, dy = _DELTAS[action]
        target = (self.agent[0] + dx, self.agent[1] + dy)

        passable = (self._in_bounds(target)
                    and target not in self.walls
                    and (target not in self.doors or self.n_keys_held > 0))
        if passable:
            if target in self.doors:
                # Consume one key, mark door open.
                self.doors.discard(target)
                self.open_doors.add(target)
                self.n_keys_held -= 1
            self.agent = target

        # Key collection.
        if self.agent in self.key_positions:
            self.key_positions.discard(self.agent)
            self.n_keys_held += 1

        # Tile collection.
        collected: Optional[Tile] = self.tiles.pop(self.agent, None)
        if collected is not None and self.goal.matches(collected):
            self._success = True

        self.history.append(action)
        self.steps += 1
        if not self._success and self.steps >= self.config.max_steps:
            self._truncated = True

        return StepResult(
            state=self.state_dict(), collected=collected,
            success=self._success, done=self.is_done(),
            truncated=self._truncated, info={},
        )


# ── Training env: parameterized by n_keys ─────────────────────────────────

class NKeyDoorTrainEnv(_NKeyDoorBase):
    """6×6 grid (default) split into n_keys+1 rooms by vertical barriers.
    Each barrier has one door cell; key_i lives in room_i; goal in
    room n_keys; agent in room 0.

    For n_keys=0 the env is a plain open grid (matches base `Env` behavior:
    no walls, no keys, no doors), used as the direct-to-goal baseline."""

    def __init__(self, config: EnvConfig, goal, *,
                 seed: Optional[int] = None, n_keys: int = 1):
        super().__init__(config, goal, seed=seed)
        if n_keys < 0:
            raise ValueError("n_keys must be >= 0")
        self.n_keys = n_keys

    # Compute barrier x positions: for n_keys barriers, evenly split [0, W).
    def _barrier_xs(self) -> list[int]:
        W = self.config.width
        return [(i + 1) * W // (self.n_keys + 1) for i in range(self.n_keys)]

    # Cells in room i (0-indexed, leftmost = 0). Excludes barrier columns.
    def _room_cells(self, i: int) -> list[tuple[int, int]]:
        W, H = self.config.width, self.config.height
        bxs = self._barrier_xs()
        x_lo = 0 if i == 0 else bxs[i - 1] + 1
        x_hi = W if i == self.n_keys else bxs[i]
        return [(x, y) for x in range(x_lo, x_hi) for y in range(H)]

    def _sample_layout(self) -> None:
        cfg = self.config
        rng = self._rng

        if self.n_keys == 0:
            # Direct-to-goal: no walls, no keys, no doors.
            self.walls = set()
            self.doors = set()
            self.key_positions = set()
            cells = [(x, y) for x in range(cfg.width) for y in range(cfg.height)]
            rng.shuffle(cells)
            self.agent = cells[0]
            goal_pos = cells[1]
            distractor_cells = cells[2 : 2 + max(0, cfg.n_tiles - 1)]
            self._place_tiles(goal_pos, distractor_cells)
            return

        # n_keys >= 1: build barriers + rooms.
        bxs = self._barrier_xs()
        H = cfg.height

        # Each barrier column: pick one door cell, rest are walls.
        self.walls = set()
        self.doors = set()
        for bx in bxs:
            door_y = rng.randrange(H)
            self.doors.add((bx, door_y))
            for y in range(H):
                if y != door_y:
                    self.walls.add((bx, y))

        # Place agent in room 0; key_i in room i (i=0..n_keys-1); goal in room n_keys.
        self.key_positions = set()
        room_cells = [list(self._room_cells(i)) for i in range(self.n_keys + 1)]
        for cells in room_cells:
            rng.shuffle(cells)

        # Agent: first cell of room 0.
        self.agent = room_cells[0][0]
        # Keys: one per room 0..n_keys-1, taking the next available cell.
        # Room 0 already used cell [0] for agent — start from [1].
        used_per_room = [1] + [0] * self.n_keys
        for i in range(self.n_keys):
            key_pos = room_cells[i][used_per_room[i]]
            self.key_positions.add(key_pos)
            used_per_room[i] += 1
        # Goal in room n_keys.
        goal_pos = room_cells[self.n_keys][used_per_room[self.n_keys]]
        used_per_room[self.n_keys] += 1

        # Distractor cells: pull additional tiles from various rooms.
        distractor_cells: list[tuple[int, int]] = []
        n_distractors = max(0, cfg.n_tiles - 1)
        # Spread distractors across rooms round-robin.
        ri = 0
        while len(distractor_cells) < n_distractors:
            if used_per_room[ri] < len(room_cells[ri]):
                distractor_cells.append(room_cells[ri][used_per_room[ri]])
                used_per_room[ri] += 1
            ri = (ri + 1) % (self.n_keys + 1)
            # Bail out if no rooms have more cells.
            if all(used_per_room[i] >= len(room_cells[i])
                   for i in range(self.n_keys + 1)):
                break

        self._place_tiles(goal_pos, distractor_cells)

    def _place_tiles(self, goal_pos: tuple[int, int],
                     distractor_cells: list[tuple[int, int]]) -> None:
        rng = self._rng
        match_attrs = self.goal.sample_matching_attrs(rng)
        tiles: dict = {goal_pos: Tile(pos=goal_pos, **match_attrs)}
        for pos in distractor_cells:
            for _ in range(20):
                col = rng.choice(COLORS); sh = rng.choice(SHAPES); pat = rng.choice(PATTERNS)
                cand = Tile(pos=pos, color=col, shape=sh, pattern=pat)
                if not self.goal.matches(cand):
                    tiles[pos] = cand
                    break
        self.tiles = tiles

    def _has_reachable_match(self) -> bool:
        """Ensure the oracle can solve the layout given the inventory it
        will accumulate. Equivalent to: BFS over (pos, n_keys, frozenset_keys,
        frozenset_doors) finds the goal."""
        target = next((p for p, t in self.tiles.items() if self.goal.matches(t)), None)
        if target is None:
            return False
        return self._oracle_can_reach(target)

    def _oracle_can_reach(self, target: tuple[int, int]) -> bool:
        start = (self.agent, self.n_keys_held,
                 frozenset(self.key_positions), frozenset(self.doors))
        if self.agent == target:
            return True
        seen = {start}
        q = deque([start])
        while q:
            pos, n_k, keys, closed = q.popleft()
            for a in ACTIONS:
                dx, dy = _DELTAS[a]
                nxt = (pos[0] + dx, pos[1] + dy)
                if not self._in_bounds(nxt) or nxt in self.walls:
                    continue
                if nxt in closed and n_k == 0:
                    continue
                new_n = n_k
                new_closed = closed
                new_keys = keys
                if nxt in closed:
                    new_closed = closed - {nxt}
                    new_n = n_k - 1
                if nxt in keys:
                    new_keys = keys - {nxt}
                    new_n += 1
                state = (nxt, new_n, new_keys, new_closed)
                if state in seen:
                    continue
                if nxt == target:
                    return True
                seen.add(state)
                q.append(state)
        return False


# ── Oracle: BFS over (pos, n_keys, remaining_keys, closed_doors) ───────────

def n_key_door_oracle(env: _NKeyDoorBase) -> Optional[str]:
    """Return the first action of a shortest plan to reach any goal-
    matching tile, accounting for keys and doors. None if unreachable."""
    targets = {p for p, t in env.tiles.items() if env.goal.matches(t)}
    if not targets:
        return None
    if env.agent in targets:
        return ACTIONS[0]  # arbitrary; episode would already be done

    start = (env.agent, env.n_keys_held,
             frozenset(env.key_positions), frozenset(env.doors))
    seen = {start}
    first_action: dict = {start: None}
    q = deque([start])

    while q:
        pos, n_k, keys, closed = q.popleft()
        prev_first = first_action[(pos, n_k, keys, closed)]
        for a in ACTIONS:
            dx, dy = _DELTAS[a]
            nxt = (pos[0] + dx, pos[1] + dy)
            if not env._in_bounds(nxt) or nxt in env.walls:
                continue
            if nxt in closed and n_k == 0:
                continue
            new_n = n_k
            new_closed = closed
            new_keys = keys
            if nxt in closed:
                new_closed = closed - {nxt}
                new_n = n_k - 1
            if nxt in keys:
                new_keys = keys - {nxt}
                new_n += 1
            state = (nxt, new_n, new_keys, new_closed)
            if state in seen:
                continue
            seen.add(state)
            first_action[state] = prev_first if prev_first is not None else a
            if nxt in targets:
                return first_action[state]
            q.append(state)
    return None


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = EnvConfig(width=6, height=6, n_tiles=5, n_walls=0, max_steps=64)
    n_tries = 60
    for n_keys in (0, 1, 2):
        for shape_value in ("circle", "square", "triangle"):
            goal = ShapeGoal(shape_value)
            n_succ = 0; n_steps = []; n_failed_layout = 0
            for ep in range(n_tries):
                try:
                    env = NKeyDoorTrainEnv(cfg, goal, seed=ep, n_keys=n_keys)
                    env.reset()
                except RuntimeError:
                    n_failed_layout += 1
                    continue
                steps = 0
                while not env.is_done() and steps < cfg.max_steps:
                    a = n_key_door_oracle(env)
                    if a is None:
                        break
                    env.step(a)
                    steps += 1
                if env._success:
                    n_succ += 1
                    n_steps.append(steps)
            avg = sum(n_steps) / max(1, len(n_steps)) if n_steps else 0
            print(f"n_keys={n_keys}  goal={shape_value:<8}  "
                  f"oracle success {n_succ}/{n_tries}  "
                  f"failed_layout={n_failed_layout}  "
                  f"avg_steps_succ={avg:.1f}")
