from random import Random

import pytest

from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld import (
    ACTIONS,
    Env,
    EnvConfig,
    Tile,
)


def make_env(seed: int = 0, **cfg_overrides) -> Env:
    cfg = EnvConfig(**cfg_overrides)
    goal = SimpleFeatureGoal(attribute="color", value="blue")
    env = Env(cfg, goal, seed=seed)
    env.reset()
    return env


def test_state_dict_shape():
    env = make_env(seed=0)
    s = env.state_dict()
    assert s["grid_size"] == [8, 8]
    assert len(s["agent"]) == 2
    assert all(len(w) == 2 for w in s["walls"])
    assert s["previous_actions"] == []
    for t in s["tiles"]:
        assert set(t.keys()) == {"pos", "color", "shape", "pattern"}


def test_reset_guarantees_reachable_match():
    goal = SimpleFeatureGoal(attribute="color", value="red")
    for seed in range(20):
        env = Env(EnvConfig(), goal, seed=seed)
        env.reset()
        # at least one tile matches goal
        matches = [t for t in env.tiles.values() if goal.matches(t)]
        assert matches, f"no matching tile at seed {seed}"


def test_walls_block_movement():
    env = make_env(seed=1)
    # Place a known wall directly to the right of agent.
    ax, ay = env.agent
    if ax + 1 >= env.config.width:
        pytest.skip("agent on right edge for this seed; rerun with different seed")
    env.walls.add((ax + 1, ay))
    env.tiles.pop((ax + 1, ay), None)
    before = env.agent
    res = env.step("right")
    assert env.agent == before
    assert "right" in res.state["previous_actions"]


def test_oob_blocked_but_action_recorded():
    env = make_env(seed=2)
    env.agent = (0, 0)
    env.walls.discard((0, 0))
    res = env.step("up")
    assert env.agent == (0, 0)
    assert res.state["previous_actions"][-1] == "up"


def test_collection_and_success():
    goal = SimpleFeatureGoal(attribute="color", value="blue")
    env = Env(EnvConfig(), goal, seed=3)
    env.reset()
    # Force a known matching tile adjacent to the agent.
    ax, ay = env.agent
    target = None
    for dx, dy, action in [(1, 0, "right"), (-1, 0, "left"),
                           (0, 1, "down"), (0, -1, "up")]:
        np = (ax + dx, ay + dy)
        if 0 <= np[0] < env.config.width and 0 <= np[1] < env.config.height:
            target = (np, action)
            break
    assert target is not None
    pos, action = target
    env.walls.discard(pos)
    env.tiles[pos] = Tile(pos=pos, color="blue", shape="circle", pattern="solid")
    res = env.step(action)
    assert res.collected is not None
    assert res.success is True
    assert res.done is True


def test_truncation_on_step_limit():
    goal = SimpleFeatureGoal(attribute="color", value="blue")
    env = Env(EnvConfig(max_steps=3), goal, seed=4)
    env.reset()
    # agent will take at most 3 actions; pick something safe
    last = None
    for _ in range(3):
        last = env.step("up")
    assert last.done is True
    # success may or may not be true depending on placement; if not, must be truncated
    if not last.success:
        assert last.truncated is True


def test_history_truncation():
    env = make_env(seed=5, history_len=3, max_steps=10)
    for a in ["up", "down", "left", "right", "up"]:
        if env.is_done():
            break
        env.step(a)
    assert len(env.state_dict()["previous_actions"]) <= 3


def test_actions_constant():
    assert ACTIONS == ("up", "down", "left", "right")


def test_simple_goal_matches():
    g = SimpleFeatureGoal(attribute="shape", value="triangle")
    t1 = Tile(pos=(0, 0), color="red", shape="triangle", pattern="solid")
    t2 = Tile(pos=(0, 0), color="red", shape="circle", pattern="solid")
    assert g.matches(t1)
    assert not g.matches(t2)
    assert g.matches_any([t2, t1])
    assert not g.matches_any([t2])


def test_simple_goal_sample_matching_attrs():
    rng = Random(0)
    g = SimpleFeatureGoal(attribute="pattern", value="striped")
    for _ in range(20):
        attrs = g.sample_matching_attrs(rng)
        assert attrs["pattern"] == "striped"


def test_simple_goal_description():
    assert (
        SimpleFeatureGoal(attribute="color", value="blue").description
        == "collect a blue tile"
    )
    assert (
        SimpleFeatureGoal(attribute="shape", value="triangle").description
        == "collect a triangle tile"
    )
    assert (
        SimpleFeatureGoal(attribute="shape", value="star").description
        == "collect a star tile"
    )


def test_invalid_action_raises():
    env = make_env(seed=6)
    with pytest.raises(ValueError):
        env.step("diagonal")


def test_step_after_done_raises():
    goal = SimpleFeatureGoal(attribute="color", value="blue")
    env = Env(EnvConfig(max_steps=1), goal, seed=7)
    env.reset()
    env.step("up")
    with pytest.raises(RuntimeError):
        env.step("up")
