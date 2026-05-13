from goal_detector.goals import SimpleFeatureGoal
from goal_detector.gridworld import Env, EnvConfig, Tile
from goal_detector.policies.oracle import bfs_optimal_action


def test_oracle_solves_random_layouts():
    """The oracle should reach the goal on every layout the env produces."""
    for seed in range(50):
        for goal in (
            SimpleFeatureGoal("color", "blue"),
            SimpleFeatureGoal("shape", "triangle"),
            SimpleFeatureGoal("pattern", "striped"),
        ):
            env = Env(EnvConfig(max_steps=200), goal, seed=seed)
            env.reset()
            for _ in range(200):
                a = bfs_optimal_action(env)
                assert a is not None, f"oracle returned None at seed={seed}, goal={goal}"
                res = env.step(a)
                if res.success:
                    break
            assert env._success, f"oracle failed at seed={seed}, goal={goal}"


def test_oracle_first_step_toward_goal():
    """In a minimal hand-crafted layout, the oracle should pick the obvious action."""
    goal = SimpleFeatureGoal("color", "blue")
    env = Env(EnvConfig(), goal, seed=0)
    env.reset()
    # Override layout: agent at (3,4), single matching tile at (3,1), no walls
    env.agent = (3, 4)
    env.walls = set()
    env.tiles = {
        (3, 1): Tile(pos=(3, 1), color="blue", shape="circle", pattern="solid")
    }
    assert bfs_optimal_action(env) == "up"

    env.tiles = {
        (3, 7): Tile(pos=(3, 7), color="blue", shape="circle", pattern="solid")
    }
    assert bfs_optimal_action(env) == "down"

    env.tiles = {
        (1, 4): Tile(pos=(1, 4), color="blue", shape="circle", pattern="solid")
    }
    assert bfs_optimal_action(env) == "left"

    env.tiles = {
        (7, 4): Tile(pos=(7, 4), color="blue", shape="circle", pattern="solid")
    }
    assert bfs_optimal_action(env) == "right"


def test_oracle_routes_around_walls():
    """If the direct path is blocked, the oracle should still find a route."""
    goal = SimpleFeatureGoal("color", "blue")
    env = Env(EnvConfig(), goal, seed=0)
    env.reset()
    env.agent = (3, 4)
    env.tiles = {
        (3, 2): Tile(pos=(3, 2), color="blue", shape="circle", pattern="solid")
    }
    env.walls = {(3, 3), (2, 3), (4, 3)}  # wall above agent except diagonals; up is blocked
    a = bfs_optimal_action(env)
    # Must be left or right (the only two ways around the wall row)
    assert a in {"left", "right"}, a
