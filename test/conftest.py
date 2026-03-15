import pytest
import numpy as np


@pytest.fixture
def small_config():
    """Small uniform environment for quick tests."""
    return {
        "num_organisms": 3,
        "height": 10.0,
        "width": 10.0,
        "food_spawn_rate": 2.0,
        "num_copies": 2,
        "dt": 0.1,
        "energy_loss": 0.1,
        "seed": 42,
    }


@pytest.fixture
def variable_config():
    """Variable-agent-count environments."""
    return {
        "num_organisms": [3, 5, 2],
        "height": 10.0,
        "width": 10.0,
        "food_spawn_rate": 1.0,
        "num_copies": 3,
        "dt": 0.5,
        "energy_loss": 0.05,
        "seed": 123,
    }


@pytest.fixture
def deterministic_config():
    """Single env, no food spawning, for precise physics checks."""
    return {
        "num_organisms": 1,
        "height": 20.0,
        "width": 20.0,
        "food_spawn_rate": 0.0,
        "num_copies": 1,
        "dt": 0.1,
        "energy_loss": 0.1,
        "seed": 0,
    }


def zero_actions(env):
    """Create a zero-action tensor matching the environment shape."""
    return np.zeros((env.num_envs, env.max_agents, 2), dtype=np.float32)
