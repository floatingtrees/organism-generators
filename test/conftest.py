import pytest
import numpy as np


@pytest.fixture
def small_config():
    return {
        "num_organisms": 3,
        "height": 10.0,
        "width": 10.0,
        "food_spawn_rate": 25.0,
        "num_copies": 2,
        "dt": 0.1,
        "energy_loss": 0.1,
        "seed": 42,
        "vision_cost": 0.0,
    }


@pytest.fixture
def variable_config():
    return {
        "num_organisms": [3, 5, 2],
        "height": 10.0,
        "width": 10.0,
        "food_spawn_rate": 10.0,
        "num_copies": 3,
        "dt": 0.5,
        "energy_loss": 0.05,
        "seed": 123,
        "vision_cost": 0.0,
    }


@pytest.fixture
def deterministic_config():
    return {
        "num_organisms": 1,
        "height": 20.0,
        "width": 20.0,
        "food_spawn_rate": 0.0,
        "num_copies": 1,
        "dt": 0.1,
        "energy_loss": 0.1,
        "seed": 0,
        "vision_cost": 0.0,
    }


NUM_ACTIONS = 3
TOTAL_CHANNELS = 16


def zero_actions(env):
    return np.zeros((env.num_envs, env.max_agents, NUM_ACTIONS), dtype=np.float32)
