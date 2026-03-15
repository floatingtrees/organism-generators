import pytest
import numpy as np
import organism_env
from conftest import TOTAL_CHANNELS


class TestInitialize:
    def test_basic_creation(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        assert env.num_envs == 2
        assert env.max_agents == 3

    def test_variable_agents(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        assert env.num_envs == 3
        assert env.max_agents == 5

    def test_scalar_num_organisms(self):
        config = {
            "num_organisms": 4, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 1.0, "num_copies": 3,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        assert env.num_envs == 3
        assert env.max_agents == 4

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            organism_env.EvolutionEnv.initialize({"num_organisms": 3})

    def test_mismatched_list_length_raises(self):
        config = {
            "num_organisms": [1, 2], "height": 10.0, "width": 10.0,
            "food_spawn_rate": 1.0, "num_copies": 3,
        }
        with pytest.raises(ValueError):
            organism_env.EvolutionEnv.initialize(config)

    def test_agents_start_alive(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        mask = env.alive_mask()
        assert np.all(mask == 1.0)

    def test_observe_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs = env.observe()
        assert obs.shape == (2, 3, env.view_res, env.view_res, TOTAL_CHANNELS)

    def test_properties(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        assert env.view_res == 32
        assert env.num_actions == 3
        assert env.total_channels == 16
