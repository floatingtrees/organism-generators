import pytest
import numpy as np
import organism_env


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
            "num_organisms": 4,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 1.0,
            "num_copies": 3,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        assert env.num_envs == 3
        assert env.max_agents == 4

    def test_list_num_organisms(self):
        config = {
            "num_organisms": [1, 7, 3],
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 1.0,
            "num_copies": 3,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        assert env.max_agents == 7

    def test_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            organism_env.EvolutionEnv.initialize({"num_organisms": 3})

    def test_mismatched_list_length_raises(self):
        config = {
            "num_organisms": [1, 2],  # length 2 != num_copies 3
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 1.0,
            "num_copies": 3,
        }
        with pytest.raises(ValueError):
            organism_env.EvolutionEnv.initialize(config)

    def test_default_dt(self):
        config = {
            "num_organisms": 1,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 1.0,
            "num_copies": 1,
        }
        # Should not raise — dt defaults to 0.5
        env = organism_env.EvolutionEnv.initialize(config)
        assert env.num_envs == 1

    def test_custom_rules(self):
        config = {
            "num_organisms": 2,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "rules": {
                "wall_bounce": False,
                "food_collection": False,
                "obstacle_collision": False,
                "agent_collision": True,
            },
        }
        env = organism_env.EvolutionEnv.initialize(config)
        assert env.num_envs == 1

    def test_agents_start_alive(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        _, mask = env.observe()
        # All agents should be alive initially
        assert np.all(mask == 1.0)

    def test_agents_within_bounds(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs, _ = env.observe()
        h = small_config["height"]
        w = small_config["width"]
        assert np.all(obs[..., 0] >= 0.0)
        assert np.all(obs[..., 0] <= w)
        assert np.all(obs[..., 1] >= 0.0)
        assert np.all(obs[..., 1] <= h)
