import pytest
import numpy as np
import organism_env
from conftest import zero_actions, TOTAL_CHANNELS, NUM_ACTIONS


class TestObserve:
    def test_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs = env.observe()
        r = env.view_res
        assert obs.shape == (2, 3, r, r, TOTAL_CHANNELS)

    def test_dtype(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs = env.observe()
        assert obs.dtype == np.float32

    def test_variable_agent_shape(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        r = env.view_res
        assert obs.shape == (3, 5, r, r, TOTAL_CHANNELS)

    def test_padded_slots_are_zero(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        # env 0 has 3 agents → slots 3,4 are zero
        np.testing.assert_array_equal(obs[0, 3:], 0.0)
        # env 2 has 2 agents → slots 2,3,4 are zero
        np.testing.assert_array_equal(obs[2, 2:], 0.0)

    def test_alive_mask_shape(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        mask = env.alive_mask()
        assert mask.shape == (3, 5)
        assert mask[0, :3].sum() == 3
        assert mask[1, :5].sum() == 5
        assert mask[2, :2].sum() == 2

    def test_food_appears_in_observation(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 200.0, "num_copies": 1, "dt": 0.1,
            "food_cap": 50, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        env.step(zero_actions(env))  # spawn food
        obs = env.observe()
        # Channel 0 = food (current frame)
        assert (obs[0, 0, :, :, 0] > 0).any(), "food should be visible"

    def test_history_channels_populate(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 200.0, "num_copies": 1, "dt": 0.1,
            "food_cap": 50, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        for _ in range(5):
            env.step(zero_actions(env))
        obs = env.observe()
        # History channels (4..16) should have some data
        assert (obs[0, 0, :, :, 4:] > 0).any(), "history should be populated"

    def test_observe_after_reset(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = np.ones((env.num_envs, env.max_agents, NUM_ACTIONS), dtype=np.float32) * 5.0
        for _ in range(10):
            env.step(actions)
        env.reset()
        mask = env.alive_mask()
        assert np.all(mask == 1.0)

    def test_reset_produces_different_obs(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs0 = env.observe().copy()
        env.reset()
        obs1 = env.observe()
        assert not np.array_equal(obs0, obs1)
