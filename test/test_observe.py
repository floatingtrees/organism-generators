import pytest
import numpy as np
import organism_env
from conftest import zero_actions


class TestObserve:
    def test_shapes(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs, mask = env.observe()
        assert obs.shape == (2, 3, 2)   # (num_envs, max_agents, features)
        assert mask.shape == (2, 3)      # (num_envs, max_agents)

    def test_dtypes(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs, mask = env.observe()
        assert obs.dtype == np.float32
        assert mask.dtype == np.float32

    def test_variable_agent_shapes(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs, mask = env.observe()
        # max_agents = max(3, 5, 2) = 5
        assert obs.shape == (3, 5, 2)
        assert mask.shape == (3, 5)

    def test_padding_mask(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        _, mask = env.observe()
        # env 0 has 3 agents → mask[:3]=1, mask[3:]=0
        assert np.all(mask[0, :3] == 1.0)
        assert np.all(mask[0, 3:] == 0.0)
        # env 1 has 5 agents → all 1
        assert np.all(mask[1, :] == 1.0)
        # env 2 has 2 agents → mask[:2]=1, mask[2:]=0
        assert np.all(mask[2, :2] == 1.0)
        assert np.all(mask[2, 2:] == 0.0)

    def test_padded_features_are_zero(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs, _ = env.observe()
        # Padded agent slots should have (0, 0)
        assert np.all(obs[0, 3:] == 0.0)
        assert np.all(obs[2, 2:] == 0.0)

    def test_positions_change_after_step(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs0, _ = env.observe()

        actions = zero_actions(env)
        actions[0, 0] = [5.0, 3.0]
        env.step(actions)

        obs1, _ = env.observe()
        assert not np.allclose(obs0, obs1)

    def test_observe_after_reset(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs0, mask0 = env.observe()

        # Step and change state
        actions = np.ones((env.num_envs, env.max_agents, 2), dtype=np.float32) * 5.0
        for _ in range(10):
            env.step(actions)

        env.reset()
        obs_reset, mask_reset = env.observe()

        # After reset, observations should match initial (same seed)
        np.testing.assert_allclose(obs0, obs_reset, atol=1e-5)
        np.testing.assert_array_equal(mask0, mask_reset)

    def test_positions_within_bounds(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = np.random.randn(env.num_envs, env.max_agents, 2).astype(np.float32) * 10
        for _ in range(100):
            env.step(actions)

        obs, mask = env.observe()
        alive = mask == 1.0
        xs = obs[..., 0][alive]
        ys = obs[..., 1][alive]
        assert np.all(xs >= 0) and np.all(xs <= small_config["width"])
        assert np.all(ys >= 0) and np.all(ys <= small_config["height"])
