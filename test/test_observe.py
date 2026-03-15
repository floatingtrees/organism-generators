import pytest
import numpy as np
import organism_env
from conftest import zero_actions, X, Y, VX, VY, ALIVE, NUM_FEATURES


class TestObserve:
    def test_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs = env.observe()
        assert obs.shape == (2, 3, NUM_FEATURES)

    def test_dtype(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs = env.observe()
        assert obs.dtype == np.float32

    def test_variable_agent_shape(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        # max_agents = max(3, 5, 2) = 5
        assert obs.shape == (3, 5, NUM_FEATURES)

    def test_alive_feature_for_real_agents(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        # env 0 has 3 agents → alive=1 for first 3
        assert np.all(obs[0, :3, ALIVE] == 1.0)
        # env 1 has 5 agents → alive=1 for all
        assert np.all(obs[1, :, ALIVE] == 1.0)
        # env 2 has 2 agents → alive=1 for first 2
        assert np.all(obs[2, :2, ALIVE] == 1.0)

    def test_padded_slots_are_zero(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        # Padded agent slots should have all features = 0
        np.testing.assert_array_equal(obs[0, 3:], 0.0)
        np.testing.assert_array_equal(obs[2, 2:], 0.0)

    def test_positions_change_after_step(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs0 = env.observe()

        actions = zero_actions(env)
        actions[0, 0] = [5.0, 3.0]
        env.step(actions)

        obs1 = env.observe()
        assert not np.allclose(obs0[..., :2], obs1[..., :2])

    def test_velocity_features(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs = env.observe()
        # Initial velocity should be zero
        assert obs[0, 0, VX] == 0.0
        assert obs[0, 0, VY] == 0.0

        actions = zero_actions(env)
        actions[0, 0] = [2.0, -1.0]
        env.step(actions)

        obs = env.observe()
        dt = deterministic_config["dt"]
        np.testing.assert_allclose(obs[0, 0, VX], 2.0 * dt, atol=1e-5)
        np.testing.assert_allclose(obs[0, 0, VY], -1.0 * dt, atol=1e-5)

    def test_observe_after_reset(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        obs0 = env.observe()

        actions = np.ones((env.num_envs, env.max_agents, 2), dtype=np.float32) * 5.0
        for _ in range(10):
            env.step(actions)

        env.reset()
        obs_reset = env.observe()

        np.testing.assert_allclose(obs0, obs_reset, atol=1e-5)

    def test_positions_within_bounds(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = np.random.randn(env.num_envs, env.max_agents, 2).astype(np.float32) * 10
        for _ in range(100):
            env.step(actions)

        obs = env.observe()
        alive = obs[..., ALIVE] == 1.0
        xs = obs[..., X][alive]
        ys = obs[..., Y][alive]
        assert np.all(xs >= 0) and np.all(xs <= small_config["width"])
        assert np.all(ys >= 0) and np.all(ys <= small_config["height"])
