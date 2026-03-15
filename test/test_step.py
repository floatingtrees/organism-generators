import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions, X, Y, VX, VY, ALIVE


class TestStep:
    def test_returns_correct_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = zero_actions(env)
        rewards = env.step(actions)
        assert rewards.shape == (env.num_envs, env.max_agents)

    def test_rewards_dtype(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        rewards = env.step(zero_actions(env))
        assert rewards.dtype == np.float32

    def test_torch_tensor_actions(self, small_config):
        """Actions from PyTorch tensors via .numpy()."""
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = torch.zeros(env.num_envs, env.max_agents, 2)
        rewards = env.step(actions.numpy())
        assert rewards.shape == (env.num_envs, env.max_agents)

    def test_wrong_shape_raises(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        wrong = np.zeros((1, 1, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            env.step(wrong)

    def test_zero_action_preserves_position(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs_before = env.observe()
        env.step(zero_actions(env))
        obs_after = env.observe()
        np.testing.assert_allclose(obs_before[..., :2], obs_after[..., :2], atol=1e-6)

    def test_acceleration_moves_agent(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        x0 = env.observe()[0, 0, X]

        actions = zero_actions(env)
        actions[0, 0, 0] = 1.0  # accelerate in +x
        env.step(actions)

        assert env.observe()[0, 0, X] > x0

    def test_acceleration_costs_energy(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        e_before = env.step(zero_actions(env))[0, 0]

        env.reset()
        actions = zero_actions(env)
        actions[0, 0] = [3.0, 4.0]  # magnitude 5
        e_after = env.step(actions)[0, 0]

        assert e_after < e_before

    def test_velocity_accumulates(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        actions = zero_actions(env)
        actions[0, 0, 0] = 1.0  # constant +x acceleration

        x0 = env.observe()[0, 0, X]
        env.step(actions)
        x1 = env.observe()[0, 0, X]
        dx1 = x1 - x0

        env.step(actions)
        x2 = env.observe()[0, 0, X]
        dx2 = x2 - x1

        # Second step should move farther (velocity accumulated)
        assert dx2 > dx1

    def test_velocity_in_features(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs0 = env.observe()
        assert obs0[0, 0, VX] == 0.0
        assert obs0[0, 0, VY] == 0.0

        actions = zero_actions(env)
        actions[0, 0] = [5.0, -3.0]
        env.step(actions)

        obs1 = env.observe()
        dt = deterministic_config["dt"]
        np.testing.assert_allclose(obs1[0, 0, VX], 5.0 * dt, atol=1e-5)
        np.testing.assert_allclose(obs1[0, 0, VY], -3.0 * dt, atol=1e-5)

    def test_initial_energy_is_10(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        rewards = env.step(zero_actions(env))
        assert abs(rewards[0, 0] - 10.0) < 1e-4

    def test_multiple_steps(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        for _ in range(100):
            rewards = env.step(zero_actions(env))
        assert rewards.shape == (env.num_envs, env.max_agents)


class TestWallBounce:
    def test_agent_stays_in_bounds(self):
        config = {
            "num_organisms": 1,
            "height": 5.0,
            "width": 5.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "dt": 0.5,
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        actions = np.zeros((1, 1, 2), dtype=np.float32)
        actions[0, 0] = [100.0, 100.0]

        for _ in range(50):
            env.step(actions)

        obs = env.observe()
        assert obs[0, 0, X] >= 0.0 and obs[0, 0, X] <= 5.0
        assert obs[0, 0, Y] >= 0.0 and obs[0, 0, Y] <= 5.0

    def test_wall_bounce_costs_energy(self):
        config = {
            "num_organisms": 1,
            "height": 5.0,
            "width": 5.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "dt": 0.5,
            "energy_loss": 0.2,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        e0 = env.step(zero_actions(env))[0, 0]

        env.reset()
        actions = np.zeros((1, 1, 2), dtype=np.float32)
        actions[0, 0] = [50.0, 0.0]
        for _ in range(5):
            env.step(actions)

        e_after = env.step(zero_actions(env))[0, 0]
        assert e_after < e0


class TestAgentDeath:
    def test_agent_dies_after_threshold(self):
        config = {
            "num_organisms": 1,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "dt": 1.0,
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        actions = np.zeros((1, 1, 2), dtype=np.float32)
        actions[0, 0] = [100.0, 0.0]
        env.step(actions)

        zeroact = zero_actions(env)
        for _ in range(15):
            env.step(zeroact)

        obs = env.observe()
        assert obs[0, 0, ALIVE] == 0.0  # agent should be dead

    def test_dead_agent_in_observations(self):
        config = {
            "num_organisms": 2,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "dt": 1.0,
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        actions = np.zeros((1, 2, 2), dtype=np.float32)
        actions[0, 0] = [200.0, 0.0]
        env.step(actions)

        zeroact = zero_actions(env)
        for _ in range(15):
            env.step(zeroact)

        obs = env.observe()
        assert obs[0, 0, ALIVE] == 0.0  # agent 0 dead
        assert obs[0, 1, ALIVE] == 1.0  # agent 1 alive
