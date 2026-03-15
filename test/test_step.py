import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions


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
        obs_before, _ = env.observe()
        env.step(zero_actions(env))
        obs_after, _ = env.observe()
        np.testing.assert_allclose(obs_before, obs_after, atol=1e-6)

    def test_acceleration_moves_agent(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        obs_before, _ = env.observe()
        x0 = obs_before[0, 0, 0]

        actions = zero_actions(env)
        actions[0, 0, 0] = 1.0  # accelerate in +x
        env.step(actions)

        obs_after, _ = env.observe()
        assert obs_after[0, 0, 0] > x0

    def test_acceleration_costs_energy(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        e_before = env.step(zero_actions(env))[0, 0]

        env.reset()
        actions = zero_actions(env)
        actions[0, 0] = [3.0, 4.0]  # magnitude 5
        e_after = env.step(actions)[0, 0]

        dt = deterministic_config["dt"]
        # Energy should decrease by magnitude * dt = 5 * 0.1 = 0.5
        assert e_after < e_before

    def test_velocity_accumulates(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        actions = zero_actions(env)
        actions[0, 0, 0] = 1.0  # constant +x acceleration

        obs0, _ = env.observe()
        env.step(actions)
        obs1, _ = env.observe()
        dx1 = obs1[0, 0, 0] - obs0[0, 0, 0]

        env.step(actions)
        obs2, _ = env.observe()
        dx2 = obs2[0, 0, 0] - obs1[0, 0, 0]

        # Second step should move farther (velocity accumulated)
        assert dx2 > dx1

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
        actions[0, 0] = [100.0, 100.0]  # huge acceleration

        for _ in range(50):
            env.step(actions)

        obs, _ = env.observe()
        assert obs[0, 0, 0] >= 0.0 and obs[0, 0, 0] <= 5.0
        assert obs[0, 0, 1] >= 0.0 and obs[0, 0, 1] <= 5.0

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
        actions[0, 0] = [50.0, 0.0]  # will hit wall
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
            "dt": 1.0,  # threshold = 10s / 1.0dt = 10 steps
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        # Drain energy with huge acceleration
        actions = np.zeros((1, 1, 2), dtype=np.float32)
        actions[0, 0] = [100.0, 0.0]  # burns 100*dt = 100 energy per step
        env.step(actions)

        # Now energy is very negative. Step with zero actions for threshold steps.
        zeroact = zero_actions(env)
        for _ in range(15):
            env.step(zeroact)

        _, mask = env.observe()
        assert mask[0, 0] == 0.0  # agent should be dead

    def test_dead_agent_masked_in_observations(self):
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

        # Kill agent 0 with huge acceleration
        actions = np.zeros((1, 2, 2), dtype=np.float32)
        actions[0, 0] = [200.0, 0.0]
        env.step(actions)

        zeroact = zero_actions(env)
        for _ in range(15):
            env.step(zeroact)

        _, mask = env.observe()
        assert mask[0, 0] == 0.0  # agent 0 dead
        assert mask[0, 1] == 1.0  # agent 1 alive
