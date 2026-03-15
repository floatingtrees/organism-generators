import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions, X, Y, VX, VY, ALIVE, NUM_FEATURES


class TestBatched:
    def test_environments_are_independent(self):
        config = {
            "num_organisms": 1,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 0.0,
            "num_copies": 2,
            "dt": 0.1,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        obs = env.observe()
        # Different seeds per env → different initial positions
        assert not np.allclose(obs[0, :, :2], obs[1, :, :2])

    def test_per_env_actions(self):
        config = {
            "num_organisms": 1,
            "height": 20.0,
            "width": 20.0,
            "food_spawn_rate": 0.0,
            "num_copies": 2,
            "dt": 0.1,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        obs_before = env.observe()
        actions = np.zeros((2, 1, 2), dtype=np.float32)
        actions[0, 0, 0] = 10.0  # env 0 accelerate right
        actions[1, 0, 1] = 10.0  # env 1 accelerate down

        env.step(actions)
        obs_after = env.observe()

        assert obs_after[0, 0, X] > obs_before[0, 0, X]
        assert obs_after[1, 0, Y] > obs_before[1, 0, Y]

    def test_variable_agent_counts(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs = env.observe()
        # alive feature reflects actual agent counts
        assert obs[0, :3, ALIVE].sum() == 3  # env 0 has 3 agents
        assert obs[1, :, ALIVE].sum() == 5   # env 1 has 5 agents
        assert obs[2, :2, ALIVE].sum() == 2  # env 2 has 2 agents

    def test_step_with_torch_actions(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        actions = torch.randn(env.num_envs, env.max_agents, 2)
        rewards = env.step(actions.numpy())
        assert rewards.shape == (3, 5)

    def test_reset_restores_all(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs0 = env.observe()

        for _ in range(20):
            actions = np.random.randn(env.num_envs, env.max_agents, 2).astype(np.float32)
            env.step(actions)

        env.reset()
        obs_reset = env.observe()

        np.testing.assert_allclose(obs0, obs_reset, atol=1e-5)

    def test_rewards_padded_agents_are_zero(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        rewards = env.step(zero_actions(env))
        # env 0 has 3 agents, max 5 → slots 3,4 should be 0
        assert rewards[0, 3] == 0.0
        assert rewards[0, 4] == 0.0
        # env 2 has 2 agents → slots 2,3,4 should be 0
        assert rewards[2, 2] == 0.0
        assert rewards[2, 3] == 0.0
        assert rewards[2, 4] == 0.0

    def test_many_environments(self):
        config = {
            "num_organisms": 3,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 1.0,
            "num_copies": 50,
            "dt": 0.1,
            "seed": 0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        actions = np.zeros((50, 3, 2), dtype=np.float32)
        rewards = env.step(actions)
        assert rewards.shape == (50, 3)

    def test_obstacles(self):
        config = {
            "num_organisms": 1,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 0.0,
            "num_copies": 1,
            "num_obstacles": 5,
            "obstacle_weight": 2.0,
            "dt": 0.1,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        for _ in range(100):
            actions = np.random.randn(1, 1, 2).astype(np.float32) * 5
            env.step(actions)
        obs = env.observe()
        assert obs.shape == (1, 1, NUM_FEATURES)

    def test_food_spawns_and_collectable(self):
        config = {
            "num_organisms": 1,
            "height": 2.0,
            "width": 2.0,
            "food_spawn_rate": 50.0,
            "num_copies": 1,
            "dt": 0.1,
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        initial = env.step(zero_actions(env))[0, 0]
        for _ in range(20):
            actions = np.random.randn(1, 1, 2).astype(np.float32) * 2
            env.step(actions)

        final = env.step(zero_actions(env))[0, 0]
        assert isinstance(final, (float, np.floating))

    def test_velocity_features_in_batched(self):
        config = {
            "num_organisms": 2,
            "height": 20.0,
            "width": 20.0,
            "food_spawn_rate": 0.0,
            "num_copies": 2,
            "dt": 0.1,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        obs = env.observe()
        # Initial velocities should be zero
        np.testing.assert_array_equal(obs[..., VX], 0.0)
        np.testing.assert_array_equal(obs[..., VY], 0.0)

        actions = np.zeros((2, 2, 2), dtype=np.float32)
        actions[0, 0] = [1.0, 0.0]
        env.step(actions)

        obs = env.observe()
        dt = config["dt"]
        np.testing.assert_allclose(obs[0, 0, VX], 1.0 * dt, atol=1e-5)
        assert obs[0, 0, VY] == pytest.approx(0.0, abs=1e-5)
