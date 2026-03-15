import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions, TOTAL_CHANNELS, NUM_ACTIONS


class TestBatched:
    def test_environments_are_independent(self):
        config = {
            "num_organisms": 3, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 200.0, "food_cap": 50,
            "num_copies": 2, "dt": 0.1, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        env.step(zero_actions(env))  # spawn food so views differ
        obs = env.observe()
        # Different seeds → different agent positions → different views
        assert not np.array_equal(obs[0], obs[1])

    def test_variable_agent_counts(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        mask = env.alive_mask()
        assert mask[0, :3].sum() == 3
        assert mask[1, :5].sum() == 5
        assert mask[2, :2].sum() == 2

    def test_step_with_torch_actions(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        actions = torch.randn(env.num_envs, env.max_agents, NUM_ACTIONS)
        rewards = env.step(actions.numpy())
        assert rewards.shape == (3, 5)

    def test_reset_restores_all(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        for _ in range(20):
            actions = np.random.randn(env.num_envs, env.max_agents, NUM_ACTIONS).astype(np.float32)
            env.step(actions)
        env.reset()
        mask = env.alive_mask()
        assert np.all(mask[0, :3] == 1.0)
        assert np.all(mask[1, :5] == 1.0)
        assert np.all(mask[2, :2] == 1.0)
        assert np.all(mask[0, 3:] == 0.0)

    def test_rewards_padded_agents_are_zero(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        rewards = env.step(zero_actions(env))
        assert rewards[0, 3] == 0.0
        assert rewards[0, 4] == 0.0
        assert rewards[2, 2] == 0.0

    def test_many_environments(self):
        config = {
            "num_organisms": 3, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 10.0, "num_copies": 50, "dt": 0.1, "seed": 0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        actions = np.zeros((50, 3, NUM_ACTIONS), dtype=np.float32)
        rewards = env.step(actions)
        assert rewards.shape == (50, 3)

    def test_obstacles(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 0.0, "num_copies": 1,
            "num_obstacles": 5, "obstacle_weight": 2.0,
            "dt": 0.1, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        for _ in range(100):
            actions = np.random.randn(1, 1, NUM_ACTIONS).astype(np.float32) * 5
            env.step(actions)
        obs = env.observe()
        r = env.view_res
        assert obs.shape == (1, 1, r, r, TOTAL_CHANNELS)

    def test_vision_cost(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 0.0, "num_copies": 1, "dt": 0.1,
            "seed": 42, "vision_cost": 1.0, "initial_view_size": 2.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        e0 = env.step(zero_actions(env))[0, 0]
        env.reset()
        # vision_cost * view_size * dt = 1.0 * 2.0 * 0.1 = 0.2 per step
        e1 = env.step(zero_actions(env))[0, 0]
        assert e1 < 10.0  # should have lost energy
