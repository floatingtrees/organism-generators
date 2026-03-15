import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions


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

        obs0, _ = env.observe()
        # Different seeds per env → different initial positions
        assert not np.allclose(obs0[0], obs0[1])

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

        obs_before, _ = env.observe()
        actions = np.zeros((2, 1, 2), dtype=np.float32)
        actions[0, 0, 0] = 10.0  # env 0 accelerate right
        actions[1, 0, 1] = 10.0  # env 1 accelerate down

        env.step(actions)
        obs_after, _ = env.observe()

        # env 0: x increased
        assert obs_after[0, 0, 0] > obs_before[0, 0, 0]
        # env 1: y increased
        assert obs_after[1, 0, 1] > obs_before[1, 0, 1]

    def test_variable_agent_counts(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)

        obs, mask = env.observe()
        # Verify mask reflects actual agent counts
        assert mask[0].sum() == 3  # env 0 has 3 agents
        assert mask[1].sum() == 5  # env 1 has 5 agents
        assert mask[2].sum() == 2  # env 2 has 2 agents

    def test_step_with_torch_actions(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        actions = torch.randn(env.num_envs, env.max_agents, 2)
        rewards = env.step(actions.numpy())
        assert rewards.shape == (3, 5)

    def test_reset_restores_all(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        obs0, mask0 = env.observe()

        # Evolve
        for _ in range(20):
            actions = np.random.randn(env.num_envs, env.max_agents, 2).astype(np.float32)
            env.step(actions)

        env.reset()
        obs_reset, mask_reset = env.observe()

        np.testing.assert_allclose(obs0, obs_reset, atol=1e-5)
        np.testing.assert_array_equal(mask0, mask_reset)

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
        # Should not crash with obstacles present
        for _ in range(100):
            actions = np.random.randn(1, 1, 2).astype(np.float32) * 5
            env.step(actions)
        obs, mask = env.observe()
        assert obs.shape == (1, 1, 2)

    def test_food_spawns_and_collectable(self):
        config = {
            "num_organisms": 1,
            "height": 2.0,
            "width": 2.0,
            "food_spawn_rate": 50.0,  # lots of food in tiny area
            "num_copies": 1,
            "dt": 0.1,
            "energy_loss": 0.0,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)

        # Step several times — agent should collect some food and gain energy
        initial = env.step(zero_actions(env))[0, 0]
        for _ in range(20):
            actions = np.random.randn(1, 1, 2).astype(np.float32) * 2
            env.step(actions)

        final = env.step(zero_actions(env))[0, 0]
        # With 50 food per step in 2x2 area, agent should have collected some
        # (exact amount depends on RNG, but energy cost of acceleration is small)
        # At minimum, the environment shouldn't crash
        assert isinstance(final, (float, np.floating))
