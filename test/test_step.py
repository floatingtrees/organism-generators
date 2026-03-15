import pytest
import numpy as np
import torch
import organism_env
from conftest import zero_actions, NUM_ACTIONS


class TestStep:
    def test_returns_correct_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        rewards = env.step(zero_actions(env))
        assert rewards.shape == (env.num_envs, env.max_agents)

    def test_rewards_dtype(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        rewards = env.step(zero_actions(env))
        assert rewards.dtype == np.float32

    def test_torch_tensor_actions(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        actions = torch.zeros(env.num_envs, env.max_agents, NUM_ACTIONS)
        rewards = env.step(actions.numpy())
        assert rewards.shape == (env.num_envs, env.max_agents)

    def test_wrong_shape_raises(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        wrong = np.zeros((1, 1, NUM_ACTIONS), dtype=np.float32)
        with pytest.raises(ValueError):
            env.step(wrong)

    def test_initial_energy_is_10(self, deterministic_config):
        env = organism_env.EvolutionEnv.initialize(deterministic_config)
        rewards = env.step(zero_actions(env))
        assert abs(rewards[0, 0] - 10.0) < 0.5  # allow small vision cost

    def test_view_delta_changes_view(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 200.0, "food_cap": 50,
            "num_copies": 1, "dt": 0.1, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        # Step once to spawn food
        env.step(zero_actions(env))
        obs0 = env.observe().copy()

        # Increase view size significantly
        actions = zero_actions(env)
        actions[0, 0, 2] = 50.0  # large view_delta
        env.step(actions)
        obs1 = env.observe()

        # Observation should change (wider view sees food at different scale)
        assert not np.array_equal(obs0, obs1)

    def test_multiple_steps(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        for _ in range(100):
            rewards = env.step(zero_actions(env))
        assert rewards.shape == (env.num_envs, env.max_agents)


class TestWallBounce:
    def test_wall_bounce_costs_energy(self):
        config = {
            "num_organisms": 1, "height": 5.0, "width": 5.0,
            "food_spawn_rate": 0.0, "num_copies": 1, "dt": 0.5,
            "energy_loss": 0.2, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        e0 = env.step(zero_actions(env))[0, 0]

        env.reset()
        actions = np.zeros((1, 1, NUM_ACTIONS), dtype=np.float32)
        actions[0, 0, 0] = 50.0
        for _ in range(5):
            env.step(actions)
        e_after = env.step(zero_actions(env))[0, 0]
        assert e_after < e0


class TestAgentDeath:
    def test_agent_dies_after_threshold(self):
        config = {
            "num_organisms": 1, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 0.0, "num_copies": 1, "dt": 1.0,
            "energy_loss": 0.0, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        actions = np.zeros((1, 1, NUM_ACTIONS), dtype=np.float32)
        actions[0, 0, 0] = 100.0
        env.step(actions)

        for _ in range(15):
            env.step(zero_actions(env))

        mask = env.alive_mask()
        assert mask[0, 0] == 0.0

    def test_dead_agent_masked(self):
        config = {
            "num_organisms": 2, "height": 10.0, "width": 10.0,
            "food_spawn_rate": 0.0, "num_copies": 1, "dt": 1.0,
            "energy_loss": 0.0, "seed": 42, "vision_cost": 0.0,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        actions = np.zeros((1, 2, NUM_ACTIONS), dtype=np.float32)
        actions[0, 0, 0] = 200.0
        env.step(actions)

        for _ in range(15):
            env.step(zero_actions(env))

        mask = env.alive_mask()
        assert mask[0, 0] == 0.0
        assert mask[0, 1] == 1.0
