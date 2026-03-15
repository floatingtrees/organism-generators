import pytest
import numpy as np
import organism_env
from conftest import zero_actions


class TestRender:
    def test_default_render_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render()
        # 10.0 * 20 ppu = 200 pixels per side
        assert img.shape == (200, 200, 3)

    def test_render_dtype(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render()
        assert img.dtype == np.uint8

    def test_custom_ppu(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render(pixels_per_unit=10.0)
        assert img.shape == (100, 100, 3)

    def test_higher_resolution(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render(pixels_per_unit=50.0)
        assert img.shape == (500, 500, 3)

    def test_render_different_envs(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        img0 = env.render(env_index=0)
        img1 = env.render(env_index=1)
        img2 = env.render(env_index=2)
        # All should have same dimensions
        assert img0.shape == img1.shape == img2.shape
        # But content should differ (different agent positions)
        assert not np.array_equal(img0, img1)

    def test_render_out_of_range_raises(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with pytest.raises(IndexError):
            env.render(env_index=99)

    def test_render_has_agent_pixels(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render(pixels_per_unit=20.0)
        # Agents should produce bright pixels (white center markers)
        bright = np.any(img > 200, axis=2)
        assert np.sum(bright) > 0

    def test_render_with_food(self):
        config = {
            "num_organisms": 1,
            "height": 10.0,
            "width": 10.0,
            "food_spawn_rate": 10.0,
            "num_copies": 1,
            "dt": 0.1,
            "seed": 42,
        }
        env = organism_env.EvolutionEnv.initialize(config)
        env.step(np.zeros((1, 1, 2), dtype=np.float32))  # spawn food
        img = env.render()
        # Should have green-ish pixels from food
        green_dominant = img[:, :, 1] > np.maximum(img[:, :, 0], img[:, :, 2])
        assert np.sum(green_dominant) > 0

    def test_render_after_steps(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        for _ in range(10):
            env.step(zero_actions(env))
        img = env.render()
        assert img.shape[2] == 3  # still valid RGB
