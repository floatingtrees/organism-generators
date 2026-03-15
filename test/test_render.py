import os
import tempfile
import pytest
import numpy as np
import organism_env
from conftest import zero_actions


class TestRender:
    def test_saves_png(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            env.render(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_custom_ppu(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            env.render(path, pixels_per_unit=50.0)
            # Higher resolution → bigger file
            size_50 = os.path.getsize(path)

            env.render(path, pixels_per_unit=10.0)
            size_10 = os.path.getsize(path)

            assert size_50 > size_10
        finally:
            os.unlink(path)

    def test_render_different_envs(self, variable_config):
        env = organism_env.EvolutionEnv.initialize(variable_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            p0 = os.path.join(tmpdir, "env0.png")
            p1 = os.path.join(tmpdir, "env1.png")
            env.render(p0, env_index=0)
            env.render(p1, env_index=1)
            assert os.path.getsize(p0) > 0
            assert os.path.getsize(p1) > 0
            # Different env states → different images
            with open(p0, "rb") as f0, open(p1, "rb") as f1:
                assert f0.read() != f1.read()

    def test_render_out_of_range_raises(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with pytest.raises(IndexError):
            env.render("/tmp/test_oob.png", env_index=99)

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
        env.step(np.zeros((1, 1, 2), dtype=np.float32))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            env.render(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_render_after_steps(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        for _ in range(10):
            env.step(zero_actions(env))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            env.render(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_render_returns_none(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = env.render(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_png_is_valid(self, small_config):
        """Verify the file starts with a PNG signature."""
        env = organism_env.EvolutionEnv.initialize(small_config)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            env.render(path)
            with open(path, "rb") as f:
                header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n"
        finally:
            os.unlink(path)
