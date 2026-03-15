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

    def test_render_out_of_range_raises(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        with pytest.raises(IndexError):
            env.render("/tmp/test_oob.png", env_index=99)

    def test_render_array_shape(self, small_config):
        env = organism_env.EvolutionEnv.initialize(small_config)
        img = env.render_array()
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_png_is_valid(self, small_config):
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
