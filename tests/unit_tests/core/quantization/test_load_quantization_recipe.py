# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64

from megatron.core.file_arg_utils import PSEUDO_FILE_PREFIX
from megatron.core.quantization.utils import load_quantization_recipe

RECIPE_YAML = """
configs:
  fp8_e4m3:
    kitchen_config_type: QLinearParams
    quant_algo: FP8_CS
matchers:
  attn_qkv_fp8:
    config: "fp8_e4m3"
    type: "glob"
    pattern: "*.linear_qkv"
    enabled: true
"""


class TestLoadQuantizationRecipe:
    def test_loads_a_recipe_from_a_path(self, tmp_path):
        """The pre-existing path form must keep working for launchers that share a filesystem."""
        path = tmp_path / "recipe.yaml"
        path.write_text(RECIPE_YAML)

        recipe = load_quantization_recipe(str(path))

        assert "fp8_e4m3" in recipe.configs

    def test_loads_the_same_recipe_from_an_inline_payload(self, tmp_path):
        """The base64: wire format is what miles sends; a path and a payload must agree exactly."""
        path = tmp_path / "recipe.yaml"
        path.write_text(RECIPE_YAML)
        encoded = base64.b64encode(RECIPE_YAML.encode()).decode()

        from_payload = load_quantization_recipe(f"{PSEUDO_FILE_PREFIX}{encoded}")
        from_path = load_quantization_recipe(str(path))

        assert from_payload.configs == from_path.configs

    def test_an_inline_recipe_keeps_its_matchers(self, tmp_path):
        """Dropping matchers would silently quantize nothing while the recipe still looks loaded."""
        encoded = base64.b64encode(RECIPE_YAML.encode()).decode()

        recipe = load_quantization_recipe(f"{PSEUDO_FILE_PREFIX}{encoded}")

        assert len(recipe.matchers) == 1
