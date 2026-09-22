# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    is_gated_delta_net_variant as module_specs_is_gdn,
    is_linear_attention_variant,
)
from megatron.core.transformer.transformer_config import is_gated_delta_net_variant


def test_gdn_is_an_alias_of_gated_delta_net():
    for name in ("gated_delta_net", "gdn"):
        assert is_gated_delta_net_variant(name)
        assert module_specs_is_gdn(name)
        assert is_linear_attention_variant(name)


def test_other_variants_are_not_gated_delta_net():
    for name in (None, "dsa", "dsv4_hybrid", "dsv4", "gdn2"):
        assert not is_gated_delta_net_variant(name)
        assert not is_linear_attention_variant(name)
