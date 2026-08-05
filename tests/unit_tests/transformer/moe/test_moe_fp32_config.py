# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.transformer.transformer_config import TransformerConfig


STANDARD_PRECISION_CONFIGS = [
    pytest.param({}, id="unquantized"),
    *[
        pytest.param({"fp8": fp8, "fp8_recipe": recipe}, id=f"fp8-{fp8}-{recipe}")
        for fp8 in ("e4m3", "hybrid")
        for recipe in ("delayed", "tensorwise", "blockwise", "mxfp8")
    ],
    pytest.param({"fp4": "e2m1", "fp4_recipe": "nvfp4"}, id="nvfp4"),
]

FP32_MOE_MODES = [
    pytest.param({"moe_activation_in_fp32": True}, id="activation"),
    pytest.param({"moe_combine_in_fp32": True}, id="combine"),
    pytest.param(
        {"moe_activation_in_fp32": True, "moe_combine_in_fp32": True}, id="activation-and-combine"
    ),
]


@pytest.fixture(autouse=True)
def assume_supported_transformer_engine(monkeypatch):
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)


def make_fp32_moe_config(**overrides):
    kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_moe_experts": 2,
        "moe_ffn_hidden_size": 64,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "transformer_impl": "transformer_engine",
        "moe_grouped_gemm": True,
        "moe_use_legacy_grouped_gemm": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_dtype": "fp32",
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "add_bias_linear": True,
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


@pytest.mark.parametrize("precision_config", STANDARD_PRECISION_CONFIGS)
@pytest.mark.parametrize("mode_config", FP32_MOE_MODES)
def test_fp32_moe_modes_accept_standard_precision_configs(precision_config, mode_config):
    config = make_fp32_moe_config(**precision_config, **mode_config)

    assert config.moe_activation_in_fp32 == mode_config.get("moe_activation_in_fp32", False)
    assert config.moe_combine_in_fp32 == mode_config.get("moe_combine_in_fp32", False)


@pytest.mark.parametrize(
    "precision_config, error",
    [
        pytest.param(
            {
                "fp8": "e4m3",
                "fp8_recipe": "custom",
                "fp8_quantizer_factory": "package.module.factory",
            },
            "standard FP8 recipes only",
            id="custom-fp8",
        ),
        pytest.param(
            {
                "fp4": "e2m1",
                "fp4_recipe": "custom",
                "fp4_quantizer_factory": "package.module.factory",
            },
            "NVFP4 only",
            id="custom-fp4",
        ),
    ],
)
def test_fp32_moe_modes_reject_custom_precision_recipes(precision_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(moe_activation_in_fp32=True, **precision_config)


@pytest.mark.parametrize(
    "invalid_config, error",
    [
        pytest.param(
            {"transformer_impl": "local"},
            "requires --transformer-impl transformer_engine",
            id="local-transformer",
        ),
        pytest.param({"moe_grouped_gemm": False}, "requires TEGroupedMLP", id="sequential-mlp"),
        pytest.param(
            {"moe_use_legacy_grouped_gemm": True}, "requires TEGroupedMLP", id="legacy-grouped-mlp"
        ),
        pytest.param(
            {"moe_permute_fusion": True},
            "does not support --moe-permute-fusion",
            id="fused-permute",
        ),
        pytest.param(
            {"activation_func_clamp_value": 7.0},
            "drop --activation-func-clamp-value",
            id="clamped-activation",
        ),
    ],
)
@pytest.mark.parametrize("mode_config", FP32_MOE_MODES[:2])
def test_fp32_moe_modes_reject_unsupported_expert_paths(mode_config, invalid_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(**mode_config, **invalid_config)


@pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall", "flex"])
def test_fp32_moe_activation_accepts_dispatchers_without_fp32_router(moe_token_dispatcher_type):
    config = make_fp32_moe_config(
        moe_activation_in_fp32=True,
        moe_router_dtype=None,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
    )

    assert config.moe_activation_in_fp32


def test_fp32_moe_combine_accepts_non_swiglu():
    config = make_fp32_moe_config(
        moe_combine_in_fp32=True, gated_linear_unit=False, activation_func=F.gelu
    )

    assert config.moe_combine_in_fp32


@pytest.mark.parametrize(
    "invalid_config, error",
    [
        pytest.param(
            {"gated_linear_unit": False}, "supports SwiGLU only", id="non-gated-activation"
        ),
        pytest.param({"activation_func": F.gelu}, "supports SwiGLU only", id="non-silu-activation"),
    ],
)
def test_fp32_moe_activation_rejects_non_swiglu(invalid_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(moe_activation_in_fp32=True, **invalid_config)


@pytest.mark.parametrize(
    "invalid_config, error",
    [
        pytest.param(
            {"moe_token_dispatcher_type": "allgather"},
            "requires --moe-token-dispatcher-type alltoall",
            id="allgather-dispatcher",
        ),
        pytest.param(
            {"moe_token_dispatcher_type": "flex"},
            "requires --moe-token-dispatcher-type alltoall",
            id="flex-dispatcher",
        ),
        pytest.param(
            {"moe_apply_probs_on_input": True},
            "incompatible with --moe-apply-probs-on-input",
            id="probs-on-input",
        ),
        pytest.param(
            {"moe_router_dtype": None}, "needs --moe-router-dtype fp32", id="router-dtype"
        ),
    ],
)
def test_fp32_moe_combine_rejects_unsupported_paths(invalid_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(moe_combine_in_fp32=True, **invalid_config)
