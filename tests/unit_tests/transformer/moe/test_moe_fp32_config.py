# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.moe.fp32_activation as fp32_activation_module
import megatron.core.transformer.moe.moe_utils as moe_utils_module
import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.activations import squared_relu
from megatron.core.transformer.moe.experts import _MoEActivationInFP32
from megatron.core.transformer.moe.fp32_activation import (
    MoEActivationInFP32Spec,
    get_moe_activation_in_fp32_spec,
    register_moe_activation_in_fp32,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

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
    for name in (
        "fused_permute",
        "fused_permute_with_probs",
        "fused_sort_chunks_by_index",
        "fused_sort_chunks_by_index_with_probs",
        "fused_unpermute",
    ):
        monkeypatch.setattr(moe_utils_module, name, object(), raising=False)


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
    ],
)
@pytest.mark.parametrize("mode_config", FP32_MOE_MODES[:2])
def test_fp32_moe_modes_reject_unsupported_expert_paths(mode_config, invalid_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(**mode_config, **invalid_config)


def test_fp32_moe_activation_rejects_clamp():
    with pytest.raises(AssertionError, match="drop --activation-func-clamp-value"):
        make_fp32_moe_config(moe_activation_in_fp32=True, activation_func_clamp_value=7.0)


@pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall", "flex"])
def test_fp32_moe_activation_accepts_dispatchers_without_fp32_router(moe_token_dispatcher_type):
    config = make_fp32_moe_config(
        moe_activation_in_fp32=True,
        moe_router_dtype=None,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
    )

    assert config.moe_activation_in_fp32


@pytest.mark.parametrize("precision_config", STANDARD_PRECISION_CONFIGS)
def test_fp32_moe_activation_accepts_squared_relu_with_fused_permute(precision_config):
    config = make_fp32_moe_config(
        moe_activation_in_fp32=True,
        gated_linear_unit=False,
        activation_func=squared_relu,
        moe_token_dispatcher_type="allgather",
        moe_permute_fusion=True,
        use_fused_weighted_squared_relu=True,
        **precision_config,
    )

    assert config.moe_activation_in_fp32
    assert config.activation_func is squared_relu
    assert not config.gated_linear_unit
    assert config.moe_permute_fusion


@pytest.mark.parametrize(
    "activation_config",
    [
        pytest.param({"gated_linear_unit": False, "activation_func": F.gelu}, id="gelu"),
        pytest.param(
            {
                "gated_linear_unit": False,
                "activation_func": squared_relu,
                "use_fused_weighted_squared_relu": True,
            },
            id="squared-relu",
        ),
    ],
)
def test_fp32_moe_combine_accepts_non_swiglu(activation_config):
    config = make_fp32_moe_config(moe_combine_in_fp32=True, **activation_config)

    assert config.moe_combine_in_fp32
    assert not config.moe_activation_in_fp32


def test_fp32_moe_activation_accepts_registered_activation(monkeypatch):
    monkeypatch.setattr(
        fp32_activation_module,
        "_MOE_ACTIVATIONS_IN_FP32",
        fp32_activation_module._MOE_ACTIVATIONS_IN_FP32.copy(),
    )

    def cubic_relu(x):
        return F.relu(x).pow(3)

    def cubic_relu_forward(fc1_out, _glu_offset):
        return F.relu(fc1_out).pow(3)

    def cubic_relu_backward(grad_output, fc1_out, _glu_offset):
        return grad_output * 3 * F.relu(fc1_out).square()

    register_moe_activation_in_fp32(
        cubic_relu,
        False,
        MoEActivationInFP32Spec(
            name="cubic_relu",
            context_factory=lambda _config: None,
            forward=cubic_relu_forward,
            backward=cubic_relu_backward,
        ),
    )

    config = make_fp32_moe_config(
        moe_activation_in_fp32=True, gated_linear_unit=False, activation_func=cubic_relu
    )

    assert config.activation_func is cubic_relu


def test_yaml_squared_relu_uses_registered_activation():
    config = make_fp32_moe_config(
        moe_activation_in_fp32=True, gated_linear_unit=False, activation_func=squared_relu
    )
    yaml_fields = vars(config).copy()
    yaml_fields["activation_func"] = "squaredrelu"
    yaml_args = SimpleNamespace(
        language_model=SimpleNamespace(**yaml_fields), model_parallel=SimpleNamespace()
    )

    parsed_config = core_transformer_config_from_yaml(yaml_args)

    assert parsed_config.activation_func is squared_relu
    assert parsed_config.moe_activation_in_fp32


@pytest.mark.parametrize(
    "invalid_config, error",
    [
        pytest.param(
            {"gated_linear_unit": False},
            "requires a registered activation/gating pair",
            id="non-gated-silu",
        ),
        pytest.param(
            {"gated_linear_unit": False, "activation_func": F.gelu},
            "requires a registered activation/gating pair",
            id="gelu",
        ),
        pytest.param(
            {"activation_func": squared_relu},
            "requires a registered activation/gating pair",
            id="gated-squared-relu",
        ),
    ],
)
def test_fp32_moe_activation_rejects_unsupported_activations(invalid_config, error):
    with pytest.raises(AssertionError, match=error):
        make_fp32_moe_config(moe_activation_in_fp32=True, **invalid_config)


@pytest.mark.parametrize(
    "invalid_config, error",
    [
        pytest.param(
            {"moe_permute_fusion": True},
            "does not support --moe-permute-fusion",
            id="fused-permute",
        ),
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_swiglu", [True, False], ids=["swiglu", "squared-relu"])
def test_activation_in_fp32_matches_reference(dtype, is_swiglu):
    actual_input = torch.randn(11, 16, dtype=dtype, requires_grad=True)
    actual_probs = torch.rand(11, 1, dtype=torch.float32, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_()
    expected_probs = actual_probs.detach().clone().requires_grad_()
    glu_offset = 0.25

    activation_func = F.silu if is_swiglu else squared_relu
    activation_spec = get_moe_activation_in_fp32_spec(activation_func, is_swiglu)
    assert activation_spec is not None
    config = make_fp32_moe_config(glu_linear_offset=glu_offset)
    activation_context = activation_spec.context_factory(config)
    actual = _MoEActivationInFP32.apply(
        actual_input, actual_probs, activation_spec, activation_context
    )
    if is_swiglu:
        gate, linear = torch.chunk(expected_input.float(), 2, dim=-1)
        expected = (F.silu(gate) * (linear + glu_offset) * expected_probs).to(dtype)
    else:
        expected = (squared_relu(expected_input.float()) * expected_probs).to(dtype)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    torch.testing.assert_close(actual_probs.grad, expected_probs.grad)
