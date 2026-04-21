# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.sglang import (
    SGLangColumnParallelLinear,
    SGLangNorm,
    SGLangRowParallelLinear,
    SGLangSpecProvider,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.matmul_tp_inv import sglang_reference_matmul
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_spec_for_backend
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from tests.unit_tests.test_utilities import Utils


def _make_config(**overrides) -> TransformerConfig:
    config_kwargs = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "ffn_hidden_size": 32,
        "normalization": "RMSNorm",
        "use_cpu_initialization": True,
        "perform_initialization": False,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "transformer_impl": "local",
    }
    config_kwargs.update(overrides)
    return TransformerConfig(**config_kwargs)


@contextmanager
def _fake_tp_init():
    parallel_state.destroy_model_parallel()
    Utils.fake_initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()


def _parse_training_args(monkeypatch, *argv):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr(sys, "argv", ["test_sglang_extension.py", *argv])
    return parse_args()


def test_sglang_extension_imports():
    assert SGLangColumnParallelLinear.backend_name == "sglang"
    assert SGLangRowParallelLinear.backend_name == "sglang"
    assert SGLangNorm.backend_name == "sglang"
    assert callable(sglang_reference_matmul)


def test_use_sglang_arg_parsing(monkeypatch):
    field_names = {field.name for field in dataclasses.fields(TransformerConfig)}
    assert "use_sglang" in field_names

    args = _parse_training_args(monkeypatch, "--use-sglang")

    assert args.use_sglang is True


def test_validate_args_rejects_incompatible_sglang_backend(monkeypatch):
    from megatron.training.arguments import validate_args

    args = _parse_training_args(monkeypatch)
    args.use_sglang = True
    args.transformer_impl = "transformer_engine"

    with pytest.raises(
        AssertionError, match="--use-sglang currently requires --transformer-impl local"
    ):
        validate_args(args)


def test_default_backend_selection_is_unchanged():
    config = _make_config(use_sglang=False)

    layer_spec = get_gpt_decoder_layer_specs(
        config,
        use_transformer_engine=False,
        normalization=config.normalization,
    )[0]

    assert layer_spec.submodules.input_layernorm is WrappedTorchNorm
    assert layer_spec.submodules.self_attention.submodules.linear_qkv is ColumnParallelLinear
    assert layer_spec.submodules.self_attention.submodules.linear_proj is RowParallelLinear


def test_use_sglang_selects_sglang_backend():
    config = _make_config(use_sglang=True)

    layer_spec = get_gpt_decoder_layer_specs(
        config,
        use_transformer_engine=False,
        normalization=config.normalization,
    )[0]

    assert layer_spec.submodules.input_layernorm is SGLangNorm
    assert layer_spec.submodules.self_attention.submodules.linear_qkv is SGLangColumnParallelLinear
    assert layer_spec.submodules.self_attention.submodules.linear_proj is SGLangRowParallelLinear


def test_transformer_config_rejects_incompatible_sglang_backend():
    with pytest.raises(AssertionError, match="use_sglang currently requires transformer_impl='local'"):
        _make_config(use_sglang=True, transformer_impl="transformer_engine")


def test_transformer_config_rejects_sglang_with_kitchen():
    with pytest.raises(AssertionError, match="use_sglang is not compatible with use_kitchen"):
        _make_config(use_sglang=True, use_kitchen=True)


def test_sglang_reference_matmul_matches_torch_linear():
    input_ = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    bias = torch.randn(5)

    actual = sglang_reference_matmul(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
    )
    expected = torch.nn.functional.linear(input_, weight, bias)

    torch.testing.assert_close(actual, expected)


def test_sglang_norm_layernorm_matches_torch():
    config = _make_config(normalization="LayerNorm")
    norm = SGLangNorm(config=config, hidden_size=4, eps=1e-5)
    x = torch.randn(2, 3, 4)

    actual = norm(x)
    expected = torch.nn.functional.layer_norm(x, (4,), norm.weight, norm.bias, norm.eps)

    torch.testing.assert_close(actual, expected)


def test_sglang_norm_rmsnorm_matches_reference():
    config = _make_config(normalization="RMSNorm")
    norm = SGLangNorm(config=config, hidden_size=4, eps=1e-5)
    x = torch.randn(2, 3, 4)

    actual = norm(x)
    x_float = x.float()
    expected = (
        x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    ).type_as(x) * norm.weight

    torch.testing.assert_close(actual, expected)


def test_sglang_rmsnorm_rejects_zero_centered_gamma():
    config = _make_config(normalization="RMSNorm")

    with pytest.raises(AssertionError, match="zero_centered_gamma is not supported"):
        SGLangNorm(config=config, hidden_size=4, zero_centered_gamma=True)


def test_sglang_spec_provider_grouped_mlp_fallback():
    provider = SGLangSpecProvider()

    module, submodules = provider.grouped_mlp_modules(
        moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=False
    )
    assert module.__name__ == "SequentialMLP"
    assert submodules.linear_fc1 is SGLangColumnParallelLinear
    assert submodules.linear_fc2 is SGLangRowParallelLinear

    grouped_module, grouped_submodules = provider.grouped_mlp_modules(
        moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=False
    )
    assert grouped_module.__name__ == "GroupedMLP"
    assert grouped_submodules is None


def test_sglang_mtp_spec_uses_sglang_backend():
    config = _make_config(use_sglang=True, mtp_num_layers=1)
    transformer_layer_spec = get_gpt_decoder_layer_specs(
        config,
        use_transformer_engine=False,
        normalization=config.normalization,
    )[0]
    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec,
        backend=SGLangSpecProvider(),
    )

    assert mtp_layer_spec.submodules.enorm is SGLangNorm
    assert mtp_layer_spec.submodules.hnorm is SGLangNorm
    assert mtp_layer_spec.submodules.eh_proj is SGLangColumnParallelLinear
    assert mtp_layer_spec.submodules.layer_norm is SGLangNorm


def test_sglang_column_parallel_linear_wrapper_forward_matches_reference():
    with _fake_tp_init():
        config = _make_config(use_cpu_initialization=True)
        layer = SGLangColumnParallelLinear(
            input_size=4,
            output_size=5,
            init_method=config.init_method,
            bias=True,
            config=config,
            gather_output=False,
            skip_bias_add=False,
        )
        with torch.no_grad():
            layer.weight.copy_(torch.arange(20, dtype=torch.float32).view(5, 4))
            layer.bias.copy_(torch.arange(5, dtype=torch.float32))

        x = torch.randn(2, 3, 4)
        actual, output_bias = layer(x)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

        torch.testing.assert_close(actual, expected)
        assert output_bias is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sglang_row_parallel_linear_wrapper_forward_matches_reference():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = _make_config(use_cpu_initialization=False)
        layer = SGLangRowParallelLinear(
            input_size=4,
            output_size=5,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=True,
            config=config,
            skip_bias_add=False,
        )
        with torch.no_grad():
            layer.weight.copy_(
                torch.arange(20, dtype=torch.float32, device=layer.weight.device).view(5, 4)
            )
            layer.bias.copy_(torch.arange(5, dtype=torch.float32, device=layer.bias.device))

        x = torch.randn(2, 3, 4, device=layer.weight.device)
        actual, output_bias = layer(x)
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

        torch.testing.assert_close(actual, expected)
        assert output_bias is None
    finally:
        Utils.destroy_model_parallel()
