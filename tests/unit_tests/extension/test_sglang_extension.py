# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import dataclasses
import os
import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from miles_megatron_plugins.true_on_policy.sglang_backend import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    MegatronTrueOnPolicyRuntimePolicy,
    SGLangColumnParallelLinear,
    SGLangCoreAttention,
    SGLangFinalRMSNorm,
    SGLangGroupedMLP,
    SGLangNorm,
    SGLangQKRMSNorm,
    SGLangRowParallelLinear,
    SGLangSpecProvider,
    _ensure_batch_invariant_mode_from_config,
    disable_sglang_rope,
    get_sglang_bias_dropout_add,
    is_sglang_rope_enabled,
    resolve_true_on_policy_runtime_policy,
)
from miles_megatron_plugins.true_on_policy.contracts import get_true_on_policy_contract
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layer_specs,
    get_gpt_layer_local_spec,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
from miles_megatron_plugins.true_on_policy.matmul import _sglang_row_parallel_matmul, sglang_reference_matmul
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    matmul_persistent,
    set_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.attention import _sglang_cast_dense_tensor_math_input
from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_spec_for_backend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import _get_block_submodules
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


class _FakeCPGroup:
    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


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
    assert MegatronTrueOnPolicyRuntimePolicy.__name__ == "MegatronTrueOnPolicyRuntimePolicy"


def test_legacy_sglang_backend_imports_match_true_on_policy_namespace():
    from megatron.core.extensions import sglang as legacy_backend
    from megatron.core.tensor_parallel import matmul_tp_inv as legacy_matmul
    from miles_megatron_plugins.true_on_policy import (
        attention_fa3,
        bias_dropout,
        cp_layout,
        linear,
        norm,
        provider,
        rope,
        runtime,
    )
    from miles_megatron_plugins.true_on_policy import matmul, sglang_backend

    assert legacy_backend.SGLangNorm is sglang_backend.SGLangNorm
    assert legacy_backend.SGLangRowParallelLinear is sglang_backend.SGLangRowParallelLinear
    assert sglang_backend.SGLangColumnParallelLinear is linear.SGLangColumnParallelLinear
    assert sglang_backend.SGLangCoreAttention is attention_fa3.SGLangCoreAttention
    assert sglang_backend.SGLangUlyssesCPLayout is cp_layout.SGLangUlyssesCPLayout
    assert sglang_backend.SGLangNorm is norm.SGLangNorm
    assert sglang_backend.SGLangSpecProvider is provider.SGLangSpecProvider
    assert sglang_backend.get_sglang_bias_dropout_add is bias_dropout.get_sglang_bias_dropout_add
    assert (
        sglang_backend.enable_sglang_batch_invariant_mode
        is runtime.enable_sglang_batch_invariant_mode
    )
    assert sglang_backend.sglang_apply_rotary_pos_emb is rope.sglang_apply_rotary_pos_emb
    assert (
        sglang_backend.resolve_true_on_policy_runtime_policy
        is resolve_true_on_policy_runtime_policy
    )
    assert legacy_matmul.sglang_reference_matmul is matmul.sglang_reference_matmul


def test_true_on_policy_contract_arg_parsing(monkeypatch):
    field_names = {field.name for field in dataclasses.fields(TransformerConfig)}
    assert "use_sglang" not in field_names
    assert "true_on_policy_contract" in field_names

    args = _parse_training_args(
        monkeypatch, "--true-on-policy-contract", QWEN3_DENSE_TRUE_ON_POLICY_V1
    )

    assert args.true_on_policy_contract == QWEN3_DENSE_TRUE_ON_POLICY_V1


def test_true_on_policy_contract_resolves_megatron_runtime_policy():
    config = _make_config(
        batch_invariant_mode=True,
        context_parallel_size=4,
        cp_comm_type="a2a",
        attention_backend=AttnBackend.flash,
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
    )

    policy = resolve_true_on_policy_runtime_policy(config)

    assert policy.contract_name == QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert policy.use_sglang_backend
    assert policy.batch_invariant_mode
    assert policy.attention_backend == "fa3_varlen"
    assert policy.cp_layout == "ulysses_a2a"
    assert policy.use_ulysses_cp_recompute_fallback


def test_contract_object_owns_megatron_runtime_policy_values():
    contract = get_true_on_policy_contract(QWEN3_DENSE_TRUE_ON_POLICY_V1)
    config = _make_config(
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
    )

    policy = contract.policy_for(config)

    assert contract.schema.name == QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert contract.schema.model_family == "qwen3_dense"
    assert policy.contract_name == QWEN3_DENSE_TRUE_ON_POLICY_V1
    assert policy.enabled
    assert policy.use_sglang_backend
    assert policy.batch_invariant_mode
    assert policy.disable_rope_fusion
    assert policy.disable_bias_swiglu_fusion
    assert policy.attention_backend == "fa3_varlen"
    assert policy.cast_attention_input_to_dense_math_dtype
    assert policy.cast_lm_head_input_to_weight_dtype
    assert policy.cast_qk_after_rope_to_dense_math_dtype
    assert policy.deterministic_row_parallel_reduce
    assert policy.defer_ulysses_cp_loss_scaling_to_grad_sum
    assert policy.apply_logits_contract
    assert policy.use_sglang_final_norm
    assert policy.use_sglang_residual_pair
    assert not policy.use_ulysses_cp_recompute_fallback


def test_qwen3_moe_contract_splits_tp_cp_and_ep_policy_values():
    contract = get_true_on_policy_contract(QWEN3_MOE_TRUE_ON_POLICY_V1)
    config = _make_config(
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        context_parallel_size=2,
        cp_comm_type="a2a",
        expert_model_parallel_size=4,
        num_moe_experts=128,
        true_on_policy_contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
    )

    policy = contract.policy_for(config)

    assert contract.schema.model_family == "qwen3_moe"
    assert policy.contract_name == QWEN3_MOE_TRUE_ON_POLICY_V1
    assert policy.use_sglang_backend
    assert policy.cp_layout == "ulysses_a2a"
    assert policy.deterministic_moe_routing
    assert policy.moe_topk_tiebreak == "stable_sort"
    assert policy.ep_invariant_moe
    assert policy.deterministic_moe_dispatch


def test_true_on_policy_moe_local_spec_keeps_pre_mlp_layernorm_checkpoint_key():
    dense_spec = get_gpt_layer_local_spec(
        normalization="RMSNorm",
        use_true_on_policy_backend=True,
    )
    moe_spec = get_gpt_layer_local_spec(
        num_experts=128,
        moe_grouped_gemm=True,
        normalization="RMSNorm",
        use_true_on_policy_backend=True,
    )

    dense_key_map = dense_spec.submodules.sharded_state_dict_keys_map
    moe_key_map = moe_spec.submodules.sharded_state_dict_keys_map

    assert dense_key_map["pre_mlp_layernorm."] == "mlp.linear_fc1.layer_norm_"
    assert "pre_mlp_layernorm." not in moe_key_map
    assert "mlp.linear_fc1.layer_norm_" not in moe_key_map.values()


def test_sglang_moe_ep_tree_all_reduce_uses_sglang_fixed_tree(monkeypatch):
    from miles_megatron_plugins.true_on_policy.moe import sglang_moe_ep_tree_all_reduce

    calls = []
    tp_invariant_ops = types.ModuleType("sglang.srt.tp_invariant_ops")

    def fake_tree_all_reduce_sum(input_, device_group=None):
        calls.append((input_, device_group))
        return input_ + 1

    tp_invariant_ops.tree_all_reduce_sum = fake_tree_all_reduce_sum
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", types.ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.tp_invariant_ops", tp_invariant_ops
    )

    ep_group = object()
    input_ = torch.zeros(2, 3)
    output = sglang_moe_ep_tree_all_reduce(input_, ep_group)

    torch.testing.assert_close(output, torch.ones_like(input_))
    assert calls == [(input_, ep_group)]


def test_qwen3_dense_contract_only_marks_ulysses_a2a_as_cp_layout():
    contract = get_true_on_policy_contract(QWEN3_DENSE_TRUE_ON_POLICY_V1)

    policy = contract.policy_for(
        _make_config(
            context_parallel_size=4,
            cp_comm_type="all_gather",
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        )
    )

    assert policy.cp_layout is None
    assert not policy.use_ulysses_cp_recompute_fallback


def test_missing_true_on_policy_contract_uses_default_policy():
    config = _make_config()

    policy = resolve_true_on_policy_runtime_policy(config)

    assert policy.contract_name is None
    assert not policy.enabled


def test_invalid_true_on_policy_contract_is_rejected_by_config():
    with pytest.raises(ValueError, match="Unsupported Megatron true-on-policy contract"):
        _make_config(true_on_policy_contract="unknown_contract")


def test_default_backend_selection_is_unchanged():
    config = _make_config()

    layer_spec = get_gpt_decoder_layer_specs(
        config, use_transformer_engine=False, normalization=config.normalization
    )[0]

    assert layer_spec.submodules.input_layernorm is WrappedTorchNorm
    assert layer_spec.submodules.self_attention.submodules.linear_qkv is ColumnParallelLinear
    assert layer_spec.submodules.self_attention.submodules.linear_proj is RowParallelLinear


def test_true_on_policy_contract_selects_sglang_backend():
    disable_sglang_rope()
    config = _make_config(true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, qk_layernorm=True)

    layer_spec = get_gpt_decoder_layer_specs(
        config, use_transformer_engine=False, normalization=config.normalization
    )[0]

    assert is_sglang_rope_enabled()
    assert isinstance(layer_spec.submodules.input_layernorm, ModuleSpec)
    assert layer_spec.submodules.input_layernorm.module is SGLangNorm
    assert layer_spec.submodules.input_layernorm.params["override_orig_dtype"] is torch.float32
    assert layer_spec.submodules.self_attn_bda is get_sglang_bias_dropout_add
    assert layer_spec.submodules.self_attention.submodules.linear_qkv is SGLangColumnParallelLinear
    assert layer_spec.submodules.self_attention.submodules.core_attention is SGLangCoreAttention
    assert layer_spec.submodules.self_attention.submodules.q_layernorm is SGLangQKRMSNorm
    assert layer_spec.submodules.self_attention.submodules.linear_proj is SGLangRowParallelLinear
    assert layer_spec.submodules.mlp_bda is get_sglang_bias_dropout_add


def test_true_on_policy_contract_layer_spec_selects_sglang_final_norm():
    config = _make_config(true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1)
    layer_spec = get_gpt_decoder_layer_specs(
        config, use_transformer_engine=False, normalization=config.normalization
    )[0]

    block_submodules = _get_block_submodules(config, layer_spec, pp_rank=0)

    assert block_submodules.layer_norm is SGLangFinalRMSNorm


def test_transformer_config_rejects_incompatible_true_on_policy_backend():
    with pytest.raises(
        AssertionError, match="true_on_policy_contract currently requires transformer_impl='local'"
    ):
        _make_config(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            transformer_impl="transformer_engine",
        )


def test_transformer_config_rejects_true_on_policy_with_kitchen():
    with pytest.raises(
        AssertionError, match="true_on_policy_contract is not compatible with use_kitchen"
    ):
        _make_config(true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, use_kitchen=True)


def test_sglang_config_allows_ulysses_context_parallel():
    config = _make_config(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        cp_comm_type="a2a",
    )

    assert config.cp_comm_type == "a2a"


def test_ulysses_rope_uses_cp_positions_for_local_sequence_shards():
    config = _make_config(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        context_parallel_size=2,
        cp_comm_type="a2a",
    )
    local_t = torch.randn(4, 2, 4)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    freqs = torch.randn(8, 1, 1, 4)
    cp_group = _FakeCPGroup(size=2, rank=1)

    actual = apply_rotary_pos_emb(
        local_t, freqs, config=config, cu_seqlens=cu_seqlens, cp_group=cp_group, ulysses_cp=True
    )
    expected = apply_rotary_pos_emb(
        local_t, freqs, config=config, cu_seqlens=cu_seqlens, cp_group=cp_group, ulysses_cp=False
    )

    torch.testing.assert_close(actual, expected)


def test_ulysses_rope_keeps_unsplit_positions_for_full_sequence_layout():
    config = _make_config(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        context_parallel_size=2,
        cp_comm_type="a2a",
    )
    full_t = torch.randn(8, 2, 4)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    freqs = torch.randn(8, 1, 1, 4)

    actual = apply_rotary_pos_emb(
        full_t,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(size=2, rank=1),
        ulysses_cp=True,
    )
    expected = apply_rotary_pos_emb(
        full_t,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(size=1, rank=0),
        ulysses_cp=False,
    )

    torch.testing.assert_close(actual, expected)


def test_local_attention_still_rejects_ulysses_context_parallel_without_true_on_policy():
    with pytest.raises(ValueError, match="only supports all_gather"):
        _make_config(tensor_model_parallel_size=1, context_parallel_size=2, cp_comm_type="a2a")


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


def test_sglang_reference_matmul_matches_sglang_mixed_dtype_contract():
    input_ = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4).to(torch.bfloat16)
    bias = torch.randn(5).to(torch.float32)

    actual = sglang_reference_matmul(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
    )
    expected = torch.matmul(input_.to(weight.dtype), weight.t()) + bias.to(weight.dtype)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_sglang_row_parallel_matmul_uses_fixed_k_block_order():
    input_ = torch.randn(2, 3, 256)
    weight = torch.randn(5, 256)
    bias = torch.randn(5)

    actual = sglang_reference_matmul(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
        row_parallel=True,
    )
    expected = _sglang_row_parallel_matmul(input_, weight, bias)

    assert torch.equal(actual, expected)


def test_sglang_column_matmul_keeps_default_linear_path_for_same_k_size():
    input_ = torch.randn(2, 3, 256)
    weight = torch.randn(5, 256)
    bias = torch.randn(5)

    actual = sglang_reference_matmul(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion=False,
        allreduce_dgrad=False,
        sequence_parallel=False,
        row_parallel=False,
    )
    expected = torch.nn.functional.linear(input_, weight, bias)

    torch.testing.assert_close(actual, expected)


def test_sglang_backend_enables_batch_invariant_mode_from_config(monkeypatch):
    from megatron.core.transformer.custom_layers import batch_invariant_kernels

    calls = []
    config = _make_config(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        batch_invariant_mode=True,
        attention_backend=AttnBackend.flash,
    )

    monkeypatch.setattr(batch_invariant_kernels, "is_batch_invariant_mode_enabled", lambda: False)
    monkeypatch.setattr(
        batch_invariant_kernels, "enable_batch_invariant_mode", lambda: calls.append(None)
    )

    _ensure_batch_invariant_mode_from_config(config)

    assert calls == [None]


def test_sglang_backend_leaves_batch_invariant_mode_disabled(monkeypatch):
    from megatron.core.transformer.custom_layers import batch_invariant_kernels

    calls = []
    config = _make_config(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, batch_invariant_mode=False
    )

    monkeypatch.setattr(batch_invariant_kernels, "is_batch_invariant_mode_enabled", lambda: False)
    monkeypatch.setattr(
        batch_invariant_kernels, "enable_batch_invariant_mode", lambda: calls.append(None)
    )

    _ensure_batch_invariant_mode_from_config(config)

    assert calls == []


def test_sglang_dense_tensor_math_cast_matches_qwen3_contract():
    input_ = torch.randn(2, 3, 4, dtype=torch.float32)

    actual = _sglang_cast_dense_tensor_math_input(input_)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, input_.to(torch.bfloat16))


def test_sglang_dense_tensor_math_cast_preserves_bfloat16_tensor():
    input_ = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual = _sglang_cast_dense_tensor_math_input(input_)

    assert actual is input_


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batch_invariant_linear_flattens_sequence_batch_for_gemm():
    input_ = torch.randn(4, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with set_batch_invariant_mode(True):
        actual = linear_with_grad_accumulation_and_async_allreduce(
            input_,
            weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
        )
        expected = matmul_persistent(input_.detach().reshape(-1, 128), weight.detach().t())
        expected = expected.reshape(4, 1, 96)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)

    actual.float().sum().backward()
    assert input_.grad is not None
    assert weight.grad is not None


def test_sglang_output_layer_casts_input_to_weight_dtype():
    with _fake_tp_init():
        config = _make_config(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, use_cpu_initialization=True
        )
        layer = LinearCrossEntropyModule(
            input_size=4,
            output_size=5,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
        )
        layer.weight.data = torch.randn_like(layer.weight.data).to(torch.bfloat16)
        x = torch.randn(2, 3, 4, dtype=torch.float32)

        actual, _ = layer(x)
        expected = torch.matmul(x.to(torch.bfloat16), layer.weight.t())

        assert actual.dtype == torch.bfloat16
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
    expected = (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)).type_as(
        x
    ) * norm.weight

    torch.testing.assert_close(actual, expected)


def test_sglang_norm_rmsnorm_keeps_affine_in_float():
    config = _make_config(normalization="RMSNorm")
    norm = SGLangNorm(config=config, hidden_size=4, eps=1e-5).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual = norm(x)
    x_float = x.float()
    expected = (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)).type_as(
        x
    ) * norm.weight.float()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_sglang_norm_rmsnorm_can_override_original_dtype():
    config = _make_config(normalization="RMSNorm")
    norm = SGLangNorm(config=config, hidden_size=4, eps=1e-5, override_orig_dtype=torch.float32).to(
        dtype=torch.bfloat16
    )
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual = norm(x)
    x_float = x.float()
    expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = norm.weight.float() * expected

    assert actual.dtype == torch.float32
    assert norm.weight.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_sglang_norm_rmsnorm_accepts_residual_pair():
    config = _make_config(normalization="RMSNorm")
    norm = SGLangNorm(config=config, hidden_size=4, eps=1e-5, override_orig_dtype=torch.float32).to(
        dtype=torch.bfloat16
    )
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual, actual_residual = norm(x, residual)
    x_float = x.float() + residual.float()
    expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = norm.weight.float() * expected

    assert actual.dtype == torch.float32
    assert actual_residual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, x_float)


def test_sglang_qk_rmsnorm_matches_source_truth_dtype_boundary():
    config = _make_config(normalization="RMSNorm")
    norm = SGLangQKRMSNorm(config=config, hidden_size=4, eps=1e-6)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    actual = norm(x)
    x_float = x.float()
    expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = norm.weight.float() * expected.to(torch.bfloat16)

    assert norm.weight.dtype == torch.float32
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_sglang_final_rmsnorm_matches_source_truth_dtype_boundary():
    config = _make_config(normalization="RMSNorm", params_dtype=torch.bfloat16)
    norm = SGLangFinalRMSNorm(config=config, hidden_size=4, eps=1e-6)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16).float()
    residual = torch.randn(2, 3, 4, dtype=torch.float32)

    actual, actual_residual = norm(x, residual)
    x_with_residual = x + residual
    x_float = x_with_residual.float()
    expected = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = norm.weight.to(torch.bfloat16) * expected.to(torch.bfloat16)

    assert norm.weight.dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_residual, x_with_residual)


def test_sglang_exact_forward_native_grad_wrapper_preserves_exact_value():
    from miles_megatron_plugins.true_on_policy.norm import _with_native_grad

    exact = torch.tensor([1.0], dtype=torch.float32)
    native = torch.tensor([1.0e8], dtype=torch.float32, requires_grad=True)

    actual = _with_native_grad(exact, native)

    torch.testing.assert_close(actual.detach(), exact, atol=0.0, rtol=0.0)
    actual.sum().backward()
    torch.testing.assert_close(native.grad, torch.ones_like(native))


def test_sglang_bias_dropout_add_keeps_residual_add_in_float():
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 3, 4, dtype=torch.float32)

    actual = get_sglang_bias_dropout_add(training=False, fused=True)((x, None), residual, 0.0)
    expected = residual + x.float()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_sglang_rmsnorm_rejects_zero_centered_gamma():
    config = _make_config(normalization="RMSNorm")

    with pytest.raises(AssertionError, match="zero_centered_gamma is not supported"):
        SGLangNorm(config=config, hidden_size=4, zero_centered_gamma=True)


def test_sglang_spec_provider_grouped_mlp_uses_sglang_no_grad_surface():
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
    assert grouped_module is SGLangGroupedMLP
    assert grouped_submodules is None


def test_sglang_grouped_mlp_weight_views_match_megatron_grouped_layout():
    config = _make_config(hidden_size=2, moe_ffn_hidden_size=3, num_moe_experts=2)
    layer = object.__new__(SGLangGroupedMLP)
    layer.config = config
    layer.num_local_experts = 2
    layer.weight1 = torch.arange(2 * 2 * 2 * 3, dtype=torch.bfloat16).reshape(2, 12)
    layer.weight2 = torch.arange(2 * 3 * 2, dtype=torch.bfloat16).reshape(6, 2)

    expected_w13 = layer.weight1.view(2, 2, 6).permute(0, 2, 1).contiguous()
    expected_w2 = layer.weight2.view(2, 3, 2).permute(0, 2, 1).contiguous()

    torch.testing.assert_close(layer._sglang_w13_weight(), expected_w13)
    torch.testing.assert_close(layer._sglang_w2_weight(), expected_w2)


def test_sglang_grouped_mlp_weight_cache_reuses_and_invalidates_on_update():
    config = _make_config(hidden_size=2, moe_ffn_hidden_size=3, num_moe_experts=2)
    layer = object.__new__(SGLangGroupedMLP)
    layer.config = config
    layer.num_local_experts = 2
    layer.weight1 = torch.arange(2 * 2 * 2 * 3, dtype=torch.bfloat16).reshape(2, 12)
    layer.weight2 = torch.arange(2 * 3 * 2, dtype=torch.bfloat16).reshape(6, 2)

    with patch.dict(os.environ, {"MILES_TRUE_ON_POLICY_CACHE_SGLANG_EXPERT_WEIGHTS": "1"}):
        with torch.no_grad():
            first = layer._sglang_w13_weight()
            second = layer._sglang_w13_weight()
            assert first.data_ptr() == second.data_ptr()

            layer.weight1.add_(1)
            updated = layer._sglang_w13_weight()

    assert updated.data_ptr() != first.data_ptr()
    torch.testing.assert_close(
        updated,
        layer.weight1.view(2, 2, 6).permute(0, 2, 1).contiguous(),
    )


def test_sglang_grouped_mlp_weight_cache_disabled_by_default():
    config = _make_config(hidden_size=2, moe_ffn_hidden_size=3, num_moe_experts=2)
    layer = object.__new__(SGLangGroupedMLP)
    layer.config = config
    layer.num_local_experts = 2
    layer.weight1 = torch.arange(2 * 2 * 2 * 3, dtype=torch.bfloat16).reshape(2, 12)

    with torch.no_grad():
        first = layer._sglang_w13_weight()
        second = layer._sglang_w13_weight()

    assert first.data_ptr() != second.data_ptr()


def test_sglang_grouped_mlp_masks_nonlocal_topk_ids_for_standard_ep_path():
    layer = object.__new__(SGLangGroupedMLP)
    layer.num_local_experts = 2
    global_topk_ids = torch.tensor([[0, 3, 2], [1, 2, 3]], dtype=torch.long)

    actual = layer._local_masked_topk_ids(global_topk_ids, local_expert_indices=[2, 3])

    torch.testing.assert_close(
        actual,
        torch.tensor([[-1, 1, 0], [-1, 0, 1]], dtype=torch.int32),
    )


def test_sglang_mtp_spec_uses_sglang_backend():
    config = _make_config(true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, mtp_num_layers=1)
    transformer_layer_spec = get_gpt_decoder_layer_specs(
        config, use_transformer_engine=False, normalization=config.normalization
    )[0]
    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec, backend=SGLangSpecProvider()
    )

    assert mtp_layer_spec.submodules.enorm.module is SGLangNorm
    assert mtp_layer_spec.submodules.hnorm.module is SGLangNorm
    assert mtp_layer_spec.submodules.eh_proj is SGLangColumnParallelLinear
    assert mtp_layer_spec.submodules.layer_norm.module is SGLangNorm


def test_sglang_column_parallel_linear_wrapper_forward_matches_reference():
    with _fake_tp_init():
        config = _make_config(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1, use_cpu_initialization=True
        )
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
