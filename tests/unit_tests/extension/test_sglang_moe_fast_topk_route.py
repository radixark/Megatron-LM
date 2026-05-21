from __future__ import annotations

import types

import torch

from miles_megatron_plugins.true_on_policy import moe_layer_ext
from miles_megatron_plugins.true_on_policy.moe_experts import SGLangGroupedMLP
from miles_megatron_plugins.true_on_policy.moe_layer_ext import (
    _try_sglang_ordered_topk_route,
    uses_true_on_policy_moe_kernel,
)
from miles_megatron_plugins.true_on_policy.sglang_backend import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
)
from sglang.srt.tp_invariant_ops import stable_topk


def _predicate_layer(contract_name, ep_size: int):
    config = types.SimpleNamespace(
        true_on_policy_contract=contract_name,
        expert_model_parallel_size=ep_size,
        moe_latent_size=None,
    )
    return types.SimpleNamespace(config=config, use_shared_expert=False)


def test_sglang_moe_kernel_mode_matches_contract_and_ep():
    assert not uses_true_on_policy_moe_kernel(_predicate_layer(None, ep_size=4))
    assert not uses_true_on_policy_moe_kernel(
        _predicate_layer(QWEN3_DENSE_TRUE_ON_POLICY_V1, ep_size=4)
    )
    assert not uses_true_on_policy_moe_kernel(
        _predicate_layer(QWEN3_MOE_TRUE_ON_POLICY_V1, ep_size=1)
    )
    assert uses_true_on_policy_moe_kernel(
        _predicate_layer(QWEN3_MOE_TRUE_ON_POLICY_V1, ep_size=4)
    )


def test_sglang_moe_fast_topk_route_matches_router_contract(monkeypatch):
    config = types.SimpleNamespace(
        num_moe_experts=4,
        moe_router_topk=2,
        params_dtype=torch.bfloat16,
        moe_router_pre_softmax=False,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_fusion=False,
        moe_expert_capacity_factor=None,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_input_jitter_eps=None,
        moe_router_topk_scaling_factor=None,
        true_on_policy_contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
    )
    logits = torch.tensor(
        [
            [[1.0, 2.0, 2.0, 0.0]],
            [[0.5, -1.0, 0.25, 0.5]],
        ],
        dtype=torch.bfloat16,
    )
    router = types.SimpleNamespace(
        routing_type="aux_loss",
        score_function="softmax",
        expert_bias=None,
        weight=torch.empty(4, 8, dtype=torch.bfloat16),
        bias=None,
        _maintain_float32_expert_bias=lambda: None,
        apply_z_loss=lambda router_logits, padding_mask=None: router_logits,
    )
    monkeypatch.setattr(
        moe_layer_ext,
        "router_gating_linear",
        lambda router_input, weight, bias, dtype: logits.to(dtype),
    )
    experts = types.SimpleNamespace(forward_sglang_local_masked_topk=lambda *args: None)
    moe_layer = types.SimpleNamespace(config=config, router=router, experts=experts)
    hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)

    topk_weights, topk_ids = _try_sglang_ordered_topk_route(
        moe_layer, hidden_states, hidden_size=8
    )

    scores = torch.softmax(logits.view(2, 4), dim=-1, dtype=torch.float32)
    expected_weights, expected_ids = stable_topk(scores, top_k=2)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(topk_ids, expected_ids)
    torch.testing.assert_close(topk_weights, expected_weights.to(torch.bfloat16))


def test_sglang_moe_fast_topk_route_falls_back_for_grouped_routing():
    config = types.SimpleNamespace(
        moe_router_pre_softmax=False,
        moe_router_num_groups=2,
        moe_router_group_topk=1,
        moe_router_fusion=False,
        moe_expert_capacity_factor=None,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_input_jitter_eps=None,
        true_on_policy_contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
    )
    router = types.SimpleNamespace(
        routing_type="aux_loss",
        score_function="softmax",
        expert_bias=None,
    )
    experts = types.SimpleNamespace(forward_sglang_local_masked_topk=lambda *args: None)
    moe_layer = types.SimpleNamespace(config=config, router=router, experts=experts)

    assert _try_sglang_ordered_topk_route(moe_layer, torch.zeros(1, 4), 4) is None


def test_sglang_expert_weight_cache_invalidates_after_inplace_update(monkeypatch):
    monkeypatch.setenv("MILES_TRUE_ON_POLICY_CACHE_SGLANG_EXPERT_WEIGHTS", "1")
    grouped_mlp = object.__new__(SGLangGroupedMLP)
    weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    first = grouped_mlp._cached_sglang_weight(
        cache_name="_test_weight_cache",
        weight=weight,
        view_shape=(2, 3, 4),
        permute_dims=(0, 2, 1),
    )

    with torch.no_grad():
        weight.add_(1000)

    second = grouped_mlp._cached_sglang_weight(
        cache_name="_test_weight_cache",
        weight=weight,
        view_shape=(2, 3, 4),
        permute_dims=(0, 2, 1),
    )
    expected = weight.view(2, 3, 4).permute(0, 2, 1).contiguous()

    assert not torch.equal(first, second)
    torch.testing.assert_close(second, expected)
