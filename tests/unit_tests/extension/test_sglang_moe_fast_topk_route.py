from __future__ import annotations

import types

import pytest
import torch

from miles_megatron_plugins.true_on_policy import moe_layer_ext
from miles_megatron_plugins.true_on_policy.moe_experts import SGLangGroupedMLP
from miles_megatron_plugins.true_on_policy.moe_layer_ext import (
    _sglang_topk_route as _try_sglang_ordered_topk_route,
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


def _ready_direct_layer(**overrides):
    config = types.SimpleNamespace(
        moe_latent_size=None, moe_router_topk=2, moe_permute_fusion=False, moe_z_loss_coeff=None
    )
    router = types.SimpleNamespace(is_aux_loss_enabled=lambda: False)
    dispatcher = types.SimpleNamespace(drop_and_pad=False, tp_size=1, ep_size=4)
    layer = types.SimpleNamespace(
        config=config,
        use_shared_expert=False,
        experts=object.__new__(SGLangGroupedMLP),
        router=router,
        token_dispatcher=dispatcher,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_sglang_moe_kernel_mode_matches_contract_and_ep():
    assert not uses_true_on_policy_moe_kernel(_predicate_layer(None, ep_size=4))
    assert not uses_true_on_policy_moe_kernel(
        _predicate_layer(QWEN3_DENSE_TRUE_ON_POLICY_V1, ep_size=4)
    )
    assert uses_true_on_policy_moe_kernel(_predicate_layer(QWEN3_MOE_TRUE_ON_POLICY_V1, ep_size=1))
    assert uses_true_on_policy_moe_kernel(_predicate_layer(QWEN3_MOE_TRUE_ON_POLICY_V1, ep_size=4))


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
    logits = torch.tensor([[[1.0, 2.0, 2.0, 0.0]], [[0.5, -1.0, 0.25, 0.5]]], dtype=torch.bfloat16)
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
    experts = types.SimpleNamespace()
    moe_layer = types.SimpleNamespace(config=config, router=router, experts=experts)
    hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)

    topk_weights, topk_ids = _try_sglang_ordered_topk_route(moe_layer, hidden_states, hidden_size=8)

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
        routing_type="aux_loss", score_function="softmax", expert_bias=None
    )
    experts = types.SimpleNamespace()
    moe_layer = types.SimpleNamespace(config=config, router=router, experts=experts)

    assert _try_sglang_ordered_topk_route(moe_layer, torch.zeros(1, 4), 4) == (None, None)
