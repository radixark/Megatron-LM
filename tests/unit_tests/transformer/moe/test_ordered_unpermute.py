# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stable FP32 combine preserves weighted sums and gradients without global mode changes."""

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import unpermute


@pytest.mark.parametrize("tokens,empty", [(0, True), (4, True), (4, False)])
def test_ordered_combine_values_and_gradients(tokens, empty):
    routing = torch.zeros(tokens, 3, dtype=torch.bool, device="cuda")
    if not empty:
        routing[:] = torch.tensor(
            [[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1]], device="cuda", dtype=torch.bool
        )
    indices = routing.T.contiguous().nonzero()[:, 1]
    source = torch.randn(indices.numel(), 5, device="cuda", dtype=torch.float64, requires_grad=True)
    probs = torch.rand(tokens, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    actual = unpermute(
        source, indices, (tokens, 5), probs=probs, routing_map=routing, deterministic=True
    )
    weights = probs.T.contiguous().masked_select(routing.T.contiguous())
    weighted = source * weights[:, None]
    expected = (
        torch.stack([weighted[indices == i].sum(0) for i in range(tokens)])
        if tokens
        else weighted.new_zeros((0, 5)) + weighted.sum() * 0
    )
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(
        (actual * upstream).sum(), (source, probs), retain_graph=True
    )
    expected_grads = torch.autograd.grad((expected * upstream).sum(), (source, probs))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_bf16_production_width_combine_repeats_exactly():
    assert not torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(9041)
    tokens, hidden, experts, topk = 1024, 4096, 256, 8
    routing = torch.zeros(tokens, experts, device="cuda", dtype=torch.bool)
    selected = (
        torch.arange(tokens, device="cuda")[:, None] + torch.arange(topk, device="cuda")
    ) % experts
    routing.scatter_(1, selected, True)
    indices = routing.T.contiguous().nonzero()[:, 1]
    probs = torch.rand(tokens, experts, device="cuda") * routing
    probs /= probs.sum(1, keepdim=True)
    source = torch.randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    first = unpermute(
        source, indices, (tokens, hidden), probs=probs, routing_map=routing, deterministic=True
    )
    for _ in range(20):
        actual = unpermute(
            source, indices, (tokens, hidden), probs=probs, routing_map=routing, deterministic=True
        )
        assert torch.equal(actual, first)
    assert not torch.are_deterministic_algorithms_enabled()
    weighted = source.float() * probs.T.contiguous().masked_select(routing.T.contiguous())[:, None]
    try:
        torch.use_deterministic_algorithms(True)
        expected = (
            torch.zeros(tokens, hidden, device="cuda").index_add_(0, indices, weighted).bfloat16()
        )
    finally:
        torch.use_deterministic_algorithms(False)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)


def test_stable_combine_rejects_fused_path():
    with pytest.raises(ValueError, match="requires unfused"):
        unpermute(
            torch.empty(0, 1),
            torch.empty(0, dtype=torch.long),
            (0, 1),
            fused=True,
            deterministic=True,
        )
