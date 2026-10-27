# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import permute, unpermute

NUM_TOKENS = 64
NUM_EXPERTS = 16
TOPK = 8
HIDDEN = 1024


def _routing_map():
    generator = torch.Generator(device="cuda").manual_seed(0)
    scores = torch.rand(NUM_TOKENS, NUM_EXPERTS, device="cuda", generator=generator)
    experts = scores.topk(TOPK, dim=-1).indices
    return torch.zeros(NUM_TOKENS, NUM_EXPERTS, dtype=torch.bool, device="cuda").scatter_(
        1, experts, True
    )


def _assert_within_one_final_rounding(actual, reference, abs_sum):
    # fp32 accumulation error of TOPK terms plus one bf16 round of the result.
    bound = reference.abs() * 2**-8 + abs_sum * TOPK * 2**-24 * 1.01
    error = (actual.double() - reference).abs()
    assert (error <= bound).all(), f"max excess {(error - bound).max().item():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unpermute_accumulates_in_fp32():
    routing_map = _routing_map()
    tokens = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _, _, sorted_indices, _, _ = permute(tokens, routing_map, num_out_tokens=NUM_TOKENS * TOPK)
    expert_outputs = torch.randn(NUM_TOKENS * TOPK, HIDDEN, dtype=torch.bfloat16, device="cuda")

    output = unpermute(
        expert_outputs, sorted_indices, restore_shape=tokens.shape, accumulate_in_fp32=True
    )

    assert output.dtype == torch.bfloat16
    reference = torch.zeros(tokens.shape, dtype=torch.float64, device="cuda")
    reference.index_add_(0, sorted_indices, expert_outputs.double())
    abs_sum = torch.zeros_like(reference).index_add_(
        0, sorted_indices, expert_outputs.double().abs()
    )
    _assert_within_one_final_rounding(output, reference, abs_sum)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_permute_backward_accumulates_in_fp32():
    routing_map = _routing_map()
    tokens = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )

    permuted, _, sorted_indices, _, _ = permute(
        tokens, routing_map, num_out_tokens=NUM_TOKENS * TOPK, accumulate_in_fp32=True
    )
    assert permuted.dtype == torch.bfloat16
    grad_output = torch.randn_like(permuted)
    permuted.backward(grad_output)

    assert tokens.grad.dtype == torch.bfloat16
    reference = torch.zeros(tokens.shape, dtype=torch.float64, device="cuda")
    reference.index_add_(0, sorted_indices, grad_output.double())
    abs_sum = torch.zeros_like(reference).index_add_(0, sorted_indices, grad_output.double().abs())
    _assert_within_one_final_rounding(tokens.grad, reference, abs_sum)
