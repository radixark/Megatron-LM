"""True-on-policy MoE layer extensions.

ONE orchestration path: gather -> route -> forward -> reduce -> slice.
The only branch is ``if torch.is_grad_enabled()``, which controls whether
the autograd wrapper (backward kernels) is attached.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from megatron.core.transformer.moe.moe_utils import router_gating_linear

from .contracts import resolve_true_on_policy_runtime_policy
from .moe_experts import SGLangGroupedMLP
from .moe_reduce import sglang_moe_ep_tree_all_reduce
from .schema import QWEN3_MOE_SGLANG_MATH

try:
    from megatron.core.transformer.moe.sgl_fused_moe.forward import sglang_moe_forward
except ImportError:
    sglang_moe_forward = None

try:
    from sglang.srt.tp_invariant_ops import stable_topk_softmax
except ImportError:
    stable_topk_softmax = None

try:
    from megatron.core.transformer.moe.sgl_fused_moe.autograd import (
        sglang_fused_experts_autograd,
    )
except ImportError:
    sglang_fused_experts_autograd = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def uses_true_on_policy_moe_kernel(moe_layer) -> bool:
    policy = resolve_true_on_policy_runtime_policy(moe_layer.config)
    return policy.enabled and policy.requires_kernel(QWEN3_MOE_SGLANG_MATH)


def forward_compacted_true_on_policy_padding(
    hidden_states: torch.Tensor,
    padding_mask: torch.Tensor,
    custom_forward,
):
    """Strip padding tokens before MoE, scatter output back."""
    hidden_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_shape[-1])
    valid_mask = ~padding_mask.reshape(-1)

    if not bool(valid_mask.any().item()):
        return torch.zeros_like(hidden_states), None

    compact = flat[valid_mask].contiguous().view(-1, 1, hidden_shape[-1])
    compact_output, mlp_bias = custom_forward(compact, None, None)
    assert mlp_bias is None

    flat_output = compact_output.new_zeros(flat.shape)
    flat_output[valid_mask] = compact_output.reshape(-1, hidden_shape[-1])
    return flat_output.view(hidden_shape), None


def run_direct_sglang_ep_forward(
    moe_layer,
    hidden_states: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    intermediate_tensors,
) -> tuple:
    """Entry point from MoELayer.forward."""
    assert uses_true_on_policy_moe_kernel(moe_layer)
    assert isinstance(moe_layer.experts, SGLangGroupedMLP)
    assert moe_layer.token_dispatcher.ep_size > 1
    assert not (padding_mask is not None and bool(padding_mask.any().item()))

    output = _forward_sglang_ep(moe_layer, hidden_states)
    assert output is not None, "Router config not supported for true-on-policy MoE"
    return output


# ---------------------------------------------------------------------------
# Unified EP forward
# ---------------------------------------------------------------------------

class _PaddedEPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_tensor, max_num_tokens, token_counts, ep_group, ep_rank, ep_size):
        ctx.max_num_tokens = max_num_tokens
        ctx.token_counts = token_counts
        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size

        padded = local_tensor.new_zeros((max_num_tokens, *local_tensor.shape[1:]))
        if local_tensor.shape[0] != 0:
            padded[: local_tensor.shape[0]] = local_tensor
        gathered = [torch.empty_like(padded) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, padded, group=ep_group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunks = grad_output.contiguous().view(
            ctx.ep_size, ctx.max_num_tokens, *grad_output.shape[1:]
        )
        grad_local = torch.empty_like(chunks[ctx.ep_rank])
        torch.distributed.reduce_scatter(
            grad_local,
            [c.contiguous() for c in chunks.unbind(0)],
            group=ctx.ep_group,
        )
        n = ctx.token_counts[ctx.ep_rank]
        return grad_local[:n], None, None, None, None, None


class _SGLangEPAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, ep_group):
        ctx.ep_group = ep_group
        with torch.no_grad():
            return sglang_moe_ep_tree_all_reduce(input_, ep_group)

    @staticmethod
    def backward(ctx, grad_output):
        grad = sglang_moe_ep_tree_all_reduce(grad_output.contiguous(), ctx.ep_group)
        if os.environ.get("MILES_TRUE_ON_POLICY_MOE_AVG_EP_REDUCE_BWD", "0") == "1":
            ws = torch.distributed.get_world_size(ctx.ep_group)
            if ws > 1:
                grad = grad / ws
        return grad, None


