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


def _under_sp_or_attn_tp(moe_layer) -> bool:
    return moe_layer.config.sequence_parallel or moe_layer.attn_tp_group.size() > 1


def should_compact_true_on_policy_padding(
    moe_layer,
    padding_mask: Optional[torch.Tensor],
    intermediate_tensors,
) -> bool:
    if padding_mask is None or intermediate_tensors is not None:
        return False
    if _under_sp_or_attn_tp(moe_layer):
        # Under SP / attention-TP, compaction can make all-padding local shards
        # skip MoE collectives while peer ranks still enter EP communication.
        return False
    if moe_layer.use_shared_expert or moe_layer.config.moe_latent_size:
        return False
    return (
        uses_true_on_policy_moe_kernel(moe_layer)
        and padding_mask.dtype == torch.bool
        and bool(padding_mask.any().item())
    )


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


def _forward_sglang_ep(moe_layer, hidden_states: torch.Tensor):
    """gather -> route -> forward -> reduce -> slice"""
    hidden_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_shape[-1])
    ep_group = moe_layer.token_dispatcher.ep_group
    ep_size = moe_layer.token_dispatcher.ep_size
    ep_rank = torch.distributed.get_rank(group=ep_group)

    max_num_tokens, token_counts = _gather_ep_token_counts(
        flat.shape[0], flat.device, ep_group, ep_size
    )
    if max_num_tokens == 0:
        return torch.zeros_like(hidden_states), None

    # Step 1: EP all-gather
    if torch.is_grad_enabled():
        global_hidden = _PaddedEPAllGather.apply(
            flat, max_num_tokens, tuple(token_counts), ep_group, ep_rank, ep_size
        )
    else:
        global_hidden = torch.cat(
            _all_gather_padded(flat, max_num_tokens, ep_group, ep_size), dim=0
        )

    # Step 2: Router
    topk_weights, topk_ids = _sglang_topk_route(moe_layer, global_hidden, hidden_shape[-1])
    if topk_weights is None:
        return None

    # Step 3: Router grad ownership (training only)
    if torch.is_grad_enabled():
        local_start = ep_rank * max_num_tokens
        local_count = token_counts[ep_rank]
        topk_weights = _with_local_router_grad_owner(topk_weights, local_start, local_count)

    # Step 4: Expert forward
    w1 = moe_layer.experts.sglang_w13_weight()
    w2 = moe_layer.experts.sglang_w2_weight()

    if torch.is_grad_enabled():
        global_output = sglang_fused_experts_autograd(
            layer_number=moe_layer.layer_number,
            hidden_states=global_hidden,
            w1=w1, w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=moe_layer.config.num_moe_experts,
            num_local_experts=moe_layer.num_local_experts,
            ep_rank=ep_rank, ep_size=ep_size, ep_group=ep_group,
            activation="silu",
            allreduce_grad_hidden=False,
        )
    else:
        global_output = sglang_moe_forward(
            hidden_states=global_hidden,
            w1=w1, w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            num_experts=moe_layer.config.num_moe_experts,
            num_local_experts=moe_layer.num_local_experts,
            ep_rank=ep_rank, ep_size=ep_size,
            activation="silu",
            layer_id=moe_layer.layer_number,
        )

    # Step 5: EP reduce
    if torch.is_grad_enabled():
        global_output = _SGLangEPAllReduceSum.apply(global_output, ep_group)
    else:
        global_output = sglang_moe_ep_tree_all_reduce(global_output, ep_group)

    # Step 6: Slice local tokens
    local_start = ep_rank * max_num_tokens
    local_count = token_counts[ep_rank]
    local_output = global_output[local_start : local_start + local_count].contiguous()

    if local_count < flat.shape[0]:
        padded_output = hidden_states.new_zeros(flat.shape)
        padded_output[:local_count] = local_output
        local_output = padded_output

    return local_output.view(hidden_shape), None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _with_local_router_grad_owner(topk_weights, local_start, local_count):
    """Keep global routing values, but route gradients only through local tokens."""
    if local_count == topk_weights.shape[0]:
        return topk_weights
    mask = torch.zeros((topk_weights.shape[0], 1), dtype=torch.bool, device=topk_weights.device)
    mask[local_start : local_start + local_count] = True
    return torch.where(mask, topk_weights, topk_weights.detach())


def _sglang_topk_route(moe_layer, global_hidden, hidden_size):
    """Compute SGLang-ordered top-k routing. Returns (weights, ids) or (None, None)."""
    config = moe_layer.config
    router = moe_layer.router
    policy = resolve_true_on_policy_runtime_policy(config)

    if not policy.deterministic_moe_routing or policy.moe_topk_tiebreak != "stable_sort":
        return None, None
    if stable_topk_softmax is None:
        return None, None
    if getattr(router, "routing_type", None) == "sinkhorn":
        return None, None
    if getattr(router, "score_function", None) != "softmax":
        return None, None
    if getattr(config, "moe_router_pre_softmax", False):
        return None, None
    if getattr(config, "moe_router_num_groups", None) is not None:
        return None, None
    if getattr(config, "moe_router_fusion", False):
        return None, None
    if getattr(router, "expert_bias", None) is not None:
        return None, None
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        return None, None

    router._maintain_float32_expert_bias()
    router_weight = router.weight
    if router_weight.device.type == "cpu" and global_hidden.is_cuda:
        router_weight.data = router_weight.data.to(device=torch.cuda.current_device())
    bias = getattr(router, "bias", None)
    if bias is not None and bias.device.type == "cpu" and global_hidden.is_cuda:
        bias.data = bias.data.to(device=torch.cuda.current_device())

    router_input = global_hidden.to(config.params_dtype).view(-1, 1, hidden_size)
    logits = router_gating_linear(
        router_input, router_weight, bias, config.params_dtype
    ).view(-1, config.num_moe_experts)
    logits = router.apply_z_loss(logits, padding_mask=None)

    topk_weights, topk_ids = stable_topk_softmax(logits, config.moe_router_topk)
    topk_scaling = getattr(config, "moe_router_topk_scaling_factor", None)
    if topk_scaling:
        topk_weights = topk_weights * topk_scaling
    return topk_weights.type_as(logits), topk_ids


# ---------------------------------------------------------------------------
# EP collective helpers
# ---------------------------------------------------------------------------

def _gather_ep_token_counts(local_num_tokens, device, ep_group, ep_size):
    local_count = torch.tensor([local_num_tokens], device=device, dtype=torch.long)
    gathered = [torch.empty_like(local_count) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, local_count, group=ep_group)
    counts = [int(c.item()) for c in gathered]
    return max(counts), counts


def _all_gather_padded(local_tensor, max_num_tokens, ep_group, ep_size):
    padded = local_tensor.new_zeros((max_num_tokens, *local_tensor.shape[1:]))
    if local_tensor.shape[0] != 0:
        padded[: local_tensor.shape[0]] = local_tensor
    gathered = [torch.empty_like(padded) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, padded, group=ep_group)
    return gathered
