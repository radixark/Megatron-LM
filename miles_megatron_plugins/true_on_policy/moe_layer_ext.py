"""True-on-policy MoE layer extensions.

EP-invariant local-masked forward and padding compaction extracted from
Megatron's ``MoELayer`` so the core file stays close to upstream.
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
    from sglang.srt.tp_invariant_ops import stable_topk_softmax
except ImportError:
    stable_topk_softmax = None

try:
    from megatron.core.transformer.moe.sgl_fused_moe.autograd import (
        HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD,
        sglang_fused_experts_autograd,
    )
except ImportError:
    HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD = False
    sglang_fused_experts_autograd = None


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def uses_true_on_policy_moe_kernel(moe_layer) -> bool:
    """Return whether this layer should use the contract-selected MoE kernel.

    The true-on-policy contract chooses kernel requirements.  The direct MoE
    path is enabled when the active contract requires the MoE SGLang math
    kernel.  Unsupported runtime layouts raise from the direct path instead of
    silently falling back to Megatron MoE.
    """
    policy = resolve_true_on_policy_runtime_policy(moe_layer.config)
    return policy.enabled and policy.requires_kernel(QWEN3_MOE_SGLANG_MATH)


def _has_true_on_policy_padding(padding_mask: Optional[torch.Tensor]) -> bool:
    return padding_mask is not None and bool(padding_mask.any().item())


def _require_direct_sglang_moe_ready(moe_layer) -> None:
    config = moe_layer.config
    if moe_layer.use_shared_expert:
        raise RuntimeError("Qwen3 true-on-policy MoE does not support shared experts")
    if config.moe_latent_size:
        raise RuntimeError("Qwen3 true-on-policy MoE does not support MoE latent projection")
    if config.moe_router_topk <= 1:
        raise RuntimeError("Qwen3 true-on-policy MoE requires moe_router_topk > 1")
    if config.moe_permute_fusion:
        raise RuntimeError("Qwen3 true-on-policy MoE requires MoE permute fusion disabled")
    if not isinstance(moe_layer.experts, SGLangGroupedMLP):
        raise TypeError(
            "Qwen3 true-on-policy MoE requires SGLangGroupedMLP experts; "
            f"got {type(moe_layer.experts).__name__}"
        )

    router = moe_layer.router
    if getattr(config, "moe_z_loss_coeff", None) is not None:
        raise RuntimeError("Qwen3 true-on-policy MoE does not support router z-loss")
    if hasattr(router, "is_aux_loss_enabled") and router.is_aux_loss_enabled():
        raise RuntimeError(
            "Qwen3 true-on-policy MoE does not support router auxiliary load-balancing loss"
        )

    dispatcher = moe_layer.token_dispatcher
    if getattr(dispatcher, "drop_and_pad", False):
        raise RuntimeError("Qwen3 true-on-policy MoE does not support drop-and-pad dispatch")
    if getattr(dispatcher, "tp_size", 1) != 1:
        raise RuntimeError("Qwen3 true-on-policy MoE currently requires MoE TP size 1")
    if getattr(dispatcher, "ep_size", 1) <= 1:
        raise RuntimeError("Qwen3 true-on-policy MoE direct path requires EP size > 1")


# ---------------------------------------------------------------------------
# Padding compaction
# ---------------------------------------------------------------------------

def forward_compacted_true_on_policy_padding(
    hidden_states: torch.Tensor,
    padding_mask: torch.Tensor,
    custom_forward,
):
    hidden_shape = hidden_states.shape
    flat_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
    flat_padding_mask = padding_mask.reshape(-1)
    valid_mask = ~flat_padding_mask

    if not bool(valid_mask.any().item()):
        return torch.zeros_like(hidden_states), None

    compact_hidden_states = flat_hidden_states[valid_mask].contiguous().view(
        -1, 1, hidden_shape[-1]
    )
    compact_output, mlp_bias = custom_forward(compact_hidden_states, None, None)
    if mlp_bias is not None:
        raise AssertionError("MoE true-on-policy padding compaction does not support bias")

    flat_output = compact_output.new_zeros(flat_hidden_states.shape)
    flat_output[valid_mask] = compact_output.reshape(-1, hidden_shape[-1])
    return flat_output.view(hidden_shape), None


# ---------------------------------------------------------------------------
# Top-level entry point called from MoELayer.forward
# ---------------------------------------------------------------------------

def run_direct_sglang_ep_forward(
    moe_layer,
    hidden_states: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    intermediate_tensors,
) -> tuple:
    """Run the SGLang local-masked EP forward path.

    The caller should gate this with
    ``uses_true_on_policy_moe_kernel``.  This function raises when the
    required true-on-policy SGLang path is not wired correctly; it never falls
    through to Megatron MoE.  The grad-enabled path uses a PyTorch autograd
    wrapper around the SGLang forward and Triton backward kernels, without
    running a side Megatron MoE forward.
    """
    if not uses_true_on_policy_moe_kernel(moe_layer):
        raise RuntimeError(
            "Direct SGLang MoE forward was called without a true-on-policy MoE policy"
        )

    _require_direct_sglang_moe_ready(moe_layer)
    if _has_true_on_policy_padding(padding_mask):
        raise RuntimeError(
            "Qwen3 true-on-policy MoE received padding in the direct SGLang path; "
            "padding should be compacted before the MoE forward"
        )

    if torch.is_grad_enabled():
        if intermediate_tensors is not None:
            raise RuntimeError(
                "Qwen3 true-on-policy MoE direct autograd path does not support "
                "intermediate_tensors"
            )
        if os.environ.get("MILES_TRUE_ON_POLICY_USE_SGLANG_MOE_TRITON_BWD", "1") != "1":
            raise RuntimeError(
                "Qwen3 true-on-policy MoE requires the SGLang Triton backward path; "
                "MILES_TRUE_ON_POLICY_USE_SGLANG_MOE_TRITON_BWD disabled it"
            )
        if not HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD:
            raise RuntimeError("SGLang fused MoE autograd path is not available")
        output = _forward_sglang_local_masked_ep_autograd(moe_layer, hidden_states)
    else:
        output = _forward_sglang_local_masked_ep(moe_layer, hidden_states)

    if output is None:
        raise RuntimeError(
            "Qwen3 true-on-policy MoE requires the direct SGLang MoE path, but the "
            "current router/config is not supported"
        )
    return output


# ---------------------------------------------------------------------------
# EP-invariant forward implementation
# ---------------------------------------------------------------------------

class _PaddedEPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_tensor: torch.Tensor,
        max_num_tokens: int,
        token_counts: tuple[int, ...],
        ep_group,
        ep_rank: int,
        ep_size: int,
    ):
        ctx.max_num_tokens = max_num_tokens
        ctx.token_counts = token_counts
        ctx.ep_group = ep_group
        ctx.ep_rank = ep_rank
        ctx.ep_size = ep_size

        padded_shape = (max_num_tokens, *local_tensor.shape[1:])
        padded_local = local_tensor.new_zeros(padded_shape)
        if local_tensor.shape[0] != 0:
            padded_local[: local_tensor.shape[0]] = local_tensor
        gathered = [torch.empty_like(padded_local) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, padded_local, group=ep_group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_chunks = grad_output.contiguous().view(
            ctx.ep_size, ctx.max_num_tokens, *grad_output.shape[1:]
        )
        grad_local = torch.empty_like(grad_chunks[ctx.ep_rank])
        torch.distributed.reduce_scatter(
            grad_local,
            [chunk.contiguous() for chunk in grad_chunks.unbind(0)],
            group=ctx.ep_group,
        )
        local_num_tokens = ctx.token_counts[ctx.ep_rank]
        return grad_local[:local_num_tokens], None, None, None, None, None


class _SGLangEPAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, ep_group):
        ctx.ep_group = ep_group
        with torch.no_grad():
            return sglang_moe_ep_tree_all_reduce(input_, ep_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = sglang_moe_ep_tree_all_reduce(grad_output.contiguous(), ctx.ep_group)
        if _average_ep_reduce_backward() and ctx.ep_group is not None:
            ep_world_size = torch.distributed.get_world_size(ctx.ep_group)
            if ep_world_size > 1:
                grad_input = grad_input / ep_world_size
        return grad_input, None


def _average_ep_reduce_backward() -> bool:
    value = os.environ.get("MILES_TRUE_ON_POLICY_MOE_AVG_EP_REDUCE_BWD", "0")
    return value.lower() in {"1", "true", "yes", "on"}


def _forward_sglang_local_masked_ep(moe_layer, hidden_states: torch.Tensor):
    hidden_shape = hidden_states.shape
    flat_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
    ep_group = moe_layer.token_dispatcher.ep_group
    ep_size = moe_layer.token_dispatcher.ep_size
    ep_rank = torch.distributed.get_rank(group=ep_group)
    local_num_tokens = flat_hidden_states.shape[0]
    max_num_tokens, token_counts = _gather_ep_token_counts(
        local_num_tokens, flat_hidden_states.device, ep_group, ep_size
    )

    hidden_chunks = _all_gather_padded_ep_tensor(
        flat_hidden_states, max_num_tokens, ep_group, ep_size
    )

    if max_num_tokens == 0:
        return torch.zeros_like(hidden_states), None

    return _forward_global_padded(
        moe_layer, hidden_states, hidden_chunks,
        max_num_tokens, token_counts, ep_group, ep_rank,
    )


def _forward_global_padded(
    moe_layer,
    hidden_states: torch.Tensor,
    hidden_chunks: list[torch.Tensor],
    max_num_tokens: int,
    token_counts: list[int],
    ep_group,
    ep_rank: int,
):
    hidden_shape = hidden_states.shape
    global_hidden_states = torch.cat(hidden_chunks, dim=0)
    topk_route = _try_sglang_ordered_topk_route(
        moe_layer, global_hidden_states, hidden_shape[-1]
    )
    if topk_route is None:
        return None

    topk_weights, global_topk_ids = topk_route
    global_output = moe_layer.experts.forward_sglang_local_masked_topk(
        global_hidden_states,
        topk_weights,
        global_topk_ids,
        moe_layer.local_expert_indices,
    )
    global_output = sglang_moe_ep_tree_all_reduce(global_output, ep_group)

    local_num_tokens = token_counts[ep_rank]
    local_start = ep_rank * max_num_tokens
    local_output = global_output[local_start : local_start + local_num_tokens].contiguous()
    if local_num_tokens < hidden_states.reshape(-1, hidden_shape[-1]).shape[0]:
        padded_output = hidden_states.new_zeros(
            hidden_states.reshape(-1, hidden_shape[-1]).shape
        )
        padded_output[:local_num_tokens] = local_output
        local_output = padded_output
    return local_output.view(hidden_shape), None


def _forward_sglang_local_masked_ep_autograd(moe_layer, hidden_states: torch.Tensor):
    hidden_shape = hidden_states.shape
    flat_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
    ep_group = moe_layer.token_dispatcher.ep_group
    ep_size = moe_layer.token_dispatcher.ep_size
    ep_rank = torch.distributed.get_rank(group=ep_group)
    local_num_tokens = flat_hidden_states.shape[0]
    max_num_tokens, token_counts = _gather_ep_token_counts(
        local_num_tokens, flat_hidden_states.device, ep_group, ep_size
    )
    if max_num_tokens == 0:
        return torch.zeros_like(hidden_states), None

    global_hidden_states = _PaddedEPAllGather.apply(
        flat_hidden_states,
        max_num_tokens,
        tuple(token_counts),
        ep_group,
        ep_rank,
        ep_size,
    )
    topk_route = _try_sglang_ordered_topk_route(
        moe_layer, global_hidden_states, hidden_shape[-1]
    )
    if topk_route is None:
        return None

    topk_weights, global_topk_ids = topk_route
    local_start = ep_rank * max_num_tokens
    local_count = token_counts[ep_rank]
    topk_weights = _with_local_router_grad_owner(
        topk_weights, local_start, local_count
    )
    w1 = moe_layer.experts._sglang_w13_weight()
    w2 = moe_layer.experts._sglang_w2_weight()
    global_output = sglang_fused_experts_autograd(
        layer_number=moe_layer.layer_number,
        hidden_states=global_hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=global_topk_ids,
        num_experts=moe_layer.config.num_moe_experts,
        num_local_experts=moe_layer.num_local_experts,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_group=ep_group,
        activation="silu",
        # The input was produced by an EP all-gather. Its backward reduce
        # already sums per-expert hidden-state gradients across EP ranks.
        allreduce_grad_hidden=False,
    )
    global_output = _SGLangEPAllReduceSum.apply(global_output, ep_group)

    local_output = global_output[
        local_start : local_start + token_counts[ep_rank]
    ].contiguous()
    return local_output.view(hidden_shape), None


def _with_local_router_grad_owner(
    topk_weights: torch.Tensor,
    local_start: int,
    local_count: int,
) -> torch.Tensor:
    """Keep global routing values, but route gradients only through local tokens."""
    if local_count == topk_weights.shape[0]:
        return topk_weights

    owner_mask = torch.zeros(
        (topk_weights.shape[0], 1),
        dtype=torch.bool,
        device=topk_weights.device,
    )
    owner_mask[local_start : local_start + local_count] = True
    return torch.where(owner_mask, topk_weights, topk_weights.detach())


def _try_sglang_ordered_topk_route(
    moe_layer,
    global_hidden_states: torch.Tensor,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return SGLang-ordered top-k weights/ids for the simple Qwen3 MoE route.

    This is deliberately narrow. When a config needs grouped routing, expert
    bias, token dropping, forced routing, or non-softmax scoring, ``None``
    marks the route unsupported; the top-level direct path raises if the
    true-on-policy MoE contract required SGLang execution.
    """
    router = moe_layer.router
    config = moe_layer.config
    policy = resolve_true_on_policy_runtime_policy(config)
    if not policy.deterministic_moe_routing or policy.moe_topk_tiebreak != "stable_sort":
        return None
    if stable_topk_softmax is None:
        raise RuntimeError("SGLang stable_topk_softmax is not available")
    if getattr(router, "routing_type", None) == "sinkhorn":
        return None
    if getattr(router, "score_function", None) != "softmax":
        return None
    if getattr(config, "moe_router_pre_softmax", False):
        return None
    if (
        getattr(config, "moe_router_num_groups", None) is not None
        or getattr(config, "moe_router_group_topk", None) is not None
    ):
        return None
    if getattr(config, "moe_router_fusion", False):
        return None
    if getattr(router, "expert_bias", None) is not None:
        return None
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        return None
    if (
        getattr(config, "moe_router_force_load_balancing", False)
        or getattr(config, "moe_router_force_biased", None) is not None
    ):
        return None
    if getattr(config, "moe_input_jitter_eps", None) is not None:
        return None

    router._maintain_float32_expert_bias()
    router_weight = router.weight
    if router_weight.device.type == "cpu" and global_hidden_states.is_cuda:
        router_weight.data = router_weight.data.to(device=torch.cuda.current_device())
    bias = getattr(router, "bias", None)
    if bias is not None and bias.device.type == "cpu" and global_hidden_states.is_cuda:
        bias.data = bias.data.to(device=torch.cuda.current_device())

    router_input = global_hidden_states.to(config.params_dtype).view(-1, 1, hidden_size)
    logits = router_gating_linear(
        router_input,
        router_weight,
        bias,
        config.params_dtype,
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

def _gather_ep_token_counts(
    local_num_tokens: int,
    device: torch.device,
    ep_group,
    ep_size: int,
) -> tuple[int, list[int]]:
    local_count = torch.tensor([local_num_tokens], device=device, dtype=torch.long)
    gathered_counts = [torch.empty_like(local_count) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered_counts, local_count, group=ep_group)
    token_counts = [int(count.item()) for count in gathered_counts]
    return max(token_counts), token_counts


def _all_gather_padded_ep_tensor(
    local_tensor: torch.Tensor,
    max_num_tokens: int,
    ep_group,
    ep_size: int,
) -> list[torch.Tensor]:
    padded_shape = (max_num_tokens, *local_tensor.shape[1:])
    padded_local = local_tensor.new_zeros(padded_shape)
    if local_tensor.shape[0] != 0:
        padded_local[: local_tensor.shape[0]] = local_tensor
    gathered = [torch.empty_like(padded_local) for _ in range(ep_size)]
    torch.distributed.all_gather(gathered, padded_local, group=ep_group)
    return gathered
