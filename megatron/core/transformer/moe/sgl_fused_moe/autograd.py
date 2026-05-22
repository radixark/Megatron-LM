"""Autograd wrapper for SGLang fused MoE.

The forward is delegated to ``forward.sglang_moe_forward`` (the single
source of truth).  This module only adds the custom backward pass using
Triton kernels for weight and activation gradients.

The ID remap uses ``forward.remap_global_to_local_expert_ids`` - the same
function used by the no-grad path, so both paths produce identical local
IDs.
"""

from __future__ import annotations

import os

import torch

try:
    import triton.language as tl
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        invoke_fused_moe_kernel,
        moe_align_block_size,
        silu_and_mul,
    )

    from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel

    HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD = True
except ImportError:
    tl = None
    invoke_fused_moe_kernel = None
    invoke_fused_moe_backward_kernel = None
    moe_align_block_size = None
    silu_and_mul = None
    HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD = False

from .forward import remap_global_to_local_expert_ids, sglang_moe_forward


def _scale_hidden_dgrad_by_ep() -> bool:
    value = os.environ.get("MILES_TRUE_ON_POLICY_MOE_SCALE_HIDDEN_DGRAD_BY_EP", "0")
    return value.lower() in {"1", "true", "yes", "on"}


class _FusedExpertsTritonBackward(torch.autograd.Function):
    """Wrap the shared SGLang forward and provide Triton-based backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        layer_id: int,
        num_experts: int,
        num_local_experts: int,
        ep_rank: int,
        ep_size: int,
        ep_group,
        allreduce_grad_hidden: bool,
    ):
        with torch.no_grad():
            output = sglang_moe_forward(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                ep_rank=ep_rank,
                ep_size=ep_size,
                activation=activation,
                layer_id=layer_id,
            )

        local_topk_ids = remap_global_to_local_expert_ids(
            topk_ids, num_experts, num_local_experts, ep_rank, ep_size
        )
        ctx.save_for_backward(hidden_states, w1, w2, topk_weights, local_topk_ids)
        ctx.activation = activation
        ctx.layer_id = layer_id
        ctx.ep_group = ep_group
        ctx.allreduce_grad_hidden = allreduce_grad_hidden
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, w1, w2, topk_weights, topk_ids = ctx.saved_tensors
        ep_group = ctx.ep_group

        num_tokens, hidden_size = hidden_states.shape
        num_experts, ffn_hidden_size, _ = w1.shape
        topk = topk_ids.shape[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        chunk_size = 64 * 1024
        for chunk_start in range(0, num_tokens, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            curr_hidden = hidden_states[chunk_start:chunk_end]
            curr_topk_ids = topk_ids[chunk_start:chunk_end]
            curr_topk_weights = topk_weights[chunk_start:chunk_end]
            curr_grad_output = grad_output[chunk_start:chunk_end]
            curr_tokens = curr_hidden.shape[0]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], num_experts
            )

            # Recompute w1 forward
            intermediate_cache1 = torch.empty(
                (curr_tokens * topk, ffn_hidden_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            invoke_fused_moe_kernel(
                curr_hidden,
                w1,
                None,
                intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

            intermediate_cache2 = torch.empty(
                (curr_tokens * topk, ffn_hidden_size // 2),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            silu_and_mul(intermediate_cache1.view(-1, ffn_hidden_size), intermediate_cache2)

            # Backward w2 (down-proj, routed)
            grad_intermediate_cache3 = curr_grad_output.unsqueeze(1).expand(
                curr_tokens, topk, hidden_size
            ).contiguous()
            sorted_token_ids_down, expert_ids_down, num_tokens_post_padded_down = (
                moe_align_block_size(curr_topk_ids, config["BLOCK_SIZE_M"], num_experts)
            )

            grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)
            invoke_fused_moe_backward_kernel(
                grad_output=grad_intermediate_cache3,
                input=intermediate_cache2,
                weight=w2,
                grad_input=grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids_down,
                expert_ids=expert_ids_down,
                num_tokens_post_padded=num_tokens_post_padded_down,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=compute_type,
            )
            grad_w2 += curr_grad_w2
            grad_topk_weights[chunk_start:chunk_end] = curr_grad_topk_weights

            # Manual SiLU backward
            x1, x2 = intermediate_cache1.view(-1, ffn_hidden_size).chunk(2, dim=-1)
            silu_x1 = torch.nn.functional.silu(x1)
            sig = torch.sigmoid(x1)
            dsilu_dx1 = sig + x1 * sig * (1 - sig)
            grad_x1 = grad_intermediate_cache2 * x2 * dsilu_dx1
            grad_x2 = grad_intermediate_cache2 * silu_x1
            grad_intermediate_cache1 = torch.cat([grad_x1, grad_x2], dim=-1)

            # Backward w1 (up-proj, unrouted)
            curr_grad_hidden = torch.zeros_like(curr_hidden)
            curr_grad_w1 = torch.zeros_like(w1)
            invoke_fused_moe_backward_kernel(
                grad_output=grad_intermediate_cache1,
                input=curr_hidden,
                weight=w1,
                grad_input=curr_grad_hidden,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=compute_type,
            )
            grad_hidden_states[chunk_start:chunk_end] = curr_grad_hidden
            grad_w1 += curr_grad_w1

        if ep_group is not None:
            ep_world_size = torch.distributed.get_world_size(ep_group)
            if ep_world_size > 1:
                torch.distributed.all_reduce(grad_topk_weights, group=ep_group)
                if ctx.allreduce_grad_hidden:
                    torch.distributed.all_reduce(grad_hidden_states, group=ep_group)
                elif _scale_hidden_dgrad_by_ep():
                    grad_hidden_states.div_(ep_world_size)

        return (
            grad_hidden_states,
            grad_w1,
            grad_w2,
            grad_topk_weights,
            None,  # topk_ids
            None,  # activation
            None,  # layer_id
            None,  # num_experts
            None,  # num_local_experts
            None,  # ep_rank
            None,  # ep_size
            None,  # ep_group
            None,  # allreduce_grad_hidden
        )


def sglang_fused_experts_autograd(
    *,
    layer_number: int,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
    ep_group,
    activation: str = "silu",
    allreduce_grad_hidden: bool = True,
) -> torch.Tensor:
    """Autograd-wrapped SGLang MoE forward with Triton backward.

    The forward uses the same ``sglang_moe_forward`` as the no-grad path.
    The ID remap uses the same ``remap_global_to_local_expert_ids``.
    This wrapper only attaches the backward pass.
    """
    if not HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD:
        raise RuntimeError("SGLang fused MoE Triton backward is not available")

    if hidden_states.dtype != w1.dtype:
        hidden_states = hidden_states.to(w1.dtype)

    return _FusedExpertsTritonBackward.apply(
        hidden_states.contiguous(),
        w1.contiguous(),
        w2.contiguous(),
        topk_weights.to(dtype=hidden_states.dtype).contiguous(),
        topk_ids.contiguous(),
        activation,
        layer_number,
        num_experts,
        num_local_experts,
        ep_rank,
        ep_size,
        ep_group,
        allreduce_grad_hidden,
    )
