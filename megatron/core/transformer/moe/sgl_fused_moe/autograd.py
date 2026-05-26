"""Autograd wrapper for SGLang fused MoE — backward only.

Forward delegates to ``forward.sglang_moe_forward``.
ID remap uses ``forward.remap_global_to_local_expert_ids``.
Both paths use the same functions — forward divergence is structurally impossible.
"""

from __future__ import annotations

import os

import torch
import triton.language as tl
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import silu_and_mul
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

from .forward import remap_global_to_local_expert_ids, sglang_moe_forward
from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel

HAVE_SGLANG_FUSED_EXPERTS_AUTOGRAD = True


class _FusedExpertsTritonBackward(torch.autograd.Function):
    """Shared forward + Triton backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        layer_id,
        num_experts,
        num_local_experts,
        ep_rank,
        ep_size,
        ep_group,
        allreduce_grad_hidden,
    ):
        """Run the fused MoE forward and save tensors for the Triton backward."""
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

        local_ids = remap_global_to_local_expert_ids(
            topk_ids, num_experts, num_local_experts, ep_rank, ep_size
        )
        ctx.save_for_backward(hidden_states, w1, w2, topk_weights, local_ids)
        ctx.ep_group = ep_group
        ctx.allreduce_grad_hidden = allreduce_grad_hidden
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute hidden-state, expert-weight, and top-k-weight gradients."""
        hidden_states, w1, w2, topk_weights, topk_ids = ctx.saved_tensors
        ep_group = ctx.ep_group

        num_tokens, hidden_size = hidden_states.shape
        num_experts, ffn_hidden_size, _ = w1.shape
        topk = topk_ids.shape[1]
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
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

            # Recompute w1 forward (mirrors SGLang fused_experts_impl)
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
            grad_inter3 = (
                curr_grad_output.unsqueeze(1).expand(curr_tokens, topk, hidden_size).contiguous()
            )
            sorted_down, experts_down, ntp_down = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], num_experts
            )

            grad_inter2 = torch.zeros_like(intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)
            invoke_fused_moe_backward_kernel(
                grad_output=grad_inter3,
                input=intermediate_cache2,
                weight=w2,
                grad_input=grad_inter2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_down,
                expert_ids=experts_down,
                num_tokens_post_padded=ntp_down,
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
            grad_x1 = grad_inter2 * x2 * (sig + x1 * sig * (1 - sig))
            grad_x2 = grad_inter2 * silu_x1
            grad_inter1 = torch.cat([grad_x1, grad_x2], dim=-1)

            # Backward w1 (up-proj, unrouted)
            curr_grad_hidden = torch.zeros_like(curr_hidden)
            curr_grad_w1 = torch.zeros_like(w1)
            invoke_fused_moe_backward_kernel(
                grad_output=grad_inter1,
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
            ws = torch.distributed.get_world_size(ep_group)
            if ws > 1:
                torch.distributed.all_reduce(grad_topk_weights, group=ep_group)
                if ctx.allreduce_grad_hidden:
                    torch.distributed.all_reduce(grad_hidden_states, group=ep_group)
                elif (
                    os.environ.get("MILES_TRUE_ON_POLICY_MOE_SCALE_HIDDEN_DGRAD_BY_EP", "0") == "1"
                ):
                    grad_hidden_states.div_(ws)

        return (
            grad_hidden_states,
            grad_w1,
            grad_w2,
            grad_topk_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sglang_fused_experts_autograd(
    *,
    layer_number,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    num_experts,
    num_local_experts,
    ep_rank,
    ep_size,
    ep_group,
    activation="silu",
    allreduce_grad_hidden=True,
):
    """Autograd-wrapped forward. Same math as no-grad path, plus Triton backward."""
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
