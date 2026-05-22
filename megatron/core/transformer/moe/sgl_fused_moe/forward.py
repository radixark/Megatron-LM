"""Canonical SGLang MoE forward.

Both inference (no-grad) and training (autograd-wrapped) paths call this
same function.  This structural guarantee makes it impossible for the two
paths to diverge, which is the core invariant of true-on-policy alignment.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

try:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    HAVE_SGLANG_FUSED_MOE = True
except ImportError:
    MoeRunnerConfig = None
    StandardTopKOutput = None
    fused_experts = None
    get_global_server_args = None
    set_global_server_args_for_scheduler = None
    HAVE_SGLANG_FUSED_MOE = False


def ensure_sglang_server_args() -> None:
    """Initialize SGLang global server args for deterministic MoE inference."""
    if not HAVE_SGLANG_FUSED_MOE:
        raise RuntimeError("SGLang fused MoE is not available")
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            SimpleNamespace(
                enable_fused_moe_sum_all_reduce=False,
                enable_deterministic_inference=True,
                rl_on_policy_target="fsdp_tp",
            )
        )


def remap_global_to_local_expert_ids(
    global_topk_ids: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
) -> torch.Tensor:
    """Map global expert IDs to local indices; non-local experts become -1.

    This is the ONE remap function used by both inference and training.
    """
    if ep_size <= 1:
        return global_topk_ids.to(torch.int32)

    local_start = ep_rank * num_local_experts
    local_expert_mapping = torch.full(
        (num_experts,), -1, dtype=torch.int32, device=global_topk_ids.device
    )
    local_expert_mapping[local_start : local_start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=global_topk_ids.device
    )
    return local_expert_mapping[global_topk_ids.long()]


def sglang_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
    activation: str = "silu",
    layer_id: int = 0,
) -> torch.Tensor:
    """The single canonical SGLang fused MoE forward.

    Both inference and autograd-wrapped training call this exact function.
    It performs:
      1. Remap global expert IDs to local (non-local -> -1)
      2. Build SGLang StandardTopKOutput and MoeRunnerConfig
      3. Call SGLang fused_experts
    """
    if not HAVE_SGLANG_FUSED_MOE:
        raise RuntimeError("SGLang fused MoE is not available")

    ensure_sglang_server_args()

    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    if flat_hidden.dtype != w1.dtype:
        flat_hidden = flat_hidden.to(w1.dtype)
    flat_hidden = flat_hidden.contiguous()

    local_topk_ids = remap_global_to_local_expert_ids(
        topk_ids, num_experts, num_local_experts, ep_rank, ep_size
    )
    topk_weights = topk_weights.to(
        device=flat_hidden.device, dtype=flat_hidden.dtype
    ).contiguous()

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=local_topk_ids,
        router_logits=None,
    )
    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=flat_hidden.shape[-1],
        intermediate_size_per_partition=w2.shape[2],
        layer_id=layer_id,
        top_k=topk_weights.shape[-1],
        params_dtype=w1.dtype,
        activation=activation,
        is_gated=True,
        apply_router_weight_on_input=False,
        inplace=True,
        no_combine=False,
    )

    return fused_experts(
        hidden_states=flat_hidden.clone(),
        w1=w1.contiguous(),
        w2=w2.contiguous(),
        topk_output=topk_output,
        moe_runner_config=runner_config,
    )
