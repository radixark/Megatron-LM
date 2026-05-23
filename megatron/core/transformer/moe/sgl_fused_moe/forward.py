"""Canonical SGLang MoE forward — the single source of truth.

Both inference (no-grad) and training (autograd-wrapped) paths call
``sglang_moe_forward``.  This makes forward divergence structurally impossible.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import get_global_server_args, set_global_server_args_for_scheduler


def _ensure_server_args():
    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            SimpleNamespace(
                enable_fused_moe_sum_all_reduce=False,
                enable_deterministic_inference=True,
            )
        )


def remap_global_to_local_expert_ids(
    global_topk_ids: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    ep_rank: int,
    ep_size: int,
) -> torch.Tensor:
    """Map global expert IDs to local indices; non-local experts become -1."""
    if ep_size <= 1:
        return global_topk_ids.to(torch.int32)

    mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=global_topk_ids.device)
    start = ep_rank * num_local_experts
    mapping[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=global_topk_ids.device
    )
    return mapping[global_topk_ids.long()]


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
    """Single canonical forward: remap IDs -> build config -> call fused_experts."""
    _ensure_server_args()

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if flat.dtype != w1.dtype:
        flat = flat.to(w1.dtype)
    flat = flat.contiguous()

    local_ids = remap_global_to_local_expert_ids(
        topk_ids, num_experts, num_local_experts, ep_rank, ep_size
    )
    topk_weights = topk_weights.to(device=flat.device, dtype=flat.dtype).contiguous()

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=local_ids, router_logits=None
    )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        hidden_size=w1.shape[1],
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
        hidden_states=flat.clone(),
        w1=w1.contiguous(),
        w2=w2.contiguous(),
        topk_output=topk_output,
        moe_runner_config=config,
    )
