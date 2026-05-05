"""Deterministic MoE combine for true-on-policy.

SGLang adds top-k expert contributions in a fixed per-token order (lowest
expert-id first within each top-k slot). Megatron's default ``unpermute``
uses expert-major storage order, which produces a different FP addition
sequence. The helpers here reorder the unpermuted outputs to match SGLang's
addition order so the combine step is bit-exact across the two engines.
"""

from __future__ import annotations

from typing import Optional

import torch


def _sglang_ordered_topk_ids_from_probs(probs: torch.Tensor, topk: int) -> torch.Tensor:
    """Rebuild SGLang's per-token expert order from routed probabilities."""
    num_experts = probs.shape[-1]
    expert_ids = torch.arange(num_experts, device=probs.device, dtype=torch.float32)
    probs_fp32 = probs.float()
    tie_step = torch.finfo(torch.float32).eps * probs_fp32.abs().clamp_min(1.0 / num_experts)
    ordered_scores = probs_fp32 - expert_ids.view(1, -1) * tie_step
    return torch.topk(ordered_scores, k=topk, dim=-1).indices


def _sglang_ordered_moe_assignment_rows(
    sorted_indices: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Map each token's SGLang-ordered top-k experts to rows in expert-major storage."""
    num_tokens, num_experts = routing_map.shape
    sorted_indices = sorted_indices.reshape(-1)
    routing_map_t = routing_map.T.contiguous()
    expert_ids = (
        torch.arange(num_experts, device=routing_map.device)
        .unsqueeze(1)
        .expand(num_experts, num_tokens)
        .masked_select(routing_map_t)
    )

    assignment_keys = sorted_indices.to(torch.long) * num_experts + expert_ids.to(torch.long)
    sorted_keys, sorted_key_rows = torch.sort(assignment_keys)

    ordered_expert_ids = _sglang_ordered_topk_ids_from_probs(probs, topk).to(torch.long)
    token_ids = torch.arange(num_tokens, device=probs.device, dtype=torch.long).unsqueeze(1)
    ordered_keys = (token_ids * num_experts + ordered_expert_ids).reshape(-1)

    key_positions = torch.searchsorted(sorted_keys, ordered_keys)
    return sorted_key_rows.index_select(0, key_positions).view(num_tokens, topk)


def _sglang_reduce_moe_slots(ordered_tokens: torch.Tensor, restore_shape: torch.Size) -> torch.Tensor:
    topk = ordered_tokens.shape[1]
    if topk == 1:
        return ordered_tokens[:, 0].contiguous()

    if topk == 2:
        return torch.add(ordered_tokens[:, 0], ordered_tokens[:, 1])

    if ordered_tokens.is_cuda:
        from sgl_kernel import moe_sum_reduce

        output = torch.empty(
            restore_shape,
            device=ordered_tokens.device,
            dtype=ordered_tokens.dtype,
        )
        moe_sum_reduce(ordered_tokens.contiguous(), output, 1.0)
        return output

    output = ordered_tokens[:, 0].clone()
    for topk_idx in range(1, topk):
        output = output + ordered_tokens[:, topk_idx]
    return output


def sglang_ordered_moe_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    topk: int,
    ep_size: int = 1,
    num_local_experts: Optional[int] = None,
) -> torch.Tensor:
    """Unpermute MoE outputs using SGLang's MoE contribution addition order."""
    ordered_rows = _sglang_ordered_moe_assignment_rows(
        sorted_indices=sorted_indices,
        routing_map=routing_map,
        probs=probs,
        topk=topk,
    )
    ordered_tokens = permuted_tokens.index_select(0, ordered_rows.reshape(-1)).view(
        restore_shape[0],
        topk,
        restore_shape[1],
    )

    if ep_size <= 1 or num_local_experts is None:
        return _sglang_reduce_moe_slots(ordered_tokens, restore_shape)

    ordered_expert_ids = _sglang_ordered_topk_ids_from_probs(probs, topk).to(torch.long)
    ep_rank_outputs = []
    for ep_rank in range(ep_size):
        expert_start = ep_rank * num_local_experts
        expert_end = expert_start + num_local_experts
        local_mask = (ordered_expert_ids >= expert_start) & (ordered_expert_ids < expert_end)
        local_tokens = ordered_tokens.masked_fill(~local_mask.unsqueeze(-1), 0)
        ep_rank_outputs.append(_sglang_reduce_moe_slots(local_tokens, restore_shape))

    return _sglang_reduce_moe_slots(
        torch.stack(ep_rank_outputs, dim=1).contiguous(),
        restore_shape,
    )


def should_use_sglang_ordered_combine(config) -> bool:
    """Check whether the SGLang-ordered combine path should be used."""
    if (
        torch.is_grad_enabled()
        or config.moe_router_topk <= 1
        or config.moe_pad_expert_input_to_capacity
        or config.moe_permute_fusion
    ):
        return False

    from miles_megatron_plugins.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

    policy = resolve_true_on_policy_runtime_policy(config)
    return policy.use_sglang_backend and policy.deterministic_moe_combine
