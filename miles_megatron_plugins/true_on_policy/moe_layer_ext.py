"""True-on-policy MoE layer extensions.

EP-invariant local-masked forward and padding compaction extracted from
Megatron's ``MoELayer`` so the core file stays close to upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.typed_torch import apply_module

from .contracts import resolve_true_on_policy_runtime_policy
from .moe_reduce import sglang_moe_ep_tree_all_reduce


@dataclass
class SglangEPResult:
    """Wrapper returned by ``try_sglang_ep_forward`` to the MoE layer."""

    is_final: bool
    output: tuple | None = None
    exact_output: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def _has_true_on_policy_padding(padding_mask: Optional[torch.Tensor]) -> bool:
    return padding_mask is not None and bool(padding_mask.any().item())


def _ep_invariant_moe_eligible(moe_layer) -> bool:
    """Common preconditions shared by the no-grad and straight-through paths."""
    if moe_layer.use_shared_expert or moe_layer.config.moe_latent_size:
        return False
    if moe_layer.config.moe_router_topk <= 1:
        return False
    if moe_layer.config.moe_permute_fusion:
        return False
    if not hasattr(moe_layer.experts, "forward_sglang_local_masked"):
        return False

    dispatcher = moe_layer.token_dispatcher
    if getattr(dispatcher, "drop_and_pad", False):
        return False
    if getattr(dispatcher, "tp_size", 1) != 1 or getattr(dispatcher, "ep_size", 1) <= 1:
        return False

    policy = resolve_true_on_policy_runtime_policy(moe_layer.config)
    return policy.ep_invariant_moe and policy.deterministic_moe_dispatch


def should_use_sglang_local_masked_ep_forward(
    moe_layer,
    padding_mask: Optional[torch.Tensor],
    shared_expert_output: Optional[torch.Tensor],
) -> bool:
    if torch.is_grad_enabled():
        return False
    if _has_true_on_policy_padding(padding_mask) or shared_expert_output is not None:
        return False
    return _ep_invariant_moe_eligible(moe_layer)


def should_use_sglang_local_masked_ep_straight_through(
    moe_layer,
    padding_mask: Optional[torch.Tensor],
    shared_expert_output: Optional[torch.Tensor],
    intermediate_tensors,
) -> bool:
    if not torch.is_grad_enabled() or intermediate_tensors is not None:
        return False
    if _has_true_on_policy_padding(padding_mask) or shared_expert_output is not None:
        return False
    return _ep_invariant_moe_eligible(moe_layer)


def should_compact_true_on_policy_padding(
    moe_layer,
    padding_mask: Optional[torch.Tensor],
    intermediate_tensors,
) -> bool:
    if padding_mask is None or intermediate_tensors is not None:
        return False
    if moe_layer.use_shared_expert or moe_layer.config.moe_latent_size:
        return False

    policy = resolve_true_on_policy_runtime_policy(moe_layer.config)
    return (
        policy.ep_invariant_moe
        and padding_mask.dtype == torch.bool
        and bool(padding_mask.any().item())
    )


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

def try_sglang_ep_forward(
    moe_layer,
    hidden_states: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    shared_expert_output: Optional[torch.Tensor],
    intermediate_tensors,
) -> SglangEPResult | None:
    """Try the SGLang local-masked EP forward path.

    Returns ``None`` when the path is not applicable, allowing the caller to
    fall through to the normal Megatron MoE forward.  Otherwise returns a
    ``SglangEPResult`` indicating whether the result is final (no-grad) or
    a straight-through exact output (grad-enabled).
    """
    if should_use_sglang_local_masked_ep_forward(
        moe_layer, padding_mask, shared_expert_output
    ):
        return SglangEPResult(
            is_final=True,
            output=_forward_sglang_local_masked_ep(moe_layer, hidden_states),
        )

    if should_use_sglang_local_masked_ep_straight_through(
        moe_layer, padding_mask, shared_expert_output, intermediate_tensors
    ):
        with torch.no_grad():
            exact_output = _forward_sglang_local_masked_ep(moe_layer, hidden_states)[0]
        return SglangEPResult(is_final=False, exact_output=exact_output)

    return None


# ---------------------------------------------------------------------------
# EP-invariant forward implementation
# ---------------------------------------------------------------------------

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
        global_probs, _ = apply_module(moe_layer.router)(
            global_hidden_states.view(-1, 1, hidden_shape[-1]), None
        )
        global_output = moe_layer.experts.forward_sglang_local_masked(
            global_hidden_states,
            global_probs,
            moe_layer.config.moe_router_topk,
            moe_layer.local_expert_indices,
        )
    else:
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


def _try_sglang_ordered_topk_route(
    moe_layer,
    global_hidden_states: torch.Tensor,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return SGLang-ordered top-k weights/ids for the simple Qwen3 MoE route.

    This is deliberately narrow.  When a config needs grouped routing, expert
    bias, token dropping, forced routing, or non-softmax scoring, the caller
    falls back to Megatron's router so those behaviors stay centralized.
    """
    router = moe_layer.router
    config = moe_layer.config
    if not hasattr(moe_layer.experts, "forward_sglang_local_masked_topk"):
        return None
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
    logits = router.gating(global_hidden_states.view(-1, 1, hidden_size)).view(
        -1, config.num_moe_experts
    )
    logits = router.apply_z_loss(logits, padding_mask=None)

    from sglang.srt.tp_invariant_ops import stable_topk_softmax

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
