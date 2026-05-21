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

_MOE_DEBUG_DUMP_COUNTS: dict[tuple[int, int, str], int] = {}


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def requires_direct_sglang_moe(moe_layer) -> bool:
    """Return whether this layer must use the direct SGLang MoE path."""
    policy = resolve_true_on_policy_runtime_policy(moe_layer.config)
    return policy.ep_invariant_moe and policy.deterministic_moe_dispatch


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

    dispatcher = moe_layer.token_dispatcher
    if getattr(dispatcher, "drop_and_pad", False):
        raise RuntimeError("Qwen3 true-on-policy MoE does not support drop-and-pad dispatch")
    if getattr(dispatcher, "tp_size", 1) != 1:
        raise RuntimeError("Qwen3 true-on-policy MoE currently requires MoE TP size 1")
    if getattr(dispatcher, "ep_size", 1) <= 1:
        raise RuntimeError("Qwen3 true-on-policy MoE direct path requires EP size > 1")


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

def run_direct_sglang_ep_forward(
    moe_layer,
    hidden_states: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    intermediate_tensors,
) -> tuple:
    """Run the SGLang local-masked EP forward path.

    The caller should gate this with ``requires_direct_sglang_moe``.  This
    function raises when the required true-on-policy SGLang path is not wired
    correctly; it never falls through to Megatron MoE.  The grad-enabled path
    uses a PyTorch autograd wrapper around the SGLang forward and Triton
    backward kernels, without running a side Megatron MoE forward.
    """
    if not requires_direct_sglang_moe(moe_layer):
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


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return -1


def _tensor_stats(tensor: torch.Tensor) -> dict:
    detached = tensor.detach()
    flat = detached.reshape(-1)
    finite = torch.isfinite(flat)
    stats = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(flat.numel()),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(flat).sum().item()),
        "inf": int(torch.isinf(flat).sum().item()),
    }
    if stats["finite"]:
        finite_values = flat[finite].float()
        stats.update(
            {
                "mean": float(finite_values.mean().item()),
                "max_abs": float(finite_values.abs().max().item()),
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
            }
        )
    return stats


def _selected_debug_layer(layer_number: int) -> bool:
    raw_layers = os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_LAYERS")
    if not raw_layers:
        return True
    return str(layer_number) in {layer.strip() for layer in raw_layers.split(",")}


def _maybe_dump_moe_debug(
    moe_layer,
    name: str,
    tensor: torch.Tensor,
    *,
    extra: Optional[dict] = None,
    force: bool = False,
) -> None:
    debug_dir = os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_DIR")
    if not debug_dir or not torch.is_tensor(tensor):
        return

    layer_number = int(getattr(moe_layer, "layer_number", -1))
    if not force and not _selected_debug_layer(layer_number):
        return

    rank = _rank()
    key = (rank, layer_number, name)
    count = _MOE_DEBUG_DUMP_COUNTS.get(key, 0)
    max_per_name = int(os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_MAX_PER_NAME", "2"))
    if count >= max_per_name:
        return
    _MOE_DEBUG_DUMP_COUNTS[key] = count + 1

    os.makedirs(debug_dir, exist_ok=True)
    payload = {
        "rank": rank,
        "layer_number": layer_number,
        "name": name,
        "stats": _tensor_stats(tensor),
        "extra": extra or {},
    }
    if os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_TENSORS", "0") == "1":
        payload["tensor"] = tensor.detach().cpu()

    path = os.path.join(
        debug_dir,
        f"rank_{rank}_layer_{layer_number:02d}_{name}_{count:05d}_pid_{os.getpid()}.pt",
    )
    torch.save(payload, path)
    print(
        f"[MILES_TRUE_ON_POLICY_MOE_DEBUG] wrote {name} layer={layer_number} "
        f"rank={rank} stats={payload['stats']} path={path}",
        flush=True,
    )


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
    _maybe_dump_moe_debug(
        moe_layer,
        "nograd_global_hidden",
        global_hidden_states,
        extra={"max_num_tokens": max_num_tokens, "token_counts": list(token_counts)},
    )
    topk_route = _try_sglang_ordered_topk_route(
        moe_layer, global_hidden_states, hidden_shape[-1]
    )
    if topk_route is None:
        return None

    topk_weights, global_topk_ids = topk_route
    _maybe_dump_moe_debug(moe_layer, "nograd_topk_weights", topk_weights)
    _maybe_dump_moe_debug(moe_layer, "nograd_topk_ids", global_topk_ids)
    global_output = moe_layer.experts.forward_sglang_local_masked_topk(
        global_hidden_states,
        topk_weights,
        global_topk_ids,
        moe_layer.local_expert_indices,
    )
    _maybe_dump_moe_debug(moe_layer, "nograd_local_expert_output", global_output)
    global_output = sglang_moe_ep_tree_all_reduce(global_output, ep_group)
    _maybe_dump_moe_debug(moe_layer, "nograd_ep_reduced_output", global_output)

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
    _maybe_dump_moe_debug(
        moe_layer,
        "autograd_global_hidden",
        global_hidden_states,
        extra={"max_num_tokens": max_num_tokens, "token_counts": list(token_counts)},
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
    _maybe_dump_moe_debug(moe_layer, "autograd_topk_weights", topk_weights)
    _maybe_dump_moe_debug(moe_layer, "autograd_topk_ids", global_topk_ids)
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
    _maybe_dump_moe_debug(moe_layer, "autograd_local_expert_output", global_output)
    global_output = _SGLangEPAllReduceSum.apply(global_output, ep_group)
    _maybe_dump_moe_debug(moe_layer, "autograd_ep_reduced_output", global_output)
    _maybe_compare_sglang_autograd_to_exact(
        moe_layer,
        global_hidden_states,
        topk_weights,
        global_topk_ids,
        global_output,
        ep_group,
    )

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


def _maybe_compare_sglang_autograd_to_exact(
    moe_layer,
    global_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    global_topk_ids: torch.Tensor,
    autograd_reduced_output: torch.Tensor,
    ep_group,
) -> None:
    if os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_COMPARE_EXACT", "0") != "1":
        return

    with torch.no_grad():
        exact_output = moe_layer.experts.forward_sglang_local_masked_topk(
            global_hidden_states.detach(),
            topk_weights.detach(),
            global_topk_ids.detach(),
            moe_layer.local_expert_indices,
        )
        exact_output = sglang_moe_ep_tree_all_reduce(exact_output, ep_group)

    diff = (autograd_reduced_output.detach().float() - exact_output.float()).abs()
    max_diff = float(diff.max().item()) if diff.numel() else 0.0
    mean_diff = float(diff.mean().item()) if diff.numel() else 0.0
    nonzero = int((diff != 0).sum().item()) if diff.numel() else 0
    extra = {"max_diff": max_diff, "mean_diff": mean_diff, "nonzero": nonzero}

    should_dump = (
        os.environ.get("MILES_TRUE_ON_POLICY_MOE_DEBUG_COMPARE_DUMP_ALL", "0") == "1"
        or max_diff != 0.0
    )
    if should_dump:
        _maybe_dump_moe_debug(
            moe_layer,
            "exact_ep_reduced_output",
            exact_output,
            extra=extra,
            force=True,
        )
        _maybe_dump_moe_debug(
            moe_layer,
            "autograd_exact_abs_diff",
            diff,
            extra=extra,
            force=True,
        )

    print(
        "[MILES_TRUE_ON_POLICY_MOE_DEBUG] "
        f"layer={getattr(moe_layer, 'layer_number', -1)} rank={_rank()} "
        f"autograd_vs_exact max_diff={max_diff} mean_diff={mean_diff} nonzero={nonzero}",
        flush=True,
    )


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
    debug_prefix = "autograd" if torch.is_grad_enabled() else "nograd"
    _maybe_dump_moe_debug(moe_layer, f"{debug_prefix}_router_logits", logits)

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
