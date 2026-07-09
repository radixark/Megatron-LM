from __future__ import annotations

from typing import Iterable, List, Optional

import torch

from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)

_ROW_LINEAR_INV_BLOCK_K = 128


def _fixed_tree_sum_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum tensors with a fixed pairwise binary tree.

    Matches SGLang's `tree_all_reduce_sum`, which reduces across a power-of-two
    number of TP ranks.
    """
    partials = list(tensors)
    if not partials:
        raise ValueError("at least one tensor is required")

    while len(partials) > 1:
        next_partials = []
        for index in range(0, len(partials), 2):
            if index + 1 < len(partials):
                next_partials.append(partials[index] + partials[index + 1])
            else:
                next_partials.append(partials[index])
        partials = next_partials

    return partials[0]


def _sglang_first_level_block(num_partials: int) -> int:
    """FIRST_LEVEL_BLOCK as derived by SGLang's `_matmul_tp_persistent_impl`.

    The kernel halves the k-tile count while it stays even and above two, so the
    first accumulator level spans `num_partials / 2**(LEVEL_K - 1)` tiles and the
    remaining levels are binary. When `num_partials` is a power of two this
    degenerates to 2, i.e. a plain binary tree.
    """
    first_level_block = num_partials
    while first_level_block > 2 and first_level_block % 2 == 0:
        first_level_block //= 2
    return first_level_block


def _sglang_kernel_order_sum(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Sum k-tile partials in SGLang's `matmul_tp_inv` accumulation order.

    SGLang accumulates FIRST_LEVEL_BLOCK partials sequentially into one register
    accumulator before carrying into the binary levels above it. A plain binary
    tree over all partials is only equivalent when FIRST_LEVEL_BLOCK == 2, so it
    diverges bitwise for any k-tile count with an odd factor (e.g. T=38 for
    Qwen3-4B `down_proj` at TP=2).
    """
    partials = list(tensors)
    if not partials:
        raise ValueError("at least one tensor is required")

    first_level_block = _sglang_first_level_block(len(partials))

    blocks: List[torch.Tensor] = []
    for start in range(0, len(partials), first_level_block):
        group = partials[start : start + first_level_block]
        accumulator = group[0]
        for partial in group[1:]:
            accumulator = accumulator + partial
        blocks.append(accumulator)

    return _fixed_tree_sum_tensors(blocks)


def _safe_group_size(group: Optional[torch.distributed.ProcessGroup]) -> int:
    if group is not None:
        return group.size()
    try:
        from megatron.core.parallel_state import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _safe_tensor_context_parallel_size() -> int:
    try:
        from megatron.core.parallel_state import get_tensor_and_context_parallel_world_size

        return get_tensor_and_context_parallel_world_size()
    except Exception:
        return _safe_group_size(None)


def _rollout_row_parallel_partition_k(
    input_: torch.Tensor, tp_group: Optional[torch.distributed.ProcessGroup]
) -> int:
    train_tp_size = _safe_group_size(tp_group)
    rollout_tp_size = _safe_tensor_context_parallel_size()
    global_k_size = input_.shape[-1] * train_tp_size
    if rollout_tp_size <= 0 or global_k_size % rollout_tp_size != 0:
        return input_.shape[-1]
    return global_k_size // rollout_tp_size


def _should_use_sglang_tp_invariant_row_linear(
    input_: torch.Tensor, row_parallel: bool, tp_group: Optional[torch.distributed.ProcessGroup]
) -> bool:
    rollout_partition_k = _rollout_row_parallel_partition_k(input_, tp_group)
    return (
        row_parallel
        and rollout_partition_k >= _ROW_LINEAR_INV_BLOCK_K
        and rollout_partition_k % _ROW_LINEAR_INV_BLOCK_K == 0
    )


def _sglang_row_parallel_matmul(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """SGLang's row-linear TP-invariant matmul contract.

    SGLang chunks the K dimension into 128-wide products, casts each product to
    the input dtype, then combines those partials in its kernel's accumulation
    order. Mirroring that order is required before the TP tree all-reduce can be
    bitwise identical.
    """
    input_shape = input_.shape
    input_2d = input_.reshape(-1, input_shape[-1])
    weight_t = weight.t()
    partials = []

    for start in range(0, input_2d.shape[1], _ROW_LINEAR_INV_BLOCK_K):
        end = min(start + _ROW_LINEAR_INV_BLOCK_K, input_2d.shape[1])
        partials.append(input_2d[:, start:end] @ weight_t[start:end, :])

    output = _sglang_kernel_order_sum(partials).to(input_.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], weight.shape[0])


def _sglang_rollout_partition_row_parallel_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    tp_group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Mirror SGLang rollout row-linear shards when train TP is smaller than rollout TP."""
    rollout_partition_k = _rollout_row_parallel_partition_k(input_, tp_group)
    if (
        rollout_partition_k <= 0
        or rollout_partition_k >= input_.shape[-1]
        or input_.shape[-1] % rollout_partition_k != 0
    ):
        return _linear_reference_matmul(input_, weight, bias)

    input_shape = input_.shape
    input_2d = input_.reshape(-1, input_shape[-1])
    weight_t = weight.t()
    partials = []

    for start in range(0, input_2d.shape[1], rollout_partition_k):
        end = start + rollout_partition_k
        partials.append(input_2d[:, start:end] @ weight_t[start:end, :])

    output = _fixed_tree_sum_tensors(partials).to(input_.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], weight.shape[0])


def _linear_reference_matmul(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    output = input_.reshape(-1, input_.shape[-1]) @ weight.t()
    output = output.reshape(*input_.shape[:-1], weight.shape[0])
    if bias is not None:
        output = output + bias
    return output


def sglang_reference_matmul(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    row_parallel: bool = False,
) -> torch.Tensor:
    """Reference TP matmul entrypoint for the SGLang-compatible backend.

    PR 6 keeps Megatron on the same local numerical path by default and introduces a
    single surface that later PRs can specialize for TP-invariant ordering. The
    implementation intentionally delegates to the existing Megatron kernels so enabling
    the backend flag does not yet change the training contract.
    """

    if input_.dtype != weight.dtype:
        input_ = input_.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)

    if _should_use_sglang_tp_invariant_row_linear(input_, row_parallel, tp_group):
        return _sglang_row_parallel_matmul(input_, weight, bias)
    if row_parallel:
        return _sglang_rollout_partition_row_parallel_matmul(
            input_, weight, bias, tp_group=tp_group
        )

    if weight.requires_grad:
        return linear_with_grad_accumulation_and_async_allreduce(
            input=input_,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            grad_output_buffer=grad_output_buffer,
            wgrad_deferral_limit=wgrad_deferral_limit or 0,
            tp_group=tp_group,
        )

    return linear_with_frozen_weight(
        input=input_,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=sequence_parallel,
        grad_output_buffer=grad_output_buffer,
        wgrad_deferral_limit=wgrad_deferral_limit,
        tp_group=tp_group,
    )
