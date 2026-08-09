from __future__ import annotations

from typing import List, Optional

import torch

from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)

from . import kernels


def _sglang_row_parallel_matmul(
    input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """SGLang row-linear local GEMM, delegated through the shared kernel seam.

    Routes to ``kernels.tp_invariant_row_linear`` -> SGLang's ``matmul_tp_inv`` (a
    two-level-tree TP-invariant matmul), so this rank's partial is bitwise-identical to
    inference; the subsequent TP all-reduce (a no-op at TP=1) then combines identical
    partials.
    """
    input_shape = input_.shape
    input_2d = input_.reshape(-1, input_shape[-1])
    output = kernels.tp_invariant_row_linear(input_2d, weight)
    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], weight.shape[0])


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
    """Matmul entry point for the SGLang-compatible (true-on-policy) backend -- a router.

    Row-parallel linears (o_proj, down_proj) reduce partials across TP ranks, so their
    reduction order must match SGLang's inference GEMM bitwise. They ALWAYS route through
    ``matmul_tp_inv`` (``_sglang_row_parallel_matmul``); SGLang does the same
    unconditionally. Because ``matmul_tp_inv`` is TP-degree-invariant, both engines then
    agree for every (train_tp, rollout_tp) -- including TP=1/TP=1, where the tree is
    marginally slower than a single GEMM but never mismatches. Using it always (rather
    than gating on tp) makes divergence impossible and needs no cross-engine tp signal.

    Column-parallel linears and frozen weights have no cross-rank K-reduction and fall
    through to the stock Megatron kernels, numerically unchanged.
    """
    if input_.dtype != weight.dtype:
        input_ = input_.to(weight.dtype)
    if bias is not None and bias.dtype != weight.dtype:
        bias = bias.to(weight.dtype)

    if row_parallel:
        return _sglang_row_parallel_matmul(input_, weight, bias)

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
