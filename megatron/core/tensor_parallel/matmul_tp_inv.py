# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import List, Optional

import torch

from .layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)


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
) -> torch.Tensor:
    """Reference TP matmul entrypoint for the SGLang-compatible backend.

    PR 6 keeps Megatron on the same local numerical path by default and introduces a
    single surface that later PRs can specialize for TP-invariant ordering. The
    implementation intentionally delegates to the existing Megatron kernels so enabling
    the backend flag does not yet change the training contract.
    """

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
