# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic EP combine for true-on-policy MoE.

Wraps SGLang's ``tree_all_reduce_sum`` so SGLang rollout and Megatron training
both perform the post-EP sum in the same fixed-tree order. Using this in place
of NCCL's ring all-reduce is what keeps the EP combine bit-exact across the
two engines.
"""

from __future__ import annotations

import torch


def sglang_moe_ep_tree_all_reduce(
    input_: torch.Tensor,
    ep_group,
) -> torch.Tensor:
    """Mirror SGLang's fixed-order EP sum for true-on-policy MoE combine."""
    from sglang.srt.tp_invariant_ops import tree_all_reduce_sum

    return tree_all_reduce_sum(input_, device_group=ep_group)
