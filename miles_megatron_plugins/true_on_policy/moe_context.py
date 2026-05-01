# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SGLang MoE rollout-segment context.

Plumbs per-sample DP partitioning information from the rollout into the
gradient-bearing Megatron MoE forward so the local-masked EP path can replay
the same per-source-rank token slices instead of the always-correct but
slower global-padded fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SGLangMoERolloutContext:
    sample_indices: tuple[int, ...] | None = None
    rollout_dp_ranks: tuple[int, ...] | None = None
    token_counts: tuple[int, ...] | None = None


_sglang_moe_rollout_context: SGLangMoERolloutContext | None = None


def _normalize_optional_ints(values) -> tuple[int, ...] | None:
    if values is None:
        return None
    if torch.is_tensor(values):
        raw_values = values.detach().cpu().flatten().tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_values = list(values)
    else:
        raw_values = [values]
    if not raw_values:
        return None
    return tuple(-1 if value is None else int(value) for value in raw_values)


@contextmanager
def sglang_moe_rollout_context(
    *,
    sample_indices=None,
    rollout_dp_ranks=None,
    token_counts=None,
):
    global _sglang_moe_rollout_context
    old_context = _sglang_moe_rollout_context
    _sglang_moe_rollout_context = SGLangMoERolloutContext(
        sample_indices=_normalize_optional_ints(sample_indices),
        rollout_dp_ranks=_normalize_optional_ints(rollout_dp_ranks),
        token_counts=_normalize_optional_ints(token_counts),
    )
    try:
        yield
    finally:
        _sglang_moe_rollout_context = old_context


def get_sglang_moe_rollout_context() -> SGLangMoERolloutContext | None:
    return _sglang_moe_rollout_context
