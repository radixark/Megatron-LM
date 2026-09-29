# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit DSA top-k policies, independent of the score and attention kernels."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def uses_external_topk(config: "TransformerConfig") -> bool:
    """Whether the caller selected top-k separately from the fused DSA backend."""
    return getattr(config, "dsa_indexer_topk_backend", None) is not None


def select_dsa_topk(scores: torch.Tensor, topk: int, config: "TransformerConfig") -> torch.Tensor:
    """Select indices with the requested policy and mark masked entries as -1.

    FlashInfer consumes flattened rows and preserves its unsorted result. Do not
    perturb the scores to resolve ties: its tie_break argument defines which
    original indices survive ties at the selection boundary.
    """
    topk = min(topk, scores.size(-1))
    if topk == 0:
        return torch.empty((*scores.shape[:-1], 0), dtype=torch.int32, device=scores.device)
    backend = config.dsa_indexer_topk_backend
    if backend == "torch":
        values, indices = torch.topk(scores, topk, dim=-1)
    elif backend == "flashinfer":
        # FlashInfer is optional unless explicitly selected.
        from flashinfer import top_k

        values, indices = top_k(
            scores.reshape(-1, scores.size(-1)),
            topk,
            sorted=False,
            deterministic=config.dsa_indexer_topk_deterministic,
            tie_break=config.dsa_indexer_topk_tie_break,
            dsa_graph_safe=True,
        )
        values = values.reshape(*scores.shape[:-1], topk)
        indices = indices.reshape(*scores.shape[:-1], topk)
    else:
        raise ValueError(f"Unsupported DSA top-k backend: {backend!r}")
    return indices.to(torch.int32).masked_fill(values == -torch.inf, -1)
