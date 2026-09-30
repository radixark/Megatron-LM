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

    Both backends return values and indices; only FlashInfer's 2-D input and
    optional deterministic/tie-break arguments need adaptation. Preserve Miles'
    sorted Torch / unsorted FlashInfer ordering without perturbing scores.
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
    else:
        raise ValueError(f"Unsupported DSA top-k backend: {backend!r}")
    # The selector owns these indices. Mask in place after the dtype conversion
    # instead of allocating another output; reshape only the final result.
    return (
        indices.to(torch.int32)
        .masked_fill_(values == -torch.inf, -1)
        .reshape(*scores.shape[:-1], topk)
    )
