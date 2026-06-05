# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Debug/test-only deterministic SUM collectives.

Replace order-sensitive SUM collectives with an all-gather plus a fixed local
fold so that different reduction topologies become bitwise-comparable. The
all-gather is pure data movement (no arithmetic), so the local summation order
is the only arithmetic order, making the result independent of the NCCL version,
topology, or padding. Never enable in production: it is slow.
"""

from typing import List

import torch

from megatron.core.tensor_parallel.mappings import _tree_reduce_sum_from_gathered

_DETERMINISTIC_COLLECTIVES: bool = False


def enable_deterministic_collectives() -> None:
    """Enable the deterministic SUM-collective fold globally."""
    global _DETERMINISTIC_COLLECTIVES
    _DETERMINISTIC_COLLECTIVES = True


def is_deterministic_collectives_enabled() -> bool:
    """Return whether the deterministic SUM-collective fold is enabled."""
    return _DETERMINISTIC_COLLECTIVES


def _is_power_of_two(value: int) -> bool:
    """Return whether the positive integer is a power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _fold_gathered_sum(gathered: List[torch.Tensor]) -> torch.Tensor:
    """Sum a per-rank gathered list in a fixed order (tree for power-of-two)."""
    world_size = len(gathered)
    if _is_power_of_two(world_size):
        stacked = torch.stack(gathered, dim=0)
        return _tree_reduce_sum_from_gathered(stacked)

    running = gathered[0].clone()
    for index in range(1, world_size):
        running = running + gathered[index]
    return running


def deterministic_sum_inplace(
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    *,
    chunk_numel: int = 64 * 1024 * 1024,
) -> None:
    """All-reduce SUM in-place via all-gather plus a fixed local fold (bitwise)."""
    assert tensor.is_contiguous(), "deterministic_sum_inplace requires a contiguous tensor"

    world_size = group.size()
    if world_size == 1:
        return

    flat = tensor.view(-1)
    total_numel = flat.numel()

    for start in range(0, total_numel, chunk_numel):
        end = min(start + chunk_numel, total_numel)
        chunk = flat[start:end].contiguous()
        gathered_list = [torch.empty_like(chunk) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_list, chunk, group=group)
        reduced = _fold_gathered_sum(gathered_list)
        flat[start:end].copy_(reduced)
