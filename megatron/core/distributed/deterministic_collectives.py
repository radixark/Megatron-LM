# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Debug/test-only deterministic SUM collectives.

Replace order-sensitive SUM collectives with an all-gather plus a fixed local
fold so that different reduction topologies become bitwise-comparable. The
all-gather is pure data movement (no arithmetic), so the local summation order
is the only arithmetic order, making the result independent of the NCCL version,
topology, or padding. Never enable in production: it is slow.
"""

import logging
from typing import Callable, List, Set

import torch

from megatron.core.tensor_parallel.mappings import _is_power_of_two, _tree_reduce_sum_from_gathered

logger = logging.getLogger(__name__)

_DETERMINISTIC_COLLECTIVES: bool = False

_LOGGED_GROUP_RANKS_TAGS: Set[str] = set()


def log_group_ranks_once(tag: str, group: torch.distributed.ProcessGroup) -> None:
    """Log ``tag`` plus the group's member ranks at INFO level exactly once per tag."""
    if tag in _LOGGED_GROUP_RANKS_TAGS:
        return
    _LOGGED_GROUP_RANKS_TAGS.add(tag)
    logger.info("%s ranks=%s", tag, torch.distributed.get_process_group_ranks(group))


def enable_deterministic_collectives() -> None:
    """Enable the deterministic SUM-collective fold globally."""
    global _DETERMINISTIC_COLLECTIVES
    _DETERMINISTIC_COLLECTIVES = True


def is_deterministic_collectives_enabled() -> bool:
    """Return whether the deterministic SUM-collective fold is enabled."""
    return _DETERMINISTIC_COLLECTIVES


def deterministic_sum_inplace(
    tensor: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    *,
    chunk_numel: int = 64 * 1024 * 1024,
) -> None:
    """All-reduce SUM in-place via all-gather plus a fixed local fold (bitwise)."""

    def _all_gather(gathered_list: List[torch.Tensor], chunk: torch.Tensor) -> None:
        torch.distributed.all_gather(gathered_list, chunk, group=group)

    deterministic_sum_inplace_with_gather(
        tensor, world_size=group.size(), all_gather_fn=_all_gather, chunk_numel=chunk_numel
    )


def deterministic_sum_inplace_with_gather(
    tensor: torch.Tensor,
    *,
    world_size: int,
    all_gather_fn: Callable[[List[torch.Tensor], torch.Tensor], None],
    chunk_numel: int = 64 * 1024 * 1024,
) -> None:
    """SUM in-place with an injectable all-gather (for non-c10d process groups).

    ``all_gather_fn(gathered_list, chunk)`` must fill ``gathered_list`` with every
    rank's ``chunk``; the local fold then defines the (fixed) summation order.
    """
    if not tensor.is_contiguous():
        work = tensor.contiguous()
        deterministic_sum_inplace_with_gather(
            work,
            world_size=world_size,
            all_gather_fn=all_gather_fn,
            chunk_numel=chunk_numel,
        )
        tensor.copy_(work)
        return

    if world_size == 1:
        return

    flat = tensor.view(-1)
    total_numel = flat.numel()

    for start in range(0, total_numel, chunk_numel):
        end = min(start + chunk_numel, total_numel)
        chunk = flat[start:end]
        gathered_list = [torch.empty_like(chunk) for _ in range(world_size)]
        all_gather_fn(gathered_list, chunk)
        chunk.copy_(fold_gathered_sum(gathered_list))


def fold_gathered_sum(gathered: List[torch.Tensor]) -> torch.Tensor:
    """Sum a per-rank gathered list in a fixed order (tree for power-of-two).

    Public: also used by miles indep_dp for the cross-cell deterministic
    reduction, so both sides share one bracketing. May reuse (mutate) the first
    gathered buffer as the accumulator.
    """
    world_size = len(gathered)
    if _is_power_of_two(world_size):
        stacked = torch.stack(gathered, dim=0)
        return _tree_reduce_sum_from_gathered(stacked)

    running = gathered[0]
    for index in range(1, world_size):
        running += gathered[index]
    return running
