from __future__ import annotations

from typing import Optional

import torch

from .contracts import sglang_residual_pair_enabled


_RESIDUAL_ATTR = "_sglang_residual"


def get_sglang_residual(hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    return getattr(hidden_states, _RESIDUAL_ATTR, None)


def attach_sglang_residual(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> torch.Tensor:
    if residual is None:
        if hasattr(hidden_states, _RESIDUAL_ATTR):
            delattr(hidden_states, _RESIDUAL_ATTR)
    else:
        setattr(hidden_states, _RESIDUAL_ATTR, residual)
    return hidden_states


def is_sglang_residual_pair_output(value) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    hidden_states, residual = value
    return (
        isinstance(hidden_states, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and get_sglang_residual(hidden_states) is residual
    )


def unpack_sglang_pipeline_input(input_tensor, *, owner: str = "TransformerBlock"):
    if isinstance(input_tensor, list):
        assert len(input_tensor) in (1, 2), (
            f"{owner} input_tensor should be length 1, or length 2 for the "
            "true-on-policy residual-pair pipeline carrier."
        )
        residual = input_tensor[1] if len(input_tensor) == 2 else None
        return input_tensor[0], residual
    return input_tensor, None


def pack_sglang_pipeline_input(hidden_states: torch.Tensor, residual: Optional[torch.Tensor]):
    if residual is None:
        return hidden_states
    return [attach_sglang_residual(hidden_states, residual), residual]


def pack_sglang_pipeline_output(config, hidden_states: torch.Tensor):
    if not sglang_residual_pair_enabled(config):
        return hidden_states

    residual = get_sglang_residual(hidden_states)
    if residual is None:
        return hidden_states
    return [hidden_states, residual]


def append_sglang_residual_shape_if_needed(
    config,
    tensor_shapes: list,
    tensor_shape,
    *,
    pipeline_model_parallel_world_size: int,
) -> None:
    if sglang_residual_pair_enabled(config) and pipeline_model_parallel_world_size > 1:
        tensor_shapes.append(tensor_shape)
