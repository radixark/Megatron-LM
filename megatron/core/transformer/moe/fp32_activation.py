# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu


ActivationFunc = Callable[[torch.Tensor], torch.Tensor]
ActivationContextFactory = Callable[[Any], Any]
ActivationForward = Callable[[torch.Tensor, Any], torch.Tensor]
ActivationBackward = Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]


@dataclass(frozen=True)
class MoEActivationInFP32Spec:
    """FP32 expert activation callbacks used by the recomputing autograd wrapper.

    ``context_factory`` binds activation-specific, non-tensor state from TransformerConfig.
    ``forward`` receives the FP32 FC1 output and bound context and returns the unweighted
    FP32 activation. ``backward`` receives the FP32 upstream gradient, FP32 FC1 output,
    and bound context and returns the FP32 FC1-output gradient. The wrapper owns router-
    probability multiplication, its gradient, and casts at the expert GEMM boundaries.
    """

    name: str
    context_factory: ActivationContextFactory
    forward: ActivationForward
    backward: ActivationBackward


_MOE_ACTIVATIONS_IN_FP32: Dict[Tuple[ActivationFunc, bool], MoEActivationInFP32Spec] = {}


def register_moe_activation_in_fp32(
    activation_func: ActivationFunc,
    gated_linear_unit: bool,
    spec: MoEActivationInFP32Spec,
) -> None:
    """Register an activation/gating pair for ``moe_activation_in_fp32``."""
    key = (activation_func, gated_linear_unit)
    if key in _MOE_ACTIVATIONS_IN_FP32:
        raise ValueError(f"FP32 MoE activation already registered: {spec.name}")
    _MOE_ACTIVATIONS_IN_FP32[key] = spec


def get_moe_activation_in_fp32_spec(
    activation_func: ActivationFunc, gated_linear_unit: bool
) -> Optional[MoEActivationInFP32Spec]:
    """Return the registered FP32 expert activation spec, if any."""
    return _MOE_ACTIVATIONS_IN_FP32.get((activation_func, gated_linear_unit))


def is_moe_activation_in_fp32_supported(
    activation_func: ActivationFunc, gated_linear_unit: bool
) -> bool:
    """Whether an activation/gating pair has an FP32 expert implementation."""
    return get_moe_activation_in_fp32_spec(activation_func, gated_linear_unit) is not None


def _swiglu_context(config: Any) -> float:
    return config.glu_linear_offset


def _swiglu_forward(fc1_out: torch.Tensor, glu_offset: Any) -> torch.Tensor:
    gate, linear = torch.chunk(fc1_out, 2, dim=-1)
    return F.silu(gate) * (linear + glu_offset)


def _swiglu_backward(
    grad_output: torch.Tensor, fc1_out: torch.Tensor, glu_offset: Any
) -> torch.Tensor:
    gate, linear = torch.chunk(fc1_out, 2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    gate_grad = grad_output * (linear + glu_offset) * (
        sigmoid + silu * (1 - sigmoid)
    )
    linear_grad = grad_output * silu
    return torch.cat([gate_grad, linear_grad], dim=-1)


def _no_activation_context(_config: Any) -> None:
    return None


def _squared_relu_forward(fc1_out: torch.Tensor, _context: Any) -> torch.Tensor:
    return F.relu(fc1_out).square()


def _squared_relu_backward(
    grad_output: torch.Tensor, fc1_out: torch.Tensor, _context: Any
) -> torch.Tensor:
    return grad_output * 2 * F.relu(fc1_out)


register_moe_activation_in_fp32(
    F.silu,
    True,
    MoEActivationInFP32Spec(
        name="swiglu",
        context_factory=_swiglu_context,
        forward=_swiglu_forward,
        backward=_swiglu_backward,
    ),
)
register_moe_activation_in_fp32(
    squared_relu,
    False,
    MoEActivationInFP32Spec(
        name="squared_relu",
        context_factory=_no_activation_context,
        forward=_squared_relu_forward,
        backward=_squared_relu_backward,
    ),
)
