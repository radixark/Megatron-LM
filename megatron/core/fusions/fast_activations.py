# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in activation arithmetic aligned with FlashInfer rollout kernels."""

import os

import torch

from megatron.core.jit import jit_fuser

_FAST_ACTIVATIONS_ENV = "MILES_USE_FAST_ACTIVATIONS"
_LOG2_E = 1.4426950408889634
_USE_FAST_ACTIVATIONS = os.environ.get(_FAST_ACTIVATIONS_ENV) == "1"


def use_fast_activations():
    """Return whether fast activations were enabled before Megatron import."""
    return _USE_FAST_ACTIVATIONS


def _flashinfer_fast_sigmoid(y):
    """Mirror the fast sigmoid arithmetic used by FlashInfer's SM100 CuTe MoE."""
    return torch.reciprocal(1.0 + torch.exp2(-y * _LOG2_E))


@jit_fuser
def flashinfer_fast_swiglu(y):
    """Perform SwiGLU with FlashInfer's fast FP32 activation operation order."""
    dtype = y.dtype
    gate, up = torch.chunk(y, 2, -1)
    gate = gate.float()
    up = up.float()
    sigmoid = _flashinfer_fast_sigmoid(gate)
    return ((sigmoid * gate) * up).to(dtype)


@jit_fuser
def flashinfer_fast_swiglu_back(grad, y):
    """Use the fast sigmoid value in Megatron's analytic SwiGLU derivative."""
    dtype = y.dtype
    gate, up = torch.chunk(y, 2, -1)
    grad = grad.float()
    gate = gate.float()
    up = up.float()
    sigmoid = _flashinfer_fast_sigmoid(gate)
    gate_grad = grad * sigmoid * (1 + gate * (1 - sigmoid)) * up
    up_grad = grad * (sigmoid * gate)
    return torch.cat((gate_grad, up_grad), -1).to(dtype)
