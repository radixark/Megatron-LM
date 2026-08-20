# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch

from megatron.core.fusions.fast_activations import (
    flashinfer_fast_swiglu,
    flashinfer_fast_swiglu_back,
    use_fast_activations,
)
from megatron.core.fusions.fused_bias_swiglu import megatron_swiglu, swiglu

_LOG2_E = 1.4426950408889634


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_flashinfer_fast_swiglu(input_dtype):
    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    grad_output = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    output = flashinfer_fast_swiglu(x)
    input_grad = flashinfer_fast_swiglu_back(grad_output, x)

    gate, up = torch.chunk(x.float(), 2, dim=-1)
    sigmoid = torch.reciprocal(1.0 + torch.exp2(-gate * _LOG2_E))
    output_ref = ((sigmoid * gate) * up).to(input_dtype)
    gate_grad_ref = grad_output.float() * sigmoid * (1 + gate * (1 - sigmoid)) * up
    up_grad_ref = grad_output.float() * (sigmoid * gate)
    input_grad_ref = torch.cat((gate_grad_ref, up_grad_ref), dim=-1).to(input_dtype)

    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-5, atol=1.0e-6)
    else:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)

    assert output.dtype == input_dtype
    assert input_grad.dtype == input_dtype
    assert torch.allclose(output, output_ref, **tols)
    assert torch.allclose(input_grad, input_grad_ref, **tols)


def test_fast_activation_environment_selection():
    env_enabled = os.environ.get("MILES_USE_FAST_ACTIVATIONS") == "1"
    expected = flashinfer_fast_swiglu if env_enabled else megatron_swiglu
    assert use_fast_activations() is env_enabled
    assert swiglu is expected
