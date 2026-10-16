# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions import fused_bias_geglu
from megatron.core.fusions.fused_bias_geglu import quick_gelu, weighted_bias_quick_geglu_impl

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference(x, bias, weights, linear_offset, clamp_value):
    if clamp_value is not None:
        y_1, y_2 = torch.chunk(x, 2, -1)
        x = torch.cat((y_1.clamp(max=clamp_value), y_2.clamp(-clamp_value, clamp_value)), -1)
    if bias is not None:
        x = x + bias
    y_1, y_2 = torch.chunk(x, 2, -1)
    return quick_gelu(y_1) * (y_2 + linear_offset) * weights


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(16, 64), (4, 4, 64)])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("clamp_value", [None, 2.0])
def test_weighted_bias_quick_geglu_matches_reference(device, shape, use_bias, clamp_value):
    torch.manual_seed(0)
    linear_offset = 1.0
    tokens = shape[0] * shape[1] if len(shape) == 3 else shape[0]
    x = (3 * torch.randn(*shape, device=device)).requires_grad_()
    weights = torch.randn(tokens, 1, device=device, requires_grad=True)
    bias = torch.randn(shape[-1], device=device) if use_bias else None

    out = weighted_bias_quick_geglu_impl(
        x, bias, weights, linear_offset=linear_offset, clamp_value=clamp_value
    )
    x_ref = x.detach().clone().requires_grad_()
    weights_ref = weights.detach().clone().requires_grad_()
    expected = _reference(
        x_ref.view(tokens, shape[-1]), bias, weights_ref, linear_offset, clamp_value
    )

    assert out.shape == (*shape[:-1], shape[-1] // 2)
    torch.testing.assert_close(out.view(tokens, -1), expected)

    grad = torch.randn_like(expected)
    out.view(tokens, -1).backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weights.grad, weights_ref.grad)


@pytest.mark.parametrize("device", DEVICES)
def test_linear_offset_tensor_is_cached_per_value_dtype_and_device(device):
    first = fused_bias_geglu._linear_offset_tensor(1.0, torch.float32, torch.device(device))
    again = fused_bias_geglu._linear_offset_tensor(1.0, torch.float32, torch.device(device))
    other_value = fused_bias_geglu._linear_offset_tensor(0.5, torch.float32, torch.device(device))
    other_dtype = fused_bias_geglu._linear_offset_tensor(1.0, torch.bfloat16, torch.device(device))

    assert first is again
    assert first.item() == 1.0 and first.dtype == torch.float32 and first.ndim == 0
    assert other_value.item() == 0.5
    assert other_dtype.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_weighted_bias_quick_geglu_does_not_synchronize():
    x = torch.randn(16, 64, device="cuda", requires_grad=True)
    weights = torch.randn(16, 1, device="cuda", requires_grad=True)
    # The first call may create the cached offset tensor; later calls must not
    # copy it from the host again.
    weighted_bias_quick_geglu_impl(x, None, weights, linear_offset=1.0)

    torch.cuda.set_sync_debug_mode("error")
    try:
        out = weighted_bias_quick_geglu_impl(x, None, weights, linear_offset=1.0)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    out.sum().backward()
