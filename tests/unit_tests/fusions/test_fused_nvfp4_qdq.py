# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact tests for the fused CuTe DSL NVFP4 QDQ kernel.

The data patterns and Four Over Six Cartesian matrix mirror FlashInfer's
``tests/utils/test_fp4_quantize.py::test_nvfp4_quantize_te_reference``. The
oracle is Transformer Engine's native quantize-then-dequantize path because
FlashInfer's per-tensor path has a different numerical contract.
"""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("cutlass")
te = pytest.importorskip("transformer_engine.pytorch")

from megatron.core.fusions.fused_nvfp4_qdq import (  # noqa: E402
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    fake_nvfp4_quantization_ste,
    fused_nvfp4_qdq,
)


_recipe_available, _recipe_unavailable_reason = te.is_nvfp4_available(
    return_reason=True
)
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(not _recipe_available, reason=_recipe_unavailable_reason),
]


SHAPES = [
    # Minimum-K and odd-row cases absent from FlashInfer's swizzled-layout matrix.
    (1, 16),
    (1, 32),
    (3, 48),
    # FlashInfer strict-test shapes and both TE BF16 dispatch routes after M padding.
    (1, 64),
    (3, 128),
    (16, 64),
    (31, 128),
    (32, 128),
    (128, 64),
    (128, 1024),
    (256, 256),
    (1024, 2048),
]


CONFIGS = [
    pytest.param(NVFP4QDQConfig(), id="nvfp4"),
]
for _error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
    for _e4m3_max in (448, 256):
        for _error_use_fp16 in (False, True):
            CONFIGS.append(
                pytest.param(
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=_e4m3_max,
                        error_mode=_error_mode,
                        error_use_fp16=_error_use_fp16,
                    ),
                    id=(
                        f"4over6-{_error_mode.name.lower()}-e4m3-{_e4m3_max}-"
                        f"{'fp16-error' if _error_use_fp16 else 'exact-error'}"
                    ),
                )
            )


def _make_input(
    shape: tuple[int, int], dtype: torch.dtype, init_data: str
) -> torch.Tensor:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m, n = shape
    if init_data == "random":
        x = torch.randn(shape, dtype=dtype, device="cuda")
        if m > 1:
            x[0].zero_()
        return x
    if init_data == "boundary":
        base = torch.linspace(
            -12.0, 12.0, steps=n // 2, dtype=torch.float32, device="cuda"
        )
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, torch.full_like(base, 1e-4))
        row = torch.empty(n, dtype=torch.float32, device="cuda")
        row[0::2] = base - eps
        row[1::2] = base + eps
        return row.unsqueeze(0).repeat(m, 1).to(dtype=dtype)
    if init_data == "zeros":
        return torch.zeros(shape, dtype=dtype, device="cuda")
    if init_data == "maxes":
        return torch.full(shape, torch.finfo(dtype).max, dtype=dtype, device="cuda")
    raise ValueError(f"Unknown init_data: {init_data}")


def _make_te_quantizer(config: NVFP4QDQConfig):
    return te.NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=config.use_4over6,
        nvfp4_e4m3_max=config.e4m3_max,
        nvfp4_4over6_err_mode=config.error_mode.name,
        with_random_sign_mask=False,
    )


def _te_reference(
    x: torch.Tensor, config: NVFP4QDQConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    padded_m = ((m + 15) // 16) * 16
    if padded_m == m:
        x_padded = x.contiguous()
    else:
        padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x.contiguous(), padding), dim=0)

    quantized = _make_te_quantizer(config).quantize(x_padded)
    reference = quantized.dequantize(dtype=x.dtype)[:m, :n].contiguous()
    assert quantized._amax_rowwise is not None
    return reference, quantized._amax_rowwise.reshape(1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}")
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", CONFIGS)
@torch.inference_mode()
def test_fused_nvfp4_qdq_is_bit_exact_with_te(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    """Cover BF16/FP16 x shapes x data patterns x the full supported feature matrix."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv(
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fp16 else "0"
    )
    x = _make_input(shape, dtype, init_data)
    amax = compute_nvfp4_amax(x)
    expected, te_amax = _te_reference(x, config)
    actual = fused_nvfp4_qdq(x, amax, config)

    assert torch.equal(amax.reshape(1).view(torch.int32), te_amax.view(torch.int32))
    # Integer views distinguish signed zero; tolerance-zero floating comparison does not.
    actual_bits = actual.view(torch.uint16)
    expected_bits = expected.view(torch.uint16)
    assert torch.equal(actual_bits, expected_bits), (
        f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fused_nvfp4_qdq_uses_straight_through_gradient_and_preserves_main_grad() -> (
    None
):
    x = torch.randn((3, 32), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    main_grad = torch.empty_like(x)
    x.main_grad = main_grad
    output = fake_nvfp4_quantization_ste(x, NVFP4QDQConfig())
    output.backward(torch.ones_like(output))

    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0.0, atol=0.0)
    assert output.main_grad is main_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_nvfp4_qdq_rejects_unsupported_input_dtype(dtype: torch.dtype) -> None:
    x = torch.randn((2, 16), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="supports BF16 and FP16"):
        fused_nvfp4_qdq(x, x.abs().amax().float(), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_non_block_aligned_k() -> None:
    x = torch.randn((2, 17), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="K divisible by 16"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())
