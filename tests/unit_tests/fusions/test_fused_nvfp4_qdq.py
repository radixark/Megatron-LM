# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact tests for the fused CuTe DSL NVFP4 QDQ kernel.

The data patterns and Four Over Six Cartesian matrix mirror FlashInfer's
``tests/utils/test_fp4_quantize.py::test_nvfp4_quantize_te_reference``. The
oracle follows Transformer Engine's strict
``tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py`` test and calls TE's native
quantize-then-dequantize path because FlashInfer's per-tensor path has a
different numerical contract.
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
    current_nvfp4_qdq_config,
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
GROUP_COUNTS = [1, 3, 8]


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
    group_count: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    init_data: str,
) -> torch.Tensor:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    m, n = shape
    if init_data == "random":
        x = torch.randn((group_count, m, n), dtype=torch.float32, device="cuda")
        if m > 1:
            x[:, 0].zero_()
        scales = torch.pow(
            2.0,
            (torch.arange(group_count, device="cuda") % 5) - 2,
        ).view(-1, 1, 1)
        return (x * scales).to(dtype)
    if init_data == "boundary":
        base = torch.linspace(
            -12.0, 12.0, steps=n // 2, dtype=torch.float32, device="cuda"
        )
        eps = torch.full_like(base, 1e-3)
        eps = torch.maximum(eps, torch.full_like(base, 1e-4))
        row = torch.empty(n, dtype=torch.float32, device="cuda")
        row[0::2] = base - eps
        row[1::2] = base + eps
        groups = []
        for group_idx in range(group_count):
            scale = 2.0 ** ((group_idx % 5) - 2)
            groups.append(torch.roll(row, shifts=2 * group_idx).repeat(m, 1) * scale)
        return torch.stack(groups).to(dtype=dtype)
    if init_data == "zeros":
        # Alternate signed zeros so the integer-view equality below exercises
        # TE's E2M1 sign-bit contract for zero-amax blocks.
        row = torch.tensor([-0.0, 0.0], dtype=torch.float32, device="cuda").repeat(
            m, n // 2
        )
        return torch.stack(
            [
                torch.roll(row, shifts=group_idx % 2, dims=1)
                for group_idx in range(group_count)
            ]
        ).to(dtype=dtype)
    if init_data == "maxes":
        x = torch.full(
            (group_count, m, n), torch.finfo(dtype).max, dtype=dtype, device="cuda"
        )
        signs = torch.where(
            torch.arange(group_count, device="cuda") % 2 == 0,
            1.0,
            -1.0,
        ).to(dtype)
        return x * signs.view(-1, 1, 1)
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
    _, m, n = x.shape
    references = []
    amaxes = []
    quantizer = _make_te_quantizer(config)
    for weight in x.unbind(0):
        padded_m = ((m + 15) // 16) * 16
        if padded_m == m:
            weight_padded = weight.contiguous()
        else:
            padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
            weight_padded = torch.cat((weight.contiguous(), padding), dim=0)

        quantized = quantizer.quantize(weight_padded)
        references.append(quantized.dequantize(dtype=x.dtype)[:m, :n].contiguous())
        assert quantized._amax_rowwise is not None
        amaxes.append(quantized._amax_rowwise.reshape(1))
    return torch.stack(references), torch.cat(amaxes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("group_count", GROUP_COUNTS, ids=lambda value: f"g{value}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: f"{shape[0]}x{shape[1]}")
@pytest.mark.parametrize("init_data", ["random", "boundary", "zeros", "maxes"])
@pytest.mark.parametrize("config", CONFIGS)
@torch.inference_mode()
def test_fused_nvfp4_qdq_is_bit_exact_with_te(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    group_count: int,
    shape: tuple[int, int],
    init_data: str,
    config: NVFP4QDQConfig,
) -> None:
    """Cover BF16/FP16 x shapes x data patterns x the full supported feature matrix."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv(
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if config.error_use_fp16 else "0"
    )
    x = _make_input(group_count, shape, dtype, init_data)
    amaxes = compute_nvfp4_amax(x)
    expected, te_amax = _te_reference(x, config)
    actual = fused_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amax.view(torch.int32))
    # Integer views distinguish signed zero; tolerance-zero floating comparison does not.
    actual_bits = actual.view(torch.uint16)
    expected_bits = expected.view(torch.uint16)
    assert torch.equal(actual_bits, expected_bits), (
        f"bit mismatch count: {torch.count_nonzero(actual_bits != expected_bits).item()}"
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("group_count", [1, 8], ids=lambda value: f"g{value}")
def test_fused_nvfp4_qdq_uses_straight_through_gradient_and_preserves_main_grad(
    group_count: int,
) -> None:
    x = torch.nn.Parameter(
        torch.randn((group_count, 3, 32), dtype=torch.bfloat16, device="cuda")
    )
    main_grad = torch.empty_like(x)
    x.main_grad = main_grad
    output = fake_nvfp4_quantization_ste(x, NVFP4QDQConfig())
    grad = (
        torch.arange(1, group_count + 1, dtype=torch.float32, device="cuda")
        .view(-1, 1, 1)
        .expand_as(output)
        .to(output.dtype)
    )
    output.backward(grad)

    assert tuple(output.shape) == tuple(x.shape)
    assert output.is_contiguous()
    torch.testing.assert_close(x.grad, grad, rtol=0.0, atol=0.0)
    assert output.main_grad is main_grad


@pytest.mark.parametrize(
    ("four_over_six_scope", "e4m3_256_scope", "expected_enabled", "expected_max"),
    [
        (
            four_over_six_scope,
            e4m3_256_scope,
            four_over_six_scope in ("weights", "all"),
            expected_max,
        )
        for four_over_six_scope in ("none", "activations", "weights", "all")
        for e4m3_256_scope in ("none", "activations", "weights", "all")
        for expected_max in [
            256
            if four_over_six_scope in ("weights", "all")
            and e4m3_256_scope in ("weights", "all")
            else 448
        ]
    ],
)
@pytest.mark.parametrize(
    ("error_mode", "error_use_fp16"),
    [("MAE", False), ("MAE", True), ("MSE", False), ("MSE", True)],
)
def test_current_nvfp4_qdq_config_maps_full_latest_te_env_contract(
    monkeypatch: pytest.MonkeyPatch,
    four_over_six_scope: str,
    e4m3_256_scope: str,
    expected_enabled: bool,
    expected_max: int,
    error_mode: str,
    error_use_fp16: bool,
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", four_over_six_scope)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", e4m3_256_scope)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", error_mode)
    monkeypatch.setenv(
        "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", "1" if error_use_fp16 else "0"
    )
    config = current_nvfp4_qdq_config()
    assert config.use_4over6 is expected_enabled
    assert config.e4m3_max == expected_max
    assert config.error_mode is NVFP4QDQErrorMode[error_mode]
    # The latest TE meaning is FP16-rounded candidate error, not a general
    # instruction-level fast-math toggle.
    assert config.error_use_fp16 is (expected_enabled and error_use_fp16)


@pytest.mark.parametrize("legacy_scope", ["inputs", "gradients"])
def test_current_nvfp4_qdq_config_rejects_stale_te_scopes(
    monkeypatch: pytest.MonkeyPatch, legacy_scope: str
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", legacy_scope)
    with pytest.raises(ValueError, match="activations"):
        current_nvfp4_qdq_config()


def test_current_nvfp4_qdq_config_rejects_quant_fast_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    with pytest.raises(ValueError, match="NVTE_USE_FAST_MATH=0"):
        current_nvfp4_qdq_config()


@pytest.mark.parametrize("group_count", [1, 3, 8])
def test_te_grouped_linear_nvfp4_flag_routes_packed_weight_once(
    monkeypatch: pytest.MonkeyPatch,
    group_count: int,
) -> None:
    from megatron.core.extensions import transformer_engine as te_extension
    from megatron.core.fusions import fused_nvfp4_qdq as qdq_module

    grouped_weight = torch.nn.Parameter(
        torch.randn((group_count, 2, 16), dtype=torch.bfloat16, device="cuda")
    )
    weights = list(grouped_weight.unbind(0))
    main_grads = [torch.empty_like(weight) for weight in weights]
    for weight, main_grad in zip(weights, main_grads):
        weight.main_grad = main_grad
    expected = torch.empty_like(grouped_weight)
    calls = []
    cached_config = NVFP4QDQConfig()

    monkeypatch.setattr(te.GroupedLinear, "_get_weight_tensors", lambda _self: weights)

    def fake_fused_qdq(value: torch.Tensor, config: NVFP4QDQConfig) -> torch.Tensor:
        calls.append((value, config))
        return expected

    monkeypatch.setattr(qdq_module, "fake_nvfp4_quantization_ste", fake_fused_qdq)
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    layer = te_extension.TEGroupedLinear.__new__(te_extension.TEGroupedLinear)
    torch.nn.Module.__init__(layer)
    layer._nvfp4_qat_config = cached_config
    layer.single_grouped_weight = True
    layer.weight = grouped_weight

    actual = te_extension.TEGroupedLinear._get_weight_tensors(layer)

    assert len(calls) == 1
    assert calls[0][0] is grouped_weight
    assert calls[0][1] is cached_config
    assert len(actual) == group_count
    assert all(value.is_contiguous() for value in actual)
    assert all(
        value.data_ptr() == expected[group_idx].data_ptr()
        for group_idx, value in enumerate(actual)
    )
    assert all(
        value.main_grad is main_grad for value, main_grad in zip(actual, main_grads)
    )


def test_te_grouped_linear_nvfp4_discrete_weights_use_singleton_3d_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.extensions import transformer_engine as te_extension
    from megatron.core.fusions import fused_nvfp4_qdq as qdq_module

    weights = [
        torch.randn((2, 16), dtype=torch.bfloat16, device="cuda") for _ in range(3)
    ]
    main_grads = [torch.empty_like(weight) for weight in weights]
    for weight, main_grad in zip(weights, main_grads):
        weight.main_grad = main_grad
    grouped_outputs = [torch.empty_like(weight.unsqueeze(0)) for weight in weights]
    calls = []
    cached_config = NVFP4QDQConfig()

    monkeypatch.setattr(te.GroupedLinear, "_get_weight_tensors", lambda _self: weights)

    def fake_fused_qdq(value: torch.Tensor, config: NVFP4QDQConfig) -> torch.Tensor:
        call_idx = len(calls)
        calls.append((value, config))
        return grouped_outputs[call_idx]

    monkeypatch.setattr(qdq_module, "fake_nvfp4_quantization_ste", fake_fused_qdq)
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    layer = te_extension.TEGroupedLinear.__new__(te_extension.TEGroupedLinear)
    layer._nvfp4_qat_config = cached_config
    layer.single_grouped_weight = False

    actual = te_extension.TEGroupedLinear._get_weight_tensors(layer)

    assert len(calls) == len(weights)
    assert all(tuple(value.shape) == (1, 2, 16) for value, _ in calls)
    assert all(config is cached_config for _, config in calls)
    assert all(
        value.data_ptr() == grouped_outputs[group_idx][0].data_ptr()
        for group_idx, value in enumerate(actual)
    )
    assert all(
        value.main_grad is main_grad for value, main_grad in zip(actual, main_grads)
    )


def test_te_grouped_linear_real_packed_weight_qdq_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.extensions import transformer_engine as te_extension

    if not te_extension.is_te_min_version("2.17.0"):
        pytest.skip("packed TE GroupedLinear weights require Transformer Engine 2.17+")

    group_count, rows, columns = 3, 2, 32
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    monkeypatch.setenv("OPEN_TRAINING_INT4_FAKE_QAT_FLAG", "0")
    monkeypatch.setenv("OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG", "1")
    layer = te_extension.TEGroupedLinear.__new__(te_extension.TEGroupedLinear)
    te.GroupedLinear.__init__(
        layer,
        num_gemms=group_count,
        in_features=columns,
        out_features=rows,
        bias=False,
        params_dtype=torch.bfloat16,
        single_grouped_weight=True,
        device="cuda",
    )
    layer.disable_parameter_transpose_cache = True
    layer.is_first_microbatch = True
    layer.te_quant_params = None
    layer.te_return_bias = False
    layer._nvfp4_qat_config = NVFP4QDQConfig()
    layer.weight.main_grad = torch.empty(
        (group_count, rows, columns), dtype=torch.bfloat16, device="cuda"
    )

    actual_views = te_extension.TEGroupedLinear._get_weight_tensors(layer)
    actual = torch.stack(actual_views)
    packed_input = layer.weight.rowwise_data.view(group_count, rows, columns)
    expected, _ = _te_reference(packed_input, layer._nvfp4_qat_config)

    assert len(actual_views) == group_count
    assert all(weight.requires_grad for weight in actual_views)
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    assert all(
        weight.main_grad.data_ptr() == layer.weight.main_grad[group_idx].data_ptr()
        for group_idx, weight in enumerate(actual_views)
    )

    m_splits = [2, 1, 3]
    inp = torch.randn(
        (sum(m_splits), columns),
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    output, bias = layer(inp, m_splits)
    output.backward(torch.ones_like(output))

    assert bias is None
    assert tuple(output.shape) == (sum(m_splits), rows)
    assert inp.grad is not None and torch.isfinite(inp.grad).all()
    assert layer.weight.grad is not None
    assert tuple(layer.weight.grad.shape) == (group_count, rows, columns)
    assert all(
        torch.count_nonzero(layer.weight.grad[group_idx]).item() > 0
        for group_idx in range(group_count)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_nvfp4_qdq_rejects_unsupported_input_dtype(dtype: torch.dtype) -> None:
    x = torch.randn((1, 2, 16), dtype=dtype, device="cuda")
    with pytest.raises(TypeError, match="supports BF16 and FP16"):
        fused_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


def test_fused_nvfp4_qdq_rejects_rank_2_input() -> None:
    x = torch.randn((2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="rank-3"):
        fused_nvfp4_qdq(x, torch.ones(1, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("group_count", [0, 2049])
def test_fused_nvfp4_qdq_rejects_group_count_outside_bound(
    group_count: int,
) -> None:
    x = torch.empty((group_count, 1, 16), dtype=torch.bfloat16, device="cuda")
    amaxes = torch.empty(group_count, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="1 <= G <= 2048"):
        fused_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_non_block_aligned_k() -> None:
    x = torch.randn((3, 2, 17), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="N divisible by 16"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_misaligned_contiguous_storage() -> None:
    storage = torch.randn(33, dtype=torch.bfloat16, device="cuda")
    x = storage[1:].view(1, 2, 16)
    assert x.is_contiguous()
    assert x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte-aligned"):
        fused_nvfp4_qdq(x, compute_nvfp4_amax(x), NVFP4QDQConfig())


def test_fused_nvfp4_qdq_rejects_wrong_amax_shape() -> None:
    x = torch.randn((3, 2, 16), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        fused_nvfp4_qdq(
            x, torch.ones(1, dtype=torch.float32, device="cuda"), NVFP4QDQConfig()
        )


@torch.inference_mode()
def test_fused_nvfp4_qdq_supports_maximum_group_count() -> None:
    config = NVFP4QDQConfig()
    x = _make_input(2048, (1, 16), torch.bfloat16, "random")
    amaxes = compute_nvfp4_amax(x)
    expected, te_amaxes = _te_reference(x, config)
    actual = fused_nvfp4_qdq(x, amaxes, config)

    assert torch.equal(amaxes.view(torch.int32), te_amaxes.view(torch.int32))
    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_nvfp4_qdq_uses_and_restores_non_current_device() -> None:
    with torch.cuda.device(0):
        with torch.cuda.device(1):
            x = _make_input(3, (3, 32), torch.bfloat16, "boundary")
            amaxes = compute_nvfp4_amax(x)
            expected, _ = _te_reference(x, NVFP4QDQConfig())

        assert torch.cuda.current_device() == 0
        actual = fused_nvfp4_qdq(x, amaxes, NVFP4QDQConfig())
        assert torch.cuda.current_device() == 0

    assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))
