# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark fused NVFP4 QDQ against the former fake-QAT TE round trip.

The model-like shape list and ``torch.utils.benchmark`` methodology follow
Transformer Engine commit 83e230873f00676d4966ca151de22c8bfc68a77f. The
primary speedup compares complete user-visible expressions:

* naive: pad + TE quantize + TE dequantize + crop + contiguous;
* fused: PyTorch FP32 per-tensor amax + register-resident CuTe DSL QDQ.

The separately labeled kernel-only number reuses a precomputed amax and is not
used for the primary speedup claim.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import cutlass
import torch
import torch.utils.benchmark as benchmark
import transformer_engine
import transformer_engine.pytorch as te

from megatron.core.fusions.fused_nvfp4_qdq import (
    NVFP4QDQConfig,
    NVFP4QDQErrorMode,
    compute_nvfp4_amax,
    fused_nvfp4_qdq,
)


BENCHMARK_SHAPES = [
    (8192, 5120),
    (8192, 10240),
    (8192, 2560),
    (8192, 11328),
    (8192, 512),
    (8192, 3584),
    (5120, 8192),
    (10240, 8192),
    (2560, 8192),
    (11328, 8192),
    (512, 8192),
    (3584, 8192),
    (4096, 16384),
    (14336, 16384),
]
DEFAULT_SHAPES = [
    (8192, 512),
    (8192, 3584),
    (3584, 8192),
    (4096, 16384),
]


def _configs() -> list[tuple[str, NVFP4QDQConfig]]:
    configs = [("nvfp4", NVFP4QDQConfig())]
    for error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
        for e4m3_max in (448, 256):
            for error_use_fp16 in (False, True):
                label = (
                    f"4over6-{error_mode.name.lower()}-e4m3-{e4m3_max}-"
                    f"{'fp16-error' if error_use_fp16 else 'exact-error'}"
                )
                configs.append(
                    (
                        label,
                        NVFP4QDQConfig(
                            use_4over6=True,
                            e4m3_max=e4m3_max,
                            error_mode=error_mode,
                            error_use_fp16=error_use_fp16,
                        ),
                    )
                )
    return configs


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


def _naive_te_qdq(x: torch.Tensor, quantizer) -> torch.Tensor:
    m, n = x.shape
    padded_m = ((m + 15) // 16) * 16
    if padded_m == m:
        x_padded = x.contiguous()
    else:
        padding = torch.zeros((padded_m - m, n), dtype=x.dtype, device=x.device)
        x_padded = torch.cat((x.contiguous(), padding), dim=0)
    return quantizer.quantize(x_padded).dequantize(dtype=x.dtype)[:m, :n].contiguous()


def _median_us(function: Callable[[], torch.Tensor], min_run_time: float) -> float:
    timing = benchmark.Timer(
        stmt="function()",
        globals={"function": function},
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)
    return timing.median * 1e6


def _benchmark_case(
    shape: tuple[int, int],
    dtype: torch.dtype,
    config: NVFP4QDQConfig,
    min_run_time: float,
) -> tuple[float, float, float]:
    os.environ["NVTE_USE_FAST_MATH"] = "0"
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = (
        "1" if config.error_use_fp16 else "0"
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    amax = compute_nvfp4_amax(x)
    quantizer = _make_te_quantizer(config)

    def naive() -> torch.Tensor:
        return _naive_te_qdq(x, quantizer)

    def fused_end_to_end() -> torch.Tensor:
        return fused_nvfp4_qdq(x, compute_nvfp4_amax(x), config)

    def fused_kernel_only() -> torch.Tensor:
        return fused_nvfp4_qdq(x, amax, config)

    # Warm native TE and every compiled CuTe specialization before timing.
    naive()
    fused_end_to_end()
    fused_kernel_only()
    torch.cuda.synchronize()
    naive_us = _median_us(naive, min_run_time)
    fused_e2e_us = _median_us(fused_end_to_end, min_run_time)
    fused_kernel_us = _median_us(fused_kernel_only, min_run_time)
    return naive_us, fused_e2e_us, fused_kernel_us


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        m, n = value.lower().split("x", maxsplit=1)
        shape = int(m), int(n)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "shapes must use MxN, for example 8192x3584"
        ) from exc
    if shape[0] <= 0 or shape[1] <= 0 or shape[1] % 16 != 0:
        raise argparse.ArgumentTypeError(
            "M and N must be positive and N must be divisible by 16"
        )
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--full-shapes", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument(
        "--image",
        default=os.getenv("MILES_IMAGE", "unknown"),
        help="Explicit Miles image tag or digest for the output metadata.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    shapes = args.shapes or (BENCHMARK_SHAPES if args.full_shapes else DEFAULT_SHAPES)
    dtypes = {
        "bf16": [torch.bfloat16],
        "fp16": [torch.float16],
        "both": [torch.bfloat16, torch.float16],
    }[args.dtype]

    print(f"image={args.image}")
    print(f"gpu={torch.cuda.get_device_name()}")
    print(f"compute_capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformer_engine={transformer_engine.__version__}")
    print(f"cutlass_dsl={getattr(cutlass, '__version__', 'unknown')}")
    print(f"min_run_time_s={args.min_run_time}")
    print()
    print(
        "| dtype | shape | mode | naive TE QDQ (us) | fused end-to-end (us) | "
        "fused kernel-only (us) | end-to-end speedup |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for dtype in dtypes:
        for shape in shapes:
            for label, config in _configs():
                naive_us, fused_e2e_us, fused_kernel_us = _benchmark_case(
                    shape, dtype, config, args.min_run_time
                )
                print(
                    f"| {str(dtype).removeprefix('torch.')} | {shape[0]}x{shape[1]} | "
                    f"{label} | {naive_us:.3f} | {fused_e2e_us:.3f} | "
                    f"{fused_kernel_us:.3f} | {naive_us / fused_e2e_us:.3f}x |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
