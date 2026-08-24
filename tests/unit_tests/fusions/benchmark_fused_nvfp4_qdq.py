# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark grouped fused NVFP4 QDQ against the former TE fake-QAT loop.

The ``torch.utils.benchmark`` methodology follows Transformer Engine commit
83e230873f00676d4966ca151de22c8bfc68a77f. The benchmark shape is written as
``block-axis x rows`` and stored in one contiguous ``[G, rows, block-axis]``
Parameter. Every 1x16 NVFP4 block therefore lies along the first reported
dimension.

The primary comparison times complete user-visible operations:

* naive TE: the exact pad + quantize + dequantize + crop + contiguous expression
  over zero-copy 2D unbound weight slices;
* grouped fused: one rank-3 QAT call, including the vectorized PyTorch FP32 amax,
  STE wrapper, output allocation, and register-resident CuTe DSL QDQ launch.

The separately labeled singleton fused loop applies the same high-level fused
API to G zero-copy ``[1, rows, block-axis]`` views. It attributes the benefit of
the grouped launch and grouped amax without adding a stack to the timed path.
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
    fake_nvfp4_quantization_ste,
)


DEFAULT_LOGICAL_SHAPES = [(6144, 4096)]
DEFAULT_GROUPS = 8
MAX_GROUPS = 2048


def _configs() -> list[tuple[str, NVFP4QDQConfig]]:
    # Exact-error remains in the zero-tolerance correctness matrix, but the
    # performance target is standard NVFP4 plus TE's current FP16-error path.
    configs = [("nvfp4", NVFP4QDQConfig())]
    for error_mode in (NVFP4QDQErrorMode.MAE, NVFP4QDQErrorMode.MSE):
        for e4m3_max in (448, 256):
            configs.append(
                (
                    f"4over6-{error_mode.name.lower()}-e4m3-{e4m3_max}-fp16-error",
                    NVFP4QDQConfig(
                        use_4over6=True,
                        e4m3_max=e4m3_max,
                        error_mode=error_mode,
                        error_use_fp16=True,
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


def _naive_te_fake_qat(x: torch.Tensor, quantizer) -> torch.Tensor:
    output = _naive_te_qdq(x, quantizer)
    if hasattr(x, "main_grad"):
        output.main_grad = x.main_grad
    return output


def _median_us(function: Callable[[], object], min_run_time: float) -> float:
    timing = benchmark.Timer(
        stmt="function()",
        globals={"function": function},
        num_threads=1,
    ).blocked_autorange(min_run_time=min_run_time)
    return timing.median * 1e6


def _benchmark_case(
    logical_shape: tuple[int, int],
    dtype: torch.dtype,
    config: NVFP4QDQConfig,
    groups: int,
    min_run_time: float,
    repeats: int,
) -> tuple[list[float], list[float], list[float]]:
    os.environ["NVTE_USE_FAST_MATH"] = "0"
    os.environ["NVTE_NVFP4_4OVER6"] = "weights" if config.use_4over6 else "none"
    os.environ["NVTE_NVFP4_4OVER6_E4M3_USE_256"] = (
        "weights" if config.e4m3_max == 256 else "none"
    )
    os.environ["NVTE_NVFP4_4OVER6_ERR_MODE"] = config.error_mode.name
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = (
        "1" if config.error_use_fp16 else "0"
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    block_axis, rows = logical_shape
    weight = torch.nn.Parameter(
        torch.randn((groups, rows, block_axis), dtype=dtype, device="cuda")
    )
    weight.main_grad = torch.empty_like(weight)

    # Build only zero-copy views outside the timed paths. The TE baseline consumes
    # rank-2 slices, while the pre-grouping fused diagnostic consumes G singleton
    # rank-3 slices. Neither path performs a stack or clone.
    weight_slices = tuple(weight.unbind(0))
    main_grad_slices = tuple(weight.main_grad.unbind(0))
    singleton_weights = tuple(
        weight_slice.unsqueeze(0) for weight_slice in weight_slices
    )
    singleton_main_grads = tuple(
        main_grad_slice.unsqueeze(0) for main_grad_slice in main_grad_slices
    )
    for weight_slice, main_grad_slice in zip(weight_slices, main_grad_slices):
        weight_slice.main_grad = main_grad_slice
    for singleton_weight, singleton_main_grad in zip(
        singleton_weights, singleton_main_grads
    ):
        singleton_weight.main_grad = singleton_main_grad

    quantizer = _make_te_quantizer(config)

    def naive() -> list[torch.Tensor]:
        return [
            _naive_te_fake_qat(weight_slice, quantizer)
            for weight_slice in weight_slices
        ]

    def grouped_fused() -> torch.Tensor:
        return fake_nvfp4_quantization_ste(weight, config)

    def singleton_fused_loop() -> list[torch.Tensor]:
        return [
            fake_nvfp4_quantization_ste(singleton_weight, config)
            for singleton_weight in singleton_weights
        ]

    # Warm native TE and every compiled CuTe specialization before timing.
    expected = naive()
    actual = grouped_fused()
    singleton_actual = singleton_fused_loop()
    if tuple(actual.shape) != tuple(weight.shape) or not actual.is_contiguous():
        raise AssertionError(
            f"grouped fused QDQ returned shape {tuple(actual.shape)} "
            f"with contiguous={actual.is_contiguous()}"
        )
    if actual.main_grad is not weight.main_grad:
        raise AssertionError("grouped fused QDQ did not preserve the packed main_grad")
    for weight_idx, (expected_weight, singleton_weight) in enumerate(
        zip(expected, singleton_actual)
    ):
        actual_weight = actual[weight_idx]
        singleton_weight_2d = singleton_weight[0]
        if not torch.equal(
            expected_weight.detach().view(torch.uint16),
            actual_weight.detach().view(torch.uint16),
        ):
            mismatch_count = torch.count_nonzero(
                expected_weight.detach().view(torch.uint16)
                != actual_weight.detach().view(torch.uint16)
            ).item()
            raise AssertionError(
                f"fused QDQ weight {weight_idx} differs from TE in "
                f"{mismatch_count} elements"
            )
        if not torch.equal(
            expected_weight.detach().view(torch.uint16),
            singleton_weight_2d.detach().view(torch.uint16),
        ):
            mismatch_count = torch.count_nonzero(
                expected_weight.detach().view(torch.uint16)
                != singleton_weight_2d.detach().view(torch.uint16)
            ).item()
            raise AssertionError(
                f"singleton fused QDQ weight {weight_idx} differs from TE in "
                f"{mismatch_count} elements"
            )
        if expected_weight.main_grad is not main_grad_slices[weight_idx]:
            raise AssertionError(
                f"naive TE QDQ weight {weight_idx} did not preserve its main_grad"
            )
        if singleton_weight.main_grad is not singleton_main_grads[weight_idx]:
            raise AssertionError(
                f"singleton fused QDQ weight {weight_idx} did not preserve its main_grad"
            )
    del expected, actual, singleton_actual
    torch.cuda.synchronize()

    naive_us = []
    grouped_fused_us = []
    singleton_fused_us = []
    for _ in range(repeats):
        # Interleaved A/B/A guards the primary comparison against clock drift
        # and neighboring-workload interference on a shared GPU node.
        naive_us.append(_median_us(naive, min_run_time))
        grouped_fused_us.append(_median_us(grouped_fused, min_run_time))
        naive_us.append(_median_us(naive, min_run_time))
        singleton_fused_us.append(_median_us(singleton_fused_loop, min_run_time))
    return naive_us, grouped_fused_us, singleton_fused_us


def _format_samples(samples: list[float]) -> str:
    raw = ", ".join(f"{sample:.3f}" for sample in samples)
    return f"{_sample_median(samples):.3f} [{raw}]"


def _sample_median(samples: list[float]) -> float:
    ordered = sorted(samples)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        block_axis, rows = value.lower().split("x", maxsplit=1)
        shape = int(block_axis), int(rows)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "shapes must use BLOCK_AXISxROWS, for example 6144x4096"
        ) from exc
    if shape[0] <= 0 or shape[1] <= 0 or shape[0] % 16 != 0:
        raise argparse.ArgumentTypeError(
            "BLOCK_AXIS and ROWS must be positive and BLOCK_AXIS must be divisible by 16"
        )
    return shape


def _parse_groups(value: str) -> int:
    try:
        groups = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("groups must be an integer") from exc
    if groups < 1 or groups > MAX_GROUPS:
        raise argparse.ArgumentTypeError(
            f"groups must be between 1 and {MAX_GROUPS}, inclusive"
        )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape, dest="shapes")
    parser.add_argument("--groups", type=_parse_groups, default=DEFAULT_GROUPS)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--image",
        default=os.getenv("MILES_IMAGE", "unknown"),
        help="Explicit Miles image tag or digest for the output metadata.",
    )
    parser.add_argument(
        "--commit",
        default=os.getenv("MEGATRON_COMMIT", "unknown"),
        help="Tested Megatron commit for output metadata.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.min_run_time <= 0.0:
        raise ValueError("--min-run-time must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(reason)

    shapes = args.shapes or DEFAULT_LOGICAL_SHAPES
    dtypes = {
        "bf16": [torch.bfloat16],
        "fp16": [torch.float16],
        "both": [torch.bfloat16, torch.float16],
    }[args.dtype]

    print(f"image={args.image}")
    print(f"megatron_commit={args.commit}")
    print(f"gpu={torch.cuda.get_device_name()}")
    print(f"compute_capability={torch.cuda.get_device_capability()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"transformer_engine={transformer_engine.__version__}")
    print(f"cutlass_dsl={getattr(cutlass, '__version__', 'unknown')}")
    print(f"min_run_time_s={args.min_run_time}")
    print(f"repeats={args.repeats}")
    print(f"groups={args.groups}")
    print(f"num_gemms={args.groups}")
    print("main_grad_present=true")
    print("shape_contract=logical_block_axis_x_rows")
    print("primary_tensor_layout=one_contiguous_Parameter_[G,rows,block_axis]")
    print("naive_te_input_views=zero_copy_unbind_[rows,block_axis]")
    print("singleton_fused_input_views=zero_copy_unsqueeze_[1,rows,block_axis]")
    print("no_stack_or_clone_in_timed_paths=true")
    print(
        "grouped_timing_includes=vectorized_fp32_amax,STE,output_allocation,QDQ_launch"
    )
    print(
        "timing_excludes=input_initialization,view_creation,quantizer_creation,warmup"
    )
    for block_axis, rows in shapes:
        print(f"in_features={block_axis}")
        print(f"out_features={rows}")
        print(f"stored_weight_shape=[{args.groups},{rows},{block_axis}]")
    print("primary_order=naive/grouped-fused/naive per repeat")
    print("diagnostic_order=singleton-fused after primary A/B/A")
    print("NVTE_USE_FAST_MATH=0")
    print()
    print(
        "| dtype | stored shape [G, rows, block-axis] | "
        "logical shape (block-axis x rows) | mode | "
        f"naive TE {args.groups}-slice loop median [A/B/A raw] (us) | "
        "grouped fused QAT one-call median [raw] (us) | "
        f"singleton fused QAT {args.groups}-call loop diagnostic median [raw] (us) | "
        "grouped vs naive | grouped vs singleton |"
    )
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for dtype in dtypes:
        for shape in shapes:
            for label, config in _configs():
                naive_us, grouped_fused_us, singleton_fused_us = _benchmark_case(
                    shape,
                    dtype,
                    config,
                    args.groups,
                    args.min_run_time,
                    args.repeats,
                )
                print(
                    f"| {str(dtype).removeprefix('torch.')} | "
                    f"[{args.groups},{shape[1]},{shape[0]}] | {shape[0]}x{shape[1]} | "
                    f"{label} | {_format_samples(naive_us)} | "
                    f"{_format_samples(grouped_fused_us)} | "
                    f"{_format_samples(singleton_fused_us)} | "
                    f"{_sample_median(naive_us) / _sample_median(grouped_fused_us):.3f}x | "
                    f"{_sample_median(singleton_fused_us) / _sample_median(grouped_fused_us):.3f}x |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
