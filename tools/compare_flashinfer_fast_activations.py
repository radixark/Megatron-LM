# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare Megatron activations with source-mirrored CuTe oracles.

This experiment intentionally does not import FlashInfer.  The CuTe DSL kernel
below mirrors the elementwise arithmetic used by FlashInfer's SM100 fast SwiGLU
path, while omitting GEMM, routing, permutation, and tensor-layout concerns.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megatron.core.activations import squared_relu  # noqa: E402
from megatron.core.fusions.fast_activations import (  # noqa: E402
    flashinfer_fast_swiglu,
    use_fast_activations,
)
from megatron.core.fusions.fused_bias_swiglu import megatron_swiglu, swiglu  # noqa: E402


_LOG2_E = 1.4426950408889634074
_THREADS = 256
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_CASES = ("edge", "sweep", "normal", "wide")


@dsl_user_op
def _fadd_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Add two FP32 values with an explicit round-to-nearest PTX instruction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Multiply two FP32 values with an explicit round-to-nearest PTX instruction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Return the FP32 maximum using the same PTX operation as FlashInfer."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _CuteFastSwiGLUOracle:
    """Plain-layout CuTe DSL oracle for FlashInfer's fast activation arithmetic."""

    @cute.kernel
    def kernel(self, input_tensor: cute.Tensor, output_tensor: cute.Tensor):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        output_idx = block_idx * _THREADS + thread_idx
        output_size = cute.size(output_tensor)
        expert_hidden = cute.size(output_tensor, mode=[1])

        if output_idx < output_size:
            row = output_idx // expert_hidden
            column = output_idx % expert_hidden
            gate = input_tensor[row, column].to(cutlass.Float32)
            up = input_tensor[row, column + expert_hidden].to(cutlass.Float32)

            exp_arg = _fmul_rn(gate, Float32(-_LOG2_E))
            neg_gate_exp = cute.math.exp2(exp_arg, fastmath=True)
            denominator = _fadd_rn(neg_gate_exp, Float32(1.0))
            sigmoid = cute.arch.rcp_approx(denominator)
            result = _fmul_rn(_fmul_rn(sigmoid, gate), up)
            output_tensor[row, column] = result.to(output_tensor.element_type)

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(input_tensor, output_tensor).launch(
            grid=(cute.ceil_div(cute.size(output_tensor), _THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


class _CuteRelu2Oracle:
    """Plain-layout CuTe DSL oracle for FlashInfer's ReLU-squared arithmetic."""

    @cute.kernel
    def kernel(self, input_tensor: cute.Tensor, output_tensor: cute.Tensor):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        output_idx = block_idx * _THREADS + thread_idx

        if output_idx < cute.size(output_tensor):
            hidden = cute.size(output_tensor, mode=[1])
            row = output_idx // hidden
            column = output_idx % hidden
            value = input_tensor[row, column].to(cutlass.Float32)
            relu = _fmax_f32(value, Float32(0.0))
            output_tensor[row, column] = _fmul_rn(relu, relu).to(
                output_tensor.element_type
            )

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(input_tensor, output_tensor).launch(
            grid=(cute.ceil_div(cute.size(output_tensor), _THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


class CuteOracleRunner:
    """Compile the CuTe oracle once for each shape and dtype."""

    def __init__(self, kernel, output_width_divisor: int):
        self._compiled = {}
        self._kernel = kernel
        self._output_width_divisor = output_width_divisor

    @staticmethod
    def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
        return from_dlpack(tensor.detach(), assumed_align=16)

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (
                *input_tensor.shape[:-1],
                input_tensor.shape[-1] // self._output_width_divisor,
            ),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        input_cute = self._as_cute(input_tensor)
        output_cute = self._as_cute(output)
        stream = cuda.CUstream(torch.cuda.current_stream(input_tensor.device).cuda_stream)
        key = (
            input_tensor.device.index,
            input_tensor.dtype,
            tuple(input_tensor.shape),
        )

        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = cute.compile(self._kernel, input_cute, output_cute, stream)
            self._compiled[key] = compiled
        compiled(input_cute, output_cute, stream)
        return output


def _repeat_pattern(pattern: torch.Tensor, count: int) -> torch.Tensor:
    repeats = math.ceil(count / pattern.numel())
    return pattern.repeat(repeats)[:count]


def _make_input(
    case: str,
    rows: int,
    expert_hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    count = rows * expert_hidden
    if case == "edge":
        gate_values = torch.tensor(
            [
                -20.0,
                -10.0,
                -5.0,
                -2.0,
                -1.0,
                -0.5,
                -2.0**-20,
                -0.0,
                0.0,
                2.0**-20,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        up_values = torch.tensor(
            [-8.0, -2.0, -1.0, -0.0, 0.0, 2.0**-20, 1.0, 2.0, 8.0],
            dtype=torch.float32,
            device=device,
        )
        gate_grid = gate_values[:, None].expand(-1, up_values.numel()).reshape(-1)
        up_grid = up_values[None, :].expand(gate_values.numel(), -1).reshape(-1)
        gate = _repeat_pattern(gate_grid, count).reshape(rows, expert_hidden)
        up = _repeat_pattern(up_grid, count).reshape(rows, expert_hidden)
    elif case == "sweep":
        gate = torch.linspace(-20.0, 20.0, count, dtype=torch.float32, device=device).reshape(
            rows, expert_hidden
        )
        up = torch.linspace(3.0, -3.0, count, dtype=torch.float32, device=device).reshape(
            rows, expert_hidden
        )
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        gate = torch.randn(
            (rows, expert_hidden),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        up = torch.randn(
            (rows, expert_hidden),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        if case == "wide":
            gate.mul_(6.0)
            up.mul_(3.0)

    return torch.cat((gate, up), dim=-1).to(dtype=dtype).contiguous()


def _ordered_float_bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        signed_bits = tensor.contiguous().view(torch.int32).to(torch.int64)
        minimum = -(1 << 31)
    elif tensor.dtype == torch.bfloat16:
        signed_bits = tensor.contiguous().view(torch.int16).to(torch.int64)
        minimum = -(1 << 15)
    else:
        raise TypeError(f"ULP distance is unsupported for {tensor.dtype}")
    return torch.where(signed_bits < 0, minimum - signed_bits, signed_bits)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    if actual.shape != reference.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {reference.shape}")
    if actual.dtype != reference.dtype:
        raise AssertionError(f"dtype mismatch: {actual.dtype} != {reference.dtype}")

    exact_mismatch = torch.count_nonzero(actual != reference)
    element_count = actual.numel()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    finite_count = int(finite.sum().item())

    if finite_count:
        actual_finite = actual[finite].to(torch.float64)
        reference_finite = reference[finite].to(torch.float64)
        difference = actual_finite - reference_finite
        abs_difference = difference.abs()
        reference_l2 = torch.linalg.vector_norm(reference_finite)
        difference_l2 = torch.linalg.vector_norm(difference)
        relative_l2 = (
            float((difference_l2 / reference_l2).item())
            if reference_l2.item() != 0.0
            else (0.0 if difference_l2.item() == 0.0 else math.inf)
        )

        actual_bits = _ordered_float_bits(actual[finite])
        reference_bits = _ordered_float_bits(reference[finite])
        ulp = (actual_bits - reference_bits).abs()
        finite_metrics = {
            "max_abs": float(abs_difference.max().item()),
            "mean_abs": float(abs_difference.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(difference * difference)).item()),
            "relative_l2": relative_l2,
            "max_ulp": int(ulp.max().item()),
            "mean_ulp": float(ulp.to(torch.float64).mean().item()),
        }
    else:
        finite_metrics = {
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "relative_l2": None,
            "max_ulp": None,
            "mean_ulp": None,
        }

    mismatch_count = int(exact_mismatch.item())
    return {
        "element_count": element_count,
        "finite_pair_count": finite_count,
        "exact_match": mismatch_count == 0,
        "exact_mismatch_count": mismatch_count,
        "exact_mismatch_fraction": mismatch_count / element_count,
        "exact_mismatch_percent": 100.0 * mismatch_count / element_count,
        **finite_metrics,
    }


def _run_implementation(
    implementation: Callable[[torch.Tensor], torch.Tensor], input_tensor: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        output = implementation(input_tensor)
    if not output.is_contiguous():
        output = output.contiguous()
    return output


def _print_result(case: str, dtype: str, implementation: str, metrics: dict) -> None:
    print(
        f"{case:>6} {dtype:>8} {implementation:>22} "
        f"mismatch={metrics['exact_mismatch_percent']:9.5f}% "
        f"max_abs={metrics['max_abs']!s:>14} "
        f"rmse={metrics['rmse']!s:>14} "
        f"rel_l2={metrics['relative_l2']!s:>14} "
        f"max_ulp={metrics['max_ulp']!s:>10}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Megatron SwiGLU and ReLU2 with source-mirrored CuTe arithmetic."
    )
    parser.add_argument("--rows", type=int, default=7168)
    parser.add_argument("--expert-hidden", type=int, default=2048)
    parser.add_argument(
        "--relu2-hidden",
        type=int,
        default=1856,
        help="Nemotron-3-Nano-30B-A3B expert hidden width used for the ReLU2 comparison.",
    )
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    parser.add_argument(
        "--dtypes", nargs="+", choices=tuple(_DTYPES), default=["float32", "bfloat16"]
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--assert-exact-cases",
        nargs="*",
        choices=_CASES,
        default=list(_CASES),
        help="FP32 cases for which the FlashInfer-compatible candidate must exactly match CuTe.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the complete machine-readable result.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rows <= 0 or args.expert_hidden <= 0 or args.relu2_hidden <= 0:
        raise ValueError("--rows, --expert-hidden, and --relu2-hidden must all be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA GPU")

    device = torch.device("cuda", torch.cuda.current_device())
    fast_enabled = bool(use_fast_activations())
    selected_name = "flashinfer_fast_swiglu" if fast_enabled else "megatron_swiglu"
    selected_implementation = flashinfer_fast_swiglu if fast_enabled else megatron_swiglu
    swiglu_oracle_runner = CuteOracleRunner(_CuteFastSwiGLUOracle(), output_width_divisor=2)
    relu2_oracle_runner = CuteOracleRunner(_CuteRelu2Oracle(), output_width_divisor=1)
    results = []
    asserted_fp32_cases = (
        set(args.cases) & set(args.assert_exact_cases)
        if "float32" in args.dtypes
        else set()
    )
    executed_fp32_cases = []
    default_fp32_gap_present = False
    asserted_default_fp32_gap_present = False
    candidate_exact_for_all_fp32_cases = True
    relu2_exact_for_all_cases = True

    print(
        f"SwiGLU shape=[{args.rows}, {2 * args.expert_hidden}] "
        f"output=[{args.rows}, {args.expert_hidden}] selected={selected_name}"
    )
    print(f"ReLU2 shape=[{args.rows}, {args.relu2_hidden}]")
    print("References: source-mirrored CuTe activation arithmetic in plain linear layouts")

    for dtype_index, dtype_name in enumerate(args.dtypes):
        dtype = _DTYPES[dtype_name]
        for case_index, case in enumerate(args.cases):
            input_tensor = _make_input(
                case,
                args.rows,
                args.expert_hidden,
                dtype,
                device,
                args.seed + dtype_index * len(_CASES) + case_index,
            )
            with torch.no_grad():
                oracle = swiglu_oracle_runner(input_tensor)
                default_output = _run_implementation(megatron_swiglu, input_tensor)
                candidate_output = _run_implementation(flashinfer_fast_swiglu, input_tensor)
                public_output = _run_implementation(swiglu, input_tensor)
                expected_public_output = _run_implementation(
                    selected_implementation, input_tensor
                )
                relu2_input = input_tensor[:, : args.relu2_hidden]
                if args.relu2_hidden > args.expert_hidden:
                    relu2_input = _make_input(
                        case,
                        args.rows,
                        args.relu2_hidden,
                        dtype,
                        device,
                        args.seed + 2 * len(_CASES) + dtype_index * len(_CASES) + case_index,
                    )[:, : args.relu2_hidden]
                relu2_input = relu2_input.contiguous()
                relu2_oracle = relu2_oracle_runner(relu2_input)
                megatron_relu2_output = _run_implementation(squared_relu, relu2_input)
            torch.cuda.synchronize(device)

            comparisons = {
                "megatron_swiglu": _metrics(default_output, oracle),
                "flashinfer_fast_swiglu": _metrics(candidate_output, oracle),
                "env_selected_swiglu": _metrics(public_output, oracle),
            }
            alias_metrics = _metrics(public_output, expected_public_output)
            relu2_metrics = _metrics(megatron_relu2_output, relu2_oracle)
            if not alias_metrics["exact_match"]:
                raise AssertionError(
                    f"public swiglu did not exactly match {selected_name} for {case}/{dtype_name}"
                )
            if dtype == torch.float32 and case in args.assert_exact_cases:
                if not comparisons["flashinfer_fast_swiglu"]["exact_match"]:
                    mismatch = comparisons["flashinfer_fast_swiglu"]["exact_mismatch_count"]
                    raise AssertionError(
                        "FlashInfer-compatible candidate did not exactly match the CuTe "
                        f"oracle for {case}/{dtype_name}: {mismatch} elements differ"
                    )
            if dtype == torch.float32:
                executed_fp32_cases.append(case)
                default_fp32_gap_present |= not comparisons["megatron_swiglu"]["exact_match"]
                if case in asserted_fp32_cases:
                    asserted_default_fp32_gap_present |= not comparisons["megatron_swiglu"][
                        "exact_match"
                    ]
                candidate_exact_for_all_fp32_cases &= comparisons["flashinfer_fast_swiglu"][
                    "exact_match"
                ]
            relu2_exact_for_all_cases &= relu2_metrics["exact_match"]
            if not relu2_metrics["exact_match"]:
                raise AssertionError(
                    "Megatron squared_relu did not exactly match the CuTe ReLU2 oracle for "
                    f"{case}/{dtype_name}: {relu2_metrics['exact_mismatch_count']} elements differ"
                )

            for implementation, metrics in comparisons.items():
                _print_result(case, dtype_name, implementation, metrics)
            _print_result(case, dtype_name, "megatron_squared_relu", relu2_metrics)
            results.append(
                {
                    "activation": "swiglu",
                    "case": case,
                    "dtype": dtype_name,
                    "comparisons_to_cute_oracle": comparisons,
                    "env_alias_contract": {
                        "selected_implementation": selected_name,
                        "comparison": alias_metrics,
                    },
                }
            )
            results.append(
                {
                    "activation": "relu2",
                    "case": case,
                    "dtype": dtype_name,
                    "comparisons_to_cute_oracle": {
                        "megatron_squared_relu": relu2_metrics,
                    },
                }
            )

            del (
                input_tensor,
                oracle,
                default_output,
                candidate_output,
                public_output,
                expected_public_output,
                relu2_input,
                relu2_oracle,
                megatron_relu2_output,
            )

    if asserted_fp32_cases and not asserted_default_fp32_gap_present:
        raise AssertionError(
            "The selected FP32 cases did not expose a default Megatron-to-CuTe mismatch"
        )
    closure = {
        "swiglu": {
            "asserted_fp32_cases": sorted(asserted_fp32_cases),
            "executed_fp32_cases": executed_fp32_cases,
            "default_megatron_gap_present": default_fp32_gap_present,
            "flashinfer_fast_candidate_exact": bool(executed_fp32_cases)
            and candidate_exact_for_all_fp32_cases,
            "observed_fp32_gap_closed": (
                bool(executed_fp32_cases)
                and candidate_exact_for_all_fp32_cases
                and default_fp32_gap_present
            ),
        },
        "relu2": {
            "megatron_squared_relu_exact": relu2_exact_for_all_cases,
            "alternative_implementation_needed": not relu2_exact_for_all_cases,
        },
    }
    print(f"Activation closure: {closure}")

    report = {
        "configuration": {
            "rows": args.rows,
            "expert_hidden": args.expert_hidden,
            "input_shape": [args.rows, 2 * args.expert_hidden],
            "output_shape": [args.rows, args.expert_hidden],
            "relu2_shape": [args.rows, args.relu2_hidden],
            "cases": args.cases,
            "dtypes": args.dtypes,
            "seed": args.seed,
            "assert_exact_cases": args.assert_exact_cases,
            "device": torch.cuda.get_device_name(device),
            "env_selected_swiglu_implementation": selected_name,
        },
        "oracles": {
            "dependency": "CuTe DSL; no FlashInfer import",
            "compute_dtype": "float32",
            "layout": "plain contiguous",
            "swiglu_formula": (
                "(rcp_approx(1 + exp2(-gate * log2(e), fastmath=True)) * gate) * up"
            ),
            "relu2_formula": "max(x, 0) * max(x, 0)",
        },
        "closure": closure,
        "results": results,
    }
    json_result = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_result + "\n", encoding="utf-8")
        print(f"Wrote JSON result to {args.json_output}")
    print(json_result)


if __name__ == "__main__":
    main()
