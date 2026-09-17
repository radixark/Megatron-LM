#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare beta1=0 AdamW with torch fused CPU AdamW or TE GPU FusedAdam.

Run directly from the repository with PYTHONPATH=.; no pytest/distributed runner
is required. Timings exclude initialization, gradient generation,
transfers, and training. Both implementations remain resident during alternating
AB/BA trials. CUDA timings include Python dispatch and a final synchronization.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from megatron.core.optimizer.optimizer import _step_with_adam_beta1_zero


def _classes(device):
    if device == "cpu":
        return torch.optim.AdamW, torch.optim.AdamW
    from transformer_engine.pytorch.optimizers import FusedAdam

    return FusedAdam, FusedAdam


def _workload(args, parser):
    if args.preset == "manual":
        return [
            {"name": f"param{index}", "shape": [args.numel or 1048576]}
            for index in range(args.num_params or 64)
        ]
    if args.num_params is not None or args.numel is not None:
        parser.error("--num-params/--numel are only used with --preset manual")
    if 256 % args.ep or 2048 % args.tp or 2048 % args.expert_tp:
        parser.error("GLM-5 requires EP to divide 256 and TP/expert-TP to divide 2048")
    # TEGroupedMLP defaults to one fc1/fc2 parameter per local expert, despite
    # executing grouped GEMMs. Shared experts use dense TP; routed experts use ETP.
    shapes = []
    for expert in range(256 // args.ep):
        shapes.extend(
            [
                {
                    "name": f"experts.linear_fc1.weight{expert}",
                    "shape": [4096 // args.expert_tp, 6144],
                },
                {
                    "name": f"experts.linear_fc2.weight{expert}",
                    "shape": [6144, 2048 // args.expert_tp],
                },
            ]
        )
    return shapes + [
        {"name": "shared_experts.linear_fc1.weight", "shape": [4096 // args.tp, 6144]},
        {"name": "shared_experts.linear_fc2.weight", "shape": [6144, 2048 // args.tp]},
        {"name": "router.weight", "shape": [256, 6144]},
    ]


def _make(optimizer_class, args, sizes, omit_exp_avg=False):
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    params, templates = [], {}
    for size in sizes:
        shape = (size,) if isinstance(size, int) else tuple(size)
        if shape not in templates:
            templates[shape] = (
                torch.randn(
                    shape, device=args.device, dtype=torch.float32, generator=generator
                ).mul_(0.1),
                torch.randn(
                    shape, device=args.device, dtype=torch.float32, generator=generator
                ).mul_(0.01),
            )
        # Repeated expert shapes share values, never storage. Templates live only
        # during construction; seeding a full GLM-5 layer otherwise dominates setup.
        param = torch.nn.Parameter(templates[shape][0].clone())
        param.grad = templates[shape][1].clone()
        params.append(param)
    options = {"fused": True} if args.device == "cpu" else {"adam_w_mode": True}
    optimizer = optimizer_class(
        params, lr=0.001, betas=(0.0, 0.95), eps=1e-8, weight_decay=args.weight_decay, **options
    )
    optimizer.omit_exp_avg = omit_exp_avg
    return optimizer, params


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _cpu_model():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()


def _state_storage(optimizer):
    storages, logical = {}, {}
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
                logical[name] = logical.get(name, 0) + value.numel() * value.element_size()
    return {"unique_bytes": sum(storages.values()), "logical_bytes_by_key": logical}


def _compare_bits(actual, reference, diagnostics, name):
    # All benchmark tensors are FP32. Count differing elements, including signed
    # zero, without allocating a second full model for validation.
    mismatches = actual.detach().view(torch.int32) != reference.detach().view(torch.int32)
    count = mismatches.count_nonzero().item()
    diagnostics["bit_mismatches"][name] += count
    if count:
        error = (actual - reference).abs().max().item()
        diagnostics["max_abs_error"][name] = max(diagnostics["max_abs_error"][name], error)


def _diagnostics():
    return {
        "bit_mismatches": {"parameter": 0, "exp_avg_sq": 0},
        "max_abs_error": {"parameter": 0.0, "exp_avg_sq": 0.0},
    }


def _assert_parity(diagnostics):
    if any(diagnostics["bit_mismatches"].values()):
        raise AssertionError(f"Bitwise parity failed: {json.dumps(diagnostics)}")


def _verify(classes, args):
    pairs = [
        _make(cls, args, [17, 1024, 4097, 3], omit_exp_avg=bool(index))
        for index, cls in enumerate(classes)
    ]
    diagnostics = _diagnostics()
    start = time.perf_counter()
    for step in range(6):
        for index, (left, right) in enumerate(zip(pairs[0][1], pairs[1][1])):
            if index == 3 or (index == 1 and step == 2):
                left.grad = right.grad = None
            else:
                values = torch.arange(left.numel(), device=args.device, dtype=torch.float32)
                grad = (values + step + index).sin() * 0.2
                left.grad, right.grad = grad.clone(), grad.clone()
        saved_grads = [None if p.grad is None else p.grad.clone() for p in pairs[1][1]]
        for optimizer, _ in pairs:
            _step_with_adam_beta1_zero(optimizer)
        for index, (left, right) in enumerate(zip(pairs[0][1], pairs[1][1])):
            reference_state, state = pairs[0][0].state.get(left, {}), pairs[1][0].state.get(
                right, {}
            )
            assert "exp_avg" not in state or state["exp_avg"].numel() == 0
            tensors = {"parameter": (left, right)}
            if "exp_avg_sq" in reference_state:
                tensors["exp_avg_sq"] = (reference_state["exp_avg_sq"], state["exp_avg_sq"])
            for name, (reference, actual) in tensors.items():
                _compare_bits(actual, reference, diagnostics, name)
            if saved_grads[index] is not None:
                torch.testing.assert_close(right.grad, saved_grads[index], rtol=0, atol=0)
    _sync(args.device)
    _assert_parity(diagnostics)
    return {
        "steps": 6,
        "tolerances": {"rtol": 0, "atol": 0},
        **diagnostics,
        "initial_validation_ms": (time.perf_counter() - start) * 1000,
    }


def _first_step(optimizer_class, args, shapes, omit_exp_avg):
    optimizer, params = _make(
        optimizer_class, args, [tuple(item["shape"]) for item in shapes], omit_exp_avg
    )
    _sync(args.device)
    before = torch.cuda.memory_allocated() if args.device == "cuda" else None
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    _step_with_adam_beta1_zero(optimizer)
    _sync(args.device)
    result = {
        "class": optimizer_class.__name__,
        "implementation": "gradient_alias_callsite" if omit_exp_avg else "stock",
        "first_step_ms": (time.perf_counter() - start) * 1000,
        "first_step_peak_delta_bytes": None,
        "state": _state_storage(optimizer),
        "trials_ms": [],
    }
    if args.device == "cuda":
        result["first_step_peak_delta_bytes"] = torch.cuda.max_memory_allocated() - before
        result["first_step_allocated_delta_bytes"] = torch.cuda.memory_allocated() - before
    for _ in range(args.warmup):
        _step_with_adam_beta1_zero(optimizer)
    _sync(args.device)
    return optimizer, params, result


def main() -> None:
    """Run numerical validation and the selected standalone optimizer benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--preset", choices=("manual", "glm5-moe"), default="manual")
    parser.add_argument(
        "--num-params", type=int, help="Manual workload parameter count (default 64)"
    )
    parser.add_argument("--numel", type=int, help="Manual elements per parameter (default 1048576)")
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="GLM-5 expert parallel size; default 1 selects all 256 experts",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="GLM-5 shared-expert tensor parallel size"
    )
    parser.add_argument(
        "--expert-tp", type=int, default=1, help="GLM-5 routed-expert tensor parallel size"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10, help="Steps per timed trial")
    parser.add_argument(
        "--repeats", type=int, default=10, help="Even count balances AB and BA order"
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--image", default="not_recorded", help="Explicit container tag/digest for provenance"
    )
    parser.add_argument(
        "--json", type=Path, help="Also write the complete JSON result to this path"
    )
    args = parser.parse_args()
    counts = [args.threads, args.warmup, args.steps, args.repeats, args.ep, args.tp, args.expert_tp]
    counts += [value for value in (args.num_params, args.numel) if value is not None]
    if min(counts) < 1:
        parser.error("Thread, workload, warmup, step, and repeat counts must be positive")
    if not args.weight_decay >= 0:
        parser.error("--weight-decay must be nonnegative")
    if args.repeats % 2:
        parser.error("--repeats must be even to balance AB/BA trial order")
    shapes = _workload(args, parser)
    total_elements = sum(torch.Size(item["shape"]).numel() for item in shapes)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    classes = _classes(args.device)
    print("Validating small changing-gradient cases...", file=sys.stderr, flush=True)
    validation = _verify(classes, args)
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        # Both parameter+gradient sets, baseline m/v, and specialized v are resident.
        # Leave room for the untimed comparison's temporary tensors and CUDA context.
        required = total_elements * 28
        if required > free * 0.95:
            parser.error(
                f"Paired states require {required / 2**30:.2f} GiB before overhead; "
                f"only {free / 2**30:.2f} GiB free. Increase EP/TP or reduce manual shapes."
            )
    cases = []
    for label, cls in zip(("baseline", "optimized"), classes):
        print(f"Preparing {label} ({cls.__name__})...", file=sys.stderr, flush=True)
        cases.append(_first_step(cls, args, shapes, omit_exp_avg=label == "optimized"))
    print("Timing alternating baseline/optimized trials...", file=sys.stderr, flush=True)
    gc.disable()
    try:
        for repeat in range(args.repeats):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                optimizer, _, result = cases[index]
                _sync(args.device)
                start = time.perf_counter()
                for _ in range(args.steps):
                    _step_with_adam_beta1_zero(optimizer)
                _sync(args.device)
                result["trials_ms"].append((time.perf_counter() - start) * 1000 / args.steps)
    finally:
        gc.enable()
    for _, _, result in cases:
        result["median_ms"] = statistics.median(result["trials_ms"])
    baseline, optimized = [case[2] for case in cases]
    print(
        f"Median ms/step: baseline={baseline['median_ms']:.6f}, "
        f"optimized={optimized['median_ms']:.6f}; "
        f"speedup={baseline['median_ms'] / optimized['median_ms']:.4f}x",
        file=sys.stderr,
        flush=True,
    )
    print("Checking full-workload parameter/state parity...", file=sys.stderr, flush=True)
    assert optimized["state"]["logical_bytes_by_key"].get("exp_avg", 0) == 0
    assert all("exp_avg" not in state for state in cases[1][0].state_dict()["state"].values())
    diagnostics = _diagnostics()
    for reference, actual in zip(cases[0][1], cases[1][1]):
        _compare_bits(actual, reference, diagnostics, "parameter")
        _compare_bits(
            cases[1][0].state[actual]["exp_avg_sq"],
            cases[0][0].state[reference]["exp_avg_sq"],
            diagnostics,
            "exp_avg_sq",
        )
    _assert_parity(diagnostics)
    validation["timed_workload"] = diagnostics
    validation["timed_workload_parameters_and_second_moments_match"] = True
    repo = Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    sources = [
        repo / "megatron/core/optimizer" / name
        for name in (
            "__init__.py",
            "optimizer.py",
            "optimizer_config.py",
            "distrib_optimizer.py",
            "cpu_offloading/hybrid_optimizer.py",
            "cpu_offloading/chunked_optimizer_state_offload.py",
        )
    ]
    output = {
        "environment": {
            "image": args.image,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "threads": torch.get_num_threads(),
            "interop_threads": 1,
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "transformer_engine": (
                importlib.metadata.version("transformer_engine") if args.device == "cuda" else None
            ),
            "commit": head.stdout.strip() if head.returncode == 0 else None,
            "optimizer_source_sha256": {
                str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources
            },
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "workload": {
            **vars(args),
            "json": str(args.json) if args.json else None,
            "dtype": "float32",
            "lr": 0.001,
            "betas": [0.0, 0.95],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
            "parameter_shapes": shapes,
            "parameter_count": len(shapes),
            "total_elements": total_elements,
            "optimizer_dp_sharding": False,
            "initialization": "seeded_template_clones",
        },
        "method": {
            "timing": "perf_counter wall time; CUDA synchronization; alternating AB/BA trials",
            "first_step": "Includes optimizer state allocation after small validation",
            "peak": "CUDA allocator increment above parameters/gradients; CPU peak unavailable",
            "excluded": ["initialization", "gradient generation", "transfers", "training"],
        },
        "validation": validation,
        "baseline": baseline,
        "optimized": optimized,
        "speedup": baseline["median_ms"] / optimized["median_ms"],
        "persistent_state_saved_bytes": baseline["state"]["unique_bytes"]
        - optimized["state"]["unique_bytes"],
    }
    encoded = json.dumps(output, indent=2) + "\n"
    print(encoded, end="", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded)


if __name__ == "__main__":
    main()
