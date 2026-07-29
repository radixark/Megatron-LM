# FlashInfer-fast activation alignment for Miles training

## Status

This is a trainer-side numerical-alignment experiment for the Miles Megatron
fork. FlashInfer remains unchanged in this phase.

The implementation is opt-in:

```bash
export MILES_USE_FAST_ACTIVATIONS=1
```

The plural name is intentional: it represents a process-wide activation
policy and leaves room for additional audited activations. Today it changes
only SwiGLU because ReLU2 does not need an alternative implementation.

The default is disabled. The value is read once when
`megatron.core.fusions.fast_activations` is imported, before `torch.compile`
and CUDA-graph capture, so it must be set at process startup.

Local setup on 2026-07-28:

- Megatron branch: `agent-flashinfer-fast-swiglu-megatron`
- Megatron base: `origin/miles-main` at `9fc14d826`
- Devbox: `cutedsl-strict-swiglu`
- Image: `radixark/miles:dev-202607281246`
- Hardware: NVIDIA B200, SM100
- PyTorch: `2.11.0+cu130`
- CUDA: 13.0
- CUTLASS DSL: 4.5.2

## Source-level conclusions

### SwiGLU

FlashInfer's standard SwiGLU backends use the same mathematical expression,

```text
up * gate * sigmoid(gate)
```

and the same broad fast-math approximation family, but source inspection does
not establish one bitwise contract across all backends:

- The SM100 CuTe DSL MoE epilogue explicitly uses approximate `exp2`,
  approximate reciprocal, then `(sigmoid * gate) * up`.
- The CUTLASS MoE epilogue uses `cutlass::SiLu<float>`, whose sigmoid uses a
  fast exponential; FlashInfer JIT also enables fast math.
- Public CUDA activation and relevant TRT-LLM paths use
  `gate / (1 + __expf(-gate))`.
- Specialized kernels may reassociate multiplication or add alpha, beta, and
  clamp behavior.

The implemented exact target is therefore the explicit SM100 CuTe DSL sequence
used by the B200 rollout kernel:

```text
q = mul.rn.f32(gate, -log2(e))
e = exp2.approx.ftz.f32(q)
d = add.rn.f32(e, 1)
s = rcp.approx.ftz.f32(d)
output = mul.rn.f32(mul.rn.f32(s, gate), up)
```

CUTLASS and CUDA paths are described only as the same approximation family.
Exact cross-backend equality requires a separate runtime comparison.

### Nemotron 3 ReLU2

The relevant Nemotron 3 activation is ReLU2, not SiLU2. The checked-in
Nemotron-3-Nano configuration selects squared ReLU and uses expert hidden width
1856.

There is no analogous fast-math activation mismatch:

- Megatron computes `torch.pow(F.relu(x), 2)`.
- FlashInfer SM100 CuTe computes `max(x, 0)` and then self-multiplies.
- FlashInfer CUTLASS applies ReLU and then self-multiplies.
- No exponential, reciprocal, or other approximate transcendental operation
  is involved.

The experiment below confirms exact agreement for the finite FP32 and BF16
inputs tested. Consequently, no alternate ReLU2 production implementation was
added. Differences caused by GEMM accumulation, casting, scaling,
quantization, or routing-weight order remain outside this activation-only
claim.

## Implementation layout

The change is intentionally easy to cherry-pick and rebase:

- `megatron/core/fusions/fast_activations.py` is the single owner of the env
  snapshot and every concrete fast activation implementation.
- `megatron/core/fusions/fused_bias_swiglu.py` keeps the original Megatron
  implementation and resolves stable forward/backward aliases at import.
- `megatron/core/transformer/moe/experts.py` contains only the small legacy
  `GroupedMLP` selection needed by the local and Miles SGLang backends.
- `tests/unit_tests/fusions/test_fast_activations.py` contains all newly added
  tests; existing test files are unchanged.
- `tools/compare_flashinfer_fast_activations.py` is the one standalone
  experiment.

The opt-in SwiGLU forward is:

```python
sigmoid = torch.reciprocal(1.0 + torch.exp2(-gate * log2_e))
output = (sigmoid * gate) * up
```

Gate and up are promoted to FP32, and output is cast back to the input dtype.
On the pinned Torch image, `torch.compile` lowers the explicit expression to
the same fast `exp2` and reciprocal sequence as the CuTe target.

The existing custom autograd wrappers remain responsible for bias handling,
optional FP8 saved activations, activation offload, token weights, reshaping,
and output dtype. The backward is a trainer-defined surrogate: it retains
Megatron's analytic SwiGLU derivative while evaluating its sigmoid with the
same fast approximation. FlashInfer's target path is inference-only, so this
is not a claim about a FlashInfer backward contract.

Legacy `GroupedMLP` uses the existing weighted custom wrapper when the env is
enabled. This avoids allowing an outer `torch.compile` graph to derive a
different gradient through `exp2` and keeps the same explicit analytic
backward across Megatron MoE paths.

## Scope

Included:

- Standard finite SwiGLU with gate in the first half and up in the second
- FP32 activation arithmetic with BF16 or FP32 input/output storage
- Megatron's analytic training backward evaluated with the fast sigmoid
- Dense, sequential, TE-grouped fused SwiGLU paths
- Legacy `GroupedMLP`
- Finite ReLU2 as a negative-control comparison

Excluded:

- GEMM and accumulator differences
- quantization and scaling
- routing, permutation, unpermutation, and MoE combine
- FlashInfer-specific tensor layouts
- nonstandard SwiGLU alpha, beta, offset, or clamp variants
- TransformerEngine-owned activation functions
- explicitly non-fused generic MLP paths
- legacy non-MCore transformer paths
- performance measurement
- exceptional NaN, infinity, and deliberate subnormal-underflow contracts

## Standalone experiment

`tools/compare_flashinfer_fast_activations.py` does not import FlashInfer. It
uses CUTLASS CuTe DSL to reproduce the owning SM100 elementwise primitive
sequences and otherwise imports only Megatron code from this checkout.

The default ordinary contiguous layouts are:

```text
DeepSeek-V3 SwiGLU input:  [7168, 4096]
DeepSeek-V3 SwiGLU output: [7168, 2048]
Nemotron-3-Nano ReLU2:     [7168, 1856]
```

For SwiGLU, the first 2048 values are gate and the second 2048 are up. No
FlashInfer tensor layout, GEMM, routing, permutation, or combine step is
reproduced.

Four deterministic input distributions are tested:

- `edge`: signed zeros, small values, and representative values in `[-20, 20]`
- `sweep`: a dense deterministic sweep through `[-20, 20]`
- `normal`: standard normal inputs
- `wide`: gate standard deviation 6 and up standard deviation 3

The script reports exact mismatch count and fraction, maximum and mean
absolute error, RMSE, relative L2, and ULP distance. It asserts exact FP32
closure for the fast SwiGLU candidate, verifies the import-time public alias,
and asserts exact ReLU2 agreement.

Reproduction:

```bash
MILES_USE_FAST_ACTIVATIONS=0 \
python tools/compare_flashinfer_fast_activations.py \
  --json-output /tmp/fast-activations-env-off.json

MILES_USE_FAST_ACTIVATIONS=1 \
python tools/compare_flashinfer_fast_activations.py \
  --json-output /tmp/fast-activations-env-on.json
```

## Final B200 validation

The final command-traced run is:

```text
megatron_fast_activations_final_20260728_235000
```

Downloaded artifacts:

- `logs/megatron_fast_activations_final_20260728_235000/run.log`
- `logs/megatron_fast_activations_final_20260728_235000/env-off.json`
- `logs/megatron_fast_activations_final_20260728_235000/env-on.json`

Each SwiGLU case compared 14,680,064 output values:

| dtype | case | default mismatch | default max ULP | fast candidate mismatch |
| --- | --- | ---: | ---: | ---: |
| FP32 | edge | 1,427,230 (9.72223%) | 1 | 0 |
| FP32 | sweep | 5,437,740 (37.04166%) | 18 | 0 |
| FP32 | normal | 1,415,280 (9.64083%) | 6 | 0 |
| FP32 | wide | 4,113,200 (28.01895%) | 29 | 0 |
| BF16 | edge | 0 | 0 | 0 |
| BF16 | sweep | 0 | 0 | 0 |
| BF16 | normal | 0 | 0 | 0 |
| BF16 | wide | 24 (0.00016%) | 1 | 0 |

Across the four FP32 cases, both the candidate and, with
`MILES_USE_FAST_ACTIVATIONS=1`, the env-selected path matched the CuTe SwiGLU
oracle on 58,720,256 values with zero mismatch, zero absolute error, and zero
ULP error. All 58,720,256 BF16 candidate outputs also matched exactly.

Each ReLU2 case compared 13,303,808 values. Megatron squared ReLU matched the
CuTe ReLU2 oracle exactly in all four cases and both dtypes: 106,430,464
values total with zero mismatch, zero absolute error, and zero ULP error.
The report therefore records `alternative_implementation_needed: false`.

An early SwiGLU stress probe included gates as large as `+/-100`. Its only
candidate differences were far-underflow outputs, with maximum absolute error
`4.27e-36`. The finite validation contract above uses `[-20, 20]` for the
edge and sweep cases; the wide random case can sample beyond that range.

## Tests and checks

The focused tests ran once with the env disabled and once enabled:

```text
7 passed with MILES_USE_FAST_ACTIVATIONS=0
7 passed with MILES_USE_FAST_ACTIVATIONS=1
```

They cover FP32/BF16 fast SwiGLU forward and analytic backward, import-time env
selection, existing weighted SwiGLU forward/backward, and existing weighted
ReLU2 forward/backward. The traced log is:

```text
logs/megatron_fast_activations_tests_20260728_234000/run.log
```

Final local static checks:

```text
python3 -m py_compile \
  tools/compare_flashinfer_fast_activations.py \
  megatron/core/fusions/fast_activations.py \
  tests/unit_tests/fusions/test_fast_activations.py

git diff --check
```

Both pass. Per the explicit task instruction, no final pre-commit run is
required for this repository.
