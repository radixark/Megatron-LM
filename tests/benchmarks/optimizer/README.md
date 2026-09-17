# Beta1-zero AdamW benchmark

Run from the repository in a PyTorch environment; GPU runs require Transformer
Engine. This script runs directly, without pytest or a distributed launcher.

To benchmark one complete GLM-5 MoE block, including all 256 routed experts,
the shared expert, and router:

```bash
PYTHONPATH=. python tests/benchmarks/optimizer/benchmark_adam_beta1_zero.py \
  --device cpu --threads 8 --preset glm5-moe --weight-decay 0 \
  --warmup 3 --steps 3 --repeats 6 --json /tmp/glm5-moe-cpu.json

PYTHONPATH=. python tests/benchmarks/optimizer/benchmark_adam_beta1_zero.py \
  --device cuda --threads 1 --preset glm5-moe --weight-decay 0 \
  --warmup 3 --steps 3 --repeats 6 --json /tmp/glm5-moe-cuda.json
```

The default is EP=TP=expert-TP=1. The paired full-layer workload uses about
253 GiB before temporary allocations. `--ep 8` selects one rank's 32 routed
experts plus shared expert and router; `--tp` partitions shared expert weights,
and `--expert-tp` partitions routed expert weights. The script checks available
CUDA memory before allocating. These represent parameter shapes before optimizer
DP sharding; they do not emulate DistributedOptimizer's buffer slicing.

For manually sized tensors:

```bash
PYTHONPATH=. python tests/benchmarks/optimizer/benchmark_adam_beta1_zero.py \
  --device cpu --threads 8 --num-params 1 --numel 16000000 \
  --warmup 5 --steps 10 --repeats 10 --json /tmp/adam-beta1-zero-cpu.json

PYTHONPATH=. python tests/benchmarks/optimizer/benchmark_adam_beta1_zero.py \
  --device cuda --threads 1 --num-params 64 --numel 1048576 \
  --warmup 5 --steps 10 --repeats 10 --json /tmp/adam-beta1-zero-cuda.json
```

Use `--image` to record the exact container tag or digest. CPU comparisons use
`torch.optim.AdamW(fused=True)`; GPU comparisons use TE `FusedAdam`. The optimized arm calls Megatron's gradient-alias step helper on the same stock
optimizer class. Both have beta1 zero and identical seeded FP32 parameters, gradients, and hyperparameters.
Weight decay defaults to zero; use `--weight-decay` to benchmark another value.
Initialization generates one seeded parameter/gradient template per distinct
shape, then clones it into independent storage for every parameter and gradient.
Repeated expert shapes therefore share values, not storage. Templates are released
after construction; JSON records `initialization: seeded_template_clones`.

The JSON includes raw wall-time samples, median speedup, unique persistent state
bytes, CUDA first-step allocation peak, source hashes, and environment metadata.
Trials alternate AB/BA order. CUDA wall time includes Python dispatch and final
synchronization. CPU allocator peaks are reported as unavailable.

The benchmark requires bitwise parameter and second-moment parity for its small
changing-gradient cases and complete timed workload on both CPU and GPU. JSON records
element mismatch counts and maximum absolute errors; any mismatch fails the run.
CPU gradient aliasing can change the sign bit of exact-zero parameters or
gradients while preserving their numerical value; a dedicated unit test covers
this edge case.
Initialization, gradient generation, transfers, and
training are excluded from steady timing. Initial validation and first-step
latencies are reported separately; the first step includes optimizer state
allocation. Run workloads
with both many small tensors and large tensors, at representative CPU thread
counts; these optimizer-only measurements do not measure full HDO training.

## GLM-5 shape provenance

The preset uses hidden size 6144, expert FFN size 2048, 256 routed experts, one
shared expert, SwiGLU, and no linear biases. Sources are the
[official GLM-5 configuration](https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/config.json)
and [Miles model arguments](https://github.com/radixark/miles/blob/78747f69123da316af95e581c78114b26282bb8f/scripts/models/glm5-744B-A40B.py).

For each local routed expert, FC1 is `[4096 / expert-TP, 6144]` and FC2 is
`[6144, 2048 / expert-TP]`. Shared FC1/FC2 use the same shapes with TP. Router
weight is `[256, 6144]` and is replicated. These follow
[TEGroupedMLP](https://github.com/radixark/Megatron-LM/blob/73b54618f7e58e0f25f619bcfecbe2640765475a/megatron/core/transformer/moe/experts.py),
[TE grouped-linear partitioning](https://github.com/radixark/Megatron-LM/blob/73b54618f7e58e0f25f619bcfecbe2640765475a/megatron/core/extensions/transformer_engine.py),
[shared experts](https://github.com/radixark/Megatron-LM/blob/73b54618f7e58e0f25f619bcfecbe2640765475a/megatron/core/transformer/moe/shared_experts.py),
and [router](https://github.com/radixark/Megatron-LM/blob/73b54618f7e58e0f25f619bcfecbe2640765475a/megatron/core/transformer/moe/router.py).

Grouped GEMM uses separate `weight0`, `weight1`, etc. parameters under the default
`moe_single_grouped_weight=False`; the preset preserves this granularity. Expert
correction bias is a buffer and has no optimizer state. Attention, surrounding
normalization, and other transformer blocks are excluded.
