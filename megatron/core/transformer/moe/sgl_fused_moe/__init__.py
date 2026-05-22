"""SGLang fused MoE helpers used by true-on-policy training.

- forward.py:  Canonical shared forward (single source of truth).
- autograd.py: Autograd wrapper attaching Triton backward to the shared forward.
- fused_moe_triton_backward_kernels.py: Low-level Triton GEMM backward kernels.
"""
