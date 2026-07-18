"""Adapter seam for true-on-policy parity-critical kernels.

This is the ONE place the Megatron true-on-policy backend imports the invariant
kernels; they delegate to SGLang's implementations (SGLang ships these as its
deterministic / true-on-policy feature). The rest of the plugin calls these functions
and never names SGLang directly, so parity comes from running the *same* kernel as
inference -- not a reimplemented copy that can silently drift (see matmul.py history).

Pattern: wrap SGLang's forward-only kernel in a ``torch.autograd.Function`` whose
backward is the standard analytic linear vjp (plain GEMMs). The backward has no
inference counterpart, so it needs no special kernel and is precision-agnostic.
"""

from __future__ import annotations

import torch


class _TpInvRowLinear(torch.autograd.Function):
    """Row-linear local GEMM delegated to SGLang's exact TP-invariant kernel."""

    @staticmethod
    def forward(ctx, input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        import sglang.srt.tp_invariant_ops  # noqa: F401  (registers torch.ops.tp_inv_ops)

        ctx.save_for_backward(input_2d, weight)
        # weight is [out, K_local] (Megatron RowParallelLinear); the kernel wants [K, N].
        return torch.ops.tp_inv_ops.matmul_tp_inv(input_2d.contiguous(), weight.t(), None)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.transpose(-2, -1), input_2d)
        return grad_input, grad_weight


def tp_invariant_row_linear(input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-rank row-linear local GEMM, bitwise-identical to SGLang's TP-invariant kernel.

    Args:
        input_2d: ``[tokens, K_local]`` activation shard.
        weight:   ``[out, K_local]`` (Megatron ``RowParallelLinear`` layout).
    Returns ``[tokens, out]``; differentiable (backward = analytic linear vjp).
    """
    return _TpInvRowLinear.apply(input_2d, weight)


class _RmsNormBatchInvariant(torch.autograd.Function):
    """RMS-normalize (weight=1) via SGLang's exact batch-invariant kernel.

    Forward runs the same forward-only Triton kernel SGLang uses (bitwise inference
    parity). That kernel is non-differentiable (writes an ``empty_like`` output, no
    grad_fn), so training needs this wrapper: backward is the analytic RMSNorm vjp
    (plain torch), which has no inference counterpart and need not match a kernel.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant

        ones = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x)
        ctx.eps = eps
        return rms_norm_batch_invariant(x, ones, eps)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        (x,) = ctx.saved_tensors
        # y = x / sqrt(mean(x^2) + eps); vjp over the last dim (H):
        #   grad_x = (grad_y - x * mean(grad_y * x) / ms) / rms
        ms = x.pow(2).mean(-1, keepdim=True) + ctx.eps
        rms = ms.sqrt()
        dot = (grad_y * x).mean(-1, keepdim=True)
        grad_x = (grad_y - x * (dot / ms)) / rms
        return grad_x, None


def rms_norm_batch_invariant(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant RMS-normalize (implicit unit weight), differentiable.

    Forward is SGLang's exact kernel (bitwise inference parity); backward is the analytic
    RMSNorm vjp. The affine ``weight`` multiply stays in the caller (already autograd-safe).
    """
    return _RmsNormBatchInvariant.apply(x, eps)
