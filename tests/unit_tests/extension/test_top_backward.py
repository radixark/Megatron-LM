"""Gradient tests for the hand-written true-on-policy backward Functions.

The analytic backwards — the RMSNorm vjp (``kernels._RmsNormBatchInvariant``) and the O(L)
per-segment attention vjp (``sglang_attention._FlashinferRaggedAttn``) — have no inference
counterpart, so they are hand-written and need direct validation. Their forwards are
CUDA-kernel-bound (SGLang batch-invariant Triton; flashinfer, Blackwell), and those kernels
run in fp32/bf16, not fp64 — so a finite-difference ``torch.autograd.gradcheck`` on the real
forward isn't possible. Instead each test invokes the real Function and compares its gradient
to a trusted, differentiable torch reference (whose own autograd is gradcheck-validated by
PyTorch). CUDA + the kernels are required, so these skip on CPU CI and run on GPU.
"""

import pytest
import torch

CUDA = torch.cuda.is_available()


def _rms_norm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Pure-torch differentiable RMSNorm with implicit unit weight (same math as the kernel).
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@pytest.mark.skipif(not CUDA, reason="TOP batch-invariant RMSNorm kernel is CUDA-only")
def test_rms_norm_batch_invariant_backward_matches_reference():
    pytest.importorskip("sglang.srt.batch_invariant_ops")
    from sglang.srt.batch_invariant_ops import rms_norm_batch_invariant as bare_kernel

    from miles_megatron_plugins.true_on_policy import kernels

    H, N, eps = 128, 8, 1e-6
    x = torch.randn(N, H, device="cuda", dtype=torch.float32)

    # Forward must stay bitwise-identical to the bare kernel (the wrapper preserves TOP parity).
    ones = torch.ones(H, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(kernels.rms_norm_batch_invariant(x, eps), bare_kernel(x, ones, eps))

    # Backward: our analytic vjp reproduces the fp32 torch RMSNorm reference's autograd.
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    go = torch.randn_like(x)
    kernels.rms_norm_batch_invariant(xa, eps).backward(go)
    _rms_norm_ref(xb, eps).backward(go)
    rel = ((xa.grad - xb.grad).norm() / xb.grad.norm().clamp_min(1e-12)).item()
    assert torch.isfinite(xa.grad).all()
    assert rel < 1e-3, f"RMSNorm analytic backward vs reference rel err {rel:.2e}"


@pytest.mark.skipif(not CUDA, reason="flashinfer ragged prefill + CUDA required")
def test_flashinfer_ragged_attn_backward_matches_sdpa():
    pytest.importorskip("flashinfer")
    import torch.nn.functional as F

    from miles_megatron_plugins.true_on_policy.sglang_attention import _FlashinferRaggedAttn

    H, HKV, D = 32, 8, 128
    scale = 1.0 / (D ** 0.5)
    rep = H // HKV
    # Packed multi-segment input including length-1 segments (the real rollout edge case).
    cu = torch.tensor([0, 137, 138, 400, 801, 1300, 1301, 2048], dtype=torch.int32, device="cuda")
    T = int(cu[-1])

    torch.manual_seed(0)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    go = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)

    # Block-diagonal causal reference via SDPA (GQA heads expanded), differentiable.
    seg = torch.zeros(T, dtype=torch.long, device="cuda")
    cl = cu.long()
    for i in range(cl.numel() - 1):
        seg[cl[i]:cl[i + 1]] = i
    idx = torch.arange(T, device="cuda")
    mask = (seg[:, None] == seg[None, :]) & (idx[:, None] >= idx[None, :])

    def _bhtd(t):  # [T, heads, D] -> [1, heads, T, D]
        return t.transpose(0, 1).unsqueeze(0)

    def _expand(t):  # GQA: expand kv heads to the query-head count
        return t.repeat_interleave(rep, 1) if rep > 1 else t

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    o_ref = (
        F.scaled_dot_product_attention(
            _bhtd(qr), _bhtd(_expand(kr)), _bhtd(_expand(vr)), attn_mask=mask[None, None], scale=scale
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    gq_ref, gk_ref, gv_ref = torch.autograd.grad(o_ref, [qr, kr, vr], go)

    qp, kp, vp = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = _FlashinferRaggedAttn.apply(qp, kp, vp, cu, cu, H, HKV, D, scale)
    out.backward(go)

    def _rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    assert _rel(out, o_ref) < 5e-3, "flashinfer forward diverged from the SDPA reference"
    for name, gp, gr in (("dq", qp.grad, gq_ref), ("dk", kp.grad, gk_ref), ("dv", vp.grad, gv_ref)):
        assert torch.isfinite(gp).all(), f"attention backward {name} has non-finite grads"
        assert _rel(gp, gr) < 5e-3, f"attention backward {name} vs SDPA reference rel err too large"


def test_flashinfer_prefill_backend_matches_sglang(monkeypatch):
    # CPU guard (no GPU): the training-side flashinfer prefill backend MUST match sglang's
    # rollout-side choice (flashinfer_backend.py: cutlass on SM100/Blackwell, auto otherwise).
    # If it drifts, train and rollout run different flashinfer kernels and parity breaks SILENTLY
    # -- exactly what happened when the backend was left unset: Blackwell coincidentally matched,
    # Hopper diverged (abs_diff 0.017). This locks the two together by rule.
    from miles_megatron_plugins.true_on_policy import sglang_attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    assert sglang_attention._fmha_backend("cuda") == "cutlass"  # SM100 / Blackwell
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert sglang_attention._fmha_backend("cuda") == "auto"  # Hopper and earlier
