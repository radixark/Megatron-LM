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


@pytest.mark.skipif(not CUDA, reason="TOP tp-invariant row-linear kernel is CUDA-only")
def test_tp_invariant_row_linear_backward_matches_reference():
    pytest.importorskip("sglang.srt.tp_invariant_ops")
    from miles_megatron_plugins.true_on_policy import kernels

    tokens, K, out = 64, 256, 128
    torch.manual_seed(0)
    x = torch.randn(tokens, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(out, K, device="cuda", dtype=torch.bfloat16)  # [out, K_local], RowParallelLinear layout
    go = torch.randn(tokens, out, device="cuda", dtype=torch.bfloat16)

    # Forward = SGLang's matmul_tp_inv (matches x @ Wᵀ within tolerance); backward = the hand-written
    # linear vjp (grad_x = go @ W, grad_w = goᵀ @ x), checked against the reference's autograd.
    xa, wa = (t.clone().requires_grad_(True) for t in (x, w))
    xb, wb = (t.clone().requires_grad_(True) for t in (x, w))
    out_k = kernels.tp_invariant_row_linear(xa, wa)
    out_ref = xb @ wb.t()
    out_k.backward(go)
    out_ref.backward(go)

    def _rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    assert _rel(out_k, out_ref) < 5e-3, "tp-invariant row-linear forward diverged from reference"
    for name, gk, gr in (("dx", xa.grad, xb.grad), ("dw", wa.grad, wb.grad)):
        assert torch.isfinite(gk).all(), f"row-linear backward {name} has non-finite grads"
        assert _rel(gk, gr) < 5e-3, f"row-linear backward {name} vs reference rel err too large"


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


def test_flashinfer_prefill_backend_is_fa2():
    # Cheap rule guard; the real check is the output-parity test below.
    from miles_megatron_plugins.true_on_policy import sglang_attention

    assert sglang_attention._fmha_backend("cuda") == "fa2"


@pytest.mark.skipif(not CUDA, reason="flashinfer required")
def test_flashinfer_output_matches_deterministic_paged_reference():
    # Our training flashinfer path (ragged + _fmha_backend) must be bitwise-equal to sglang's
    # deterministic prefill kernel (paged + fa2) on identical q/k/v. Catches a backend drift.
    pytest.importorskip("flashinfer")
    from flashinfer import (
        BatchPrefillWithRaggedKVCacheWrapper as Ragged,
        BatchPrefillWithPagedKVCacheWrapper as Paged,
    )
    from miles_megatron_plugins.true_on_policy import sglang_attention

    dev, HD, NQ, NKV, L = "cuda", 128, 16, 4, 1024
    torch.manual_seed(0)
    q = torch.randn(L, NQ, HD, device=dev, dtype=torch.bfloat16)
    k = torch.randn(L, NKV, HD, device=dev, dtype=torch.bfloat16)
    v = torch.randn(L, NKV, HD, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, L], dtype=torch.int32, device=dev)

    # our training path: ragged wrapper with the backend _fmha_backend selects
    be = sglang_attention._fmha_backend(dev)
    wr = Ragged(torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), kv_layout="NHD", backend=be)
    wr.plan(cu, cu, NQ, NKV, HD, causal=True, q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16, fixed_split_size=4096)
    o_ours = wr.run(q, k, v)

    # sglang deterministic reference: paged wrapper + "fa2", page_size=1 over the same K/V
    wp = Paged(torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), kv_layout="NHD", backend="fa2")
    wp.plan(cu, cu, torch.arange(L, dtype=torch.int32, device=dev),
            torch.tensor([1], dtype=torch.int32, device=dev), NQ, NKV, HD, 1, causal=True,
            q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, fixed_split_size=4096)
    o_ref = wp.run(q, (k.view(L, 1, NKV, HD).contiguous(), v.view(L, 1, NKV, HD).contiguous()))

    assert torch.equal(o_ours, o_ref), (
        f"training flashinfer backend {be!r} does not bitwise-match sglang deterministic paged+fa2 "
        f"(max abs diff {(o_ours.float() - o_ref.float()).abs().max().item():.3e})"
    )
