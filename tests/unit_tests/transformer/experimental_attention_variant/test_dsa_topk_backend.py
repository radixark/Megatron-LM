# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
)
from megatron.core.transformer.experimental_attention_variant.dsa_topk import select_dsa_topk


def _config(backend="flashinfer", tie_break=1):
    return SimpleNamespace(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_topk_backend=backend,
        dsa_indexer_topk_deterministic=backend == "flashinfer",
        dsa_indexer_topk_tie_break=tie_break if backend == "flashinfer" else 0,
    )


def _require_flashinfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashInfer top-k")
    return pytest.importorskip("flashinfer")


def _indexer_inputs(sq, sk, batch, heads, dim, nonzero):
    q = torch.zeros(sq, batch, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros(sk, batch, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(sq, batch, heads, dtype=torch.bfloat16, device="cuda")
    if nonzero:
        positions = torch.arange(sk, device="cuda")
        k[:, :, 0] = (positions % 17 - 8)[:, None]
        k[:, :, 1] = ((positions * 5) % 19 - 9)[:, None]
        k[:, :, 2] = ((positions + 1) / 1024)[:, None]
        for head in range(heads - 1):
            q[:, :, head, head % 2] = 1
        q[:, :, -1, 2] = 1
        weights[:] = torch.arange(heads, device="cuda") % 4 + 1
        # Integer channels exercise ReLU and unequal head weights. A separate
        # positive fractional head keeps scores unique without BF16 rounding.
    return q, k, weights


def _reference_scores(q, k, weights, starts, ends):
    scores = torch.einsum("sbhd,tbd->bsht", q.float(), k.float()).relu()
    scores = torch.einsum("bsht,sbh->bst", scores, weights.float())
    positions = torch.arange(k.size(0), device=k.device)
    valid = (positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None])
    return scores.masked_fill(~valid[None, :, :], -torch.inf), valid


def test_external_topk_cli_fields_are_generated():
    from argparse import ArgumentParser
    from dataclasses import fields

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.training.argument_utils import ArgumentGroupFactory

    selected = {
        "dsa_indexer_topk_backend",
        "dsa_indexer_topk_deterministic",
        "dsa_indexer_topk_tie_break",
    }
    parser = ArgumentParser()
    excluded = [field.name for field in fields(TransformerConfig) if field.name not in selected]
    ArgumentGroupFactory(TransformerConfig, exclude=excluded).build_group(parser)
    defaults = parser.parse_args([])
    assert defaults.dsa_indexer_topk_backend is None
    assert defaults.dsa_indexer_topk_deterministic is False
    assert defaults.dsa_indexer_topk_tie_break == 0
    parsed = parser.parse_args(
        [
            "--dsa-indexer-topk-backend",
            "flashinfer",
            "--dsa-indexer-topk-deterministic",
            "--dsa-indexer-topk-tie-break",
            "2",
        ]
    )
    assert parsed.dsa_indexer_topk_backend == "flashinfer"
    assert parsed.dsa_indexer_topk_deterministic is True
    assert parsed.dsa_indexer_topk_tie_break == 2


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"dsa_indexer_loss_coeff": 0.1}, "auxiliary loss"),
        ({"dsa_kernel_backend": "tilelang"}, "requires dsa_kernel_backend"),
        ({"dsa_indexer_scoring_relu": False}, "requires dsa_kernel_backend"),
        ({"dsa_kernel_backend": "none", "dsa_indexer_loss_coeff": 0.1}, None),
        ({"attention_backend": AttnBackend.unfused, "dsa_indexer_loss_coeff": 0.1}, None),
    ],
)
def test_explicit_fused_topk_config_rejects_unsupported_training(monkeypatch, overrides, error):
    from megatron.core.transformer import transformer_config

    monkeypatch.setattr(
        transformer_config, "_validate_dsa_kernel_backend_dependencies", lambda _: None
    )
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_kernel_backend="cudnn",
        dsa_indexer_topk_backend="torch",
        dsa_indexer_loss_coeff=0.0,
    )
    kwargs.update(overrides)
    if error:
        with pytest.raises(ValueError, match=error):
            transformer_config.TransformerConfig(**kwargs)
    else:
        assert transformer_config.TransformerConfig(**kwargs).dsa_indexer_loss_coeff == 0.1


@pytest.mark.parametrize("shape", [(4, 128), (2, 2, 128)])
@pytest.mark.parametrize("tie_break", [0, 1, 2])
@pytest.mark.parametrize("deterministic", [False, True])
def test_flashinfer_selector_matches_public_api(shape, tie_break, deterministic):
    flashinfer = _require_flashinfer()
    # Both a tied k-boundary and rows with fewer than k valid keys.
    scores = torch.zeros(shape, device="cuda", dtype=torch.float32)
    scores.reshape(-1, 128)[1, 5:] = -torch.inf
    scores.reshape(-1, 128)[2, :] = -torch.inf
    config = _config(tie_break=tie_break)
    config.dsa_indexer_topk_deterministic = deterministic
    indices = select_dsa_topk(scores, 32, config)
    values_ref, indices_ref = flashinfer.top_k(
        scores.reshape(-1, 128),
        32,
        sorted=False,
        deterministic=deterministic,
        tie_break=tie_break,
        dsa_graph_safe=True,
    )
    expected = indices_ref.to(torch.int32).masked_fill(values_ref == -torch.inf, -1)
    # No order guarantee is imposed when deterministic=False/tie_break=0.
    if deterministic or tie_break:
        torch.testing.assert_close(indices, expected.reshape(*shape[:-1], 32), rtol=0, atol=0)
    assert indices.dtype == torch.int32
    assert (indices.reshape(-1, 32)[2] == -1).all()
    if tie_break:
        expected_boundary = torch.arange(32, device="cuda")
        if tie_break == 2:
            expected_boundary += 96
        torch.testing.assert_close(
            indices.reshape(-1, 32)[0].sort().values.long(), expected_boundary
        )


def test_torch_selector_keeps_torch_topk_ties_and_masks():
    scores = torch.tensor([[[1.0, 1.0, 1.0, 0.0], [-torch.inf, 3.0, -torch.inf, -torch.inf]]])
    values, indices = torch.topk(scores, 2, dim=-1)
    expected = indices.to(torch.int32).masked_fill(values == -torch.inf, -1)
    torch.testing.assert_close(select_dsa_topk(scores, 2, _config("torch")), expected)


@pytest.mark.parametrize("tie_break", [1, 2])
@pytest.mark.parametrize("packed", [False, True], ids=["bshd", "packed-thd"])
@pytest.mark.parametrize("nonzero", [False, True], ids=["tied", "nonzero"])
def test_cudnn_score_hook_uses_external_tie_policy(monkeypatch, tie_break, packed, nonzero):
    _require_flashinfer()
    pytest.importorskip("cudnn")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN DSA requires SM90+")
    seq, batch, heads, dim, topk = 128, 1 if packed else 2, 64, 128, 32
    q, k, weights = _indexer_inputs(seq, seq, batch, heads, dim, nonzero)
    starts = torch.zeros(seq, dtype=torch.int32, device="cuda")
    if packed:
        starts[64:] = 64
    ends = torch.arange(1, seq + 1, dtype=torch.int32, device="cuda")

    def internal_topk_must_not_run(*args, **kwargs):
        raise AssertionError("The internal cuDNN top-k bypasses the requested tie policy")

    monkeypatch.setattr(
        dsa_cudnn_kernels, "_indexer_top_k_wrapper_chunked", internal_topk_must_not_run
    )
    original_score = dsa_cudnn_kernels._indexer_forward_wrapper_with_warning
    score_calls = []

    def score_hook(*args, **kwargs):
        result = original_score(*args, **kwargs)
        score_calls.append(result["scores"])
        return result

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_forward_wrapper_with_warning", score_hook)
    config = _config(tie_break=tie_break)
    indices, lengths = dsa_kernels.run_fused_qk_topk(
        config, q, k, weights, topk, starts, ends, block_size=128, cp_size=1
    )
    assert score_calls, "The cuDNN score hook must execute"
    assert lengths is None
    expected_scores, valid = _reference_scores(q, k, weights, starts, ends)
    torch.testing.assert_close(
        score_calls[0].masked_fill(~valid[None, :, :], -torch.inf), expected_scores, rtol=0, atol=0
    )
    expected = select_dsa_topk(expected_scores, topk, config)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tie_break", [1, 2])
@pytest.mark.parametrize("nonzero", [False, True], ids=["tied", "nonzero"])
def test_external_score_chunks_preserve_global_mask_offsets(monkeypatch, tie_break, nonzero):
    _require_flashinfer()
    pytest.importorskip("cudnn")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cuDNN DSA requires SM90+")
    seq, heads, dim, topk = 128, 64, 128, 16
    positions = torch.arange(seq, device="cuda")
    q, k, weights = _indexer_inputs(seq, seq, 1, heads, dim, nonzero)
    starts = torch.where(positions < 64, 0, 64).to(torch.int32)
    ends = (positions + 1).to(torch.int32)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_score_chunk_rows", lambda *_: 17)

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit fused chunks must use cuDNN scoring and external top-k")

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_top_k_wrapper_chunked", forbidden)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_compute_indexer_scores_chunk_with_global_rows", forbidden
    )
    original_score = dsa_cudnn_kernels._indexer_forward_wrapper_with_warning
    score_calls = []

    def score_hook(*args, **kwargs):
        result = original_score(*args, **kwargs)
        score_calls.append((kwargs.get("q_causal_offsets"), result["scores"]))
        return result

    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_forward_wrapper_with_warning", score_hook)
    config = _config(tie_break=tie_break)
    indices, lengths = dsa_kernels.run_fused_qk_topk(
        config, q, k, weights, topk, starts, ends, block_size=128, cp_size=1
    )
    expected_scores, valid = _reference_scores(q, k, weights, starts, ends)
    assert len(score_calls) == 8
    for chunk, (offsets, scores) in enumerate(score_calls):
        row_start = chunk * 17
        if row_start == 0:
            assert offsets is None
        else:
            torch.testing.assert_close(offsets, torch.full_like(offsets, row_start))
        row_end = row_start + scores.size(1)
        torch.testing.assert_close(
            scores.masked_fill(~valid[None, row_start:row_end], -torch.inf),
            expected_scores[:, row_start:row_end],
            rtol=0,
            atol=0,
        )
    expected = select_dsa_topk(expected_scores, topk, config)
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    assert lengths is None


@pytest.mark.parametrize("layout", ["plain", "padded"])
def test_skip_layer_reuses_external_selection_without_recomputing(monkeypatch, layout):
    config = _config("torch")
    config.dsa_indexer_topk = 2
    config.dsa_indexer_topk_freq = 4
    config.dsa_indexer_skip_topk_offset = 1
    config.sequence_parallel = False
    config.dsa_indexer_loss_coeff = 0.0
    config.dsa_indexer_use_sparse_loss = False
    config.calculate_per_token_loss = False
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
        pg_collection=SimpleNamespace(),
    )
    mask = torch.ones(1)
    packed = None
    if layout != "plain":
        cu = torch.tensor([0, 3], dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4
        )
        if layout == "padded":
            packed.real_token_mask_q = torch.tensor([True, True, True, False])
    scores = torch.zeros(1, 4, 4)
    scores.masked_fill_(torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1), -torch.inf)
    selected = select_dsa_topk(scores, 2, config)
    attention._get_index_share_topk_holder(packed, mask)[1] = selected
    forwarded = []

    def sparse_attention(**kwargs):
        forwarded.append(kwargs["topk_indices"])
        return kwargs["query"].flatten(2)

    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.dsa._run_sparse_attention",
        sparse_attention,
    )
    tensors = [torch.randn(4, 1, 2, 8) for _ in range(3)]
    attention(
        *tensors,
        mask,
        torch.randn(4, 1, 16),
        torch.randn(4, 1, 8),
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=packed,
    )
    assert forwarded == [selected]
    assert forwarded[0] is selected


@pytest.mark.parametrize("segments", [1, 2])
@pytest.mark.parametrize("tie_break", [1, 2])
@pytest.mark.parametrize("nonzero", [False, True], ids=["tied", "nonzero"])
def test_external_selection_preserves_fused_packed_cp_scores(
    monkeypatch, segments, tie_break, nonzero
):
    _require_flashinfer()
    pytest.importorskip("cudnn")
    seq, local_seq, heads, dim, topk = 128, 64, 64, 128, 32
    q, k, weights = _indexer_inputs(local_seq, seq, 1, heads, dim, nonzero)
    seq_len = seq // segments
    positions = torch.cat(
        [
            torch.arange(start + seq_len // 4, start + 3 * seq_len // 4, device="cuda")
            for start in range(0, seq, seq_len)
        ]
    )
    starts = (positions // seq_len * seq_len).to(torch.int32)
    ends = (positions + 1).to(torch.int32)
    cu = torch.arange(0, seq + 1, seq_len, dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Packed CP must retain fused scoring and external selection")

    monkeypatch.setattr(
        dsa_cudnn_kernels, "_compute_indexer_scores_chunk_with_global_rows", forbidden
    )
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_top_k_wrapper_chunked", forbidden)
    config = _config(tie_break=tie_break)
    indices, lengths = dsa_kernels.run_fused_qk_topk(
        config,
        q,
        k,
        weights,
        topk,
        starts,
        ends,
        block_size=128,
        cp_size=2,
        use_local_indexer_varlen=True,
        single_packed_thd_sequence=segments == 1,
        local_packed_cp_rank=1,
        packed_seq_params=packed,
    )
    scores, _ = _reference_scores(q, k, weights, starts, ends)
    expected = select_dsa_topk(scores, topk, config)
    # Unsorted FlashInfer output order may depend on the segment's score width;
    # the selected original indices and boundary tie policy must be identical.
    torch.testing.assert_close(indices.sort().values, expected.sort().values, rtol=0, atol=0)
    assert lengths is None


def test_multi_packed_cp_local_tp_slice_rejects_unsupported_layout(monkeypatch):
    q = torch.zeros(32, 1, 2, 8)
    k = torch.zeros(128, 1, 8)
    weights = torch.ones(32, 1, 2)
    positions = torch.cat((torch.arange(32, 48), torch.arange(80, 96)))
    starts = (positions // 64 * 64).to(torch.int32)
    ends = (positions + 1).to(torch.int32)
    cu = torch.tensor([0, 64, 128], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=64, max_seqlen_kv=64
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Whole-CP cu_seqlens cannot describe a TP-local query slice")

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_indexer_topk_multi_packed_cp_thd", forbidden)
    monkeypatch.setattr(
        dsa_cudnn_kernels, "_compute_indexer_scores_chunk_with_global_rows", forbidden
    )
    with pytest.raises(RuntimeError, match="query-position layout"):
        dsa_kernels.run_fused_qk_topk(
            _config("torch"),
            q,
            k,
            weights,
            16,
            starts,
            ends,
            block_size=128,
            cp_size=2,
            use_local_indexer_varlen=True,
            packed_seq_params=packed,
            local_packed_cp_rank=1,
            local_packed_cp_query_start=16,
            local_packed_cp_query_len=64,
        )


@pytest.mark.parametrize(
    "capability,row_layout",
    [
        pytest.param((9, 0), "mixed", id="sm90-mixed"),
        pytest.param((9, 0), "empty", id="sm90-empty"),
        pytest.param((10, 0), "nonempty", id="sm100-nonempty"),
        pytest.param((10, 0), "mixed", id="sm100-mixed"),
        pytest.param((10, 3), "empty", id="sm103-empty"),
        pytest.param((10, 1), "mixed", id="unaudited-sm101-mixed"),
    ],
)
def test_split_attention_backward_uses_device_capability(monkeypatch, capability, row_layout):
    direct = capability in ((10, 0), (10, 3))
    seen = []

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(q, kv, out, dO, lse, sink, indices, **kwargs):
            lengths = kwargs["topk_length"]
            seen.append((q.size(0), lengths.clone()))
            dq = torch.ones_like(q).masked_fill((lengths == 0)[:, None, None], 0)
            return {"dq": dq, "dkv": torch.zeros_like(kv)}

    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA)

    def fake_forward(q, kv, indices, softmax_scale, d_v, attn_sink, topk_length):
        return torch.zeros(q.size(0), q.size(1), d_v), torch.zeros(q.size(0), q.size(1))

    monkeypatch.setattr(dsa_cudnn_kernels, "_dsa_fwd_flash_mla", fake_forward)
    query = torch.zeros(2, 1, 1, 512, requires_grad=True)
    key = torch.zeros(4, 1, 1, 512, requires_grad=True)
    indices = torch.tensor([[[0, -1], [1, 0]]], dtype=torch.int32)
    if row_layout != "nonempty":
        indices[:, 0] = -1
    if row_layout == "empty":
        indices[:, 1] = -1
    lengths = (indices >= 0).sum(-1).flatten().to(torch.int32)
    output = dsa_kernels.run_fused_absorbed_sparse_attention(
        _config("torch"), query, key, indices, 1.0, 512
    )
    original_nonzero = torch.nonzero
    original_index_select = torch.Tensor.index_select
    compaction_calls = []
    devices = []

    def checked_nonzero(*args, **kwargs):
        assert not direct, "Audited SM100-family kernels must not synchronize to compact empty rows"
        compaction_calls.append("nonzero")
        return original_nonzero(*args, **kwargs)

    def checked_index_select(tensor, *args, **kwargs):
        assert not direct, "Audited SM100-family kernels must receive the original rows"
        compaction_calls.append("index_select")
        return original_index_select(tensor, *args, **kwargs)

    def device_sm(device):
        devices.append(device)
        return capability

    with monkeypatch.context() as patch:
        # Exercise the CUDA-only dispatch using CPU tensor arithmetic and a GPU kernel double.
        patch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
        patch.setattr(dsa_cudnn_kernels, "_device_sm", device_sm)
        patch.setattr(dsa_cudnn_kernels, "_get_head_padding", lambda heads: heads)
        patch.setattr(torch, "nonzero", checked_nonzero)
        patch.setattr(torch.Tensor, "index_select", checked_index_select)
        output.sum().backward()

    assert devices == [query.device.index]
    assert ("nonzero" in compaction_calls) is (not direct)
    assert ("index_select" in compaction_calls) is (not direct)
    expected_lengths = (
        lengths if direct else torch.cat((lengths[lengths > 0], torch.ones(1, dtype=torch.int32)))
    )
    assert seen[0][0] == expected_lengths.numel()
    torch.testing.assert_close(seen[0][1], expected_lengths)
    expected_grad = (lengths > 0).to(query.dtype).view(2, 1, 1, 1).expand_as(query)
    torch.testing.assert_close(query.grad, expected_grad)
    torch.testing.assert_close(key.grad, torch.zeros_like(key))


@pytest.mark.parametrize("row_layout", ["mixed", "empty"])
def test_sm100_split_attention_empty_rows_match_filtered_reference(row_layout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("This numerical regression requires audited SM100/SM103 cuDNN sparse attention")
    pytest.importorskip("cudnn")
    dsa_cudnn_kernels._ensure_dsa_namespace()
    pytest.importorskip("flash_mla")
    torch.manual_seed(713)
    sq, sk, heads, dim, value_dim, topk = 8, 128, 32, 576, 512, 512
    query = torch.randn(sq, 1, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    key = torch.randn(sk, 1, 1, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
    lengths_list = [4, 67, 128, 9, 31, 85, 3, 64]
    if row_layout == "mixed":
        lengths_list[::2] = [0] * (sq // 2)
    elif row_layout == "empty":
        lengths_list = [0] * sq
    lengths = torch.tensor([lengths_list], dtype=torch.int32, device="cuda")
    indices = torch.full((1, sq, topk), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths_list):
        indices[0, row, :length] = torch.arange(length, dtype=torch.int32, device="cuda")

    output = dsa_kernels.run_fused_absorbed_sparse_attention(
        _config("torch"), query, key, indices, dim**-0.5, value_dim, topk_length=lengths
    )
    assert output is not None
    grad_output = torch.randn_like(output)
    assert grad_output.abs().sum() > 0
    output.backward(grad_output)
    expected_output = torch.zeros_like(output)
    expected_query_grad = torch.zeros_like(query)
    expected_key_grad = torch.zeros_like(key)
    valid_rows = torch.tensor(
        [row for row, length in enumerate(lengths_list) if length], device="cuda", dtype=torch.long
    )
    if valid_rows.numel():
        # Physically remove empty queries; they cannot affect the reference kernel's dKV reduction.
        reference_query = query.detach().index_select(0, valid_rows).requires_grad_()
        reference_key = key.detach().clone().requires_grad_()
        reference_output = dsa_kernels.run_fused_absorbed_sparse_attention(
            _config("torch"),
            reference_query,
            reference_key,
            indices.index_select(1, valid_rows),
            dim**-0.5,
            value_dim,
            topk_length=lengths.index_select(1, valid_rows),
        )
        reference_output.backward(grad_output.index_select(0, valid_rows))
        expected_output.index_copy_(0, valid_rows, reference_output.detach())
        expected_query_grad.index_copy_(0, valid_rows, reference_query.grad)
        expected_key_grad.copy_(reference_key.grad)
        assert expected_query_grad.abs().sum() > 0
        assert expected_key_grad.abs().sum() > 0
    for actual, expected in (
        (output, expected_output),
        (query.grad, expected_query_grad),
        (key.grad, expected_key_grad),
    ):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-3)
    empty_rows = lengths.flatten() == 0
    assert torch.count_nonzero(query.grad[empty_rows]) == 0
    assert torch.count_nonzero(output[empty_rows]) == 0
    if row_layout == "empty":
        assert torch.count_nonzero(key.grad) == 0


@pytest.mark.parametrize("sparse_loss", [False, True])
def test_reference_indexer_loss_retains_external_indices_and_backward(sparse_loss):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)
    q = torch.randn(8, 1, 2, 4, device=device, requires_grad=True)
    k = torch.randn(8, 1, 4, device=device, requires_grad=True)
    weights = torch.randn(8, 1, 2, device=device, requires_grad=True)
    query = torch.randn(8, 1, 2, 4, device=device)
    key = torch.randn(8, 1, 2, 4, device=device)
    mask = torch.triu(torch.full((1, 8, 8), -torch.inf, device=device), diagonal=1)
    config = _config("torch")
    config.dsa_kernel_backend = "none"
    pg = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1))
    _, expected = fused_qk_topk_naive(q, k, weights, 4, mask=mask, topk_config=config)
    indices, loss = FusedDSAIndexerLoss.apply(
        q,
        weights,
        k,
        query,
        key,
        0.5,
        4,
        0.1,
        mask,
        sparse_loss,
        pg,
        None,
        None,
        None,
        None,
        False,
        True,
        None,
        config,
    )
    torch.testing.assert_close(indices, expected)
    loss.backward()
    for tensor in (q, k, weights):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
