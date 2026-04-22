# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import torch

from megatron.core.extensions import sglang as sglang_ext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.attention import _is_ulysses_cp
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class _FakeCPGroup:
    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _DummyAttention(torch.nn.Module):
    kind = "dummy"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.current_max_attn_logits = None

    def forward(self, *args, **kwargs):
        raise AssertionError("This test should not execute the backend attention kernel.")


class _DummyUlyssesAttention(_DummyAttention):
    kind = "ulysses"


class _DummyFallbackAttention(_DummyAttention):
    kind = "fallback"


def _make_config(*, context_parallel_size: int, cp_comm_type=None) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        kv_channels=4,
        num_query_groups=2,
        attention_dropout=0.0,
        apply_rope_fusion=False,
        context_parallel_size=context_parallel_size,
        cp_comm_type=cp_comm_type,
    )


def test_is_ulysses_cp_detects_a2a_mode():
    assert not _is_ulysses_cp(_make_config(context_parallel_size=1, cp_comm_type="a2a"))
    assert not _is_ulysses_cp(_make_config(context_parallel_size=2, cp_comm_type="p2p"))
    assert _is_ulysses_cp(_make_config(context_parallel_size=2, cp_comm_type="a2a"))
    assert _is_ulysses_cp(_make_config(context_parallel_size=2, cp_comm_type=["a2a", "p2p"]))


def test_apply_rotary_pos_emb_ulysses_matches_unsplit_sequence():
    config = _make_config(context_parallel_size=2, cp_comm_type="a2a")
    t = torch.randn(4, 2, 4, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    freqs = torch.randn(4, 1, 1, 4, dtype=torch.float32)

    ulysses_output = apply_rotary_pos_emb(
        t,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(size=2, rank=1),
        ulysses_cp=True,
    )
    unsplit_output = apply_rotary_pos_emb(
        t,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        cp_group=_FakeCPGroup(size=1, rank=0),
        ulysses_cp=False,
    )

    torch.testing.assert_close(ulysses_output, unsplit_output)


def test_sglang_core_attention_dispatches_ulysses_backend(monkeypatch):
    monkeypatch.setattr(sglang_ext, "SGLangFlashAttention", _DummyUlyssesAttention)
    monkeypatch.setattr(sglang_ext, "DotProductAttention", _DummyFallbackAttention)

    attn = sglang_ext.SGLangCoreAttention(
        config=_make_config(context_parallel_size=2, cp_comm_type="a2a"),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="a2a",
        pg_collection=None,
    )

    assert isinstance(attn.impl, _DummyUlyssesAttention)


def test_sglang_core_attention_falls_back_outside_ulysses(monkeypatch):
    monkeypatch.setattr(sglang_ext, "SGLangFlashAttention", _DummyUlyssesAttention)
    monkeypatch.setattr(sglang_ext, "DotProductAttention", _DummyFallbackAttention)

    attn = sglang_ext.SGLangCoreAttention(
        config=_make_config(context_parallel_size=2, cp_comm_type="p2p"),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="p2p",
        pg_collection=None,
    )

    assert isinstance(attn.impl, _DummyFallbackAttention)
