from __future__ import annotations

import inspect
import math
import os
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from .cp_layout import SGLangUlyssesCPLayout
from megatron.core.utils import divide

try:
    from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func

    HAVE_FA3_VARLEN = True
except ImportError:
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func

        HAVE_FA3_VARLEN = True
    except ImportError:
        HAVE_FA3_VARLEN = False
        fa3_varlen_func = None


_ragged_prefill_wrapper = None

# Scratch for flashinfer's deterministic ragged prefill (batch_prefill_tmp_v etc.). Sized to
# the TRAINING side's parallelism/window, independent of sglang's rollout-side
# SGLANG_FLASHINFER_WORKSPACE_SIZE: the requirement scales with per-rank heads
# (total_heads / TP) and batching, which differ between train and rollout, and the size is
# scratch that does not affect parity. 256 MiB overflowed the canonical config (8192 window,
# batch 256, ~406 MiB), so default 1 GiB.
_WORKSPACE_SIZE = int(
    os.environ.get("MEGATRON_TRUE_ON_POLICY_FLASHINFER_WORKSPACE_SIZE", 1024 * 1024 * 1024)
)


def _fmha_backend(device) -> str:
    """The flashinfer prefill backend, matching sglang's rollout-side choice EXACTLY so training
    and rollout run the IDENTICAL kernel (contractual parity, not coincidental). sglang's
    flashinfer_backend.py uses "cutlass" on SM100 (Blackwell), "auto" otherwise. Omitting the
    backend only accidentally matched sglang on Blackwell (both resolved to cutlass) and diverged
    on Hopper (train "auto" default != rollout "auto" kernel -> abs_diff 0.017)."""
    return "cutlass" if torch.cuda.get_device_capability(device)[0] >= 10 else "auto"


def _get_ragged_prefill_wrapper(device):
    """One shared deterministic ragged-prefill wrapper (Blackwell / flashinfer)."""
    global _ragged_prefill_wrapper
    if _ragged_prefill_wrapper is None:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        ws = torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
        _ragged_prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            ws, kv_layout="NHD", backend=_fmha_backend(device)
        )
    return _ragged_prefill_wrapper


class _FlashinferRaggedAttn(torch.autograd.Function):
    """Deterministic flashinfer ragged prefill. Forward = flashinfer (matches sglang
    inference bitwise). flashinfer's run is forward-only, so backward is the fused
    mem-efficient attention backward dispatched per packed segment (needn't bit-match
    inference)."""

    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, num_q_heads, num_kv_heads, head_dim, scale):
        w = _get_ragged_prefill_wrapper(q.device)
        # Match sglang's call exactly: plan WITHOUT sm_scale, apply sm_scale at forward-time
        # via .forward (sglang uses fast_prefill_plan (no sm_scale) + .forward(..., sm_scale=...)).
        w.plan(
            cu_q, cu_k, num_q_heads, num_kv_heads, head_dim,
            causal=True,
            q_data_type=q.dtype, kv_data_type=k.dtype, fixed_split_size=4096,
        )
        out = w.forward(q, k, v, causal=True, sm_scale=scale, logits_soft_cap=0.0)
        ctx.save_for_backward(q, k, v, cu_q)
        ctx.num_q_heads, ctx.num_kv_heads, ctx.scale = num_q_heads, num_kv_heads, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Fused mem-efficient attention backward, dispatched DIRECTLY per packed segment.
        # Two deliberate properties:
        #   * O(L) memory: each segment uses is_causal=True with NO dense [L, L] mask
        #     tensor, so the fused backend tiles the attention (never materializes the
        #     score matrix). Peak memory is O(max segment length), not O(seq_len^2) --
        #     essential for long single segments (a full-length sample) and long context.
        #   * NO torch.autograd.grad: dispatching aten's fused backward kernel directly is
        #     reentrant-safe, unlike a recompute-then-autograd.grad, which nests a second
        #     autograd traversal inside the Function.backward. Same rule as the norm and
        #     row-linear backwards.
        # Flashinfer is forward-only and FA3/flash-attn varlen is Hopper-only, so on
        # Blackwell the portable fused varlen backward is aten's mem-efficient kernel run
        # per segment with is_causal (the packed sequence's segments are single samples).
        q, k, v, cu_q = ctx.saved_tensors
        H, HKV, scale = ctx.num_q_heads, ctx.num_kv_heads, ctx.scale
        rep = H // HKV
        head_dim = q.shape[-1]
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        cu = cu_q.long()
        eff_attn = torch.ops.aten._scaled_dot_product_efficient_attention
        eff_attn_bwd = torch.ops.aten._scaled_dot_product_efficient_attention_backward
        for i in range(cu.numel() - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            seg_len = e - s
            if seg_len == 0:
                continue
            # [1, H, L, D]; expand GQA kv heads to the query-head count for the dense kernel.
            qs = q[s:e].transpose(0, 1).unsqueeze(0)
            ks = (k[s:e].repeat_interleave(rep, dim=1) if rep > 1 else k[s:e]).transpose(0, 1).unsqueeze(0)
            vs = (v[s:e].repeat_interleave(rep, dim=1) if rep > 1 else v[s:e]).transpose(0, 1).unsqueeze(0)
            gos = grad_out[s:e].transpose(0, 1).unsqueeze(0)
            # is_causal=True, attn_bias=None -> no O(L^2) mask; compute_log_sumexp=True for the bwd.
            out, lse, philox_seed, philox_offset = eff_attn(qs, ks, vs, None, True, 0.0, True, scale=scale)
            dqi, dki, dvi, _ = eff_attn_bwd(
                gos, qs, ks, vs, None, out, lse, philox_seed, philox_offset,
                0.0, [True, True, True, False], True, scale=scale,
            )
            dq[s:e] = dqi.squeeze(0).transpose(0, 1)
            dki = dki.squeeze(0).transpose(0, 1)  # [L, H, D] (expanded query heads)
            dvi = dvi.squeeze(0).transpose(0, 1)
            if rep > 1:  # reduce expanded query heads back to the kv-head count (GQA)
                dk[s:e] = dki.reshape(seg_len, HKV, rep, head_dim).sum(dim=2)
                dv[s:e] = dvi.reshape(seg_len, HKV, rep, head_dim).sum(dim=2)
            else:
                dk[s:e] = dki
                dv[s:e] = dvi
        return dq, dk, dv, None, None, None, None, None, None


class SGLangFlashAttention(MegatronModule):
    """SGLang-compatible FA3 attention path with packed-sequence support."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.current_max_attn_logits = None
        self.cp_size = config.context_parallel_size
        self.cp_comm_type = cp_comm_type

        if self.cp_size > 1:
            assert cp_comm_type == "a2a", (
                "SGLangFlashAttention currently supports packed CP only with Ulysses "
                f"(cp_comm_type='a2a'), got {cp_comm_type!r}"
            )
            assert pg_collection is not None and hasattr(
                pg_collection, "cp"
            ), "ProcessGroupCollection must provide a CP group for SGLangFlashAttention"
            self.cp_layout = SGLangUlyssesCPLayout(pg_collection.cp, self.cp_size)
        else:
            self.cp_layout = None

        kv_channels = config.kv_channels
        assert kv_channels is not None, "kv_channels must be set"
        projection_size = kv_channels * config.num_attention_heads
        tp_world_size = pg_collection.tp.size() if pg_collection is not None else 1

        self.hidden_size_per_partition = divide(projection_size, tp_world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, tp_world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, tp_world_size)

        self.softmax_scale = (
            1.0 / math.sqrt(self.hidden_size_per_attention_head)
            if softmax_scale is None
            else softmax_scale
        )
        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.attention_dropout = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )
        # Attention backend is an explicit flag (TOP_ATTN_BACKEND), NOT a GPU arch-gate:
        # miles' TrueOnPolicyKernelPolicy sets it to match sglang's --sglang-attention-backend
        # so both engines agree. "fa3" (Hopper, faster) or "flashinfer" (Blackwell / no-FA3,
        # and usable on Hopper for e2e parity tests). flashinfer does GQA natively (no KV repeat)
        # + takes fixed_split_size=4096 to match sglang's deterministic ragged prefill bitwise.
        # Fallback "fa3" applies only when run standalone without the flag set.
        self.attention_backend = os.environ.get("TOP_ATTN_BACKEND", "fa3")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        del attention_mask

        assert attention_bias is None, "Attention bias is not supported for SGLangFlashAttention"
        assert (
            attn_mask_type is None or attn_mask_type == AttnMaskType.causal
        ), "Only causal attention is supported for SGLangFlashAttention"
        if self.attention_backend == "fa3" and (not HAVE_FA3_VARLEN or fa3_varlen_func is None):
            raise ImportError("Flash Attention 3 varlen is required for SGLangFlashAttention")

        is_packed = packed_seq_params is not None
        input_ndim = query.dim()
        head_dim_idx = -2 if is_packed and input_ndim >= 3 else 2
        if self.attention_backend != "flashinfer" and (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1
        ):
            repeat_factor = (
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            )
            key = key.repeat_interleave(repeat_factor, dim=head_dim_idx)
            value = value.repeat_interleave(repeat_factor, dim=head_dim_idx)

        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if is_packed:
            if input_ndim == 3:
                total_tokens, _, _ = query.shape
            else:
                total_tokens, batch_size, _, _ = query.shape
                assert batch_size == 1, f"Packed sequences should use batch=1, got {batch_size}"
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)

            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_k = packed_seq_params.cu_seqlens_kv
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_k = packed_seq_params.max_seqlen_kv
            if cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            if cu_seqlens_k.dtype != torch.int32:
                cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        else:
            seq_len, batch_size, num_heads, head_dim = query.shape
            key_seq_len = key.shape[0]
            query = query.transpose(0, 1).reshape(batch_size * seq_len, num_heads, head_dim)
            key = key.transpose(0, 1).reshape(batch_size * key_seq_len, num_heads, head_dim)
            value = value.transpose(0, 1).reshape(batch_size * key_seq_len, num_heads, head_dim)
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * key_seq_len,
                key_seq_len,
                dtype=torch.int32,
                device=query.device,
            )
            max_seqlen_q = seq_len
            max_seqlen_k = key_seq_len

        if self.cp_size > 1:
            assert is_packed, "SGLang Ulysses CP currently requires packed THD inputs."
            assert self.cp_layout is not None
            local_tokens, local_heads, _ = query.shape
            query = self.cp_layout.sequence_to_head_parallel(query, cu_seqlens_q)
            key = self.cp_layout.sequence_to_head_parallel(key, cu_seqlens_k)
            value = self.cp_layout.sequence_to_head_parallel(value, cu_seqlens_k)

        if self.attention_backend == "flashinfer":
            # Deterministic ragged prefill matching sglang's flashinfer TOP path (GQA-native,
            # fixed_split_size=4096); autograd.Function supplies the backward (flashinfer is fwd-only).
            output = _FlashinferRaggedAttn.apply(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                self.num_attention_heads_per_partition,
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
                self.softmax_scale,
            )
        else:
            sig = inspect.signature(fa3_varlen_func)
            fa3_kwargs = {
                "q": query,
                "k": key,
                "v": value,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "softmax_scale": self.softmax_scale,
                "causal": True,
            }
            if "dropout_p" in sig.parameters:
                fa3_kwargs["dropout_p"] = self.attention_dropout if self.training else 0.0
            if "window_size" in sig.parameters:
                fa3_kwargs["window_size"] = (-1, -1)
            if "softcap" in sig.parameters:
                fa3_kwargs["softcap"] = 0.0
            if "return_attn_probs" in sig.parameters:
                fa3_kwargs["return_attn_probs"] = False
            if "return_softmax_lse" in sig.parameters:
                fa3_kwargs["return_softmax_lse"] = False
            if "num_splits" in sig.parameters:
                fa3_kwargs["num_splits"] = 1
            if (
                "deterministic" in sig.parameters
                and os.environ.get("MEGATRON_TRUE_ON_POLICY_FA3_DETERMINISTIC_BWD") == "1"
            ):
                fa3_kwargs["deterministic"] = True

            output = fa3_varlen_func(**fa3_kwargs)
            if isinstance(output, tuple):
                output = output[0]

        if self.cp_size > 1:
            assert self.cp_layout is not None
            output = self.cp_layout.head_to_sequence_parallel(
                output, cu_seqlens_q, local_tokens, local_heads
            )

        if is_packed:
            if input_ndim == 3:
                return output.view(total_tokens, self.hidden_size_per_partition)
            return output.view(total_tokens, 1, self.hidden_size_per_partition)

        output = output.view(batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(0, 1)
        return output.reshape(seq_len, batch_size, self.hidden_size_per_partition)


class SGLangCoreAttention(MegatronModule):
    """Core-attention wrapper used by the SGLang backend."""

    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        super().__init__(config=config)
        self.impl = SGLangFlashAttention(*args, **kwargs)
        self._current_max_attn_logits = getattr(self.impl, "current_max_attn_logits", None)

    @property
    def current_max_attn_logits(self):
        return getattr(self.impl, "current_max_attn_logits", self._current_max_attn_logits)

    @current_max_attn_logits.setter
    def current_max_attn_logits(self, value):
        self._current_max_attn_logits = value
        if hasattr(self.impl, "current_max_attn_logits"):
            self.impl.current_max_attn_logits = value

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)
