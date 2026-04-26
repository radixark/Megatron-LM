# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
import math
import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.matmul_tp_inv import sglang_reference_matmul
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
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


def enable_sglang_batch_invariant_mode() -> None:
    """Enable deterministic runtime knobs expected by the SGLang backend."""

    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"


_USE_SGLANG_ROPE = False


def enable_sglang_rope() -> None:
    """Enable the SGLang-compatible RoPE path used by dense true-on-policy."""

    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = True


def disable_sglang_rope() -> None:
    global _USE_SGLANG_ROPE
    _USE_SGLANG_ROPE = False


def is_sglang_rope_enabled() -> bool:
    return _USE_SGLANG_ROPE


def sglang_apply_rotary_pos_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool = True,
) -> Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float()
    sin = sin.float()

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim < x.shape[-1]:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = sglang_apply_rotary_pos_emb(x_rot, cos, sin, is_neox_style)
        return torch.cat((x_rot, x_pass), dim=-1).to(orig_dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1).to(orig_dtype)

    return torch.stack((o1, o2), dim=-1).flatten(-2).to(orig_dtype)


def sglang_apply_rotary_pos_emb_with_freqs(
    x: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    layer_number: Optional[int] = None,
) -> Tensor:
    del layer_number

    x_seq_len = x.shape[0]
    freqs_seq_len = freqs.shape[0]

    freqs_flat = freqs.squeeze(1).squeeze(1)
    head_dim = x.shape[-1]
    raw_angles = freqs_flat[..., : head_dim // 2]
    cos = torch.cos(raw_angles)
    sin = torch.sin(raw_angles)
    is_neox_style = not getattr(config, "rotary_interleaved", False)

    if x_seq_len == freqs_seq_len:
        return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)
    if freqs_seq_len < x_seq_len:
        x_valid = x[:freqs_seq_len]
        x_valid = sglang_apply_rotary_pos_emb(x_valid, cos, sin, is_neox_style)
        return torch.cat([x_valid, x[freqs_seq_len:]], dim=0)

    cos = cos[:x_seq_len]
    sin = sin[:x_seq_len]
    return sglang_apply_rotary_pos_emb(x, cos, sin, is_neox_style)


class SGLangColumnParallelLinear(ColumnParallelLinear):
    """Megatron local column-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
        _ensure_batch_invariant_mode_from_config(self.config)
        bias = kwargs.pop("bias", None)
        return sglang_reference_matmul(
            input,
            weight,
            bias,
            gradient_accumulation_fusion=kwargs.pop("gradient_accumulation_fusion"),
            allreduce_dgrad=kwargs.pop("allreduce_dgrad"),
            sequence_parallel=kwargs.pop("sequence_parallel"),
            grad_output_buffer=kwargs.pop("grad_output_buffer", None),
            wgrad_deferral_limit=kwargs.pop("wgrad_deferral_limit", None),
            tp_group=kwargs.pop("tp_group", None),
            row_parallel=False,
        )


class SGLangRowParallelLinear(RowParallelLinear):
    """Megatron local row-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
        _ensure_batch_invariant_mode_from_config(self.config)
        bias = kwargs.pop("bias", None)
        return sglang_reference_matmul(
            input,
            weight,
            bias,
            gradient_accumulation_fusion=kwargs.pop("gradient_accumulation_fusion"),
            allreduce_dgrad=kwargs.pop("allreduce_dgrad"),
            sequence_parallel=kwargs.pop("sequence_parallel"),
            grad_output_buffer=kwargs.pop("grad_output_buffer", None),
            wgrad_deferral_limit=kwargs.pop("wgrad_deferral_limit", None),
            tp_group=kwargs.pop("tp_group", None),
            row_parallel=True,
        )


def _ensure_batch_invariant_mode_from_config(config: TransformerConfig) -> None:
    if not getattr(config, "batch_invariant_mode", False):
        return

    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()


def _sglang_bias_dropout_add(x_with_bias, residual, prob, training):
    x, bias = x_with_bias
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    return out.float() + residual.float()


def get_sglang_bias_dropout_add(training, fused):
    del fused

    def _bias_dropout_add(x_with_bias, residual, prob):
        return _sglang_bias_dropout_add(x_with_bias, residual, prob, training)

    return _bias_dropout_add


class SGLangNorm(torch.nn.Module):
    """Norm wrapper with Megatron-compatible parameters and SGLang backend identity."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
        cast_x_before_out_mul: bool = True,
        override_orig_dtype: Optional[torch.dtype] = None,
        keep_weight_fp32: bool = True,
    ) -> None:
        super().__init__()

        del persist_layer_norm
        del normalization

        self.config = config
        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.normalization = config.normalization
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma or zero_centered_gamma
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.override_orig_dtype = override_orig_dtype
        self.keep_weight_fp32 = keep_weight_fp32

        if self.normalization == "LayerNorm":
            self.weight = torch.nn.Parameter(torch.empty(hidden_size))
            self.bias = torch.nn.Parameter(torch.empty(hidden_size))
            self.reset_parameters()
            setattr(self.bias, "sequence_parallel", config.sequence_parallel)
        elif self.normalization == "RMSNorm":
            if self.zero_centered_gamma:
                raise AssertionError("zero_centered_gamma is not supported with SGLang RMSNorm.")
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.register_parameter("bias", None)
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def reset_parameters(self) -> None:
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(self.weight)
        else:
            torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def _apply(self, fn):
        super()._apply(fn)
        if self.normalization == "RMSNorm" and self.keep_weight_fp32:
            self.weight.data = self.weight.data.float()
            if self.weight.grad is not None:
                self.weight.grad.data = self.weight.grad.data.float()
        return self

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.normalization == "LayerNorm":
            if residual is not None:
                x = x + residual
                if post_residual_addition is not None:
                    x = x + post_residual_addition
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            return F.layer_norm(x, self.hidden_size, weight, self.bias, self.eps)

        orig_dtype = self.override_orig_dtype or x.dtype
        x_float = x.float()
        if residual is not None:
            x_float = x_float + residual.float()
            if post_residual_addition is not None:
                x_float = x_float + post_residual_addition.float()
            residual = x_float.to(orig_dtype)

        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.cast_x_before_out_mul:
            output = self.weight.float() * output.to(orig_dtype)
        else:
            output = (output * self.weight.float()).to(orig_dtype)

        if residual is None:
            return output
        return output, residual


class SGLangQKRMSNorm(torch.nn.Module):
    """Q/K RMSNorm matching the SGLang true-on-policy dense path."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        super().__init__()

        del config
        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        try:
            from sglang.srt.server_args import get_global_server_args

            target = getattr(get_global_server_args(), "rl_on_policy_target", None)
        except Exception:
            target = None
        self.cast_x_before_out_mul = target in ("fsdp", "fsdp_tp", None)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        if os.environ.get("MEGATRON_ROPE_BF16", "0") == "1":
            return (x_float.to(orig_dtype) * self.weight.to(orig_dtype)).to(orig_dtype)

        if self.cast_x_before_out_mul:
            return self.weight.float() * x_float.to(orig_dtype)
        return (x_float * self.weight.float()).to(orig_dtype)


class SGLangFinalRMSNorm(torch.nn.Module):
    """Final block RMSNorm matching the SGLang dense path."""

    backend_name = "sglang"

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-6,
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "RMSNorm",
    ) -> None:
        super().__init__()

        del config
        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            x = x.contiguous()

        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            if post_residual_addition is not None:
                x = x + post_residual_addition
            residual = x.clone()

        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = self.weight * x_float.to(orig_dtype)

        if residual is not None:
            return output, residual
        return output


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
            self.cp_group = pg_collection.cp
        else:
            self.cp_group = None

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

    def _local_packed_lengths(self, cu_seqlens: Tensor, local_tokens: int) -> list[int]:
        """Return per-packed-sequence local CP lengths from global cu_seqlens."""
        global_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        local_lengths = []
        for length in global_lengths:
            assert length % self.cp_size == 0, (
                f"Ulysses CP requires padded sequence lengths divisible by cp_size; "
                f"got length={length}, cp_size={self.cp_size}"
            )
            local_lengths.append(length // self.cp_size)

        assert sum(local_lengths) == local_tokens, (
            f"Packed cu_seqlens do not match local CP shard length: "
            f"sum(local_lengths)={sum(local_lengths)}, local_tokens={local_tokens}"
        )
        return local_lengths

    def _sequence_to_head_parallel(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Ulysses CP all-to-all: local zigzag sequence shard -> full sequence, head shard."""
        local_tokens, num_heads, head_dim = x.shape
        assert num_heads % self.cp_size == 0, (
            f"num_heads={num_heads} must be divisible by cp_size={self.cp_size}"
        )
        local_lengths = self._local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(local_tokens, 1, num_heads * head_dim)
        hidden_per_rank = x.shape[-1] // self.cp_size
        rank_ordered = torch.cat(torch.split(x.reshape(local_tokens, -1), hidden_per_rank, dim=1), dim=0)
        rank_ordered = all_to_all(self.cp_group, rank_ordered)
        rank_ordered = rank_ordered.reshape(local_tokens * self.cp_size, 1, hidden_per_rank)

        per_source_offsets = []
        offset = 0
        for _ in range(self.cp_size):
            per_source_offsets.append(offset)
            offset += local_tokens

        sequential = []
        for seq_index, local_length in enumerate(local_lengths):
            assert local_length % 2 == 0, (
                f"Ulysses CP expects two equal zigzag chunks per rank; got local_length={local_length}"
            )
            chunk = local_length // 2
            seq_offset = sum(local_lengths[:seq_index])

            for source_rank in range(self.cp_size):
                start = per_source_offsets[source_rank] + seq_offset
                sequential.append(rank_ordered[start : start + chunk])
            for source_rank in range(self.cp_size - 1, -1, -1):
                start = per_source_offsets[source_rank] + seq_offset + chunk
                sequential.append(rank_ordered[start : start + chunk])

        x = torch.cat(sequential, dim=0)
        return x.view(local_tokens * self.cp_size, num_heads // self.cp_size, head_dim)

    def _head_to_sequence_parallel(self, x: Tensor, cu_seqlens: Tensor, local_tokens: int, num_heads: int) -> Tensor:
        """Ulysses CP inverse all-to-all: full sequence, head shard -> local zigzag sequence shard."""
        global_tokens, heads_per_cp_rank, head_dim = x.shape
        assert global_tokens == local_tokens * self.cp_size, (
            f"Unexpected Ulysses global token count: {global_tokens} vs {local_tokens * self.cp_size}"
        )
        assert heads_per_cp_rank * self.cp_size == num_heads, (
            f"Unexpected Ulysses head shard: {heads_per_cp_rank} * {self.cp_size} != {num_heads}"
        )
        local_lengths = self._local_packed_lengths(cu_seqlens, local_tokens)

        x = x.reshape(global_tokens, 1, heads_per_cp_rank * head_dim)
        rank_ordered = [[] for _ in range(self.cp_size)]
        seq_start = 0
        for local_length in local_lengths:
            chunk = local_length // 2
            chunks = torch.split(x[seq_start : seq_start + local_length * self.cp_size], chunk, dim=0)
            assert len(chunks) == 2 * self.cp_size
            for rank in range(self.cp_size):
                rank_ordered[rank].append(chunks[rank])
                rank_ordered[rank].append(chunks[2 * self.cp_size - rank - 1])
            seq_start += local_length * self.cp_size

        rank_ordered = torch.cat([torch.cat(parts, dim=0) for parts in rank_ordered], dim=0)
        rank_ordered = all_to_all(self.cp_group, rank_ordered.reshape(global_tokens, -1))
        output = torch.cat(torch.split(rank_ordered, local_tokens, dim=0), dim=-1)
        return output.view(local_tokens, num_heads, head_dim)

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
        if not HAVE_FA3_VARLEN or fa3_varlen_func is None:
            raise ImportError("Flash Attention 3 varlen is required for SGLangFlashAttention")

        is_packed = packed_seq_params is not None
        input_ndim = query.dim()
        head_dim_idx = -2 if is_packed and input_ndim >= 3 else 2
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
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
            local_tokens, local_heads, _ = query.shape
            query = self._sequence_to_head_parallel(query, cu_seqlens_q)
            key = self._sequence_to_head_parallel(key, cu_seqlens_k)
            value = self._sequence_to_head_parallel(value, cu_seqlens_k)

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
            output = self._head_to_sequence_parallel(output, cu_seqlens_q, local_tokens, local_heads)

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


class SGLangSpecProvider(BackendSpecProvider):
    """Backend provider for the correctness-first SGLang-compatible Megatron surface."""

    def column_parallel_linear(self) -> type:
        return SGLangColumnParallelLinear

    def row_parallel_linear(self) -> type:
        return SGLangRowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False):
        if for_qk:
            return SGLangQKRMSNorm if rms_norm else SGLangNorm
        return ModuleSpec(
            module=SGLangNorm,
            params={
                "cast_x_before_out_mul": True,
                "override_orig_dtype": torch.float32,
                "keep_weight_fp32": True,
            },
        )

    def core_attention(self) -> type:
        return SGLangCoreAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        del moe_use_legacy_grouped_gemm

        if moe_use_grouped_gemm:
            warnings.warn(
                "SGLang backend falls back to Megatron's existing GroupedMLP surface "
                "until the deterministic MoE backend is introduced."
            )
            return GroupedMLP, None

        return SequentialMLP, MLPSubmodules(
            linear_fc1=SGLangColumnParallelLinear,
            linear_fc2=SGLangRowParallelLinear,
        )

    def activation_func(self) -> type:
        return None
