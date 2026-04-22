# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import inspect
import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.matmul_tp_inv import sglang_reference_matmul
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
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


class SGLangColumnParallelLinear(ColumnParallelLinear):
    """Megatron local column-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
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
        )


class SGLangRowParallelLinear(RowParallelLinear):
    """Megatron local row-parallel layer with an explicit SGLang backend identity."""

    backend_name = "sglang"

    def _forward_impl(self, input, weight, *args, **kwargs):
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
        )


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
    ) -> None:
        super().__init__()

        del persist_layer_norm
        del normalization

        self.config = config
        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.normalization = config.normalization
        self.zero_centered_gamma = config.layernorm_zero_centered_gamma or zero_centered_gamma

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "LayerNorm":
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            return F.layer_norm(x, self.hidden_size, weight, self.bias, self.eps)

        x_float = x.float()
        output = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class SGLangFlashAttention(MegatronModule):
    """Minimal SGLang-compatible FlashAttention path for Ulysses CP.

    This path is intentionally narrow: it is only used when the SGLang backend
    is selected and Megatron is configured with Ulysses context parallelism
    (`cp_comm_type == "a2a"`). For non-Ulysses layouts we continue to use the
    existing DotProductAttention path.
    """

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
                f"SGLangFlashAttention only supports Ulysses (a2a) context parallelism, "
                f"got cp_comm_type={cp_comm_type!r}"
            )
            assert pg_collection is not None and hasattr(
                pg_collection, "cp"
            ), "ProcessGroupCollection must have a 'cp' group for context parallelism"
            self.cp_group = pg_collection.cp
        else:
            self.cp_group = None

        kv_channels = config.kv_channels
        assert kv_channels is not None, "kv_channels must be set"
        projection_size = kv_channels * config.num_attention_heads
        assert pg_collection is not None and hasattr(
            pg_collection, "tp"
        ), "ProcessGroupCollection must have a 'tp' group for SGLangFlashAttention"

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        if self.cp_size > 1:
            assert self.num_attention_heads_per_partition % self.cp_size == 0, (
                f"num_attention_heads_per_partition ({self.num_attention_heads_per_partition}) "
                f"must be divisible by cp_size ({self.cp_size})"
            )

        self.softmax_scale = (
            1.0 / math.sqrt(self.hidden_size_per_attention_head)
            if softmax_scale is None
            else softmax_scale
        )
        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= layer_number

        self.attention_dropout = (
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def _ulysses_slice_heads(self, x: Tensor) -> Tensor:
        cp_rank = self.cp_group.rank()
        h_local = x.shape[1] // self.cp_size
        start = cp_rank * h_local
        return x[:, start : start + h_local, :].contiguous()

    def _ulysses_gather_heads(self, x: Tensor) -> Tensor:
        out_list = [torch.empty_like(x) for _ in range(self.cp_size)]
        torch.distributed.all_gather(out_list, x.contiguous(), group=self.cp_group)
        return torch.cat(out_list, dim=1)

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
        """Run FA3 varlen attention with Ulysses head-parallel gather when cp_size > 1."""
        del attention_mask

        assert attention_bias is None, "Attention bias not supported for SGLangFlashAttention"
        assert (
            attn_mask_type is None or attn_mask_type == AttnMaskType.causal
        ), "Only causal mask is supported for SGLangFlashAttention"
        if not HAVE_FA3_VARLEN or fa3_varlen_func is None:
            raise ImportError(
                "Flash Attention 3 with flash_attn_varlen_func is required "
                "for SGLangFlashAttention."
            )

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
                t, np, hn = query.shape
            else:
                t, b, np, hn = query.shape
                assert b == 1, f"Packed sequences should have batch=1, got {b}"
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
            sq, b, np, hn = query.shape
            sk = key.shape[0]
            query = query.transpose(0, 1).reshape(b * sq, np, hn)
            key = key.transpose(0, 1).reshape(b * sk, np, hn)
            value = value.transpose(0, 1).reshape(b * sk, np, hn)
            cu_seqlens_q = torch.arange(0, (b + 1) * sq, sq, dtype=torch.int32, device=query.device)
            cu_seqlens_k = torch.arange(0, (b + 1) * sk, sk, dtype=torch.int32, device=query.device)
            max_seqlen_q = sq
            max_seqlen_k = sk

        if self.cp_size > 1:
            query = self._ulysses_slice_heads(query)
            key = self._ulysses_slice_heads(key)
            value = self._ulysses_slice_heads(value)

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

        output = fa3_varlen_func(**fa3_kwargs)
        if isinstance(output, tuple):
            output = output[0]

        if self.cp_size > 1:
            output = self._ulysses_gather_heads(output)

        if is_packed:
            if input_ndim == 3:
                output = output.view(t, self.hidden_size_per_partition)
            else:
                output = output.view(t, 1, self.hidden_size_per_partition)
        else:
            output = output.view(b, sq, np, hn)
            output = output.transpose(0, 1)
            output = output.reshape(sq, b, self.hidden_size_per_partition)

        return output


class SGLangCoreAttention(MegatronModule):
    """Dispatch Ulysses CP to SGLang FA3 and keep all other paths unchanged."""

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

        impl_cls = (
            SGLangFlashAttention
            if config.context_parallel_size > 1 and cp_comm_type == "a2a"
            else DotProductAttention
        )
        self.impl = impl_cls(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self._current_max_attn_logits = getattr(self.impl, "current_max_attn_logits", None)

    @property
    def current_max_attn_logits(self):
        """Proxy to the underlying attention impl for Megatron's max-logit tracking."""
        return getattr(self.impl, "current_max_attn_logits", self._current_max_attn_logits)

    @current_max_attn_logits.setter
    def current_max_attn_logits(self, value):
        self._current_max_attn_logits = value
        if hasattr(self.impl, "current_max_attn_logits"):
            self.impl.current_max_attn_logits = value

    def forward(self, *args, **kwargs):
        """Delegate to the selected attention implementation."""
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

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        del rms_norm
        del for_qk
        return SGLangNorm

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
            linear_fc1=SGLangColumnParallelLinear, linear_fc2=SGLangRowParallelLinear
        )

    def activation_func(self) -> type:
        return None
