# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.matmul_tp_inv import sglang_reference_matmul
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.transformer_config import TransformerConfig


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
        return DotProductAttention

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
