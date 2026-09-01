# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine integration for fused NVFP4 fake QAT."""

import os

import torch

from megatron.core.extensions.transformer_engine_int4_fake_qat import INT4_FAKE_QAT_FLAG
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import is_te_min_version

NVFP4_FAKE_QAT_FLAG = "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG"
_MIN_TE_VERSION = "2.17.0"


def _validate_nvfp4_fake_qat_support(
    config: ModelParallelConfig, delay_wgrad_compute: bool, weight_tensors: list[torch.Tensor]
) -> None:
    """Reject TE weight layouts and gradient paths that cannot safely use STE tensors."""
    if not is_te_min_version(_MIN_TE_VERSION):
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 requires Transformer Engine >= {_MIN_TE_VERSION}."
        )
    if os.getenv(INT4_FAKE_QAT_FLAG, "0") == "1":
        raise RuntimeError(
            f"{INT4_FAKE_QAT_FLAG}=1 and {NVFP4_FAKE_QAT_FLAG}=1 are mutually exclusive."
        )
    if config.gradient_accumulation_fusion:
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 is not supported with "
            "gradient_accumulation_fusion because TE fused wgrad accumulation mutates "
            "Python attributes on the original weight tensors."
        )
    if delay_wgrad_compute:
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 is not supported with delayed wgrad compute because "
            "the delayed TE path mutates Python attributes on the original weight tensors."
        )
    if getattr(config, "moe_single_grouped_weight", False):
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 requires TE's discrete grouped-linear parameters; "
            "moe_single_grouped_weight is not supported."
        )
    if any(
        hasattr(weight, "__fsdp_param__") or hasattr(weight, "get_main_grad")
        for weight in weight_tensors
    ):
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 is not supported with Megatron FSDP because "
            "FSDP patches weight tensors with main-gradient attributes and methods."
        )
    if any(getattr(weight, "ndim", None) != 2 for weight in weight_tensors):
        raise RuntimeError(
            f"{NVFP4_FAKE_QAT_FLAG}=1 requires one rank-2 tensor per grouped-linear GEMM."
        )


def maybe_fake_quantize_nvfp4_weight_tensors(
    config: ModelParallelConfig, delay_wgrad_compute: bool, weight_tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Optionally apply fused NVFP4 fake QAT to discrete TE grouped-linear weights."""
    if os.getenv(NVFP4_FAKE_QAT_FLAG, "0") != "1":
        return weight_tensors

    _validate_nvfp4_fake_qat_support(config, delay_wgrad_compute, weight_tensors)

    # Keep CuTe DSL optional for every Megatron process that does not enable this path.
    from megatron.core.fusions.fused_nvfp4_qdq import (
        current_nvfp4_qdq_config,
        fake_nvfp4_quantization_ste,
    )

    qdq_config = current_nvfp4_qdq_config()
    return [fake_nvfp4_quantization_ste(weight, qdq_config) for weight in weight_tensors]


__all__ = ["NVFP4_FAKE_QAT_FLAG", "maybe_fake_quantize_nvfp4_weight_tensors"]
