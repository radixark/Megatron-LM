from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig
from .contracts import resolve_true_on_policy_runtime_policy


_FUSED_RMSNORM_ENV = "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM"
_FUSED_RMSNORM_DEBUG_ENV = "SGLANG_TRUE_ON_POLICY_FUSED_RMSNORM_DEBUG"
_fused_rmsnorm_debug_keys: set[tuple[str, tuple[int, ...], bool, bool]] = set()
_fused_rmsnorm_entry_debug_keys: set[tuple[str, tuple[int, ...], bool, bool]] = set()

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _should_use_fused_rms_norm(*tensors: Optional[torch.Tensor]) -> bool:
    return _env_flag_enabled(_FUSED_RMSNORM_ENV) and any(
        tensor is not None and tensor.is_cuda for tensor in tensors
    )


def _needs_native_grad(*tensors: Optional[torch.Tensor]) -> bool:
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def _with_native_grad(
    exact: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    native: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(exact, tuple):
        exact_output, exact_residual = exact
        native_output, native_residual = native
        return (
            exact_output.detach() + (native_output - native_output.detach()),
            exact_residual.detach() + (native_residual - native_residual.detach()),
        )
    return exact.detach() + (native - native.detach())


def _debug_fused_rms_norm(
    label: str,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
) -> None:
    if not _env_flag_enabled(_FUSED_RMSNORM_DEBUG_ENV):
        return

    key = (
        label,
        tuple(x.shape),
        residual is not None,
        post_residual_addition is not None,
    )
    if key in _fused_rmsnorm_debug_keys or len(_fused_rmsnorm_debug_keys) >= 16:
        return

    _fused_rmsnorm_debug_keys.add(key)
    logger.warning(
        "[true-on-policy] Megatron fused RMSNorm "
        f"label={label} shape={tuple(x.shape)} dtype={x.dtype} "
        f"residual={residual is not None} "
        f"post_residual={post_residual_addition is not None}"
    )


def _debug_rms_norm_entry(
    label: str,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
) -> None:
    if not _env_flag_enabled(_FUSED_RMSNORM_DEBUG_ENV):
        return

    key = (
        label,
        tuple(x.shape),
        residual is not None,
        post_residual_addition is not None,
    )
    if key in _fused_rmsnorm_entry_debug_keys or len(_fused_rmsnorm_entry_debug_keys) >= 16:
        return

    _fused_rmsnorm_entry_debug_keys.add(key)
    logger.warning(
        "[true-on-policy] Megatron RMSNorm entry "
        f"label={label} shape={tuple(x.shape)} dtype={x.dtype} "
        f"residual={residual is not None} "
        f"post_residual={post_residual_addition is not None} "
        f"env={os.environ.get(_FUSED_RMSNORM_ENV)!r} "
        f"cuda={x.is_cuda} grad_enabled={torch.is_grad_enabled()} "
        f"requires_grad={x.requires_grad}"
    )


def _maybe_true_on_policy_fused_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    debug_label: str,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    cast_x_before_out_mul: bool = True,
    norm_cast_dtype: torch.dtype,
    weight_cast_dtype: torch.dtype,
    output_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    if not _should_use_fused_rms_norm(x, weight, residual, post_residual_addition):
        return None

    try:
        from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm
    except Exception:
        return None

    output = true_on_policy_rms_norm(
        x,
        weight,
        eps,
        residual=residual,
        post_residual_addition=post_residual_addition,
        cast_x_before_out_mul=cast_x_before_out_mul,
        norm_cast_dtype=norm_cast_dtype,
        weight_cast_dtype=weight_cast_dtype,
        output_dtype=output_dtype,
        residual_dtype=residual_dtype,
    )
    _debug_fused_rms_norm(debug_label, x, residual, post_residual_addition)
    return output


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
        _debug_rms_norm_entry("SGLangNorm", x, residual, post_residual_addition)
        if self.normalization == "LayerNorm":
            if residual is not None:
                x = x + residual
                if post_residual_addition is not None:
                    x = x + post_residual_addition
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            return F.layer_norm(x, self.hidden_size, weight, self.bias, self.eps)

        fused = None
        if residual is None:
            fused = _maybe_true_on_policy_fused_rms_norm(
                x,
                self.weight.float(),
                self.eps,
                debug_label="SGLangNorm",
                residual=residual,
                post_residual_addition=post_residual_addition,
                cast_x_before_out_mul=self.cast_x_before_out_mul,
                norm_cast_dtype=self.override_orig_dtype or x.dtype,
                weight_cast_dtype=torch.float32,
                residual_dtype=x.dtype,
            )
        if fused is not None:
            if not _needs_native_grad(x, self.weight, residual, post_residual_addition):
                return fused
            return _with_native_grad(
                fused,
                self._forward_rmsnorm_native(x, residual, post_residual_addition),
            )

        return self._forward_rmsnorm_native(x, residual, post_residual_addition)

    def _forward_rmsnorm_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = self.override_orig_dtype or x.dtype
        x_float = x.float()
        if residual is not None:
            x_float = x_float + residual.float()
            if post_residual_addition is not None:
                x_float = x_float + post_residual_addition.float()
            residual = x_float.to(x.dtype)
            x_float = residual.float()

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

        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        policy = resolve_true_on_policy_runtime_policy(config)
        self.cast_x_before_out_mul = policy.cast_qk_norm_input_before_weight_mul
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _debug_rms_norm_entry("SGLangQKRMSNorm", x, None, None)
        if not x.is_contiguous():
            x = x.contiguous()

        fused = _maybe_true_on_policy_fused_rms_norm(
            x,
            self.weight.float(),
            self.eps,
            debug_label="SGLangQKRMSNorm",
            cast_x_before_out_mul=self.cast_x_before_out_mul,
            norm_cast_dtype=x.dtype,
            weight_cast_dtype=torch.float32,
        )
        if fused is not None:
            if not _needs_native_grad(x, self.weight):
                return fused
            return _with_native_grad(fused, self._forward_native(x))

        return self._forward_native(x)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)

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

        del persist_layer_norm
        del zero_centered_gamma
        del normalization

        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.source_truth_orig_dtype = getattr(config, "params_dtype", None)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _debug_rms_norm_entry("SGLangFinalRMSNorm", x, residual, post_residual_addition)
        if not x.is_contiguous():
            x = x.contiguous()

        return self._forward_native(x, residual, post_residual_addition)

    def _forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = self.source_truth_orig_dtype or x.dtype
        if residual is not None:
            x = x + residual
            if post_residual_addition is not None:
                x = x + post_residual_addition
            residual = x.clone()

        x_float = x.to(torch.float32)
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = self.weight.to(orig_dtype) * x_float.to(orig_dtype)

        if residual is not None:
            return output, residual
        return output
