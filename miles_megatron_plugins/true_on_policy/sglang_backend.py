"""Compatibility facade for the Megatron true-on-policy SGLang backend."""

from .sglang_attention import (
    HAVE_FA3_VARLEN,
    SGLangCoreAttention,
    SGLangFlashAttention,
    fa3_varlen_func,
)
from .bias_dropout import (
    _sglang_bias_dropout_add,
    get_sglang_bias_dropout_add,
)
from .contracts import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    MegatronTrueOnPolicyRuntimePolicy,
    resolve_true_on_policy_runtime_policy,
)
from .cp_layout import SGLangUlyssesCPLayout
from .linear import SGLangColumnParallelLinear, SGLangRowParallelLinear
from .norm import SGLangFinalRMSNorm, SGLangNorm, SGLangQKRMSNorm
from .provider import SGLangSpecProvider
from .runtime import (
    enable_sglang_batch_invariant_mode,
    ensure_batch_invariant_mode_from_config,
)

_ensure_batch_invariant_mode_from_config = ensure_batch_invariant_mode_from_config

__all__ = [
    "HAVE_FA3_VARLEN",
    "QWEN3_DENSE_TRUE_ON_POLICY_V1",
    "MegatronTrueOnPolicyRuntimePolicy",
    "SGLangColumnParallelLinear",
    "SGLangCoreAttention",
    "SGLangFinalRMSNorm",
    "SGLangFlashAttention",
    "SGLangNorm",
    "SGLangQKRMSNorm",
    "SGLangRowParallelLinear",
    "SGLangSpecProvider",
    "SGLangUlyssesCPLayout",
    "_sglang_bias_dropout_add",
    "enable_sglang_batch_invariant_mode",
    "fa3_varlen_func",
    "get_sglang_bias_dropout_add",
    "resolve_true_on_policy_runtime_policy",
]
