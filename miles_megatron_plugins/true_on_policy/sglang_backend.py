"""Compatibility facade for the Megatron true-on-policy SGLang backend."""

from .attention_fa3 import (
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
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    MegatronTrueOnPolicyRuntimePolicy,
    resolve_true_on_policy_runtime_policy,
)
from .cp_layout import SGLangUlyssesCPLayout
from .linear import SGLangColumnParallelLinear, SGLangRowParallelLinear
from .moe import SGLangGroupedMLP
from .norm import SGLangFinalRMSNorm, SGLangNorm, SGLangQKRMSNorm
from .provider import SGLangSpecProvider
from .rope import (
    disable_sglang_rope,
    enable_sglang_rope,
    is_sglang_rope_enabled,
    sglang_apply_rotary_pos_emb,
    sglang_apply_rotary_pos_emb_with_freqs,
)
from .schema import QWEN3_DENSE_SGLANG_MATH, QWEN3_MOE_SGLANG_MATH
from .runtime import (
    enable_sglang_batch_invariant_mode,
    ensure_batch_invariant_mode_from_config,
)

_ensure_batch_invariant_mode_from_config = ensure_batch_invariant_mode_from_config

__all__ = [
    "HAVE_FA3_VARLEN",
    "QWEN3_DENSE_SGLANG_MATH",
    "QWEN3_DENSE_TRUE_ON_POLICY_V1",
    "QWEN3_MOE_SGLANG_MATH",
    "QWEN3_MOE_TRUE_ON_POLICY_V1",
    "MegatronTrueOnPolicyRuntimePolicy",
    "SGLangColumnParallelLinear",
    "SGLangCoreAttention",
    "SGLangFinalRMSNorm",
    "SGLangFlashAttention",
    "SGLangGroupedMLP",
    "SGLangNorm",
    "SGLangQKRMSNorm",
    "SGLangRowParallelLinear",
    "SGLangSpecProvider",
    "SGLangUlyssesCPLayout",
    "_sglang_bias_dropout_add",
    "disable_sglang_rope",
    "enable_sglang_batch_invariant_mode",
    "enable_sglang_rope",
    "fa3_varlen_func",
    "get_sglang_bias_dropout_add",
    "is_sglang_rope_enabled",
    "resolve_true_on_policy_runtime_policy",
    "sglang_apply_rotary_pos_emb",
    "sglang_apply_rotary_pos_emb_with_freqs",
]
