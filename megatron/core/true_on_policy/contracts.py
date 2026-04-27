# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings


QWEN3_DENSE_TRUE_ON_POLICY_V1 = "qwen3_dense_true_on_policy_v1"


@dataclass(frozen=True)
class MegatronTrueOnPolicyRuntimePolicy:
    """Megatron-local behavior implied by a true-on-policy parity contract."""

    contract_name: Optional[str]
    enabled: bool
    use_sglang_backend: bool
    batch_invariant_mode: bool
    disable_rope_fusion: bool
    disable_bias_swiglu_fusion: bool
    attention_backend: str
    cp_layout: Optional[str]
    cast_qk_norm_input_before_weight_mul: bool


DEFAULT_RUNTIME_POLICY = MegatronTrueOnPolicyRuntimePolicy(
    contract_name=None,
    enabled=False,
    use_sglang_backend=False,
    batch_invariant_mode=False,
    disable_rope_fusion=False,
    disable_bias_swiglu_fusion=False,
    attention_backend="default",
    cp_layout=None,
    cast_qk_norm_input_before_weight_mul=True,
)


def validate_true_on_policy_contract(contract_name: Optional[str]) -> None:
    if contract_name is None:
        return
    if contract_name != QWEN3_DENSE_TRUE_ON_POLICY_V1:
        raise ValueError(f"Unsupported Megatron true-on-policy contract: {contract_name!r}")


def resolve_true_on_policy_runtime_policy(config) -> MegatronTrueOnPolicyRuntimePolicy:
    contract_name = getattr(config, "true_on_policy_contract", None)
    if contract_name is None and getattr(config, "use_sglang", False):
        contract_name = QWEN3_DENSE_TRUE_ON_POLICY_V1
        warnings.warn(
            "--use-sglang without --true-on-policy-contract defaults to "
            f"{QWEN3_DENSE_TRUE_ON_POLICY_V1!r} for backward compatibility. "
            "Pass the contract explicitly for new true-on-policy runs.",
            stacklevel=2,
        )
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY

    validate_true_on_policy_contract(contract_name)
    return MegatronTrueOnPolicyRuntimePolicy(
        contract_name=contract_name,
        enabled=True,
        use_sglang_backend=getattr(config, "use_sglang", False),
        batch_invariant_mode=getattr(config, "batch_invariant_mode", False),
        disable_rope_fusion=True,
        disable_bias_swiglu_fusion=True,
        attention_backend="fa3_varlen",
        cp_layout="ulysses_a2a"
        if getattr(config, "context_parallel_size", 1) > 1
        else None,
        cast_qk_norm_input_before_weight_mul=True,
    )
