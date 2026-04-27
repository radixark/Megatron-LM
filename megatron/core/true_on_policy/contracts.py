# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

from megatron.core.true_on_policy.schema import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
    TrueOnPolicyContractName,
    TrueOnPolicyContractSchema,
)

QWEN3_DENSE_TRUE_ON_POLICY_V1 = QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA.name
_WARNED_IMPLICIT_QWEN3_DENSE_CONTRACT = False


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
    cast_lm_head_input_to_weight_dtype: bool
    deterministic_row_parallel_reduce: bool
    defer_ulysses_cp_loss_scaling_to_grad_sum: bool
    apply_logits_contract: bool


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
    cast_lm_head_input_to_weight_dtype=False,
    deterministic_row_parallel_reduce=False,
    defer_ulysses_cp_loss_scaling_to_grad_sum=False,
    apply_logits_contract=False,
)


@dataclass(frozen=True)
class MegatronTrueOnPolicyContract:
    """Megatron-local adapter from a shared contract schema to runtime policy."""

    schema: TrueOnPolicyContractSchema

    @property
    def name(self) -> TrueOnPolicyContractName:
        return self.schema.name

    def policy_for(self, config) -> MegatronTrueOnPolicyRuntimePolicy:
        return MegatronTrueOnPolicyRuntimePolicy(
            contract_name=self.name,
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
            cast_lm_head_input_to_weight_dtype=True,
            deterministic_row_parallel_reduce=True,
            defer_ulysses_cp_loss_scaling_to_grad_sum=True,
            apply_logits_contract=True,
        )


QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT = MegatronTrueOnPolicyContract(
    schema=QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA,
)


_CONTRACT_BY_NAME = {
    QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT.name: QWEN3_DENSE_TRUE_ON_POLICY_CONTRACT,
}


def get_true_on_policy_contract(contract_name: str) -> MegatronTrueOnPolicyContract:
    try:
        return _CONTRACT_BY_NAME[contract_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(
            f"Unsupported Megatron true-on-policy contract {contract_name!r}. "
            f"Supported contracts: {supported}"
        ) from exc


def validate_true_on_policy_contract(contract_name: Optional[str]) -> None:
    if contract_name is None:
        return
    get_true_on_policy_contract(contract_name)


def resolve_true_on_policy_runtime_policy(config) -> MegatronTrueOnPolicyRuntimePolicy:
    global _WARNED_IMPLICIT_QWEN3_DENSE_CONTRACT

    contract_name = getattr(config, "true_on_policy_contract", None)
    if contract_name is None and getattr(config, "use_sglang", False):
        contract_name = QWEN3_DENSE_TRUE_ON_POLICY_V1
        if not _WARNED_IMPLICIT_QWEN3_DENSE_CONTRACT:
            warnings.warn(
                "--use-sglang without --true-on-policy-contract defaults to "
                f"{QWEN3_DENSE_TRUE_ON_POLICY_V1!r} for backward compatibility. "
                "Pass the contract explicitly for new true-on-policy runs.",
                stacklevel=2,
            )
            _WARNED_IMPLICIT_QWEN3_DENSE_CONTRACT = True
    if contract_name is None:
        return DEFAULT_RUNTIME_POLICY

    validate_true_on_policy_contract(contract_name)
    return get_true_on_policy_contract(contract_name).policy_for(config)
