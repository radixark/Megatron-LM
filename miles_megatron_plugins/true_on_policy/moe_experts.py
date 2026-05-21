"""``SGLangGroupedMLP`` helpers for direct SGLang MoE execution.

The MoE layer calls these helpers only on the direct local-masked EP path:
no-grad/reference uses SGLang ``fused_experts`` directly, while grad-enabled
training uses the layer-level autograd wrapper with the same SGLang weight
layout helpers.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.experts import GroupedMLP

try:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    HAVE_SGLANG_FUSED_MOE_FORWARD = True
except ImportError:
    MoeRunnerConfig = None
    StandardTopKOutput = None
    fused_experts = None
    get_global_server_args = None
    set_global_server_args_for_scheduler = None
    HAVE_SGLANG_FUSED_MOE_FORWARD = False


class SGLangGroupedMLP(GroupedMLP):
    """GroupedMLP subclass exposing the SGLang local-masked fused expert call."""

    def forward_sglang_local_masked_topk(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        global_topk_ids: torch.Tensor,
        local_expert_indices: list[int],
    ) -> torch.Tensor:
        """Run SGLang's local-masked fused expert path with precomputed top-k.

        The caller can use this when it already has SGLang-ordered top-k ids and
        weights.  This avoids materializing a full sparse routing matrix and
        avoids running stable-topk twice in the Megatron exact no-grad path.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("SGLang local-masked MoE fused path is inference-only")
        if not HAVE_SGLANG_FUSED_MOE_FORWARD:
            raise RuntimeError("SGLang fused MoE forward path is not available")
        if hidden_states.numel() == 0:
            return torch.empty_like(hidden_states)

        self._ensure_sglang_server_args()

        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat_hidden_states.dtype != self.weight1.dtype:
            flat_hidden_states = flat_hidden_states.to(self.weight1.dtype)

        local_topk_ids = self._local_masked_topk_ids(global_topk_ids, local_expert_indices)
        topk_weights = topk_weights.to(
            device=flat_hidden_states.device,
            dtype=flat_hidden_states.dtype,
        )

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=local_topk_ids,
            router_logits=None,
        )
        runner_config = MoeRunnerConfig(
            # Keep this as the global expert count. SGLang uses num_experts !=
            # num_local_experts under EP, which selects the filtered activation
            # kernel path in fused_experts.
            num_experts=self.config.num_moe_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.config.hidden_size,
            intermediate_size_per_partition=self.config.moe_ffn_hidden_size,
            top_k=topk_weights.shape[-1],
            params_dtype=self.config.params_dtype,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            inplace=True,
            no_combine=False,
        )

        return fused_experts(
            hidden_states=flat_hidden_states.contiguous(),
            w1=self._sglang_w13_weight(),
            w2=self._sglang_w2_weight(),
            topk_output=topk_output,
            moe_runner_config=runner_config,
        )

    def _local_masked_topk_ids(
        self,
        global_topk_ids: torch.Tensor,
        local_expert_indices: list[int],
    ) -> torch.Tensor:
        expert_start = int(local_expert_indices[0])
        expert_end = expert_start + self.num_local_experts
        local_ids = global_topk_ids.to(torch.int32) - expert_start
        local_mask = (global_topk_ids >= expert_start) & (global_topk_ids < expert_end)
        return torch.where(local_mask, local_ids, torch.full_like(local_ids, -1))

    def _sglang_w13_weight(self) -> torch.Tensor:
        return self._cached_sglang_weight(
            cache_name="_sglang_w13_weight_cache",
            weight=self.weight1,
            view_shape=(self.num_local_experts, self.config.hidden_size, -1),
            permute_dims=(0, 2, 1),
        )

    def _sglang_w2_weight(self) -> torch.Tensor:
        return self._cached_sglang_weight(
            cache_name="_sglang_w2_weight_cache",
            weight=self.weight2,
            view_shape=(self.num_local_experts, -1, self.config.hidden_size),
            permute_dims=(0, 2, 1),
        )

    def _cached_sglang_weight(
        self,
        *,
        cache_name: str,
        weight: torch.Tensor,
        view_shape: tuple[int, ...],
        permute_dims: tuple[int, ...],
    ) -> torch.Tensor:
        if (
            os.environ.get("MILES_TRUE_ON_POLICY_CACHE_SGLANG_EXPERT_WEIGHTS", "0")
            != "1"
            or (torch.is_grad_enabled() and weight.requires_grad)
        ):
            return weight.view(*view_shape).permute(*permute_dims).contiguous()

        key = (
            weight.data_ptr(),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
            weight.device,
            getattr(weight, "_version", None),
        )
        cached = getattr(self, cache_name, None)
        if cached is None or cached[0] != key:
            cached = (
                key,
                weight.view(*view_shape).permute(*permute_dims).contiguous(),
            )
            setattr(self, cache_name, cached)
        return cached[1]

    @staticmethod
    def _ensure_sglang_server_args() -> None:
        if not HAVE_SGLANG_FUSED_MOE_FORWARD:
            raise RuntimeError("SGLang server args helpers are not available")

        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(
                SimpleNamespace(
                    enable_fused_moe_sum_all_reduce=False,
                    enable_deterministic_inference=True,
                )
            )
