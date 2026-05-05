"""``SGLangGroupedMLP``: Megatron grouped expert with an SGLang no-grad forward.

The autograd path stays on Megatron's ``GroupedMLP``. Inference (reference and
logprob recompute) routes to SGLang's ``fused_experts`` Triton entry point so
the forward numerics match the rollout engine exactly.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.experts import GroupedMLP


class SGLangGroupedMLP(GroupedMLP):
    """Grouped MoE expert surface with an SGLang-compatible no-grad forward.

    The autograd path stays on Megatron's GroupedMLP. The true-on-policy
    reference/logprob path runs under no-grad, where using the same fused expert
    primitive as rollout removes grouped-GEMM arithmetic drift.
    """

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        if self._should_use_sglang_fused_forward(permuted_local_hidden_states):
            source_counts = self._consume_sglang_alltoall_source_counts()
            if source_counts is not None:
                return (
                    self._forward_sglang_fused_by_source(
                        permuted_local_hidden_states,
                        source_counts,
                        permuted_probs,
                    ),
                    None,
                )
            return (
                self._forward_sglang_fused(
                    permuted_local_hidden_states,
                    tokens_per_expert,
                    permuted_probs,
                ),
                None,
            )

        return super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)

    def set_sglang_alltoall_source_counts(self, counts: torch.Tensor | None) -> None:
        self._sglang_alltoall_source_counts = counts

    def forward_sglang_local_masked(
        self,
        hidden_states: torch.Tensor,
        routing_probs: torch.Tensor,
        topk: int,
        local_expert_indices: list[int],
    ) -> torch.Tensor:
        """Run SGLang's standard EP fused-expert path for this rank's local experts.

        SGLang's non-DeepEP expert-parallel path keeps the token stream intact on
        every EP rank, maps non-local top-k expert ids to -1, runs fused_experts
        with the original top-k width, and then all-reduces the local expert
        contributions. This helper mirrors the per-rank fused_experts call; the
        caller owns the EP all-reduce.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("SGLang local-masked MoE fused path is inference-only")
        if hidden_states.numel() == 0:
            return torch.empty_like(hidden_states)

        self._ensure_sglang_server_args()

        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat_hidden_states.dtype != self.weight1.dtype:
            flat_hidden_states = flat_hidden_states.to(self.weight1.dtype)

        global_topk_ids = self._ordered_topk_ids_from_probs(routing_probs, topk)
        local_topk_ids = self._local_masked_topk_ids(global_topk_ids, local_expert_indices)
        topk_weights = torch.gather(
            routing_probs.to(device=flat_hidden_states.device),
            dim=-1,
            index=global_topk_ids.to(device=flat_hidden_states.device),
        ).to(flat_hidden_states.dtype)

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
            top_k=topk,
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

    def _consume_sglang_alltoall_source_counts(self) -> torch.Tensor | None:
        counts = getattr(self, "_sglang_alltoall_source_counts", None)
        self._sglang_alltoall_source_counts = None
        if counts is None or counts.dim() != 2 or counts.shape[1] != self.num_local_experts:
            return None
        if counts.shape[0] <= 1:
            return None
        return counts.to(device="cpu", dtype=torch.long)

    def _should_use_sglang_fused_forward(self, hidden_states: torch.Tensor) -> bool:
        return (
            hidden_states.is_cuda
            and not torch.is_grad_enabled()
            and self.config.bf16
            and self.config.gated_linear_unit
            and self.config.activation_func is F.silu
            and not self.config.moe_apply_probs_on_input
            and not self.activation_recompute
        )

    def _forward_sglang_fused(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("SGLang MoE fused path is inference-only")
        if permuted_local_hidden_states.numel() == 0:
            return super().forward(
                permuted_local_hidden_states, tokens_per_expert, permuted_probs
            )[0]

        self._ensure_sglang_server_args()

        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        hidden_states = permuted_local_hidden_states
        if hidden_states.dtype != self.weight1.dtype:
            hidden_states = hidden_states.to(self.weight1.dtype)

        topk_output = StandardTopKOutput(
            topk_weights=permuted_probs.reshape(-1, 1).to(hidden_states.dtype),
            topk_ids=self._local_topk_ids(tokens_per_expert, hidden_states.device),
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
            top_k=1,
            params_dtype=self.config.params_dtype,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            inplace=True,
            no_combine=False,
        )

        return fused_experts(
            hidden_states=hidden_states.contiguous(),
            w1=self._sglang_w13_weight(),
            w2=self._sglang_w2_weight(),
            topk_output=topk_output,
            moe_runner_config=runner_config,
        )

    def _forward_sglang_fused_by_source(
        self,
        permuted_local_hidden_states: torch.Tensor,
        source_counts: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        if int(source_counts.sum().item()) != permuted_local_hidden_states.shape[0]:
            return self._forward_sglang_fused(
                permuted_local_hidden_states,
                source_counts.sum(dim=0),
                permuted_probs,
            )

        num_sources = source_counts.shape[0]
        expert_source_hidden: list[list[torch.Tensor]] = []
        expert_source_probs: list[list[torch.Tensor]] = []
        offset = 0
        for expert_idx in range(self.num_local_experts):
            hidden_chunks = []
            prob_chunks = []
            for source_idx in range(num_sources):
                count = int(source_counts[source_idx, expert_idx].item())
                hidden_chunks.append(permuted_local_hidden_states[offset : offset + count])
                prob_chunks.append(permuted_probs[offset : offset + count])
                offset += count
            expert_source_hidden.append(hidden_chunks)
            expert_source_probs.append(prob_chunks)

        expert_outputs: list[list[torch.Tensor]] = [[] for _ in range(self.num_local_experts)]
        for source_idx in range(num_sources):
            source_hidden_chunks = [
                expert_source_hidden[expert_idx][source_idx]
                for expert_idx in range(self.num_local_experts)
            ]
            source_prob_chunks = [
                expert_source_probs[expert_idx][source_idx]
                for expert_idx in range(self.num_local_experts)
            ]
            source_total = sum(chunk.shape[0] for chunk in source_hidden_chunks)
            if source_total == 0:
                continue

            source_hidden = torch.cat(source_hidden_chunks, dim=0)
            source_probs = torch.cat(source_prob_chunks, dim=0)
            source_tokens_per_expert = source_counts[source_idx]
            source_output = self._forward_sglang_fused(
                source_hidden,
                source_tokens_per_expert,
                source_probs,
            )
            split_outputs = source_output.split(source_tokens_per_expert.tolist(), dim=0)
            for expert_idx, output in enumerate(split_outputs):
                if output.shape[0] != 0:
                    expert_outputs[expert_idx].append(output)

        ordered_outputs = []
        for expert_idx in range(self.num_local_experts):
            if expert_outputs[expert_idx]:
                ordered_outputs.append(torch.cat(expert_outputs[expert_idx], dim=0))
        if not ordered_outputs:
            return torch.empty_like(permuted_local_hidden_states)
        return torch.cat(ordered_outputs, dim=0)

    def _local_topk_ids(self, tokens_per_expert: torch.Tensor, device: torch.device) -> torch.Tensor:
        counts = tokens_per_expert.to(device=device, dtype=torch.long)
        expert_ids = torch.arange(
            self.num_local_experts,
            device=device,
            dtype=torch.int32,
        )
        return torch.repeat_interleave(expert_ids, counts).reshape(-1, 1)

    @staticmethod
    def _ordered_topk_ids_from_probs(routing_probs: torch.Tensor, topk: int) -> torch.Tensor:
        from sglang.srt.tp_invariant_ops import stable_topk

        _, selected_ids = stable_topk(routing_probs, topk)
        return selected_ids

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
        return (
            self.weight1.view(
                self.num_local_experts,
                self.config.hidden_size,
                -1,
            )
            .permute(0, 2, 1)
            .contiguous()
        )

    def _sglang_w2_weight(self) -> torch.Tensor:
        return (
            self.weight2.view(
                self.num_local_experts,
                -1,
                self.config.hidden_size,
            )
            .permute(0, 2, 1)
            .contiguous()
        )

    @staticmethod
    def _ensure_sglang_server_args() -> None:
        from sglang.srt.server_args import (
            get_global_server_args,
            set_global_server_args_for_scheduler,
        )

        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(
                SimpleNamespace(
                    enable_fused_moe_sum_all_reduce=False,
                    enable_deterministic_inference=True,
                )
            )
