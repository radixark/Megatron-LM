# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoECudaGraphPartialCaptureSignal,
    MoECudaGraphTensorStore,
    get_default_pg_collection,
    maybe_skip_or_early_return_by_cudagraph,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import internal_api

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TELinear, te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class RouterInterface(Protocol):
    """Interface for the router used in an MoELayer."""

    def forward(self, input: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the router.

        Returns:
            A tuple of (probabilities, routing_map).
        """
        ...

    def set_layer_number(self, layer_number: int) -> None:
        """Set the layer number for the router.

        Called from transformer_layer during initialization.
        """
        ...


class RouterBuilder(Protocol):
    """Protocol for building a Router."""

    def __call__(
        self, /, *, config: TransformerConfig, pg_collection: ProcessGroupCollection | None
    ) -> RouterInterface: ...


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None
    router: RouterBuilder = TopKRouter


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: RouterInterface = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)

    def set_is_mtp(self):
        """Mark this MoE layer as an MTP layer."""
        self.router.is_mtp = True


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        # If using mcore cudagraphs, recompute is handled by transformer_layer.MoETransformerLayer
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective'
            and "moe" in config.recompute_modules
            and config.cuda_graph_impl != 'local'
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        self.tp_group = pg_collection.tp

        # Initialize router.
        self.router = submodules.router(config=self.config, pg_collection=pg_collection)
        self.tp_group = pg_collection.tp

        # Initialize latent projections.
        if self.config.moe_latent_size:
            assert HAVE_TE, "TransformerEngine is required for MoE latent projections."
            self.fc1_latent_proj = TELinear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )
            self.fc2_latent_proj = TELinear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                parallel_mode="duplicated",
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                is_expert=False,
            )

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts,
                config=self.config,
                pg_collection=pg_collection,
                gate=self.config.moe_shared_expert_gate,
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

        # Cudagraph tensor store for resuming the forward pass from the end of the cudagraph.
        self.cudagraph_tensor_store = MoECudaGraphTensorStore()
        self.fwd_execution_map = ["route", "expert_compute", "postprocess"]

    @maybe_skip_or_early_return_by_cudagraph("route")
    def route(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Compute token routing for preprocessing.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping.
        """
        probs, routing_map = apply_module(self.router)(hidden_states, padding_mask)
        return probs, routing_map

    @maybe_skip_or_early_return_by_cudagraph("preprocess")
    def preprocess(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        """Preprocess token routing for dispatch.

        This method preprocesses the hidden states and routing probabilities for the token
        dispatcher.
        """
        # Project the hidden_states from hidden dimension down to latent dimenion.
        if self.config.moe_latent_size:
            assert (
                not self.shared_expert_overlap
            ), "Shared expert overlap not supported when MoE latent projections are used."
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.

        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    @maybe_skip_or_early_return_by_cudagraph("shared_experts_compute")
    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8 or self.config.fp4:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    @internal_api
    def routed_experts_compute(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if hasattr(self.experts, "set_sglang_alltoall_source_counts"):
            self.experts.set_sglang_alltoall_source_counts(
                getattr(self.token_dispatcher, "num_global_tokens_per_local_expert", None)
            )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication).
        """
        output = self.token_dispatcher.token_combine(output)
        return output

    def postprocess(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Project the output back from latent dimension to hidden dimension after combine
        in latent dimension if needed. Combine expert output with shared_experts if needed."""

        output = self.token_dispatcher.combine_postprocess(output)
        if self.config.moe_latent_size:
            output, _ = self.fc2_latent_proj(output)

        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """This method is a combined method of route and preprocess. Deprecated."""

        probs, routing_map = self.route(hidden_states)
        hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
        return hidden_states, probs, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.
        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )
        # Transpose from [bsz, seq_length] to [seq_length, bsz] to align with hidden_states
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states, intermediate_tensors, padding_mask=None):
            sglang_exact_output = None
            try:
                if "route" in self.fwd_execution_map:
                    shared_expert_output = self.shared_experts_compute(hidden_states)
                    if self._should_use_sglang_local_masked_ep_forward(
                        padding_mask, shared_expert_output
                    ):
                        return self._forward_sglang_local_masked_ep(hidden_states)
                    if self._should_use_sglang_local_masked_ep_straight_through(
                        padding_mask, shared_expert_output, intermediate_tensors
                    ):
                        with torch.no_grad():
                            sglang_exact_output = self._forward_sglang_local_masked_ep(
                                hidden_states
                            )[0]
                    probs, routing_map = self.route(hidden_states, padding_mask)
                    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

                    if intermediate_tensors is not None:
                        return hidden_states, probs, shared_expert_output

            except MoECudaGraphPartialCaptureSignal as e:
                # This signal is raised from the maybe_skip_or_early_return_by_cudagraph decorator.
                # It means we should early-return from the MoE layer forward pass.
                # This happens when we are partially capturing the CUDA graph of the MoE layer,
                # like cuda_graph_scope=["moe_router", "moe_preprocess"].
                # We need to return the intermediate tensors as CUDA graph outputs.
                return e.get_early_return_outputs(hidden_states, shared_expert_output)

            if "expert_compute" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    hidden_states, probs = intermediate_tensors

                dispatched_input, probs = self.dispatch(hidden_states, probs)
                output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
                assert (
                    mlp_bias is None
                ), f"mlp_bias is not supported for {type(self.token_dispatcher)}"
                output = self.combine(output)

                if intermediate_tensors is not None:
                    return output, mlp_bias

            if "postprocess" in self.fwd_execution_map:
                if intermediate_tensors is not None:
                    output, shared_expert_output = intermediate_tensors

                output = self.postprocess(output, shared_expert_output)

                if intermediate_tensors is not None:
                    return output

            if sglang_exact_output is not None:
                output = sglang_exact_output + (output - output.detach())

            return output, mlp_bias

        if self._should_compact_true_on_policy_padding(padding_mask, intermediate_tensors):
            outputs = self._forward_compacted_true_on_policy_padding(
                hidden_states, padding_mask, custom_forward
            )
        elif self.moe_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                outputs = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    padding_mask,
                )
            else:
                outputs = tensor_parallel.checkpoint(
                    custom_forward, False, hidden_states, padding_mask
                )
        else:
            outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask)

        return outputs

    def _should_compact_true_on_policy_padding(
        self,
        padding_mask: Optional[torch.Tensor],
        intermediate_tensors,
    ) -> bool:
        if padding_mask is None or intermediate_tensors is not None:
            return False
        if self.use_shared_expert or self.config.moe_latent_size:
            return False

        from megatron.core.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

        policy = resolve_true_on_policy_runtime_policy(self.config)
        return (
            policy.ep_invariant_moe
            and padding_mask.dtype == torch.bool
            and bool(padding_mask.any().item())
        )

    def _forward_compacted_true_on_policy_padding(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor,
        custom_forward,
    ):
        hidden_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        flat_padding_mask = padding_mask.reshape(-1)
        valid_mask = ~flat_padding_mask

        if not bool(valid_mask.any().item()):
            return torch.zeros_like(hidden_states), None

        compact_hidden_states = flat_hidden_states[valid_mask].contiguous().view(
            -1, 1, hidden_shape[-1]
        )
        compact_output, mlp_bias = custom_forward(compact_hidden_states, None, None)
        if mlp_bias is not None:
            raise AssertionError("MoE true-on-policy padding compaction does not support bias")

        flat_output = compact_output.new_zeros(flat_hidden_states.shape)
        flat_output[valid_mask] = compact_output.reshape(-1, hidden_shape[-1])
        return flat_output.view(hidden_shape), None

    def _should_use_sglang_local_masked_ep_forward(
        self,
        padding_mask: Optional[torch.Tensor],
        shared_expert_output: Optional[torch.Tensor],
    ) -> bool:
        if torch.is_grad_enabled():
            return False
        if self._has_true_on_policy_padding(padding_mask) or shared_expert_output is not None:
            return False
        if self.use_shared_expert or self.config.moe_latent_size:
            return False
        if self.config.moe_router_topk <= 1:
            return False
        if self.config.moe_permute_fusion:
            return False
        if not hasattr(self.experts, "forward_sglang_local_masked"):
            return False

        dispatcher = self.token_dispatcher
        if getattr(dispatcher, "drop_and_pad", False):
            return False
        if getattr(dispatcher, "tp_size", 1) != 1 or getattr(dispatcher, "ep_size", 1) <= 1:
            return False

        from megatron.core.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

        policy = resolve_true_on_policy_runtime_policy(self.config)
        return policy.ep_invariant_moe and policy.deterministic_moe_dispatch

    def _should_use_sglang_local_masked_ep_straight_through(
        self,
        padding_mask: Optional[torch.Tensor],
        shared_expert_output: Optional[torch.Tensor],
        intermediate_tensors,
    ) -> bool:
        if not torch.is_grad_enabled() or intermediate_tensors is not None:
            return False
        if self._has_true_on_policy_padding(padding_mask) or shared_expert_output is not None:
            return False
        if self.use_shared_expert or self.config.moe_latent_size:
            return False
        if self.config.moe_router_topk <= 1:
            return False
        if self.config.moe_permute_fusion:
            return False
        if not hasattr(self.experts, "forward_sglang_local_masked"):
            return False

        dispatcher = self.token_dispatcher
        if getattr(dispatcher, "drop_and_pad", False):
            return False
        if getattr(dispatcher, "tp_size", 1) != 1 or getattr(dispatcher, "ep_size", 1) <= 1:
            return False

        from megatron.core.true_on_policy.contracts import resolve_true_on_policy_runtime_policy

        policy = resolve_true_on_policy_runtime_policy(self.config)
        return policy.ep_invariant_moe and policy.deterministic_moe_dispatch

    @staticmethod
    def _has_true_on_policy_padding(padding_mask: Optional[torch.Tensor]) -> bool:
        return padding_mask is not None and bool(padding_mask.any().item())

    def _forward_sglang_local_masked_ep(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        ep_group = self.token_dispatcher.ep_group
        ep_size = self.token_dispatcher.ep_size
        ep_rank = torch.distributed.get_rank(group=ep_group)
        local_num_tokens = flat_hidden_states.shape[0]
        max_num_tokens, token_counts = self._gather_sglang_ep_token_counts(
            local_num_tokens, flat_hidden_states.device, ep_group, ep_size
        )

        hidden_chunks = self._all_gather_padded_sglang_ep_tensor(
            flat_hidden_states, max_num_tokens, ep_group, ep_size
        )

        if max_num_tokens == 0:
            return torch.zeros_like(hidden_states), None

        rollout_segments = self._gather_sglang_ep_rollout_segments(
            local_num_tokens,
            flat_hidden_states.device,
            ep_group,
            ep_size,
        )
        if rollout_segments is None:
            return self._forward_sglang_local_masked_ep_global_padded(
                hidden_states,
                hidden_chunks,
                max_num_tokens,
                token_counts,
                ep_group,
                ep_rank,
            )

        local_output = flat_hidden_states.new_zeros(flat_hidden_states.shape)
        for source_rank, source_segments in enumerate(rollout_segments):
            source_offset = 0
            for source_num_tokens, source_active_rank in source_segments:
                if source_num_tokens == 0:
                    continue

                source_global_hidden = flat_hidden_states.new_zeros(
                    (ep_size * source_num_tokens, hidden_shape[-1])
                )
                source_start = source_active_rank * source_num_tokens
                source_global_hidden[source_start : source_start + source_num_tokens] = hidden_chunks[
                    source_rank
                ][source_offset : source_offset + source_num_tokens]

                source_probs, _ = apply_module(self.router)(
                    source_global_hidden.view(-1, 1, hidden_shape[-1]), None
                )
                source_output = self.experts.forward_sglang_local_masked(
                    source_global_hidden,
                    source_probs,
                    self.config.moe_router_topk,
                    self.local_expert_indices,
                )
                from megatron.core.true_on_policy.moe import sglang_moe_ep_tree_all_reduce

                source_output = sglang_moe_ep_tree_all_reduce(source_output, ep_group)

                if source_rank == ep_rank:
                    local_output[source_offset : source_offset + source_num_tokens] = source_output[
                        source_start : source_start + source_num_tokens
                    ]
                source_offset += source_num_tokens

        return local_output.view(hidden_shape), None

    def _forward_sglang_local_masked_ep_global_padded(
        self,
        hidden_states: torch.Tensor,
        hidden_chunks: list[torch.Tensor],
        max_num_tokens: int,
        token_counts: list[int],
        ep_group,
        ep_rank: int,
    ):
        hidden_shape = hidden_states.shape
        global_hidden_states = torch.cat(hidden_chunks, dim=0)
        global_probs, _ = apply_module(self.router)(
            global_hidden_states.view(-1, 1, hidden_shape[-1]), None
        )
        global_output = self.experts.forward_sglang_local_masked(
            global_hidden_states,
            global_probs,
            self.config.moe_router_topk,
            self.local_expert_indices,
        )
        from megatron.core.true_on_policy.moe import sglang_moe_ep_tree_all_reduce

        global_output = sglang_moe_ep_tree_all_reduce(global_output, ep_group)

        local_num_tokens = token_counts[ep_rank]
        local_start = ep_rank * max_num_tokens
        local_output = global_output[local_start : local_start + local_num_tokens].contiguous()
        if local_num_tokens < hidden_states.reshape(-1, hidden_shape[-1]).shape[0]:
            padded_output = hidden_states.new_zeros(hidden_states.reshape(-1, hidden_shape[-1]).shape)
            padded_output[:local_num_tokens] = local_output
            local_output = padded_output
        return local_output.view(hidden_shape), None

    def _gather_sglang_ep_rollout_segments(
        self,
        local_num_tokens: int,
        device: torch.device,
        ep_group,
        ep_size: int,
    ) -> list[list[tuple[int, int]]] | None:
        try:
            from megatron.core.true_on_policy.moe import get_sglang_moe_rollout_context
        except Exception:
            return None

        context = get_sglang_moe_rollout_context()
        if context is None or context.rollout_dp_ranks is None:
            return None

        local_token_counts = context.token_counts
        if local_token_counts is None:
            local_token_counts = (local_num_tokens,)
        if sum(local_token_counts) != local_num_tokens:
            return None

        local_active_ranks = context.rollout_dp_ranks
        if len(local_active_ranks) < len(local_token_counts):
            return None

        local_segments = [
            (int(num_tokens), int(active_rank))
            for num_tokens, active_rank in zip(local_token_counts, local_active_ranks, strict=False)
        ]
        if any(active_rank < 0 or active_rank >= ep_size for _, active_rank in local_segments):
            return None

        max_num_segments, segment_counts = self._gather_sglang_ep_token_counts(
            len(local_segments), device, ep_group, ep_size
        )
        if max_num_segments == 0:
            return [[] for _ in range(ep_size)]

        gathered_num_tokens = self._all_gather_padded_sglang_ep_ints(
            [segment[0] for segment in local_segments],
            max_num_segments,
            device,
            ep_group,
            ep_size,
        )
        gathered_active_ranks = self._all_gather_padded_sglang_ep_ints(
            [segment[1] for segment in local_segments],
            max_num_segments,
            device,
            ep_group,
            ep_size,
        )

        rollout_segments: list[list[tuple[int, int]]] = []
        for source_rank, num_segments in enumerate(segment_counts):
            source_segments = []
            for segment_idx in range(num_segments):
                num_tokens = gathered_num_tokens[source_rank][segment_idx]
                active_rank = gathered_active_ranks[source_rank][segment_idx]
                if num_tokens < 0 or active_rank < 0 or active_rank >= ep_size:
                    return None
                source_segments.append((num_tokens, active_rank))
            rollout_segments.append(source_segments)
        return rollout_segments

    @staticmethod
    def _gather_sglang_ep_token_counts(
        local_num_tokens: int,
        device: torch.device,
        ep_group,
        ep_size: int,
    ) -> tuple[int, list[int]]:
        local_count = torch.tensor([local_num_tokens], device=device, dtype=torch.long)
        gathered_counts = [torch.empty_like(local_count) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered_counts, local_count, group=ep_group)
        token_counts = [int(count.item()) for count in gathered_counts]
        return max(token_counts), token_counts

    @staticmethod
    def _all_gather_padded_sglang_ep_tensor(
        local_tensor: torch.Tensor,
        max_num_tokens: int,
        ep_group,
        ep_size: int,
    ) -> list[torch.Tensor]:
        padded_shape = (max_num_tokens, *local_tensor.shape[1:])
        padded_local = local_tensor.new_zeros(padded_shape)
        if local_tensor.shape[0] != 0:
            padded_local[: local_tensor.shape[0]] = local_tensor
        gathered = [torch.empty_like(padded_local) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, padded_local, group=ep_group)
        return gathered

    @staticmethod
    def _all_gather_padded_sglang_ep_ints(
        local_values: list[int],
        max_num_values: int,
        device: torch.device,
        ep_group,
        ep_size: int,
    ) -> list[list[int]]:
        padded_local = torch.full(
            (max_num_values,),
            -1,
            device=device,
            dtype=torch.long,
        )
        if local_values:
            local_tensor = torch.tensor(local_values, device=device, dtype=torch.long)
            padded_local[: local_tensor.shape[0]] = local_tensor
        gathered = [torch.empty_like(padded_local) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, padded_local, group=ep_group)
        return [[int(value.item()) for value in rank_values] for rank_values in gathered]

    def backward_dw(self, routed_experts: bool = True, shared_experts: bool = False):
        """Compute weight gradients for experts and shared experts."""
        if routed_experts:
            self.experts.backward_dw()
        if shared_experts and self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8/fp4."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
