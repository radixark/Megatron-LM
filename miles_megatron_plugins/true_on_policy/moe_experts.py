"""``SGLangGroupedMLP`` weight-layout adapter for SGLang MoE kernels.

This module subclasses Megatron's ``GroupedMLP`` and exposes weight views in
SGLang's expected layout.  It contains NO forward logic; the canonical
forward lives in ``sgl_fused_moe.forward.sglang_moe_forward``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.moe_utils import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class SGLangGroupedMLP(GroupedMLP):
    """GroupedMLP subclass that provides SGLang-compatible weight accessors.

    The only responsibility of this class is to reshape/permute Megatron's
    weight tensors into the layout that SGLang's fused_experts kernel expects:
      - w1 (gate+up): [num_local_experts, hidden_size, 2*intermediate] ->
                       SGLang wants [num_local_experts, 2*intermediate, hidden_size]
      - w2 (down):    [num_local_experts, intermediate, hidden_size] ->
                       SGLang wants [num_local_experts, hidden_size, intermediate]

    All forward computation is in sgl_fused_moe.forward or autograd.py.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        MegatronModule.__init__(self, config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        if pg_collection is None:
            raise ValueError("SGLangGroupedMLP requires a ProcessGroupCollection")
        assert (
            config.add_bias_linear == False
        ), "bias not supported in SGLangGroupedMLP, please set '--disable-bias-linear'."
        assert (
            config.moe_latent_size is None
        ), "MoE latent projection not supported in SGLangGroupedMLP."

        self.expert_parallel = config.expert_model_parallel_size > 1
        self.ep_group = pg_collection.ep
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp

        tp_size = self.tp_group.size()
        tp_rank = self.tp_group.rank()

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                    rank=tp_rank,
                    world_size=tp_size,
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1, config.init_method, partition_dim=1, is_expert=True
                )
                _initialize_affine_weight_gpu(
                    self.weight2, config.output_layer_init_method, partition_dim=0, is_expert=True
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight1, is_parallel=True, dim=1, stride=1
                )
                set_tensor_model_parallel_attributes(
                    tensor=self.weight2, is_parallel=True, dim=0, stride=1
                )
        setattr(self.weight1, "allreduce", not self.expert_parallel)
        setattr(self.weight2, "allreduce", not self.expert_parallel)

        def remove_extra_states_check(self, incompatible_keys):
            keys = list(incompatible_keys.unexpected_keys)
            for key in keys:
                if "_extra_state" in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "SGLangGroupedMLP is only a weight-layout adapter; MoE execution must "
            "go through the true-on-policy SGLang fused MoE path."
        )

    def sglang_w13_weight(self) -> torch.Tensor:
        """Return w1 (gate+up projection) in SGLang layout: [E, 2*I, H]."""
        return self._cached_sglang_weight(
            cache_name="_sglang_w13_weight_cache",
            weight=self.weight1,
            view_shape=(self.num_local_experts, self.config.hidden_size, -1),
            permute_dims=(0, 2, 1),
        )

    def sglang_w2_weight(self) -> torch.Tensor:
        """Return w2 (down projection) in SGLang layout: [E, H, I]."""
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
