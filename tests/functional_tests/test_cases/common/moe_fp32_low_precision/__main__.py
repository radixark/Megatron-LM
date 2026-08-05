# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""8-GPU B200 smoke coverage for FP32 MoE activation/combine low-precision paths.

Launch one EP8 case from the repository root with::

    PYTHONPATH=. python -m torch.distributed.run --standalone --nproc_per_node=8 \
        -m tests.functional_tests.test_cases.common.moe_fp32_low_precision \
        --recipe mxfp8 --mode both

Each invocation validates one recipe/mode/activation/dispatcher combination.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import transformer_engine as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

import megatron.core.parallel_state as parallel_state
import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_module
from megatron.core.activations import squared_relu
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

PRECISIONS = {
    "delayed": {"fp8": "hybrid", "fp8_recipe": "delayed"},
    "tensorwise": {"fp8": "hybrid", "fp8_recipe": "tensorwise"},
    "blockwise": {"fp8": "hybrid", "fp8_recipe": "blockwise"},
    "mxfp8": {"fp8": "hybrid", "fp8_recipe": "mxfp8"},
    "nvfp4": {"fp4": "e2m1", "fp4_recipe": "nvfp4"},
}

EXPECTED_RECIPE_CLASSES = {
    "delayed": "TEDelayedScaling",
    "tensorwise": "Float8CurrentScaling",
    "blockwise": "Float8BlockScaling",
    "mxfp8": "MXFP8BlockScaling",
    "nvfp4": "NVFP4BlockScaling",
}


def graph_contains(tensor, needle):
    stack = [tensor.grad_fn]
    seen = set()
    while stack:
        function = stack.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        if needle in type(function).__name__:
            return True
        stack.extend(next_function for next_function, _ in function.next_functions)
    return False


def make_hidden(device, rank, world_size, step, hidden_size=256):
    generator = torch.Generator(device=device)
    generator.manual_seed(9000 + 101 * rank + step)
    hidden = 0.01 * torch.randn(
        (33, 2, hidden_size), device=device, dtype=torch.bfloat16, generator=generator
    )

    # Force deterministic top-2 routing. Every source rank sends tokens to
    # every expert rank, with unaligned expert token counts for TE padding.
    flat = hidden.view(-1, hidden_size)
    flat[:, :world_size].zero_()
    for token in range(flat.size(0)):
        first = (token + step) % world_size
        second = (first + 1) % world_size
        flat[token, first] = 4.0
        flat[token, second] = 2.0

    return hidden.requires_grad_()


def assert_finite_grads(layer, hidden):
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.float().abs().sum() > 0

    for prefix in ("router.weight", "experts.linear_fc1", "experts.linear_fc2"):
        params = [
            (name, param) for name, param in layer.named_parameters() if name.startswith(prefix)
        ]
        assert params, f"no parameters matched {prefix}"

        grad_sum = 0.0
        for name, param in params:
            assert param.grad is not None, f"missing grad: {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad: {name}"
            grad_sum += param.grad.float().abs().sum().item()
        assert grad_sum > 0, f"zero aggregate grad: {prefix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, choices=PRECISIONS)
    parser.add_argument(
        "--mode", default="both", choices=("control", "activation", "combine", "both")
    )
    parser.add_argument("--activation", default="swiglu", choices=("swiglu", "squared-relu"))
    parser.add_argument("--dispatcher", default="alltoall", choices=("alltoall", "allgather"))
    parser.add_argument("--fp8-format", default="hybrid", choices=("e4m3", "hybrid"))
    parser.add_argument("--permute-fusion", action="store_true")
    cli = parser.parse_args()

    combine = cli.mode in ("combine", "both")
    if combine and cli.dispatcher != "alltoall":
        parser.error("combine mode requires --dispatcher alltoall")
    if combine and cli.permute_fusion:
        parser.error("combine mode does not support --permute-fusion")

    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 8, f"expected the 8-GPU devbox, got WORLD_SIZE={world_size}"
    assert (
        local_world_size == 8
    ), f"expected a single 8-GPU node, got LOCAL_WORLD_SIZE={local_world_size}"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    device_name = torch.cuda.get_device_name(local_rank)
    assert capability[0] == 10, f"expected Blackwell on rank {local_rank}, got {capability}"
    assert "B200" in device_name, f"expected B200 on rank {local_rank}, got {device_name}"

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=world_size,
        expert_tensor_parallel_size=1,
    )
    padding_hook = None
    original_unpermute = None
    try:
        rank = torch.distributed.get_rank()
        _set_random_seed(seed_=1234, data_parallel_random_init=False)

        activation = cli.mode in ("activation", "both")
        squared_relu_activation = cli.activation == "squared-relu"
        precision_config = PRECISIONS[cli.recipe].copy()
        if "fp8" in precision_config:
            precision_config["fp8"] = cli.fp8_format

        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            ffn_hidden_size=512,
            num_attention_heads=8,
            num_moe_experts=world_size,
            moe_ffn_hidden_size=512,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=world_size,
            expert_tensor_parallel_size=1,
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            transformer_impl="transformer_engine",
            moe_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_token_dispatcher_type=cli.dispatcher,
            moe_router_load_balancing_type="none",
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_aux_loss_coeff=0.0,
            moe_permute_fusion=cli.permute_fusion,
            moe_router_padding_for_quantization=False,
            moe_apply_probs_on_input=False,
            add_bias_linear=True,
            gated_linear_unit=not squared_relu_activation,
            activation_func=squared_relu if squared_relu_activation else F.silu,
            use_fused_weighted_squared_relu=squared_relu_activation,
            bias_activation_fusion=False,
            moe_activation_in_fp32=activation,
            moe_combine_in_fp32=combine,
            **precision_config,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=world_size, moe_grouped_gemm=True
        )
        layer = MoELayer(config, layer_spec.submodules.mlp.submodules, layer_number=1).to(device)
        layer.train()

        assert isinstance(layer.experts, TEGroupedMLP)
        expected_dispatcher_type = (
            MoEAlltoAllTokenDispatcher
            if cli.dispatcher == "alltoall"
            else MoEAllGatherTokenDispatcher
        )
        assert isinstance(layer.token_dispatcher, expected_dispatcher_type)
        assert layer.experts.num_local_experts == 1
        assert layer.token_dispatcher.ep_size == world_size
        assert hasattr(layer.experts, "quantization_padding")

        padding_calls = 0
        padding_observed = False

        def record_padding(_module, call_args, result):
            nonlocal padding_calls, padding_observed
            original_counts = call_args[1]
            padded_counts = result[1]
            padding_calls += 1
            padding_observed |= any(
                int(padded) > int(original)
                for padded, original in zip(padded_counts, original_counts)
            )

        padding_hook = layer.experts.quantization_padding.register_forward_hook(record_padding)

        with torch.no_grad():
            layer.router.weight.zero_()
            for expert in range(world_size):
                layer.router.weight[expert, expert] = 1.0

        original_unpermute = token_dispatcher_module.unpermute
        combine_calls = 0

        def traced_unpermute(*call_args, **call_kwargs):
            nonlocal combine_calls
            probs = call_kwargs.get("probs")
            if probs is None and len(call_args) > 3:
                probs = call_args[3]
            if probs is not None:
                assert combine
                assert probs.dtype == torch.float32
                combine_calls += 1
            return original_unpermute(*call_args, **call_kwargs)

        token_dispatcher_module.unpermute = traced_unpermute

        losses = []
        active_recipe_name = None

        for step in range(2):
            layer.zero_grad(set_to_none=True)
            hidden = make_hidden(device, rank, world_size, step)

            quant_context = (
                get_fp8_context(config, layer_no=0)
                if config.fp8
                else get_fp4_context(config, layer_no=0)
            )

            with quant_context:
                enabled = FP8GlobalStateManager.is_fp8_enabled()
                assert enabled

                active_recipe_name = type(FP8GlobalStateManager.get_fp8_recipe()).__name__
                assert active_recipe_name == EXPECTED_RECIPE_CLASSES[cli.recipe]
                assert layer.experts.linear_fc1.will_execute_quantized(enabled)
                assert layer.experts.linear_fc2.will_execute_quantized(enabled)

                output, bias = layer(hidden)

            if cli.dispatcher == "alltoall":
                assert all(int(split) > 0 for split in layer.token_dispatcher.input_splits)
                assert all(int(split) > 0 for split in layer.token_dispatcher.output_splits)
                routed_probs = layer.token_dispatcher.probs
            else:
                assert layer.token_dispatcher.hidden_shape_before_permute[0] == 66 * world_size
                routed_probs = layer.token_dispatcher.local_probs

            assert bias is None
            assert output.shape == hidden.shape
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output).all()
            assert routed_probs.dtype == torch.float32
            assert graph_contains(output, "MoEActivationInFP32") == activation

            # Keep the upstream gradient O(1) so FP8 backward quantization does not
            # erase the signal before these smoke-test gradient assertions.
            loss = output.float().sum()
            loss.backward()
            assert_finite_grads(layer, hidden)
            losses.append(loss.detach())

        assert combine_calls == (2 if combine else 0)
        assert padding_calls == 4
        padding_observed_any_rank = torch.tensor(
            int(padding_observed), device=device, dtype=torch.int32
        )
        torch.distributed.all_reduce(padding_observed_any_rank, op=torch.distributed.ReduceOp.MAX)
        assert padding_observed_any_rank.item() == 1

        torch.cuda.synchronize()

        if rank == 0:
            print(
                f"PASS recipe={cli.recipe} active={active_recipe_name} mode={cli.mode} "
                f"fp8_format={config.fp8 or 'none'} "
                f"activation={cli.activation} dispatcher={cli.dispatcher} "
                f"permute_fusion={cli.permute_fusion} "
                f"te={te.__version__} gpu={torch.cuda.get_device_name(0)} "
                f"cc={torch.cuda.get_device_capability(0)} "
                f"losses={[loss.item() for loss in losses]}"
            )
    finally:
        try:
            if padding_hook is not None:
                padding_hook.remove()
        finally:
            if original_unpermute is not None:
                token_dispatcher_module.unpermute = original_unpermute
            parallel_state.destroy_model_parallel()
            Utils.inited = False
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
