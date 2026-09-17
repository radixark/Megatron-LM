# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint continuation with beta1-zero Adam and chunked GPU state offload."""

import io

import torch
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
    ChunkedOptimizerStateOffloader,
)
from megatron.core.optimizer.optimizer import (
    _initialize_adam_beta1_zero_state,
    _step_with_adam_beta1_zero,
)
from tests.unit_tests.test_utilities import Utils


def test_adam_beta1_zero_chunked_state_checkpoint_continuation():
    """Only variance is staged, including a reload that bypasses optimizer.load_state_dict."""
    Utils.initialize_distributed()
    params = [
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, count, device="cuda"))
        for count in (128, 64, 32)
    ]
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    options = dict(lr=0.01, betas=(0.0, 0.95), eps=1e-8, weight_decay=0.1, adam_w_mode=True)

    def groups(parameters):
        return [{"params": parameters[:2]}, {"params": parameters[2:], "lr": 0.02}]

    def initialize_state(optimizer, config):
        for group in optimizer.param_groups:
            for param in group["params"]:
                _initialize_adam_beta1_zero_state(optimizer, param)

    def make_offloaded(parameters):
        optimizer = FusedAdam(groups(parameters), **options)
        optimizer.omit_exp_avg = True
        manager = ChunkedOptimizerStateOffloader(
            optimizer,
            master_params=[],
            chunk_size_bytes=512,
            offload_fraction=1.0,
            state_dtypes=(torch.float32,),
            step_fn=lambda: _step_with_adam_beta1_zero(optimizer),
        )
        assert len(manager.chunks) == 2
        manager.initialize_state_for_loading(initialize_state, None)
        return optimizer, manager

    def assert_cpu_canonical(optimizer, manager):
        for param in manager.selected_params:
            assert set(manager._cpu_state[param]) == {"exp_avg_sq"}
            variance = optimizer.state[param]["exp_avg_sq"]
            assert variance.device.type == "cpu"
            assert variance.is_pinned()
            assert "exp_avg" not in optimizer.state[param]

    optimizer, manager = make_offloaded(params)
    reference = FusedAdam(groups(reference_params), **options)
    assert_cpu_canonical(optimizer, manager)
    for step in range(5):
        for index, (param, reference_param) in enumerate(zip(params, reference_params)):
            if index == 1 and step == 1:
                param.grad = reference_param.grad = None
            else:
                gradient = torch.full_like(param, (step + index + 1) / 32)
                param.grad = gradient
                reference_param.grad = gradient.clone()
        manager.step()
        reference.step()
        manager.synchronize_for_checkpoint()
        assert_cpu_canonical(optimizer, manager)
        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
            torch.testing.assert_close(
                optimizer.state[param]["exp_avg_sq"],
                reference.state[reference_param]["exp_avg_sq"].cpu(),
                rtol=0,
                atol=0,
            )
        assert [group["step"] for group in optimizer.param_groups] == [step + 1, step + 1]

        if step == 1:
            # Saving with gradients cleared must not retain a first-moment alias.
            optimizer.zero_grad()
            checkpoint = optimizer.state_dict()
            assert all("exp_avg" not in state for state in checkpoint["state"].values())
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=False)
            # Legacy checkpoints may still carry momentum; the offloader must
            # discard it before restoring canonical CPU state.
            for state in checkpoint["state"].values():
                state["exp_avg"] = torch.ones_like(state["exp_avg_sq"])
            params = [torch.nn.Parameter(param.detach().clone()) for param in params]
            optimizer, manager = make_offloaded(params)
            manager.load_state_dict_without_device_cast(checkpoint)
            assert_cpu_canonical(optimizer, manager)
