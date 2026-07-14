# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import copy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
    from transformer_engine.pytorch.optimizers import FusedSGD as GPUSGD
except:
    # Handle environment where transformer_engine is not installed
    from torch.optim import SGD as GPUSGD
    from torch.optim import Adam as GPUAdam

from megatron.core.optimizer import OptimizerConfig, _get_megatron_optimizer_based_on_param_groups
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def setup_seed(seed):
    random.seed(seed)  # Set Python's built-in random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('n_steps', [1, 10])
@pytest.mark.parametrize('overlap_cpu_optimizer_d2h_h2d', [False, True])
@pytest.mark.parametrize('offload_fraction', [0, 0.5, 1.0])
@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
@pytest.mark.parametrize('with_param_groups', [False, True])
def test_multi_device_hybrid_optimizer(
    with_param_groups, optimizer, offload_fraction, overlap_cpu_optimizer_d2h_h2d, n_steps
):
    setup_seed(42)
    net1 = Net().cuda()
    net2 = Net().cuda()
    net2.load_state_dict(net1.state_dict())
    base_lr = 1e-3
    params = list(net1.parameters())
    ref_params = list(net2.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": params[len(params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        params = param_groups
        ref_param_groups = [
            {"params": ref_params[: len(ref_params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": ref_params[len(ref_params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        ref_params = ref_param_groups

    if optimizer == 'adam':
        cls_kwargs = dict(cpu_optimizer_cls=Adam, gpu_optimizer_cls=GPUAdam)
    else:
        cls_kwargs = dict(cpu_optimizer_cls=SGD, gpu_optimizer_cls=GPUSGD)

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=base_lr,
        overlap_cpu_optimizer_d2h_h2d=overlap_cpu_optimizer_d2h_h2d,
        **cls_kwargs,
    )

    ref_optimizer = cls_kwargs['gpu_optimizer_cls'](ref_params, lr=base_lr)

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net1(input)
    output.sum().backward()
    hdo.step()
    output = net2(input)
    output.sum().backward()
    ref_optimizer.step()
    # PyTorch SGD will not generate state
    if optimizer != 'sgd':
        assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    if optimizer == 'adam':
        first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
        last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
        if offload_fraction > 0:
            assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
        if offload_fraction < 1:
            assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda

    # 3. check parameters allclose
    for _ in range(1, n_steps):
        input = torch.randn(1, 3, 32, 32).cuda()
        output = net1(input)
        output.sum().backward()
        hdo.step()
        output = net2(input)
        output.sum().backward()
        ref_optimizer.step()

    params = net1.state_dict()
    ref_params = net2.state_dict()
    for k, v in params.items():
        assert (v.isnan() == ref_params[k].isnan()).all()
        torch.nan_to_num_(v, 0)
        torch.nan_to_num_(ref_params[k], 0)
        assert torch.allclose(
            v, ref_params[k], atol=1e-03
        ), f"Weight {k} value mismatch, max error: {(v - ref_params[k]).abs().max()}"


@pytest.mark.skipif(torch.__version__ < '2.3.0', reason="Requires PyTorch 2.3.0 or higher")
@pytest.mark.parametrize("offload_fraction", [0.5, 1.0])
@pytest.mark.parametrize("model_params_dtype", [torch.bfloat16, torch.float16])
def test_precision_aware_cpu_offload_state_dtypes(offload_fraction, model_params_dtype):
    model_params = [
        torch.nn.Parameter(torch.ones(32, dtype=model_params_dtype, device="cuda"))
        for _ in range(2)
    ]
    config = OptimizerConfig(
        optimizer="adam",
        lr=1e-2,
        fp16=model_params_dtype == torch.float16,
        bf16=model_params_dtype == torch.bfloat16,
        params_dtype=model_params_dtype,
        use_distributed_optimizer=True,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=offload_fraction,
        overlap_cpu_optimizer_d2h_h2d=False,
        use_precision_aware_optimizer=True,
        main_params_dtype=torch.float16,
        exp_avg_dtype=torch.float16,
        exp_avg_sq_dtype=torch.float16,
        decoupled_weight_decay=True,
    )

    with (
        patch(
            "megatron.core.optimizer.DistributedOptimizer",
            side_effect=lambda optimizer, *args, **kwargs: optimizer,
        ),
        patch(
            "megatron.core.optimizer.parallel_state.get_tensor_model_parallel_group",
            return_value=None,
        ),
    ):
        optimizer = _get_megatron_optimizer_based_on_param_groups(
            config=config, model_chunks=[], param_groups=[{"params": model_params}]
        )

    assert isinstance(optimizer, HybridDeviceOptimizer)
    assert not optimizer.state
    assert not optimizer.param_to_fp32_param
    assert not optimizer.fp32_param_to_orig_param
    initial_inner_params = {
        model_param: optimizer.param_to_inner_param[model_param].detach().clone()
        for model_param in model_params
    }

    for model_param in model_params:
        model_param.decoupled_grad = torch.ones_like(model_param, dtype=torch.float32)
    optimizer.step()
    for model_param in model_params:
        assert not torch.equal(
            initial_inner_params[model_param], optimizer.param_to_inner_param[model_param]
        )

    def assert_state_dtypes():
        devices = set()
        for model_param in model_params:
            master_param = optimizer.param_to_inner_param[model_param]
            state = optimizer.state[model_param]
            devices.add(master_param.device.type)
            assert state["master_param"] is master_param
            assert master_param.dtype == torch.float16
            assert master_param.element_size() == 2
            assert state["exp_avg"].device == master_param.device
            assert state["exp_avg"].dtype == torch.float16
            assert state["exp_avg_sq"].device == master_param.device
            assert state["exp_avg_sq"].dtype == torch.float16

        expected_devices = {"cpu", "cuda"} if offload_fraction == 0.5 else {"cpu"}
        assert devices == expected_devices

    assert_state_dtypes()

    expected_state = {
        model_param: {
            key: optimizer.state[model_param][key].detach().clone()
            for key in ("master_param", "exp_avg", "exp_avg_sq")
        }
        for model_param in model_params
    }
    state_dict = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        for model_param in model_params:
            inner_param = optimizer.param_to_inner_param[model_param]
            if inner_param is not model_param:
                model_param.fill_(7)
                inner_param.zero_()
            optimizer.state[model_param]["exp_avg"].zero_()
            optimizer.state[model_param]["exp_avg_sq"].zero_()

    optimizer.load_state_dict(state_dict)
    assert_state_dtypes()
    for model_param in model_params:
        state = optimizer.state[model_param]
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key], expected_state[model_param][key], rtol=0, atol=0)

    for model_param in model_params:
        model_param.decoupled_grad = torch.ones_like(model_param, dtype=torch.float32)
    optimizer.step()
    assert_state_dtypes()


def test_precision_aware_cpu_offload_rejects_mixed_state_dtypes():
    with pytest.raises(AssertionError, match="must use the same dtype"):
        OptimizerConfig(
            optimizer="adam",
            use_distributed_optimizer=True,
            optimizer_cpu_offload=True,
            use_precision_aware_optimizer=True,
            main_params_dtype=torch.float32,
            exp_avg_dtype=torch.float16,
            exp_avg_sq_dtype=torch.bfloat16,
        )


def test_hybrid_optimizer_rejects_legacy_fp32_flag_with_low_precision_dtypes():
    with pytest.raises(AssertionError, match="param_update_in_fp32=True conflicts"):
        HybridDeviceOptimizer(
            [],
            param_update_in_fp32=True,
            main_params_dtype=torch.float16,
            exp_avg_dtype=torch.float16,
            exp_avg_sq_dtype=torch.float16,
        )
