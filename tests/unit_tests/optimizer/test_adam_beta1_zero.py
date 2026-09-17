# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import io

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.optimizer import (
    _initialize_adam_beta1_zero_state,
    _step_with_adam_beta1_zero,
    _strip_adam_beta1_zero_state,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def _initialize_distributed():
    Utils.initialize_distributed()


def _parameters(device):
    return [
        torch.nn.Parameter(torch.linspace(-0.8, 0.9, size, device=device))
        for size in (17, 10, 1, 3)
    ]


def _groups(params):
    return [
        {"params": params[:2], "lr": 0.025, "weight_decay": 0.125},
        {"params": params[2:], "lr": 0.01, "weight_decay": 0.05},
    ]


def _omit_first_moment(optimizer):
    optimizer.omit_exp_avg = True
    return optimizer


def _optimizers(backend, params, reference_params):
    kwargs = dict(betas=(0.0, 0.95), eps=1e-7)
    if backend == "cpu":
        return (
            _omit_first_moment(torch.optim.AdamW(_groups(params), fused=True, **kwargs)),
            torch.optim.AdamW(_groups(reference_params), fused=True, **kwargs),
        )

    from transformer_engine.pytorch.optimizers import FusedAdam

    return (
        _omit_first_moment(FusedAdam(_groups(params), adam_w_mode=True, **kwargs)),
        FusedAdam(_groups(reference_params), adam_w_mode=True, **kwargs),
    )


def _set_gradients(params, step):
    for index, param in enumerate(params):
        # One parameter never receives a gradient; another skips intermittent steps.
        if index == 3 or (index == 1 and step % 3 == 1):
            param.grad = None
        else:
            values = torch.arange(param.numel(), dtype=torch.float32, device=param.device)
            param.grad = ((values + index + step).sin() * 0.2).reshape_as(param).to(param.dtype)


def _assert_no_first_moment(optimizer):
    for state in optimizer.state.values():
        assert "exp_avg" not in state
    for state in optimizer.state_dict()["state"].values():
        assert "exp_avg" not in state


def _assert_bitwise(actual, reference):
    torch.testing.assert_close(
        actual.detach().reshape(-1).view(torch.uint8),
        reference.detach().reshape(-1).view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _assert_matches(optimizer, reference, params, reference_params, backend):
    # Compare against the fused optimizers selected by Megatron on both backends.
    for param, reference_param in zip(params, reference_params):
        _assert_bitwise(param, reference_param)
        reference_state = reference.state.get(reference_param, {})
        if "exp_avg_sq" in reference_state:
            _assert_bitwise(optimizer.state[param]["exp_avg_sq"], reference_state["exp_avg_sq"])
        if "step" in reference_state:
            torch.testing.assert_close(optimizer.state[param]["step"], reference_state["step"])
    _assert_no_first_moment(optimizer)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("initialize_state", [False, True])
def test_adam_beta1_zero_matches_existing_optimizer(backend, initialize_state):
    params = _parameters(backend)
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer, reference = _optimizers(backend, params, reference_params)
    if initialize_state:
        for param in params:
            _initialize_adam_beta1_zero_state(optimizer, param)
        _assert_no_first_moment(optimizer)

    for step in range(6):
        _set_gradients(params, step)
        _set_gradients(reference_params, step)
        gradients = [None if param.grad is None else param.grad.clone() for param in params]
        _step_with_adam_beta1_zero(optimizer)
        reference.step()
        _assert_matches(optimizer, reference, params, reference_params, backend)
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                torch.testing.assert_close(param.grad, gradient, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_adam_beta1_zero_checkpoint_continuation(backend, legacy_checkpoint):
    params = _parameters(backend)
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer, reference = _optimizers(backend, params, reference_params)
    for step in range(3):
        _set_gradients(params, step)
        _set_gradients(reference_params, step)
        _step_with_adam_beta1_zero(optimizer)
        reference.step()

    checkpoint = reference.state_dict() if legacy_checkpoint else optimizer.state_dict()
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    restored_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    restored, _ = _optimizers(backend, restored_params, reference_params)
    restored.load_state_dict(
        _strip_adam_beta1_zero_state(restored, torch.load(stream, weights_only=False))
    )
    _assert_no_first_moment(restored)

    for step in range(3, 6):
        _set_gradients(restored_params, step)
        _set_gradients(reference_params, step)
        _step_with_adam_beta1_zero(restored)
        reference.step()
        _assert_matches(restored, reference, restored_params, reference_params, backend)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_adam_beta1_zero_cpu_vectorized_rounding(weight_decay):
    """Repeated vector updates retain parity near cancellation in the parameter update."""
    generator = torch.Generator().manual_seed(1234)
    param = torch.nn.Parameter(torch.randn(1000000, generator=generator).mul_(0.1))
    param.grad = torch.randn(param.shape, generator=generator).mul_(0.01)
    reference_param = torch.nn.Parameter(param.detach().clone())
    reference_param.grad = param.grad.clone()
    kwargs = dict(lr=0.001, betas=(0.0, 0.95), eps=1e-8, weight_decay=weight_decay)
    optimizer = _omit_first_moment(torch.optim.AdamW([param], fused=True, **kwargs))
    reference = torch.optim.AdamW([reference_param], fused=True, **kwargs)
    for _ in range(22):
        _step_with_adam_beta1_zero(optimizer)
        reference.step()
    _assert_matches(optimizer, reference, [param], [reference_param], "cpu")
    torch.testing.assert_close(param.grad, reference_param.grad, rtol=0, atol=0)
    for index in range(8):
        for opt in (optimizer, reference):
            opt.param_groups[0]["lr"] = 0.001 * (index + 1) / 8
            opt.param_groups[0]["weight_decay"] = (
                weight_decay + index * 0.001 if weight_decay else 0.0
            )
            _step_with_adam_beta1_zero(opt)
        _assert_matches(optimizer, reference, [param], [reference_param], "cpu")


@pytest.mark.parametrize("numel", [17, 65536])
def test_adam_beta1_zero_cpu_weight_decay_schedule(numel):
    """Switching decay off/on preserves stock AdamW's update and state."""
    param = torch.nn.Parameter(torch.linspace(-0.1, 0.1, numel))
    param.grad = torch.linspace(0.01, 0.02, numel)
    reference_param = torch.nn.Parameter(param.detach().clone())
    reference_param.grad = param.grad.clone()
    kwargs = dict(lr=0.001, betas=(0.0, 0.95), weight_decay=0.0)
    optimizer = _omit_first_moment(torch.optim.AdamW([param], fused=True, **kwargs))
    reference = torch.optim.AdamW([reference_param], fused=True, **kwargs)
    for weight_decay in (0.0, 0.1, 0.0, 0.25, 0.0, 0.5, 0.0):
        for opt in (optimizer, reference):
            opt.param_groups[0]["weight_decay"] = weight_decay
            _step_with_adam_beta1_zero(opt)
        _assert_matches(optimizer, reference, [param], [reference_param], "cpu")


@pytest.mark.parametrize("values", [[0.0, -0.0], [-0.0] * 17])
def test_adam_beta1_zero_cpu_signed_zero_values(values):
    """Aliasing may change a zero's sign bit while preserving its numerical value."""
    param = torch.nn.Parameter(torch.tensor(values))
    reference_param = torch.nn.Parameter(param.detach().clone())
    param.grad = torch.tensor(values)
    reference_param.grad = param.grad.clone()
    saved_gradient = param.grad.clone()
    kwargs = dict(lr=0.001, betas=(0.0, 0.95), eps=1e-8, weight_decay=0.0)
    optimizer = _omit_first_moment(torch.optim.AdamW([param], fused=True, **kwargs))
    reference = torch.optim.AdamW([reference_param], fused=True, **kwargs)
    for _ in range(3):
        _step_with_adam_beta1_zero(optimizer)
        reference.step()
        # Stock scalar/SIMD lerp can choose different zero signs when momentum
        # aliases the gradient. Both parameter and gradient values remain zero.
        torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
        torch.testing.assert_close(param.grad, saved_gradient, rtol=0, atol=0)
        assert param.count_nonzero().item() == 0
        _assert_bitwise(
            optimizer.state[param]["exp_avg_sq"], reference.state[reference_param]["exp_avg_sq"]
        )
        _assert_no_first_moment(optimizer)


def test_adam_beta1_zero_cpu_step_hooks_and_closure():
    param = torch.nn.Parameter(torch.ones(17))
    optimizer = _omit_first_moment(torch.optim.AdamW([param], betas=(0.0, 0.999), fused=True))
    events = []

    def pre_hook(opt, args, kwargs):
        events.append("pre")
        _assert_no_first_moment(opt)

    def post_hook(opt, args, kwargs):
        events.append("post")
        assert opt.state[param]["exp_avg"] is param.grad

    def closure():
        events.append("closure")
        assert torch.is_grad_enabled()
        param.grad = torch.full_like(param, 0.125)
        return 42

    optimizer.register_step_pre_hook(pre_hook)
    optimizer.register_step_post_hook(post_hook)
    assert _step_with_adam_beta1_zero(optimizer, closure) == 42
    assert events == ["pre", "closure", "post"]
    assert optimizer.state[param]["step"].item() == 1
    _assert_no_first_moment(optimizer)


def test_adam_beta1_zero_cpu_removes_alias_after_step_error(monkeypatch):
    param = torch.nn.Parameter(torch.ones(17))
    param.grad = torch.full_like(param, 0.125)
    optimizer = _omit_first_moment(torch.optim.AdamW([param], betas=(0.0, 0.999), fused=True))

    def fail_step(closure=None):
        assert optimizer.state[param]["exp_avg"] is param.grad
        raise RuntimeError("injected stock step error")

    monkeypatch.setattr(optimizer, "step", fail_step)
    with pytest.raises(RuntimeError, match="injected stock step error"):
        _step_with_adam_beta1_zero(optimizer)
    _assert_no_first_moment(optimizer)
    assert optimizer.state[param]["step"].item() == 0


@pytest.mark.parametrize("store_param_remainders", [False, True])
def test_adam_beta1_zero_decoupled_grad_and_bf16_master(store_param_remainders):
    from transformer_engine.pytorch.optimizers import FusedAdam

    params = [torch.nn.Parameter(p.detach().bfloat16()) for p in _parameters("cuda")]
    reference_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    kwargs = dict(
        lr=0.025,
        betas=(0.0, 0.95),
        eps=1e-7,
        weight_decay=0.125,
        master_weights=True,
        use_decoupled_grad=True,
        store_param_remainders=store_param_remainders,
    )
    optimizer = _omit_first_moment(FusedAdam(params, **kwargs))
    reference = FusedAdam(reference_params, **kwargs)
    for step in range(6):
        for index, (param, reference_param) in enumerate(zip(params, reference_params)):
            values = torch.arange(param.numel(), dtype=torch.float32, device="cuda")
            grad = None if index == 3 or (index == 1 and step == 2) else (values + step).sin()
            param.decoupled_grad = grad
            reference_param.decoupled_grad = None if grad is None else grad.clone()
        _step_with_adam_beta1_zero(optimizer)
        reference.step()
        _assert_matches(optimizer, reference, params, reference_params, "cuda")
        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(
                optimizer.state[param]["master_param"],
                reference.state[reference_param]["master_param"],
                rtol=0,
                atol=0,
            )
            if param.decoupled_grad is not None:
                torch.testing.assert_close(
                    param.decoupled_grad, reference_param.decoupled_grad, rtol=0, atol=0
                )
        if step == 2:
            optimizer.load_state_dict(
                _strip_adam_beta1_zero_state(optimizer, copy.deepcopy(optimizer.state_dict()))
            )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_adam_beta1_zero_rejects_restored_nonzero_beta1(backend):
    params = _parameters(backend)
    reference_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer, _ = _optimizers(backend, params, reference_params)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0]["betas"] = (0.9, 0.95)
    with pytest.raises(ValueError, match="beta1"):
        optimizer.load_state_dict(_strip_adam_beta1_zero_state(optimizer, checkpoint))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adam_beta1_zero_hybrid_full_cpu_offload(overlap, dtype):
    from transformer_engine.pytorch.optimizers import FusedAdam

    params = [torch.nn.Parameter(param.detach().to(dtype)) for param in _parameters("cuda")[:3]]
    reference_params = [torch.nn.Parameter(param.detach().float().cpu()) for param in params]
    kwargs = dict(lr=0.025, betas=(0.0, 0.95), eps=1e-7, weight_decay=0.125)

    def make_hybrid(model_params):
        return HybridDeviceOptimizer(
            model_params,
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=FusedAdam,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=overlap,
            fused=True,
            **kwargs,
        )

    optimizer = make_hybrid(params)
    reference = torch.optim.AdamW(reference_params, fused=True, **kwargs)
    assert optimizer.gpu_optimizer is None
    assert len(optimizer.cpu_optimizers) == (len(params) if overlap else 1)

    for step in range(5):
        for index, (param, reference_param) in enumerate(zip(params, reference_params)):
            if index == 1 and step == 1:
                param.grad = None
                reference_param.grad = None
                continue
            gradient = torch.full_like(param, (index + step + 1) / 32)
            param.grad = gradient
            reference_param.grad = gradient.float().cpu()
        optimizer.step()
        reference.step()
        _assert_no_first_moment(optimizer)
        for child in optimizer.cpu_optimizers:
            _assert_no_first_moment(child)
        for param, reference_param in zip(params, reference_params):
            inner_param = optimizer.param_to_inner_param[param]
            _assert_bitwise(inner_param, reference_param)
            _assert_bitwise(
                optimizer.state[param]["exp_avg_sq"], reference.state[reference_param]["exp_avg_sq"]
            )
            _assert_bitwise(param.cpu(), reference_param.detach().to(dtype))

        if step == 2:
            # HDO assigns child state directly during load: child load hooks are insufficient.
            checkpoint = copy.deepcopy(optimizer.state_dict())
            for state in checkpoint["state"].values():
                state["exp_avg"] = torch.ones_like(state["exp_avg_sq"])
            params = [torch.nn.Parameter(param.detach().clone()) for param in params]
            optimizer = make_hybrid(params)
            optimizer.load_state_dict(checkpoint)
            _assert_no_first_moment(optimizer)
            for child in optimizer.cpu_optimizers:
                _assert_no_first_moment(child)


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_adam_beta1_zero_rejects_partial_cpu_offload(fraction):
    with pytest.raises(AssertionError, match="only full optimizer offload"):
        OptimizerConfig(
            adam_beta1=0.0, optimizer_cpu_offload=True, optimizer_offload_fraction=fraction
        )


@pytest.mark.parametrize("beta1,fraction", [(0.0, 1.0), (0.9, 0.5)])
def test_adam_beta1_zero_offload_config_accepts_supported_combinations(beta1, fraction):
    OptimizerConfig(
        adam_beta1=beta1, optimizer_cpu_offload=True, optimizer_offload_fraction=fraction
    )


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("sharding_type", ["dp_reshardable", "fully_reshardable"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adam_beta1_zero_distributed_checkpoint(
    tmp_path_dist_ckpt, cpu_offload, sharding_type, dtype
):
    from transformer_engine.pytorch.optimizers import FusedAdam

    Utils.initialize_model_parallel()

    def make_optimizer():
        config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=1, bf16=True)
        module = torch.nn.Linear(32, 32, bias=False, dtype=dtype, device="cuda")
        module.weight.data.fill_(0.5)
        model = DistributedDataParallel(
            config, DistributedDataParallelConfig(use_distributed_optimizer=True), module
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.025,
                adam_beta1=0.0,
                adam_beta2=0.95,
                weight_decay=0.125,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=cpu_offload,
                optimizer_cpu_offload=cpu_offload,
                optimizer_offload_fraction=1.0 if cpu_offload else 0.0,
            ),
            [model],
        )
        inner = optimizer.chained_optimizers[0].optimizer
        if cpu_offload:
            assert isinstance(inner, HybridDeviceOptimizer)
            assert inner.gpu_optimizer is None
            assert all(type(child) is torch.optim.AdamW for child in inner.cpu_optimizers)
            assert all(child.omit_exp_avg for child in inner.cpu_optimizers)
        else:
            assert type(inner) is FusedAdam
        assert inner.omit_exp_avg
        return model, optimizer, inner

    def step(model, optimizer, value):
        optimizer.zero_grad()
        model.zero_grad_buffer()
        for param in model.parameters():
            param.main_grad.fill_(value)
        assert optimizer.step()[0]

    def sharded_state(model, optimizer, is_loading=False):
        model_state = {
            name: ShardedTensor.from_rank_offsets(
                name, param, replica_id=(0, 0, torch.distributed.get_rank())
            )
            for name, param in model.module.named_parameters()
        }
        return {
            "optimizer": optimizer.sharded_state_dict(
                model_state,
                is_loading=is_loading,
                metadata={"distrib_optim_sharding_type": sharding_type},
            )
        }

    try:
        model, optimizer, inner = make_optimizer()
        step(model, optimizer, 0.125)
        step(model, optimizer, 0.25)
        _assert_no_first_moment(inner)
        with TempNamedDir(tmp_path_dist_ckpt / "adam_beta1_zero", sync=True) as checkpoint_dir:
            dist_checkpointing.save(sharded_state(model, optimizer), checkpoint_dir)
            restored_model, restored, restored_inner = make_optimizer()
            destination = sharded_state(restored_model, restored, is_loading=True)
            _assert_no_first_moment(restored_inner)
            restored.load_state_dict(
                dist_checkpointing.load(destination, checkpoint_dir)["optimizer"]
            )
            _assert_no_first_moment(restored_inner)
            step(model, optimizer, 0.375)
            step(restored_model, restored, 0.375)
            torch.testing.assert_close(
                restored_model.module.weight, model.module.weight, rtol=0, atol=0
            )
            for group, restored_group in zip(inner.param_groups, restored_inner.param_groups):
                for param, restored_param in zip(group["params"], restored_group["params"]):
                    torch.testing.assert_close(
                        restored_inner.state[restored_param]["exp_avg_sq"],
                        inner.state[param]["exp_avg_sq"],
                        rtol=0,
                        atol=0,
                    )
    finally:
        Utils.destroy_model_parallel()
