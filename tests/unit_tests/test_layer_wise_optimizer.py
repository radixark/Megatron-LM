# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer import layer_wise_optimizer as layer_wise_optimizer_module
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import Utils

skip_layerwise_lts = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip the legacy layer-wise optimizer suite for LTS images",
)


class _RankGroup:
    """Minimal process-group stand-in for duplicate-filter tests."""

    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank


@pytest.mark.parametrize("expert_rank", [0, 1])
def test_shard_params_keeps_expert_plane_when_dense_dp_is_singleton(monkeypatch, expert_rank):
    """Expert ownership must not be disabled by a singleton dense-DP group."""
    dp_group = object()
    expert_dp_group = object()
    monkeypatch.setattr(
        layer_wise_optimizer_module, "get_pg_size", lambda group: 1 if group is dp_group else 2
    )
    monkeypatch.setattr(
        layer_wise_optimizer_module,
        "get_pg_rank",
        lambda group: 0 if group is dp_group else expert_rank,
    )

    dense_param = nn.Parameter(torch.ones(1))
    expert_params = [nn.Parameter(torch.ones(size)) for size in (1, 2, 3)]
    base_optimizer = torch.optim.SGD(
        [
            {"params": [dense_param], "is_expert_parallel": False},
            {"params": expert_params, "is_expert_parallel": True},
        ],
        lr=0.1,
    )
    optimizer = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    optimizer.pg_collection = SimpleNamespace(dp_cp=dp_group, expt_dp=expert_dp_group)

    optimizer.shard_params([base_optimizer])

    assert optimizer.dp_cp_params_list is None
    assert optimizer.expt_dp_params_list is not None
    owned_ids = [id(param) for shard in optimizer.expt_dp_params_list for param in shard]
    assert sorted(owned_ids) == sorted(id(param) for param in expert_params)
    local_ids = {id(param) for param in base_optimizer.param_groups[1]["params"]}
    assert local_ids == {id(param) for param in optimizer.expt_dp_params_list[expert_rank]}
    assert [id(param) for param in base_optimizer.param_groups[0]["params"]] == [id(dense_param)]


def test_allgather_params_ignores_completely_empty_plane(monkeypatch):
    """An ownership plane with no parameters is a valid no-op."""
    optimizer = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    optimizer.pg_collection = SimpleNamespace(dp_cp=object(), expt_dp=object())
    optimizer.dp_cp_params_list = [[], []]
    optimizer.expt_dp_params_list = None

    def fail_all_gather(*args, **kwargs):
        raise AssertionError("all_gather must not run for a completely empty plane")

    monkeypatch.setattr(torch.distributed, "all_gather", fail_all_gather)
    optimizer.allgather_params()


def test_allgather_params_accepts_empty_first_shard(monkeypatch):
    """Device and dtype discovery must use the first non-empty ownership shard."""
    group = object()
    param = nn.Parameter(torch.zeros(3))
    optimizer = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    optimizer.pg_collection = SimpleNamespace(dp_cp=group, expt_dp=object())
    optimizer.dp_cp_params_list = [[], [param]]
    optimizer.expt_dp_params_list = None
    monkeypatch.setattr(layer_wise_optimizer_module, "get_pg_rank", lambda _: 0)

    def fake_all_gather(output_list, src, group):
        assert src.numel() == 0
        output_list[1].fill_(5.0)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    optimizer.allgather_params()
    torch.testing.assert_close(param, torch.full_like(param, 5.0))


def test_broadcast_params_runs_expert_plane_without_dense_plane(monkeypatch):
    """Dense and expert broadcasts are independent ownership domains."""
    expert_group = object()
    expert_params = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0]))]
    optimizer = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    optimizer.pg_collection = SimpleNamespace(dp_cp=object(), expt_dp=expert_group)
    optimizer.dp_cp_params_list = None
    optimizer.expt_dp_params_list = [[expert_params[0]], [expert_params[1]]]
    calls = []
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _, rank: rank + 10)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda param, src, group: calls.append((id(param), src, group)),
    )

    optimizer.broadcast_params()

    assert calls == [
        (id(expert_params[0]), 10, expert_group),
        (id(expert_params[1]), 11, expert_group),
    ]


def test_expert_parameters_use_expert_tp_duplicate_filter():
    """Expert parameters use ETP, while ordinary parameters use dense TP."""
    param = nn.Parameter(torch.ones(1))
    param.tensor_model_parallel = False
    dense_tp_group = _RankGroup(rank=0)
    expert_tp_group = _RankGroup(rank=1)

    param.allreduce = True
    assert param_is_not_tensor_parallel_duplicate(param, dense_tp_group, expert_tp_group)

    param.allreduce = False
    assert not param_is_not_tensor_parallel_duplicate(param, dense_tp_group, expert_tp_group)

    param.tensor_model_parallel = True
    assert param_is_not_tensor_parallel_duplicate(param, dense_tp_group, expert_tp_group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_main_parameter_preserves_expert_reduction_metadata():
    """The FP32 optimizer master must retain expert-topology routing metadata."""
    param = nn.Parameter(torch.ones(4, dtype=torch.bfloat16, device="cuda"))
    param.allreduce = False
    base_optimizer = torch.optim.SGD([param], lr=0.1)
    config = OptimizerConfig(optimizer="sgd", lr=0.1, bf16=True)

    optimizer = Float16OptimizerWithFloat16Params(base_optimizer, config, None, None)

    main_param = optimizer.fp32_from_float16_groups[0][0]
    assert main_param.allreduce is False


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or int(os.getenv("WORLD_SIZE", "1")) < 2
    or int(os.getenv("WORLD_SIZE", "1")) % 2,
    reason="Requires an even multi-GPU world",
)
class TestLayerWiseExpertTopology:
    """Distributed regressions for independent dense-DP and expert-DP ownership."""

    @pytest.fixture(autouse=True, params=["etp_split", "ep_split"])
    def setup_and_teardown(self, request):
        world_size = int(os.environ["WORLD_SIZE"])
        if request.param == "etp_split":
            tensor_parallel_size = world_size
            self.expert_model_parallel_size = 1
            self.expert_tensor_parallel_size = world_size // 2
        else:
            if world_size < 4:
                pytest.skip("EP topology requires at least four GPUs")
            tensor_parallel_size = world_size // 2
            self.expert_model_parallel_size = 2
            self.expert_tensor_parallel_size = 1
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            expert_tensor_parallel_size=self.expert_tensor_parallel_size,
        )
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _process_groups():
        return ProcessGroupCollection.use_mpu_process_groups(["tp", "expt_tp", "dp_cp", "expt_dp"])

    def _make_optimizer(self, params, clip_grad):
        config = OptimizerConfig(
            optimizer="sgd",
            lr=0.1,
            min_lr=0.0,
            weight_decay=0.0,
            sgd_momentum=0.0,
            clip_grad=clip_grad,
            log_num_zeros_in_grad=True,
            bf16=False,
            use_distributed_optimizer=False,
            params_dtype=torch.float32,
        )
        base_optimizer = torch.optim.SGD(
            [{"params": params, "is_expert_parallel": True}], lr=config.lr
        )
        optimizer = LayerWiseDistributedOptimizer(
            [FP32Optimizer(base_optimizer, config, None)], config, self._process_groups()
        )
        return optimizer

    @staticmethod
    def _assert_group_replicas_equal(param, group):
        replicas = [
            torch.empty_like(param) for _ in range(torch.distributed.get_world_size(group=group))
        ]
        torch.distributed.all_gather(replicas, param, group=group)
        for replica in replicas[1:]:
            torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)

    def test_expert_shards_have_one_edp_owner_and_exact_norm(self):
        """Each ETP shard contributes once to norm and clipping, not once per EDP replica."""
        expert_scale = parallel_state.get_expert_model_parallel_rank() + 1
        param = nn.Parameter(torch.full((8,), expert_scale, dtype=torch.float32, device="cuda"))
        param.tensor_model_parallel = True
        param.allreduce = False
        param.main_grad = torch.full_like(param, 3.0 * expert_scale)
        optimizer = self._make_optimizer([param], clip_grad=1.0)

        assert optimizer.dp_cp_params_list is None
        assert optimizer.expt_dp_params_list is not None
        local_owner_count = len(optimizer.chained_optimizers[0].get_parameters())
        global_owner_count = torch.tensor(local_owner_count, dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(global_owner_count)
        expected_shards = self.expert_model_parallel_size * self.expert_tensor_parallel_size
        assert global_owner_count.item() == expected_shards

        update_successful, grad_norm, num_zeros = optimizer.step()

        expert_scale_sq_sum = sum(
            scale**2 for scale in range(1, self.expert_model_parallel_size + 1)
        )
        expected_norm = (
            self.expert_tensor_parallel_size * param.numel() * 3.0**2 * expert_scale_sq_sum
        ) ** 0.5
        assert update_successful
        assert grad_norm == pytest.approx(expected_norm, rel=1e-6, abs=1e-6)
        assert num_zeros == 0
        clip_coefficient = 1.0 / (expected_norm + 1.0e-6)
        expected_param = torch.full_like(
            param, expert_scale - 0.1 * 3.0 * expert_scale * clip_coefficient
        )
        torch.testing.assert_close(param, expected_param, rtol=1e-6, atol=1e-6)
        self._assert_group_replicas_equal(param, optimizer.pg_collection.expt_dp)

    def test_nonpartitioned_expert_params_use_expert_tp_for_stats(self):
        """ETP replicas are deduplicated without dropping EDP-owned expert parameters."""
        expert_scale = parallel_state.get_expert_model_parallel_rank() + 1
        first = nn.Parameter(torch.full((8,), expert_scale, dtype=torch.float32, device="cuda"))
        second = nn.Parameter(torch.full((8,), expert_scale, dtype=torch.float32, device="cuda"))
        for param in (first, second):
            param.tensor_model_parallel = False
            param.allreduce = False
        first_grad = torch.tensor([0, 0, 0, 0, 3, 3, 3, 3], device="cuda", dtype=torch.float32)
        second_grad = torch.tensor([0, 0, 4, 4, 4, 4, 4, 4], device="cuda", dtype=torch.float32)
        first_grad *= expert_scale
        second_grad *= expert_scale
        first.main_grad = first_grad.clone()
        second.main_grad = second_grad.clone()
        per_expert_norm_sq = 4 * 3.0**2 + 6 * 4.0**2
        expert_scale_sq_sum = sum(
            scale**2 for scale in range(1, self.expert_model_parallel_size + 1)
        )
        expected_norm = (expert_scale_sq_sum * per_expert_norm_sq) ** 0.5
        optimizer = self._make_optimizer([first, second], clip_grad=expected_norm / 2.0)

        local_owner_count = len(optimizer.chained_optimizers[0].get_parameters())
        global_owner_count = torch.tensor(local_owner_count, dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(global_owner_count)
        expected_owned_params = (
            2 * self.expert_model_parallel_size * self.expert_tensor_parallel_size
        )
        assert global_owner_count.item() == expected_owned_params

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful
        assert grad_norm == pytest.approx(expected_norm, rel=1e-6, abs=1e-6)
        assert num_zeros == 6 * self.expert_model_parallel_size
        clip_coefficient = (expected_norm / 2.0) / (expected_norm + 1.0e-6)
        torch.testing.assert_close(
            first, torch.full_like(first, expert_scale) - 0.1 * first_grad * clip_coefficient
        )
        torch.testing.assert_close(
            second, torch.full_like(second, expert_scale) - 0.1 * second_grad * clip_coefficient
        )
        self._assert_group_replicas_equal(first, optimizer.pg_collection.expt_dp)
        self._assert_group_replicas_equal(second, optimizer.pg_collection.expt_dp)


class SimpleModel(nn.Module):
    """Simple model for testing LayerWiseDistributedOptimizer.

    Model with 5 layers to ensure more than 8 parameters (10 total: 5 weights + 5 biases).
    """

    def __init__(self, input_size=80, hidden_size=48, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class TinyModel(nn.Module):
    """Tiny model with only 1 layer (2 parameters: weight and bias)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
@skip_layerwise_lts
class TestLayerWiseOptimizer:
    """Test class for LayerWiseDistributedOptimizer with common setup code."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        rank = int(os.getenv('RANK', '0'))
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_model_and_optimizer(
        self,
        model_class=SimpleModel,
        clip_grad=1.0,
        model_kwargs=None,
        use_layer_wise=True,
        copy_from=None,
    ):
        """Create model, DDP wrapper, and optimizer.

        Args:
            model_class: Model class to instantiate
            clip_grad: Optional gradient clipping value
            model_kwargs: Optional kwargs for model initialization
            use_layer_wise: If True, wrap optimizer in LayerWiseDistributedOptimizer;
                          if False, use get_megatron_optimizer instead (for reference)

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        if model_kwargs is None:
            model_kwargs = {}

        model = model_class(**model_kwargs).bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        if copy_from:
            model.module.load_state_dict(copy_from.module.state_dict())
        else:
            model.broadcast_params()

        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=0.01,
            weight_decay=0.01,
            bf16=not use_layer_wise,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = get_megatron_optimizer(optimizer_config, [model])
        if use_layer_wise:
            optimizer_config.bf16 = True
            optimizer = LayerWiseDistributedOptimizer(
                optimizer.chained_optimizers, optimizer_config, pg_collection
            )
        return model, optimizer, pg_collection

    def create_reference_model(self, model):
        """Create a reference model by cloning the current model."""
        reference_model = type(model.module)().bfloat16().cuda()
        reference_model.load_state_dict(model.module.state_dict())
        return reference_model

    def test_basic(self):
        """Test basic LayerWiseDistributedOptimizer initialization and step with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"

        reference_model = self.create_reference_model(model)

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

        # Verify parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # Verify all ranks have the same updated parameters (test allgather)
        dp_size = get_pg_size(pg_collection.dp_cp)

        if dp_size > 1:
            for name, param in model.named_parameters():
                # Gather parameters from all ranks
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(param_list, param.data, group=pg_collection.dp_cp)

                # Verify all ranks have the same parameter values
                for i in range(1, dp_size):
                    try:
                        torch.testing.assert_close(param_list[0], param_list[i])
                    except AssertionError as e:
                        # Append additional context without overwriting the default message
                        raise AssertionError(
                            f"Parameter {name} differs between rank 0 and rank {i}. {str(e)}"
                        ) from None

    def test_get_grad_norm(self):
        """Test LayerWiseDistributedOptimizer gradient norm computation."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            use_layer_wise=False
        )

        # Set same gradients on both models
        # note that model is different at this point but we're only testing grad norm here
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.float().detach()
            ref_param.main_grad = grad_value.float().detach()

        # Test get_grad_norm on both optimizers
        optimizer.prepare_grads()
        grad_norm = optimizer.get_grad_norm()

        reference_optimizer.prepare_grads()
        reference_grad_norm = reference_optimizer.get_grad_norm()

        assert grad_norm is not None, "Grad norm should not be None"
        assert grad_norm >= 0, "Grad norm should be non-negative"

        # Compare with reference optimizer grad norm
        torch.testing.assert_close(grad_norm, reference_grad_norm, rtol=1e-5, atol=1e-5)

    def test_state_dict(self):
        """Test LayerWiseDistributedOptimizer state dict save and load."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Test state_dict
        state_dict = optimizer.state_dict()

        # Test load_state_dict
        # TODO(deyuf): fix this. not going through get() will cause missing keys like wd_mult
        # optimizer.load_state_dict(state_dict)

    def test_sharded_state_dict(self):
        """Test LayerWiseDistributedOptimizer sharded_state_dict method."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Get model sharded state dict
        model_sharded_state_dict = model.sharded_state_dict()

        # Test sharded_state_dict
        sharded_state_dict = optimizer.sharded_state_dict(model_sharded_state_dict)

        # Verify the sharded_state_dict is not None and has expected structure
        assert sharded_state_dict is not None, "Sharded state dict should not be None"
        assert (
            'optimizer' in sharded_state_dict
        ), "Sharded state dict should contain 'optimizer' key"

        # Verify that replica_id is set correctly (should be 0 for DP dimension)
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.dict_utils import nested_values

        for sh_base in nested_values(sharded_state_dict):
            if isinstance(sh_base, ShardedTensor):
                assert (
                    len(sh_base.replica_id) == 3
                ), f'Expected replica_id format (PP, TP, DP), got: {sh_base.replica_id}'
                assert (
                    sh_base.replica_id[2] == 0
                ), f'Expected DP replica_id to be 0 for layer-wise optimizer, got: {sh_base.replica_id[2]}'

    def test_multiple_optimizers(self):
        """Test LayerWiseDistributedOptimizer with multiple chained optimizers.

        This test properly tests allgather functionality with multiple ranks.
        """
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Split parameters into two groups for testing multiple optimizers
        params = list(model.parameters())
        mid_point = len(params) // 2
        param_groups_1 = [{'params': params[:mid_point]}]
        param_groups_2 = [{'params': params[mid_point:]}]

        # Create two separate base optimizers
        base_optimizer_1 = torch.optim.Adam(param_groups_1, lr=optimizer_config.lr)
        base_optimizer_2 = torch.optim.Adam(param_groups_2, lr=optimizer_config.lr)

        wrapped_optimizer_1 = FP32Optimizer(base_optimizer_1, optimizer_config, None)
        wrapped_optimizer_2 = FP32Optimizer(base_optimizer_2, optimizer_config, None)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = LayerWiseDistributedOptimizer(
            [wrapped_optimizer_1, wrapped_optimizer_2], optimizer_config, pg_collection
        )

        assert len(optimizer.chained_optimizers) == 2, "Should have two chained optimizers"

        # Set gradients and test optimizer step - this will trigger allgather
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_wrapping(self):
        """Test LayerWiseDistributedOptimizer automatically wraps optimizer with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify bf16 wrapping happened
        assert isinstance(
            optimizer.chained_optimizers[0], Float16OptimizerWithFloat16Params
        ), "Optimizer should be wrapped in Float16OptimizerWithFloat16Params"

        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_error(self):
        """Test LayerWiseDistributedOptimizer raises error when receiving pre-wrapped Float16 optimizer."""
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Create base optimizer and manually wrap in Float16 optimizer
        param_groups = [{'params': list(model.parameters())}]
        base_optimizer = torch.optim.Adam(param_groups, lr=optimizer_config.lr)
        wrapped_optimizer = Float16OptimizerWithFloat16Params(
            base_optimizer, optimizer_config, None, None
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        # Should raise TypeError when receiving already-wrapped Float16 optimizer
        with pytest.raises(
            TypeError, match='LayerWiseDistributedOptimizer received Float16 optimizer already'
        ):
            LayerWiseDistributedOptimizer([wrapped_optimizer], optimizer_config, pg_collection)

    def _run_parameter_update_test(self, model_class=SimpleModel):
        """Helper method to test parameter updates with a given model class.

        Args:
            model_class: Model class to use for testing
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=model_class)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=model_class, use_layer_wise=False, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)

    def test_parameter_updates(self):
        """Test LayerWiseDistributedOptimizer actually updates model parameters."""
        self._run_parameter_update_test()

    def test_parameter_updates_insufficient_parameters(self):
        """Test LayerWiseDistributedOptimizer when there are insufficient parameters for all ranks.

        Uses a tiny model with only 1 layer (2 parameters: weight and bias).
        This will be insufficient when world size > 2.
        """
        self._run_parameter_update_test(model_class=TinyModel)

    def test_broadcast_vs_allgather(self):
        """Test LayerWiseDistributedOptimizer allgather code agains broadcast code."""
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=SimpleModel)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=SimpleModel, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # step() internal call allgather_params. replace reference object with bcast
        reference_optimizer.allgather_params = reference_optimizer.broadcast_params
        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)
