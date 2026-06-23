import pytest
import torch

from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:
    def __init__(self, world_size):
        self._world_size = world_size

    def size(self):
        return self._world_size

    def rank(self):
        return 0


def _make_tree_gather(monkeypatch, partials):
    def _fake_all_gather(output, input_, group=None):
        gathered = torch.cat(partials, dim=0)
        output.copy_(gathered)

    monkeypatch.setattr(mappings, "dist_all_gather_func", _fake_all_gather)


def _manual_tree_sum(partials):
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


@pytest.mark.parametrize("world_size", [2, 8])
def test_tree_all_reduce_sum_matches_fixed_pairwise_order(monkeypatch, world_size):
    partials = [
        torch.full((2, 3), float(index + 1), dtype=torch.float32) for index in range(world_size)
    ]
    _make_tree_gather(monkeypatch, partials)

    actual = mappings._tree_all_reduce_sum(partials[0].clone(), _FakeGroup(world_size))
    expected = _manual_tree_sum(partials)

    torch.testing.assert_close(actual, expected)


def test_tree_all_reduce_sum_rejects_non_power_of_two(monkeypatch):
    partials = [torch.ones((2, 3), dtype=torch.float32) for _ in range(3)]
    _make_tree_gather(monkeypatch, partials)

    with pytest.raises(RuntimeError, match="power-of-two"):
        mappings._tree_all_reduce_sum(partials[0].clone(), _FakeGroup(3))


def test_row_parallel_default_path_keeps_standard_reduce(monkeypatch):
    Utils.fake_initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            use_cpu_initialization=True,
            perform_initialization=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        layer = RowParallelLinear(
            input_size=8,
            output_size=8,
            config=config,
            init_method=lambda _: None,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )

        calls = []

        def _fake_reduce(input_, group=None, deterministic=False):
            calls.append(deterministic)
            return input_

        monkeypatch.setattr(
            "megatron.core.tensor_parallel.layers.reduce_from_tensor_model_parallel_region",
            _fake_reduce,
        )

        output, _ = layer(torch.randn(2, 1, 8))
        assert output.shape == (2, 1, 8)
        assert calls == [False]
    finally:
        Utils.destroy_model_parallel()


def test_row_parallel_sglang_path_uses_deterministic_reduce(monkeypatch):
    Utils.fake_initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            use_cpu_initialization=True,
            perform_initialization=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            transformer_impl="local",
            true_on_policy_contract="qwen3_dense_true_on_policy_v1",
        )
        layer = RowParallelLinear(
            input_size=8,
            output_size=8,
            config=config,
            init_method=lambda _: None,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )

        calls = []

        def _fake_reduce(input_, group=None, deterministic=False):
            calls.append(deterministic)
            return input_

        monkeypatch.setattr(
            "megatron.core.tensor_parallel.layers.reduce_from_tensor_model_parallel_region",
            _fake_reduce,
        )

        output, _ = layer(torch.randn(2, 1, 8))
        assert output.shape == (2, 1, 8)
        assert calls == [True]
    finally:
        Utils.destroy_model_parallel()
