import functools

import pytest
import torch

from megatron.core.distributed import deterministic_collectives


class _FakeGroup:
    def __init__(self, world_size):
        self._world_size = world_size

    def size(self):
        return self._world_size

    def rank(self):
        return 0


def _patch_all_gather(monkeypatch, partials):
    def _fake_all_gather(output_list, input_, group=None):
        for dst, src in zip(output_list, partials):
            dst.copy_(src)

    monkeypatch.setattr(torch.distributed, "all_gather", _fake_all_gather)


def _manual_tree_sum(partials):
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


def _manual_ascending_sum(partials):
    running = partials[0].clone()
    for partial in partials[1:]:
        running = running + partial
    return running


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_deterministic_sum_inplace_matches_true_sum(monkeypatch, world_size):
    """deterministic_sum_inplace equals the true element-wise sum."""
    partials = [torch.randn(5, dtype=torch.float32) for _ in range(world_size)]
    _patch_all_gather(monkeypatch, partials)

    tensor = partials[0].clone()
    deterministic_collectives.deterministic_sum_inplace(tensor, _FakeGroup(world_size))

    torch.testing.assert_close(tensor, _manual_tree_sum(partials))


def test_deterministic_sum_inplace_is_bitwise_identical_across_groups(monkeypatch):
    """Two process-group objects over the same ranks give bitwise-identical results."""
    world_size = 8
    partials = [torch.randn(7, dtype=torch.float32) for _ in range(world_size)]

    _patch_all_gather(monkeypatch, partials)
    first = partials[0].clone()
    deterministic_collectives.deterministic_sum_inplace(first, _FakeGroup(world_size))

    _patch_all_gather(monkeypatch, partials)
    second = partials[0].clone()
    deterministic_collectives.deterministic_sum_inplace(second, _FakeGroup(world_size))

    assert torch.equal(first, second)


def test_chunking_matches_single_shot(monkeypatch):
    """Chunked folding equals the result of a single unchunked fold."""
    world_size = 4
    partials = [torch.randn(100, dtype=torch.float32) for _ in range(world_size)]
    _patch_all_gather(monkeypatch, partials)

    chunked = partials[0].clone()
    deterministic_collectives.deterministic_sum_inplace(
        chunked, _FakeGroup(world_size), chunk_numel=16
    )

    torch.testing.assert_close(chunked, _manual_tree_sum(partials))


def test_non_power_of_two_falls_back_to_ascending_fold():
    """Non-power-of-two world sizes use the ascending-rank sequential fold."""
    partials = [torch.randn(4, dtype=torch.float32) for _ in range(3)]
    expected = _manual_ascending_sum(partials)

    # fold_gathered_sum may reuse the first buffer as its accumulator.
    actual = deterministic_collectives.fold_gathered_sum([p.clone() for p in partials])

    torch.testing.assert_close(actual, expected)


def test_power_of_two_uses_tree_fold():
    """Power-of-two world sizes use the fixed pairwise tree fold."""
    partials = [torch.randn(4, dtype=torch.float32) for _ in range(4)]
    expected = _manual_tree_sum(partials)

    actual = deterministic_collectives.fold_gathered_sum([p.clone() for p in partials])

    torch.testing.assert_close(actual, expected)


def test_with_gather_custom_fn_matches_group_path():
    """The injectable-gather variant gives the same bits as the process-group path."""
    world_size = 4
    partials = [torch.randn(9, dtype=torch.float32) for _ in range(world_size)]

    offset = 0

    def _fake_gather(gathered_list, chunk):
        nonlocal offset
        for dst, src in zip(gathered_list, partials):
            dst.copy_(src[offset : offset + chunk.numel()])
        offset += chunk.numel()

    tensor = partials[0].clone()
    deterministic_collectives.deterministic_sum_inplace_with_gather(
        tensor, world_size=world_size, all_gather_fn=_fake_gather, chunk_numel=4
    )

    torch.testing.assert_close(tensor, _manual_tree_sum(partials))


def test_world_size_one_is_noop(monkeypatch):
    """A single-rank group leaves the tensor unchanged."""
    called = functools.partial(pytest.fail, "all_gather should not be called for world_size 1")
    monkeypatch.setattr(torch.distributed, "all_gather", lambda *a, **k: called())

    tensor = torch.randn(5, dtype=torch.float32)
    expected = tensor.clone()
    deterministic_collectives.deterministic_sum_inplace(tensor, _FakeGroup(1))

    assert torch.equal(tensor, expected)


def test_requires_contiguous_input():
    """A non-contiguous tensor is rejected by an assertion."""
    base = torch.randn(4, 4, dtype=torch.float32)
    non_contiguous = base.t()

    with pytest.raises(AssertionError, match="contiguous"):
        deterministic_collectives.deterministic_sum_inplace(non_contiguous, _FakeGroup(2))
