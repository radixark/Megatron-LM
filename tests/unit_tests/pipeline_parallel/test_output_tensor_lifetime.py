# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise output ownership in the real schedules without distributed workers.

Non-loss callbacks retain model outputs; loss callbacks retain the reduced loss.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import schedules
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def set_input_tensor(self, input_tensor):
        pass


@pytest.fixture
def cpu_schedule(monkeypatch):
    # Only the schedule's token accumulator requires CUDA for these small models.
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, 'zeros', cpu_zeros)
    monkeypatch.setattr(schedules.MoEAuxLossAutoScaler, 'main_loss_backward_scale', None)
    config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = SimpleNamespace(size=lambda: 1)
    pg_collection.cp = SimpleNamespace(size=lambda: 1)
    return config, pg_collection


def _assert_released(refs):
    assert all(ref() is None for ref in refs), 'previous microbatch output is still live'


def _check_final_release(config, refs):
    # Check before returning from the schedule, including one-microbatch runs.
    def timer(name, **kwargs):
        def stop():
            if name == 'forward-backward':
                _assert_released(refs)

        return SimpleNamespace(start=lambda **kwargs: None, stop=stop)

    config.timers = timer


def _forward_callback(refs, mode, *, dictionary_output=False):
    def forward(data_iterator, model):
        _assert_released(refs)
        value = next(data_iterator)
        output = model.weight * torch.full((2, 4), float(value))
        refs.append(weakref.ref(output))

        def reduce(output, non_loss_data=False):
            if dictionary_output:
                output = output['lm']
            assert non_loss_data == (mode == 'non_loss')
            result = {'value': output.sum().item()}
            if non_loss_data:
                return result
            loss = output.sum()
            refs.append(weakref.ref(loss))
            if mode == 'loss_tokens':
                return loss, torch.tensor(output.numel()), result
            return loss, result

        return {'lm': output} if dictionary_output else output, reduce

    return forward


@pytest.mark.parametrize('num_microbatches', [1, 3])
@pytest.mark.parametrize('mode', ['non_loss', 'loss', 'loss_tokens'])
@pytest.mark.parametrize('moe', [False, True])
def test_no_pipeline_forward_only_releases_outputs(cpu_schedule, num_microbatches, mode, moe):
    config, pg_collection = cpu_schedule
    if moe:
        config.num_moe_experts = 2
        # Forward-only must use the regular schedule even with overlap configured.
        config.overlap_moe_expert_parallel_comm = True
    refs = []
    _check_final_release(config, refs)
    with torch.no_grad():
        results = schedules.forward_backward_no_pipelining(
            forward_step_func=_forward_callback(refs, mode),
            data_iterator=iter(range(1, num_microbatches + 1)),
            model=_Model(config),
            num_microbatches=num_microbatches,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=mode == 'non_loss',
            pg_collection=pg_collection,
        )
    assert results == [{'value': 16.0 * i} for i in range(1, num_microbatches + 1)]
    _assert_released(refs)
    if moe:
        torch.testing.assert_close(
            schedules.MoEAuxLossAutoScaler.main_loss_backward_scale,
            torch.tensor([1.0 / num_microbatches]),
        )


@pytest.mark.parametrize('num_microbatches', [1, 3])
@pytest.mark.parametrize('mode', ['loss', 'loss_tokens'])
def test_no_pipeline_training_preserves_backward(cpu_schedule, num_microbatches, mode):
    config, pg_collection = cpu_schedule
    model = _Model(config)
    refs = []
    _check_final_release(config, refs)
    results = schedules.forward_backward_no_pipelining(
        forward_step_func=_forward_callback(refs, mode),
        data_iterator=iter(range(1, num_microbatches + 1)),
        model=model,
        num_microbatches=num_microbatches,
        seq_length=2,
        micro_batch_size=1,
        pg_collection=pg_collection,
    )
    expected_grad = (num_microbatches + 1) / 2
    if mode == 'loss':
        expected_grad *= 8
    torch.testing.assert_close(model.weight.grad, torch.tensor(expected_grad))
    assert results == [{'value': 16.0 * i} for i in range(1, num_microbatches + 1)]


class _BlockingPipeline:
    """Stand in for transport only; do not retain tensors as mock call histories do."""

    def __init__(self, config, stage, *, dictionary_output=False):
        self.config = config
        self.total_stages = 3
        self.current_stage = stage
        self.is_pp_first_stage = stage == 0
        self.is_pp_last_stage = stage == 2
        self.dictionary_output = dictionary_output
        self.sent = []

    def recv_forward(self, shapes, is_first_stage):
        # Model does not use pipeline inputs. Preserve each transport's container shape.
        return {} if self.dictionary_output else [None]

    def send_forward(self, output, is_last_stage):
        if not is_last_stage:
            tensor = output['lm'] if self.dictionary_output else output[0]
            # Read the full payload at send time: it must be valid until send returns.
            self.sent.append(tensor.sum().item())


@pytest.mark.parametrize(
    'stage,num_microbatches',
    [(0, 1), (0, 4), (1, 4), (2, 1), (2, 4)],
    ids=['warmup_only', 'warmup_to_steady', 'middle_stage', 'last_single', 'last_steady'],
)
@pytest.mark.parametrize('mode', ['non_loss', 'loss', 'loss_tokens'])
@pytest.mark.parametrize('multimodule', [False, True])
def test_pipeline_forward_only_releases_outputs(
    cpu_schedule, stage, num_microbatches, mode, multimodule
):
    config, pg_collection = cpu_schedule
    if multimodule:
        config.variable_seq_lengths = True
        pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={'lm': pg_collection}, language_model_module_name='lm'
        )
    communicator = _BlockingPipeline(config, stage, dictionary_output=multimodule)
    refs = []
    _check_final_release(config, refs)
    with torch.no_grad():
        results = schedules.forward_backward_pipelining_without_interleaving(
            forward_step_func=_forward_callback(refs, mode, dictionary_output=multimodule),
            data_iterator=iter(range(1, num_microbatches + 1)),
            model=_Model(config),
            num_microbatches=num_microbatches,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=mode == 'non_loss',
            p2p_communicator=communicator,
            pg_collection=pg_collection,
        )
    expected_values = [16.0 * i for i in range(1, num_microbatches + 1)]
    assert results == ([{'value': value} for value in expected_values] if stage == 2 else [])
    assert communicator.sent == ([] if stage == 2 else expected_values)
    _assert_released(refs)


@pytest.mark.parametrize('callback', ['non_loss', 'none'])
def test_no_pipeline_preserves_outputs_owned_by_caller(cpu_schedule, callback):
    config, pg_collection = cpu_schedule
    refs = []

    def forward(data_iterator, model):
        output = torch.full((2, 4), float(next(data_iterator)))
        refs.append(weakref.ref(output))
        return output, (lambda output, non_loss_data: output) if callback == 'non_loss' else None

    with torch.no_grad():
        results = schedules.forward_backward_no_pipelining(
            forward_step_func=forward,
            data_iterator=iter(range(3)),
            model=_Model(config),
            num_microbatches=3,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=True,
            pg_collection=pg_collection,
        )
    for i, output in enumerate(results):
        assert refs[i]() is output
        torch.testing.assert_close(output, torch.full((2, 4), float(i)))
    del output, results
    _assert_released(refs)


@pytest.mark.parametrize('mode', ['non_loss', 'loss', 'loss_tokens'])
@pytest.mark.parametrize('overlap', [False, True])
def test_interleaved_forward_only_releases_last_stage_outputs(cpu_schedule, mode, overlap):
    config, pg_collection = cpu_schedule
    config.virtual_pipeline_model_parallel_size = 2
    config.microbatch_group_size_per_vp_stage = 2
    config.overlap_p2p_comm = overlap
    config.overlap_p2p_comm_warmup_flush = overlap
    config.batch_p2p_comm = not overlap
    config.overlap_moe_expert_parallel_comm = True
    refs = []
    _check_final_release(config, refs)
    models = [_Model(config), _Model(config)]
    for i, model in enumerate(models):
        model.vp_stage = i
    last_stage_forward = _forward_callback(refs, mode)

    def forward(data_iterator, model):
        # Final-stage logits must also be gone before a different chunk runs.
        _assert_released(refs)
        if model.vp_stage == 1:
            return last_stage_forward(data_iterator, model)
        return torch.ones((2, 1, 4)), None

    def exchange(output_tensor, recv_prev, tensor_shape, overlap_p2p_comm=False):
        received = torch.zeros(tensor_shape) if recv_prev else None
        if not overlap_p2p_comm:
            return received
        handles = {}
        if recv_prev:
            handles['recv_prev'] = SimpleNamespace(wait=lambda: None)
        if output_tensor is not None:
            handles['send_next'] = SimpleNamespace(wait=lambda: None)
        return received, handles

    communicator = SimpleNamespace(
        config=config,
        virtual_pipeline_model_parallel_size=2,
        pp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
        recv_forward=lambda shape, is_first_stage: torch.zeros(shape),
        send_forward_recv_forward=exchange,
    )
    with torch.no_grad():
        results = schedules.forward_backward_pipelining_with_interleaving(
            forward_step_func=forward,
            data_iterator=[iter(range(1, 5)), iter(range(1, 5))],
            model=models,
            num_microbatches=4,
            seq_length=2,
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=mode == 'non_loss',
            p2p_communicator=communicator,
            pg_collection=pg_collection,
        )
    assert results == [{'value': 16.0 * i} for i in range(1, 5)]
    _assert_released(refs)


@pytest.mark.parametrize('stage,num_microbatches', [(0, 1), (0, 4), (1, 4), (2, 4)])
def test_pipeline_training_preserves_backward(cpu_schedule, stage, num_microbatches):
    config, pg_collection = cpu_schedule
    communicator = _BlockingPipeline(config, stage)
    model = _Model(config)
    backward_order = []

    def forward(data_iterator, model):
        value = next(data_iterator)
        output = model.weight * torch.full((2, 4), float(value))
        output.register_hook(lambda grad: backward_order.append(value))
        return output, lambda output: (output.sum(), {'value': output.sum().item()})

    def recv_backward(shapes, is_last_stage):
        return [None if is_last_stage else torch.full((2, 4), 1.0 / num_microbatches)]

    def send_forward_recv_backward(output, shapes, is_last_stage):
        communicator.send_forward(output, is_last_stage)
        return recv_backward(shapes, is_last_stage)

    communicator.recv_backward = recv_backward
    communicator.send_forward_recv_backward = send_forward_recv_backward
    communicator.send_backward = lambda grad, is_first_stage: None
    communicator.send_backward_recv_forward = lambda grad, shapes, is_first_stage: [None]
    results = schedules.forward_backward_pipelining_without_interleaving(
        forward_step_func=forward,
        data_iterator=iter(range(1, num_microbatches + 1)),
        model=model,
        num_microbatches=num_microbatches,
        seq_length=2,
        micro_batch_size=1,
        p2p_communicator=communicator,
        pg_collection=pg_collection,
    )
    assert backward_order == list(range(1, num_microbatches + 1))
    torch.testing.assert_close(model.weight.grad, torch.tensor(4.0 * (num_microbatches + 1)))
    assert len(results) == (num_microbatches if stage == 2 else 0)
