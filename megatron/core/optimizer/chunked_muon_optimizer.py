# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""BF16 Muon updates with CPU masters and a single GPU parameter window."""

import torch

from megatron.core.optimizer.cpu_offloading.chunked_optimizer_state_offload import (
    ChunkedOptimizerStateOffloader,
    OptimizerStateChunk,
)
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params


class _MuonMasterOffloader(ChunkedOptimizerStateOffloader):
    def __init__(self, optimizer, master_to_model, chunk_size_bytes):
        super().__init__(optimizer, list(master_to_model), chunk_size_bytes, 1.0, (torch.float32,))
        self.master_to_model = master_to_model
        self._param_devices.update({master: model.device for master, model in master_to_model.items()})
        self._master_weights_resident = False

    def prefetch_for_step(self):
        # Gradient finalization must not restore the complete master window.
        return

    def prefetch_master_for_step(self):
        return

    def ensure_master_for_param_sync(self):
        raise RuntimeError("Chunked Muon copies model parameters inside each update chunk")

    def _restore_chunk(self, params, grads):
        for param in params:
            if param in self.master_to_model:
                self._cpu_master[param] = param.data
                param.data = param.data.to(self._param_devices[param])
            param.grad = grads[param]

    def _offload_chunk(self, params):
        for param in params:
            if param in self.master_to_model:
                self.master_to_model[param].copy_(param)
                cpu_master = self._cpu_master[param]
                cpu_master.copy_(param)
                param.grad = None
                param.data = cpu_master
            else:
                param.grad = None
        self._schedule_state_d2h(params)
        self._d2h_stream.synchronize()
        self._release_state_staging_slots()

    @torch.no_grad()
    def step_with_gradients(self, grads):
        original_groups = list(self.optimizer.param_groups)
        base_metadata = self._snapshot_group_metadata(original_groups)
        resulting_metadata = {}
        try:
            for chunk in self._chunks:
                active = tuple(param for param in chunk.params if grads.get(param) is not None)
                if not active:
                    continue
                self._restore_chunk(active, grads)
                self._wait_state_prefetch(self._prefetch_state(OptimizerStateChunk(active)))
                self._step_subset(active, original_groups, base_metadata, resulting_metadata)
                self._offload_chunk(active)
        finally:
            self.optimizer.param_groups = original_groups
        for index, metadata in resulting_metadata.items():
            group = original_groups[index]
            for key in tuple(group):
                if key != "params" and key not in metadata:
                    del group[key]
            group.update(metadata)


class ChunkedMuonOptimizer(Float16OptimizerWithFloat16Params):
    """Keep masters on CPU across construction, model reload and completed updates.

    Gradients remain in the model's FP32 CUDA buffers. Optimizer checkpoint I/O,
    clipping and FP8 parameter staging are outside this bounded BF16 path.
    """

    def __init__(self, optimizer, config, grad_scaler, init_state_fn):
        if optimizer is not None and type(optimizer) is not TensorParallelMuon:
            raise ValueError("Chunked Muon masters require exactly TensorParallelMuon")
        if not (
            config.bf16
            and not config.fp16
            and config.use_layer_wise_distributed_optimizer
            and config.chunked_optimizer_state_offload
            and config.optimizer_state_offload_fraction == 1.0
            and config.optimizer_state_offload_chunk_size_mb > 0
        ):
            raise ValueError("Chunked Muon masters require BF16 LayerWise and full chunked state offload")
        if (
            grad_scaler is not None
            or config.clip_grad != 0
            or config.log_num_zeros_in_grad
            or config.optimizer_cpu_offload
            or config.optimizer_cuda_graph
            or config.use_precision_aware_optimizer
            or config.reuse_grad_buf_for_mxfp8_param_ag
            or config.overlap_param_gather
            or config.overlap_param_gather_with_optimizer_step
        ):
            raise ValueError("Unsupported scaling, clipping, FP8, graph or gather mode for chunked Muon masters")
        self._cuda_grads = {}
        super().__init__(optimizer, config, grad_scaler, init_state_fn, main_params_on_cpu=True)
        self.is_stub_optimizer = self.is_stub_optimizer or not self.get_parameters()
        self._master_to_model = {}
        if not self.is_stub_optimizer:
            self._master_to_model = {
                master: model
                for models, masters in zip(self.float16_groups, self.fp32_from_float16_groups)
                for model, master in zip(models, masters)
            }

    def enable_chunked_optimizer_state_offload(self, master_params, state_dtypes, **kwargs):
        if self.optimizer is not None:
            self._optimizer_state_offloader = _MuonMasterOffloader(
                self.optimizer,
                self._master_to_model,
                self.config.optimizer_state_offload_chunk_size_mb * 1024**2,
            )

    @torch.no_grad()
    def prepare_grads(self):
        self._cuda_grads.clear()
        if self.is_stub_optimizer:
            return False
        pairs = list(self._master_to_model.items())
        pairs.extend((param, param) for group in self.fp32_from_fp32_groups for param in group)
        for master, model in pairs:
            if not hasattr(model, "main_grad"):
                raise ValueError("Chunked Muon requires persistent model main_grad buffers")
            grad = model.main_grad
            if grad is not None:
                if grad.dtype != torch.float32 or grad.device != model.device or not grad.is_cuda:
                    raise ValueError("Chunked Muon requires existing FP32 CUDA main_grad buffers")
                self._cuda_grads[master] = grad
            model.grad = None
        return False

    def _get_param_grad_for_norm(self, param):
        return self._cuda_grads.get(param)

    @torch.no_grad()
    def step_with_ready_grads(self):
        if not self.is_stub_optimizer:
            self._optimizer_state_offloader.step_with_gradients(self._cuda_grads)
        self._cuda_grads.clear()
        return True

    def zero_grad(self, set_to_none=True):
        self._cuda_grads.clear()
        super().zero_grad(set_to_none)

    @torch.no_grad()
    def _copy_model_params_to_main_params(self, state_dict=None):
        if state_dict is not None:
            raise ValueError("Chunked Muon accepts model reload only, not optimizer state")
        self._cuda_grads.clear()
        for master, model in self._master_to_model.items():
            if master.device.type != "cpu":
                raise RuntimeError("Model reload requires CPU canonical Muon masters")
            master.copy_(model)

    def _copy_main_params_to_model_params(self):
        raise RuntimeError("Chunked Muon copies model parameters inside each update chunk")

    def state_dict(self, *args, **kwargs):
        raise RuntimeError("Optimizer checkpoint I/O is unsupported with chunked Muon masters")

    def sharded_state_dict(self, *args, **kwargs):
        raise RuntimeError("Optimizer checkpoint I/O is unsupported with chunked Muon masters")

    def load_state_dict(self, *args, **kwargs):
        raise RuntimeError("Optimizer checkpoint I/O is unsupported with chunked Muon masters")
