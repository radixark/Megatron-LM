import atexit
import json
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

logger = logging.getLogger(__name__)

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


class _BucketSpec:
    """One DDP bucket's slice of optimizer state and its backing file.

    The file holds three equally-sized segments, [main | exp_avg | exp_avg_sq],
    each laid out as the bucket's param shards concatenated in group order.
    """

    def __init__(self, index: int, path: str, entries: List[Tuple[torch.nn.Parameter, torch.Tensor, int]]):
        self.index = index
        self.path = path
        self.entries = entries  # (model_param, shard_main_param, master_group_idx)
        self.numel = sum(main.numel() for _, main, _ in entries)
        offsets = []
        pos = 0
        for _, main, _ in entries:
            offsets.append(pos)
            pos += main.numel()
        self.entry_offsets = offsets
        self.fd = -1
        self.adam = None
        self.group_master_indices: List[int] = []
        self.main_on_disk = False
        self.moments_on_disk = False


class NVMeOptimizerStateStore:
    """Owns residency and I/O of one DistributedOptimizer's state."""

    # ChainedOptimizer's dense/expert members can share distributed_optimizer_instance_id;
    # this per-process counter keeps their store directories distinct.
    _next_uid = 0

    def __init__(self, distrib_optimizer: "DistributedOptimizer", dir_root: str, chunk_mb: int):
        self.dist_opt = distrib_optimizer
        self.uid = NVMeOptimizerStateStore._next_uid
        NVMeOptimizerStateStore._next_uid += 1
        config = distrib_optimizer.config

        assert not config.use_precision_aware_optimizer, (
            "NVMe state store requires the non-precision-aware optimizer "
            "(fp32 main params held by mcore)."
        )
        assert not config.optimizer_cpu_offload, "NVMe state store is mutually exclusive with CPU offload."
        assert not config.offload_optimizer_states, (
            "NVMe state store is mutually exclusive with --offload-optimizer-states."
        )
        assert not distrib_optimizer.ddp_config.use_megatron_fsdp
        assert all(len(g) == 0 for g in distrib_optimizer.model_fp32_groups), (
            "NVMe state store only supports pure bf16/fp16 models (no fp32 model params)."
        )

        rank = torch.distributed.get_rank()
        instance = distrib_optimizer.distributed_optimizer_instance_id
        self.dir = os.path.join(dir_root, f"rank{rank}", f"opt{instance}_{self.uid}")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir, exist_ok=True)
        atexit.register(shutil.rmtree, self.dir, ignore_errors=True)

        self._chunk = torch.empty(chunk_mb * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
        self._chunk_np = self._chunk.numpy()

        self.specs = self._build_specs()
        self._build_bucket_optimizers()
        for spec in self.specs:
            spec.fd = os.open(spec.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.posix_fallocate(spec.fd, 0, 3 * spec.numel * 4)

        for spec in self.specs:
            for tensor, offset in self._segment(spec, "main"):
                self._stream(spec.fd, offset, tensor, to_disk=True)
                self._release(tensor)
            spec.main_on_disk = True

        total_gb = sum(3 * s.numel * 4 for s in self.specs) / 1024**3
        logger.info(
            f"NVMe optimizer state store: {len(self.specs)} buckets, "
            f"{total_gb:.1f} GB state at {self.dir}"
        )

    def _build_specs(self) -> List["_BucketSpec"]:
        by_bucket: Dict[Tuple, List] = {}
        groups = zip(
            self.dist_opt.model_float16_groups, self.dist_opt.shard_fp32_from_float16_groups
        )
        for group_idx, (model_group, main_group) in enumerate(groups):
            for model_param, main_param in zip(model_group, main_group):
                assert main_param is not None and main_param.dtype == torch.float32
                key = self.dist_opt.model_param_gbuf_map[model_param]
                by_bucket.setdefault(key, []).append((model_param, main_param, group_idx))
        limit = 200_000_000
        chunked = []
        for _, entries in sorted(by_bucket.items(), key=lambda kv: kv[0]):
            cur, cur_numel = [], 0
            for entry in entries:
                cur.append(entry)
                cur_numel += entry[1].numel()
                if cur_numel >= limit:
                    chunked.append(cur)
                    cur, cur_numel = [], 0
            if cur:
                chunked.append(cur)
        return [
            _BucketSpec(i, os.path.join(self.dir, f"bucket{i:05d}.bin"), entries)
            for i, entries in enumerate(chunked)
        ]

    def _build_bucket_optimizers(self) -> None:
        from megatron.core.optimizer import Adam

        master_groups = self.dist_opt.optimizer.param_groups
        for spec in self.specs:
            groups = []
            spec.group_master_indices = sorted({gi for _, _, gi in spec.entries})
            for g_idx in spec.group_master_indices:
                group = {k: v for k, v in master_groups[g_idx].items() if k != "params"}
                group["params"] = [main for _, main, gi in spec.entries if gi == g_idx]
                groups.append(group)
            spec.adam = Adam(groups, adam_w_mode=self.dist_opt.config.decoupled_weight_decay)

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def save_to(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        manifest = {
            "buckets": [
                {
                    "numel": spec.numel,
                    "entry_numels": [main.numel() for _, main, _ in spec.entries],
                    "steps": [g.get("step", 0) for g in spec.adam.param_groups],
                    "file": os.path.basename(spec.path),
                }
                for spec in self.specs
            ]
        }
        for spec in self.specs:
            shutil.copyfile(spec.path, os.path.join(dirpath, os.path.basename(spec.path)))
        with open(os.path.join(dirpath, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        logger.info(f"NVMe optimizer state saved: {len(self.specs)} buckets -> {dirpath}")

    @torch.no_grad()
    def load_from(self, dirpath: str) -> None:
        with open(os.path.join(dirpath, "manifest.json")) as f:
            manifest = json.load(f)
        assert len(manifest["buckets"]) == len(self.specs), (
            f"NVMe state layout mismatch: checkpoint has {len(manifest['buckets'])} buckets, "
            f"current topology builds {len(self.specs)} (same-topology resume only)"
        )
        for spec, meta in zip(self.specs, manifest["buckets"]):
            assert meta["numel"] == spec.numel
            assert meta["entry_numels"] == [main.numel() for _, main, _ in spec.entries]
            shutil.copyfile(os.path.join(dirpath, meta["file"]), spec.path)
            for group, step in zip(spec.adam.param_groups, meta["steps"]):
                if step:
                    group["step"] = step
            for _, main, _ in spec.entries:
                state = spec.adam.state.setdefault(main, {})
                for key in _MOMENT_KEYS:
                    if key not in state:
                        t = torch.empty_like(main)
                        t.untyped_storage().resize_(0)
                        state[key] = t
            spec.main_on_disk = True
            spec.moments_on_disk = True
        logger.info(f"NVMe optimizer state loaded: {len(self.specs)} buckets <- {dirpath}")

    @torch.no_grad()
    def refresh_main_from_model_params(self, copy_fn) -> None:
        for spec in self.specs:
            for tensor, _ in self._segment(spec, "main"):
                self._materialize(tensor)
        copy_fn()
        for spec in self.specs:
            for tensor, offset in self._segment(spec, "main"):
                self._stream(spec.fd, offset, tensor, to_disk=True)
                self._release(tensor)
            spec.main_on_disk = True

    @torch.no_grad()
    def step(self) -> bool:
        t0 = time.monotonic()
        read_bytes = written_bytes = 0
        for spec in self.specs:
            read_bytes += self._load_bucket(spec)
            self._sync_hyperparams(spec)
            spec.adam.step()
            self.dist_opt._copy_main_params_to_model_params_for(
                (main, model) for model, main, _ in spec.entries
            )
            written_bytes += self._store_bucket(spec)
        logger.info(
            f"NVMe streaming step: {len(self.specs)} buckets, "
            f"read {read_bytes / 1024**3:.1f} GB, wrote {written_bytes / 1024**3:.1f} GB "
            f"in {time.monotonic() - t0:.1f}s"
        )
        return True

    def _sync_hyperparams(self, spec: "_BucketSpec") -> None:
        master_groups = self.dist_opt.optimizer.param_groups
        for group, g_idx in zip(spec.adam.param_groups, spec.group_master_indices):
            group["lr"] = master_groups[g_idx]["lr"]
            group["weight_decay"] = master_groups[g_idx]["weight_decay"]

    def _load_bucket(self, spec: "_BucketSpec") -> int:
        nbytes = 0
        if spec.main_on_disk:
            for tensor, offset in self._segment(spec, "main"):
                self._materialize(tensor)
                self._stream(spec.fd, offset, tensor, to_disk=False)
                nbytes += tensor.numel() * 4
        if spec.moments_on_disk:
            for key in _MOMENT_KEYS:
                for tensor, offset in self._segment(spec, key):
                    self._materialize(tensor)
                    self._stream(spec.fd, offset, tensor, to_disk=False)
                    nbytes += tensor.numel() * 4
        return nbytes

    def _store_bucket(self, spec: "_BucketSpec") -> int:
        nbytes = 0
        for key in ("main",) + _MOMENT_KEYS:
            for tensor, offset in self._segment(spec, key):
                self._stream(spec.fd, offset, tensor, to_disk=True)
                self._release(tensor)
                nbytes += tensor.numel() * 4
        spec.main_on_disk = True
        spec.moments_on_disk = True
        return nbytes

    # ------------------------------------------------------- residency & I/O

    def _segment(self, spec: "_BucketSpec", key: str):
        segment_index = ("main",) + _MOMENT_KEYS
        base = segment_index.index(key) * spec.numel * 4
        for (_, main, _), entry_offset in zip(spec.entries, spec.entry_offsets):
            tensor = main if key == "main" else spec.adam.state[main][key]
            yield tensor, base + entry_offset * 4

    @staticmethod
    def _materialize(tensor: torch.Tensor) -> None:
        tensor.untyped_storage().resize_(tensor.numel() * tensor.element_size())

    @staticmethod
    def _release(tensor: torch.Tensor) -> None:
        tensor.untyped_storage().resize_(0)

    def _stream(self, fd: int, base_offset: int, tensor: torch.Tensor, *, to_disk: bool) -> None:
        flat = tensor.view(-1)
        chunk_numel = self._chunk.numel()
        pos = 0
        while pos < flat.numel():
            n = min(chunk_numel, flat.numel() - pos)
            byte_offset = base_offset + pos * 4
            if to_disk:
                self._chunk[:n].copy_(flat[pos : pos + n])
                self._rw_full(os.pwritev, fd, byte_offset, self._chunk_np[:n])
            else:
                self._rw_full(os.preadv, fd, byte_offset, self._chunk_np[:n])
                flat[pos : pos + n].copy_(self._chunk[:n])
            pos += n

    @staticmethod
    def _rw_full(op, fd: int, offset: int, array) -> None:
        mv = memoryview(array).cast("B")
        done = 0
        while done < len(mv):
            n = op(fd, [mv[done:]], offset + done)
            if n <= 0:
                raise IOError(f"short {op.__name__} ({n}) on optimizer state file at offset {offset + done}")
            done += n
