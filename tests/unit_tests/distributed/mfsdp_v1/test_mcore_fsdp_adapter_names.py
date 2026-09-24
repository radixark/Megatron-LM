# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.distributed.fsdp import mcore_fsdp_adapter
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel,
    FullyShardedDataParallelV1,
    FullyShardedDataParallelV2,
)


def test_v1_is_the_shipped_implementation():
    assert FullyShardedDataParallelV1 is FullyShardedDataParallel


def test_v2_name_is_importable_but_not_constructible():
    assert issubclass(FullyShardedDataParallelV2, _BaseDataParallel)
    assert not issubclass(FullyShardedDataParallel, FullyShardedDataParallelV2)
    with pytest.raises(NotImplementedError, match="5865"):
        FullyShardedDataParallelV2(None, None, None)


def test_upstream_style_isinstance_tuple_imports():
    # The tuple Megatron-Bridge's unwrap_model builds after NVIDIA-NeMo/Megatron-Bridge#5933.
    module_instances = (
        mcore_fsdp_adapter.FullyShardedDataParallelV1,
        mcore_fsdp_adapter.FullyShardedDataParallelV2,
    )
    assert not isinstance(object(), module_instances)
