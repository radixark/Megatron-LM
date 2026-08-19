# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    copy_tensor_model_parallel_attributes,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestTPAttributesWithoutInitialization:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("use_cpu_init", [True, False])
    def test_vocab_parallel_embedding_tp_attrs_no_init(self, use_cpu_init):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=use_cpu_init,
            perform_initialization=False,
        )

        emb = VocabParallelEmbedding(
            num_embeddings=16, embedding_dim=8, init_method=cfg.init_method, config=cfg
        )
        w = emb.weight
        assert hasattr(w, "tensor_model_parallel") and w.tensor_model_parallel is True
        assert hasattr(w, "partition_dim") and w.partition_dim == 0
        assert hasattr(w, "partition_stride") and w.partition_stride == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("use_cpu_init", [True, False])
    def test_column_parallel_linear_tp_attrs_no_init(self, use_cpu_init):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=use_cpu_init,
            perform_initialization=False,
        )

        layer = ColumnParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            config=cfg,
            skip_bias_add=False,
        )
        w = layer.weight
        assert hasattr(w, "tensor_model_parallel") and w.tensor_model_parallel is True
        assert hasattr(w, "partition_dim") and w.partition_dim == 0
        assert hasattr(w, "partition_stride") and w.partition_stride == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("use_cpu_init", [True, False])
    def test_row_parallel_linear_tp_attrs_no_init(self, use_cpu_init):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            use_cpu_initialization=use_cpu_init,
            perform_initialization=False,
        )

        layer = RowParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            input_is_parallel=True,
            config=cfg,
            skip_bias_add=False,
        )
        w = layer.weight
        assert hasattr(w, "tensor_model_parallel") and w.tensor_model_parallel is True
        assert hasattr(w, "partition_dim") and w.partition_dim == 1
        assert hasattr(w, "partition_stride") and w.partition_stride == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_expert_linear_parameters_use_expert_topology_metadata(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            use_cpu_initialization=True,
            perform_initialization=False,
        )

        column = ColumnParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            config=cfg,
            gather_output=False,
            skip_bias_add=False,
            is_expert=True,
        )
        row = RowParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            input_is_parallel=True,
            config=cfg,
            skip_bias_add=False,
            is_expert=True,
        )

        for param in (*column.parameters(), *row.parameters()):
            assert param.allreduce is False

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_TE,
        reason="CUDA and Transformer Engine are required",
    )
    def test_te_expert_cpu_init_bias_uses_expert_topology_metadata(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
        cfg = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            use_cpu_initialization=True,
            perform_initialization=False,
        )

        column = TEColumnParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            config=cfg,
            gather_output=False,
            skip_bias_add=False,
            is_expert=True,
            stride=2,
        )
        row = TERowParallelLinear(
            input_size=8,
            output_size=8,
            init_method=cfg.init_method,
            bias=True,
            input_is_parallel=True,
            config=cfg,
            skip_bias_add=False,
            is_expert=True,
        )

        for param in (*column.parameters(), *row.parameters()):
            assert param.allreduce is False
        assert column.bias.tensor_model_parallel is True
        assert column.weight.partition_stride == 2
        assert column.bias.partition_stride == 2
        assert row.bias.tensor_model_parallel is False


def test_copy_tensor_model_parallel_attributes_preserves_qkv_split_shapes():
    source = torch.empty(4, 4)
    destination = torch.empty_like(source)
    source.is_qkv = True
    source.qkv_split_shapes = [256, 64, 64]

    copy_tensor_model_parallel_attributes(destination, source)

    assert destination.is_qkv is True
    assert destination.qkv_split_shapes == source.qkv_split_shapes
