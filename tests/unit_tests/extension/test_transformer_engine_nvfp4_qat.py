# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import megatron.core.extensions.transformer_engine_nvfp4_fake_qat as nvfp4_qat
from megatron.core.extensions.transformer_engine_int4_fake_qat import INT4_FAKE_QAT_FLAG


def _config(gradient_accumulation_fusion: bool = False, moe_single_grouped_weight: bool = False):
    return SimpleNamespace(
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        moe_single_grouped_weight=moe_single_grouped_weight,
    )


@pytest.fixture(autouse=True)
def _supported_te(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nvfp4_qat, "is_te_min_version", lambda _version: True)
    monkeypatch.setenv(INT4_FAKE_QAT_FLAG, "0")


class TestTransformerEngineNVFP4FakeQAT:
    def test_supported_with_discrete_rank2_weights(self):
        weights = [torch.nn.Parameter(torch.empty(4, 16)) for _ in range(3)]

        nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), False, weights)

    def test_rejects_transformer_engine_before_2_17(self, monkeypatch: pytest.MonkeyPatch):
        weight = torch.nn.Parameter(torch.empty(4, 16))
        monkeypatch.setattr(nvfp4_qat, "is_te_min_version", lambda _version: False)

        with pytest.raises(RuntimeError, match="Transformer Engine >= 2.17.0"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), False, [weight])

    def test_rejects_simultaneous_int4_and_nvfp4(self, monkeypatch: pytest.MonkeyPatch):
        weight = torch.nn.Parameter(torch.empty(4, 16))
        monkeypatch.setenv(INT4_FAKE_QAT_FLAG, "1")

        with pytest.raises(RuntimeError, match="mutually exclusive"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), False, [weight])

    def test_rejects_gradient_accumulation_fusion(self):
        weight = torch.nn.Parameter(torch.empty(4, 16))

        with pytest.raises(RuntimeError, match="gradient_accumulation_fusion"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(
                _config(gradient_accumulation_fusion=True), False, [weight]
            )

    def test_rejects_delayed_wgrad_compute(self):
        weight = torch.nn.Parameter(torch.empty(4, 16))

        with pytest.raises(RuntimeError, match="delayed wgrad"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), True, [weight])

    def test_rejects_fsdp_weight_attributes(self):
        weight = torch.nn.Parameter(torch.empty(4, 16))
        weight.__fsdp_param__ = True
        weight.get_main_grad = lambda: torch.empty_like(weight)

        with pytest.raises(RuntimeError, match="Megatron FSDP"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), False, [weight])

    def test_rejects_single_grouped_weight(self):
        weight = torch.nn.Parameter(torch.empty(3, 4, 16))

        with pytest.raises(RuntimeError, match="moe_single_grouped_weight"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(
                _config(moe_single_grouped_weight=True), False, [weight]
            )

    def test_rejects_non_rank2_weight(self):
        weight = torch.nn.Parameter(torch.empty(3, 4, 16))

        with pytest.raises(RuntimeError, match="rank-2 tensor"):
            nvfp4_qat._validate_nvfp4_fake_qat_support(_config(), False, [weight])

    def test_disabled_path_returns_original_list(self, monkeypatch: pytest.MonkeyPatch):
        weights = [torch.nn.Parameter(torch.empty(4, 16))]
        monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "0")
        monkeypatch.setattr(nvfp4_qat, "is_te_min_version", lambda _version: False)

        actual = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(_config(), False, weights)

        assert actual is weights

    def test_enabled_path_resolves_config_once_and_maps_arbitrary_weight_count(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        weights = [torch.nn.Parameter(torch.empty(4, 16)) for _ in range(3)]
        expected = [torch.empty_like(weight) for weight in weights]
        qdq_config = object()
        config_calls = 0
        calls = []

        fake_module = ModuleType("megatron.core.fusions.fused_nvfp4_qdq")

        def current_config():
            nonlocal config_calls
            config_calls += 1
            return qdq_config

        def fake_qdq(weight, config):
            calls.append((weight, config))
            return expected[len(calls) - 1]

        setattr(fake_module, "current_nvfp4_qdq_config", current_config)
        setattr(fake_module, "fake_nvfp4_quantization_ste", fake_qdq)
        monkeypatch.setitem(sys.modules, "megatron.core.fusions.fused_nvfp4_qdq", fake_module)
        monkeypatch.setenv(nvfp4_qat.NVFP4_FAKE_QAT_FLAG, "1")

        actual = nvfp4_qat.maybe_fake_quantize_nvfp4_weight_tensors(_config(), False, weights)

        assert config_calls == 1
        assert len(actual) == len(weights)
        assert all(value is expected_value for value, expected_value in zip(actual, expected))
        assert all(weight is call[0] for weight, call in zip(weights, calls))
        assert all(call[1] is qdq_config for call in calls)
