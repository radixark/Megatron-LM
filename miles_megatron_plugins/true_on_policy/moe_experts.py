"""``SGLangGroupedMLP`` — weight-layout adapter for SGLang MoE kernels.

No forward logic. The canonical forward is in ``sgl_fused_moe/forward.py``.
"""

from __future__ import annotations

import torch

from megatron.core.transformer.moe.experts import GroupedMLP


class SGLangGroupedMLP(GroupedMLP):
    """GroupedMLP subclass providing SGLang-compatible weight views.

    Megatron: weight1 [H, E*2*I], weight2 [E*I, H]
    SGLang:   w1 [E, 2*I, H],    w2 [E, H, I]
    """

    def sglang_w13_weight(self) -> torch.Tensor:
        """w1 (gate+up) in SGLang layout: [E, 2*I, H]."""
        return (
            self.weight1
            .view(self.num_local_experts, self.config.hidden_size, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

    def sglang_w2_weight(self) -> torch.Tensor:
        """w2 (down) in SGLang layout: [E, H, I]."""
        return (
            self.weight2
            .view(self.num_local_experts, -1, self.config.hidden_size)
            .permute(0, 2, 1)
            .contiguous()
        )
