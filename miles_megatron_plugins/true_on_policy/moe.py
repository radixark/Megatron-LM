"""True-on-policy MoE re-exports.

The implementation is split across:
  * :mod:`moe_context`   -- ``SGLangMoERolloutContext`` and the rollout-segment context manager.
  * :mod:`moe_reduce`    -- ``sglang_moe_ep_tree_all_reduce`` deterministic EP combine.
  * :mod:`moe_experts`   -- ``SGLangGroupedMLP`` with the SGLang-compatible no-grad forward.
  * :mod:`moe_combine`   -- SGLang-ordered MoE unpermute for deterministic combine.
  * :mod:`moe_layer_ext` -- EP-invariant local-masked forward and padding compaction.

Importers should prefer the dedicated submodules; this module is kept as a
backward-compatible re-export point for older call sites.
"""

from .moe_combine import sglang_ordered_moe_unpermute
from .moe_context import (
    SGLangMoERolloutContext,
    get_sglang_moe_rollout_context,
    sglang_moe_rollout_context,
)
from .moe_experts import SGLangGroupedMLP
from .moe_reduce import sglang_moe_ep_tree_all_reduce

__all__ = [
    "SGLangGroupedMLP",
    "SGLangMoERolloutContext",
    "get_sglang_moe_rollout_context",
    "sglang_moe_ep_tree_all_reduce",
    "sglang_moe_rollout_context",
    "sglang_ordered_moe_unpermute",
]
