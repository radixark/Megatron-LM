"""True-on-policy MoE re-exports.

The implementation is split across:
  * :mod:`moe_reduce`    -- ``sglang_moe_ep_tree_all_reduce`` deterministic EP combine.
  * :mod:`moe_experts`   -- ``SGLangGroupedMLP`` SGLang fused-expert helpers.
  * :mod:`moe_layer_ext` -- EP-invariant local-masked forward and padding compaction.

Importers should prefer the dedicated submodules; this module is kept as a
backward-compatible re-export point for older call sites.
"""

from .moe_experts import SGLangGroupedMLP
from .moe_reduce import sglang_moe_ep_tree_all_reduce

__all__ = [
    "SGLangGroupedMLP",
    "sglang_moe_ep_tree_all_reduce",
]
