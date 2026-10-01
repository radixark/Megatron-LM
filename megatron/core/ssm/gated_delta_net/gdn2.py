# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.jit import jit_fuser
from megatron.core.ssm.gated_delta_net.common import (
    a2a_cp_to_hp,
    causal_conv1d,
    get_parameter_local_cp,
    l2norm,
)
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

try:
    # The GDN2 kernel is only available in flash-linear-attention >= 0.5.1.
    from fla.ops.gdn2.chunk import chunk_gdn2

    HAVE_FLA_GDN2 = True
except ImportError:
    chunk_gdn2 = None
    HAVE_FLA_GDN2 = False

logger = logging.getLogger(__name__)


class GatedDeltaNet2(GatedDeltaNet):
    """GDN2 (Gated DeltaNet-2) layer class.

    GDN2 replaces GDN's per-head scalar decay and write strength with channel-wise
    gates, decoupling erase and write:

        S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

    where ``g_t`` is a per-key-channel log-decay, ``b_t`` (in R^{d_k}) is the
    channel-wise erase gate, and ``w_t`` (in R^{d_v}) is the channel-wise write gate.
    Reference: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
    (https://github.com/NVlabs/GatedDeltaNet-2).

    Note: unlike the GDN2 reference implementation, which uses low-rank decay and
    output-gate projections, all GDN2 projections are fused full-rank into the single
    column-parallel in_proj for TP/CP/SP simplicity.

    Subclasses ``GatedDeltaNet`` to reuse its ``forward``/headwise-and-chunkwise-CP/
    selective-recompute plumbing and only overrides the variant-specific projection
    sizing, gate math, and kernel dispatch. GDN2 only supports ``linear_cp_mode=
    'headwise'`` (or ``context_parallel_size=1``) and does not support
    ``gdn_pre_gated_delta_rule_fusion``; both are rejected in ``_setup_variant_attrs``.

    The layer takes input with size [s, b, h] and returns output of the same size.
    """

    def _setup_variant_attrs(self):
        """Set the GDN2 in_proj sizing, split tables, gate parameter dims, and kernel."""
        assert (
            chunk_gdn2 is not None or self.config.deterministic_mode
        ), "GDN2 requires flash-linear-attention >= 0.5.1 with the fla.ops.gdn2 kernel."
        assert self.config.linear_cp_mode != "chunkwise", (
            "GDN2 does not support linear_cp_mode='chunkwise'; use 'headwise' or "
            "context_parallel_size=1."
        )
        assert not self.config.gdn_pre_gated_delta_rule_fusion, (
            "gdn_pre_gated_delta_rule_fusion is not supported for "
            "experimental_attention_variant='gdn2'."
        )
        self.gdn_pre_gated_delta_rule_fusion = False

        # f (decay pre-activation), b (erase gate), w (write gate), on top of the
        # q/k/v/z sections the base class already accounts for.
        # TODO: for now, output gate is forced for GDN2.
        # We may remove this restriction in the future.
        self.in_proj_extra_dim = self.qk_dim * 2 + self.v_dim

        # Per-section sizes (and names) of the in_proj output, local to this TP rank.
        # Used for the CP head permutation (pre-a2a), for splitting the projection
        # output (post-a2a), and for the sharded checkpoint split of in_proj.weight.
        self.in_proj_split_names = ["query", "key", "value", "z", "f", "b", "w"]
        self.in_proj_split_sections = (
            self.qk_dim_local_tp,  # q
            self.qk_dim_local_tp,  # k
            self.v_dim_local_tp,  # v
            self.v_dim_local_tp,  # gate (z)
            self.qk_dim_local_tp,  # f (decay pre-activation)
            self.qk_dim_local_tp,  # b (erase gate)
            self.v_dim_local_tp,  # w (write gate)
        )

        # Time step projection (discretization): per-key-channel dt_bias and
        # per-key-head A_log, following the GDN2 reference implementation.
        self.dt_bias_dim = self.qk_dim_local_tp
        self.a_log_dim = self.num_key_heads // self.tp_size

        if self.config.deterministic_mode:
            self.gated_delta_rule = torch_chunk_gdn2
        else:
            self.gated_delta_rule = chunk_gdn2

    def _get_feat_dim_split(self, cp_size_headwise: int) -> tuple[int, int, int, int, int]:
        """Return GDN2 qkv/z/f/b/w split sizes for a runtime headwise CP size."""
        return (
            (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // cp_size_headwise,
            self.v_dim_local_tp // cp_size_headwise,
            self.qk_dim_local_tp // cp_size_headwise,
            self.qk_dim_local_tp // cp_size_headwise,
            self.v_dim_local_tp // cp_size_headwise,
        )

    def _reset_dt_bias(self):
        """Softplus-inverse init of dt_bias.

        Initializes so the initial per-channel step size lands in [1e-3, 0.1],
        following the GDN2 reference implementation.
        """
        dt = torch.exp(
            torch.rand(
                self.dt_bias.shape[0], dtype=torch.float32, device=torch.cuda.current_device()
            )
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)

    @jit_fuser
    def _compute_gates(
        self,
        A_log_local_cp: torch.Tensor,
        dt_bias_local_cp: torch.Tensor,
        batch: int,
        seq_len: int,
        *gate_feats: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the per-channel log-decay g and the erase/write gates b/w."""
        f, b, w = gate_feats
        # Channel-wise log-decay, computed in fp32 for numerical stability. A_log is a
        # per-key-head rate broadcast over the head's key channels; dt_bias is per-channel.
        g = -A_log_local_cp.float().exp().repeat_interleave(self.key_head_dim) * F.softplus(
            f.float() + dt_bias_local_cp
        )
        g = g.reshape(batch, seq_len, -1, self.key_head_dim)

        # Channel-wise erase (key axis) and write (value axis) gates, squashed to [0, 1]
        b = b.sigmoid().reshape(batch, seq_len, -1, self.key_head_dim)
        w = w.sigmoid().reshape(batch, seq_len, -1, self.value_head_dim)

        # Expand key-side gates across value-head groups (grouped value attention)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            g = g.repeat_interleave(repeat_factor, dim=2)
            b = b.repeat_interleave(repeat_factor, dim=2)

        return g, {"b": b.contiguous(), "w": w.contiguous()}

    def pre_gated_delta_rule(
        self,
        qkvzfbw,
        batch,
        seq_len,
        cp_size_headwise,
        cp_group_headwise,
        cu_seqlens_q=None,
        chunkwise_cp_context=None,
        packed_seq_params=None,
    ):
        """Prepare QKV, gate, and the erase/write/decay tensors before the gated delta rule."""

        qkvzfbw = qkvzfbw.transpose(0, 1)
        qkv, gate, f, b, w = torch.split(
            qkvzfbw, self._get_feat_dim_split(cp_size_headwise), dim=-1
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)

        nvtx_range_push(suffix="conv1d")
        kernel_seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=cp_group_headwise,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=cp_group_headwise,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // cp_size_headwise,
            )
            qkv = self.act_fn(conv_out[..., :kernel_seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ["silu", "swish"]
            orig_seq = qkv.shape[1]
            pad_n = 0
            conv_input = qkv.contiguous()
            conv_cu_seqlens = cu_seqlens_q
            if self.config.gdn_conv_pad_alignment is not None:
                if packed_seq_params is None or cu_seqlens_q is None:
                    raise ValueError(
                        "gdn_conv_pad_alignment is only supported with packed sequence "
                        "parameters in THD format. SBHD inputs do not need causal-conv padding."
                    )
                pad_n = -orig_seq % self.config.gdn_conv_pad_alignment
            if pad_n > 0:
                conv_input = torch.nn.functional.pad(conv_input, (0, 0, 0, pad_n))
                conv_cu_seqlens = cu_seqlens_q.clone()
                conv_cu_seqlens[-1] += pad_n
            qkv, _ = causal_conv1d(
                x=conv_input,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=conv_cu_seqlens,
                cp_context=None,
            )
            if pad_n > 0:
                qkv = qkv[:, :orig_seq, :]
        nvtx_range_pop(suffix="conv1d")

        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group_headwise)
        dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group_headwise)

        nvtx_range_push(suffix="prepare_input_for_gated_delta_rule")
        kernel_inputs = self._prepare_input_for_gated_delta_rule(
            qkv,
            gate,
            A_log_local_cp,
            dt_bias_local_cp,
            batch,
            kernel_seq_len,
            f,
            b,
            w,
            cp_size_headwise=cp_size_headwise,
        )
        nvtx_range_pop(suffix="prepare_input_for_gated_delta_rule")

        gate = kernel_inputs.pop("gate")

        return (
            kernel_inputs["q"],
            kernel_inputs["k"],
            kernel_inputs["v"],
            gate,
            kernel_inputs["b"],
            kernel_inputs["w"],
            kernel_inputs["g"],
        )

    def _forward_compute(
        self,
        hidden_states,
        batch,
        seq_len_post_headwise,
        cp_size_headwise,
        cp_group_headwise,
        cp_size_chunkwise,
        cp_group_chunkwise,
        cu_seqlens_q,
        packed_seq_params,
        chunkwise_cp_context,
    ):
        """Core GDN2 computation (in_proj -> conv1d -> gated_delta_rule -> norm -> out_proj)."""
        # ``_setup_variant_attrs`` already rejects linear_cp_mode='chunkwise' for GDN2, so
        # this path never actually sees chunkwise CP (cp_size_chunkwise is always 1 and
        # cp_group_chunkwise/chunkwise_cp_context are always None).
        assert cp_size_chunkwise == 1, "GDN2 does not support chunkwise context parallelism."

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzfbw, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvzfbw, thd_cp_a2a_inv = a2a_cp_to_hp(
            qkvzfbw,
            self.in_proj_split_sections,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            seq_len_post_headwise,
            packed_seq_params,
        )

        nvtx_range_push(suffix="pre_gated_delta_rule")
        query, key, value, gate, b, w, g = self.pre_gated_delta_rule(
            qkvzfbw,
            batch,
            seq_len_post_headwise,
            cp_size_headwise,
            cp_group_headwise,
            cu_seqlens_q,
            chunkwise_cp_context,
            packed_seq_params=packed_seq_params,
        )
        kernel_inputs = {"q": query, "k": key, "v": value, "g": g, "b": b, "w": w}
        nvtx_range_pop(suffix="pre_gated_delta_rule")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            **kernel_inputs,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        if self.recompute_norm_out and self.training:
            self.norm_out_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            norm_func = partial(
                self._gated_norm_and_layout_restore,
                thd_cp_a2a_inv=thd_cp_a2a_inv,
                batch=batch,
                seq_len=seq_len_post_headwise,
                packed_seq_params=packed_seq_params,
                cp_size_headwise=cp_size_headwise,
                cp_group_headwise=cp_group_headwise,
                cp_size_chunkwise=cp_size_chunkwise,
                cp_group_chunkwise=cp_group_chunkwise,
                cu_seqlens_q=cu_seqlens_q,
            )
            norm_out = self.norm_out_checkpoint.checkpoint(norm_func, core_attn_out, gate)
        else:
            norm_out = self._gated_norm_and_layout_restore(
                core_attn_out,
                gate,
                thd_cp_a2a_inv,
                batch,
                seq_len_post_headwise,
                packed_seq_params,
                cp_size_headwise,
                cp_group_headwise,
                cp_size_chunkwise,
                cp_group_chunkwise,
                cu_seqlens_q,
            )

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        if self.recompute_norm_out and self.training:
            self.norm_out_checkpoint.discard_output_and_register_recompute(out)

        return out, out_bias


####################
# Torch native gated delta rule 2
####################
def torch_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Torch-native chunkwise Gated Delta Rule-2, for deterministic mode.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: channel-wise log-decay of shape ``[B, T, H, K]``.
        b: channel-wise erase gate of shape ``[B, T, H, K]``.
        w: channel-wise write gate of shape ``[B, T, H, V]``.
        scale: attention scale. Defaults to ``1 / sqrt(K)``.
        chunk_size: chunk length of the WY schedule.
        initial_state: optional ``[B, H, K, V]`` initial state.
        output_final_state: whether to also return the final recurrent state.
        use_qk_l2norm_in_kernel: L2-normalize q and k here rather than in the caller.
        cu_seqlens: packed-sequence offsets; unsupported, must be ``None``.
        kwargs: accepted and ignored, so this stays interchangeable with the FLA
            kernel, which takes several options this implementation does not model.

    Returns:
        (tuple[Tensor, Tensor | None]): output of shape ``[B, T, H, V]`` and the
        final state, or ``None`` when ``output_final_state`` is ``False``.
    """
    assert cu_seqlens is None, "cu_seqlens is not supported for torch_chunk_gdn2 for now."

    initial_dtype = q.dtype
    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    # b s h d -> b h s d, and compute the whole recurrence in fp32
    query, key, value, g, b, w = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, g, b, w)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    # Zero padding is inert: it leaves the erase/write rows empty and, because the
    # padded log-decay is 0, leaves the chunk's cumulative decay at its last real value.
    query, key, value, g, b, w = [
        F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value, g, b, w)
    ]
    total_sequence_length = sequence_length + pad_size
    if scale is None:
        scale = 1 / (k_head_dim**0.5)
    query = query * scale

    # reshape to chunks
    query, key, value, g, b, w = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, b, w)
    ]

    # Channel-wise cumulative log-decay within each chunk.
    g = g.cumsum(dim=-2)
    decay = g.exp()

    # The pairwise decay exp(G_r - G_j) is carried on the operands, as
    # exp(G_r - c) * exp(c - G_j) for any per-channel c. Centering on half the
    # chunk's total decay halves the exponent range each operand has to represent,
    # which keeps exp() in fp32 range for roughly twice the decay strength.
    center = g[..., -1:, :] * 0.5
    decay_centered = (g - center).exp()
    inv_decay_centered = (center - g).exp()

    erase = decay * b * key  # E = exp(G) * b * k
    erase_centered = decay_centered * b * key
    key_inv_decay = key * inv_decay_centered  # Khat = exp(c - G) * k
    write = w * value  # Z = w * v

    # T = (I + A)^{-1} with A = tril(E @ Khat^T, -1), by forward substitution.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    attn = -(erase_centered @ key_inv_decay.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    write = attn @ write  # T @ Z
    k_cumdecay = attn @ erase  # T @ E

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(write)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )
    query_decay = query * decay  # Qtilde = exp(G) * q, exact: multiplies the incoming state
    query_decay_centered = query * decay_centered  # centered: only used pairwise against Khat

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        attn_i = query_decay_centered[:, :, i] @ key_inv_decay[:, :, i].transpose(-1, -2)
        attn_i = attn_i.masked_fill_(mask, 0)
        # U = T @ (Z - E @ S), the chunk's delta residuals against the incoming state
        u_i = write[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
        attn_inter = query_decay[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ u_i
        # Carry the state across the chunk: decay it by the chunk total, then add
        # the delta residuals mapped back through the keys. exp(G_C - G) <= 1, so this
        # ratio needs no centering.
        g_chunk = g[:, :, i, -1:]  # G_C, the chunk's total log-decay, [b, h, 1, k]
        key_bar = key[:, :, i] * (g_chunk - g[:, :, i]).exp()
        last_recurrent_state = (
            last_recurrent_state * g_chunk.squeeze(-2).unsqueeze(-1).exp()
            + key_bar.transpose(-1, -2) @ u_i
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
