# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused CuTe DSL NVFP4 quantize-dequantize for fake QAT.

The kernel keeps the E4M3 block scale and packed E2M1 values in registers and
writes only the dequantized BF16/FP16 result. Its arithmetic order mirrors
Transformer Engine's 1D, 1x16, per-tensor NVFP4 implementation. The vectorized
load, FP4 conversion, and Four Over Six structure are adapted from FlashInfer's
CuTe DSL NVFP4 quantizer.

Supported contract:

* contiguous rank-2 BF16 or FP16 input on SM10x;
* 1x16 block scaling and a caller-provided FP32 per-tensor amax;
* round-to-nearest quantization with ordinary quant fast math disabled;
* standard NVFP4, plus the full Four Over Six MAE/MSE, E4M3-max 256/448,
  and exact/FP16-error matrix;
* no stochastic rounding, RHT, 2D quantization, transpose, or row scaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import cuda.bindings.driver as cuda  # type: ignore
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


_FP32_MAX = 3.4028234663852886e38
_FP4_BLOCK_SIZE = 16
_THREADS = 512
_BLOCKS_PER_SM = 4


class NVFP4QDQErrorMode(IntEnum):
    """Four Over Six candidate error metric."""

    MAE = 0
    MSE = 1


@dataclass(frozen=True)
class NVFP4QDQConfig:
    """Compile-time numerical configuration for fused NVFP4 QDQ."""

    use_4over6: bool = False
    e4m3_max: int = 448
    error_mode: NVFP4QDQErrorMode = NVFP4QDQErrorMode.MAE
    error_use_fp16: bool = False

    def __post_init__(self) -> None:
        if self.e4m3_max not in (256, 448):
            raise ValueError(f"NVFP4 E4M3 max must be 256 or 448, got {self.e4m3_max}.")
        if not self.use_4over6 and self.e4m3_max != 448:
            raise ValueError("E4M3 max 256 is only supported by Four Over Six.")
        if not self.use_4over6 and self.error_use_fp16:
            raise ValueError("The FP16 error contract only applies to Four Over Six.")


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _env_applies_to_weights(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in ("none", "inputs", "weights", "gradients", "all"):
        raise ValueError(
            f"{name} must be one of none, inputs, weights, gradients, or all; got {value!r}."
        )
    return value in ("weights", "all")


def current_nvfp4_qdq_config() -> NVFP4QDQConfig:
    """Resolve the weight QDQ contract from Transformer Engine environment variables."""
    if _env_flag("NVTE_USE_FAST_MATH"):
        raise ValueError(
            "Fused NVFP4 QDQ requires NVTE_USE_FAST_MATH=0; ordinary quant fast math "
            "is outside its numerical contract."
        )

    use_4over6 = _env_applies_to_weights("NVTE_NVFP4_4OVER6", "none")
    use_e4m3_256 = _env_applies_to_weights("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all")
    error_mode_name = os.getenv("NVTE_NVFP4_4OVER6_ERR_MODE", "MAE").strip().upper()
    try:
        error_mode = NVFP4QDQErrorMode[error_mode_name]
    except KeyError as exc:
        raise ValueError(
            f"NVTE_NVFP4_4OVER6_ERR_MODE must be MAE or MSE, got {error_mode_name!r}."
        ) from exc

    return NVFP4QDQConfig(
        use_4over6=use_4over6,
        e4m3_max=256 if use_4over6 and use_e4m3_256 else 448,
        error_mode=error_mode,
        # Despite the legacy variable name, this selects TE's FP16 candidate-error contract.
        error_use_fp16=(
            use_4over6 and _env_flag("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")
        ),
    )


@dsl_user_op
def _get_ptr(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    elem_ptr = tensor.iterator + offset
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _load_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _store_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fadd_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fsub_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fdiv_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmin(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fabs(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_abs(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bfloat2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_max_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bfloat2_max_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half2_to_f32x2(x: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bfloat2_to_f32x2(x: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e4m3_to_f32(a: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2;
                .reg .b16 lo, hi;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                mov.b32 {lo, hi}, h2;
                cvt.f32.f16 $0, lo;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cvt_e2m1x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;
                mov.b32 $0, {b0, b1, b2, b3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e2m1x2_to_f32x2(a: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(a).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            .reg .b32 h2;
            .reg .b16 lo, hi;
            .reg .b32 code_lo, code_hi, bits_lo, bits_hi;
            .reg .f32 f_lo, f_hi;
            .reg .pred negzero_lo, negzero_hi;

            mov.b32 {byte0, byte1, byte2, byte3}, $2;
            cvt.rn.f16x2.e2m1x2 h2, byte0;
            mov.b32 {lo, hi}, h2;
            cvt.f32.f16 f_lo, lo;
            cvt.f32.f16 f_hi, hi;
            mov.b32 bits_lo, f_lo;
            mov.b32 bits_hi, f_hi;
            and.b32 code_lo, $2, 0xF;
            shr.u32 code_hi, $2, 4;
            and.b32 code_hi, code_hi, 0xF;
            setp.eq.u32 negzero_lo, code_lo, 0x8;
            setp.eq.u32 negzero_hi, code_hi, 0x8;
            selp.u32 bits_lo, 0x80000000, bits_lo, negzero_lo;
            selp.u32 bits_hi, 0x80000000, bits_hi, negzero_hi;
            mov.b32 $0, bits_lo;
            mov.b32 $1, bits_hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _scaled_e2m1x2_e4m3_to_f32x2(
    packed: Uint32, scale: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Uint32(scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 fp8_pair, scale_h, unused_h, lo, hi;
            .reg .b32 q_h2, scale_h2, product_h2;
            mov.b32 {b0, b1, b2, b3}, $2;
            cvt.rn.f16x2.e2m1x2 q_h2, b0;
            cvt.u16.u32 fp8_pair, $3;
            cvt.rn.f16x2.e4m3x2 scale_h2, fp8_pair;
            mov.b32 {scale_h, unused_h}, scale_h2;
            mov.b32 scale_h2, {scale_h, scale_h};
            mul.rn.f16x2 product_h2, q_h2, scale_h2;
            mov.b32 {lo, hi}, product_h2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_f32x2_to_half2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.f16.f32 lo, $1;
                cvt.rn.f16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_f32x2_to_bfloat2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.bf16.f32 lo, $1;
                cvt.rn.bf16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _normal_block_scale(
    block_amax: Float32, global_encode_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """TE's intentionally associated ``amax * (S_enc * (1/6))`` expression."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_encode_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred zero;
                .reg .f32 scale_mul, result;
                setp.eq.f32 zero, $1, 0f00000000;
                mul.rn.f32 scale_mul, $2, 0f3E2AAAAB;
                mul.rn.f32 result, $1, scale_mul;
                selp.f32 $0, 0f00000000, result, zero;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _input_values(words: tuple, is_bfloat16: bool) -> tuple:
    values = ()
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr(is_bfloat16):
            lo, hi = _bfloat2_to_f32x2(words[i])
        else:
            lo, hi = _half2_to_f32x2(words[i])
        values = values + (lo, hi)
    return values


@cute.jit
def _block_amax(words: tuple, is_bfloat16: bool) -> Float32:
    maxima = ()
    for i in cutlass.range_constexpr(8):
        value = _half2_abs(words[i])
        maxima = maxima + (value,)

    if cutlass.const_expr(is_bfloat16):
        max01 = _bfloat2_max(maxima[0], maxima[1])
        max23 = _bfloat2_max(maxima[2], maxima[3])
        max45 = _bfloat2_max(maxima[4], maxima[5])
        max67 = _bfloat2_max(maxima[6], maxima[7])
        max_value = _bfloat2_max(_bfloat2_max(max01, max23), _bfloat2_max(max45, max67))
        return _bfloat2_max_to_f32(max_value)

    max01 = _half2_max(maxima[0], maxima[1])
    max23 = _half2_max(maxima[2], maxima[3])
    max45 = _half2_max(maxima[4], maxima[5])
    max67 = _half2_max(maxima[6], maxima[7])
    max_value = _half2_max(_half2_max(max01, max23), _half2_max(max45, max67))
    return _half2_max_to_f32(max_value)


@cute.jit
def _scale_values(values: tuple, scale: Float32) -> tuple:
    scaled = ()
    for i in cutlass.range_constexpr(16):
        scaled = scaled + (_fmul_rn(values[i], scale),)
    return scaled


@cute.jit
def _pack_e2m1(values: tuple) -> tuple[Uint32, Uint32]:
    lo = _cvt_e2m1x8(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
    )
    hi = _cvt_e2m1x8(
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        values[13],
        values[14],
        values[15],
    )
    return lo, hi


@cute.jit
def _decode_e2m1(lo: Uint32, hi: Uint32) -> tuple:
    values = ()
    for i in cutlass.range_constexpr(4):
        q0, q1 = _e2m1x2_to_f32x2(lo >> Uint32(8 * i))
        values = values + (q0, q1)
    for i in cutlass.range_constexpr(4):
        q0, q1 = _e2m1x2_to_f32x2(hi >> Uint32(8 * i))
        values = values + (q0, q1)
    return values


@cute.jit
def _decode_scaled_fp16(lo: Uint32, hi: Uint32, scale: Uint32) -> tuple:
    values = ()
    for i in cutlass.range_constexpr(4):
        q0, q1 = _scaled_e2m1x2_e4m3_to_f32x2(lo >> Uint32(8 * i), scale)
        values = values + (q0, q1)
    for i in cutlass.range_constexpr(4):
        q0, q1 = _scaled_e2m1x2_e4m3_to_f32x2(hi >> Uint32(8 * i), scale)
        values = values + (q0, q1)
    return values


@cute.jit
def _global_encode_scale(amax: Float32, e4m3_max: int) -> Float32:
    scale = _fdiv_rn(Float32(float(e4m3_max * 6)), amax)
    scale = _fmin(scale, Float32(_FP32_MAX))
    if amax == Float32(0.0):
        scale = Float32(1.0)
    if scale == Float32(0.0):
        scale = Float32(1.0)
    return scale


@cute.jit
def _candidate_inverse_scale(
    scale_f32: Float32, global_encode_scale: Float32, block_amax: Float32
) -> Float32:
    global_decode_scale = _fdiv_rn(Float32(1.0), global_encode_scale)
    product = _fmul_rn(scale_f32, global_decode_scale)
    inverse = _fmin(_fdiv_rn(Float32(1.0), product), Float32(_FP32_MAX))
    if block_amax == Float32(0.0):
        inverse = Float32(0.0)
    return inverse


@cute.jit
def _standard_quantize(
    values: tuple, block_amax: Float32, global_encode_scale: Float32
) -> tuple[Uint32, Uint32, Uint32]:
    scale_high_precision = _normal_block_scale(block_amax, global_encode_scale)
    scale = _cvt_f32_to_e4m3(scale_high_precision)
    scale_f32 = _cvt_e4m3_to_f32(scale)
    inverse = _candidate_inverse_scale(scale_f32, global_encode_scale, block_amax)
    lo, hi = _pack_e2m1(_scale_values(values, inverse))
    return scale, lo, hi


@cute.jit
def _candidate_error(
    original: tuple,
    lo: Uint32,
    hi: Uint32,
    scale: Uint32,
    global_amax: Float32,
    global_encode_scale: Float32,
    config: NVFP4QDQConfig,
) -> Float32:
    error = Float32(0.0)
    if cutlass.const_expr(config.error_use_fp16):
        candidate = _decode_scaled_fp16(lo, hi, scale)
        for i in cutlass.range_constexpr(16):
            original_scaled = _fmul_rn(original[i], global_encode_scale)
            diff = _fsub_rn(candidate[i], original_scaled)
            if cutlass.const_expr(config.error_mode == NVFP4QDQErrorMode.MSE):
                term = _fmul_rn(diff, diff)
            else:
                term = _fabs(diff)
            error = _fadd_rn(error, term)
        return error

    candidate = _decode_e2m1(lo, hi)
    denominator = Float32(float(6 * config.e4m3_max))
    scale_f32 = _cvt_e4m3_to_f32(scale)
    for i in cutlass.range_constexpr(16):
        dequant = _fmul_rn(candidate[i], scale_f32)
        dequant = _fmul_rn(dequant, global_amax)
        dequant = _fdiv_rn(dequant, denominator)
        diff = _fsub_rn(dequant, original[i])
        if cutlass.const_expr(config.error_mode == NVFP4QDQErrorMode.MSE):
            term = _fmul_rn(diff, diff)
        else:
            term = _fabs(diff)
        error = _fadd_rn(error, term)
    return error


@cute.jit
def _four_over_six_quantize(
    values: tuple,
    block_amax: Float32,
    global_amax: Float32,
    global_encode_scale: Float32,
    config: NVFP4QDQConfig,
) -> tuple[Uint32, Uint32, Uint32]:
    selected_scale = Uint32(0)
    selected_lo = Uint32(0)
    selected_hi = Uint32(0)

    if block_amax != Float32(0.0):
        # TE intentionally associates this differently from standard NVFP4.
        scale6_hp = _fmul_rn(_fdiv_rn(block_amax, Float32(6.0)), global_encode_scale)
        scale4_hp = _fmul_rn(scale6_hp, Float32(1.5))
        scale4 = _cvt_f32_to_e4m3(_fmin(scale4_hp, Float32(448.0)))
        scale6 = _cvt_f32_to_e4m3(_fmin(scale6_hp, Float32(448.0)))
        scale4_f32 = _cvt_e4m3_to_f32(scale4)
        scale6_f32 = _cvt_e4m3_to_f32(scale6)
        inv4 = _candidate_inverse_scale(scale4_f32, global_encode_scale, block_amax)
        inv6 = _candidate_inverse_scale(scale6_f32, global_encode_scale, block_amax)
        lo4, hi4 = _pack_e2m1(_scale_values(values, inv4))
        lo6, hi6 = _pack_e2m1(_scale_values(values, inv6))
        error4 = _candidate_error(
            values, lo4, hi4, scale4, global_amax, global_encode_scale, config
        )
        error6 = _candidate_error(
            values, lo6, hi6, scale6, global_amax, global_encode_scale, config
        )

        # Strict comparison is part of the contract: ties select map-to-6.
        selected_scale = scale6
        selected_lo = lo6
        selected_hi = hi6
        if error4 < error6:
            selected_scale = scale4
            selected_lo = lo4
            selected_hi = hi4

    return selected_scale, selected_lo, selected_hi


@cute.jit
def _dequantize(
    lo: Uint32,
    hi: Uint32,
    scale: Uint32,
    global_amax: Float32,
    e4m3_max: int,
) -> tuple:
    quantized = _decode_e2m1(lo, hi)
    scale_f32 = _cvt_e4m3_to_f32(scale)
    final_scale = _fmul_rn(scale_f32, global_amax)
    final_scale = _fmul_rn(final_scale, Float32(1.0 / float(6 * e4m3_max)))
    output = ()
    for i in cutlass.range_constexpr(16):
        output = output + (_fmul_rn(quantized[i], final_scale),)
    return output


@cute.jit
def _store_output(
    output: cute.Tensor, offset: Int32, values: tuple, is_bfloat16: bool
) -> None:
    packed = ()
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr(is_bfloat16):
            pair = _pack_f32x2_to_bfloat2(values[2 * i], values[2 * i + 1])
        else:
            pair = _pack_f32x2_to_half2(values[2 * i], values[2 * i + 1])
        packed = packed + (pair,)

    ptr0 = _get_ptr(output, offset)
    ptr1 = _get_ptr(output, offset + Int32(8))
    _store_v4_u32(ptr0, packed[0], packed[1], packed[2], packed[3])
    _store_v4_u32(ptr1, packed[4], packed[5], packed[6], packed[7])


class _NVFP4QDQKernel:
    """One thread processes one contiguous 1x16 quantization block."""

    def __init__(self, is_bfloat16: bool, config: NVFP4QDQConfig) -> None:
        self.is_bfloat16 = is_bfloat16
        self.config = config

    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amax: cute.Tensor,
        total_blocks: Int32,
        num_ctas: Int32,
        stream,
    ) -> None:
        self.kernel(input_tensor, output_tensor, global_amax, total_blocks).launch(
            grid=[num_ctas, 1, 1],
            block=[_THREADS, 1, 1],
            max_number_threads=[_THREADS, 1, 1],
            min_blocks_per_mp=1,
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        global_amax: cute.Tensor,
        total_blocks: Int32,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        grid_dim, _, _ = cute.arch.grid_dim()

        amax = Float32(global_amax[Int32(0)])
        global_encode_scale = _global_encode_scale(amax, self.config.e4m3_max)
        block = block_idx * Int32(_THREADS) + thread_idx
        stride = grid_dim * Int32(_THREADS)
        while block < total_blocks:
            offset = block * Int32(_FP4_BLOCK_SIZE)
            ptr0 = _get_ptr(input_tensor, offset)
            ptr1 = _get_ptr(input_tensor, offset + Int32(8))
            w0, w1, w2, w3 = _load_v4_u32(ptr0)
            w4, w5, w6, w7 = _load_v4_u32(ptr1)
            words = (w0, w1, w2, w3, w4, w5, w6, w7)
            values = _input_values(words, self.is_bfloat16)
            block_amax = _block_amax(words, self.is_bfloat16)

            if cutlass.const_expr(self.config.use_4over6):
                scale, lo, hi = _four_over_six_quantize(
                    values, block_amax, amax, global_encode_scale, self.config
                )
            else:
                scale, lo, hi = _standard_quantize(
                    values, block_amax, global_encode_scale
                )

            dequantized = _dequantize(lo, hi, scale, amax, self.config.e4m3_max)
            _store_output(output_tensor, offset, dequantized, self.is_bfloat16)
            block = block + stride


_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _validate_input(x: torch.Tensor, amax: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("Fused NVFP4 QDQ requires a CUDA tensor.")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Fused NVFP4 QDQ supports BF16 and FP16, got {x.dtype}.")
    if x.ndim != 2:
        raise ValueError(
            f"Fused NVFP4 QDQ requires a rank-2 tensor, got shape {tuple(x.shape)}."
        )
    if not x.is_contiguous():
        raise ValueError("Fused NVFP4 QDQ requires a contiguous tensor.")
    if x.shape[1] % _FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"Fused NVFP4 QDQ requires K divisible by {_FP4_BLOCK_SIZE}, got {x.shape[1]}."
        )
    if x.numel() == 0:
        raise ValueError("Fused NVFP4 QDQ does not support empty tensors.")
    if not amax.is_cuda or amax.device != x.device:
        raise ValueError(
            "The FP32 per-tensor amax must be on the input tensor's CUDA device."
        )
    if amax.dtype != torch.float32 or amax.numel() != 1:
        raise TypeError("The per-tensor amax must contain exactly one FP32 value.")
    capability = torch.cuda.get_device_capability(x.device)
    if capability[0] != 10:
        raise ValueError(
            f"Fused NVFP4 QDQ requires SM10x, got compute capability {capability}."
        )


def compute_nvfp4_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute the TE-compatible FP32 per-tensor amax with PyTorch."""
    if x.numel() == 0:
        raise ValueError("Cannot compute NVFP4 amax for an empty tensor.")
    return x.detach().abs().amax().to(torch.float32)


def fused_nvfp4_qdq(
    x: torch.Tensor,
    amax: torch.Tensor,
    config: Optional[NVFP4QDQConfig] = None,
) -> torch.Tensor:
    """Run register-resident NVFP4 QDQ and return a detached high-precision tensor."""
    if config is None:
        config = current_nvfp4_qdq_config()
    _validate_input(x, amax)

    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        input_flat = x.detach().view(-1)
        output_flat = output.view(-1)
        input_packed = from_dlpack(
            input_flat, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0)
        output_packed = from_dlpack(
            output_flat, assumed_align=16
        ).mark_compact_shape_dynamic(mode=0)
        amax_packed = from_dlpack(amax.detach().reshape(1), assumed_align=4)
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        total_blocks = x.numel() // _FP4_BLOCK_SIZE
        multiprocessors = torch.cuda.get_device_properties(
            x.device
        ).multi_processor_count
        num_ctas = max(
            1,
            min(
                (total_blocks + _THREADS - 1) // _THREADS,
                multiprocessors * _BLOCKS_PER_SM,
            ),
        )
        key = (
            x.device.index,
            torch.cuda.get_device_capability(x.device),
            x.dtype,
            config,
        )
        compiled = _KERNEL_CACHE.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up fused NVFP4 QDQ before CUDA graph capture.")
            kernel = _NVFP4QDQKernel(x.dtype == torch.bfloat16, config)
            compiled = cute.compile(
                kernel,
                input_packed,
                output_packed,
                amax_packed,
                total_blocks,
                num_ctas,
                stream,
            )
            _KERNEL_CACHE[key] = compiled
        compiled(
            input_packed,
            output_packed,
            amax_packed,
            total_blocks,
            num_ctas,
            stream,
        )
        return output


class _FusedNVFP4QDQSTE(torch.autograd.Function):
    """Identity backward around the non-differentiable fused QDQ kernel."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        amax: torch.Tensor,
        config: NVFP4QDQConfig,
    ) -> torch.Tensor:
        del ctx
        return fused_nvfp4_qdq(x, amax, config)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        del ctx
        return grad_output, None, None


def fake_nvfp4_quantization_ste(
    x: torch.Tensor, config: Optional[NVFP4QDQConfig] = None
) -> torch.Tensor:
    """Apply fused NVFP4 QDQ in forward and the straight-through estimator in backward."""
    if config is None:
        config = current_nvfp4_qdq_config()
    amax = compute_nvfp4_amax(x)
    output = _FusedNVFP4QDQSTE.apply(x, amax, config)
    if hasattr(x, "main_grad"):
        output.main_grad = x.main_grad
    return output


__all__ = [
    "NVFP4QDQConfig",
    "NVFP4QDQErrorMode",
    "compute_nvfp4_amax",
    "current_nvfp4_qdq_config",
    "fake_nvfp4_quantization_ste",
    "fused_nvfp4_qdq",
]
