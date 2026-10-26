"""Optional floating-point Triton kernels for the vLLM ``fast`` execution mode.

No packed weights or low-bit GEMM. Import only on the CUDA inference path.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _hadamard_kernel(X, Y, WIDTH: tl.constexpr, STAGES: tl.constexpr, SCALE: tl.constexpr):
    column = tl.arange(0, WIDTH)
    row = tl.program_id(0)
    value = tl.load(X + row * WIDTH + column).to(tl.float32)
    for stage in tl.static_range(STAGES):
        other = tl.gather(value, column ^ (1 << stage), 0)
        value = tl.where((column & (1 << stage)) == 0, value + other, other - value)
    tl.store(Y + row * WIDTH + column, value * SCALE)


def hadamard_transform(value):
    value = value.contiguous()
    width = value.shape[-1]
    result = torch.empty_like(value, dtype=torch.float32)
    _hadamard_kernel[(value.numel() // width,)](
        value, result, width, width.bit_length() - 1, 1.0 / math.sqrt(width),
        num_warps=4 if width <= 2048 else 8, enable_fp_fusion=False)
    return result


@triton.jit
def _permute_quantize_kernel(X, P, C, Y, TOTAL_GROUPS, WIDTH: tl.constexpr,
                            QMAX: tl.constexpr, HAS_PERM: tl.constexpr,
                            GROUP_CLIP: tl.constexpr, BLOCK_GROUPS: tl.constexpr):
    group = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    channel = (group % (WIDTH // 32))[:, None] * 32 + tl.arange(0, 32)[None, :]
    source = channel
    if HAS_PERM:
        source = tl.load(P + channel)
    value = tl.load(X + (group // (WIDTH // 32))[:, None] * WIDTH + source,
                    group[:, None] < TOTAL_GROUPS, other=0).to(tl.float32)
    clip_index = group % (WIDTH // 32) if GROUP_CLIP else tl.full((BLOCK_GROUPS,), 0, tl.int32)
    ratio = tl.load(C + clip_index)
    # PyTorch CUDA divides by a Python scalar via an FP32 reciprocal multiply
    # (BinaryDivTrueKernel.cu), unlike the tensor/tensor division below.
    scale = (tl.maximum(tl.max(tl.abs(value), 1), 1.0e-8) * ratio) * tl.full((), 1.0 / QMAX, tl.float32)
    divided = libdevice.div_rn(value, scale[:, None])
    rounded = libdevice.nearbyint(divided)
    # Retain the reference STE's forward arithmetic (there is no backward here).
    rounded = (rounded - divided) + divided
    quantized = tl.minimum(tl.maximum(rounded, -QMAX), QMAX) * scale[:, None]
    tl.store(Y + group[:, None] * 32 + tl.arange(0, 32)[None, :],
             quantized, group[:, None] < TOTAL_GROUPS)


def permute_quantize(value, permutation, clip, bits):
    value = value.contiguous()
    width = value.shape[-1]
    result = torch.empty_like(value)
    groups = value.numel() // 32
    _permute_quantize_kernel[(triton.cdiv(groups, 8),)](
        value, permutation, clip, result, groups, width, 2 ** (bits - 1) - 1,
        permutation.numel() != 0, clip.numel() != 1, 8,
        num_warps=4, enable_fp_fusion=False)
    return result


@triton.jit
def _affine_kernel(X, G, B, Y, SIZE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < SIZE, other=0).to(tl.float32)
    gain = tl.load(G + index % WIDTH)
    bias = tl.load(B + index % WIDTH)
    tl.store(Y + index, value * gain + bias, index < SIZE)


def output_affine(value, gain, bias):
    value = value.contiguous()
    result = torch.empty_like(value)
    _affine_kernel[(triton.cdiv(value.numel(), 256),)](
        value, gain, bias, result, value.numel(), value.shape[-1], 256,
        num_warps=4, enable_fp_fusion=False)
    return result
