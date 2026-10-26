"""Inference-only LoopQuant operators; checkpoint and calibration code stay unchanged.

``cached`` keeps the reference arithmetic, including its dtype boundaries. The
optional ``fast`` path uses the same transforms with different reduction orders.
All preparation happens after checkpoint restoration, before engine profiling.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .lctc import _fake_quantize_with_group_clip
from .quantization import _hadamard, _small_orthogonal, fake_quantize_activation


EXECUTION_MODES = ("cached", "compiled", "fast")


def fast_hadamard_reference(value: torch.Tensor) -> torch.Tensor:
    """CPU/test oracle for the normalized Sylvester map on the last dimension."""
    width = value.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = value.float()
    step = 1
    while step < width:
        paired = result.reshape(*result.shape[:-1], -1, 2, step)
        first, second = paired.unbind(-2)
        result = torch.stack((first + second, first - second), -2).reshape_as(result)
        step *= 2
    return result / math.sqrt(width)


class TransformCache:
    """One device copy per matrix shape, shared by every projection in an engine."""

    def __init__(self):
        self.matrices = {}

    def matrix(self, size, device, *, small=False):
        key = (size, torch.device(device), small)
        if key not in self.matrices:
            self.matrices[key] = (
                _small_orthogonal(size).to(device=device, dtype=torch.float32)
                if small else _hadamard(size, torch.device(device), torch.float32)
            )
        return self.matrices[key]


class InferenceOrthogonalTransform(nn.Module):
    def __init__(self, width, *, fast=False):
        super().__init__()
        self.odd, self.power = int(width), 1
        while self.odd % 2 == 0:
            self.odd //= 2
            self.power *= 2
        self.fast = fast
        self.register_buffer("hadamard", None, persistent=False)
        self.register_buffer("small", None, persistent=False)

    def prepare(self, cache, device):
        if self.power > 1 and not self.fast:
            self.hadamard = cache.matrix(self.power, device)
        if self.odd > 1:
            self.small = cache.matrix(self.odd, device, small=True)

    def forward(self, value):
        shape, dtype = value.shape, value.dtype
        result = value.reshape(-1, self.odd, self.power).float()
        if self.power > 1:
            if not self.fast:
                result = result @ self.hadamard.T
            elif value.is_cuda:
                from .vllm_kernels import hadamard_transform
                result = hadamard_transform(result)
            else:
                result = fast_hadamard_reference(result)
        if self.odd > 1:
            result = self.small @ result
        return result.reshape(shape).to(dtype)


class InferenceLinear(nn.Module):
    """A static (module, loop) view sharing the original projection's weight."""

    def __init__(self, linear, bits, loop, cgb=None, *, fast=False):
        super().__init__()
        self.linear = linear
        self.bits, self.loop, self.fast = int(bits), int(loop), fast
        self.has_lctc = cgb is not None
        # Keep the entry module, not its meta parameter objects: assign=True
        # restoration replaces the latter. prepare() reads the restored values.
        self.entry = cgb.entry(linear.name, loop) if cgb is not None else None
        self.transform = (InferenceOrthogonalTransform(linear.in_features, fast=fast)
                          if linear.cldc_hadamard else None)
        for name in ("scale", "permutation", "clip", "gain", "bias"):
            self.register_buffer(name, None, persistent=False)
        self.prepared = False

    def prepare(self, cache):
        linear = self.linear
        device, dtype = linear.weight.device, linear.weight.dtype
        self.scale = (None if linear.cldc_scale is None else
                      linear.cldc_scale.detach().to(device=device, dtype=dtype))
        self.permutation = linear.cldc_permutation.to(device)
        if self.transform is not None:
            self.transform.prepare(cache, device)
        if self.has_lctc:
            self.clip = self.entry.clip.detach()
            self.gain = self.entry.expanded(self.entry.gain.detach(), linear.out_features, device)
            # Preserve the reference's FP32 multiply before expansion.
            self.bias = self.entry.expanded(
                self.entry.sigma * self.entry.beta.detach(), linear.out_features, device)
        else:
            self.clip = linear.clip_ratios[min(self.loop, linear.clip_ratios.numel() - 1)].detach()
        self.prepared = True

    def forward(self, value):
        if not self.prepared:
            raise RuntimeError("prepare inference operators after restoring checkpoint tensors")
        if self.scale is not None:
            value = value * self.scale
        if self.transform is not None:
            value = self.transform(value)
        if self.fast and value.is_cuda and self.bits < 16:
            from .vllm_kernels import permute_quantize
            value = permute_quantize(value, self.permutation, self.clip, self.bits)
        else:
            if self.permutation.numel():
                value = value.index_select(-1, self.permutation)
            if self.has_lctc:
                value = _fake_quantize_with_group_clip(value, self.bits, self.clip, self.linear.group_size)
            else:
                value = fake_quantize_activation(value, self.bits, self.linear.group_size, self.clip)
        output = F.linear(value, self.linear.weight, self.linear.bias)
        if self.has_lctc:
            if self.fast and output.is_cuda:
                from .vllm_kernels import output_affine
                return output_affine(output, self.gain, self.bias)
            return (output.float() * self.gain + self.bias).to(output.dtype)
        return output


class InferenceResidualNorm(nn.Module):
    """Adapt the original Norm to vLLM's deferred-residual interface.

    The addition is explicitly rounded in the input dtype before the rotated
    Norm, matching HF. Native fused add+RMSNorm would move rounding boundaries.
    """

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden, residual=None):
        if residual is None:
            return self.norm(hidden)
        residual = residual + hidden
        return self.norm(residual), residual
