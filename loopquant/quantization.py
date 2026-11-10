"""Group-wise weight and activation fake-quantization primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grouping import GroupSize, effective_group_size, validate_input_width


def round_ste(value: torch.Tensor) -> torch.Tensor:
    return (value.round() - value).detach() + value


def fake_quantize_groupwise(
    value: torch.Tensor,
    bits: int,
    group_size: GroupSize = 32,
    clip_ratio: float | torch.Tensor = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Symmetric fake quantization over groups on the last tensor dimension."""
    if bits >= 16:
        return value
    width = value.shape[-1]
    validate_input_width(width, group_size, name="last dimension")
    resolved = effective_group_size(group_size, width)
    original_dtype = value.dtype
    grouped = value.float().reshape(-1, width // resolved, resolved)
    ratio = torch.as_tensor(clip_ratio, device=grouped.device, dtype=grouped.dtype)
    if ratio.ndim == 1 and ratio.numel() == width // resolved:
        ratio = ratio.reshape(1, -1, 1)
    qmax = 2 ** (int(bits) - 1) - 1
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(eps) * ratio / qmax
    quantized = torch.clamp(round_ste(grouped / scale), -qmax, qmax) * scale
    return quantized.reshape_as(value).to(original_dtype)


def fake_quantize_weight(weight: torch.Tensor, bits: int = 4, group_size: GroupSize = 32) -> torch.Tensor:
    return fake_quantize_groupwise(weight, bits=bits, group_size=group_size)


def fake_quantize_activation(
    activation: torch.Tensor,
    bits: int,
    group_size: GroupSize = 32,
    clip_ratio: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    return fake_quantize_groupwise(activation, bits=bits, group_size=group_size, clip_ratio=clip_ratio)


class WeightQuantizer:
    """Per-output-channel quantizer used by the validated GPTQ reconstruction path."""

    def __init__(self) -> None:
        self.bits = 16
        self.mse = False
        self.scale: torch.Tensor | None = None
        self.maxq: torch.Tensor | None = None

    def configure(self, bits: int, mse: bool = True) -> None:
        self.bits = int(bits)
        self.mse = bool(mse)

    def find_params(self, value: torch.Tensor) -> None:
        flat = value.detach().float().flatten(1)
        maximum = torch.maximum(flat.amin(1).abs(), flat.amax(1).abs())
        qmax = 2 ** (self.bits - 1) - 1
        scale = maximum.clamp_min(1e-8) / qmax
        if self.mse:
            best_scale = scale.clone()
            best_error = torch.full_like(scale, float("inf"))
            for proportion in torch.linspace(1.0, 0.8, 20, device=flat.device):
                candidate = scale * proportion
                quantized = torch.clamp(torch.round(flat / candidate[:, None]), -qmax, qmax)
                error = ((quantized * candidate[:, None] - flat) ** 2).mean(1)
                use = error < best_error
                best_error = torch.where(use, error, best_error)
                best_scale = torch.where(use, candidate, best_scale)
            scale = best_scale
        self.scale = scale[:, None]
        self.maxq = torch.tensor(float(qmax), device=value.device)

    def quantize(self, value: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.maxq is None:
            return value
        quantized = torch.clamp(torch.round(value.float() / self.scale), -self.maxq, self.maxq)
        return (quantized * self.scale).to(value.dtype)


class GPTQWeightReconstruction:
    """Second-order reconstruction for one shared recurrent Linear weight."""

    def __init__(self, module: nn.Module, bits: int = 4, weight_clip: bool = True):
        self.module = module
        columns = module.weight.shape[1]
        self.hessian = torch.zeros(columns, columns, device=module.weight.device, dtype=torch.float32)
        self.num_samples = 0
        self.hook_count = 0
        self.quantizer = WeightQuantizer()
        self.quantizer.configure(bits, mse=weight_clip)

    def add_batch(self, activation: torch.Tensor) -> None:
        value = activation.detach()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        batch = value.shape[0]
        matrix = value.reshape(-1, value.shape[-1]).float().T
        self.hessian *= self.num_samples / (self.num_samples + batch)
        self.num_samples += batch
        matrix = math.sqrt(2 / self.num_samples) * matrix
        self.hessian += matrix @ matrix.T
        self.hook_count += 1

    def reconstruct(
        self,
        block_size: int = 128,
        damp_percent: float = 0.01,
        group_size: GroupSize = 32,
        activation_order: bool = False,
    ) -> None:
        weight = self.module.weight.data.clone().float()
        validate_input_width(weight.shape[1], group_size, name="weight width")
        resolved_group_size = effective_group_size(group_size, weight.shape[1])
        self.quantizer.find_params(weight)
        hessian = self.hessian
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0
        if activation_order:
            permutation = torch.argsort(torch.diag(hessian), descending=True)
            weight = weight[:, permutation]
            hessian = hessian[permutation][:, permutation]
            inverse = torch.argsort(permutation)
        else:
            inverse = None
        diagonal = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[diagonal, diagonal] += damp_percent * torch.mean(torch.diag(hessian))
        inverse_hessian = torch.linalg.cholesky(hessian)
        inverse_hessian = torch.cholesky_inverse(inverse_hessian)
        inverse_hessian = torch.linalg.cholesky(inverse_hessian, upper=True)
        reconstructed = torch.zeros_like(weight)
        for start in range(0, weight.shape[1], block_size):
            stop = min(start + block_size, weight.shape[1])
            block = weight[:, start:stop].clone()
            quantized_block = torch.zeros_like(block)
            errors = torch.zeros_like(block)
            local_inverse = inverse_hessian[start:stop, start:stop]
            for offset in range(stop - start):
                column = block[:, offset]
                divisor = local_inverse[offset, offset]
                if (start + offset) % resolved_group_size == 0:
                    self.quantizer.find_params(
                        weight[:, start + offset : start + offset + resolved_group_size]
                    )
                quantized = self.quantizer.quantize(column[:, None]).flatten()
                quantized_block[:, offset] = quantized
                error = (column - quantized) / divisor
                block[:, offset:] -= error[:, None] @ local_inverse[offset, offset:][None, :]
                errors[:, offset] = error
            reconstructed[:, start:stop] = quantized_block
            weight[:, stop:] -= errors @ inverse_hessian[start:stop, stop:]
        if inverse is not None:
            reconstructed = reconstructed[:, inverse]
        self.module.weight.data = reconstructed.to(self.module.weight.dtype)
        self.hessian = torch.empty(0, device=self.module.weight.device)


@torch.no_grad()
def quantize_weights_gptq(
    model: nn.Module,
    calibration_data: Iterable[torch.Tensor],
    device: str,
    bits: int = 4,
    group_size: GroupSize = 32,
    damp_percent: float = 0.01,
    activation_order: bool = False,
) -> dict[str, object]:
    """Accumulate Hessians over every recurrent invocation, then reconstruct each shared W4 weight once."""
    modules = {name: module for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}
    reconstructions = {name: GPTQWeightReconstruction(module, bits) for name, module in modules.items()}
    handles = []
    for name, module in modules.items():
        def make_hook(key: str):
            def hook(current: QuantizedLinear, inputs, _output):
                reconstructions[key].add_batch(current.transform_activation(inputs[0]))
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))
    previous = QUANTIZATION_CONTEXT.enabled
    QUANTIZATION_CONTEXT.enabled = False
    try:
        model.eval()
        for sequence in calibration_data:
            ids = sequence.unsqueeze(0) if sequence.ndim == 1 else sequence
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        QUANTIZATION_CONTEXT.enabled = previous
    hook_counts = {name: reconstruction.hook_count for name, reconstruction in reconstructions.items()}
    for reconstruction in reconstructions.values():
        reconstruction.reconstruct(
            damp_percent=damp_percent,
            group_size=group_size,
            activation_order=activation_order,
        )
    return {"module_count": len(modules), "hook_counts": hook_counts, "weight_bits": bits, "group_size": group_size}


@dataclass
class LoopQuantizationContext:
    num_loops: int = 4
    current_loop: int = 0
    activation_bits: tuple[int, ...] = (4, 4, 4, 4)
    enabled: bool = False

    def configure(self, num_loops: int, allocation: Sequence[int] | None = None) -> None:
        self.num_loops = int(num_loops)
        self.current_loop = 0
        self.set_allocation(allocation or [4] * self.num_loops)

    def set_allocation(self, allocation: Sequence[int]) -> None:
        values = tuple(int(bit) for bit in allocation)
        if len(values) != self.num_loops or any(bit not in (4, 8, 16) for bit in values):
            raise ValueError(f"invalid loop allocation {values} for {self.num_loops} loops")
        self.activation_bits = values

    @property
    def current_bits(self) -> int:
        return self.activation_bits[self.current_loop] if self.enabled else 16


QUANTIZATION_CONTEXT = LoopQuantizationContext()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@lru_cache(maxsize=32)
def _small_orthogonal(size: int, seed: int = 42) -> torch.Tensor:
    if size == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed + size)
    matrix = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    return q * torch.sign(torch.diag(r)).unsqueeze(0)


def _hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _is_power_of_two(size):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix / math.sqrt(size)


def orthogonal_transform(value: torch.Tensor) -> torch.Tensor:
    """Apply the validated Hadamard/Kronecker map on the last dimension."""
    width = value.shape[-1]
    power = 1
    odd = width
    while odd % 2 == 0:
        odd //= 2
        power *= 2
    shape, dtype = value.shape, value.dtype
    transformed = value.reshape(-1, odd, power).float()
    if power > 1:
        transformed = transformed @ _hadamard(power, transformed.device, transformed.dtype).T
    if odd > 1:
        transformed = _small_orthogonal(odd).to(transformed.device, transformed.dtype) @ transformed
    return transformed.reshape(shape).to(dtype)


def absorb_orthogonal_transform(module: nn.Linear) -> None:
    module.weight.data = orthogonal_transform(module.weight.data.float()).to(module.weight.dtype)


def _random_hadamard_matrix(size: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if _is_power_of_two(size):
        matrix = _hadamard(size, torch.device("cpu"), torch.float64)
        signs = torch.randint(0, 2, (size,), generator=generator).mul(2).sub(1).double()
        return (signs[:, None] * matrix).to(device)
    gaussian = torch.randn(size, size, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return (q * torch.sign(torch.diag(r)).unsqueeze(0)).to(device)


class _RotatedRMSNorm(nn.Module):
    def __init__(self, norm: nn.Module, rotation: torch.Tensor):
        super().__init__()
        self.norm = norm
        self.register_buffer("rotation", rotation.detach().float(), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        rotation = self.rotation.to(hidden_states.device)
        unrotated = hidden_states.float() @ rotation.T
        return (self.norm(unrotated.to(dtype)).float() @ rotation).to(dtype)


@torch.no_grad()
def rotate_ouro_backbone(model: nn.Module, seed: int = 42) -> torch.Tensor:
    """Apply the validated shared Ouro basis rotation used before CLDC/GPTQ."""
    if hasattr(model, "_loopquant_rotation"):
        return model._loopquant_rotation
    core = model.model
    hidden_size = int(model.config.hidden_size)
    rotation = _random_hadamard_matrix(hidden_size, next(model.parameters()).device, seed)

    def rotate_input(linear: nn.Linear) -> None:
        dtype = linear.weight.dtype
        linear.weight.data = (linear.weight.data.double() @ rotation.to(linear.weight.device)).to(dtype)

    def rotate_output(linear: nn.Linear) -> None:
        dtype = linear.weight.dtype
        linear.weight.data = (rotation.T.to(linear.weight.device) @ linear.weight.data.double()).to(dtype)
        if linear.bias is not None:
            linear.bias.data = (rotation.T.to(linear.bias.device) @ linear.bias.data.double()).to(linear.bias.dtype)

    dtype = core.embed_tokens.weight.dtype
    core.embed_tokens.weight.data = (core.embed_tokens.weight.data.double() @ rotation).to(dtype)
    rotate_input(model.lm_head)
    if hasattr(core, "early_exit_gate"):
        rotate_input(core.early_exit_gate)
    for layer in core.layers:
        attention = layer.self_attn
        mlp = layer.mlp
        for module in (attention.q_proj, attention.k_proj, attention.v_proj, mlp.gate_proj, mlp.up_proj):
            rotate_input(module)
        rotate_output(attention.o_proj)
        rotate_output(mlp.down_proj)
    for owner in (core, *list(core.layers)):
        for name, child in list(owner.named_children()):
            if "RMSNorm" in type(child).__name__:
                setattr(owner, name, _RotatedRMSNorm(child, rotation))
    model._loopquant_rotation = rotation
    model._loopquant_rotation_seed = int(seed)
    return rotation


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class QuantizedLinear(nn.Module):
    """Linear layer with shared W4 weights and loop-selected A4/A8 fake quantization."""

    def __init__(self, linear: nn.Linear, name: str, group_size: GroupSize = 32):
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        validate_input_width(self.in_features, group_size, name=f"{name} input width")
        self.group_size = group_size
        self.cldc_hadamard = False
        self.cldc_scale: nn.Parameter | None = None
        self.register_buffer("cldc_permutation", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("clip_ratios", torch.ones(QUANTIZATION_CONTEXT.num_loops), persistent=True)

    def set_cldc_scale(self, scale: torch.Tensor) -> None:
        self.cldc_scale = nn.Parameter(scale.detach().clone(), requires_grad=False)

    def transform_activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.cldc_scale is not None:
            value = value * self.cldc_scale.to(value.device, value.dtype)
        if self.cldc_hadamard:
            value = orthogonal_transform(value)
        if self.cldc_permutation.numel():
            value = value.index_select(-1, self.cldc_permutation.to(value.device))
        return value

    def quantize_activation(self, value: torch.Tensor) -> torch.Tensor:
        bits = QUANTIZATION_CONTEXT.current_bits
        if bits >= 16:
            return value
        loop = QUANTIZATION_CONTEXT.current_loop
        ratio = self.clip_ratios[min(loop, self.clip_ratios.numel() - 1)]
        return fake_quantize_activation(value, bits, self.group_size, ratio)

    def set_clip_ratios(self, ratios: Sequence[float]) -> None:
        values = torch.as_tensor(ratios, device=self.clip_ratios.device, dtype=torch.float32)
        if values.numel() == 1:
            values = values.repeat(self.clip_ratios.numel())
        self.clip_ratios.copy_(values[: self.clip_ratios.numel()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.transform_activation(value)
        from .lctc import ACTIVE_LCTC_RUNTIME

        runtime = ACTIVE_LCTC_RUNTIME.get()
        if runtime is not None and runtime.enabled:
            return runtime.forward_linear(self, value)
        return F.linear(self.quantize_activation(value), self.weight, self.bias)


def is_quantization_target(name: str) -> bool:
    return not any(part in name for part in ("lm_head", "embed", "early_exit")) and name.endswith(TARGET_SUFFIXES)


def wrap_quantized_modules(model: nn.Module, group_size: GroupSize = 32) -> dict[str, QuantizedLinear]:
    wrapped: dict[str, QuantizedLinear] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not is_quantization_target(name):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = QuantizedLinear(module, name, group_size)
        setattr(parent, child_name, replacement)
        wrapped[name] = replacement
    return wrapped
