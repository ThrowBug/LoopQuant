"""Cross-Loop Distribution Canonicalization (CLDC).

CLDC learns one shared activation/weight representation from all recurrent
invocations, then freezes that representation before precision allocation and
task calibration.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import (
    QUANTIZATION_CONTEXT,
    QuantizedLinear,
    fake_quantize_activation,
    fake_quantize_weight,
    orthogonal_transform,
    quantize_weights_gptq,
    rotate_ouro_backbone,
    wrap_quantized_modules,
)
from .grouping import GroupSize, validate_group_size


def _normalized_mse(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - reference.float()).square().sum() / reference.float().square().sum().clamp_min(1e-8)


def _channel_quantile(value: torch.Tensor, probability: float = 0.999) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).float()
    if flat.shape[0] > 8192:
        flat = flat[:8192]
    return torch.quantile(flat.abs(), probability, dim=0)


class _SharedCanonicalTransform(nn.Module):
    """Shared diagonal scale followed by the fixed structured orthogonal map."""

    def __init__(self, width: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(int(width)))

    def scales(self) -> torch.Tensor:
        return F.softplus(self.log_scale) + 1e-4

    def activation(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.scales().to(value.device, torch.float32)
        return orthogonal_transform(value.float() * scale).to(value.dtype)

    def weight(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.scales().to(value.device, torch.float32)
        return orthogonal_transform(value.float() / scale.unsqueeze(0)).to(value.dtype)


@dataclass
class _Canonicalization:
    scale: torch.Tensor
    permutation: torch.Tensor
    optimization: dict[str, float | int | None]


@torch.no_grad()
def collect_cross_loop_statistics(
    model: nn.Module,
    calibration_data: Iterable[torch.Tensor],
    device: str,
    max_sequences: int | None = None,
    max_tokens_per_loop: int = 4096,
) -> dict[str, dict[int, torch.Tensor]]:
    """Collect module inputs as CPU FP16 matrices indexed by recurrent loop."""
    store: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            loop = int(QUANTIZATION_CONTEXT.current_loop)
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            remaining = max_tokens_per_loop - counts[name][loop]
            if remaining > 0:
                part = value[:remaining].to("cpu", dtype=torch.float16)
                store[name][loop].append(part)
                counts[name][loop] += int(part.shape[0])
        return hook

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            handles.append(module.register_forward_hook(make_hook(name)))
    previous = QUANTIZATION_CONTEXT.enabled
    QUANTIZATION_CONTEXT.enabled = False
    try:
        model.eval()
        for index, sequence in enumerate(calibration_data):
            if max_sequences is not None and index >= int(max_sequences):
                break
            ids = sequence.unsqueeze(0) if sequence.ndim == 1 else sequence
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        QUANTIZATION_CONTEXT.enabled = previous
        for handle in handles:
            handle.remove()
    return {
        name: {loop: torch.cat(parts) for loop, parts in by_loop.items() if parts}
        for name, by_loop in store.items()
    }


def build_cross_loop_canonicalization(
    module: QuantizedLinear,
    activations_by_loop: dict[int, torch.Tensor],
    *,
    steps: int = 80,
    learning_rate: float = 1e-3,
    robust_temperature: float = 0.1,
    synchronization_weight: float = 0.1,
    regularization_weight: float = 1e-4,
) -> _Canonicalization:
    """Fit the shared representation for one recurrently reused module."""
    device = module.weight.device
    weight = module.weight.detach().float()
    transform = _SharedCanonicalTransform(module.in_features).to(device)
    optimizer = torch.optim.AdamW(transform.parameters(), lr=float(learning_rate))
    samples = {
        loop: value[:1024].to(device=device, dtype=torch.float32)
        for loop, value in activations_by_loop.items()
        if value.numel()
    }
    history: list[float] = []
    with torch.enable_grad():
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            transformed_weight = transform.weight(weight)
            quantized_weight = fake_quantize_weight(transformed_weight, 4, module.group_size)
            errors = []
            anomaly_profiles = []
            for loop in sorted(samples):
                original = samples[loop]
                transformed = transform.activation(original)
                quantized = fake_quantize_activation(transformed, 4, module.group_size)
                reference = F.linear(original, weight)
                reconstruction = F.linear(quantized, quantized_weight)
                errors.append(_normalized_mse(reconstruction, reference))
                profile = _channel_quantile(transformed).clamp_min(1e-8)
                anomaly_profiles.append(profile / profile.sum())
            if not errors:
                break
            stacked = torch.stack(errors)
            loss = robust_temperature * torch.logsumexp(stacked / robust_temperature, dim=0)
            if len(anomaly_profiles) > 1:
                profiles = torch.stack(anomaly_profiles)
                loss = loss + synchronization_weight * (profiles - profiles.mean(0)).square().mean()
            loss = loss + regularization_weight * (transform.scales() - 1.0).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transform.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
    with torch.no_grad():
        transformed_samples = [transform.activation(value.to(device).float()).cpu() for value in samples.values()]
        if transformed_samples:
            score = torch.stack([_channel_quantile(value) for value in transformed_samples]).amax(0)
            permutation = torch.argsort(score)
        else:
            permutation = torch.arange(module.in_features)
        scale = transform.scales().detach().cpu()
    return _Canonicalization(
        scale=scale,
        permutation=permutation.cpu(),
        optimization={
            "steps": len(history),
            "initial_loss": history[0] if history else None,
            "final_loss": history[-1] if history else None,
        },
    )


@torch.no_grad()
def apply_cldc_transform(module: QuantizedLinear, canonicalization: _Canonicalization) -> None:
    """Absorb a fitted CLDC transform into the weight and activation wrapper."""
    scale = canonicalization.scale.to(module.weight.device, module.weight.dtype)
    module.weight.data = orthogonal_transform(module.weight.data.float() / scale.float().unsqueeze(0)).to(module.weight.dtype)
    module.set_cldc_scale(scale)
    module.cldc_hadamard = True
    permutation = canonicalization.permutation.to(module.weight.device)
    module.weight.data = module.weight.data.index_select(1, permutation)
    module.cldc_permutation = permutation


class CrossLoopDistributionCanonicalization:
    """Build and freeze one shared quantization-friendly recurrent representation."""

    def __init__(
        self,
        *,
        group_size: GroupSize = 32,
        optimization_steps: int = 80,
        learning_rate: float = 1e-3,
        rotation_seed: int = 42,
        statistics_sequences: int = 16,
    ):
        self.group_size = validate_group_size(group_size)
        self.optimization_steps = int(optimization_steps)
        self.learning_rate = float(learning_rate)
        self.rotation_seed = int(rotation_seed)
        self.statistics_sequences = int(statistics_sequences)
        self.statistics: dict[str, dict[int, torch.Tensor]] = {}
        self.canonicalizations: dict[str, _Canonicalization] = {}

    def collect(self, model: nn.Module, calibration_data: torch.Tensor, device: str) -> dict[str, dict[int, torch.Tensor]]:
        rotate_ouro_backbone(model, seed=self.rotation_seed)
        modules = dict((name, module) for name, module in model.named_modules() if isinstance(module, QuantizedLinear))
        if not modules:
            wrap_quantized_modules(model, self.group_size)
        self.statistics = collect_cross_loop_statistics(
            model,
            calibration_data,
            device,
            max_sequences=min(self.statistics_sequences, len(calibration_data)),
        )
        return self.statistics

    def build(self, model: nn.Module) -> dict[str, _Canonicalization]:
        modules = {name: module for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}
        self.canonicalizations = {
            name: build_cross_loop_canonicalization(
                module,
                self.statistics.get(name, {}),
                steps=self.optimization_steps,
                learning_rate=self.learning_rate,
            )
            for name, module in modules.items()
        }
        return self.canonicalizations

    def apply(self, model: nn.Module, calibration_data: torch.Tensor, device: str) -> dict[str, Any]:
        modules = {name: module for name, module in model.named_modules() if isinstance(module, QuantizedLinear)}
        for name, canonicalization in self.canonicalizations.items():
            apply_cldc_transform(modules[name], canonicalization)
        gptq = quantize_weights_gptq(
            model,
            calibration_data,
            device,
            bits=4,
            group_size=self.group_size,
        )
        return {
            "modules": len(modules),
            "cross_loop_statistics_modules": len(self.statistics),
            "weight_quantization": "GPTQ W4 fake quantization",
            "weight_group_size": self.group_size,
            "gptq": gptq,
            "optimization": {name: value.optimization for name, value in self.canonicalizations.items()},
        }

    def run(self, model: nn.Module, calibration_data: torch.Tensor, device: str) -> dict[str, Any]:
        self.collect(model, calibration_data, device)
        self.build(model)
        return self.apply(model, calibration_data, device)
