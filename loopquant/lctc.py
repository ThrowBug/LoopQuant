"""Loop-Conditioned Task Calibration (LCTC).

LCTC owns calibration of a frozen quantized recurrent operator. Its deployed
state is a static loop/module-specific clipping, gain, and bias table. The
task-space training objective is deliberately replaceable; the included
objective follows the validated task-transport secant formulation.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .quantization import QUANTIZATION_CONTEXT, QuantizedLinear, round_ste


@dataclass(frozen=True)
class _CGBSpec:
    module_name: str
    loop_id: int
    input_groups: int
    output_groups: int
    out_features: int


class _CGBEntry(nn.Module):
    def __init__(self, spec: _CGBSpec):
        super().__init__()
        self.spec = spec
        self.clip = nn.Parameter(torch.ones(spec.input_groups, dtype=torch.float32))
        self.gain = nn.Parameter(torch.ones(spec.output_groups, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(spec.output_groups, dtype=torch.float32))
        self.register_buffer("sigma", torch.ones(spec.output_groups, dtype=torch.float32), persistent=True)
        self.register_buffer("clip_initial", torch.ones(spec.input_groups, dtype=torch.float32), persistent=True)

    def expanded(self, value: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
        return value.to(device=device, dtype=torch.float32).repeat_interleave(32)[:width]


class LoopConditionedCGB(nn.Module):
    """Static clipping/gain/bias parameters for every module and recurrent loop."""

    def __init__(self, specs: Iterable[_CGBSpec], group_size: int = 32):
        super().__init__()
        if int(group_size) != 32:
            raise ValueError("the validated LCTC deployment uses group size 32")
        self.group_size = int(group_size)
        self.entries = nn.ModuleDict()
        self._lookup: dict[tuple[str, int], str] = {}
        self._specs: dict[str, _CGBSpec] = {}
        for spec in specs:
            key = "m_" + hashlib.blake2b(
                spec.module_name.encode(), digest_size=8, person=b"LQ-CGB-v1"
            ).hexdigest() + f"_t{spec.loop_id}"
            self.entries[key] = _CGBEntry(spec)
            self._lookup[(spec.module_name, spec.loop_id)] = key
            self._specs[key] = spec

    @classmethod
    def from_model(cls, model: nn.Module, num_loops: int, group_size: int = 32) -> "LoopConditionedCGB":
        specs = []
        for name, module in model.named_modules():
            if not isinstance(module, QuantizedLinear):
                continue
            for loop_id in range(int(num_loops)):
                specs.append(
                    _CGBSpec(
                        module_name=module.name or name,
                        loop_id=loop_id,
                        input_groups=math.ceil(module.in_features / group_size),
                        output_groups=math.ceil(module.out_features / group_size),
                        out_features=module.out_features,
                    )
                )
        return cls(specs, group_size)

    def entry(self, module_name: str, loop_id: int) -> _CGBEntry:
        return self.entries[self._lookup[(module_name, int(loop_id))]]

    def iter_entries(self):
        for key, entry in self.entries.items():
            yield self._specs[key], entry

    @torch.no_grad()
    def project_(
        self,
        clip_range: tuple[float, float] = (0.5, 1.0),
        gain_range: tuple[float, float] = (0.8, 1.2),
        beta_range: tuple[float, float] = (-0.1, 0.1),
    ) -> None:
        for _, entry in self.iter_entries():
            entry.clip.clamp_(*clip_range)
            entry.gain.clamp_(*gain_range)
            entry.beta.clamp_(*beta_range)

    def regularizer(self) -> torch.Tensor:
        pieces = []
        for _, entry in self.iter_entries():
            pieces.extend(
                ((entry.clip - entry.clip_initial).square().mean(),
                 (entry.gain - 1.0).square().mean(), entry.beta.square().mean())
            )
        return torch.stack(pieces).mean() if pieces else torch.tensor(0.0)

    def summary(self) -> dict[str, Any]:
        entries = list(self.iter_entries())
        coordinates = {
            "clip": sum(entry.clip.numel() for _, entry in entries),
            "gain": sum(entry.gain.numel() for _, entry in entries),
            "bias": sum(entry.beta.numel() for _, entry in entries),
        }
        return {
            "physical_modules": len({spec.module_name for spec, _ in entries}),
            "module_loop_entries": len(entries),
            "loops": len({spec.loop_id for spec, _ in entries}),
            "coordinates": coordinates,
            "total_coordinates": sum(coordinates.values()),
            "static_bytes": sum(value.numel() * value.element_size() for value in self.parameters()),
            "request_history_bytes": 0,
        }


def _fake_quantize_with_group_clip(
    value: torch.Tensor,
    bits: int,
    clip: torch.Tensor,
    group_size: int = 32,
    eps: float = 1e-8,
) -> torch.Tensor:
    if bits >= 16:
        return value
    width = value.shape[-1]
    if width % group_size:
        raise ValueError(f"last dimension {width} is not divisible by group size {group_size}")
    groups = width // group_size
    if clip.numel() != groups:
        raise ValueError(f"clip group count {clip.numel()} does not match {groups}")
    original_dtype = value.dtype
    grouped = value.float().reshape(-1, groups, group_size)
    ratio = clip.to(value.device).reshape(1, groups, 1)
    qmax = 2 ** (int(bits) - 1) - 1
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(eps) * ratio / qmax
    quantized = torch.clamp(round_ste(grouped / scale), -qmax, qmax) * scale
    return quantized.reshape_as(value).to(original_dtype)


class _LCTCRuntime:
    def __init__(self, cgb: LoopConditionedCGB, enabled: bool = True):
        self.cgb = cgb
        self.enabled = bool(enabled)

    def forward_linear(self, module: QuantizedLinear, value: torch.Tensor) -> torch.Tensor:
        loop_id = int(QUANTIZATION_CONTEXT.current_loop)
        entry = self.cgb.entry(module.name, loop_id)
        quantized = _fake_quantize_with_group_clip(
            value, QUANTIZATION_CONTEXT.current_bits, entry.clip, module.group_size
        )
        output = F.linear(quantized, module.weight, module.bias)
        gain = entry.expanded(entry.gain, output.shape[-1], output.device)
        bias = entry.expanded(entry.sigma * entry.beta, output.shape[-1], output.device)
        return (output.float() * gain + bias).to(output.dtype)


ACTIVE_LCTC_RUNTIME: ContextVar[_LCTCRuntime | None] = ContextVar(
    "loopquant_lctc_runtime", default=None
)


@contextmanager
def _activate_runtime(runtime: _LCTCRuntime | None) -> Iterator[None]:
    if runtime is None:
        yield
        return
    token = ACTIVE_LCTC_RUNTIME.set(runtime)
    try:
        yield
    finally:
        ACTIVE_LCTC_RUNTIME.reset(token)


def initialize_loop_conditioned_cgb(
    model: nn.Module,
    num_loops: int,
    group_size: int = 32,
) -> LoopConditionedCGB:
    """Create neutral static LCTC parameters for all quantized module-loop pairs."""
    return LoopConditionedCGB.from_model(model, num_loops, group_size)


@torch.no_grad()
def _collect_teacher_module_samples(
    teacher: nn.Module,
    student: nn.Module,
    calibration_data: Sequence[torch.Tensor],
    device: str,
    tokens_per_sequence: int = 4,
) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    student_names = {name for name, module in student.named_modules() if isinstance(module, QuantizedLinear)}
    collected: dict[tuple[str, int], dict[str, list[torch.Tensor]]] = {}
    sample_id = {"value": 0}
    handles = []
    for name, module in teacher.named_modules():
        if name not in student_names or not isinstance(module, nn.Linear):
            continue

        def make_hook(module_name: str):
            def hook(_module, inputs, output):
                length = inputs[0].shape[-2]
                count = min(int(tokens_per_sequence), length)
                positions = torch.linspace(0, length - 1, steps=count + 2, device=inputs[0].device)[1:-1].round().long()
                positions = (positions + sample_id["value"] % max(length // max(count, 1), 1)).remainder(length).sort().values
                key = (module_name, int(QUANTIZATION_CONTEXT.current_loop))
                row = collected.setdefault(key, {"x": [], "y": []})
                row["x"].append(inputs[0].index_select(-2, positions).detach().cpu().to(torch.bfloat16))
                row["y"].append(output.index_select(-2, positions).detach().cpu().to(torch.bfloat16))
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    previous = QUANTIZATION_CONTEXT.enabled
    QUANTIZATION_CONTEXT.enabled = False
    try:
        for index, sequence in enumerate(calibration_data):
            sample_id["value"] = index
            ids = sequence.unsqueeze(0) if sequence.ndim == 1 else sequence
            teacher(input_ids=ids.to(device), use_cache=False)
    finally:
        QUANTIZATION_CONTEXT.enabled = previous
        for handle in handles:
            handle.remove()
    return {
        key: {
            kind: torch.cat(parts, dim=-2).reshape(-1, parts[0].shape[-1])
            for kind, parts in values.items()
        }
        for key, values in collected.items()
    }


@torch.no_grad()
def _initialize_from_local_reconstruction(
    cgb: LoopConditionedCGB,
    student: nn.Module,
    teacher_samples: dict[tuple[str, int], dict[str, torch.Tensor]],
    allocation: Sequence[int],
    clip_grid: Sequence[float],
) -> None:
    modules = {name: module for name, module in student.named_modules() if isinstance(module, QuantizedLinear)}
    for (name, loop_id), samples in teacher_samples.items():
        module = modules[name]
        raw = samples["x"].to(module.weight.device, module.weight.dtype)
        reference = samples["y"].to(module.weight.device).float()
        transformed = module.transform_activation(raw)
        entry = cgb.entry(name, loop_id)
        best_ratio, best_error = 1.0, float("inf")
        for ratio in clip_grid:
            clip = torch.full_like(entry.clip, float(ratio))
            quantized = _fake_quantize_with_group_clip(
                transformed, int(allocation[loop_id]), clip, module.group_size
            )
            error = float((F.linear(quantized, module.weight, module.bias).float() - reference).square().mean())
            if error < best_error:
                best_ratio, best_error = float(ratio), error
        entry.clip.fill_(best_ratio)
        entry.clip_initial.fill_(best_ratio)
        width = reference.shape[-1]
        groups = math.ceil(width / cgb.group_size)
        if width % cgb.group_size:
            reference = F.pad(reference, (0, groups * cgb.group_size - width))
        sigma = reference.reshape(-1, groups, cgb.group_size).square().mean((0, 2)).sqrt().clamp_min(1e-6)
        entry.sigma.copy_(sigma)


@dataclass
class _LoopMetadata:
    causal_masks: dict[str, torch.Tensor | None]
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attention_mask: torch.Tensor | None


def _prepare_loop_inputs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    core = model.model
    embeddings = core.embed_tokens(input_ids)
    cache_position = torch.arange(embeddings.shape[1], device=embeddings.device)
    position_ids = cache_position.unsqueeze(0)
    implementation = importlib.import_module(type(core).__module__)
    mask_arguments = {
        "config": core.config,
        "input_embeds": embeddings,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {"full_attention": implementation.create_causal_mask(**mask_arguments)}
    if getattr(core, "has_sliding_layers", False):
        masks["sliding_attention"] = implementation.create_sliding_window_causal_mask(**mask_arguments)
    metadata = _LoopMetadata(
        masks, position_ids, cache_position, core.rotary_emb(embeddings, position_ids), attention_mask
    )
    return embeddings, metadata


def _run_one_loop(model, state: torch.Tensor, loop_id: int, metadata: _LoopMetadata) -> torch.Tensor:
    hidden = state
    core = model.model
    for layer in core.layers[: core.config.num_hidden_layers]:
        active_runtime = ACTIVE_LCTC_RUNTIME.get()

        def layer_forward(value, current_layer=layer, runtime=active_runtime):
            # Checkpoint recomputation occurs after the surrounding context
            # manager has exited, so restore the same static LCTC runtime.
            with _activate_runtime(runtime):
                return current_layer(
                    value,
                    attention_mask=metadata.causal_masks[current_layer.attention_type],
                    position_ids=metadata.position_ids,
                    past_key_value=None,
                    use_cache=False,
                    cache_position=metadata.cache_position,
                    position_embeddings=metadata.position_embeddings,
                    current_ut=int(loop_id),
                )

        hidden = (
            checkpoint(layer_forward, hidden, use_reentrant=False)
            if torch.is_grad_enabled()
            else layer_forward(hidden)
        )
    return core.norm(hidden)


def _run_trajectory(model, input_ids: torch.Tensor, runtime: _LCTCRuntime | None = None):
    state, metadata = _prepare_loop_inputs(model, input_ids)
    entries, outputs = [], []
    with _activate_runtime(runtime):
        for loop_id in range(int(model.model.total_ut_steps)):
            entries.append(state)
            state = _run_one_loop(model, state, loop_id, metadata)
            outputs.append(state)
    return entries, outputs, metadata


def _run_suffix(model, state: torch.Tensor, start_loop: int, metadata: _LoopMetadata) -> torch.Tensor:
    hidden = state
    for loop_id in range(int(start_loop), int(model.model.total_ut_steps)):
        hidden = _run_one_loop(model, hidden, loop_id, metadata)
    return model.lm_head(hidden)


def _token_mask(input_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask[:, 0] = False
    return mask


def _fisher_energy(
    teacher_logits: torch.Tensor,
    vector: torch.Tensor,
    token_mask: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    logits, values = teacher_logits.float(), vector.float()
    normalizer = torch.logsumexp(logits, -1, keepdim=True)
    mean = torch.zeros_like(logits[..., 0])
    for start in range(0, logits.shape[-1], chunk_size):
        probability = torch.exp(logits[..., start : start + chunk_size] - normalizer)
        mean += (probability * values[..., start : start + chunk_size]).sum(-1)
    energy = torch.zeros_like(mean)
    for start in range(0, logits.shape[-1], chunk_size):
        probability = torch.exp(logits[..., start : start + chunk_size] - normalizer)
        centered = values[..., start : start + chunk_size] - mean.unsqueeze(-1)
        energy += (probability * centered.square()).sum(-1)
    mask = token_mask.to(energy.device, energy.dtype)
    return (energy * mask).sum() / mask.sum().clamp_min(1)


def compute_task_transport_secant_loss(
    student: nn.Module,
    teacher: nn.Module,
    runtime: _LCTCRuntime,
    teacher_state: torch.Tensor,
    inherited_drift: torch.Tensor,
    loop_id: int,
    alpha: float,
    teacher_logits: torch.Tensor,
    student_metadata: _LoopMetadata,
    teacher_metadata: _LoopMetadata,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal LCTC objective measuring local and propagated task-space defects."""

    def defect(state: torch.Tensor) -> torch.Tensor:
        with _activate_runtime(runtime):
            quantized_next = _run_one_loop(student, state, loop_id, student_metadata)
        return _run_suffix(teacher, quantized_next, loop_id + 1, teacher_metadata) - _run_suffix(
            teacher, state, loop_id, teacher_metadata
        )

    anchor = defect(teacher_state)
    perturbed = defect(teacher_state + float(alpha) * inherited_drift.detach())
    secant = perturbed - anchor
    return (
        _fisher_energy(teacher_logits, anchor, token_mask),
        _fisher_energy(teacher_logits, secant, token_mask) / float(alpha) ** 2,
    )


def _distillation_terms(student_logits, teacher_logits, input_ids):
    mask = _token_mask(input_ids)[:, 1:]
    student = student_logits[:, :-1].float()
    teacher = teacher_logits[:, :-1].float().detach()
    log_student = F.log_softmax(student, -1)
    teacher_probability = F.softmax(teacher, -1)
    kl = F.kl_div(log_student, teacher_probability, reduction="none").sum(-1)
    nll = F.cross_entropy(student.transpose(1, 2), input_ids[:, 1:], reduction="none")
    denominator = mask.sum().clamp_min(1)
    return (kl * mask).sum() / denominator, (nll * mask).sum() / denominator


def calibrate_loop_conditioned_task_parameters(
    calibration: "LoopConditionedTaskCalibration",
    student: nn.Module,
    teacher: nn.Module,
    cgb: LoopConditionedCGB,
    calibration_data: Sequence[torch.Tensor],
    device: str,
    objective: Callable = compute_task_transport_secant_loss,
) -> dict[str, Any]:
    """Calibrate CGB while all backbone, allocation, and teacher parameters stay frozen."""
    for model in (student, teacher):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    for parameter in cgb.parameters():
        parameter.requires_grad_(True)
    cgb.train()
    runtime = _LCTCRuntime(cgb)
    parameters = list(cgb.parameters())
    optimizer = torch.optim.Adam(parameters, lr=calibration.learning_rate, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(calibration.seed + 1201)
    probabilities = torch.tensor(calibration.alpha_probabilities)

    references_anchor, references_secant = [], []
    with torch.no_grad():
        for sample_id, sequence in enumerate(calibration_data[: min(16, len(calibration_data))]):
            input_ids = sequence.unsqueeze(0).to(device)
            loop_id = sample_id % calibration.num_loops
            alpha = calibration.alpha_values[
                int(torch.multinomial(probabilities, 1, generator=generator).item())
            ]
            student_entries, _, student_metadata = _run_trajectory(student, input_ids, runtime)
            teacher_entries, teacher_outputs, teacher_metadata = _run_trajectory(teacher, input_ids)
            teacher_logits = teacher.lm_head(teacher_outputs[-1])
            anchor, secant = objective(
                student, teacher, runtime, teacher_entries[loop_id],
                student_entries[loop_id] - teacher_entries[loop_id], loop_id, alpha,
                teacher_logits, student_metadata, teacher_metadata, _token_mask(input_ids),
            )
            references_anchor.append(float(anchor))
            if loop_id > 0:
                references_secant.append(float(secant))
    anchor_reference = max(sum(references_anchor) / len(references_anchor), 1e-8)
    secant_reference = max(sum(references_secant) / max(len(references_secant), 1), 1e-8)
    generator.manual_seed(calibration.seed + 1201)
    history = []
    for update in range(calibration.updates):
        sequence = calibration_data[update % len(calibration_data)]
        input_ids = sequence.unsqueeze(0).to(device)
        loop_id = update % calibration.num_loops
        alpha = calibration.alpha_values[
            int(torch.multinomial(probabilities, 1, generator=generator).item())
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_entries, teacher_outputs, teacher_metadata = _run_trajectory(teacher, input_ids)
            teacher_logits = teacher.lm_head(teacher_outputs[-1]).detach()
        student_entries, student_outputs, student_metadata = _run_trajectory(student, input_ids, runtime)
        student_logits = student.lm_head(student_outputs[-1])
        kl, nll = _distillation_terms(student_logits, teacher_logits, input_ids)
        anchor, secant = objective(
            student, teacher, runtime, teacher_entries[loop_id],
            (student_entries[loop_id] - teacher_entries[loop_id]).detach(), loop_id, alpha,
            teacher_logits, student_metadata, teacher_metadata, _token_mask(input_ids),
        )
        if loop_id == 0:
            secant = secant * 0.0
        loss = (
            kl + 0.1 * nll + 0.1 * anchor / anchor_reference
            + calibration.task_objective_weight * secant / secant_reference
            + 1e-4 * cgb.regularizer()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite LCTC loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, calibration.gradient_clip)
        optimizer.step()
        cgb.project_(calibration.clip_range, calibration.gain_range, calibration.beta_range)
        history.append(float(loss.detach()))
    cgb.eval()
    return {
        "updates": calibration.updates,
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "task_objective": objective.__name__,
        "cgb": cgb.summary(),
    }


def apply_lctc_calibration(model: nn.Module, cgb: LoopConditionedCGB) -> _LCTCRuntime:
    """Attach the frozen static CGB table to ordinary model forwards."""
    if hasattr(model, "lctc_cgb"):
        raise RuntimeError("LCTC calibration is already attached")
    model.lctc_cgb = cgb
    runtime = _LCTCRuntime(cgb)
    model._loopquant_lctc_runtime = runtime
    original_forward = model.forward

    # Preserve the HF forward signature for generate()'s model-kwargs validation.
    from functools import wraps

    @wraps(original_forward)
    def wrapped(*args: Any, **kwargs: Any):
        token = ACTIVE_LCTC_RUNTIME.set(model._loopquant_lctc_runtime)
        try:
            return original_forward(*args, **kwargs)
        finally:
            ACTIVE_LCTC_RUNTIME.reset(token)

    model.forward = wrapped
    return runtime


class LoopConditionedTaskCalibration:
    """Own initialization, objective-driven calibration, and static LCTC deployment."""

    def __init__(
        self,
        num_loops: int,
        *,
        group_size: int = 32,
        updates: int = 192,
        learning_rate: float = 1e-3,
        task_objective_weight: float = 0.03,
        alpha_values: tuple[float, ...] = (0.5, 1.0, 1.5),
        alpha_probabilities: tuple[float, ...] = (0.25, 0.50, 0.25),
        gradient_clip: float = 1.0,
        clip_range: tuple[float, float] = (0.5, 1.0),
        gain_range: tuple[float, float] = (0.8, 1.2),
        beta_range: tuple[float, float] = (-0.1, 0.1),
        seed: int = 0,
        objective: Callable = compute_task_transport_secant_loss,
    ):
        self.num_loops = int(num_loops)
        self.group_size = int(group_size)
        self.updates = int(updates)
        self.learning_rate = float(learning_rate)
        self.task_objective_weight = float(task_objective_weight)
        self.alpha_values = tuple(float(value) for value in alpha_values)
        self.alpha_probabilities = tuple(float(value) for value in alpha_probabilities)
        self.gradient_clip = float(gradient_clip)
        self.clip_range = clip_range
        self.gain_range = gain_range
        self.beta_range = beta_range
        self.seed = int(seed)
        self.objective = objective
        self.cgb: LoopConditionedCGB | None = None

    def initialize(
        self,
        student: nn.Module,
        teacher: nn.Module,
        calibration_data: Sequence[torch.Tensor],
        allocation: Sequence[int],
        device: str,
    ) -> LoopConditionedCGB:
        self.cgb = initialize_loop_conditioned_cgb(student, self.num_loops, self.group_size).to(device)
        samples = _collect_teacher_module_samples(teacher, student, calibration_data, device)
        grid = tuple(round(0.50 + 0.01 * index, 2) for index in range(51))
        _initialize_from_local_reconstruction(self.cgb, student, samples, allocation, grid)
        return self.cgb

    def calibrate(
        self,
        student: nn.Module,
        teacher: nn.Module,
        calibration_data: Sequence[torch.Tensor],
        device: str,
    ) -> dict[str, Any]:
        if self.cgb is None:
            raise RuntimeError("initialize LCTC before calibration")
        return calibrate_loop_conditioned_task_parameters(
            self, student, teacher, self.cgb, calibration_data, device, self.objective
        )

    def apply(self, model: nn.Module) -> _LCTCRuntime:
        if self.cgb is None:
            raise RuntimeError("initialize LCTC before deployment")
        return apply_lctc_calibration(model, self.cgb)
