"""Standalone symmetric fake quantization and GPTQ weight updates."""

import torch

from ..grouping import GroupSize, effective_group_size, validate_input_width


def activation_bits(config):
    """Use one activation precision for every projection and loop."""
    return {"w4a4": 4, "w4a8": 8}[config.quantization["activation_setting"].lower()]


def quantize_activation(value, bits, group_size: GroupSize = 32, clip=1.0):
    if bits >= 16:
        return value
    width = value.shape[-1]
    validate_input_width(width, group_size, name="activation width")
    resolved = effective_group_size(group_size, width)
    grouped = value.float().reshape(*value.shape[:-1], -1, resolved)
    maximum = 2 ** (int(bits) - 1) - 1
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-8) * clip / maximum
    result = (grouped / scale).round().clamp(-maximum, maximum) * scale
    return result.reshape_as(value).to(value.dtype)


def fit_clip(value, bits, group_size: GroupSize, ratios):
    return torch.stack([(quantize_activation(value, bits, group_size, ratio).float()
                         - value.float()).square().sum() for ratio in ratios])


@torch.no_grad()
def gptq(weight, hessian, *, bits=4, group_size: GroupSize = 32, block_size=128, damp_percent=0.001,
         inverse_factor=None):
    """Quantize one shared matrix; caller defines the multi-loop Hessian."""
    if weight.ndim != 2:
        raise ValueError("gptq weight must be a matrix")
    validate_input_width(weight.shape[1], group_size, name="weight width")
    work = weight.detach().float().clone()
    if inverse_factor is None:
        hessian = hessian.to(device=work.device, dtype=torch.float32).clone()
        dead = hessian.diag() == 0
        work[:, dead] = 0
        hessian[dead, dead] = 1
        diagonal = torch.arange(work.shape[1], device=work.device)
        hessian[diagonal, diagonal] += damp_percent * hessian.diag().mean()
        inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)
    else:
        # The alternating solver reuses the already regularized factor. Do not
        # add damping again or erase columns using a different dead-channel rule.
        inverse = inverse_factor
    result = torch.empty_like(work)
    qmax = 2 ** (bits - 1) - 1
    resolved_group_size = effective_group_size(group_size, work.shape[1])
    scale = None
    for start in range(0, work.shape[1], block_size):
        stop = min(start + block_size, work.shape[1])
        block = work[:, start:stop].clone()
        errors = torch.zeros_like(block)
        for local in range(stop - start):
            column = start + local
            if column % resolved_group_size == 0:
                # Include within-block error feedback in group parameter fitting.
                group = work[:, column:column + resolved_group_size].clone()
                overlap = min(group.shape[1], stop - column)
                group[:, :overlap] = block[:, local:local + overlap]
                base = group.abs().amax(1).clamp_min(1e-8) / qmax
                candidates = base[:, None] * torch.linspace(1.0, 0.8, 20, device=work.device)[None, :]
                error = ((group[:, None, :] / candidates[:, :, None]).round().clamp(-qmax, qmax)
                         * candidates[:, :, None] - group[:, None, :]).square().mean(-1)
                scale = candidates.gather(1, error.argmin(1, keepdim=True)).squeeze(1)
            current = block[:, local]
            quantized = (current / scale).round().clamp(-qmax, qmax) * scale
            result[:, column] = quantized
            error = (current - quantized) / inverse[column, column]
            errors[:, local] = error
            block[:, local:] -= error[:, None] * inverse[column, column:stop][None, :]
        work[:, stop:] -= errors @ inverse[start:stop, stop:]
    return result.to(weight.dtype)
