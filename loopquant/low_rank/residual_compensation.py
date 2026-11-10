"""Fixed-target residual compensation using cached output moments.

The constrained residual update is inspired by GPTAQ and supports arbitrary
targets E, including E_t=Y_t-L_t Z_t. It does not use the paper's input-neuron residual
decomposition, which cannot be recovered from output cross moments alone.
"""

import torch

from ..grouping import GroupSize, effective_group_size, validate_input_width


def _weight_scale(group, qmax):
    """Same symmetric group MSE search as the existing GPTQ implementation."""
    base = group.abs().amax(1).clamp_min(1e-8) / qmax
    candidates = base[:, None] * torch.linspace(1.0, 0.8, 20, device=group.device)[None, :]
    error = ((group[:, None, :] / candidates[:, :, None]).round().clamp(-qmax, qmax)
             * candidates[:, :, None] - group[:, None, :]).square().mean(-1)
    return candidates.gather(1, error.argmin(1, keepdim=True)).squeeze(1)


@torch.no_grad()
def residual_compensation(weight, residual_cross, *, inverse_factor, bits=4,
                          group_size: GroupSize = 32, block_size=128):
    """Quantize W0 directly, then compensate remaining weights for a fixed E.

    For J(V)=sum_t lambda_t ||V Z_t-E_t||^2/N_t + rho||V-S||^2,
    the caller supplies C=sum_t lambda_t(E_t-W0 Z_t) Z_t'/N_t
    + rho(S-W0) and U=chol((H+rho I)^-1, upper=True). No extra damping is added.

    Fixing the first rounded column with increment d gives the exact OBS
    correction on remaining columns R:
        delta_R = C_R (H+rho I)_RR^-1 - d (H+rho I)_0R (H+rho I)_RR^-1.
    U_RR' U_RR equals that remaining inverse. The first term is computed in
    FP64; standard GPTQ feedback supplies the second. The remaining continuous
    gradient is then zero, so later steps only need rounding-error feedback.

    This is not a global discrete optimum. The first group uses W0's grid,
    rather than the unrestricted regression solution's grid. Weight work and
    feedback use FP32, matching GPTQ; no full unrestricted optimum is formed.
    """
    if weight.ndim != 2 or residual_cross.shape != weight.shape:
        raise ValueError("residual compensation weight and residual_cross must be matching matrices")
    width = weight.shape[1]
    validate_input_width(width, group_size, name="weight width")
    if inverse_factor.shape != (width, width):
        raise ValueError("residual compensation inverse_factor must match the input width")
    if block_size <= 0:
        raise ValueError("residual compensation block_size must be positive")
    if (not torch.isfinite(weight).all() or not torch.isfinite(residual_cross).all()
            or not torch.isfinite(inverse_factor).all() or (inverse_factor.diag() <= 0).any()):
        raise ValueError("residual compensation requires finite inputs and a positive inverse-factor diagonal")
    work = weight.detach().float().clone()
    inverse = inverse_factor.to(device=work.device, dtype=torch.float32)
    correction = None
    if width > 1:
        remaining = inverse_factor[1:, 1:].to(device=work.device, dtype=torch.float64)
        correction = ((residual_cross[:, 1:].to(remaining) @ remaining.T) @ remaining).float()
        if not torch.isfinite(correction).all():
            raise FloatingPointError("non-finite residual compensation")
    result = torch.empty_like(work)
    qmax = 2 ** (bits - 1) - 1
    resolved_group_size = effective_group_size(group_size, width)
    scale = None
    for start in range(0, width, block_size):
        stop = min(start + block_size, width)
        block = work[:, start:stop].clone()
        errors = torch.zeros_like(block)
        for local in range(stop - start):
            column = start + local
            if column % resolved_group_size == 0:
                group = work[:, column:column + resolved_group_size].clone()
                overlap = min(group.shape[1], stop - column)
                group[:, :overlap] = block[:, local:local + overlap]
                scale = _weight_scale(group, qmax)
            current = block[:, local]
            quantized = (current / scale).round().clamp(-qmax, qmax) * scale
            result[:, column] = quantized
            error = (current - quantized) / inverse[column, column]
            errors[:, local] = error
            block[:, local:] -= error[:, None] * inverse[column, column:stop][None, :]
            if column == 0 and correction is not None:
                block[:, 1:] += correction[:, :stop - 1]
                work[:, stop:] += correction[:, stop - 1:]
                correction = None
        work[:, stop:] -= errors @ inverse[start:stop, stop:]
    return result.to(weight.dtype)
