"""Fixed-grid discrete coordinate descent for either Q update method."""

import torch

from ..grouping import effective_group_size


@torch.no_grad()
def refine_quantized(weight, scales, hessian, residual_cross, *, reference, bits,
                     group_size, sweeps):
    """Minimize the centered quadratic using the actual saved-dtype grid.

    residual_cross is the negative half-gradient at reference and hessian
    already includes damping. No new grids or additional damping are fit.
    """
    result = weight.clone()
    delta = result.double() - reference.double()
    gradient = residual_cross.double() - delta @ hessian
    size = effective_group_size(group_size, result.shape[1])
    qmax = 2 ** (bits - 1) - 1
    accepted = torch.zeros((), device=result.device, dtype=torch.int64)
    for _ in range(sweeps):
        for column in range(result.shape[1]):
            diagonal = hessian[column, column]
            optimum = result[:, column].double() + gradient[:, column] / diagonal
            scale = scales[:, column // size].double()
            candidate = ((optimum / scale).round().clamp(-qmax, qmax) * scale).to(result.dtype)
            change = candidate.double() - result[:, column].double()
            difference = diagonal * change.square() - 2 * change * gradient[:, column]
            change = torch.where(difference <= 0, change, 0)
            accepted.add_(torch.count_nonzero(change))
            result[:, column] = (result[:, column].double() + change).to(result.dtype)
            gradient -= change[:, None] * hessian[column][None, :]
    return result, int(accepted.item())
