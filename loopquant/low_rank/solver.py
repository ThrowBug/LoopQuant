"""Alternating discrete regression and activation-weighted low-rank fitting."""

import torch

from .quantization import gptq


def regularized_cholesky(hessian, damp_percent):
    """Relative ridge; a numerical floor also defines all-zero input Grams."""
    matrix = (hessian + hessian.T) * 0.5
    damping = damp_percent * matrix.diag().mean().clamp_min(1e-8)
    matrix = matrix.clone()
    matrix.diagonal().add_(damping)
    factor, info = torch.linalg.cholesky_ex(matrix)
    if info.item():
        raise ValueError("calibration Gram is not positive definite after damping; increase solver.damp_percent")
    return factor, float(damping)


def continuous_weight(cross, factor):
    """Solve W(H+rho I)=D without explicitly forming an inverse."""
    return torch.cholesky_solve(cross.T, factor).T


def low_rank_factors(cross, factor, rank):
    """Exact rank-k optimum of ||LZ-R||²/N + rho||L||²."""
    whitened = torch.linalg.solve_triangular(factor, cross.T, upper=False).T
    u, singular, vh = torch.linalg.svd(whitened, full_matrices=False)
    root = singular[:rank].clamp_min(0).sqrt()
    b = u[:, :rank] * root[None, :]
    right = root[:, None] * vh[:rank]
    a = torch.linalg.solve_triangular(factor.T, right.T, upper=True).T
    return a, b


def reconstruction_loss(weight, a, b, statistics, loop_weights):
    """Original unregularized MSE from sufficient statistics, in weight space.

    Scalars are reduced in FP64 to limit cancellation. This measures the stated
    linear objective; intermediate BF16 GEMM rounding at inference is separate.
    """
    losses = []
    for loop in range(len(a)):
        effective = weight.float() + b[loop].float() @ a[loop].float()
        quadratic = ((effective @ statistics["hessian"][loop]) * effective).sum(dtype=torch.float64)
        cross = (effective * statistics["cross"][loop]).sum(dtype=torch.float64)
        losses.append((quadratic - 2 * cross + statistics["target_energy"][loop]).clamp_min(0))
    values = torch.stack(losses)
    return float((values * loop_weights.double()).sum()), values.tolist()


@torch.no_grad()
def solve_linear(weight, statistics, *, rank, options, bits=4, group_size=32,
                 logger=None, label=""):
    """Fit one Q shared by every loop and independent rank-k corrections."""
    if options["weight_update"] != "regression_gptq":
        raise NotImplementedError("solver.weight_update=gptaq is reserved and not implemented")
    device, dtype = weight.device, weight.dtype
    stats = {name: value.to(device) for name, value in statistics.items()}
    hessians, crosses = stats["hessian"], stats["cross"]
    loops = len(hessians)
    weights = torch.tensor(options["loop_weights"] or [1.0] * loops, device=device, dtype=torch.float64)
    weights /= weights.sum()
    shared_hessian = (hessians * weights.float()[:, None, None]).sum(0)
    shared_cross = (crosses * weights.float()[:, None, None]).sum(0)
    shared_factor, q_damping = regularized_cholesky(shared_hessian, options["damp_percent"])
    inverse_factor = torch.linalg.cholesky(torch.cholesky_inverse(shared_factor), upper=True)
    factors, l_damping = zip(*(regularized_cholesky(h, options["damp_percent"]) for h in hessians))
    a = [torch.zeros(rank, weight.shape[1], device=device, dtype=dtype) for _ in range(loops)]
    b = [torch.zeros(weight.shape[0], rank, device=device, dtype=dtype) for _ in range(loops)]
    best, history = None, []
    for iteration in range(1, options["iterations"] + 1):
        adjusted = shared_cross.clone()
        for loop in range(loops):
            adjusted.addmm_(b[loop].float(), a[loop].float() @ hessians[loop], alpha=-float(weights[loop]))
        continuous = continuous_weight(adjusted, shared_factor)
        # Match the final saved base dtype before computing its low-rank residual.
        candidate = gptq(continuous, shared_hessian, bits=bits, group_size=group_size,
                         block_size=options["block_size"], inverse_factor=inverse_factor).to(dtype)
        next_a, next_b = [], []
        for loop in range(loops):
            residual_cross = crosses[loop] - candidate.float() @ hessians[loop]
            current_a, current_b = low_rank_factors(residual_cross, factors[loop], rank)
            next_a.append(current_a.to(dtype))
            next_b.append(current_b.to(dtype))
        loss, per_loop = reconstruction_loss(candidate, next_a, next_b, stats, weights)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError(f"non-finite low_rank objective: {label}")
        accepted = best is None or loss < best[3]
        record = {"iteration": iteration, "loss": loss, "per_loop_loss": per_loop, "accepted": accepted}
        if best is None:
            record["base_loss"], _ = reconstruction_loss(candidate, a, b, stats, weights)
        history.append(record)
        if logger is not None:
            logger.info("[LowRank] %s iteration=%d/%d loss=%.6g accepted=%s", label,
                        iteration, options["iterations"], loss, accepted)
        if not accepted:
            break
        a, b = next_a, next_b
        best = candidate, a, b, loss, iteration
    q, a, b, loss, iteration = best
    report = {"best_iteration": iteration, "final_loss": loss, "base_loss": history[0]["base_loss"],
              "q_damping": q_damping, "low_rank_damping": list(l_damping), "iterations": history}
    return q, a, b, report
