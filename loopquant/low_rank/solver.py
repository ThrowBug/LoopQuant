"""Alternating discrete regression and activation-weighted low-rank fitting."""

import math

import torch

from .quantization import gptq


def regularized_cholesky(hessian, damp_percent, *, logger=None, label=""):
    """Factor in FP64, with bounded, explicitly reported stability retries."""
    matrix = hessian.double()
    if not torch.isfinite(matrix).all():
        raise ValueError(f"non-finite calibration Gram: {label}, shape={tuple(matrix.shape)}")
    matrix = (matrix + matrix.T) * 0.5
    diagonal = matrix.diag()
    scale = float(diagonal.mean().clamp_min(1e-8))
    requested = float(damp_percent) * scale
    damping = requested
    # Never silently repair an arbitrary indefinite matrix with a huge ridge.
    ceiling = max(0.01, float(damp_percent)) * scale
    for _ in range(8):
        regularized = matrix.clone()
        regularized.diagonal().add_(damping)
        factor, info = torch.linalg.cholesky_ex(regularized)
        if not info.item():
            if damping > requested and logger is not None:
                logger.warning("[LowRank] %s Cholesky increased damping: requested=%.6g actual=%.6g",
                               label, requested, damping)
            return factor, damping
        next_damping = max(damping * 10, scale * 1e-12)
        if next_damping > ceiling * (1 + 1e-12):
            break
        damping = min(next_damping, ceiling)
    raise ValueError(
        f"calibration Gram is not positive definite after FP64 stability retries: {label}, "
        f"shape={tuple(matrix.shape)}, diag_min={float(diagonal.min()):.6g}, "
        f"diag_max={float(diagonal.max()):.6g}, requested_damping={requested:.6g}, "
        f"last_damping={damping:.6g}; check statistics or increase solver.damp_percent")


def continuous_weight(cross, factor):
    """Solve a regression system; callers define the reference/correction."""
    return torch.cholesky_solve(cross.T, factor).T


def svd_factors(matrix, rank):
    """Balanced factors of an ordinary rank-k truncated SVD."""
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    root = singular[:rank].clamp_min(0).sqrt()
    b = u[:, :rank] * root[None, :]
    return root[:, None] * vh[:rank], b


def low_rank_factors(cross, factor, rank):
    """Rank-k weighted solve; cross already includes any ridge reference."""
    whitened = torch.linalg.solve_triangular(factor, cross.T, upper=False).T
    right, b = svd_factors(whitened, rank)
    a = torch.linalg.solve_triangular(factor.T, right.T, upper=True).T
    return a, b


def reconstruction_loss(weight, a, b, statistics, loop_weights, *, reference):
    """Original MSE using FP64 moments of R0=Y-W0 Z, not large Y terms.

    All tokens contribute. This is the linear weight-space objective, separate
    from intermediate BF16 GEMM rounding and upstream student-state drift.
    """
    losses = []
    for loop in range(len(a)):
        delta = weight.double() - reference.double() + b[loop].double() @ a[loop].double()
        quadratic = ((delta @ statistics["hessian"][loop]) * delta).sum()
        cross = (delta * statistics["residual_cross"][loop]).sum()
        energy = statistics["residual_energy"][loop]
        error = quadratic - 2 * cross + energy
        tolerance = 128 * delta.shape[1] * torch.finfo(torch.float64).eps * (
            quadratic.abs() + 2 * cross.abs() + energy.abs())
        if error < -tolerance:
            raise FloatingPointError("negative reconstruction loss: inconsistent residual statistics")
        losses.append(error.clamp_min(0))
    values = torch.stack(losses)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite reconstruction loss")
    return float((values * loop_weights.double()).sum()), values.tolist()


@torch.no_grad()
def solve_linear(weight, statistics, *, rank, options, bits=4, group_size=32,
                 logger=None, label=""):
    """Fit one Q shared by every loop and independent rank-k corrections."""
    if options["weight_update"] != "regression_gptq":
        raise NotImplementedError("solver.weight_update=gptaq is reserved and not implemented")
    q_target = options.get("quantized_damping_target", "update")
    low_rank_target = options.get("low_rank_damping_target", "weight")
    if q_target not in ("weight", "update") or low_rank_target not in ("weight", "update"):
        raise ValueError("damping targets must be weight or update")
    device, dtype = weight.device, weight.dtype
    stats = {name: value.to(device=device, dtype=torch.float64) for name, value in statistics.items()}
    hessians, crosses = stats["hessian"], stats["residual_cross"]
    if not all(torch.isfinite(stats[key]).all() for key in ("residual_cross", "residual_energy")):
        raise ValueError(f"non-finite calibration residual statistics: {label}")
    original = weight.detach().double()
    loops = len(hessians)
    weights = torch.tensor(options["loop_weights"] or [1.0] * loops, device=device, dtype=torch.float64)
    weights /= weights.sum()
    shared_hessian = (hessians * weights[:, None, None]).sum(0)
    shared_cross = (crosses * weights[:, None, None]).sum(0)
    shared_factor, q_damping = regularized_cholesky(
        shared_hessian, options["damp_percent"], logger=logger, label=f"{label} shared Q")
    inverse_factor = torch.linalg.cholesky(torch.cholesky_inverse(shared_factor), upper=True).float()
    factors, l_damping = zip(*(regularized_cholesky(
        h, options["damp_percent"], logger=logger, label=f"{label} loop={loop + 1}")
        for loop, h in enumerate(hessians)))

    # A feasible weight-only baseline participates in selection too. It uses
    # the same Z Gram and clip as the alternating candidates (not the old X Gram).
    baseline_q = gptq(original, shared_hessian, bits=bits, group_size=group_size,
                      block_size=options["block_size"], inverse_factor=inverse_factor).to(dtype)
    baseline_a, baseline_b = svd_factors(original - baseline_q.double(), rank)
    baseline_a = [baseline_a.to(dtype) for _ in range(loops)]
    baseline_b = [baseline_b.to(dtype) for _ in range(loops)]
    baseline_loss, _ = reconstruction_loss(
        baseline_q, baseline_a, baseline_b, stats, weights, reference=original)
    best = baseline_q, baseline_a, baseline_b, baseline_loss, 0
    if logger is not None:
        logger.info("[LowRank] %s weight-only baseline loss=%.6g quantized_damping_target=%s low_rank_damping_target=%s",
                    label, baseline_loss, q_target, low_rank_target)

    # The first alternating round still starts with L_t=0 and FP W as reference.
    a = [torch.zeros(rank, weight.shape[1], device=device, dtype=dtype) for _ in range(loops)]
    b = [torch.zeros(weight.shape[0], rank, device=device, dtype=dtype) for _ in range(loops)]
    reference, history, previous_loss = original, [], None
    for iteration in range(1, options["iterations"] + 1):
        q_reference = reference if q_target == "update" else torch.zeros_like(original)
        adjusted = shared_cross - (q_reference - original) @ shared_hessian
        for loop in range(loops):
            adjusted.addmm_(b[loop].double(), a[loop].double() @ hessians[loop], alpha=-float(weights[loop]))
        # A zero reference penalizes Q itself; otherwise penalize its update.
        continuous = q_reference + continuous_weight(adjusted, shared_factor)
        # Match the final saved base dtype before computing its low-rank residual.
        candidate = gptq(continuous, shared_hessian, bits=bits, group_size=group_size,
                         block_size=options["block_size"], inverse_factor=inverse_factor).to(dtype)
        next_a, next_b = [], []
        for loop in range(loops):
            residual_cross = crosses[loop] - (candidate.double() - original) @ hessians[loop]
            if low_rank_target == "update":
                # Constrain the full new L to rank k, not just its increment.
                previous = b[loop].double() @ a[loop].double()
                residual_cross = residual_cross + l_damping[loop] * previous
            current_a, current_b = low_rank_factors(residual_cross, factors[loop], rank)
            next_a.append(current_a.to(dtype))
            next_b.append(current_b.to(dtype))
        loss, per_loop = reconstruction_loss(candidate, next_a, next_b, stats, weights, reference=original)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite low_rank objective: {label}")
        accepted = previous_loss is None or loss < previous_loss
        selected = loss < best[3]
        record = {"iteration": iteration, "loss": loss, "per_loop_loss": per_loop,
                  "accepted": accepted, "selected": selected}
        if iteration == 1:
            record["base_loss"], _ = reconstruction_loss(candidate, a, b, stats, weights, reference=original)
        history.append(record)
        if logger is not None:
            logger.info("[LowRank] %s iteration=%d/%d loss=%.6g accepted=%s selected=%s", label,
                        iteration, options["iterations"], loss, accepted, selected)
        if selected:
            best = candidate, next_a, next_b, loss, iteration
        if not accepted or loss == 0:
            break
        a, b = next_a, next_b
        reference = candidate.double()
        previous_loss = loss
    q, a, b, loss, iteration = best
    report = {"best_iteration": iteration, "final_loss": loss, "base_loss": history[0]["base_loss"],
              "baseline_loss": baseline_loss, "loss_method": "fp64_reference_residual_moments",
              "quantized_damping_target": q_target, "low_rank_damping_target": low_rank_target,
              "q_damping": q_damping, "low_rank_damping": list(l_damping), "iterations": history}
    return q, a, b, report
