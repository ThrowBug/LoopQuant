"""Alternating discrete regression and activation-weighted low-rank fitting."""

import math

import torch

from ..grouping import GroupSize, validate_input_width
from .quantization import gptq
from .residual_compensation import residual_compensation
from .refinement import refine_quantized
from .rotation import inverse_block_hadamard_transform


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


def svd_factors(matrix, rank, *, logger=None, label="", diagnostics=None):
    """Balanced factors of an ordinary rank-k truncated SVD."""
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    if diagnostics is not None:
        call_index = diagnostics["calls"]
        diagnostics["calls"] += 1
        max_calls = diagnostics["max_calls_per_linear"]
        if diagnostics["enabled"] and logger is not None and (max_calls == 0 or call_index < max_calls):
            top_k = min(singular.numel(), max(1, math.ceil(rank * diagnostics["top_multiplier"])))
            values = ", ".join(f"{float(value):.6g}" for value in singular[:top_k].detach().cpu())
            logger.info("[LowRank] %s SVD call=%d rank=%d top%d singular_values=[%s]",
                        label, call_index + 1, rank, top_k, values)
    root = singular[:rank].clamp_min(0).sqrt()
    b = u[:, :rank] * root[None, :]
    return root[:, None] * vh[:rank], b


def low_rank_factors(cross, factor, rank, *, logger=None, label="", diagnostics=None, whitened=None):
    """Minimize tr(L H L') - 2<L,C> subject to rank(L)<=rank.

    H=G G', M=C G^{-T}, and L=[M]_rank G^{-1}. The caller incorporates
    any ridge/reference in H and C; truncating C H^{-1} directly is different.
    """
    if whitened is None:
        whitened = torch.linalg.solve_triangular(factor, cross.T, upper=False).T
    right, b = svd_factors(whitened, rank, logger=logger, label=label, diagnostics=diagnostics)
    a = torch.linalg.solve_triangular(factor.T, right.T, upper=True).T
    return a, b


def _loop_factors(a, b, loop, sharing):
    """FP64 factors of the actual correction used by one recurrent loop."""
    if sharing == "independent":
        return a[loop].double(), b[loop].double()
    return (torch.cat([value.double() for value in a[:loop + 1]], 0),
            torch.cat([value.double() for value in b[:loop + 1]], 1))


def _prefix_factors(hessians, weights, damp_percent, *, logger=None, label=""):
    """Factor suffix Grams, preserving a consistent ridge on each complete L_t.

    If a suffix needs a stability increase, assign the extra ridge to its first
    loop. Earlier suffixes then include it; already factored later suffixes do
    not change. Thus every coordinate solves the same regularized objective.
    """
    loops = len(hessians)
    factors, suffix_damping = [None] * loops, [0.0] * loops
    damping = [float(damp_percent) * float(h.diag().mean().clamp_min(1e-8)) for h in hessians]
    suffix_hessian = torch.zeros_like(hessians[0])
    later_ridge = 0.0
    for group in reversed(range(loops)):
        coefficient = float(weights[group])
        suffix_hessian.add_(hessians[group], alpha=coefficient)
        requested = later_ridge + coefficient * damping[group]
        scale = float(suffix_hessian.diag().mean().clamp_min(1e-8))
        factor, actual = regularized_cholesky(
            suffix_hessian, requested / scale, logger=logger, label=f"{label} prefix_group={group + 1}")
        # Preserve a tiny loop's nominal ridge even if its weighted contribution
        # rounds away when added to a much larger later suffix.
        damping[group] += max(0.0, actual - requested) / coefficient
        factors[group], suffix_damping[group] = factor, actual
        later_ridge = actual
    return factors, damping, suffix_damping


def _rank_objective(a, b, cross, factor, *, whitened=False):
    """Variable part of the coordinate objective, evaluated at stored precision."""
    a, b = a.double(), b.double()
    transformed = a @ factor
    quadratic = ((b.T @ b) * (transformed @ transformed.T)).sum()
    right = transformed if whitened else a
    return float(quadratic - 2 * ((b.T @ cross) * right).sum())


def _low_rank_statistics(statistics):
    """Missing mixed fields denote legacy single-input solver callers."""
    fields = ("low_rank_hessian", "input_cross", "low_rank_cross")
    present = [name in statistics for name in fields]
    if any(present) and not all(present):
        raise ValueError("incomplete dual-input statistics")
    if all(present):
        return tuple(statistics[name] for name in fields)
    return statistics["hessian"], statistics["hessian"], statistics["residual_cross"]


def _whitening_plan(statistics, weights, factors, damping, sharing):
    """Cache fixed whitened targets and couplings once per physical linear."""
    h, xz, cross = _low_rank_statistics(statistics)
    plan = []
    suffix_h, suffix_ridge = [], []
    if sharing == "prefix":
        for i in range(len(h)):
            suffix_h.append(sum(float(weights[t]) * h[t] for t in range(i, len(h))))
            suffix_ridge.append(sum(float(weights[t]) * damping[t] for t in range(i, len(h))))
    for i, g in enumerate(factors):
        whiten = lambda matrix: torch.linalg.solve_triangular(g, matrix.T, upper=False).T
        if sharing == "independent":
            plan.append((whiten(cross[i]), whiten(xz[i].T), None))
        else:
            base = sum(float(weights[t]) * cross[t] for t in range(i, len(h)))
            mixed = sum(float(weights[t]) * xz[t].T for t in range(i, len(h)))
            kernels = {}
            for j in range(len(h)):
                if j == i:
                    continue
                if j < i:
                    kernels[j] = g  # (H_i + ridge_i I) G_i^-T = G_i
                else:
                    matrix = suffix_h[j].clone()
                    matrix.diagonal().add_(suffix_ridge[j])
                    kernels[j] = whiten(matrix)
            plan.append((whiten(base), whiten(mixed), kernels))
    return plan


def _update_prefix(weight, original, a, b, statistics, weights, factors, damping, target,
                   *, logger=None, label="", diagnostics=None, whitening=None):
    """One forward coordinate pass; outer-iteration damping references stay fixed."""
    hessians, xz, crosses = _low_rank_statistics(statistics)
    delta = weight.double() - original
    cached = whitening is not None and "low_rank_hessian" in statistics
    base_crosses = None if cached else [crosses[t] - delta @ xz[t].T for t in range(len(a))]
    next_a, next_b, updates = list(a), list(b), []
    rank = a[0].shape[0]
    for group in range(len(a)):
        cross = None
        if not cached:
            cross = torch.zeros_like(base_crosses[0])
            for loop in range(group, len(a)):
                coefficient, rho = float(weights[loop]), damping[loop]
                cross.add_(base_crosses[loop], alpha=coefficient)
                for other in range(loop + 1):
                    if other != group:
                        right = next_a[other].double()
                        cross.addmm_(next_b[other].double(), right @ hessians[loop] + rho * right,
                                     alpha=-coefficient)
                if target == "update":
                    reference_a, reference_b = _loop_factors(a, b, loop, "prefix")
                    cross.addmm_(reference_b, reference_a, alpha=coefficient * rho)
        whitened = None
        if whitening is not None:
            base, mixed, kernels = whitening[group]
            whitened = base - delta @ mixed
            for other, kernel in kernels.items():
                whitened -= next_b[other].double() @ (next_a[other].double() @ kernel)
            if target == "update":
                for other in range(len(a)):
                    coefficient = sum(float(weights[t]) * damping[t]
                                      for t in range(max(group, other), len(a)))
                    right = torch.linalg.solve_triangular(
                        factors[group], a[other].double().T, upper=False).T
                    whitened.addmm_(b[other].double(), right, alpha=coefficient)
        if cached:
            cross = whitened
        current_a, current_b = low_rank_factors(
            cross, factors[group], rank, logger=logger,
            label=f"{label} prefix_group={group + 1}", diagnostics=diagnostics, whitened=whitened)
        current_a, current_b = current_a.to(a[group].dtype), current_b.to(b[group].dtype)
        before = _rank_objective(next_a[group], next_b[group], cross, factors[group], whitened=cached)
        after = _rank_objective(current_a, current_b, cross, factors[group], whitened=cached)
        if not math.isfinite(before) or not math.isfinite(after):
            raise FloatingPointError(f"non-finite prefix coordinate objective: {label} group={group + 1}")
        # Casting factors to BF16 can undo a small mathematical improvement.
        accepted = after <= before
        if accepted:
            next_a[group], next_b[group] = current_a, current_b
        updates.append({"group": group + 1, "accepted": accepted,
                        "objective_change": after - before if accepted else 0.0})
        if logger is not None:
            logger.info("[LowRank] %s prefix_group=%d/%d accepted=%s", label, group + 1, len(a), accepted)
    return next_a, next_b, updates


def reconstruction_loss(weight, a, b, statistics, loop_weights, *, reference, sharing="independent"):
    """Original MSE using FP64 moments of R0=Y-W0 Z, not large Y terms.

    All tokens contribute. This is the linear weight-space objective, separate
    from intermediate BF16 GEMM rounding and upstream student-state drift.
    """
    losses = []
    hx, xz, kx = _low_rank_statistics(statistics)
    for loop in range(len(statistics["hessian"])):
        delta = weight.double() - reference.double()
        quadratic = ((delta @ statistics["hessian"][loop]) * delta).sum()
        cross = (delta * statistics["residual_cross"][loop]).sum()
        magnitude = quadratic.abs() + 2 * cross.abs()
        if a:
            loop_a, loop_b = _loop_factors(a, b, loop, sharing)
            low_rank = loop_b @ loop_a
            low_quadratic = ((low_rank @ hx[loop]) * low_rank).sum()
            interaction = ((low_rank @ xz[loop]) * delta).sum()
            low_cross = (low_rank * kx[loop]).sum()
            quadratic = quadratic + low_quadratic + 2 * interaction
            cross = cross + low_cross
            magnitude += low_quadratic.abs() + 2 * interaction.abs() + 2 * low_cross.abs()
        energy = statistics["residual_energy"][loop]
        error = quadratic - 2 * cross + energy
        tolerance = 128 * delta.shape[1] * torch.finfo(torch.float64).eps * (
            magnitude + energy.abs())
        if error < -tolerance:
            raise FloatingPointError("negative reconstruction loss: inconsistent residual statistics")
        losses.append(error.clamp_min(0))
    values = torch.stack(losses)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite reconstruction loss")
    return float((values * loop_weights.double()).sum()), values.tolist()


@torch.no_grad()
def solve_linear(weight, statistics, *, rank, options, bits=4, group_size: GroupSize = 32,
                 sharing="independent", logger=None, label="", rotation=None):
    """Fit shared Q with independent corrections or cumulative rank-k increments."""
    if sharing not in ("independent", "prefix"):
        raise ValueError("sharing must be independent or prefix")
    if options.get("initialization", "quantized_first") != "quantized_first":
        raise ValueError("solver.initialization supports quantized_first only; low_rank_first is not implemented")
    sweeps = options.get("coordinate_sweeps", 0)
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
        raise ValueError("solver.coordinate_sweeps must be a nonnegative integer")
    weight_update = options["weight_update"]
    if weight_update not in ("regression_gptq", "residual_compensation"):
        raise ValueError("solver.weight_update must be regression_gptq or residual_compensation")
    q_target = options.get("quantized_damping_target", "update")
    low_rank_target = options.get("low_rank_damping_target", "update")
    if q_target not in ("weight", "update") or low_rank_target not in ("weight", "update"):
        raise ValueError("damping targets must be weight or update")
    diagnostic_options = options.get("svd_diagnostics", {})
    if not isinstance(diagnostic_options, dict):
        raise ValueError("solver.svd_diagnostics must be a mapping")
    svd_diagnostics = {
        "enabled": bool(diagnostic_options.get("enabled", False)),
        "top_multiplier": float(diagnostic_options.get("top_multiplier", 1.2)),
        "max_calls_per_linear": int(diagnostic_options.get("max_calls_per_linear", 1)),
        "calls": 0,
    }
    if weight.ndim != 2:
        raise ValueError("solve_linear weight must be a matrix")
    validate_input_width(weight.shape[1], group_size, name=f"{label or 'linear'} weight width")
    device, dtype = weight.device, weight.dtype
    stats = {name: value.to(device=device, dtype=torch.float64) for name, value in statistics.items()}
    hessians, crosses = stats["hessian"], stats["residual_cross"]
    low_hessians, input_crosses, low_crosses = _low_rank_statistics(stats)
    if not all(torch.isfinite(value).all() for value in (low_hessians, input_crosses, low_crosses)):
        raise ValueError("non-finite dual-input calibration statistics")
    if not all(torch.isfinite(stats[key]).all() for key in ("residual_cross", "residual_energy")):
        raise ValueError(f"non-finite calibration residual statistics: {label}")
    original = weight.detach().double()
    loops = len(hessians)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    active_rank = rank * loops if sharing == "prefix" else rank
    if active_rank > min(weight.shape):
        raise ValueError("active low-rank schedule exceeds the linear dimensions")
    weights = torch.tensor(options["loop_weights"] or [1.0] * loops, device=device, dtype=torch.float64)
    weights /= weights.sum()
    shared_hessian = (hessians * weights[:, None, None]).sum(0)
    shared_cross = (crosses * weights[:, None, None]).sum(0)
    shared_factor, q_damping = regularized_cholesky(
        shared_hessian, options["damp_percent"], logger=logger, label=f"{label} shared Q")
    regularized_hessian = shared_hessian.clone()
    regularized_hessian.diagonal().add_(q_damping)
    inverse_factor64 = torch.linalg.cholesky(torch.cholesky_inverse(shared_factor), upper=True)
    inverse_factor = inverse_factor64.float()
    if weight_update != "residual_compensation":
        inverse_factor64 = None
    if rank == 0:
        factors, l_damping, suffix_damping = [], [], []
    elif sharing == "prefix":
        factors, l_damping, suffix_damping = _prefix_factors(
            low_hessians, weights, options["damp_percent"], logger=logger, label=label)
    else:
        factors, l_damping = zip(*(regularized_cholesky(
            h, options["damp_percent"], logger=logger, label=f"{label} loop={loop + 1}")
            for loop, h in enumerate(low_hessians)))
    whitening = _whitening_plan(stats, weights, factors, l_damping, sharing) if rank else []

    # GPTQ (plus weight-error SVD when enabled) participates in selection too. It uses
    # the same Z Gram and clip as the alternating candidates (not the old X Gram).
    baseline_q = gptq(original, shared_hessian, bits=bits, group_size=group_size,
                      block_size=options["block_size"], inverse_factor=inverse_factor).to(dtype)
    if rank == 0:
        baseline_a, baseline_b = [], []
    else:
        # The baseline SVD is only a comparison candidate. Diagnostics are
        # reserved for the weighted SVDs used by the actual low-rank updates.
        baseline_residual = original - baseline_q.double()
        if rotation and rotation.get("enabled", False):
            baseline_residual = inverse_block_hadamard_transform(
                baseline_residual, rotation["block_size"], rotation.get("seed", 42))
        baseline_a, baseline_b = svd_factors(baseline_residual, active_rank)
        if sharing == "prefix":
            baseline_a = [value.to(dtype) for value in baseline_a.split(rank, 0)]
            baseline_b = [value.to(dtype) for value in baseline_b.split(rank, 1)]
        else:
            baseline_a = [baseline_a.to(dtype) for _ in range(loops)]
            baseline_b = [baseline_b.to(dtype) for _ in range(loops)]
    baseline_loss, _ = reconstruction_loss(
        baseline_q, baseline_a, baseline_b, stats, weights, reference=original, sharing=sharing)
    best = baseline_q, baseline_a, baseline_b, baseline_loss, 0
    if logger is not None:
        logger.info("[LowRank] %s %s baseline loss=%.6g weight_update=%s sharing=%s "
                    "quantized_damping_target=%s low_rank_damping_target=%s",
                    label, "weight-only" if rank else "GPTQ-only", baseline_loss,
                    weight_update, sharing, q_target, low_rank_target)

    # The first alternating round still starts with L_t=0 and FP W as reference.
    banks = loops if rank else 0
    a = [torch.zeros(rank, weight.shape[1], device=device, dtype=dtype) for _ in range(banks)]
    b = [torch.zeros(weight.shape[0], rank, device=device, dtype=dtype) for _ in range(banks)]
    reference, history, previous_loss = original, [], None
    for iteration in range(1, options["iterations"] + 1):
        q_reference = reference if q_target == "update" else torch.zeros_like(original)
        if weight_update == "regression_gptq":
            adjusted = shared_cross - (q_reference - original) @ shared_hessian
        else:
            # Negative half-gradient at W0, including the chosen ridge reference.
            # Keep it centered; never reconstruct a large Y Z' cross moment.
            adjusted = shared_cross + q_damping * (q_reference - original)
        if rank:
            for loop in range(loops):
                loop_a, loop_b = _loop_factors(a, b, loop, sharing)
                adjusted.addmm_(loop_b, loop_a @ input_crosses[loop], alpha=-float(weights[loop]))
        if weight_update == "regression_gptq":
            continuous = q_reference + continuous_weight(adjusted, shared_factor)
            candidate = gptq(continuous, shared_hessian, bits=bits, group_size=group_size,
                             block_size=options["block_size"], inverse_factor=inverse_factor,
                             **({"return_scales": True} if sweeps else {}))
        else:
            candidate = residual_compensation(original, adjusted, bits=bits, group_size=group_size,
                                               block_size=options["block_size"], inverse_factor=inverse_factor64,
                                               **({"return_scales": True} if sweeps else {}))
        refined = 0
        unrefined = candidate[0] if sweeps else candidate
        q_unrefined_loss, _ = reconstruction_loss(
            unrefined.to(dtype), a, b, stats, weights, reference=original, sharing=sharing)
        if sweeps:
            candidate, scales = candidate
            if weight_update == "regression_gptq":
                residual_gradient = adjusted + (q_reference - original) @ regularized_hessian
            else:
                residual_gradient = adjusted
            candidate, refined = refine_quantized(
                candidate.to(dtype), scales, regularized_hessian,
                residual_gradient, reference=original, bits=bits, group_size=group_size, sweeps=sweeps)
        # Match the final saved base dtype before computing its low-rank residual.
        candidate = candidate.to(dtype)
        q_loss, _ = reconstruction_loss(candidate, a, b, stats, weights, reference=original, sharing=sharing)
        if rank == 0:
            next_a, next_b, prefix_updates = [], [], []
        elif sharing == "prefix":
            next_a, next_b, prefix_updates = _update_prefix(
                candidate, original, a, b, stats, weights, factors, l_damping, low_rank_target,
                logger=logger, label=f"{label} iteration={iteration}", diagnostics=svd_diagnostics,
                whitening=whitening)
        else:
            next_a, next_b = [], []
            for loop in range(loops):
                delta = candidate.double() - original
                base, mixed, _ = whitening[loop]
                white = base - delta @ mixed
                cached = "low_rank_hessian" in stats
                residual_cross = None if cached else low_crosses[loop] - delta @ input_crosses[loop].T
                if low_rank_target == "update":
                    # Constrain the full new L to rank k, not just its increment.
                    if not cached:
                        previous = b[loop].double() @ a[loop].double()
                        residual_cross = residual_cross + l_damping[loop] * previous
                    white += l_damping[loop] * b[loop].double() @ torch.linalg.solve_triangular(
                        factors[loop], a[loop].double().T, upper=False).T
                if cached:
                    residual_cross = white
                current_a, current_b = low_rank_factors(
                    residual_cross, factors[loop], rank, logger=logger,
                    label=f"{label} iteration={iteration} loop={loop + 1}",
                    diagnostics=svd_diagnostics, whitened=white)
                current_a, current_b = current_a.to(dtype), current_b.to(dtype)
                if _rank_objective(current_a, current_b, residual_cross, factors[loop], whitened=cached) > _rank_objective(
                        a[loop], b[loop], residual_cross, factors[loop], whitened=cached):
                    current_a, current_b = a[loop], b[loop]
                next_a.append(current_a)
                next_b.append(current_b)
        loss, per_loop = reconstruction_loss(
            candidate, next_a, next_b, stats, weights, reference=original, sharing=sharing)
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite low_rank objective: {label}")
        accepted = previous_loss is None or loss < previous_loss
        selected = loss < best[3]
        record = {"iteration": iteration, "loss": loss, "per_loop_loss": per_loop,
                  "accepted": accepted, "selected": selected, "q_loss": q_loss,
                  "q_unrefined_loss": q_unrefined_loss, "previous_loss": previous_loss,
                  "refined_weights": refined}
        if sharing == "prefix":
            record["prefix_updates"] = prefix_updates
        if iteration == 1:
            record["base_loss"], _ = reconstruction_loss(
                candidate, a, b, stats, weights, reference=original, sharing=sharing)
        history.append(record)
        if logger is not None:
            logger.info("[LowRank] %s iteration=%d/%d q_loss=%.6g refined_q_loss=%.6g "
                        "loss=%.6g accepted=%s selected=%s", label,
                        iteration, options["iterations"], q_unrefined_loss, q_loss, loss, accepted, selected)
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
              "weight_update": weight_update,
              "weight_update_implementation": ("continuous_regression_gptq" if weight_update == "regression_gptq"
                                                else weight_update),
              "selected_candidate": "baseline" if iteration == 0 else weight_update,
              "low_rank_enabled": rank > 0,
              "sharing": sharing,
              "rank_schedule": [(t + 1) * rank if sharing == "prefix" else rank for t in range(loops)],
              "quantized_damping_target": q_target, "low_rank_damping_target": low_rank_target,
              "q_damping": q_damping, "low_rank_damping": list(l_damping), "iterations": history}
    report.update(initialization="quantized_first", coordinate_sweeps=sweeps)
    if sharing == "prefix":
        report["prefix_group_damping"] = suffix_damping
    return q, a, b, report
