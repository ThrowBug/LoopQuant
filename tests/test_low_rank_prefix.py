"""Token-space oracles for one-pass shared-prefix coordinate fitting."""

import pytest
import torch

from loopquant.low_rank.solver import (
    _prefix_factors, _update_prefix, reconstruction_loss, solve_linear,
)


def problem():
    torch.manual_seed(179)
    original = torch.randn(7, 8, dtype=torch.float64) * 0.2
    q = original + torch.randn_like(original) * 0.03
    z = [torch.randn(8, count, dtype=torch.float64) * (loop + 1)
         for loop, count in enumerate((19, 23, 27))]
    y = [original @ (value + torch.randn_like(value) * 0.1) for value in z]
    a = [torch.randn(2, 8, dtype=torch.float64) * 0.1 for _ in z]
    b = [torch.randn(7, 2, dtype=torch.float64) * 0.1 for _ in z]
    weights = torch.tensor([0.15, 0.25, 0.6], dtype=torch.float64)
    residual = [target - original @ value for value, target in zip(z, y)]
    stats = {
        "hessian": torch.stack([value @ value.T / value.shape[1] for value in z]),
        "residual_cross": torch.stack([r @ value.T / value.shape[1] for r, value in zip(residual, z)]),
        "residual_energy": torch.stack([r.square().sum() / r.shape[1] for r in residual]),
    }
    return original, q, z, y, a, b, weights, stats


def test_prefix_moment_loss_matches_cumulative_token_outputs():
    original, q, z, y, a, b, weights, stats = problem()
    loss, per_loop = reconstruction_loss(q, a, b, stats, weights, reference=original, sharing="prefix")
    total, expected = torch.zeros_like(q), []
    for loop in range(3):
        total += b[loop] @ a[loop]
        expected.append(float(((q + total) @ z[loop] - y[loop]).square().sum() / z[loop].shape[1]))
    torch.testing.assert_close(torch.tensor(per_loop), torch.tensor(expected))
    assert loss == pytest.approx(float(torch.tensor(expected, dtype=torch.float64) @ weights), rel=1e-12)


@pytest.mark.parametrize("target", ["weight", "update"])
def test_prefix_pass_matches_augmented_token_qr_rank_constrained_oracle(target):
    original, q, z, y, a, b, weights, stats = problem()
    factors, damping, _ = _prefix_factors(stats["hessian"], weights, 0.003)
    actual_a, actual_b, updates = _update_prefix(
        q, original, a, b, stats, weights, factors, damping, target)
    expected = [right @ left for left, right in zip(a, b)]
    anchors = [sum(expected[:loop + 1]) if target == "update" else torch.zeros_like(q) for loop in range(3)]
    for group in range(3):
        inputs, targets = [], []
        for loop in range(group, 3):
            other = sum((expected[j] for j in range(loop + 1) if j != group), torch.zeros_like(q))
            coefficient = (weights[loop] / z[loop].shape[1]).sqrt()
            inputs.append(coefficient * z[loop].T)
            targets.append(coefficient * (y[loop] - (q + other) @ z[loop]).T)
            ridge = (weights[loop] * damping[loop]).sqrt()
            inputs.append(ridge * torch.eye(8, dtype=torch.float64))
            targets.append(ridge * (anchors[loop] - other).T)
        # QR of explicit augmented tokens gives an independent rank-k oracle;
        # it uses neither the cached moments nor the production whitening code.
        orthogonal, triangular = torch.linalg.qr(torch.cat(inputs), mode="reduced")
        projected = orthogonal.T @ torch.cat(targets)
        u, singular, vh = torch.linalg.svd(projected, full_matrices=False)
        truncated = (u[:, :2] * singular[:2]) @ vh[:2]
        expected[group] = torch.linalg.solve_triangular(triangular, truncated, upper=True).T
        torch.testing.assert_close(actual_b[group] @ actual_a[group], expected[group], rtol=1e-11, atol=1e-13)
    assert [entry["group"] for entry in updates] == [1, 2, 3]
    assert all(entry["accepted"] and entry["objective_change"] <= 0 for entry in updates)


def test_prefix_suffix_ridge_matches_full_loop_penalty_after_stability_retry(caplog):
    h = torch.stack((torch.eye(4), torch.diag(torch.tensor([-0.0011, 1.0, 1.0, 1.0])))).double()
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    import logging
    with caplog.at_level(logging.WARNING):
        factors, damping, suffix_damping = _prefix_factors(h, weights, 0.001, logger=logging.getLogger("prefix"))
    assert "Cholesky increased damping" in caplog.text
    assert damping[1] > 0.001 * float(h[1].diag().mean())
    for group in range(2):
        expected = sum(weights[t] * (h[t] + damping[t] * torch.eye(4, dtype=torch.float64)) for t in range(group, 2))
        torch.testing.assert_close(factors[group] @ factors[group].T, expected)
        assert suffix_damping[group] == pytest.approx(sum(float(weights[t]) * damping[t] for t in range(group, 2)))


def test_zero_grams_keep_consistent_positive_prefix_damping():
    h, weights = torch.zeros(3, 8, 8, dtype=torch.float64), torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    factors, damping, _ = _prefix_factors(h, weights, 0.001)
    for group, factor in enumerate(factors):
        rho = sum(float(weights[t]) * damping[t] for t in range(group, 3))
        assert rho > 0
        torch.testing.assert_close(factor @ factor.T, rho * torch.eye(8, dtype=torch.float64))


def test_small_loop_weight_does_not_erase_its_requested_ridge():
    h = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    weights = torch.tensor([1e-20, 1.0], dtype=torch.float64)
    _, damping, _ = _prefix_factors(h, weights, 0.001)
    assert damping == pytest.approx([0.001, 0.001])


@pytest.mark.parametrize("weight_update", ["regression_gptq", "residual_compensation"])
@pytest.mark.parametrize("target", ["weight", "update"])
def test_q_updates_use_all_prefix_groups_and_current_outer_reference(monkeypatch, weight_update, target):
    import loopquant.low_rank.solver as solver
    original, _, z, y, _, _, weights, stats = problem()
    captured, coordinates = [], []
    gptq, direct, coordinate_update = solver.gptq, solver.residual_compensation, solver._update_prefix

    def quantize(value, *args, **kwargs):
        if weight_update == "regression_gptq":
            captured.append(value.clone())
        return gptq(value, *args, **kwargs)

    def compensate(value, cross, **kwargs):
        captured.append(cross.clone())
        return direct(value, cross, **kwargs)

    def coordinate(q, w0, a, b, *args, **kwargs):
        corrections = [sum((b[j].double() @ a[j].double() for j in range(t + 1)), torch.zeros_like(w0))
                       for t in range(3)]
        coordinates.append((q.clone(), corrections))
        return coordinate_update(q, w0, a, b, *args, **kwargs)

    monkeypatch.setattr(solver, "gptq", quantize)
    monkeypatch.setattr(solver, "residual_compensation", compensate)
    monkeypatch.setattr(solver, "_update_prefix", coordinate)
    options = {"weight_update": weight_update, "iterations": 3, "damp_percent": 0.001,
               "loop_weights": weights.tolist(), "block_size": 4, "quantized_damping_target": target}
    _, _, _, report = solve_linear(original, stats, rank=2, options=options, group_size=2, sharing="prefix")
    if weight_update == "regression_gptq":
        captured = captured[1:]  # the common comparison baseline is also GPTQ
    assert len(coordinates) == len(captured) >= 2
    h = sum(weights[t] * z[t] @ z[t].T / z[t].shape[1] for t in range(3))
    rho = report["q_damping"]
    previous_q = original
    for actual, (q, corrections) in zip(captured, coordinates):
        d = sum(weights[t] * (y[t] - corrections[t] @ z[t]) @ z[t].T / z[t].shape[1] for t in range(3))
        reference = previous_q if target == "update" else torch.zeros_like(original)
        if weight_update == "regression_gptq":
            expected = torch.linalg.solve(h + rho * torch.eye(8, dtype=torch.float64), (d + rho * reference).T).T
        else:
            expected = d - original @ h + rho * (reference - original)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)
        previous_q = q


def test_prefix_baseline_uses_disjoint_svd_groups_instead_of_repeating_rank_k(monkeypatch):
    import loopquant.low_rank.solver as solver
    w = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64))
    stats = {"hessian": torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
             "residual_cross": torch.zeros(2, 4, 4, dtype=torch.float64),
             "residual_energy": torch.zeros(2, dtype=torch.float64)}
    monkeypatch.setattr(solver, "gptq", lambda value, *args, **kwargs: torch.zeros_like(value))
    options = {"weight_update": "regression_gptq", "iterations": 1, "damp_percent": 0.001,
               "loop_weights": None, "block_size": 4}
    q, a, b, report = solve_linear(w, stats, rank=1, options=options, group_size=2, sharing="prefix")
    torch.testing.assert_close(b[0] @ a[0], torch.diag(torch.tensor([8.0, 0, 0, 0], dtype=torch.float64)))
    torch.testing.assert_close(b[0] @ a[0] + b[1] @ a[1],
                               torch.diag(torch.tensor([8.0, 4.0, 0, 0], dtype=torch.float64)))
    assert report["best_iteration"] == 0 and report["final_loss"] == pytest.approx(13.0)
    assert report["rank_schedule"] == [1, 2]


def test_prefix_coordinate_keeps_old_factors_when_stored_candidate_is_worse(monkeypatch):
    import loopquant.low_rank.solver as solver
    original, q, _, _, a, b, weights, stats = problem()
    factors, damping, _ = _prefix_factors(stats["hessian"], weights, 0.001)
    monkeypatch.setattr(solver, "low_rank_factors", lambda cross, factor, rank:
                        (cross.new_full((rank, 8), 1e5), cross.new_full((7, rank), 1e5)))
    next_a, next_b, updates = _update_prefix(q, original, a, b, stats, weights, factors, damping, "update")
    assert all(torch.equal(left, right) for left, right in zip(a + b, next_a + next_b))
    assert all(not entry["accepted"] and entry["objective_change"] == 0 for entry in updates)


@pytest.mark.parametrize("weight_update", ["regression_gptq", "residual_compensation"])
@pytest.mark.parametrize("q_target", ["weight", "update"])
@pytest.mark.parametrize("low_rank_target", ["weight", "update"])
def test_single_loop_prefix_matches_independent(weight_update, q_target, low_rank_target):
    original, _, _, _, _, _, _, stats = problem()
    stats = {name: value[:1] for name, value in stats.items()}
    options = {"weight_update": weight_update, "iterations": 3, "damp_percent": 0.001,
               "loop_weights": None, "block_size": 4, "quantized_damping_target": q_target,
               "low_rank_damping_target": low_rank_target}
    independent = solve_linear(original, stats, rank=2, options=options, group_size=2)
    prefix = solve_linear(original, stats, rank=2, options=options, group_size=2, sharing="prefix")
    torch.testing.assert_close(independent[0], prefix[0])
    torch.testing.assert_close(independent[2][0] @ independent[1][0], prefix[2][0] @ prefix[1][0])
    assert independent[3]["final_loss"] == pytest.approx(prefix[3]["final_loss"], rel=1e-10)
