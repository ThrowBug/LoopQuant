"""Numerical checks against token-space regression, independent of model code."""

import pytest
import torch

from loopquant.low_rank.solver import (
    continuous_weight, low_rank_factors, reconstruction_loss, regularized_cholesky, solve_linear,
)


def moments(z, y):
    return {"hessian": z @ z.transpose(-1, -2) / z.shape[-1],
            "cross": y @ z.transpose(-1, -2) / z.shape[-1],
            "target_energy": y.square().sum((-1, -2), dtype=torch.float64) / z.shape[-1],
            "num_tokens": torch.full((len(z),), z.shape[-1], dtype=torch.int64)}


def test_continuous_regression_completion_and_weighted_concatenation():
    torch.manual_seed(91)
    z = torch.randn(3, 6, 30, dtype=torch.float64)
    e = torch.randn(3, 4, 30, dtype=torch.float64)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    stats = moments(z, e)
    h = (stats["hessian"] * weights[:, None, None]).sum(0)
    d = (stats["cross"] * weights[:, None, None]).sum(0)
    w = continuous_weight(d, torch.linalg.cholesky(h))
    joined_z = torch.cat([weights[t].sqrt() * z[t] for t in range(3)], 1)
    joined_e = torch.cat([weights[t].sqrt() * e[t] for t in range(3)], 1)
    reference = torch.linalg.lstsq(joined_z.T, joined_e.T).solution.T
    torch.testing.assert_close(w, reference)
    q = torch.randn_like(w)
    original = sum(weights[t] * (q @ z[t] - e[t]).square().sum() / 30 for t in range(3))
    irreducible = sum(weights[t] * (w @ z[t] - e[t]).square().sum() / 30 for t in range(3))
    projected = sum(weights[t] * ((q - w) @ z[t]).square().sum() / 30 for t in range(3))
    torch.testing.assert_close(original, irreducible + projected)


def test_weighted_low_rank_matches_svd_bound_and_beats_unweighted_truncation():
    torch.manual_seed(12)
    z = torch.randn(6, 80, dtype=torch.float64) * torch.tensor([0.1, 0.3, 1, 2, 5, 10])[:, None]
    residual = torch.randn(5, 6, dtype=torch.float64) @ z + torch.randn(5, 80, dtype=torch.float64)
    h, c = z @ z.T / 80, residual @ z.T / 80
    g = torch.linalg.cholesky(h)
    a, b = low_rank_factors(c, g, 2)
    l = b @ a
    m = torch.linalg.solve_triangular(g, c.T, upper=False).T
    singular = torch.linalg.svdvals(m)
    torch.testing.assert_close((l @ g - m).square().sum(), singular[2:].square().sum())
    dense = continuous_weight(c, g)
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    ordinary = (u[:, :2] * s[:2]) @ vh[:2]
    assert (l @ z - residual).square().sum() < (ordinary @ z - residual).square().sum()
    assert torch.linalg.matrix_rank(l) == 2


def test_relative_damping_and_full_rank_ridge_solution():
    torch.manual_seed(17)
    z = torch.randn(6, 20, dtype=torch.float64)
    target = torch.randn(4, 20, dtype=torch.float64)
    h, c = z @ z.T / 20, target @ z.T / 20
    g, rho = regularized_cholesky(h, 0.01)
    assert rho == pytest.approx(0.01 * h.diag().mean().item())
    regularized = h + rho * torch.eye(6, dtype=torch.float64)
    a, b = low_rank_factors(c, g, 4)
    torch.testing.assert_close(b @ a @ regularized, c)
    torch.testing.assert_close(continuous_weight(c, g), b @ a)
    # Whitening is exactly the regularized objective up to a constant.
    trial = torch.randn(4, 6, dtype=torch.float64)
    m = torch.linalg.solve_triangular(g, c.T, upper=False).T
    left = (trial @ z - target).square().sum() / 20 + rho * trial.square().sum()
    right = (trial @ g - m).square().sum() + target.square().sum() / 20 - m.square().sum()
    torch.testing.assert_close(left, right)


def test_singular_and_zero_grams_are_regularized():
    h = torch.ones(8, 8)
    g, _ = regularized_cholesky(h, 0.01)
    assert torch.isfinite(g).all()
    zero, rho = regularized_cholesky(torch.zeros(8, 8), 0.01)
    assert rho > 0 and torch.isfinite(zero).all()
    with pytest.raises(ValueError, match="positive definite"):
        regularized_cholesky(torch.diag(torch.tensor([-2.0, 1.0])), 0.01)


def test_sufficient_statistics_loss_matches_direct_token_loss():
    torch.manual_seed(23)
    z, y = torch.randn(3, 32, 70), torch.randn(3, 12, 70)
    q = torch.randn(12, 32)
    a, b = [torch.randn(2, 32) for _ in range(3)], [torch.randn(12, 2) for _ in range(3)]
    weights = torch.tensor([0.1, 0.2, 0.7])
    loss, per_loop = reconstruction_loss(q, a, b, moments(z, y), weights)
    expected = torch.stack([((q + b[t] @ a[t]) @ z[t] - y[t]).square().sum() / 70 for t in range(3)])
    torch.testing.assert_close(torch.tensor(per_loop), expected)
    assert loss == pytest.approx(float((weights * expected).sum()), rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_solver_keeps_best_full_pair_and_reuses_factors(monkeypatch, dtype):
    import loopquant.low_rank.solver as solver
    torch.manual_seed(71)
    z = torch.randn(3, 32, 80)
    w = torch.randn(16, 32) * 0.1
    y = torch.stack([w @ value + torch.randn(16, 80) * 0.01 for value in z])
    options = {"weight_update": "regression_gptq", "iterations": 3, "damp_percent": 0.01,
               "loop_weights": [1, 2, 3], "block_size": 32}
    calls, cholesky_calls = [], []
    original_gptq, original_cholesky = solver.gptq, solver.regularized_cholesky

    def update(*args, **kwargs):
        calls.append(kwargs["inverse_factor"])
        if len(calls) == 1:
            return original_gptq(*args, **kwargs)
        return torch.full_like(args[0], 1000)

    def factor(*args):
        cholesky_calls.append(1)
        return original_cholesky(*args)

    original_low_rank = solver.low_rank_factors
    def low_rank(cross, factor, rank):
        if len(calls) > 1:
            return cross.new_zeros(rank, cross.shape[1]), cross.new_zeros(cross.shape[0], rank)
        return original_low_rank(cross, factor, rank)

    monkeypatch.setattr(solver, "gptq", update)
    monkeypatch.setattr(solver, "regularized_cholesky", factor)
    monkeypatch.setattr(solver, "low_rank_factors", low_rank)
    q, a, b, report = solve_linear(w.to(dtype), moments(z, y), rank=2, options=options)
    assert report["best_iteration"] == 1
    assert [step["accepted"] for step in report["iterations"]] == [True, False]
    assert len(cholesky_calls) == 4  # one shared, one per loop; not per round
    assert calls[0] is calls[1]
    assert q.dtype == dtype and all(p.dtype == dtype for p in a + b)
    assert q.abs().max() < 1000
    actual, _ = reconstruction_loss(q, a, b, moments(z, y), torch.tensor([1, 2, 3]) / 6)
    assert actual == pytest.approx(report["final_loss"])
