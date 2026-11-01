"""Independent token-space checks of fixed-target residual compensation."""

import importlib

import pytest
import torch

from loopquant.low_rank.residual_compensation import residual_compensation
from loopquant.low_rank.quantization import gptq
from loopquant.low_rank.solver import regularized_cholesky, solve_linear


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("target", ["weight", "update"])
def test_residual_compensation_matches_repeated_constrained_token_lstsq(monkeypatch, block_size, target):
    backend = importlib.import_module("loopquant.low_rank.residual_compensation")
    torch.manual_seed(137)
    w = torch.randn(3, 8) * 0.1
    z = torch.randn(8, 40, dtype=torch.float64)
    e = torch.randn(3, 40, dtype=torch.float64)
    reference = torch.zeros_like(w).double() if target == "weight" else torch.randn_like(w).double() * 0.2
    h = z @ z.T / z.shape[1]
    factor, rho = regularized_cholesky(h, 0.07)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    c = (e - w.double() @ z) @ z.T / z.shape[1] + rho * (reference - w.double())
    # Fix the grid so this test isolates the OBS solve, including saturation.
    scale = torch.tensor([0.03, 0.04, 0.05])
    monkeypatch.setattr(backend, "_weight_scale", lambda *args: scale)
    actual = residual_compensation(w, c, inverse_factor=inverse, group_size=2, block_size=block_size)

    # Refit all free weights by an explicit augmented token least-squares solve
    # after every rounded column. This uses neither inverse factors nor moments.
    inputs = torch.cat((z / z.shape[1]**0.5, rho**0.5 * torch.eye(8, dtype=torch.float64)), 1)
    targets = torch.cat((e / z.shape[1]**0.5, rho**0.5 * reference), 1)
    expected, work = torch.empty_like(w), w.double().clone()
    for column in range(8):
        expected[:, column] = ((work[:, column] / scale.double()).round().clamp(-7, 7)
                               * scale.double()).float()
        if column < 7:
            rhs = targets - expected[:, :column + 1].double() @ inputs[:column + 1]
            work[:, column + 1:] = torch.linalg.lstsq(inputs[column + 1:].T, rhs.T).solution.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_asymmetric_residual_reduces_exactly_to_existing_gptq(block_size, dtype):
    torch.manual_seed(173)
    w = (torch.randn(5, 16) * 0.1).to(dtype)
    z = torch.randn(16, 40, dtype=torch.float64)
    h = z @ z.T / 40
    factor, _ = regularized_cholesky(h, 0.001)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    expected = gptq(w, h, group_size=4, block_size=block_size, inverse_factor=inverse.float())
    actual = residual_compensation(w, torch.zeros_like(w).double(), group_size=4, block_size=block_size,
                                   inverse_factor=inverse)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_target_changes_compensated_weights_without_changing_first_column():
    torch.manual_seed(19)
    w = torch.randn(4, 32) * 0.1
    z = torch.randn(32, 80, dtype=torch.float64)
    h = z @ z.T / 80
    factor, _ = regularized_cholesky(h, 0.001)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)
    unchanged = residual_compensation(w, torch.zeros_like(w).double(), inverse_factor=inverse)
    shifted = residual_compensation(w, torch.randn_like(w).double() * 0.1, inverse_factor=inverse)
    assert torch.equal(unchanged[:, 0], shifted[:, 0])
    assert not torch.equal(unchanged[:, 1:], shifted[:, 1:])


@pytest.mark.parametrize("target", ["weight", "update"])
def test_multiloop_residual_and_damping_are_forwarded_without_unrestricted_regression(monkeypatch, target):
    import loopquant.low_rank.solver as solver
    torch.manual_seed(83)
    w = torch.randn(4, 6) * 0.1
    z = torch.randn(2, 6, 40) * torch.tensor([1.0, 2.0])[:, None, None]
    y = (2 * w)[None] @ z
    residual = y.double() - w.double()[None] @ z.double()
    stats = {"hessian": z.double() @ z.double().transpose(-1, -2) / 40,
             "residual_cross": residual @ z.double().transpose(-1, -2) / 40,
             "residual_energy": residual.square().sum((-1, -2)) / 40}
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    records, low_rank = [], []
    original_compensation, original_low_rank = solver.residual_compensation, solver.low_rank_factors

    def direct(value, cross, **kwargs):
        q = original_compensation(value, cross, **kwargs)
        records.append((value.clone(), cross.clone(), q.to(w.dtype)))
        return q

    def fit(*args):
        a, b = original_low_rank(*args)
        low_rank.append(b.to(w.dtype).double() @ a.to(w.dtype).double())
        return a, b

    monkeypatch.setattr(solver, "residual_compensation", direct)
    monkeypatch.setattr(solver, "low_rank_factors", fit)
    monkeypatch.setattr(solver, "continuous_weight", lambda *args: pytest.fail("unrestricted regression called"))
    options = {"weight_update": "residual_compensation", "iterations": 3, "damp_percent": 0.001,
               "loop_weights": [1, 3], "block_size": 4, "quantized_damping_target": target}
    _, _, _, report = solve_linear(w, stats, rank=2, options=options, group_size=2)
    assert len(records) >= 2
    assert report["weight_update_implementation"] == "residual_compensation"
    assert report["selected_candidate"] in ("baseline", "residual_compensation")
    previous_q = w.double()
    for iteration, (initial, cross, q) in enumerate(records):
        previous_l = (torch.stack(low_rank[2 * (iteration - 1):2 * iteration]) if iteration
                      else torch.zeros(2, 4, 6, dtype=torch.float64))
        expected = ((stats["residual_cross"] - previous_l @ stats["hessian"])
                    * weights[:, None, None]).sum(0)
        reference = previous_q if target == "update" else torch.zeros_like(previous_q)
        expected += report["q_damping"] * (reference - w.double())
        torch.testing.assert_close(initial, w.double(), rtol=0, atol=0)
        torch.testing.assert_close(cross, expected, rtol=1e-12, atol=1e-14)
        previous_q = q.double()


def test_single_input_channel_has_no_remaining_weights_to_compensate():
    w = torch.tensor([[0.1], [-0.2]])
    q = residual_compensation(w, torch.ones_like(w), inverse_factor=torch.ones(1, 1), group_size=1)
    assert torch.isfinite(q).all() and q.shape == w.shape


@pytest.mark.parametrize("invalid", ["residual_shape", "factor_shape", "nonfinite", "diagonal", "block_size"])
def test_invalid_residual_compensation_inputs_are_rejected(invalid):
    w, c, u = torch.ones(2, 4), torch.zeros(2, 4), torch.eye(4)
    block_size = 4
    if invalid == "residual_shape":
        c = c[:, :2]
    elif invalid == "factor_shape":
        u = u[:2, :2]
    elif invalid == "nonfinite":
        c[0, 0] = float("nan")
    elif invalid == "diagonal":
        u[0, 0] = 0
    else:
        block_size = 0
    with pytest.raises(ValueError, match="residual compensation"):
        residual_compensation(w, c, inverse_factor=u, group_size=2, block_size=block_size)
