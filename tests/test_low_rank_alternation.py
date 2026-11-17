"""Rotated alternating Q/BA fitting checked in explicit token space."""

import pytest
import torch

from loopquant.low_rank.quantization import quantize_activation
from loopquant.low_rank.rotation import block_hadamard_transform, rotate_weight
from loopquant.low_rank.solver import solve_linear


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_rotation_preserves_fp_outputs_and_weight_singular_values(dtype):
    torch.manual_seed(613)
    value = torch.randn(19, 24, dtype=dtype)
    weight = torch.randn(13, 24, dtype=dtype)
    rotated_value = block_hadamard_transform(value, 8, 19)
    rotated_weight = rotate_weight(weight, 8, 19)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(rotated_value @ rotated_weight.T, value @ weight.T,
                               rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(torch.linalg.svdvals(rotated_weight), torch.linalg.svdvals(weight),
                               rtol=tolerance, atol=tolerance)


def rank_constrained_regression(inputs, targets, rank):
    """Independent QR oracle, including explicit ridge rows."""
    orthogonal, triangular = torch.linalg.qr(torch.cat(inputs), mode="reduced")
    projected = orthogonal.T @ torch.cat(targets)
    u, singular, vh = torch.linalg.svd(projected, full_matrices=False)
    truncated = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    return torch.linalg.solve_triangular(triangular, truncated, upper=True).T


@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("q_target", ["weight", "update"])
@pytest.mark.parametrize("l_target", ["weight", "update"])
def test_rotated_rounds_match_token_regression_and_rank_constrained_oracles(
        monkeypatch, sharing, bits, q_target, l_target):
    import loopquant.low_rank.solver as solver

    torch.manual_seed(827)
    width, outputs, rank, loops = 16, 9, 2, 3
    original = torch.randn(outputs, width, dtype=torch.float64) * 0.1
    reference = rotate_weight(original, 8, 19)
    x = [torch.randn(width, count, dtype=torch.float64) * (loop + 1)
         for loop, count in enumerate((31, 37, 43))]
    for value in x:
        value[0] *= 7
    y = [original @ value for value in x]
    # Inputs and weights enter the same rotated basis; teacher outputs stay
    # in their original output coordinates. BA and Q both consume this Z.
    z = [quantize_activation(block_hadamard_transform(value.T, 8, 19), bits,
                             group_size=4, clip=0.9).T for value in x]
    residual = [target - reference @ value for target, value in zip(y, z)]
    stats = {
        "hessian": torch.stack([value @ value.T / value.shape[1] for value in z]),
        "residual_cross": torch.stack([r @ value.T / value.shape[1] for r, value in zip(residual, z)]),
        "residual_energy": torch.stack([r.square().sum() / r.shape[1] for r in residual]),
    }
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    rounds, factor_calls = [], []
    original_gptq, original_low_rank = solver.gptq, solver.low_rank_factors
    calls = 0

    def quantize(continuous, *args, **kwargs):
        nonlocal calls
        result = original_gptq(continuous, *args, **kwargs)
        if calls:
            rounds.append((continuous.clone(), result.to(reference).clone()))
        calls += 1  # the first call is the retained comparison baseline
        return result

    def fit(cross, factor, fitted_rank, **kwargs):
        a, b = original_low_rank(cross, factor, fitted_rank, **kwargs)
        factor_calls.append((a.clone(), b.clone()))
        return a, b

    monkeypatch.setattr(solver, "gptq", quantize)
    monkeypatch.setattr(solver, "low_rank_factors", fit)
    options = {"weight_update": "regression_gptq", "iterations": 3, "damp_percent": 0.04,
               "loop_weights": weights.tolist(), "block_size": 6,
               "quantized_damping_target": q_target, "low_rank_damping_target": l_target}
    final_q, final_a, final_b, report = solve_linear(reference, stats, rank=rank,
                                                  options=options, group_size=4, sharing=sharing)
    assert rounds and len(factor_calls) == loops * len(rounds)
    identity = torch.eye(width, dtype=torch.float64)
    groups = [torch.zeros_like(reference) for _ in range(loops)]
    previous_q = reference
    for iteration, (continuous, q) in enumerate(rounds):
        old_groups = list(groups)
        old_l = [sum(old_groups[:t + 1]) if sharing == "prefix" else old_groups[t] for t in range(loops)]
        q_reference = previous_q if q_target == "update" else torch.zeros_like(reference)
        inputs, targets = [], []
        for t in range(loops):
            scale = (weights[t] / z[t].shape[1]).sqrt()
            inputs.append(scale * z[t].T)
            targets.append(scale * (y[t] - old_l[t] @ z[t]).T)
        ridge = report["q_damping"]**0.5
        inputs.append(ridge * identity)
        targets.append(ridge * q_reference.T)
        expected_q = torch.linalg.lstsq(torch.cat(inputs), torch.cat(targets)).solution.T
        torch.testing.assert_close(continuous, expected_q, rtol=1e-10, atol=1e-12)

        for j in range(loops):
            inputs, targets = [], []
            affected = range(j, loops) if sharing == "prefix" else (j,)
            for t in affected:
                other = (sum((groups[l] for l in range(t + 1) if l != j), torch.zeros_like(reference))
                         if sharing == "prefix" else torch.zeros_like(reference))
                coefficient = weights[t] if sharing == "prefix" else 1.0
                scale = (coefficient / z[t].shape[1])**0.5
                inputs.append(scale * z[t].T)
                targets.append(scale * (y[t] - (q + other) @ z[t]).T)
                ridge = (coefficient * report["low_rank_damping"][t])**0.5
                anchor = old_l[t] if l_target == "update" else torch.zeros_like(reference)
                inputs.append(ridge * identity)
                targets.append(ridge * (anchor - other).T)
            expected = rank_constrained_regression(inputs, targets, rank)
            a, b = factor_calls[iteration * loops + j]
            groups[j] = b @ a
            torch.testing.assert_close(groups[j], expected, rtol=1e-9, atol=1e-11)
            assert torch.linalg.matrix_rank(groups[j]) <= rank
        previous_q = q
        per_loop = [sum(groups[:t + 1]) if sharing == "prefix" else groups[t] for t in range(loops)]
        data_loss = sum(weights[t] * ((q + per_loop[t]) @ z[t] - y[t]).square().sum()
                        / z[t].shape[1] for t in range(loops))
        assert report["iterations"][iteration]["loss"] == pytest.approx(float(data_loss), rel=1e-10, abs=1e-12)

    final_l = [sum((final_b[j] @ final_a[j] for j in range(t + 1)), torch.zeros_like(reference))
               if sharing == "prefix" else final_b[t] @ final_a[t] for t in range(loops)]
    final_loss = sum(weights[t] * ((final_q + final_l[t]) @ z[t] - y[t]).square().sum()
                     / z[t].shape[1] for t in range(loops))
    assert report["final_loss"] == pytest.approx(float(final_loss), rel=1e-10, abs=1e-12)
    assert report["final_loss"] <= report["baseline_loss"]
