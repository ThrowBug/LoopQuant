"""Initialization control flow and diagnostics, separate from token-space oracles."""

import logging

import pytest
import torch

from loopquant.low_rank.solver import solve_linear
from test_low_rank_prefix import problem


@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("weight_update", ["regression_gptq", "residual_compensation"])
def test_zero_rank_skips_initialization_and_matches_quantized_first(monkeypatch, sharing, weight_update):
    import loopquant.low_rank.solver as solver

    original, _, _, _, _, _, weights, stats = problem()
    options = {"weight_update": weight_update, "iterations": 2, "damp_percent": 0.001,
               "loop_weights": weights.tolist(), "block_size": 6, "coordinate_sweeps": 1}

    def unexpected(*args, **kwargs):
        raise AssertionError("zero rank must not fit any low-rank matrices")

    monkeypatch.setattr(solver, "_update_independent", unexpected)
    monkeypatch.setattr(solver, "_update_prefix", unexpected)
    quantized = solve_linear(original, stats, rank=0, options=options, sharing=sharing, group_size=4)
    low_rank = solve_linear(original, stats, rank=0, options={**options, "initialization": "low_rank_first"},
                            sharing=sharing, group_size=4)
    torch.testing.assert_close(quantized[0], low_rank[0], rtol=0, atol=0)
    assert quantized[1:3] == low_rank[1:3] == ([], [])
    assert quantized[3]["iterations"] == low_rank[3]["iterations"]
    assert low_rank[3]["initialization"] == "low_rank_first"
    assert low_rank[3]["initialization_report"] == {"performed": False}


@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("max_calls", [0, 1])
def test_initialization_and_each_loop_have_distinct_logs(caplog, sharing, max_calls):
    original, _, _, _, _, _, weights, stats = problem()
    # Exercise the production mixed-statistics path, including the X=Z ablation.
    stats.update(low_rank_hessian=stats["hessian"], input_cross=stats["hessian"],
                 low_rank_cross=stats["residual_cross"])
    options = {"weight_update": "regression_gptq", "iterations": 1, "damp_percent": 0.001,
               "initialization": "low_rank_first", "loop_weights": weights.tolist(), "block_size": 6,
               "svd_diagnostics": {"enabled": True, "max_calls_per_linear": max_calls}}
    with caplog.at_level(logging.INFO):
        _, _, _, report = solve_linear(original, stats, rank=2, options=options, group_size=4,
                                        sharing=sharing, logger=logging.getLogger("init-test"), label="linear")
    messages = [record.getMessage() for record in caplog.records]
    coordinate = "loop" if sharing == "independent" else "prefix_group"
    for stage in ("phase=initialization", "phase=alternation iteration=1/1"):
        for loop in range(1, 4):
            assert any(f"{stage} {coordinate}={loop}/3 accepted=" in message for message in messages)
    assert any("phase=initialization completed loss=" in message for message in messages)
    svd_logs = [message for message in messages if "singular_values=" in message]
    assert len(svd_logs) == (6 if max_calls == 0 else 1)
    assert "phase=initialization" in svd_logs[0]
    assert len(report["iterations"]) == 1  # the extra pass is not iteration 1
    assert report["initialization_report"]["performed"]
    assert report["best_iteration"] in (0, 1)
    assert report["final_loss"] <= report["baseline_loss"]


@pytest.mark.parametrize("invalid", ["typo", None, False, []])
def test_direct_solver_rejects_unknown_initialization(invalid):
    original, _, _, _, _, _, _, stats = problem()
    with pytest.raises(ValueError, match="solver.initialization"):
        solve_linear(original, stats, rank=2, options={"initialization": invalid})
