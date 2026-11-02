"""Pure-quantization behavior when rank_step=0."""

import json

import pytest
import torch

from loopquant.checkpoint import save_quantized_checkpoint
from loopquant.low_rank.modules import LowRankLinear, iter_linears
from loopquant.low_rank.pipeline import LowRankPipeline
from loopquant.low_rank.quantization import quantize_activation
from loopquant.models import load_model_for_evaluation
from test_low_rank import configuration, tiny, DATA


def forbid_low_rank(monkeypatch):
    import loopquant.low_rank.solver as solver

    def forbidden(*args, **kwargs):
        pytest.fail("rank_step=0 must skip every low-rank solve and product")

    for name in ("svd_factors", "low_rank_factors", "_prefix_factors", "_update_prefix", "_loop_factors"):
        monkeypatch.setattr(solver, name, forbidden)


@pytest.mark.parametrize("weight_update", ["regression_gptq", "residual_compensation"])
@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("q_target", ["weight", "update"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_rank_updates_only_q_and_scores_the_uncompensated_target(monkeypatch, weight_update, sharing,
                                                                   q_target, dtype):
    import loopquant.low_rank.solver as solver
    torch.manual_seed(211)
    w = (torch.randn(5, 32) * 0.2).to(dtype)
    z = [torch.randn(32, count, dtype=torch.float64) * (t + 1) for t, count in enumerate((40, 51))]
    target = w.double() + torch.randn(5, 32, dtype=torch.float64) * 0.03
    y = [target @ value for value in z]
    residual = [output - w.double() @ value for output, value in zip(y, z)]
    stats = {"hessian": torch.stack([value @ value.T / value.shape[1] for value in z]),
             "residual_cross": torch.stack([r @ value.T / value.shape[1] for r, value in zip(residual, z)]),
             "residual_energy": torch.stack([r.square().sum() / r.shape[1] for r in residual])}
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    original_gptq, original_direct, original_cholesky = solver.gptq, solver.residual_compensation, solver.regularized_cholesky
    captured, factorizations = [], []

    def quantize(value, *args, **kwargs):
        q = original_gptq(value, *args, **kwargs)
        if weight_update == "regression_gptq":
            captured.append((value.clone(), q.to(dtype)))
        return q

    def compensate(value, cross, **kwargs):
        q = original_direct(value, cross, **kwargs)
        captured.append((cross.clone(), q.to(dtype)))
        return q

    def factor(*args, **kwargs):
        factorizations.append(kwargs.get("label"))
        return original_cholesky(*args, **kwargs)

    forbid_low_rank(monkeypatch)
    monkeypatch.setattr(solver, "gptq", quantize)
    monkeypatch.setattr(solver, "residual_compensation", compensate)
    monkeypatch.setattr(solver, "regularized_cholesky", factor)
    options = {"weight_update": weight_update, "iterations": 3, "damp_percent": 0.001,
               "loop_weights": [1, 3], "block_size": 32, "quantized_damping_target": q_target,
               "low_rank_damping_target": "update"}
    q, a, b, report = solver.solve_linear(w, stats, rank=0, options=options, sharing=sharing)
    assert a == b == [] and not report["low_rank_enabled"]
    assert report["low_rank_damping"] == [] and report["rank_schedule"] == [0, 0]
    assert len(factorizations) == 1  # the shared Q system only
    if sharing == "prefix":
        assert report["prefix_group_damping"] == []
        assert all(step["prefix_updates"] == [] for step in report["iterations"])
    if weight_update == "regression_gptq":
        captured = captured[1:]  # the common GPTQ-only comparison baseline
    assert len(captured) == len(report["iterations"]) > 0
    h = sum(weights[t] * stats["hessian"][t] for t in range(2))
    d = sum(weights[t] * y[t] @ z[t].T / z[t].shape[1] for t in range(2))
    rho, previous = report["q_damping"], w.double()
    for actual, candidate in captured:
        reference = previous if q_target == "update" else torch.zeros_like(previous)
        if weight_update == "regression_gptq":
            expected = torch.linalg.solve(h + rho * torch.eye(32, dtype=torch.float64),
                                           (d + rho * reference).T).T
        else:
            expected = d - w.double() @ h + rho * (reference - w.double())
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)
        previous = candidate.double()
    expected_loss = sum(weights[t] * (q.double() @ z[t] - y[t]).square().sum() / z[t].shape[1] for t in range(2))
    assert report["final_loss"] == pytest.approx(float(expected_loss), rel=1e-11)


@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_rank_linear_keeps_activation_quantization_and_executes_one_linear(monkeypatch, sharing, bits, dtype):
    import loopquant.low_rank.modules as modules
    torch.manual_seed(223)
    original = torch.nn.Linear(32, 16, bias=True).to(dtype)
    module = LowRankLinear(original, num_loops=3, rank_step=0, sharing=sharing, activation_bits=bits)
    module.clip_ratios.copy_(torch.tensor([0.81, 0.92, 0.87]))
    assert len(module.A) == len(module.B) == 0
    assert set(dict(module.named_parameters())) == {"weight", "bias"}
    assert set(module.state_dict()) == {"weight", "bias", "clip_ratios"}
    module.prepare_inference()  # must not concatenate empty prefix banks
    assert module._inference_cache is None
    original_linear, calls = modules.F.linear, []

    def linear(*args, **kwargs):
        calls.append(1)
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(modules.F, "linear", linear)
    monkeypatch.setattr(module, "factors", lambda *args: pytest.fail("zero-rank inference must skip factors"))
    value = torch.randn(2, 3, 32).to(dtype)
    for loop in range(3):
        expected = original_linear(quantize_activation(value, bits, clip=module.clip_ratios[loop]),
                                   original.weight, original.bias)
        torch.testing.assert_close(module.forward_at_loop(value, loop), expected, rtol=0, atol=0)
    assert len(calls) == 3


@pytest.mark.parametrize("weight_update", ["regression_gptq", "residual_compensation"])
@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("setting", ["w4a4", "w4a8"])
def test_zero_rank_pipeline_checkpoint_and_evaluation_loader_have_no_low_rank_weights(
        tmp_path, monkeypatch, weight_update, sharing, setting):
    model, tokenizer = tiny(loops=3, layers=1)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter.ndim == 1 or name.startswith(("model.embed_tokens", "lm_head"))}
    config = configuration(tmp_path, loops=3, sharing=sharing, setting=setting, rank_step=0)
    config.method["solver"]["weight_update"] = weight_update
    forbid_low_rank(monkeypatch)
    result = LowRankPipeline(config, device="cpu").run(model, tokenizer, DATA)
    assert not result.summary["low_rank_enabled"]
    assert result.summary["low_rank_parameters"] == result.summary["low_rank_bytes"] == 0
    assert result.summary["rank_schedule"] == [0, 0, 0]
    assert result.summary["stage_order"][-1] == f"quantization_{weight_update}"
    for name, value in frozen.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    for block in model.model.layers:
        for _, module in iter_linears(block):
            assert len(module.A) == len(module.B) == 0
    with torch.no_grad():
        logits = model(DATA).logits
        tokens = model.generate(DATA[:1], max_new_tokens=2, do_sample=False, use_cache=False)
    checkpoint = tmp_path / "checkpoint"
    save_quantized_checkpoint(model, tokenizer, checkpoint)
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rank_schedule"] == [0, 0, 0]
    assert all(".A." not in name and ".B." not in name for name in manifest["weight_map"])
    restored, _, metadata = load_model_for_evaluation(checkpoint, num_loops=3, device="cpu")
    assert metadata["algorithm"] == "low_rank" and restored._low_rank_config["method"]["rank_step"] == 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    with torch.no_grad():
        torch.testing.assert_close(restored(DATA).logits, logits, rtol=0, atol=0)
        assert torch.equal(restored.generate(DATA[:1], max_new_tokens=2, do_sample=False, use_cache=False), tokens)
