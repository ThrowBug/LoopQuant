from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from loopquant import cli


def test_store_true_defaults_and_suite_selection(tmp_path):
    protocol = tmp_path / "protocol.yaml"
    # Suite selection is independent of the YAML protocol.
    protocol.write_text("evaluation: {}\n")
    base = ["quantize", "--model", "unused", "--config", "unused", "--output-dir", "unused",
            "--eval-config", str(protocol)]
    args = cli.build_parser().parse_args(base)
    assert not args.save and not cli._evaluation_requested(args)
    assert cli._evaluation_suites(args) == []
    assert cli._evaluation_suites(cli.build_parser().parse_args(base + ["--generative"])) == ["generative"]
    assert cli.build_parser().parse_args(base + ["--save"]).save
    with pytest.raises(SystemExit):
        cli.main(["evaluate", "--model", "unused", "--output-dir", "unused"])


def test_evaluate_requires_loop_count():
    parser = cli.build_parser()
    base = ["evaluate", "--model", "unused", "--output-dir", "unused", "--wikitext2"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    assert parser.parse_args(base + ["--num-loops", "4"]).num_loops == 4
    with pytest.raises(SystemExit):
        cli.main(base + ["--num-loops", "0"])


def test_quantize_requires_loop_count_and_rejects_removed_parameters():
    from loopquant.config import LoopQuantConfig
    with pytest.raises(ValueError, match="num_loops is required"):
        LoopQuantConfig.from_dict({})
    for value in (None, True, 0, 4.5):
        with pytest.raises(ValueError, match="positive integer"):
            LoopQuantConfig.from_dict({"num_loops": value})
    for removed in ({"evaluation": {}}, {"unused_parameter": True}):
        with pytest.raises(ValueError, match="unknown configuration parameter"):
            LoopQuantConfig.from_dict({"num_loops": 4, **removed})


def test_model_key_and_method_parameters_preserved():
    from loopquant.config import LoopQuantConfig
    config = LoopQuantConfig.from_dict({
        "num_loops": 4, "model_key": "ouro_1.4b",
        "method": {"cldc": {"enabled": True, "lambda_sync": 0.2}},
    })
    saved = config.to_dict()
    assert saved["model_key"] == "ouro_1.4b"
    assert saved["method"]["cldc"]["lambda_sync"] == 0.2
    assert saved["method"]["cldc"]["implementation"] == "learned_rotation_group_aware_v2"
    assert all(saved["method"][stage]["enabled"] for stage in ("cldc", "ilpa", "lctc"))


@pytest.mark.parametrize("save,evaluate", [(False, False), (True, False), (False, True), (True, True)])
def test_quantize_route_combinations(tmp_path, monkeypatch, save, evaluate):
    import loopquant.checkpoint
    from loopquant.pipeline import LoopQuantPipeline
    events = []
    model = SimpleNamespace()
    result = SimpleNamespace(model=model, allocation=[4] * 4, cldc={}, ilpa={}, lctc={},
                             stage_order=("CLDC", "ILPA", "LCTC"))
    monkeypatch.setattr(cli, "load_looped_transformer", lambda *a, **k: (model, "tokenizer"))
    monkeypatch.setattr(cli, "load_calibration_data", lambda *a, **k: torch.zeros(128, 2048))
    monkeypatch.setattr(LoopQuantPipeline, "run", lambda *a, **k: events.append("quantize") or result)

    def fake_save(model, tokenizer, path, **kwargs):
        events.append("save")
        path.mkdir()
        (path / "complete").write_text("saved")

    def fake_evaluation(*args, **kwargs):
        events.append("evaluate")
        assert args[3] == tmp_path / "evaluation" / "hf"
        if save:
            assert (tmp_path / "checkpoint/complete").exists()
        raise RuntimeError("scorer failure")

    monkeypatch.setattr(loopquant.checkpoint, "save_quantized_checkpoint", fake_save)
    monkeypatch.setattr(cli, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(cli, "preflight_evaluation", lambda *a: pytest.fail("preflight before snapshot"))
    root = Path(__file__).resolve().parents[1]
    argv = ["quantize", "--model", "unused", "--config", str(root / "configs/ouro_1.4b_w4a4.yaml"),
            "--output-dir", str(tmp_path), "--device", "cpu"]
    if save:
        argv.append("--save")
    if evaluate:
        argv.append("--generative")
        with pytest.raises(RuntimeError, match="scorer failure"):
            cli.main(argv)
    else:
        assert cli.main(argv) == 0
    assert events == ["quantize"] + (["save"] if save else []) + (["evaluate"] if evaluate else [])


def test_failed_save_does_not_evaluate(tmp_path, monkeypatch):
    import loopquant.checkpoint
    from loopquant.pipeline import LoopQuantPipeline
    model = SimpleNamespace()
    monkeypatch.setattr(cli, "load_looped_transformer", lambda *a, **k: (model, None))
    monkeypatch.setattr(cli, "load_calibration_data", lambda *a, **k: torch.zeros(128, 2048))
    monkeypatch.setattr(LoopQuantPipeline, "run", lambda *a, **k: SimpleNamespace(
        model=model, allocation=[4] * 4, cldc={}, ilpa={}, lctc={}, stage_order=()))
    monkeypatch.setattr(loopquant.checkpoint, "save_quantized_checkpoint",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("save failure")))
    monkeypatch.setattr(cli, "run_evaluation", lambda *a, **k: pytest.fail("evaluated after save failure"))
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(RuntimeError, match="save failure"):
        cli.main(["quantize", "--model", "unused", "--config", str(root / "configs/ouro_1.4b_w4a4.yaml"),
                  "--output-dir", str(tmp_path), "--device", "cpu", "--save", "--generative"])
