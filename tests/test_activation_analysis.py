"""FP-only metric identities, aligned collection/cache reading and CLI artifacts."""

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from loopquant import activation_analysis as analysis


def test_identical_and_zero_states():
    torch.manual_seed(8)
    value = torch.randn(2, 5, 7)
    metrics = analysis.layer_metrics(torch.stack([value, value, value]))
    assert torch.tensor(metrics["cosine_similarity"]).allclose(torch.ones(3, 3), atol=1e-6)
    assert all(entry["relative_change"] == 0 and entry["r90"] == 0 for entry in metrics["adjacent"])
    zero = analysis.layer_metrics(torch.zeros(2, 2, 3, 4))
    assert zero["cosine_similarity"] == [[None, None], [None, None]]
    assert zero["cosine_valid_tokens"] == [[0, 0], [0, 0]]
    assert zero["adjacent"][0]["relative_change"] == 0
    assert zero["adjacent"][0]["r90"] == 0


def test_scale_direction_and_sample_mean_are_distinguished():
    base = torch.tensor([[[10., 0.]], [[0., 1.]]])
    changed = base.clone()
    changed[1] *= 2
    metrics = analysis.layer_metrics(torch.stack([base, changed, -changed]))
    assert metrics["cosine_similarity"][0] == pytest.approx([1, 1, -1])
    first, second = metrics["adjacent"]
    # A global Frobenius ratio would be 1/sqrt(101), not the required sample mean.
    assert first["relative_change"] == pytest.approx(0.5)
    assert first["relative_change_by_sample"] == [0, 1]
    assert first["r90"] == 1
    assert second["relative_change"] == pytest.approx(2)


@pytest.mark.parametrize("transpose", [False, True])
def test_r90_known_spectrum_and_scale_invariance(transpose):
    # Singular-value energies are 8,1,1: two directions explain 90%.
    delta = torch.cat([torch.diag(torch.tensor([8.0 ** 0.5, 1., 1.])), torch.zeros(2, 3)])
    if transpose:
        delta = delta.T
    for scale in (1e-8, 1, 1e8):
        assert analysis.increment_r90(delta * scale) == 2


def test_undefined_relative_change_and_invalid_states():
    metrics = analysis.layer_metrics(torch.stack([torch.zeros(1, 2, 3), torch.ones(1, 2, 3)]))
    assert metrics["adjacent"][0]["relative_change"] is None
    assert metrics["adjacent"][0]["relative_change_valid_samples"] == 0
    for states in (torch.zeros(1, 2, 3, 4), torch.zeros(2, 0, 3, 4),
                   torch.full((2, 1, 2, 3), float("nan"))):
        with pytest.raises(ValueError):
            analysis.layer_metrics(states)


@pytest.fixture
def fp_cache(tmp_path):
    # A real pre-existing teacher cache, generated without quantizing the model.
    from loopquant.config import LoopQuantConfig
    from loopquant.low_rank.cache import prepare_teacher_cache
    from loopquant.models import configure_looped_model
    from modeling_tiny_ouro import TinyOuroConfig, TinyOuroForCausalLM
    from test_evaluation import tiny_model_and_tokenizer

    torch.manual_seed(7)
    model = TinyOuroForCausalLM(TinyOuroConfig(total_ut_steps=3, num_hidden_layers=2)).eval()
    configure_looped_model(model, 3, quantization_hooks=False)
    _, tokenizer = tiny_model_and_tokenizer()
    sequences = torch.tensor([[3, 4, 3, 4, 3], [4, 3, 4, 3, 4]])
    config = LoopQuantConfig.from_dict({
        "algorithm": "low_rank", "model_key": "test_fp_analysis", "num_loops": 3,
        "calibration": {"cache_dir": str(tmp_path / "cache"), "batch_size": 2},
        "method": {"rank_step": 1},
    })
    cache = prepare_teacher_cache(model, tokenizer, sequences, config, "cpu", logging.getLogger("test"))
    return model, tokenizer, sequences, cache.directory


def test_collection_matches_existing_cache_without_modifying_model(tmp_path, fp_cache):
    model, _, sequences, cache_dir = fp_cache
    source = analysis.FPTeacherOutputs(cache_dir, 2, 3, seed=2)
    assert source.sample_ids == [0, 1]
    original = {name: value.clone() for name, value in model.state_dict().items()}
    directory = tmp_path / "sampled"
    blocks = analysis.collect_fp_outputs(
        model, sequences, directory, num_loops=3, positions=source.positions,
        batch_size=2, device="cpu", logger=logging.getLogger("test"))
    assert blocks == 2
    for block in range(blocks):
        actual = analysis._read_collected_layer(directory, block, 3, 2, 2)
        assert torch.equal(actual, source.read_layer(block))
        assert actual.shape == (3, 2, 3, 32)
        assert not model.model.layers[block]._forward_hooks
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


def test_cache_checksum_and_loop_count_validation(fp_cache):
    _, _, _, directory = fp_cache
    with pytest.raises(ValueError, match="conflicts"):
        analysis.FPTeacherOutputs(directory, 2, 3, 0, num_loops=4)
    source = analysis.FPTeacherOutputs(directory, 2, 3, 0)
    path = directory / "teacher/block_000/loop_01/shard_00000.safetensors"
    tensors = load_file(path)
    tensors["output"] += 1
    save_file(tensors, path)
    with pytest.raises(ValueError, match="corrupt"):
        source.read_layer(0)


def test_hooks_removed_if_execution_is_incomplete(tmp_path, fp_cache):
    model, _, sequences, _ = fp_cache
    with pytest.raises(ValueError, match="every configured loop"):
        analysis.collect_fp_outputs(model, sequences, tmp_path / "partial", num_loops=4,
                                    positions=[torch.arange(3)] * 2, batch_size=2, device="cpu",
                                    logger=logging.getLogger("test"))
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_cache_cli_artifacts_are_read_only_and_do_not_load_a_model(tmp_path, fp_cache, monkeypatch):
    import loopquant.models
    import loopquant.data
    model, _, _, directory = fp_cache
    monkeypatch.setattr(loopquant.models, "load_looped_transformer", lambda *a, **k: pytest.fail("cache mode loaded model"))
    monkeypatch.setattr(loopquant.data, "load_calibration_data", lambda *a, **k: pytest.fail("cache mode loaded data"))
    before = {path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.rglob("*") if path.is_file()}
    output = tmp_path / "analysis"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    assert analysis.main(["--teacher-cache", str(directory), "--output-dir", str(output),
                          "--num-samples", "2", "--tokens-per-sample", "3", "--device", "cpu"]) == 0
    result = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete" and result["num_loops"] == 3
    assert len(result["layers"]) == 2 and result["source"]["mode"] == "teacher_cache"
    assert result["layers"][0]["shape"] == [3, 2, 3, model.config.hidden_size]
    with (output / "cosine_similarity.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 3 * 3
    with (output / "adjacent_changes.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 2
    for stem in ("cosine_similarity", "adjacent_changes"):
        assert (output / "figures" / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (output / "figures" / f"{stem}.svg").read_text(encoding="utf-8")
    after = {path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in directory.rglob("*") if path.is_file()}
    assert before == after


def test_model_cli_stays_fp_and_cleans_temporary_outputs(tmp_path, fp_cache, monkeypatch):
    import loopquant.models
    import loopquant.data
    model, tokenizer, sequences, _ = fp_cache
    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return model, tokenizer

    monkeypatch.setattr(loopquant.models, "load_looped_transformer", load)
    monkeypatch.setattr(loopquant.data, "load_calibration_data", lambda *a, **kw: sequences)
    output = tmp_path / "model_analysis"
    assert analysis.main(["--model", "original/ouro", "--num-loops", "3", "--output-dir", str(output),
                          "--num-samples", "2", "--tokens-per-sample", "3", "--no-plots", "--device", "cpu"]) == 0
    assert calls[0][1]["algorithm"] != "loopquant" and not calls[0][1]["configure_context"]
    result = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete" and result["source"]["mode"] == "original_model"
    assert not list(output.glob(".fp-outputs-*"))
    assert not (output / "figures").exists()


def test_cli_rejects_quantized_checkpoint_and_cache_writes(tmp_path, fp_cache):
    _, _, _, directory = fp_cache
    with pytest.raises(SystemExit):
        analysis.main(["--teacher-cache", str(directory), "--output-dir", str(directory / "analysis")])
    checkpoint = tmp_path / "quantized"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_manifest.json").write_text("{}")
    with pytest.raises(SystemExit):
        analysis.main(["--model", str(checkpoint), "--num-loops", "4", "--output-dir", str(tmp_path / "out")])
    for args in (["--model", "unused"], ["--model", "unused", "--num-loops", "1"],
                 ["--teacher-cache", "unused", "--num-samples", "0"]):
        with pytest.raises(SystemExit):
            analysis.main([*args, "--output-dir", str(tmp_path / "out")])


def test_analysis_launcher_routes_arguments_and_exit_status(tmp_path):
    bash = os.environ.get("LOOPQUANT_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "launcher repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("analyze_activations.sh", "_launch_common.sh"):
        (root / "scripts" / name).write_text((source / "scripts" / name).read_text(), newline="\n")
    recorder = tmp_path / "record.py"
    recorder.write_text("import json,os,pathlib,sys\npathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\nsys.exit(int(os.environ.get('TEST_EXIT','0')))\n")
    wrapper = tmp_path / "python wrapper.sh"
    wrapper.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$RECORDER" "$@"\n', newline="\n")
    wrapper.chmod(0o755)
    capture = tmp_path / "capture.json"
    env = dict(os.environ, LOOPQUANT_PYTHON=wrapper.as_posix(), REAL_PYTHON=Path(sys.executable).as_posix(),
               RECORDER=recorder.as_posix(), CAPTURE=str(capture))
    env["PATH"] = str(Path(bash).parent) + os.pathsep + env.get("PATH", "")
    command = [bash, (root / "scripts/analyze_activations.sh").as_posix(), "fp experiment",
               "--teacher-cache", (tmp_path / "FP cache").as_posix()]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert completed.returncode == 0, completed.stderr
    args = json.loads(capture.read_text())
    assert args[:3] == ["-s", "-m", "loopquant.activation_analysis"]
    assert args[args.index("--teacher-cache") + 1].replace("\\", "/") == (tmp_path / "FP cache").as_posix()
    assert args[args.index("--output-dir") + 1].replace("\\", "/") == (root / "outputs/fp experiment/activation_analysis").as_posix()
    assert "outputs subdirectory: fp experiment" in completed.stdout
    failed = subprocess.run(command, env={**env, "TEST_EXIT": "7"}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert failed.returncode == 7
