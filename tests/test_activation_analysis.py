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


def test_channel_scale_sign_and_zero_channels():
    base = torch.tensor([[[1., 2., 0., 0.], [-1., 2., 0., 0.]]])
    channels = analysis.layer_metrics(torch.stack([base, base * 2, -base * 2]),
                                      channel_topk_ratio=0.25)["channels"]
    assert torch.tensor(channels["rms"]).allclose(torch.tensor([[1., 2., 0., 0.],
                                                               [2., 4., 0., 0.], [2., 4., 0., 0.]]))
    for share in channels["energy_share"]:
        assert share == pytest.approx([0.2, 0.8, 0, 0])
    assert channels["adjacent_cosine"] == [[1, 1, None, None], [-1, -1, None, None]]
    assert channels["topk_k"] == 1 and channels["topk_indices"] == [[1], [1], [1]]
    assert channels["topk_overlap"] == [1, 1]


def test_channel_energy_migration_uses_ceil_rank_and_preserves_ids():
    states = torch.tensor([[[[3., 2., 1.], [3., 2., 1.]]],
                           [[[1., 2., 3.], [1., 2., 3.]]]])
    channels = analysis.layer_metrics(states, channel_topk_ratio=0.5)["channels"]
    assert channels["topk_k"] == 2  # ceil(3*0.5), rather than truncating to one.
    assert channels["topk_indices"] == [[0, 1], [2, 1]]
    assert channels["topk_overlap"] == [0.5]
    assert channels["adjacent_cosine"][0] == pytest.approx([1, 1, 1])
    assert channels["energy_share"][0] == pytest.approx([9 / 14, 4 / 14, 1 / 14])
    assert channels["energy_share"][1] == pytest.approx([1 / 14, 4 / 14, 9 / 14])


def test_channel_response_can_change_at_constant_rms():
    states = torch.tensor([[[[1., 0.], [0., 1.]]], [[[0., 1.], [1., 0.]]]])
    channels = analysis.layer_metrics(states)["channels"]
    assert channels["rms"][0] == channels["rms"][1]
    assert channels["adjacent_cosine"] == [[0, 0]]


def test_channel_ties_and_zero_total_energy():
    tied = analysis.layer_metrics(torch.ones(2, 1, 3, 4), channel_topk_ratio=0.5)["channels"]
    assert tied["topk_indices"] == [[0, 1], [0, 1]]
    assert tied["topk_overlap"] == [1]
    zero = analysis.layer_metrics(torch.zeros(2, 1, 3, 4))["channels"]
    assert zero["rms"] == [[0] * 4] * 2
    assert zero["energy_share"] == [[None] * 4] * 2
    assert zero["adjacent_cosine"] == [[None] * 4]
    assert zero["topk_indices"] == [None, None] and zero["topk_overlap"] == [None]


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_channel_ratio_validation(ratio):
    with pytest.raises(ValueError, match="channel_topk_ratio"):
        analysis.layer_metrics(torch.ones(2, 1, 2, 3), channel_topk_ratio=ratio)


def test_channel_plot_layer_selection():
    assert analysis.select_plot_layers(24) == [1, 13, 24]
    assert analysis.select_plot_layers(2) == [1, 2]
    assert analysis.select_plot_layers(1) == [1]
    assert analysis.select_plot_layers(5, [4, 1, 4]) == [4, 1]
    for requested in ([], [0], [25]):
        with pytest.raises(ValueError, match="plot-layers"):
            analysis.select_plot_layers(24, requested)


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
        assert analysis.layer_metrics(actual)["channels"] == analysis.layer_metrics(source.read_layer(block))["channels"]
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
    (output / "figures").mkdir(parents=True)
    for suffix in (".png", ".svg"):
        (output / "figures" / f"cosine_similarity{suffix}").write_bytes(b"obsolete plot")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    assert analysis.main(["--teacher-cache", str(directory), "--output-dir", str(output),
                          "--num-samples", "2", "--tokens-per-sample", "3", "--device", "cpu",
                          "--plot-layers", "2", "--channel-topk-ratio", "0.1"]) == 0
    result = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete" and result["num_loops"] == 3
    assert len(result["layers"]) == 2 and result["source"]["mode"] == "teacher_cache"
    assert result["plot_layers"] == [2]  # Statistics still include every layer.
    assert result["layers"][0]["shape"] == [3, 2, 3, model.config.hidden_size]
    with (output / "cosine_similarity.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 3 * 3
    with (output / "adjacent_changes.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 2
    with (output / "channel_statistics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * 3 * model.config.hidden_size
        assert {int(row["channel"]) for row in rows} == set(range(model.config.hidden_size))
        assert all(row["cosine_with_previous_loop"] == "" for row in rows if row["loop"] == "1")
        assert all(row["cosine_with_previous_loop"] != "" for row in rows if row["loop"] != "1")
    with (output / "channel_overlap.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * 2 and all(row["topk_k"] == "4" for row in rows)
    for stem in ("adjacent_changes", "channel_rms", "channel_changes"):
        assert (output / "figures" / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (output / "figures" / f"{stem}.svg").read_text(encoding="utf-8")
    assert {path.name for path in (output / "figures").iterdir()} == {
        "adjacent_changes.png", "adjacent_changes.svg", "channel_rms.png", "channel_rms.svg",
        "channel_changes.png", "channel_changes.svg"}
    after = {path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in directory.rglob("*") if path.is_file()}
    assert before == after


def test_wide_rms_plot_preserves_a_single_bright_channel(tmp_path, monkeypatch):
    import numpy as np
    plt = analysis._plotting()
    rms = np.full((4, 2048), 0.1)
    peak = 173
    rms[:, peak] = 100
    result = {"num_loops": 4, "plot_layers": [1], "layers": [{"layer": 1, "channels": {
        "rms": rms.tolist(), "adjacent_cosine": np.ones((3, 2048)).tolist(),
        "topk_overlap": [1, 1, 1], "topk_ratio": 0.01, "topk_k": 21,
    }}]}
    checked = []

    def inspect_figure(figure, path):
        if path.name != "channel_rms":
            return
        figure.canvas.draw()
        axis = figure.axes[0]
        assert axis.get_window_extent().width >= 2 * rms.shape[1]
        pixels = np.asarray(figure.canvas.buffer_rgba())
        x, y = axis.transData.transform((peak, 1))
        actual = pixels[pixels.shape[0] - 1 - int(y), int(x), :3] / 255
        expected = plt.get_cmap("viridis")(1.0)[:3]
        assert np.allclose(actual, expected, atol=0.01)
        checked.append(path.name)

    monkeypatch.setattr(analysis, "_save_figure", inspect_figure)
    analysis._plot_channels(result, tmp_path, plt)
    assert checked == ["channel_rms"]


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
    assert result["plot_layers"] == [1, 2]
    assert (output / "channel_statistics.csv").is_file() and (output / "channel_overlap.csv").is_file()
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
                 ["--teacher-cache", "unused", "--num-samples", "0"],
                 ["--teacher-cache", "unused", "--channel-topk-ratio", "0"],
                 ["--teacher-cache", "unused", "--channel-topk-ratio", "nan"],
                 ["--teacher-cache", "unused", "--plot-layers", "0"]):
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
               "--teacher-cache", (tmp_path / "FP cache").as_posix(),
               "--channel-topk-ratio", "0.05", "--plot-layers", "2", "5"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert completed.returncode == 0, completed.stderr
    args = json.loads(capture.read_text())
    assert args[:3] == ["-s", "-m", "loopquant.activation_analysis"]
    assert args[args.index("--channel-topk-ratio") + 1] == "0.05"
    assert args[args.index("--plot-layers") + 1:args.index("--plot-layers") + 3] == ["2", "5"]
    assert args[args.index("--teacher-cache") + 1].replace("\\", "/") == (tmp_path / "FP cache").as_posix()
    assert args[args.index("--output-dir") + 1].replace("\\", "/") == (root / "outputs/fp experiment/activation_analysis").as_posix()
    assert "outputs subdirectory: fp experiment" in completed.stdout
    failed = subprocess.run(command, env={**env, "TEST_EXIT": "7"}, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert failed.returncode == 7
