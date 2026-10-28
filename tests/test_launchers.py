"""Exercise launcher routing, quoting and exit status without model inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def launchers(tmp_path):
    bash = os.environ.get("LOOPQUANT_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "launcher repo"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "tests").mkdir()
    for name in ("quantize.sh", "eval.sh", "_launch_common.sh"):
        (root / "scripts" / name).write_text(
            (source / "scripts" / name).read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n",
        )
    (root / "configs/evaluation_generative.yaml").write_text("evaluation: {}\n")
    capture = tmp_path / "capture.json"
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['LQ_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ.get('LQ_EXIT', '0')))\n"
    )
    wrapper = tmp_path / "python wrapper.sh"
    wrapper.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == "-m" || ( "$1" == "-s" && "${2:-}" == "-m" ) ]]; then\n'
        '  exec "$REAL_PYTHON" "$RECORD_SCRIPT" "$@"\n'
        'else\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        'fi\n', newline="\n", encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env = dict(os.environ, LOOPQUANT_PYTHON=wrapper.as_posix(),
               REAL_PYTHON=Path(sys.executable).as_posix(), RECORD_SCRIPT=recorder.as_posix(),
               LOOPQUANT_MODEL_ROOT=tmp_path.as_posix(), LQ_CAPTURE=str(capture))
    env["PATH"] = str(Path(bash).parent) + os.pathsep + env.get("PATH", "")
    for name in ("USE_VLLM", "LOOPQUANT_EVAL_CONFIG", "LOOPQUANT_EVAL_OUTPUT_DIR"):
        env.pop(name, None)

    def run(name, *args, **overrides):
        capture.unlink(missing_ok=True)
        completed = subprocess.run(
            [bash, (root / "scripts" / f"{name}.sh").as_posix(), *map(str, args)],
            env={**env, **overrides}, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        recorded = json.loads(capture.read_text()) if capture.exists() else None
        return completed, recorded

    return root, run


def option(args, name):
    return args[args.index(name) + 1].replace("\\", "/")


@pytest.mark.parametrize("model_source", ["relative", "hf", "absolute", "existing_root",
                                         "existing_repo", "explicit_relative"])
def test_quantize_model_resolution_and_tagged_output(tmp_path, launchers, model_source):
    root, run = launchers
    model_path = "models/model with spaces"
    expected_model = (tmp_path / model_path).as_posix()
    if model_source == "hf":
        model_path = expected_model = "ByteDance/Ouro-1.4B"
    elif model_source == "absolute":
        model_path = expected_model = (tmp_path / "absolute model").as_posix()
    elif model_source == "existing_root":
        model_path = "local namespace/model with spaces"
        (tmp_path / model_path).mkdir(parents=True)
        expected_model = (tmp_path / model_path).as_posix()
    elif model_source == "existing_repo":
        model_path = expected_model = "tests"
    elif model_source == "explicit_relative":
        model_path = "./missing model"
        expected_model = tmp_path.as_posix() + "/./missing model"
    config = tmp_path / "model run.yaml"
    config.write_text(yaml.safe_dump({
        "model_path": model_path, "num_loops": 4, "output_dir": "ignored/by/launcher",
    }))
    protocol = tmp_path / "evaluation protocol.yaml"
    protocol.write_text("evaluation: {}\n")
    completed, args = run("quantize", config.as_posix(), "--tag", "init only",
                          "--eval-config", protocol.as_posix())
    assert completed.returncode == 0, completed.stderr
    assert args[:4] == ["-s", "-m", "loopquant.cli", "quantize"]
    assert option(args, "--model") == expected_model
    assert option(args, "--output-dir") == (root / "outputs/model run_init only").as_posix()
    assert option(args, "--eval-config") == protocol.as_posix()
    assert all(flag in args for flag in ("--save", "--wikitext2", "--c4", "--zero-shot"))
    assert "--vllm" not in args and "--generative" not in args


@pytest.mark.parametrize("tag_args", [[], ["--tag", ""]])
def test_quantize_without_tag_and_legacy_protocol(tmp_path, launchers, tag_args):
    root, run = launchers
    config = tmp_path / "model.yaml"
    # Launchers no longer require the legacy output_dir field.
    config.write_text("model_path: ByteDance/Ouro-1.4B\nnum_loops: 4\n")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("evaluation: {}\n")
    completed, args = run("quantize", config.as_posix(), protocol.as_posix(), *tag_args)
    assert completed.returncode == 0, completed.stderr
    assert option(args, "--output-dir") == (root / "outputs/model").as_posix()
    assert option(args, "--eval-config") == protocol.as_posix()


@pytest.mark.parametrize("use_vllm,backend", [(None, "vllm"), ("1", "vllm"), ("0", "hf")])
def test_checkpoint_eval_uses_manifest_and_backend_directory(launchers, use_vllm, backend):
    root, run = launchers
    checkpoint = root / "outputs/model_tag/checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"num_loops": 3}))
    overrides = {} if use_vllm is None else {"USE_VLLM": use_vllm}
    completed, args = run("eval", "model_tag", **overrides)
    assert completed.returncode == 0, completed.stderr
    assert args[:4] == ["-s", "-m", "loopquant.cli", "evaluate"]
    assert option(args, "--model") == checkpoint.as_posix()
    assert option(args, "--num-loops") == "3"
    assert option(args, "--output-dir") == (root / f"outputs/model_tag/evaluation/{backend}").as_posix()
    assert ("--vllm" in args) == (backend == "vllm")
    assert all(flag in args for flag in ("--wikitext2", "--c4", "--zero-shot", "--generative"))
    assert "--save" not in args


@pytest.mark.parametrize("local", [False, True])
def test_bf16_eval_and_protocol_override(tmp_path, launchers, local):
    root, run = launchers
    model = (tmp_path / "original model").as_posix() if local else "ByteDance/Ouro-1.4B"
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("evaluation: {}\n")
    completed, args = run("eval", "model_bf16", "--model", model, "--num-loops", "4",
                          LOOPQUANT_EVAL_CONFIG=protocol.as_posix(), USE_VLLM="0")
    assert completed.returncode == 0, completed.stderr
    assert option(args, "--model") == model
    assert option(args, "--num-loops") == "4"
    assert option(args, "--eval-config") == protocol.as_posix()
    assert option(args, "--output-dir") == (root / "outputs/model_bf16/evaluation/hf").as_posix()
    assert not (root / "outputs/model_bf16/checkpoint").exists()
    completed, args = run("eval", "model_bf16", "--model", model, "--num-loops", "4",
                          "--eval-config", protocol.as_posix(), LQ_EXIT="7")
    assert completed.returncode == 7
    assert option(args, "--eval-config") == protocol.as_posix()


@pytest.mark.parametrize("manifest", [None, "not json", "{}", '{"num_loops": true}',
                                     '{"num_loops": 0}', '{"num_loops": "4"}'])
def test_eval_rejects_missing_or_invalid_checkpoint(launchers, manifest):
    root, run = launchers
    if manifest is not None:
        checkpoint = root / "outputs/model/checkpoint"
        checkpoint.mkdir(parents=True)
        (checkpoint / "checkpoint_manifest.json").write_text(manifest)
    completed, args = run("eval", "model")
    assert completed.returncode != 0 and args is None
    assert "manifest" in completed.stderr
    assert not (root / "outputs/model/evaluation").exists()


@pytest.mark.parametrize("name,args", [
    ("eval", []), ("eval", ["../outside"]), ("eval", ["..\\outside"]),
    ("eval", ["."]), ("eval", ["model", "--model", "hf/model"]),
    ("eval", ["model", "--num-loops", "4"]),
    ("eval", ["model", "--model", "hf/model", "--num-loops", "0"]),
    ("eval", ["model", "--eval-config"]), ("eval", ["model", "--unknown"]),
    ("quantize", []), ("quantize", ["model.yaml", "--tag"]),
    ("quantize", ["model.yaml", "--tag", "../outside"]),
    ("quantize", ["model.yaml", "--unknown"]),
])
def test_invalid_launcher_arguments(launchers, name, args):
    _, run = launchers
    completed, recorded = run(name, *args)
    assert completed.returncode == 2 and recorded is None
