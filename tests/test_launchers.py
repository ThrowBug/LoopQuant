"""Exercise launcher quoting and exit status without starting model inference."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize("model_source", ["relative", "hf", "absolute", "existing_root",
                                         "existing_repo", "explicit_relative"])
def test_shell_launchers(tmp_path, model_source):
    bash = os.environ.get("LOOPQUANT_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    root = Path(__file__).resolve().parents[1]
    capture = tmp_path / "capture.json"
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['LQ_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ.get('LQ_EXIT', '0')))\n"
    )
    launcher = tmp_path / "python wrapper.sh"
    launcher.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == "-m" ]]; then\n'
        '  exec "$REAL_PYTHON" "$RECORD_SCRIPT" "$@"\n'
        'else\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        'fi\n', newline="\n", encoding="utf-8"
    )
    launcher.chmod(0o755)
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
    run = tmp_path / "model run.yaml"
    run.write_text(yaml.safe_dump({
        "model_path": model_path,
        "num_loops": 4, "output_dir": (tmp_path / "quantized output").as_posix(),
    }))
    protocol = tmp_path / "evaluation protocol.yaml"
    protocol.write_text("evaluation: {}\n")
    env = dict(os.environ, LOOPQUANT_PYTHON=launcher.as_posix(),
               REAL_PYTHON=Path(sys.executable).as_posix(), RECORD_SCRIPT=recorder.as_posix(),
               LOOPQUANT_MODEL_ROOT=tmp_path.as_posix(), LQ_CAPTURE=str(capture),
               LOOPQUANT_EVAL_OUTPUT_DIR=(tmp_path / "bf16 output").as_posix())
    env["PATH"] = str(Path(bash).parent) + os.pathsep + env.get("PATH", "")
    env.pop("USE_VLLM", None)
    for name, command in (("quantize", "quantize"), ("eval", "evaluate")):
        source = run.as_posix() if name == "quantize" else "ByteDance/Ouro-1.4B"
        completed = subprocess.run(
            [bash, (root / "scripts" / f"{name}.sh").as_posix(),
             source, *(["4"] if name == "eval" else []), protocol.as_posix()], env=env, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stderr
        args = json.loads(capture.read_text())
        assert args[:3] == ["-m", "loopquant.cli", command]
        assert args[args.index("--eval-config") + 1].replace("\\", "/").endswith("evaluation protocol.yaml")
        assert all(flag in args for flag in ("--wikitext2", "--c4", "--zero-shot"))
        if name == "quantize":
            assert "--vllm" not in args
            assert "--generative" not in args
            assert args[args.index("--model") + 1].replace("\\", "/") == expected_model
            assert "--save" in args
        else:
            assert "--vllm" in args  # eval.sh defaults to vLLM.
            assert "--generative" in args
            assert args[args.index("--model") + 1] == "ByteDance/Ouro-1.4B"
            assert args[args.index("--num-loops") + 1] == "4"
            assert "--save" not in args
    env["USE_VLLM"] = "1"
    enabled = subprocess.run([bash, (root / "scripts/eval.sh").as_posix(),
                              "ByteDance/Ouro-1.4B", "4", protocol.as_posix()], env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert enabled.returncode == 0, enabled.stderr
    assert "--vllm" in json.loads(capture.read_text())
    env["USE_VLLM"] = "0"
    disabled = subprocess.run([bash, (root / "scripts/eval.sh").as_posix(),
                               "ByteDance/Ouro-1.4B", "4", protocol.as_posix()], env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert disabled.returncode == 0, disabled.stderr
    assert "--vllm" not in json.loads(capture.read_text())
    env["LQ_EXIT"] = "7"
    failed = subprocess.run([bash, (root / "scripts/eval.sh").as_posix(),
                             "ByteDance/Ouro-1.4B", "4", protocol.as_posix()], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert failed.returncode == 7
