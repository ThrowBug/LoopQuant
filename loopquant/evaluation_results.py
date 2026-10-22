"""Incremental, JSON-safe evaluation artifacts without model dependencies."""

import dataclasses
import json
import math
from pathlib import Path


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_harness_result(directory, result, save_samples):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "samples"}
    save_json(directory / "metrics.json", summary)
    samples_path = directory / "samples.jsonl"
    # Remove stale samples even when a rerun disables sample logging.
    if samples_path.exists():
        samples_path.unlink()
    if save_samples:
        with samples_path.open("w", encoding="utf-8") as stream:
            for task_name, samples in result.get("samples", {}).items():
                for sample in samples:
                    row = {"task": task_name, **sample}
                    stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    return summary
