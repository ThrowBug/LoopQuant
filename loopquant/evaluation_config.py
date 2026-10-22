"""Model-independent evaluation protocol loading and validation."""

from copy import deepcopy
from pathlib import Path
import re

import yaml


GENERATION_DEFAULTS = {
    "batch_size": 1,
    "max_length": 4096,
    "use_cache": True,
    "allow_truncation": False,
    "apply_chat_template": False,
    "fewshot_seed": 0,
    "save_samples": True,
    "confirm_run_unsafe_code": False,
    "tasks": [],
}


def resolve_evaluation_config(config):
    """Resolve only evaluation settings; no model/quantization configuration needed."""
    result = {
        "batch_size": 1, "sequence_length": 2048,
        "limit": None, "seed": 0,
    }
    if not isinstance(config, dict) or set(config) - (set(result) | {"generative"}):
        raise ValueError("unknown evaluation parameters; select suites using CLI flags or suites")
    result.update(deepcopy(config))
    generation = deepcopy(GENERATION_DEFAULTS)
    supplied = result.get("generative", {})
    if not isinstance(supplied, dict) or set(supplied) - set(generation):
        raise ValueError("unknown generative parameters; select suites using CLI flags or suites")
    generation.update(supplied)
    result["generative"] = generation
    limit = result["limit"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("evaluation.limit must be a positive sample/window count or null")
    for section, keys in ((result, ("batch_size", "sequence_length")),
                          (generation, ("batch_size", "max_length"))):
        for key in keys:
            if isinstance(section[key], bool) or not isinstance(section[key], int) or section[key] <= 0:
                raise ValueError(f"evaluation {key} must be a positive integer")
    for key in ("use_cache", "allow_truncation", "apply_chat_template",
                "save_samples", "confirm_run_unsafe_code"):
        if not isinstance(generation[key], bool):
            raise ValueError(f"generative.{key} must be boolean")
    if not isinstance(generation["tasks"], list):
        raise ValueError("generative.tasks must be a list")
    names = set()
    for task in generation["tasks"]:
        if not isinstance(task, dict) or set(task) - {"name", "num_fewshot", "max_gen_toks"}:
            raise ValueError("task entries accept name, num_fewshot, max_gen_toks only")
        name = task.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"invalid or duplicate task name: {name!r}")
        names.add(name)
        shots = task.get("num_fewshot")
        if shots is not None and (isinstance(shots, bool) or not isinstance(shots, int) or shots < 0):
            raise ValueError("num_fewshot must be a nonnegative integer or null")
        tokens = task.get("max_gen_toks")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens < generation["max_length"]:
            raise ValueError("max_gen_toks must be positive and less than generative.max_length")
    return result


def load_evaluation_config(path):
    """Load the standalone evaluation protocol."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation configuration must be a mapping")
    return resolve_evaluation_config(payload.get("evaluation", payload))
