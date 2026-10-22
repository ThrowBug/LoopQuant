"""Configuration loading and validation for LoopQuant."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "model_family": "ouro",
    "model_variant": None,
    "model_key": None,
    "model_path": None,
    "quantization": {
        "weight_bits": 4,
        "group_size": 32,
        "activation_setting": "w4a4",
        "activation_candidates": [4, 8],
        "average_activation_bits": 6.0,
    },
    "calibration": {
        "dataset": "wikitext2",
        "split": "train",
        "num_samples": 128,
        "sequence_length": 2048,
        "seed": 0,
    },
    "method": {
        "cldc": {
            "enabled": True,
            "implementation": "learned_rotation_group_aware_v2",
            "learnable_rotation": True,
            "scaling_enabled": True,
            "rotation_enabled": True,
            "reordering_enabled": True,
            "rotation_block_size": 32,
            "reordering": "group_aware",
            "reorder_max_passes": 3,
            "optimization_steps": 80,
            "learning_rate": 1.0e-2,
            "robust_temperature": 0.1,
            "lambda_sync": 0.1,
            "lambda_reg": 1.0e-4,
            "rotation_seed": 42,
            "statistics_sequences": 16,
        },
        "ilpa": {"enabled": True, "selection_samples": 32, "interaction_shrinkage": 0.5},
        "lctc": {
            "enabled": True,
            "updates": 192,
            "learning_rate": 1.0e-3,
            "task_objective_weight": 0.03,
        },
    },
    "output_dir": "outputs/loopquant",
}


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"unknown configuration parameter: {key}")
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class LoopQuantConfig:
    """Compact configuration with dict sections matching the public YAML files."""

    num_loops: int
    model_family: str = "ouro"
    model_variant: str | None = None
    model_key: str | None = None
    model_path: str | None = None
    quantization: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULTS["quantization"]))
    calibration: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULTS["calibration"]))
    method: dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULTS["method"]))
    output_dir: str = "outputs/loopquant"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LoopQuantConfig":
        if "num_loops" not in values:
            raise ValueError("num_loops is required in the quantization configuration")
        merged = _merge(DEFAULTS, {key: value for key, value in values.items() if key != "num_loops"})
        config = cls(num_loops=values["num_loops"], **merged)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoopQuantConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "model_key": self.model_key,
            "model_path": self.model_path,
            "num_loops": self.num_loops,
            "quantization": deepcopy(self.quantization),
            "calibration": deepcopy(self.calibration),
            "method": deepcopy(self.method),
            "output_dir": self.output_dir,
        }

    def validate(self) -> None:
        if isinstance(self.num_loops, bool) or not isinstance(self.num_loops, int) or self.num_loops <= 0:
            raise ValueError("num_loops must be a positive integer")
        if self.model_family.lower() not in {"ouro", "nanbeige", "huginn"}:
            raise ValueError("model_family must be ouro, nanbeige, or huginn")
        expected_loops = {"ouro": 4, "nanbeige": 2, "huginn": 6}[self.model_family.lower()]
        if int(self.num_loops) != expected_loops:
            raise ValueError(
                f"{self.model_family} formal runs require num_loops={expected_loops}"
            )
        q = self.quantization
        if int(q["weight_bits"]) != 4 or int(q["group_size"]) != 32:
            raise ValueError("the validated public path is W4 with group size 32")
        setting = str(q["activation_setting"]).lower()
        if setting not in {"w4a4", "w4a6"}:
            raise ValueError("activation_setting must be w4a4 or w4a6")
        if setting == "w4a6":
            candidates = tuple(int(x) for x in q["activation_candidates"])
            if candidates != (4, 8) or float(q["average_activation_bits"]) != 6.0:
                raise ValueError("W4A6 requires average activation precision 6 with A4/A8 ILPA candidates")
        cldc = self.method["cldc"]
        scaling = bool(cldc.get("scaling_enabled", True))
        rotation = bool(cldc.get("rotation_enabled", True))
        reordering = bool(cldc.get("reordering_enabled", True))
        if reordering and not (scaling and rotation):
            raise ValueError("CLDC Pi/reordering requires both D/scaling and R/rotation")
        if rotation and not bool(cldc.get("learnable_rotation", False)):
            raise ValueError("CLDC R requires learnable_rotation=true")
        if self.method["lctc"].get("enabled", True) and not self.method["ilpa"].get("enabled", True):
            raise ValueError("ILPA must run before LCTC")
        calibration = self.calibration
        if str(calibration["dataset"]).lower() not in {"wikitext2", "wikitext-2"}:
            raise ValueError("the validated calibration dataset is WikiText-2")
        if int(calibration["num_samples"]) <= 0 or int(calibration["sequence_length"]) <= 1:
            raise ValueError("calibration sample count and sequence length must be positive")

    def apply_overrides(self, **values: Any) -> None:
        for key, value in values.items():
            if value is None:
                continue
            section, _, name = key.partition("__")
            if name:
                getattr(self, section)[name] = value
            else:
                setattr(self, key, value)
        self.validate()
