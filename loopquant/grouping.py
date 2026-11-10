"""Shared group-size semantics for weight, activation, and LCTC paths."""

from __future__ import annotations

import math
from typing import TypeAlias


GroupSize: TypeAlias = int | None


def validate_group_size(value: object, *, name: str = "group_size") -> GroupSize:
    """Validate a public group size while preserving ``None`` as a sentinel."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return int(value)


def effective_group_size(group_size: GroupSize, in_features: int) -> int:
    """Resolve null to one group spanning a layer's input width."""
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    validate_group_size(group_size)
    value = in_features if group_size is None else int(group_size)
    if value <= 0:
        raise ValueError("effective group size must be positive")
    return value


def validate_input_width(in_features: int, group_size: GroupSize, *, name: str = "input width") -> None:
    """Require exact groups for integer group sizes; null is always exact."""
    if in_features <= 0:
        raise ValueError(f"{name} must be positive")
    validate_group_size(group_size)
    if group_size is not None and in_features % group_size:
        raise ValueError(f"{name} {in_features} is not divisible by group_size {group_size}")


def activation_group_count(in_features: int, group_size: GroupSize) -> int:
    """Number of activation groups in one token vector."""
    validate_input_width(in_features, group_size)
    return 1 if group_size is None else in_features // group_size


def output_group_count(out_features: int, in_features: int, group_size: GroupSize) -> int:
    """Number of LCTC output groups.

    Null means one gain/bias/sigma value per output channel. Integer sizes retain
    the original group-wise output calibration behavior.
    """
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    validate_input_width(in_features, group_size)
    return out_features if group_size is None else math.ceil(out_features / group_size)

