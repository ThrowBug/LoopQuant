"""Structured online rotations used by the low-rank fake-quantized path."""

from __future__ import annotations

from functools import lru_cache
import math

import torch


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_block_hadamard(width: int, block_size: int) -> None:
    """Validate a block-Hadamard transform for a linear input width."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or not _is_power_of_two(block_size):
        raise ValueError("rotation.block_size must be a positive power of two")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("rotation width must be a positive integer")
    if width % block_size:
        raise ValueError(
            f"rotation.block_size={block_size} must divide the linear input width={width}"
        )


@lru_cache(maxsize=16)
def _hadamard_cpu(block_size: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < block_size:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1),
                            torch.cat((matrix, -matrix), dim=1)), dim=0)
    return matrix / math.sqrt(block_size)


@lru_cache(maxsize=64)
def _signs_cpu(width: int, block_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
    return signs.mul(2).sub(1).reshape(width // block_size, block_size).to(torch.float32)


def _device_key(device: torch.device) -> tuple[str, int]:
    return device.type, -1 if device.index is None else int(device.index)


@lru_cache(maxsize=64)
def _hadamard_device(block_size: int, device_type: str, device_index: int,
                     dtype: torch.dtype) -> torch.Tensor:
    device = torch.device(device_type, None if device_index < 0 else device_index)
    return _hadamard_cpu(block_size).to(device=device, dtype=dtype)


@lru_cache(maxsize=128)
def _signs_device(width: int, block_size: int, seed: int, device_type: str,
                 device_index: int, dtype: torch.dtype) -> torch.Tensor:
    device = torch.device(device_type, None if device_index < 0 else device_index)
    return _signs_cpu(width, block_size, seed).to(device=device, dtype=dtype)


def block_hadamard_transform(value: torch.Tensor, block_size: int, seed: int = 42) -> torch.Tensor:
    """Apply a deterministic signed block-Hadamard transform on the last dimension.

    The implemented matrix is block diagonal with blocks ``H @ diag(signs)``,
    where ``H`` is a normalized Hadamard matrix. For row-vector inputs this is
    exactly ``value @ R``. The transform is evaluated in FP32 (or FP64 for FP64
    inputs) and returned in the input dtype.
    """
    if value.ndim == 0:
        raise ValueError("rotation input must have at least one dimension")
    width = int(value.shape[-1])
    validate_block_hadamard(width, block_size)
    compute_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
    shape = value.shape
    grouped = value.to(dtype=compute_dtype).reshape(-1, width // block_size, block_size)
    device_type, device_index = _device_key(value.device)
    hadamard = _hadamard_device(block_size, device_type, device_index, compute_dtype)
    signs = _signs_device(width, block_size, int(seed), device_type, device_index, compute_dtype)
    transformed = torch.matmul(grouped, hadamard.T) * signs.unsqueeze(0)
    return transformed.reshape(shape).to(dtype=value.dtype)


def rotate_weight(weight: torch.Tensor, block_size: int, seed: int = 42) -> torch.Tensor:
    """Rotate a ``[out_features, in_features]`` weight on its input columns."""
    if weight.ndim != 2:
        raise ValueError("linear weight must be a matrix")
    return block_hadamard_transform(weight, block_size, seed)

