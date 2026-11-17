"""Blocked quantization versus an explicit conditional least-squares oracle."""

import pytest
import torch

from loopquant.low_rank.quantization import gptq
from loopquant.low_rank.residual_compensation import residual_compensation


def reference_scale(group):
    """Search the public clipping grid one candidate at a time."""
    group = group.float()
    base = group.abs().amax(1).clamp_min(1e-8) / 7
    scales, losses = [], []
    for ratio in torch.linspace(1.0, 0.8, 20):
        scale = base * ratio
        rounded = (group / scale[:, None]).round().clamp(-7, 7) * scale[:, None]
        scales.append(scale)
        losses.append((rounded - group).square().mean(1))
    grid = torch.stack(scales, 1)
    return grid.gather(1, torch.stack(losses, 1).argmin(1, keepdim=True)).squeeze(1)


def sequential_oracle(weight, gram, cross, group_size):
    """Round, then solve the normal equations for *all* free columns.

    Starts on W0's grid. With cross=W0 H this is ordinary GPTQ; otherwise
    the first fixed column triggers the asymmetric residual compensation.
    No inverse Cholesky factors or blocked feedback are used here.
    """
    width = weight.shape[1]
    group_size = group_size or width
    work, result = weight.double().clone(), torch.empty_like(weight)
    for column in range(width):
        if column % group_size == 0:
            scale = reference_scale(work[:, column:column + group_size])
        result[:, column] = (work[:, column].float() / scale).round().clamp(-7, 7) * scale
        if column + 1 < width:
            remaining = slice(column + 1, width)
            rhs = cross[:, remaining] - result[:, :column + 1].double() @ gram[:column + 1, remaining]
            work[:, remaining] = torch.linalg.solve(gram[remaining, remaining], rhs.T).T
    return result


@pytest.mark.parametrize("method", ["gptq", "residual_compensation"])
@pytest.mark.parametrize("group_size", [4, None])
@pytest.mark.parametrize("block_size", [1, 3, 4, 6, 16, 32])
def test_group_grids_use_fully_updated_free_weights(method, group_size, block_size):
    torch.manual_seed(43)
    weight = torch.randn(3, 16) * 0.1
    z = torch.randn(16, 64, dtype=torch.float64)
    gram = z @ z.T / 64 + 0.03 * torch.eye(16, dtype=torch.float64)
    inverse = torch.linalg.cholesky(torch.linalg.inv(gram), upper=True)
    residual = torch.zeros_like(weight).double()
    if method == "residual_compensation":
        residual = torch.randn_like(residual) * 0.03
        actual = residual_compensation(weight, residual, inverse_factor=inverse,
                                       group_size=group_size, block_size=block_size)
    else:
        actual = gptq(weight, gram, inverse_factor=inverse.float(),
                      group_size=group_size, block_size=block_size)
    expected = sequential_oracle(weight, gram, weight.double() @ gram + residual, group_size)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("method", ["gptq", "residual_compensation"])
@pytest.mark.parametrize("block_size", [16, 32, 48, 64, 128])
def test_g32_results_match_sequential_quantization_for_arbitrary_blocks(method, block_size):
    torch.manual_seed(71)
    weight = torch.randn(3, 96) * 0.1
    z = torch.randn(96, 192, dtype=torch.float64)
    gram = z @ z.T / 192 + 0.01 * torch.eye(96, dtype=torch.float64)
    inverse = torch.linalg.cholesky(torch.linalg.inv(gram), upper=True)
    residual = torch.randn_like(weight).double() * 0.005
    if method == "gptq":
        expected = gptq(weight, gram, inverse_factor=inverse.float(), group_size=32, block_size=1)
        actual = gptq(weight, gram, inverse_factor=inverse.float(), group_size=32, block_size=block_size)
    else:
        expected = residual_compensation(weight, residual, inverse_factor=inverse, group_size=32, block_size=1)
        actual = residual_compensation(weight, residual, inverse_factor=inverse,
                                       group_size=32, block_size=block_size)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
