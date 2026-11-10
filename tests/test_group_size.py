"""Contracts for integer and full-width/null group-size modes."""

import pytest
import torch
from torch import nn

from loopquant.cli import build_parser
from loopquant.config import LoopQuantConfig
from loopquant.grouping import activation_group_count, effective_group_size, validate_input_width
from loopquant.lctc import LoopConditionedCGB
from loopquant.low_rank.quantization import gptq, quantize_activation
from loopquant.low_rank.residual_compensation import residual_compensation
from loopquant.quantization import QuantizedLinear, fake_quantize_groupwise


def test_null_resolves_to_one_full_input_group():
    assert effective_group_size(None, 8) == 8
    assert activation_group_count(8, None) == 1
    validate_input_width(8, None)
    with pytest.raises(ValueError, match="not divisible"):
        validate_input_width(10, 4)


def test_cli_null_override_is_preserved():
    parser = build_parser()
    args = parser.parse_args([
        "quantize", "--model", "unused", "--config", "unused", "--output-dir", "unused",
        "--group-size", "null",
    ])
    config = LoopQuantConfig.from_dict({"num_loops": 4})
    config.apply_overrides(quantization__group_size=args.group_size)
    assert config.quantization["group_size"] is None


@pytest.mark.parametrize("width,group_size", [(8, 4), (128, 128), (8, None)])
def test_fake_quant_and_low_rank_quantizers_accept_supported_group_sizes(width, group_size):
    value = torch.randn(3, width)
    assert fake_quantize_groupwise(value, 4, group_size).shape == value.shape
    assert quantize_activation(value, 4, group_size).shape == value.shape


def test_lctc_null_has_channelwise_output_affines():
    wrapped = QuantizedLinear(nn.Linear(8, 6), "q_proj", None)
    cgb = LoopConditionedCGB.from_model(nn.Sequential(wrapped), 2, group_size=None)
    spec, entry = next(cgb.iter_entries())
    assert spec.input_groups == 1
    assert spec.output_groups == 6
    assert spec.output_group_size == 1
    assert tuple(entry.clip.shape) == (1,)
    assert tuple(entry.gain.shape) == (6,)
    assert tuple(entry.beta.shape) == (6,)
    assert tuple(entry.sigma.shape) == (6,)


def test_lctc_integer_group_size_uses_input_groups_and_legacy_output_groups():
    wrapped = QuantizedLinear(nn.Linear(128, 64), "q_proj", 128)
    cgb = LoopConditionedCGB.from_model(nn.Sequential(wrapped), 1, group_size=128)
    spec, entry = next(cgb.iter_entries())
    assert spec.input_groups == 1
    assert spec.output_groups == 1
    assert spec.output_group_size == 128
    assert tuple(entry.clip.shape) == (1,)
    assert tuple(entry.gain.shape) == (1,)


def test_low_rank_weight_updates_treat_null_as_one_row_group():
    weight = torch.randn(4, 8)
    hessian = torch.eye(8)
    quantized = gptq(weight, hessian, group_size=None)
    compensated = residual_compensation(
        weight, torch.zeros_like(weight), inverse_factor=torch.eye(8), group_size=None
    )
    assert quantized.shape == weight.shape
    assert compensated.shape == weight.shape
    with pytest.raises(ValueError, match="not divisible"):
        gptq(weight, hessian, group_size=3)
