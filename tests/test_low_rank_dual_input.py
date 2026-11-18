"""Dual-input inference, fixed-grid refinement and snapshot compatibility."""

import json

import pytest
import torch
from torch.nn import functional as F

from loopquant.low_rank.modules import LowRankLinear, wrap_linears, prepare_inference
from loopquant.low_rank.quantization import gptq, quantize_activation
from loopquant.low_rank.refinement import refine_quantized
from loopquant.low_rank.rotation import block_hadamard_transform, inverse_block_hadamard_transform
from loopquant.checkpoint import save_quantized_checkpoint, load_quantized_checkpoint
from test_low_rank import configuration, tiny, DATA


@pytest.mark.parametrize("sharing", ["independent", "prefix"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mode", ["full_precision", "quantized"])
def test_forward_uses_the_declared_input(sharing, bits, mode):
    torch.manual_seed(374)
    module = LowRankLinear(torch.nn.Linear(8, 5), num_loops=3, rank_step=1,
                           sharing=sharing, activation_bits=bits, group_size=4,
                           rotation_enabled=True, rotation_block_size=4, low_rank_input=mode)
    for parameter in list(module.A) + list(module.B):
        parameter.data.normal_()
    x = torch.randn(7, 8)
    z = quantize_activation(block_hadamard_transform(x, 4), bits, 4)
    for loop in range(3):
        a, b = module.factors(loop)
        expected = F.linear(z, module.weight, module.bias) + F.linear(F.linear(x if mode == "full_precision" else z, a), b)
        torch.testing.assert_close(module.forward_at_loop(x, loop), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("group_size", [4, None])
def test_refinement_uses_fixed_grids_and_decreases_saved_dtype_objective(dtype, group_size):
    torch.manual_seed(387)
    reference = torch.randn(5, 8, dtype=torch.float64)
    inputs = torch.randn(8, 37, dtype=torch.float64)
    h = inputs @ inputs.T / 37 + 0.02 * torch.eye(8, dtype=torch.float64)
    c = torch.randn_like(reference) * 0.1
    continuous = reference + torch.linalg.solve(h, c.T).T
    inverse = torch.linalg.cholesky(torch.linalg.inv(h), upper=True)
    q, scales = gptq(continuous, h, inverse_factor=inverse.float(), group_size=group_size, return_scales=True)
    q = q.to(dtype)
    result, _ = refine_quantized(q, scales, h, c, reference=reference, bits=4, group_size=group_size, sweeps=2)
    def loss(w):
        delta = w.double() - reference
        return ((delta @ h) * delta).sum() - 2 * (delta * c).sum()
    assert loss(result) <= loss(q) + 1e-12
    same, changes = refine_quantized(q, scales, h, c, reference=reference, bits=4, group_size=group_size, sweeps=0)
    assert changes == 0
    torch.testing.assert_close(same, q, rtol=0, atol=0)


def test_signed_rotation_inverse_is_not_a_second_forward_rotation():
    torch.manual_seed(138)
    x = torch.randn(4, 16, dtype=torch.float64)
    rotated = block_hadamard_transform(x, 8, 19)
    torch.testing.assert_close(inverse_block_hadamard_transform(rotated, 8, 19), x)
    assert not torch.allclose(block_hadamard_transform(rotated, 8, 19), x)


def test_initialization_is_explicit_and_only_quantized_first_is_implemented(tmp_path):
    config = configuration(tmp_path)
    assert config.method["solver"]["initialization"] == "quantized_first"
    config.method["solver"]["initialization"] = "low_rank_first"
    with pytest.raises(ValueError, match="not implemented"):
        config.validate()


def test_old_snapshot_missing_input_semantics_keeps_quantized_branch(tmp_path):
    config = configuration(tmp_path)
    config.method["low_rank_input"] = "quantized"
    model, tokenizer = tiny()
    wrap_linears(model, config)
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            for parameter in list(module.A) + list(module.B):
                parameter.data.normal_(std=0.1)
    model._quantization_algorithm = "low_rank"
    model._low_rank_config = config.to_dict()
    model._loopquant_allocation = (4, 4)
    prepare_inference(model)
    expected = model(DATA).logits.detach()
    directory = tmp_path / "legacy"
    save_quantized_checkpoint(model, tokenizer, directory)
    path = directory / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("low_rank_input")
    method = manifest["quantization_config"]["method"]
    method.pop("low_rank_input")
    method["solver"].pop("initialization")
    method["solver"].pop("coordinate_sweeps")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    restored, _ = load_quantized_checkpoint(directory, device="cpu", num_loops=2)
    assert restored._low_rank_config["method"]["low_rank_input"] == "quantized"
    with torch.no_grad():
        torch.testing.assert_close(restored(DATA).logits, expected, rtol=0, atol=0)
