"""CPU arithmetic parity tests; no vLLM or Triton installation is required."""

import pytest
import torch
from torch import nn

from loopquant import vllm_ops
from loopquant.lctc import LoopConditionedCGB, _LCTCRuntime, _activate_runtime
from loopquant.quantization import (
    QUANTIZATION_CONTEXT, QuantizedLinear, _RotatedRMSNorm, fake_quantize_weight, orthogonal_transform,
)
from loopquant.vllm_ops import (
    InferenceLinear, InferenceOrthogonalTransform, InferenceResidualNorm, TransformCache,
    fast_hadamard_reference,
)
from modeling_paged_ouro import TestRMSNorm as ReferenceRMSNorm


@pytest.fixture(autouse=True)
def restore_context():
    previous = vars(QUANTIZATION_CONTEXT).copy()
    yield
    vars(QUANTIZATION_CONTEXT).update(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [32, 64, 96, 160, 192])
def test_cached_transform_preserves_reference_arithmetic(width, dtype, monkeypatch):
    torch.manual_seed(92)
    value = torch.randn(2, 3, width).to(dtype)
    expected = orthogonal_transform(value)
    transform = InferenceOrthogonalTransform(width)
    cache = TransformCache()
    transform.prepare(cache, "cpu")
    sibling = InferenceOrthogonalTransform(width)
    sibling.prepare(cache, "cpu")
    assert sibling.hadamard is transform.hadamard
    assert sibling.small is transform.small
    monkeypatch.setattr(vllm_ops, "_hadamard", lambda *a: pytest.fail("matrix rebuilt during forward"))
    monkeypatch.setattr(vllm_ops, "_small_orthogonal", lambda *a: pytest.fail("matrix rebuilt during forward"))
    torch.testing.assert_close(transform(value), expected, rtol=0, atol=0)
    assert not transform.state_dict()  # Caches do not enter saved checkpoints.


@pytest.mark.parametrize("width", [1, 8, 32, 128, 96, 160])
def test_fast_transform_has_same_normalized_map(width):
    torch.manual_seed(93)
    value = torch.randn(2, 3, width)
    transform = InferenceOrthogonalTransform(width, fast=True)
    transform.prepare(TransformCache(), "cpu")
    torch.testing.assert_close(transform(value), orthogonal_transform(value), rtol=4e-6, atol=4e-6)
    if width & (width - 1) == 0:
        result = fast_hadamard_reference(value)
        torch.testing.assert_close(fast_hadamard_reference(result), value, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_lctc", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_linear_matches_all_loop_parameters_and_shares_weight(dtype, use_lctc, transformed):
    torch.manual_seed(94)
    QUANTIZATION_CONTEXT.configure(4, (8, 4, 8, 4))
    QUANTIZATION_CONTEXT.enabled = True
    module = QuantizedLinear(nn.Linear(96, 64, bias=True).to(dtype), "projection")
    module.weight.data = fake_quantize_weight(module.weight)
    module.set_clip_ratios((0.9, 0.8, 0.7, 0.6))
    if transformed:
        module.set_cldc_scale(torch.linspace(0.8, 1.2, 96).to(dtype))
        module.cldc_hadamard = True
        module.cldc_permutation = torch.randperm(96)
    cgb = LoopConditionedCGB.from_model(nn.Sequential(module), 4) if use_lctc else None
    if cgb is not None:
        with torch.no_grad():
            for spec, entry in cgb.iter_entries():
                entry.clip.copy_(torch.linspace(0.6, 1.0 - spec.loop_id * 0.02, 3))
                entry.gain.copy_(torch.tensor([0.9, 1.1]))
                entry.sigma.copy_(torch.tensor([0.5, 2.0]))
                entry.beta.copy_(torch.tensor([0.01, -0.02]))
    value = torch.randn(2, 3, 96).to(dtype)
    cache = TransformCache()
    for loop, bits in enumerate((8, 4, 8, 4)):
        QUANTIZATION_CONTEXT.configure(4, (8, 4, 8, 4))
        QUANTIZATION_CONTEXT.enabled = True
        QUANTIZATION_CONTEXT.current_loop = loop
        with torch.no_grad(), _activate_runtime(_LCTCRuntime(cgb) if cgb is not None else None):
            expected = module(value)
        plan = InferenceLinear(module, bits, loop, cgb)
        with pytest.raises(RuntimeError, match="prepare"):
            plan(value)
        with torch.no_grad():
            plan.prepare(cache)
        assert plan.linear.weight is module.weight
        QUANTIZATION_CONTEXT.configure(1, (16,))
        QUANTIZATION_CONTEXT.enabled = False
        with torch.no_grad():
            actual = plan(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_refreshes_lctc_constants():
    QUANTIZATION_CONTEXT.configure(4)
    module = QuantizedLinear(nn.Linear(32, 64), "projection")
    cgb = LoopConditionedCGB.from_model(nn.Sequential(module), 4)
    plan = InferenceLinear(module, 4, 0, cgb)
    cache = TransformCache()
    with torch.no_grad():
        plan.prepare(cache)
        assert plan.gain.eq(1).all()
        cgb.entry("projection", 0).gain.fill_(1.1)
        assert plan.gain.eq(1).all()
        plan.prepare(cache)
    assert plan.gain.eq(1.1).all()


def test_residual_norm_preserves_addition_dtype_and_rotated_gamma():
    torch.manual_seed(95)
    norm = ReferenceRMSNorm(32).to(torch.bfloat16)
    norm.weight.data = torch.linspace(0.8, 1.2, 32).to(torch.bfloat16)
    rotation = torch.linalg.qr(torch.randn(32, 32)).Q
    original = _RotatedRMSNorm(norm, rotation)
    prepared = InferenceResidualNorm(original)
    hidden, residual = (torch.randn(1, 3, 32).to(torch.bfloat16) for _ in range(2))
    with torch.no_grad():
        actual, saved = prepared(hidden, residual)
        expected = residual + hidden
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, original(expected), rtol=0, atol=0)


def test_static_linear_is_a_full_tensor_graph_without_context_reads():
    QUANTIZATION_CONTEXT.configure(4)
    module = QuantizedLinear(nn.Linear(96, 64), "projection")
    module.cldc_hadamard = True
    module.set_cldc_scale(torch.linspace(0.8, 1.2, 96))
    module.cldc_permutation = torch.arange(95, -1, -1)
    cgb = LoopConditionedCGB.from_model(nn.Sequential(module), 4)
    plan = InferenceLinear(module, 4, 2, cgb)
    with torch.no_grad():
        plan.prepare(TransformCache())
    # Dynamo capture on CPU, not a claim to run Inductor or CUDA Graph here.
    compiled = torch.compile(plan, backend="eager", fullgraph=True, dynamic=True)
    # Match vLLM's setting: shape guards need no local C++ compiler.
    with torch.no_grad(), torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=False):
        for tokens in (2, 5):
            value = torch.randn(1, tokens, 96)
            torch.testing.assert_close(compiled(value), plan(value), rtol=0, atol=0)
