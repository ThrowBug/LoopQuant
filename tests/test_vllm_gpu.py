"""Opt-in smoke test against the actual vLLM CUDA engine, with no dataset downloads.

LOOPQUANT_VLLM_TEST_MODEL=/models/Ouro-or-loopquant python -m pytest tests/test_vllm_gpu.py
Requires requirements-vllm.txt and pytest in the active Linux/CUDA environment.
Set LOOPQUANT_VLLM_COMPARE_HF=1 to also compare teacher-forced scores with HF.
"""

import math
import os
import json
from pathlib import Path
from time import perf_counter

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("LOOPQUANT_VLLM_TEST_MODEL"), reason="opt-in vLLM GPU tests")


@pytest.fixture(scope="module")
def engine():
    from loopquant.evaluation_config import resolve_evaluation_config
    from loopquant.vllm_backend import load_vllm_backend
    source = os.environ["LOOPQUANT_VLLM_TEST_MODEL"]
    loops = int(os.environ.get("LOOPQUANT_VLLM_TEST_LOOPS", "4"))
    settings = resolve_evaluation_config({"batch_size": 2, "generative": {
        "batch_size": 2, "max_length": 256}})
    backend, metadata = load_vllm_backend(source, settings, num_loops=loops)
    return backend, metadata


def test_actual_vllm_generates_and_scores(engine):
    from lm_eval.api.instance import Instance

    backend, metadata = engine
    source = os.environ["LOOPQUANT_VLLM_TEST_MODEL"]
    loops = int(os.environ.get("LOOPQUANT_VLLM_TEST_LOOPS", "4"))
    request = Instance(request_type="generate_until", doc={}, idx=0,
                       arguments=("The capital of France is", {"max_gen_toks": 4, "do_sample": False, "until": []}))
    second = Instance(request_type="generate_until", doc={}, idx=1,
                      arguments=("The sum of 2 and 2 is", {"max_gen_toks": 4, "do_sample": False, "until": []}))
    answer = backend.generate_until([request, second], disable_tqdm=True)
    assert len(answer) == 2 and all(isinstance(text, str) for text in answer)
    assert len(backend.generation_receipts) == 2
    assert all(row["generated_tokens"] > 0 for row in backend.generation_receipts)
    scores = [Instance(request_type="loglikelihood", doc={}, idx=index, arguments=pair)
              for index, pair in enumerate([
                  ("Hello", " world"), ("The capital of France is", " Paris"),
              ])]
    result = backend.loglikelihood(scores, disable_tqdm=True)
    assert len(result) == 2 and all(math.isfinite(row[0]) for row in result)
    if os.environ.get("LOOPQUANT_VLLM_COMPARE_HF") == "1":
        import torch
        from loopquant.models import load_model_for_evaluation

        model, _, _ = load_model_for_evaluation(source, num_loops=loops)
        for request, actual in zip(scores, result, strict=True):
            context, target = backend._encode_pair(*request.args)
            tokens = torch.tensor([context + target], device="cuda:0")
            with torch.inference_mode():
                logits = model(input_ids=tokens, use_cache=False).logits
                predicted = logits[0, len(context) - 1:-1].float().log_softmax(-1)
                reference = predicted.gather(-1, tokens[0, len(context):, None]).sum().item()
            # Different attention/GEMM kernels need not be bitwise identical.
            assert actual[0] == pytest.approx(reference, abs=0.15 * len(target))
    rolling = Instance(request_type="loglikelihood_rolling", doc={}, idx=0, arguments=("Hello world",))
    result = backend.loglikelihood_rolling([rolling], disable_tqdm=True)
    assert len(result) == 1 and math.isfinite(result[0])
    assert metadata["num_loops"] == loops


@pytest.mark.parametrize("width", [32, 96, 2048, 8192])
def test_cuda_fast_orthogonal_map(width):
    import torch
    from loopquant.quantization import orthogonal_transform
    from loopquant.vllm_ops import InferenceOrthogonalTransform, TransformCache

    torch.manual_seed(17)
    value = torch.randn(7, width, device="cuda", dtype=torch.float32)
    fast = InferenceOrthogonalTransform(width, fast=True)
    fast.prepare(TransformCache(), value.device)
    torch.testing.assert_close(fast(value), orthogonal_transform(value), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_clip", [False, True])
def test_cuda_fused_permutation_quantization_and_output(bits, group_clip):
    import torch
    from loopquant.lctc import _fake_quantize_with_group_clip
    from loopquant.quantization import fake_quantize_activation
    from loopquant.vllm_kernels import output_affine, permute_quantize

    torch.manual_seed(18)
    value = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16)
    value[0].zero_()
    permutation = torch.randperm(96, device="cuda")
    clip = torch.tensor([0.6, 0.8, 1.0] if group_clip else [0.9], device="cuda")
    permuted = value.index_select(-1, permutation)
    expected = (_fake_quantize_with_group_clip(permuted, bits, clip) if group_clip else
                fake_quantize_activation(permuted, bits, clip_ratio=clip))
    actual = permute_quantize(value, permutation, clip, bits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gain = torch.linspace(0.8, 1.2, 96, device="cuda")
    bias = torch.linspace(-0.03, 0.03, 96, device="cuda")
    expected = (actual.float() * gain + bias).to(actual.dtype)
    torch.testing.assert_close(output_affine(actual, gain, bias), expected, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("LOOPQUANT_VLLM_BENCHMARK") != "1", reason="opt-in benchmark")
def test_benchmark_engine_and_reference_operators(engine):
    """Report measurements; never impose an arbitrary speedup threshold."""
    import torch
    from torch import nn
    from vllm import SamplingParams, TokensPrompt
    from loopquant.lctc import LoopConditionedCGB, _LCTCRuntime, _activate_runtime
    from loopquant.quantization import QUANTIZATION_CONTEXT, QuantizedLinear
    from loopquant.vllm_ops import InferenceLinear, TransformCache

    backend, metadata = engine
    prompt = backend.tok_encode("The quick brown fox jumps over the lazy dog. " * 16)[:128]
    prompts = [TokensPrompt(prompt_token_ids=prompt), TokensPrompt(prompt_token_ids=prompt[:-8])]
    parameters = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    backend.model.generate(prompts, parameters, use_tqdm=False)  # Warmup, excluded below.
    started = perf_counter()
    outputs = backend.model.generate(prompts, parameters, use_tqdm=False)
    seconds = perf_counter() - started
    report = {"model_source": metadata["model_source"], "execution_mode": metadata.get("execution_mode"),
              "engine_seconds": seconds, "generated_tokens": sum(len(out.outputs[0].token_ids) for out in outputs),
              "requests": []}
    for output in outputs:
        metrics = output.metrics
        if (metrics is not None and metrics.first_token_time is not None
                and metrics.last_token_time is not None and metrics.first_token_time >= metrics.arrival_time):
            report["requests"].append({
                "time_to_first_token_seconds": metrics.first_token_time - metrics.arrival_time,
                "decode_seconds": metrics.last_token_time - metrics.first_token_time,
            })

    # Same-process CUDA event benchmark isolates our operators from scheduling.
    # Peak deltas exclude already resident engine/cache tensors, not total VRAM.
    previous = vars(QUANTIZATION_CONTEXT).copy()
    try:
        QUANTIZATION_CONTEXT.configure(4)
        QUANTIZATION_CONTEXT.enabled = True
        module = QuantizedLinear(nn.Linear(2048, 2048, bias=False, device="cuda", dtype=torch.bfloat16), "bench")
        module.cldc_hadamard = True
        module.set_cldc_scale(torch.ones(2048, device="cuda", dtype=torch.bfloat16))
        module.cldc_permutation = torch.arange(2047, -1, -1, device="cuda")
        cgb = LoopConditionedCGB.from_model(nn.Sequential(module), 4).to("cuda")
        runtime = _LCTCRuntime(cgb)
        cached, fast = InferenceLinear(module, 4, 0, cgb), InferenceLinear(module, 4, 0, cgb, fast=True)
        with torch.inference_mode(), _activate_runtime(runtime):
            cache = TransformCache()
            cached.prepare(cache)
            fast.prepare(cache)
            report["operators"] = []
            for tokens in (1, 16, 512):
                value = torch.randn(1, tokens, 2048, device="cuda", dtype=torch.bfloat16)
                for name, operation in (("reference", module), ("cached", cached), ("fast", fast)):
                    for _ in range(3):
                        operation(value)
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        operation(value)
                    end.record()
                    end.synchronize()
                    report["operators"].append({"mode": name, "tokens": tokens,
                        "milliseconds": start.elapsed_time(end) / 10,
                        "peak_extra_bytes": torch.cuda.max_memory_allocated() - allocated})
    finally:
        vars(QUANTIZATION_CONTEXT).update(previous)
    output = os.environ.get("LOOPQUANT_VLLM_BENCHMARK_OUTPUT")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
