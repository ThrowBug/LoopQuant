"""Opt-in smoke test against the actual vLLM CUDA engine, with no dataset downloads.

LOOPQUANT_VLLM_TEST_MODEL=/models/Ouro-or-loopquant python -m pytest tests/test_vllm_gpu.py
Requires requirements-vllm.txt and pytest in the active Linux/CUDA environment.
Set LOOPQUANT_VLLM_COMPARE_HF=1 to also compare teacher-forced scores with HF.
"""

import math
import os

import pytest


@pytest.mark.skipif(not os.environ.get("LOOPQUANT_VLLM_TEST_MODEL"), reason="opt-in vLLM GPU smoke test")
def test_actual_vllm_generates_and_scores():
    from lm_eval.api.instance import Instance
    from loopquant.evaluation_config import resolve_evaluation_config
    from loopquant.vllm_backend import load_vllm_backend
    source = os.environ["LOOPQUANT_VLLM_TEST_MODEL"]
    loops = int(os.environ.get("LOOPQUANT_VLLM_TEST_LOOPS", "4"))
    settings = resolve_evaluation_config({"batch_size": 2, "generative": {
        "batch_size": 2, "max_length": 256}})
    backend, metadata = load_vllm_backend(source, settings, num_loops=loops)
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
