"""Opt-in smoke test against the actual vLLM CUDA engine, with no dataset downloads.

LOOPQUANT_VLLM_TEST_MODEL=/models/Ouro-or-loopquant pytest tests/test_vllm_gpu.py
Requires requirements-vllm.txt and pytest in the active Linux/CUDA environment.
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
    settings = resolve_evaluation_config({"generative": {"max_length": 256}})
    backend, metadata = load_vllm_backend(source, settings, num_loops=loops)
    request = Instance(request_type="generate_until", doc={}, idx=0,
                       arguments=("The capital of France is", {"max_gen_toks": 4, "do_sample": False, "until": []}))
    answer = backend.generate_until([request], disable_tqdm=True)
    assert len(answer) == 1 and isinstance(answer[0], str)
    assert backend.generation_receipts[0]["generated_tokens"] > 0
    score = Instance(request_type="loglikelihood", doc={}, idx=0, arguments=("Hello", " world"))
    result = backend.loglikelihood([score], disable_tqdm=True)
    assert len(result) == 1 and math.isfinite(result[0][0])
    rolling = Instance(request_type="loglikelihood_rolling", doc={}, idx=0, arguments=("Hello world",))
    result = backend.loglikelihood_rolling([rolling], disable_tqdm=True)
    assert len(result) == 1 and math.isfinite(result[0])
    assert metadata["num_loops"] == loops
