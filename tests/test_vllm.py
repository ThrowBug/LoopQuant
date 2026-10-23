"""vLLM routing/manifest tests and CPU reference tests of the paged-attention bridge.

The reference cache below is deliberately not a claim to exercise CUDA/vLLM.
Real vLLM tests are opt-in and require a compatible Linux GPU environment.
"""

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from loopquant import cli
from loopquant.checkpoint import load_quantized_checkpoint, save_quantized_checkpoint
from loopquant.evaluation_config import resolve_evaluation_config
from loopquant.lctc import LoopConditionedCGB, apply_lctc_calibration
from loopquant.models import configure_looped_model
from loopquant.quantization import QUANTIZATION_CONTEXT, fake_quantize_weight, rotate_ouro_backbone, wrap_quantized_modules
from loopquant.vllm_backend import build_vllm_arguments
from modeling_paged_ouro import PagedOuroConfig, PagedOuroForCausalLM
from test_evaluation import tiny_model_and_tokenizer


def test_vllm_is_evaluate_only_and_rejects_ppl_only():
    base = ["--model", "unused", "--output-dir", "unused"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["quantize", *base, "--config", "unused", "--vllm"])
    with pytest.raises(SystemExit):
        cli.main(["evaluate", *base, "--num-loops", "4", "--vllm", "--wikitext2"])


def test_vllm_cli_skips_ppl_without_loading_hf(tmp_path, monkeypatch):
    import loopquant.vllm_backend as loader
    calls = []
    backend = SimpleNamespace(tokenizer=object())
    monkeypatch.setattr(cli, "preflight_evaluation", lambda config, **kw: calls.append(kw) or "manager")
    monkeypatch.setattr(loader, "load_vllm_backend", lambda *a, **kw: (backend, {"backend": "vllm"}))
    monkeypatch.setattr(cli, "load_model_for_evaluation", lambda *a, **kw: pytest.fail("loaded HF model"))
    monkeypatch.setattr(cli, "run_evaluation", lambda *a, **kw: calls.append(kw))
    cli.main(["evaluate", "--model", "unused", "--num-loops", "4", "--output-dir", str(tmp_path),
              "--device", "cpu", "--vllm", "--wikitext2", "--c4", "--zero-shot"])
    assert calls[0]["suites"] == ["zero_shot"]
    assert calls[1]["lm_backend"] is backend
    assert calls[1]["skipped_suites"] == ["wikitext2", "c4"]


@pytest.mark.parametrize("phase", ["preflight", "model_loading"])
def test_early_evaluation_failure_has_manifest(tmp_path, monkeypatch, phase):
    def fail(*a, **kw):
        raise RuntimeError("expected failure")
    monkeypatch.setattr(cli, "preflight_evaluation", fail if phase == "preflight" else lambda *a, **kw: None)
    monkeypatch.setattr(cli, "load_model_for_evaluation", fail)
    with pytest.raises(RuntimeError, match="expected failure"):
        cli.main(["evaluate", "--model", "unused", "--num-loops", "4", "--output-dir", str(tmp_path),
                  "--device", "cpu", "--wikitext2"])
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failed_task"] == phase


def test_original_vllm_arguments_are_explicit(monkeypatch):
    import loopquant.vllm_backend as loader
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="ouro"))
    args, metadata = build_vllm_arguments("ByteDance/Ouro-1.4B", resolve_evaluation_config({}), num_loops=2)
    assert args["pretrained"] == "ByteDance/Ouro-1.4B"
    assert args["hf_overrides"] == {"total_ut_steps": 2}
    assert args["enforce_eager"] and args["tensor_parallel_size"] == 1
    assert metadata["mode"] == "full_precision"
    with pytest.raises(ValueError, match="always uses"):
        build_vllm_arguments("unused", resolve_evaluation_config({"generative": {"use_cache": False}}), num_loops=4)


def test_backend_factory_preserves_effective_config(monkeypatch):
    import loopquant.vllm_backend as loader
    monkeypatch.setattr(loader, "check_vllm_environment", lambda device: None)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="ouro"))
    captured = []
    fake = ModuleType("loopquant.vllm_adapter")
    def construct(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(_config=SimpleNamespace(total_ut_steps=4))
    fake.EvaluationVLLM = construct
    monkeypatch.setitem(sys.modules, "loopquant.vllm_adapter", fake)
    backend, metadata = loader.load_vllm_backend(
        "ByteDance/Ouro-1.4B", resolve_evaluation_config({}), num_loops=2, device="cpu")
    assert captured[0]["hf_overrides"] == {"total_ut_steps": 2}
    assert backend._config.total_ut_steps == metadata["num_loops"] == 2


def test_vllm_runner_uses_shared_task_protocol_and_records_backend(tmp_path, monkeypatch):
    import lm_eval.evaluator
    from loopquant.evaluation import run_evaluation
    calls = []
    backend = SimpleNamespace(generation_receipts=[], tokenizer=SimpleNamespace(), batch_size=1)
    def evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"drop": {"exact_match": 0.5}}}
    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", evaluate)
    settings = resolve_evaluation_config({"generative": {
        "batch_size": 8, "tasks": [{"name": "drop", "num_fewshot": 1, "max_gen_toks": 128}]}})
    metrics = run_evaluation(None, backend.tokenizer, settings, tmp_path, suites=["generative"],
                             task_manager=object(), lm_backend=backend, backend_name="vllm",
                             skipped_suites=["wikitext2", "c4"])
    assert backend.batch_size == 8
    assert calls[0]["num_fewshot"] == 1
    assert calls[0]["gen_kwargs"] == {"do_sample": False, "max_gen_toks": 128}
    assert metrics["generative"]["drop"]
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["backend"] == "vllm" and manifest["status"] == "complete"
    assert manifest["skipped_suites"] == ["wikitext2", "c4"]


def test_vllm_adapter_checks_budget_and_keeps_receipts(monkeypatch):
    class Base:
        max_gen_toks, max_length = 4, 10
        def __init__(self, **kwargs): pass
        def tok_encode(self, prompt): return list(range(len(prompt)))
        def generate_until(self, requests, **kwargs): return ["answer"] * len(requests)
        def _model_generate(self, requests, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2], finish_reason="stop")])]
    fake = ModuleType("lm_eval.models.vllm_causallms")
    fake.VLLM = Base
    monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", fake)
    monkeypatch.delitem(sys.modules, "loopquant.vllm_adapter", raising=False)
    try:
        adapter = importlib.import_module("loopquant.vllm_adapter").EvaluationVLLM()
        request = SimpleNamespace(args=("long prompt", {"max_gen_toks": 4}))
        with pytest.raises(ValueError, match="Prompt requires"):
            adapter.generate_until([request])
        adapter.allow_prompt_truncation = True
        assert adapter.generate_until([request]) == ["answer"]
        adapter._model_generate([[1, 2]], generate=True, sampling_params=SimpleNamespace(max_tokens=4))
        assert adapter.generation_receipts[0]["finish_reason"] == "stop"
        assert adapter.generation_receipts[0]["generated_tokens"] == 2
    finally:
        sys.modules.pop("loopquant.vllm_adapter", None)


class ReferenceAttention(torch.nn.Module):
    """One sequence cache per logical (layer, loop); a CPU oracle, not vLLM."""
    def __init__(self, heads, head_dim, scale, num_kv_heads, **kwargs):
        super().__init__()
        self.heads, self.kv_heads, self.dim, self.scale = heads, num_kv_heads, head_dim, scale
        self.keys, self.values = None, None
        self.prefix = kwargs["prefix"]

    def forward(self, query, key, value):
        query = query.view(-1, self.heads, self.dim).transpose(0, 1)
        key = key.view(-1, self.kv_heads, self.dim).transpose(0, 1)
        value = value.view(-1, self.kv_heads, self.dim).transpose(0, 1)
        previous = 0 if self.keys is None else self.keys.shape[1]
        self.keys = key if self.keys is None else torch.cat((self.keys, key), 1)
        self.values = value if self.values is None else torch.cat((self.values, value), 1)
        key = self.keys.repeat_interleave(self.heads // self.kv_heads, 0)
        value = self.values.repeat_interleave(self.heads // self.kv_heads, 0)
        scores = query @ key.transpose(-1, -2) * self.scale
        mask = torch.arange(key.shape[1])[None, :] <= previous + torch.arange(query.shape[1])[:, None]
        result = torch.softmax(scores.masked_fill(~mask, -torch.inf).float(), -1).to(query.dtype) @ value
        return result.transpose(0, 1).reshape(-1, self.heads * self.dim)


@pytest.fixture
def bridge_module(monkeypatch):
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.attention"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm.model_executor.layers.attention"].Attention = ReferenceAttention
    # Undo the imported bridge as well, so fake vLLM does not leak to other tests.
    monkeypatch.delitem(sys.modules, "loopquant.vllm_model", raising=False)
    module = importlib.import_module("loopquant.vllm_model")
    yield module
    sys.modules.pop("loopquant.vllm_model", None)


def save_paged_model(path, allocation):
    torch.manual_seed(27)
    model = PagedOuroForCausalLM(PagedOuroConfig()).to(torch.bfloat16).eval()
    configure_looped_model(model, 4)
    QUANTIZATION_CONTEXT.configure(4, allocation)
    rotate_ouro_backbone(model)
    for module in wrap_quantized_modules(model).values():
        module.set_cldc_scale(torch.linspace(0.8, 1.2, module.in_features).to(module.weight.dtype))
        module.cldc_hadamard = True
        module.cldc_permutation = torch.arange(module.in_features - 1, -1, -1)
        module.weight.data = fake_quantize_weight(module.weight, 4)
    cgb = LoopConditionedCGB.from_model(model, 4)
    with torch.no_grad():
        for spec, entry in cgb.iter_entries():
            entry.clip.fill_(0.9 - spec.loop_id * 0.03)
            entry.gain.fill_(1.0 + spec.loop_id * 0.01)
            entry.beta.fill_(0.01)
    apply_lctc_calibration(model, cgb)
    model._loopquant_allocation = tuple(allocation)
    QUANTIZATION_CONTEXT.enabled = True
    _, tokenizer = tiny_model_and_tokenizer()
    save_quantized_checkpoint(model, tokenizer, path)
    return model


@pytest.mark.parametrize("allocation", [(4, 4, 4, 4), (8, 4, 8, 4)])
def test_bridge_restores_prefill_and_incremental_decode(tmp_path, bridge_module, allocation):
    original = save_paged_model(tmp_path / "loopquant", allocation)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        quant_config=None, model_config=SimpleNamespace(model=str(tmp_path / "loopquant")),
        cache_config=None, device_config=SimpleNamespace(device="cpu"))
    bridge = bridge_module.LoopQuantOuroForCausalLM(vllm_config=config)
    bridge.load_weights(iter(()))
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, bridge.hf_model.state_dict()[name]), name
    assert bridge.runtime.cgb.entry("model.layers.0.self_attn.q_proj", 0).clip.dtype == torch.float32
    assert len({cache.prefix for cache in bridge.attentions}) == 4
    ids = torch.tensor([[3, 4, 3]])
    with torch.no_grad():
        expected = original(ids).logits[0].float()
        actual = bridge.compute_logits(bridge(ids[0], torch.arange(3)))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        extended = torch.cat((ids, torch.tensor([[4]])), -1)
        expected = original(extended).logits[0, -1].float()
        decoded = bridge.compute_logits(bridge(torch.tensor([4]), torch.tensor([3])))[0]
        torch.testing.assert_close(expected, decoded, rtol=0, atol=0)
    args, metadata = build_vllm_arguments(tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=4)
    assert args["dtype"] == "bfloat16"
    assert args["hf_overrides"]["architectures"] == ["LoopQuantOuroForCausalLM"]
    assert metadata["allocation"] == list(allocation)
    with pytest.raises(ValueError, match="conflicts"):
        build_vllm_arguments(tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=2)
