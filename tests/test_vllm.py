"""vLLM routing/manifest tests and CPU reference tests of the paged-attention bridge.

The reference cache below is deliberately not a claim to exercise CUDA/vLLM.
Real vLLM tests are opt-in and require a compatible Linux GPU environment.
"""

import importlib
import json
import os
import subprocess
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


@pytest.mark.parametrize("installed,supported", [
    ("0.11.2", True), ("0.11.2+cu128", True),
    ("0.8.5.post1", False), ("0.11.1", False), ("0.26.0", False), ("0.11.2rc1", False),
])
def test_vllm_environment_accepts_cuda_builds_of_supported_release(monkeypatch, installed, supported):
    import loopquant.vllm_backend as loader
    monkeypatch.setattr(loader, "version", lambda name: installed)
    monkeypatch.setattr(loader, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setattr(loader, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.delenv("VLLM_USE_V1", raising=False)
    if supported:
        loader.check_vllm_environment("cuda:0")
    else:
        with pytest.raises(RuntimeError, match="Ouro integration requires"):
            loader.check_vllm_environment("cuda:0")


def test_vllm_environment_rejects_v0_engine(monkeypatch):
    import loopquant.vllm_backend as loader
    monkeypatch.setattr(loader, "version", lambda name: "0.11.2")
    monkeypatch.setattr(loader, "os", SimpleNamespace(name="posix", environ={"VLLM_USE_V1": "0"}))
    monkeypatch.setattr(loader, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    with pytest.raises(RuntimeError, match="V1 engine"):
        loader.check_vllm_environment("cuda:0")


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
    settings = resolve_evaluation_config({"batch_size": 16, "generative": {"batch_size": 8}})
    args, metadata = build_vllm_arguments("ByteDance/Ouro-1.4B", settings, num_loops=2)
    assert args["pretrained"] == "ByteDance/Ouro-1.4B"
    assert args["hf_overrides"] == {"total_ut_steps": 2}
    assert args["enforce_eager"] and args["tensor_parallel_size"] == 1
    assert args["compilation_config"] == {"mode": 0}
    assert args["max_num_seqs"] == 16 and args["enable_prefix_caching"] is False
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


@pytest.mark.parametrize("previous,previous_method", [(None, None), ("0", "fork"), ("1", "spawn")])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_backend_isolates_plain_python_children_and_restores_environment(
        monkeypatch, previous, previous_method, snapshot, fail):
    import loopquant.vllm_backend as loader
    if previous is None:
        monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    else:
        monkeypatch.setenv("PYTHONNOUSERSITE", previous)
    if previous_method is None:
        monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    else:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", previous_method)
    monkeypatch.setattr(loader, "check_vllm_environment", lambda device: None)
    monkeypatch.setattr(loader, "build_vllm_arguments", lambda *a, **kw: (
        {"hf_overrides": {"total_ut_steps": 2}}, {"fake_quantization": snapshot}))
    monkeypatch.setattr(loader, "entry_points", lambda **kw: [SimpleNamespace(name="loopquant")])

    def register():
        assert os.environ["PYTHONNOUSERSITE"] == "1"
        assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    monkeypatch.setattr(loader, "register_vllm_models", register)

    def construct(**kwargs):
        # Reproduce the registry's plain `python -m ...` subprocess: no -s.
        output = subprocess.check_output([sys.executable, "-c",
            "import json, multiprocessing as mp, os, site, sys; print(json.dumps(["
            "os.environ.get('PYTHONNOUSERSITE'), sys.flags.no_user_site, site.ENABLE_USER_SITE,"
            "mp.get_context(os.environ['VLLM_WORKER_MULTIPROC_METHOD']).get_start_method()]))"],
            text=True)
        assert json.loads(output) == ["1", 1, False, "spawn"]
        if fail:
            raise RuntimeError("engine initialization failed")
        return SimpleNamespace(_config=SimpleNamespace())

    fake = ModuleType("loopquant.vllm_adapter")
    fake.EvaluationVLLM = construct
    monkeypatch.setitem(sys.modules, "loopquant.vllm_adapter", fake)
    if fail:
        with pytest.raises(RuntimeError, match="engine initialization failed"):
            loader.load_vllm_backend("unused", resolve_evaluation_config({}), num_loops=2)
    else:
        backend, _ = loader.load_vllm_backend("unused", resolve_evaluation_config({}), num_loops=2)
        assert backend._config.total_ut_steps == 2
    assert os.environ.get("PYTHONNOUSERSITE") == previous
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == previous_method


def test_engine_spawn_starts_with_fresh_process_state(tmp_path):
    from loopquant.vllm_backend import _vllm_subprocess_environment

    script = tmp_path / "spawn_probe.py"
    script.write_text(
        "import json, multiprocessing as mp, os, site\n"
        "state = 'fresh'\n"
        "def inspect_child(queue):\n"
        "    queue.put([mp.get_start_method(), state, site.ENABLE_USER_SITE])\n"
        "if __name__ == '__main__':\n"
        "    state = 'parent runtime initialized'\n"
        "    context = mp.get_context(os.environ['VLLM_WORKER_MULTIPROC_METHOD'])\n"
        "    queue = context.Queue()\n"
        "    child = context.Process(target=inspect_child, args=(queue,))\n"
        "    child.start()\n"
        "    result = queue.get(timeout=10)\n"
        "    child.join(timeout=10)\n"
        "    assert child.exitcode == 0\n"
        "    print(json.dumps(result))\n",
        encoding="utf-8")
    with _vllm_subprocess_environment():
        output = subprocess.check_output([sys.executable, str(script)], text=True, timeout=30)
    assert json.loads(output) == ["spawn", "fresh", False]


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
        assert adapter.generate_until([]) == []
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


class NativeDecoderControlFlow(torch.nn.Module):
    """vLLM 0.11.2 OuroDecoderLayer.forward, for the CPU-only import stub.

    Real native modules and graph execution are exercised by test_vllm_gpu.py.
    """

    def forward(self, positions, hidden_states, current_ut, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, current_ut=current_ut)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        return hidden_states, residual


@pytest.fixture
def bridge_module(monkeypatch):
    for name in ("vllm", "vllm.attention", "vllm.compilation", "vllm.compilation.decorators",
                 "vllm.model_executor", "vllm.model_executor.models", "vllm.model_executor.models.ouro"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm.attention"].Attention = ReferenceAttention
    sys.modules["vllm.compilation.decorators"].support_torch_compile = lambda **kw: lambda cls: cls
    sys.modules["vllm.model_executor.models.ouro"].OuroDecoderLayer = NativeDecoderControlFlow
    # Undo the imported bridge as well, so fake vLLM does not leak to other tests.
    monkeypatch.delitem(sys.modules, "loopquant.vllm_model", raising=False)
    module = importlib.import_module("loopquant.vllm_model")
    previous = vars(QUANTIZATION_CONTEXT).copy()
    try:
        yield module
    finally:
        vars(QUANTIZATION_CONTEXT).update(previous)
        sys.modules.pop("loopquant.vllm_model", None)


def save_paged_model(path, allocation, *, num_layers=1):
    torch.manual_seed(27)
    model = PagedOuroForCausalLM(PagedOuroConfig(num_hidden_layers=num_layers)).to(torch.bfloat16).eval()
    with torch.no_grad():
        for module in model.modules():
            if "RMSNorm" in type(module).__name__:
                module.weight.copy_(torch.linspace(0.8, 1.2, module.weight.numel()))
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
@pytest.mark.parametrize("num_layers", [1, 2])
def test_bridge_restores_prefill_and_incremental_decode(tmp_path, bridge_module, allocation, num_layers):
    original = save_paged_model(tmp_path / "loopquant", allocation, num_layers=num_layers)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        quant_config=None, model_config=SimpleNamespace(model=str(tmp_path / "loopquant"), dtype=torch.bfloat16),
        cache_config=None, device_config=SimpleNamespace(device="cpu"))
    bridge = bridge_module.LoopQuantOuroForCausalLM(vllm_config=config)
    bridge.load_weights(iter(()))
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, bridge.hf_model.state_dict()[name]), name
    plan = bridge.model.loops[0][0].self_attn.q_proj
    assert plan.clip.dtype == torch.float32
    assert plan.linear.weight is bridge.hf_model.model.layers[0].self_attn.q_proj.weight
    other = bridge.model.loops[1][0].self_attn.q_proj
    assert other.linear.weight is plan.linear.weight
    assert other.transform.hadamard is plan.transform.hadamard
    assert {cache.prefix for cache in bridge.attentions} == {
        f"model.layers.{index}.self_attn.attn" for index in range(4 * num_layers)}
    ids = torch.tensor([[3, 4, 3]])
    with torch.no_grad():
        expected = original(ids).logits[0].float()
        actual = bridge.compute_logits(bridge(ids[0], torch.arange(3)))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        extended = torch.cat((ids, torch.tensor([[4]])), -1)
        expected = original(extended).logits[0, -1].float()
        decoded = bridge.compute_logits(bridge(torch.tensor([4]), torch.tensor([3])))[0]
        torch.testing.assert_close(expected, decoded, rtol=0, atol=0)
        assert all(cache.keys.shape[1] == 4 for cache in bridge.attentions)
    args, metadata = build_vllm_arguments(tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=4)
    assert args["dtype"] == "bfloat16"
    assert args["hf_overrides"]["architectures"] == ["LoopQuantOuroForCausalLM"]
    assert metadata["allocation"] == list(allocation)
    with pytest.raises(ValueError, match="conflicts"):
        build_vllm_arguments(tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=2)


def test_bridge_rejects_dtype_override_and_restores_quantization_context(tmp_path, bridge_module):
    save_paged_model(tmp_path / "loopquant", (4, 4, 4, 4))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        quant_config=None,
        model_config=SimpleNamespace(model=str(tmp_path / "loopquant"), dtype=torch.float16),
        cache_config=None, device_config=SimpleNamespace(device="cpu"))
    bridge = bridge_module.LoopQuantOuroForCausalLM(vllm_config=config)
    with pytest.raises(ValueError, match="dtype must match"):
        bridge.load_weights(iter(()))
    bridge.dtype = torch.bfloat16
    bridge.load_weights(iter(()))
    QUANTIZATION_CONTEXT.configure(2, (8, 16))
    QUANTIZATION_CONTEXT.current_loop = 1
    QUANTIZATION_CONTEXT.enabled = False
    with torch.no_grad():
        bridge(torch.tensor([3]), torch.tensor([0]))
    assert (QUANTIZATION_CONTEXT.num_loops, QUANTIZATION_CONTEXT.current_loop,
            QUANTIZATION_CONTEXT.activation_bits, QUANTIZATION_CONTEXT.enabled) == (2, 1, (8, 16), False)


@pytest.mark.parametrize("mode", ["cached", "compiled", "fast"])
def test_snapshot_execution_mode_is_forwarded_to_worker(tmp_path, monkeypatch, mode):
    save_paged_model(tmp_path / "loopquant", (8, 4, 8, 4))
    monkeypatch.setenv("LOOPQUANT_VLLM_EXECUTION", mode)
    arguments, metadata = build_vllm_arguments(
        tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=4)
    assert arguments["hf_overrides"]["loopquant_execution"] == mode
    assert metadata["execution_mode"] == mode
    assert arguments["enforce_eager"] == (mode == "cached")
    assert arguments["compilation_config"]["mode"] == (0 if mode == "cached" else 3)
    monkeypatch.setenv("LOOPQUANT_VLLM_EXECUTION", "typo")
    with pytest.raises(ValueError, match="LOOPQUANT_VLLM_EXECUTION"):
        build_vllm_arguments(tmp_path / "loopquant", resolve_evaluation_config({}), num_loops=4)


@pytest.mark.parametrize("compiled", [False, True])
def test_static_executor_never_calls_context_quantized_forward(tmp_path, bridge_module, monkeypatch, compiled):
    from loopquant.quantization import QuantizedLinear

    original = save_paged_model(tmp_path / "loopquant", (8, 4, 8, 4), num_layers=2)
    with torch.no_grad():
        expected = original(torch.tensor([[3, 4, 3]])).logits[0].float()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        quant_config=None, model_config=SimpleNamespace(model=str(tmp_path / "loopquant"), dtype=torch.bfloat16),
        cache_config=None, device_config=SimpleNamespace(device="cpu"))
    bridge = bridge_module.LoopQuantOuroForCausalLM(vllm_config=config)
    bridge.load_weights(iter(()))
    if compiled:
        bridge.model = torch.compile(bridge.model, backend="eager", fullgraph=True, dynamic=True)
    monkeypatch.setattr(QuantizedLinear, "forward", lambda *a, **kw: pytest.fail("context-based forward"))
    QUANTIZATION_CONTEXT.configure(1, (16,))
    QUANTIZATION_CONTEXT.enabled = False
    with torch.no_grad(), torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=False):
        actual = bridge.compute_logits(bridge(torch.tensor([3, 4, 3]), torch.arange(3)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
