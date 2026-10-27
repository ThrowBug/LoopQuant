"""Low-rank algorithm, recurrent freezing, cache and checkpoint contracts."""

import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from loopquant.checkpoint import save_quantized_checkpoint, load_quantized_checkpoint
from loopquant.config import LoopQuantConfig
from loopquant.low_rank.adapter import run_block
from loopquant.low_rank.cache import prepare_teacher_cache, block_statistics
from loopquant.low_rank.modules import LowRankLinear, iter_linears, wrap_linears, prepare_inference
from loopquant.low_rank.pipeline import LowRankPipeline
from loopquant.low_rank.quantization import quantize_activation
from loopquant.low_rank.reconstruction import initialize_increment
from loopquant.models import configure_looped_model, load_model_for_evaluation
from modeling_tiny_ouro import TinyOuroConfig, TinyOuroForCausalLM
from modeling_paged_ouro import PagedOuroConfig, PagedOuroForCausalLM
from test_evaluation import tiny_model_and_tokenizer
from test_vllm import bridge_module


def configuration(tmp_path, *, loops=2, sharing="prefix", mode="mean_hessian", setting="w4a4"):
    return LoopQuantConfig.from_dict({
        "algorithm": "low_rank", "num_loops": loops, "model_key": "tiny",
        "quantization": {"activation_setting": setting},
        "calibration": {"cache_dir": str(tmp_path / "cache"), "num_samples": 2, "sequence_length": 3},
        "method": {"sharing": sharing, "rank_step": 2, "gptq": {"hessian_mode": mode},
                   "activation": {"clip_min": 0.8, "clip_max": 1.0, "clip_step": 0.1},
                   "optimization": {"steps": 3, "learning_rate": 0.01}},
    })


def tiny(loops=2, layers=2):
    torch.manual_seed(73)
    model = TinyOuroForCausalLM(TinyOuroConfig(total_ut_steps=loops, num_hidden_layers=layers)).to(torch.bfloat16).eval()
    configure_looped_model(model, loops, quantization_hooks=False)
    _, tokenizer = tiny_model_and_tokenizer()
    return model, tokenizer


DATA = torch.tensor([[3, 4, 3], [4, 3, 4]])


@pytest.mark.parametrize("loops", [1, 2, 3, 5])
@pytest.mark.parametrize("sharing", ["prefix", "independent"])
def test_config_uses_configured_loops_and_equal_storage(tmp_path, loops, sharing):
    config = configuration(tmp_path, loops=loops, sharing=sharing)
    model, _ = tiny(loops=loops)
    wrap_linears(model, config)
    total = 0
    for block in model.model.layers:
        for _, module in iter_linears(block):
            parameters = sum(p.numel() for bank in (module.A, module.B) for p in bank)
            assert parameters == loops * 2 * (module.in_features + module.out_features)
            total += parameters
    assert total > 0
    assert config.to_dict()["algorithm"] == "low_rank"
    assert "cldc" not in config.method


@pytest.mark.parametrize("update", [
    {"sharing": "typo"}, {"rank_step": 0}, {"rotation": {"enabled": True}},
    {"channel_scaling": {"enabled": True}}, {"gptq": {"hessian_mode": "hessian_of_mean"}},
    {"activation_bits": [4, 8]}, {"activation_bits": 6}, {"optimization": {"steps": 0}},
])
def test_invalid_method_options_are_rejected(tmp_path, update):
    values = configuration(tmp_path).to_dict()
    for key, value in update.items():
        if isinstance(value, dict):
            values["method"][key].update(value)
        else:
            values["method"][key] = value
    with pytest.raises(ValueError):
        LoopQuantConfig.from_dict(values)


@pytest.mark.parametrize("setting,bits", [("w4a4", 4), ("w4a8", 8)])
def test_activation_precision_is_uniform_across_projections_and_loops(tmp_path, setting, bits):
    config = configuration(tmp_path, loops=3, setting=setting)
    assert "activation_bits" not in config.method
    assert "activation_candidates" not in config.quantization
    assert "average_activation_bits" not in config.quantization
    model, _ = tiny(loops=3)
    wrap_linears(model, config)
    values = torch.randn(2, 3, 32).to(torch.bfloat16)
    for block in model.model.layers:
        for _, module in iter_linears(block):
            assert module.activation_bits == bits
            # Zero initial factors isolate the activation quantizer from compensation.
            inputs = values[..., :module.in_features]
            if module.in_features != inputs.shape[-1]:
                inputs = torch.randn(2, 3, module.in_features).to(torch.bfloat16)
            expected = torch.nn.functional.linear(quantize_activation(inputs, bits), module.weight, module.bias)
            for loop in range(config.num_loops):
                torch.testing.assert_close(module.forward_at_loop(inputs, loop), expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_activation_quantization_is_token_group_symmetric_and_differentiable(bits):
    values = torch.linspace(-2, 3, 64).reshape(2, 32).requires_grad_()
    result = quantize_activation(values, bits, clip=0.8)
    maximum = 2 ** (bits - 1) - 1
    scales = values.detach().abs().amax(-1, keepdim=True) * 0.8 / maximum
    expected = (values.detach() / scales).round().clamp(-maximum, maximum) * scales
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


@pytest.mark.parametrize("sharing", ["prefix", "independent"])
def test_only_new_increment_changes_and_prior_loop_is_identical(sharing):
    torch.manual_seed(13)
    original = torch.nn.Linear(32, 32, bias=False).to(torch.bfloat16)
    module = LowRankLinear(original, num_loops=3, rank_step=2, sharing=sharing,
                           activation_bits=4)
    reference = original.weight.detach().clone()
    module.weight.data = quantize_activation(module.weight, 4)
    initialize_increment(module, reference, 0, seed=0, iterations=2)
    for bank in (module.A, module.B):
        bank[0].requires_grad_(False)
    values = torch.randn(2, 3, 32).to(torch.bfloat16)
    before = module.forward_at_loop(values, 0).detach().clone()
    old = [p.detach().clone() for bank in (module.A, module.B) for p in bank[:1]]
    initialize_increment(module, reference, 1, seed=1, iterations=2)
    optimizer = torch.optim.AdamW([module.A[1], module.B[1]], lr=0.1, weight_decay=0.1)
    loss = module.forward_at_loop(values, 1).float().square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(old, [module.A[0], module.B[0]]))
    torch.testing.assert_close(module.forward_at_loop(values, 0), before, rtol=0, atol=0)


def test_teacher_cache_and_hessian_modes_match_independent_formulas(tmp_path, monkeypatch):
    import loopquant.low_rank.cache as caching
    model, tokenizer = tiny(layers=1)
    config = configuration(tmp_path)
    teacher = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert not teacher.hit
    repeat = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert repeat.hit and repeat.directory == teacher.directory
    block = model.model.layers[0]
    seen = []
    handle = block.self_attn.q_proj.register_forward_pre_hook(lambda module, args: seen.append(args[0].float().reshape(-1, 32)))
    for sample in range(2):
        for loop in range(2):
            run_block(model, block, teacher.read(0, loop, sample)["input"], loop)
    handle.remove()
    expected_mean_hessian = sum(x.T @ x for x in seen) / 6
    expected_mean_activation = sum(((seen[i] + seen[i + 1]) / 2).T @ ((seen[i] + seen[i + 1]) / 2)
                                   for i in (0, 2)) / 3
    h1, clip1 = block_statistics(model, 0, teacher, config, "cpu")
    torch.testing.assert_close(h1["self_attn.q_proj"], expected_mean_hessian)
    config.method["gptq"]["hessian_mode"] = "mean_activation"
    h2, clip2 = block_statistics(model, 0, teacher, config, "cpu")
    torch.testing.assert_close(h2["self_attn.q_proj"], expected_mean_activation)
    assert not torch.equal(h1["self_attn.q_proj"], h2["self_attn.q_proj"])
    assert all(torch.equal(clip1[name], clip2[name]) for name in clip1)
    clip_bits = []
    original_fit_clip = caching.fit_clip

    def record_clip(value, bits, group_size, ratios):
        clip_bits.append(bits)
        return original_fit_clip(value, bits, group_size, ratios)

    monkeypatch.setattr(caching, "fit_clip", record_clip)
    config.quantization["activation_setting"] = "w4a8"
    h8, _ = block_statistics(model, 0, teacher, config, "cpu")
    assert clip_bits and set(clip_bits) == {8}
    assert all(torch.equal(h2[name], h8[name]) for name in h2)
    assert len(list((teacher.directory / "activation_clip").glob("*/block_000.safetensors"))) == 2
    monkeypatch.setattr(caching, "run_block", lambda *a, **kw: pytest.fail("statistics cache was not reused"))
    block_statistics(model, 0, teacher, config, "cpu")


def test_teacher_cache_invalidates_changed_tokens_and_weights(tmp_path):
    model, tokenizer = tiny(layers=1)
    config = configuration(tmp_path)
    original = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    changed = prepare_teacher_cache(model, tokenizer, DATA.flip(-1).clone().roll(1, -1), config, "cpu", logging.getLogger())
    assert changed.directory != original.directory
    with torch.no_grad():
        model.model.embed_tokens.weight.add_(0.01)
    changed_weight = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert changed_weight.directory != original.directory


def test_teacher_collection_batches_samples_and_writes_immutable_states_off_thread(tmp_path, monkeypatch):
    import loopquant.low_rank.cache as caching
    model, tokenizer = tiny(loops=3)
    config = configuration(tmp_path, loops=3)
    config.calibration["batch_size"] = 2
    data = torch.tensor([[3, 4, 3], [4, 3, 4], [3, 3, 4], [4, 4, 3], [4, 3, 3]])
    inputs, outputs, batches, writers = {}, {}, [], []
    handles = [model.register_forward_pre_hook(
        lambda module, args, kwargs: batches.append(len(kwargs["input_ids"])), with_kwargs=True)]
    for index, block in enumerate(model.model.layers):
        def pre_hook(module, args, kwargs, block_id=index):
            inputs.setdefault((block_id, kwargs["current_ut"]), []).append(args[0].clone())

        def post_hook(module, args, kwargs, output, block_id=index):
            outputs.setdefault((block_id, kwargs["current_ut"]), []).append(output.clone())

        handles.extend((block.register_forward_pre_hook(pre_hook, with_kwargs=True),
                        block.register_forward_hook(post_hook, with_kwargs=True)))
    original_save = caching.save_file

    def record_writer(tensors, path):
        if Path(path).parent.name.startswith("loop_"):
            writers.append(threading.get_ident())
        return original_save(tensors, path)

    monkeypatch.setattr(caching, "save_file", record_writer)
    try:
        teacher = prepare_teacher_cache(model, tokenizer, data, config, "cpu", logging.getLogger())
    finally:
        for handle in handles:
            handle.remove()
    assert batches == [2, 2, 1]
    assert writers and set(writers).isdisjoint({threading.get_ident()})
    assert teacher.manifest["collection_batch_size"] == 2
    for key in inputs:
        expected_input, expected_output = torch.cat(inputs[key]), torch.cat(outputs[key])
        for sample in range(len(data)):
            states = teacher.read(*key, sample)
            torch.testing.assert_close(states["input"], expected_input[sample:sample + 1], rtol=0, atol=0)
            torch.testing.assert_close(states["output"], expected_output[sample:sample + 1], rtol=0, atol=0)


def test_teacher_collection_batch_size_separates_caches(tmp_path):
    model, tokenizer = tiny(layers=1)
    config = configuration(tmp_path)
    config.calibration["batch_size"] = 1
    single = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert "collection_batch_size" not in single.manifest["identity"]
    config.calibration["batch_size"] = 2
    batched = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert single.directory != batched.directory
    assert batched.manifest["identity"]["collection_batch_size"] == 2
    config.calibration["batch_size"] = 1
    assert prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger()).hit


def test_teacher_writer_failure_propagates_and_cleans_up(tmp_path, monkeypatch):
    import loopquant.low_rank.cache as caching
    model, tokenizer = tiny(layers=1)
    config = configuration(tmp_path)
    original_save = caching.save_file

    def broken_save(tensors, path):
        if Path(path).parent.name.startswith("loop_"):
            raise OSError("teacher write failed")
        return original_save(tensors, path)

    monkeypatch.setattr(caching, "save_file", broken_save)
    with pytest.raises(OSError, match="teacher write failed"):
        prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert not list(Path(config.calibration["cache_dir"]).rglob("manifest.json"))
    assert not list(Path(config.calibration["cache_dir"]).rglob(".teacher-*"))
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in model.model.layers)
    monkeypatch.setattr(caching, "save_file", original_save)
    teacher = prepare_teacher_cache(model, tokenizer, DATA, config, "cpu", logging.getLogger())
    assert not teacher.hit
    assert teacher.read(0, 0, 0)["input"].shape == (1, 3, 32)


@pytest.mark.parametrize("batch_size", [0, False, 1.5])
def test_teacher_collection_requires_positive_integer_batch_size(tmp_path, batch_size):
    values = configuration(tmp_path).to_dict()
    values["calibration"]["batch_size"] = batch_size
    with pytest.raises(ValueError, match="calibration.batch_size"):
        LoopQuantConfig.from_dict(values)


@pytest.mark.parametrize("sharing,mode", [("prefix", "mean_hessian"), ("independent", "mean_activation")])
@pytest.mark.parametrize("setting,bits", [("w4a4", 4), ("w4a8", 8)])
def test_pipeline_and_checkpoint_roundtrip_in_a_new_process(tmp_path, sharing, mode, setting, bits):
    model, tokenizer = tiny()
    config = configuration(tmp_path, sharing=sharing, mode=mode, setting=setting)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter.ndim == 1 or name.startswith(("model.embed_tokens", "lm_head"))}
    result = LowRankPipeline(config, device="cpu").run(model, tokenizer, DATA)
    assert result.summary["activation_allocation"] == [bits, bits]
    assert result.summary["rank_schedule"] == ([2, 4] if sharing == "prefix" else [2, 2])
    assert not any(p.requires_grad for p in result.model.parameters())
    for name, value in frozen.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    with torch.no_grad():
        expected = model(DATA).logits
        tokens = model.generate(DATA[:1], max_new_tokens=2, do_sample=False, use_cache=False)
    checkpoint = tmp_path / "loopquant"
    save_quantized_checkpoint(model, tokenizer, checkpoint, quantization_config=config.to_dict(), max_shard_bytes=15000)
    restored, _ = load_quantized_checkpoint(checkpoint, device="cpu", num_loops=2)
    assert restored._loopquant_allocation == (bits, bits)
    assert all(module.activation_bits == bits for block in restored.model.layers for _, module in iter_linears(block))
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    with torch.no_grad():
        torch.testing.assert_close(restored(DATA).logits, expected, rtol=0, atol=0)
    _, _, metadata = load_model_for_evaluation(checkpoint, num_loops=2, device="cpu")
    assert metadata["algorithm"] == "low_rank"
    script = (
        "import sys,torch; from loopquant.checkpoint import load_quantized_checkpoint; "
        "m,t=load_quantized_checkpoint(sys.argv[1],device='cpu'); "
        "x=torch.tensor([[3,4,3],[4,3,4]]); "
        "torch.save({'logits':m(x).logits.detach(),'tokens':m.generate(x[:1],max_new_tokens=2,"
        "do_sample=False,use_cache=False)},sys.argv[2])"
    )
    output = tmp_path / "new_process.pt"
    env = dict(os.environ, HF_MODULES_CACHE=str(tmp_path / "module_cache"))
    completed = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(output)],
                               env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    loaded = torch.load(output, weights_only=True)
    torch.testing.assert_close(loaded["logits"], expected, rtol=0, atol=0)
    assert torch.equal(loaded["tokens"], tokens)


@pytest.mark.parametrize("sharing", ["prefix", "independent"])
@pytest.mark.parametrize("setting,bits", [("w4a4", 4), ("w4a8", 8)])
def test_low_rank_vllm_prefill_decode_and_method_dispatch(tmp_path, bridge_module, monkeypatch, sharing, setting, bits):
    import loopquant.vllm_backend as backend
    from loopquant.evaluation_config import resolve_evaluation_config
    monkeypatch.delitem(sys.modules, "loopquant.low_rank.vllm_model", raising=False)
    bridge_class = importlib.import_module("loopquant.low_rank.vllm_model").LowRankOuroForCausalLM
    torch.manual_seed(89)
    config = configuration(tmp_path, loops=3, sharing=sharing, setting=setting)
    model = PagedOuroForCausalLM(PagedOuroConfig(num_hidden_layers=2, total_ut_steps=3)).to(torch.bfloat16).eval()
    configure_looped_model(model, 3, quantization_hooks=False)
    wrap_linears(model, config)
    with torch.no_grad():
        for block in model.model.layers:
            for _, module in iter_linears(block):
                module.weight.copy_(quantize_activation(module.weight, 4))
                module.clip_ratios.copy_(torch.tensor([0.81, 0.92, 0.87]))
                for bank in (module.A, module.B):
                    for parameter in bank:
                        parameter.normal_(0, 0.02)
    model._quantization_algorithm, model._low_rank_config = "low_rank", config.to_dict()
    model._loopquant_allocation = (bits,) * config.num_loops
    prepare_inference(model)
    _, tokenizer = tiny_model_and_tokenizer()
    snapshot = tmp_path / "loopquant"
    save_quantized_checkpoint(model, tokenizer, snapshot)
    engine = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                             quant_config=None, model_config=SimpleNamespace(model=str(snapshot), dtype=torch.bfloat16),
                             cache_config=None, device_config=SimpleNamespace(device="cpu"))
    bridge = bridge_class(vllm_config=engine)
    bridge.load_weights(iter(()))
    ids = torch.tensor([[3, 4, 3]])
    with torch.no_grad():
        expected = model(ids).logits[0].float()
        actual = bridge.compute_logits(bridge(ids[0], torch.arange(3)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoded = bridge.compute_logits(bridge(torch.tensor([4]), torch.tensor([3])))[0]
        expected = model(torch.tensor([[3, 4, 3, 4]])).logits[0, -1].float()
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    arguments, metadata = backend.build_vllm_arguments(snapshot, resolve_evaluation_config({}), num_loops=3)
    assert arguments["hf_overrides"]["architectures"] == ["LowRankOuroForCausalLM"]
    assert metadata["algorithm"] == "low_rank"
    monkeypatch.setenv("LOOPQUANT_VLLM_EXECUTION", "fast")
    with pytest.raises(ValueError, match="cached"):
        backend.build_vllm_arguments(snapshot, resolve_evaluation_config({}), num_loops=3)


def test_reconstruction_states_match_final_full_model_at_every_loop(tmp_path, monkeypatch):
    import loopquant.low_rank.reconstruction as training
    model, tokenizer = tiny(loops=3)
    config = configuration(tmp_path, loops=3)
    captured = {}
    original = training.run_block

    def record(model, block, hidden, loop):
        output = original(model, block, hidden, loop)
        if not torch.is_grad_enabled():
            index = next(i for i, candidate in enumerate(model.model.layers) if candidate is block)
            captured.setdefault((index, loop), []).append(output.clone())
        return output

    monkeypatch.setattr(training, "run_block", record)
    LowRankPipeline(config, device="cpu").run(model, tokenizer, DATA)
    actual, handles = {}, []
    for index, block in enumerate(model.model.layers):
        def hook(_block, _args, kwargs, output, key=index):
            actual[(key, kwargs["current_ut"])] = output
        handles.append(block.register_forward_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            model(DATA)
        for key, value in actual.items():
            torch.testing.assert_close(value, torch.cat(captured[key]), rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()


def test_token_cache_tracks_tokenizer_behavior_and_detects_corruption(tmp_path, monkeypatch):
    import loopquant.data as data_module
    from tokenizers.normalizers import Lowercase
    _, tokenizer = tiny_model_and_tokenizer()
    calls = []
    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **kw:
                        calls.append(1) or {"text": ["hello world hello world hello world"]})
    args = {"cache_dir": tmp_path, "num_samples": 2, "sequence_length": 3}
    first = data_module.load_calibration_data(tokenizer, **args)
    assert torch.equal(first, data_module.load_calibration_data(tokenizer, **args))
    assert len(calls) == 1
    tokenizer.backend_tokenizer.normalizer = Lowercase()
    data_module.load_calibration_data(tokenizer, **args)
    assert len(calls) == 2
    newest = max(tmp_path.glob("*.pt"), key=lambda path: path.stat().st_mtime_ns)
    payload = torch.load(newest, weights_only=True)
    payload["input_ids"][0, 0] += 1
    torch.save(payload, newest)
    with pytest.raises(ValueError, match="checksum"):
        data_module.load_calibration_data(tokenizer, **args)


def test_low_rank_cli_uses_same_save_then_evaluate_route(tmp_path, monkeypatch):
    import yaml
    from loopquant import cli
    model, tokenizer = tiny(layers=1)
    config = configuration(tmp_path)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    monkeypatch.setattr(cli, "load_looped_transformer", lambda *a, **kw: (model, tokenizer))
    monkeypatch.setattr(cli, "load_calibration_data", lambda *a, **kw: DATA)
    output = tmp_path / "output"
    events = []

    def evaluate(prepared, *args, **kwargs):
        manifest = json.loads((output / "loopquant/checkpoint_manifest.json").read_text(encoding="utf-8"))
        assert manifest["algorithm"] == "low_rank"
        assert manifest["allocation"] == [8, 8]
        assert prepared is model
        events.append("evaluate")

    monkeypatch.setattr(cli, "run_evaluation", evaluate)
    assert cli.main(["quantize", "--config", str(path), "--model", "unused", "--device", "cpu",
                     "--output-dir", str(output), "--activation-setting", "w4a8", "--save", "--zero-shot"]) == 0
    assert events == ["evaluate"]
    assert json.loads((output / "quantization_summary.json").read_text(encoding="utf-8"))["rank_schedule"] == [2, 4]


def test_all_low_rank_model_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "configs").glob("*_low_rank.yaml"))
    assert len(paths) == 4
    for path in paths:
        config = LoopQuantConfig.from_yaml(path)
        assert config.algorithm == "low_rank" and config.method["rank_step"] == 16
        assert "activation_bits" not in config.method
        assert config.output_dir == f"outputs/{path.stem}"
    assert {LoopQuantConfig.from_yaml(path).quantization["activation_setting"] for path in paths} == {"w4a4", "w4a8"}


@pytest.mark.parametrize("algorithm,setting", [("low_rank", "w4a6"), ("loopquant", "w4a8")])
def test_activation_presets_are_validated_for_the_selected_algorithm(tmp_path, algorithm, setting):
    values = configuration(tmp_path).to_dict() if algorithm == "low_rank" else {"num_loops": 4}
    values["quantization"] = {"activation_setting": setting}
    with pytest.raises(ValueError, match=f"{algorithm} activation_setting"):
        LoopQuantConfig.from_dict(values)
