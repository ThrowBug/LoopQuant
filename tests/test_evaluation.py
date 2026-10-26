"""Protocol/runner regressions plus real local HF generation (no downloads)."""

import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from loopquant.evaluation_config import load_evaluation_config, resolve_evaluation_config
from loopquant import evaluation
from loopquant.evaluation_results import save_harness_result


def protocol():
    return {
        "limit": 2,
        "generative": {"tasks": [
            {"name": "gsm8k_cot", "num_fewshot": 8, "max_gen_toks": 64},
            {"name": "bbh_cot_fewshot", "num_fewshot": None, "max_gen_toks": 64},
        ]},
    }


def test_protocol_and_old_configs():
    root = Path(__file__).resolve().parents[1]
    from loopquant.config import LoopQuantConfig
    for path in (root / "configs").glob("*.yaml"):
        if path.name.startswith("evaluation_"):
            config = load_evaluation_config(path)
            assert len(config["generative"]["tasks"]) == 5
            assert next(t for t in config["generative"]["tasks"] if t["name"] == "drop")["num_fewshot"] == 1
        else:
            config = LoopQuantConfig.from_yaml(path)
            assert "evaluation" not in config.to_dict()
    resolved = resolve_evaluation_config(protocol())
    assert resolved["generative"]["tasks"][1]["num_fewshot"] is None


@pytest.mark.parametrize("change", [
    {"full_eval": True, "limit": 2},
    {"limit": -1},
    {"generative": {"enabled": True}},
    {"ppl_batch_size": 16},
    {"wikitext2": True},
    {"c4": True},
    {"zero_shot": True},
    {"generative": {"tasks": [{"name": "../bad", "max_gen_toks": 1}]}},
    {"generative": {"tasks": [{"name": "gsm8k_cot", "max_gen_toks": 4096}]}},
])
def test_invalid_protocol(change):
    with pytest.raises(ValueError):
        resolve_evaluation_config(change)


def test_generative_requires_task_selection():
    with pytest.raises(ValueError, match="requires tasks"):
        evaluation.preflight_evaluation({}, suites=["generative"])
    assert evaluation.preflight_evaluation({}, suites=["wikitext2"]) is None


def test_drop_renders_real_reference_in_one_shot_prompt(monkeypatch):
    import datasets

    def document(question, number="", spans=None, date=None):
        return {
            "query_id": question, "passage": "The passage.", "question": question,
            "answer": {"number": number, "spans": spans or [],
                       "date": date or {"day": "", "month": "", "year": ""},
                       "worker_id": "", "hit_id": ""},
            "validated_answers": {"number": [], "spans": [], "date": []},
        }

    data = datasets.DatasetDict({
        "train": datasets.Dataset.from_list([
            document("How many people per 12 square miles?", number="1320"),
            document("How many inhabitants per 12 square miles?", number="1320"),
        ]),
        "validation": datasets.Dataset.from_list([
            document("How many field goals?", number="3"),
            document("Who scored?", spans=["Chaz Schilens", "Johnnie Lee Higgins"]),
            document("On what date?", date={"day": "3", "month": "October", "year": "2026"}),
        ]),
    })
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: data)
    config = {"generative": {"tasks": [
        {"name": "drop", "num_fewshot": 1, "max_gen_toks": 128},
    ]}}
    manager = evaluation.preflight_evaluation(config, suites=["generative"])
    task = evaluation._load_tasks(manager, ["drop"])["drop"]
    docs = list(task.validation_docs())
    assert [task.doc_to_target(doc) for doc in docs] == [
        "3", "Chaz Schilens, Johnnie Lee Higgins", "3 October 2026",
    ]
    task.set_fewshot_seed(0)
    prompt = task.fewshot_context(docs[0], num_fewshot=1)
    assert "12 square miles?1320" in prompt
    assert prompt.endswith("How many field goals?")
    assert "number,date,spans,worker_id,hit_id" not in prompt
    assert task.process_results(docs[0], ["3"]) == {"em": 1.0, "f1": 1.0}
    assert task.get_config("generation_kwargs")["until"] == ["."]


@pytest.mark.parametrize("backend_name", ["hf", "vllm"])
def test_drop_target_override_reaches_both_backends(tmp_path, monkeypatch, backend_name):
    import lm_eval.evaluator
    import loopquant.harness_adapter

    calls = []
    adapter = SimpleNamespace(generation_receipts=[], batch_size_per_gpu=1, batch_size=1)
    monkeypatch.setattr(loopquant.harness_adapter, "EvaluationHFLM", lambda **kwargs: adapter)
    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", lambda **kwargs:
                        calls.append(kwargs) or {"results": {"drop": {"em": 1.0, "f1": 1.0}}})
    config = {"generative": {"tasks": [
        {"name": "drop", "num_fewshot": 1, "max_gen_toks": 128},
    ]}}
    evaluation.run_evaluation(
        SimpleNamespace(config=SimpleNamespace(), eval=lambda: None), SimpleNamespace(),
        config, tmp_path, suites=["generative"], task_manager=object(), device="cpu",
        lm_backend=adapter if backend_name == "vllm" else None, backend_name=backend_name,
    )
    assert calls[0]["tasks"][0]["task"] == "drop"
    assert calls[0]["tasks"][0]["doc_to_target"] == "{{ answers[0] | join(', ') }}"
    assert calls[0]["num_fewshot"] == 1


def test_runner_forwards_protocol_and_preserves_partial_results(tmp_path, monkeypatch):
    import lm_eval.evaluator
    import loopquant.harness_adapter
    from loopquant.quantization import QUANTIZATION_CONTEXT

    calls = []
    instance = SimpleNamespace(generation_receipts=[], batch_size_per_gpu=1)
    monkeypatch.setattr(loopquant.harness_adapter, "EvaluationHFLM", lambda **kw: instance)

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("test interruption")
        return {"results": {"gsm8k_cot": {"exact_match": 0.5}},
                "configs": {"gsm8k_cot": {"num_fewshot": 8}},
                "samples": {"gsm8k_cot": [{"doc_id": 0, "resps": [["answer"]]}]}}

    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", fake_evaluate)
    model = SimpleNamespace(config=SimpleNamespace(), eval=lambda: None)
    before = (QUANTIZATION_CONTEXT.enabled, QUANTIZATION_CONTEXT.activation_bits)
    with pytest.raises(RuntimeError, match="test interruption"):
        evaluation.run_evaluation(model, SimpleNamespace(), protocol(), tmp_path,
                                  task_manager=object(), suites=["generative"], device="cpu")
    assert calls[0]["num_fewshot"] == 8
    assert calls[1]["num_fewshot"] is None
    assert calls[0]["gen_kwargs"] == {"do_sample": False, "max_gen_toks": 64}
    assert calls[0]["fewshot_random_seed"] == 0
    assert not calls[0]["apply_chat_template"]
    assert (QUANTIZATION_CONTEXT.enabled, QUANTIZATION_CONTEXT.activation_bits) == before
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed"] == ["gsm8k_cot"]
    assert (tmp_path / "generative/gsm8k_cot/samples.jsonl").exists()
    assert "gsm8k_cot" in json.loads((tmp_path / "metrics.json").read_text())["generative"]


def test_mbpp_execution_gate():
    config = protocol()
    config["generative"]["tasks"] = [{"name": "mbpp", "max_gen_toks": 64}]
    with pytest.raises(RuntimeError, match="MBPP"):
        evaluation.preflight_evaluation(config, suites=["generative"])


def test_stale_samples_removed(tmp_path):
    result = {"results": {}, "samples": {"task": [{"doc_id": 1}]}}
    save_harness_result(tmp_path, result, True)
    assert (tmp_path / "samples.jsonl").exists()
    save_harness_result(tmp_path, result, False)
    assert not (tmp_path / "samples.jsonl").exists()


def test_evaluate_command_skips_pipeline(tmp_path, monkeypatch):
    from loopquant import cli
    from loopquant.pipeline import LoopQuantPipeline
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump({"evaluation": {}}))
    calls = []
    monkeypatch.setattr(cli, "preflight_evaluation", lambda config, **kwargs: "manager")
    monkeypatch.setattr(cli, "load_model_for_evaluation", lambda *a, **k:
                        ("model", "tokenizer", {"mode": "full_precision"}))
    monkeypatch.setattr(cli, "run_evaluation", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(LoopQuantPipeline, "run", lambda *a, **k: pytest.fail("BF16 invoked PTQ"))
    assert cli.main(["evaluate", "--model", "unused", "--eval-config", str(path), "--wikitext2",
                     "--output-dir", str(tmp_path / "out"), "--device", "cpu", "--num-loops", "4"]) == 0
    assert calls[0][0][:2] == ("model", "tokenizer")
    assert calls[0][1]["run_metadata"]["mode"] == "full_precision"


def tiny_model_and_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                                   "hello": 3, "world": 4, "answer": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        eos_token="[EOS]", unk_token="[UNK]", model_max_length=64)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=6, n_positions=64, n_embd=16,
                                      n_layer=1, n_head=2, bos_token_id=1,
                                      eos_token_id=1, pad_token_id=0)).eval()
    return model, tokenizer


def test_real_hf_cached_and_uncached_generation():
    from lm_eval.api.instance import Instance
    from loopquant.harness_adapter import EvaluationHFLM
    model, tokenizer = tiny_model_and_tokenizer()
    adapter = EvaluationHFLM(pretrained=model, tokenizer=tokenizer, backend="causal",
                             device="cpu", max_length=64, batch_size=1)
    request = Instance(request_type="generate_until", doc={},
                       arguments=("hello world", {"max_gen_toks": 4, "do_sample": False, "until": []}),
                       idx=0)
    cached = adapter.generate_until([request])
    adapter.generation_use_cache = False
    uncached = adapter.generate_until([request])
    assert cached == uncached
    assert len(adapter.generation_receipts) == 2
    assert adapter.generation_receipts[0]["use_cache"] is True
    assert adapter.generation_receipts[1]["use_cache"] is False
    with pytest.raises(ValueError, match="Prompt requires"):
        adapter.tok_batch_encode(["hello world"], left_truncate_len=1)
    adapter.allow_prompt_truncation = True
    ids, _ = adapter.tok_batch_encode(["hello world"], left_truncate_len=1)
    assert ids.shape[1] == 1


def test_lctc_wrapper_keeps_generation_signature():
    from loopquant.lctc import apply_lctc_calibration
    model, _ = tiny_model_and_tokenizer()
    signature = inspect.signature(model.forward)
    apply_lctc_calibration(model, torch.nn.Module())
    assert inspect.signature(model.forward) == signature


@pytest.fixture
def short_dataset_cache():
    # datasets embeds the absolute cache path into its lock filename. A regular
    # pytest tmp_path can exceed Windows' 260-character path limit after that.
    with tempfile.TemporaryDirectory(prefix="lq") as directory:
        yield directory


def test_real_harness_local_dataset(tmp_path, short_dataset_cache, monkeypatch):
    import lm_eval.tasks
    from lm_eval.tasks import TaskManager
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"question": "hello world", "answer": "answer"}) + "\n")
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "local_generation.yaml").write_text(yaml.safe_dump({
        "task": "local_generation", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(data)},
                           "cache_dir": short_dataset_cache},
        "test_split": "test", "output_type": "generate_until",
        "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
        "num_fewshot": 0,
        "generation_kwargs": {"do_sample": False, "until": []},
        # Avoid downloading an external metric script.
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
        "process_results": "!unused",
    }))
    # A local function avoids metric downloads and makes the whole harness path
    # deterministic even when the randomly initialized LM answers incorrectly.
    (task_dir / "local_metrics.py").write_text(
        "def score(doc, results):\n    return {'acc': float(results[0].strip() == doc['answer'])}\n"
    )
    yaml_path = task_dir / "local_generation.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("'!unused'", "!function local_metrics.score"))
    (task_dir / "local_group.yaml").write_text(yaml.safe_dump({
        "group": "local_group", "task": ["local_generation"],
    }))
    manager = TaskManager(include_path=str(task_dir), include_defaults=False)
    monkeypatch.setattr(lm_eval.tasks, "TaskManager", lambda: manager)
    # Exercise the public loader and nested group format on both harness versions.
    assert evaluation.preflight_evaluation({"generative": {"tasks": [
        {"name": "local_group", "num_fewshot": 0, "max_gen_toks": 4},
    ]}}, suites=["generative"]) is manager
    model, tokenizer = tiny_model_and_tokenizer()
    config = {"generative": {"max_length": 64,
                              "tasks": [{"name": "local_generation", "num_fewshot": 0,
                                         "max_gen_toks": 4}]}}
    metrics = evaluation.run_evaluation(model, tokenizer, config, tmp_path / "output",
                                        device="cpu", suites=["generative"], task_manager=manager)
    assert "local_generation" in metrics["generative"]
    saved = tmp_path / "output/generative/local_generation"
    assert json.loads((saved / "generation_receipts.json").read_text())[0]["use_cache"]
    assert (saved / "samples.jsonl").read_text().strip()
    assert json.loads((tmp_path / "output/evaluation_manifest.json").read_text())["status"] == "complete"
