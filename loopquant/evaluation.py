"""Evaluation of prepared models, independent of how they were loaded or quantized."""

from __future__ import annotations

import math
import logging
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset

from .evaluation_config import resolve_evaluation_config
from .evaluation_results import save_harness_result, save_json


ZERO_SHOT_TASKS = [
    "arc_challenge", "arc_easy", "hellaswag", "lambada_openai",
    "lambada_standard", "piqa", "winogrande",
]


def _validate_suites(suites):
    if not suites or set(suites) - {"wikitext2", "c4", "zero_shot", "generative"}:
        raise ValueError("suites must select wikitext2, c4, zero_shot or generative")


def preflight_evaluation(config, *, suites):
    """Check dependencies/tasks/scorers before starting evaluation.

    Task loading may download datasets. MBPP also runs its code-eval self-check.
    Returns a task manager for reuse, or None for PPL-only evaluation.
    """
    _validate_suites(suites)
    config = resolve_evaluation_config(config)
    generation = config["generative"]
    entries = generation["tasks"] if "generative" in suites else []
    if "generative" in suites and not entries:
        raise ValueError("generative evaluation requires tasks")
    if any(task["name"].startswith("mbpp") for task in entries):
        if os.name == "nt":
            raise RuntimeError("MBPP code_eval requires Linux/WSL; native Windows is unsupported")
        if not generation["confirm_run_unsafe_code"] or os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
            raise RuntimeError(
                "MBPP requires generative.confirm_run_unsafe_code=true and "
                "HF_ALLOW_CODE_EVAL=1 in the isolated execution environment"
            )
    if not entries and "zero_shot" not in suites:
        return None
    try:
        from lm_eval.tasks import TaskManager
        # Check the backend import too; a base harness install is insufficient.
        from .harness_adapter import EvaluationHFLM  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install the pinned evaluation requirements before running evaluation") from error
    manager = TaskManager()
    if "zero_shot" in suites:
        manager.load(ZERO_SHOT_TASKS)
    for task in entries:
        loaded = manager.load([task["name"]])
        if not loaded["tasks"]:
            raise ValueError(f"No tasks resolved for {task['name']}")
        for name, instance in loaded["tasks"].items():
            if instance.get_config("output_type") != "generate_until":
                raise ValueError(f"{name} is not a generate_until task")
    return manager


def _evaluation_versions():
    result = {}
    for package in ("torch", "transformers", "lm-eval", "evaluate", "datasets", "huggingface-hub"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def run_evaluation(model, tokenizer, config, output_dir, *, suites, device="cuda:0",
                   run_metadata=None, task_manager=None):
    """Evaluate an already prepared HF model. Never loads weights or runs PTQ.

    Callers restoring custom checkpoints own their modules and runtime state.
    No LoopQuant globals, bit allocations, or model weights are modified here.
    Results are written after every suite/task; failures are saved and re-raised.
    """
    _validate_suites(suites)
    config = resolve_evaluation_config(config)
    output = Path(output_dir)
    logger = logging.getLogger("loopquant")
    started = time.time()
    manifest = {
        "status": "running", "evaluation": config, "suites": list(suites),
        "run_metadata": run_metadata or {}, "versions": _evaluation_versions(),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_config": model.config.to_dict() if hasattr(model.config, "to_dict") else str(model.config),
        "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        "full_eval": config["limit"] is None, "completed": [],
    }
    metrics = {}
    active = "preflight"
    save_json(output / "evaluation_manifest.json", manifest)
    try:
        if task_manager is None:
            task_manager = preflight_evaluation(config, suites=suites)
        model.eval()
        for dataset in ("wikitext2", "c4"):
            if dataset in suites:
                active = dataset
                logger.info("[Evaluation] %s perplexity", dataset)
                metrics[dataset] = evaluate_perplexity(
                    model, tokenizer, dataset, device=device,
                    sequence_length=config["sequence_length"], limit=config["limit"],
                )
                if not math.isfinite(metrics[dataset]["perplexity"]):
                    raise FloatingPointError(f"non-finite {dataset} perplexity")
                manifest["completed"].append(dataset)
                save_json(output / "metrics.json", metrics)
                save_json(output / "evaluation_manifest.json", manifest)
        generation = config["generative"]
        if "zero_shot" in suites or "generative" in suites:
            from lm_eval import evaluator
            from .harness_adapter import EvaluationHFLM

            adapter = EvaluationHFLM(
                pretrained=model, tokenizer=tokenizer, device=device,
                backend="causal", trust_remote_code=True,
                batch_size=config["batch_size"], max_length=generation["max_length"],
                generation_use_cache=generation["use_cache"],
                allow_truncation=generation["allow_truncation"],
            )
            seed = config["seed"]
            harness_tasks = []
            if "zero_shot" in suites:
                harness_tasks.append(("zero_shot", ZERO_SHOT_TASKS,
                               0, config["batch_size"], None))
            if "generative" in suites:
                harness_tasks.extend((entry["name"], [entry["name"]], entry.get("num_fewshot"),
                               generation["batch_size"], entry["max_gen_toks"])
                              for entry in generation["tasks"])
            for name, tasks, shots, batch_size, max_tokens in harness_tasks:
                active = name
                logger.info("[Evaluation] %s (fewshot=%s, batch=%s)", name, shots, batch_size)
                adapter.batch_size_per_gpu = batch_size
                adapter.generation_receipts.clear()
                result = evaluator.simple_evaluate(
                    model=adapter, tasks=tasks, task_manager=task_manager,
                    num_fewshot=shots, batch_size=batch_size, device=device,
                    limit=config["limit"], random_seed=seed, numpy_random_seed=seed,
                    torch_random_seed=seed, fewshot_random_seed=generation["fewshot_seed"],
                    apply_chat_template=generation["apply_chat_template"] if max_tokens else False,
                    fewshot_as_multiturn=False,
                    gen_kwargs={"do_sample": False, "max_gen_toks": max_tokens} if max_tokens else None,
                    confirm_run_unsafe_code=generation["confirm_run_unsafe_code"],
                    log_samples=generation["save_samples"],
                )
                if result is None:
                    raise RuntimeError("No harness result returned; this runner expects one process")
                directory = output / ("zero_shot" if name == "zero_shot" else "generative")
                if name != "zero_shot":
                    directory /= name
                summary = save_harness_result(directory, result, generation["save_samples"])
                save_json(directory / "generation_receipts.json", adapter.generation_receipts)
                if name == "zero_shot":
                    metrics[name] = {"tasks": tasks, "limit": config["limit"],
                                     "results": summary.get("results", {})}
                else:
                    metrics.setdefault("generative", {})[name] = summary.get("results", {})
                manifest["completed"].append(name)
                save_json(output / "metrics.json", metrics)
                save_json(output / "evaluation_manifest.json", manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", failed_task=active,
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["runtime_seconds"] = time.time() - started
        save_json(output / "evaluation_manifest.json", manifest)
        save_json(output / "metrics.json", metrics)
    return metrics


@torch.no_grad()
def calibration_nll(model, sequences: torch.Tensor | list[torch.Tensor], device: str) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    for sequence in sequences:
        input_ids = sequence.unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        labels = input_ids[:, 1:]
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_nll += float(loss)
        total_tokens += int(labels.numel())
    mean = total_nll / max(total_tokens, 1)
    return {"nll": mean, "ppl": math.exp(mean), "tokens": total_tokens}


def _tokenize_wikitext2_test(tokenizer) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer(
        "\n\n".join(dataset["text"]), add_special_tokens=False, return_tensors="pt"
    ).input_ids[0]


def _tokenize_c4_validation(tokenizer) -> torch.Tensor:
    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-validation.00000-of-00008.json.gz",
        split="train",
    )
    return tokenizer(" ".join(dataset[:1100]["text"]), return_tensors="pt").input_ids[0][: 256 * 2048]


@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext2",
    device: str = "cuda:0",
    sequence_length: int = 2048,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate full fixed windows using the recurrent model forward path."""
    if dataset.lower() in {"wikitext2", "wikitext-2"}:
        tokens = _tokenize_wikitext2_test(tokenizer)
        name = "wikitext2"
    elif dataset.lower() == "c4":
        tokens = _tokenize_c4_validation(tokenizer)
        name = "c4"
    else:
        raise ValueError("dataset must be wikitext2 or c4")
    windows = math.ceil(tokens.numel() / int(sequence_length))
    if limit is not None:
        windows = min(windows, int(limit))
    if windows == 0:
        raise RuntimeError("evaluation corpus does not contain one complete window")
    total_nll = 0.0
    total_targets = 0
    evaluated_windows = 0
    for index in range(windows):
        window = tokens[index * sequence_length : (index + 1) * sequence_length]
        if window.numel() < 2:
            continue
        batch = window.unsqueeze(0).to(device)
        logits = model(input_ids=batch, use_cache=False).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss)
        total_targets += int(batch.shape[1] - 1)
        evaluated_windows += 1
    mean_nll = total_nll / total_targets
    return {
        "dataset": name,
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "windows": evaluated_windows,
        "sequence_length": int(sequence_length),
        "target_tokens": total_targets,
        "formal_full_dataset": limit is None,
    }
