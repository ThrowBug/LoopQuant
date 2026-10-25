"""Lazy vLLM construction; importing this module does not require vLLM."""

import os
from importlib.metadata import PackageNotFoundError, entry_points, version
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from .checkpoint import MANIFEST_NAME, read_checkpoint_manifest


VLLM_VERSION = "0.11.2"
MODEL_ARCHITECTURE = "LoopQuantOuroForCausalLM"


def register_vllm_models():
    """vLLM general-plugin entry point, also run in spawned worker processes."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(MODEL_ARCHITECTURE, "loopquant.vllm_model:LoopQuantOuroForCausalLM")


def check_vllm_environment(device):
    try:
        installed = version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError("Install requirements-vllm.txt in a separate Linux GPU environment") from error
    # Local build suffixes do not change the supported Python model API.
    if installed.split("+", 1)[0] != VLLM_VERSION:
        raise RuntimeError(f"Ouro integration requires vllm=={VLLM_VERSION}; found {installed}")
    if os.name == "nt" or not torch.cuda.is_available() or device not in {"cuda", "cuda:0"}:
        raise RuntimeError("vLLM requires Linux/WSL with CUDA; select the GPU using CUDA_VISIBLE_DEVICES")
    if os.environ.get("VLLM_USE_V1", "1") == "0":
        raise RuntimeError("LoopQuant vLLM evaluation requires the V1 engine; unset VLLM_USE_V1=0")


def build_vllm_arguments(source, config, *, num_loops, dtype=None):
    """Prepare explicit engine settings for original HF weights or our snapshot."""
    generation = config["generative"]
    if not generation["use_cache"]:
        raise ValueError("vLLM always uses its KV cache; generative.use_cache must be true")
    path = Path(source).expanduser()
    source = str(path.resolve()) if path.exists() else str(source)
    if not path.exists() and (path.is_absolute() or str(source).startswith((".", "~"))):
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    snapshot = path.is_dir() and (path / MANIFEST_NAME).exists()
    manifest = read_checkpoint_manifest(path, num_loops=num_loops) if snapshot else None
    hf_config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    if hf_config.model_type != "ouro":
        raise ValueError("the current vLLM integration supports Ouro only")
    overrides = {"total_ut_steps": num_loops}
    if manifest is not None:
        if dtype is not None:
            raise ValueError("quantized checkpoints preserve saved dtypes; omit dtype overrides")
        if int(hf_config.total_ut_steps) != num_loops:
            raise ValueError("checkpoint loop count conflicts with model config")
        name = "model.embed_tokens.weight"
        with safe_open(path / manifest["weight_map"][name], framework="pt") as tensors:
            dtype = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[
                tensors.get_slice(name).get_dtype()]
        overrides["architectures"] = [MODEL_ARCHITECTURE]
    arguments = {
        "pretrained": source, "dtype": dtype or "bfloat16", "trust_remote_code": True,
        "tensor_parallel_size": 1, "data_parallel_size": 1,
        "max_model_len": generation["max_length"], "batch_size": config["batch_size"],
        "max_num_seqs": max(config["batch_size"], generation["batch_size"]),
        "gpu_memory_utilization": 0.8, "seed": config["seed"],
        "hf_overrides": overrides, "enforce_eager": True,
        # Eager execution also excludes torch.compile: Python loop state and
        # LCTC's ContextVar must be read on every forward, not captured once.
        "compilation_config": {"level": 0},
        "enable_prefix_caching": False,
        # Weight dtype does not describe FP32 CGB tables, so never run a second
        # model-wide cast or online quantizer after restoring our saved tensors.
        "quantization": None,
    }
    metadata = {"mode": "quantized" if snapshot else "full_precision", "backend": "vllm",
                "model_source": source, "num_loops": num_loops, "fake_quantization": snapshot,
                "engine_arguments": arguments, "vllm_version": VLLM_VERSION}
    if manifest is not None:
        metadata["allocation"] = manifest["allocation"]
    return arguments, metadata


def load_vllm_backend(source, config, *, num_loops, dtype=None, device="cuda:0"):
    check_vllm_environment(device)
    arguments, metadata = build_vllm_arguments(source, config, num_loops=num_loops, dtype=dtype)
    if metadata["fake_quantization"]:
        if not any(entry.name == "loopquant" for entry in entry_points(group="vllm.general_plugins")):
            raise RuntimeError("Install requirements-vllm.txt, including its editable LoopQuant worker plugin")
        if "VLLM_PLUGINS" in os.environ:
            plugins = set(filter(None, os.environ["VLLM_PLUGINS"].split(",")))
            os.environ["VLLM_PLUGINS"] = ",".join(sorted(plugins | {"loopquant"}))
        register_vllm_models()
    from .vllm_adapter import EvaluationVLLM
    backend = EvaluationVLLM(allow_truncation=config["generative"]["allow_truncation"], **arguments)
    # Harness reloads the original config separately from the engine. Reflect
    # the effective loop/architecture overrides in evaluation artifacts too.
    for key, value in arguments["hf_overrides"].items():
        setattr(backend._config, key, value)
    return backend, metadata
