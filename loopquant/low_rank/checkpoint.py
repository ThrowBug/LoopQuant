"""Low-rank structure metadata; shared checkpoint I/O preserves exact tensors."""

from pathlib import Path

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ..config import LoopQuantConfig
from ..models import configure_looped_model
from .modules import LowRankLinear, wrap_linears, prepare_inference
from .quantization import activation_bits


def save_checkpoint(model, tokenizer, path, *, quantization_config=None, max_shard_bytes=1_000_000_000):
    from ..checkpoint import _runtime_state, _publish_checkpoint
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    config = LoopQuantConfig.from_dict(quantization_config or model._low_rank_config)
    modules = [module for module in model.modules() if isinstance(module, LowRankLinear)]
    if not modules or any(module.num_loops != config.num_loops for module in modules):
        raise ValueError("low_rank structure does not match checkpoint configuration")
    bits = activation_bits(config)
    allocation = (bits,) * config.num_loops
    if (model.config.total_ut_steps != config.num_loops or tuple(model._loopquant_allocation) != allocation
            or any(module.sharing != config.method["sharing"] or module.rank_step != config.method["rank_step"]
                   or module.group_size != config.quantization["group_size"] or module.activation_bits != bits
                   for module in modules)):
        raise ValueError("low_rank runtime and checkpoint configuration disagree")
    if any(parameter.requires_grad for module in modules for bank in (module.A, module.B) for parameter in bank):
        raise ValueError("freeze low_rank increments before saving")
    state, runtime = _runtime_state(model)
    manifest = {
        "format": "low-rank-fake-quantized", "format_version": 1, "algorithm": "low_rank",
        "fake_quantization": True, "num_loops": config.num_loops,
        "allocation": list(model._loopquant_allocation), "quantization_config": config.to_dict(),
        "rank_schedule": ([(loop + 1) * config.method["rank_step"] for loop in range(config.num_loops)]
                          if config.method["sharing"] == "prefix" else [config.method["rank_step"]] * config.num_loops),
        "linears": [{"path": name, "in_features": module.in_features, "out_features": module.out_features,
                     "sharing": module.sharing, "rank_step": module.rank_step, "group_size": module.group_size}
                    for name, module in model.named_modules() if isinstance(module, LowRankLinear)],
        "has_rotation": False, "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "model_source": getattr(model, "_loopquant_model_path", None), **runtime,
    }
    return _publish_checkpoint(model, tokenizer, target, state, manifest, max_shard_bytes)


def restore_structure(model, manifest):
    config = LoopQuantConfig.from_dict(manifest["quantization_config"])
    if config.algorithm != "low_rank" or config.num_loops != manifest["num_loops"]:
        raise ValueError("low_rank manifest and method configuration disagree")
    if [activation_bits(config)] * config.num_loops != manifest["allocation"]:
        raise ValueError("low_rank activation allocation metadata disagrees")
    ranks = ([(loop + 1) * config.method["rank_step"] for loop in range(config.num_loops)]
             if config.method["sharing"] == "prefix" else [config.method["rank_step"]] * config.num_loops)
    if ranks != manifest["rank_schedule"]:
        raise ValueError("low_rank rank schedule metadata disagrees")
    wrap_linears(model, config)
    actual = {name: module for name, module in model.named_modules() if isinstance(module, LowRankLinear)}
    if set(actual) != {spec["path"] for spec in manifest["linears"]}:
        raise ValueError("low_rank checkpoint linear paths differ")
    for spec in manifest["linears"]:
        module = actual[spec["path"]]
        if any(getattr(module, key) != spec[key]
               for key in ("in_features", "out_features", "sharing", "rank_step", "group_size")):
            raise ValueError(f"low_rank linear metadata differs: {spec['path']}")
    model._quantization_algorithm = "low_rank"
    model._low_rank_config = config.to_dict()
    model._loopquant_num_loops = config.num_loops
    model._loopquant_allocation = tuple(manifest["allocation"])


def load_checkpoint(directory, manifest, *, device):
    from ..checkpoint import restore_checkpoint_tensors
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    if config.model_type != "ouro" or config.total_ut_steps != manifest["num_loops"]:
        raise ValueError("low_rank model config and saved loop count disagree")
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                               attn_implementation=manifest["attn_implementation"])
        restore_structure(model, manifest)
    restore_checkpoint_tensors(model, directory, manifest, device=device)
    configure_looped_model(model, manifest["num_loops"], quantization_hooks=False)
    model._loopquant_model_path = str(directory)
    prepare_inference(model)
    if (directory / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    return model, tokenizer
