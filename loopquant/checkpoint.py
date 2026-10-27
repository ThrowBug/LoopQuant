"""Self-contained, versioned fake-quantized Ouro snapshots (not packed INT4)."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.dynamic_module_utils import custom_object_save

from .evaluation_results import save_json
from .lctc import LoopConditionedCGB, apply_lctc_calibration
from .models import configure_looped_model
from .quantization import QUANTIZATION_CONTEXT, QuantizedLinear, _RotatedRMSNorm


MANIFEST_NAME = "checkpoint_manifest.json"
FORMAT = "loopquant-fake-quantized"
FORMAT_VERSION = 1


def _checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or invalid checkpoint file: {name}")
    return path


def _copy_model_code(obj, directory, config, auto_class):
    if not type(obj).__module__.startswith("transformers."):
        if type(obj).__module__ == "__main__":
            raise ValueError("checkpoint model/config classes must be defined in an importable module")
        custom_object_save(obj, directory)
        module = sys.modules[type(obj).__module__]
        config.auto_map = dict(getattr(config, "auto_map", None) or {})
        config.auto_map[auto_class] = f"{Path(module.__file__).stem}.{type(obj).__name__}"


def _write_weights(state, directory, max_shard_bytes):
    weight_map, shard, size, index = {}, {}, 0, 1

    def flush():
        nonlocal shard, size, index
        if shard:
            filename = f"weights-{index:05d}.safetensors"
            save_file(shard, directory / filename)
            weight_map.update({name: filename for name in shard})
            shard, size, index = {}, 0, index + 1

    for name, tensor in state.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if shard and size + tensor_size > max_shard_bytes:
            flush()
        # Independent CPU storage handles tied/view tensors and bounds transient
        # copies to one shard, without cloning the entire model on the GPU.
        shard[name] = tensor.detach().to(device="cpu", copy=True).contiguous()
        size += tensor_size
    flush()
    return weight_map


def save_quantized_checkpoint(model, tokenizer, path, *, quantization_config=None,
                              max_shard_bytes=1_000_000_000):
    """Publish a complete snapshot atomically. Never overwrite an existing path."""
    if getattr(model, "_quantization_algorithm", "loopquant") == "low_rank":
        from .low_rank.checkpoint import save_checkpoint
        return save_checkpoint(model, tokenizer, path, quantization_config=quantization_config,
                               max_shard_bytes=max_shard_bytes)
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    loops = int(model._loopquant_num_loops)
    allocation = tuple(model._loopquant_allocation)
    if (QUANTIZATION_CONTEXT.num_loops != loops or
            QUANTIZATION_CONTEXT.activation_bits != allocation or
            not QUANTIZATION_CONTEXT.enabled):
        raise RuntimeError("active quantization context does not match the model being saved")
    linears, norms, runtime_tensors, rope_scalars = [], [], [], []
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            linears.append({
                "path": name, "name": module.name, "group_size": module.group_size,
                "in_features": module.in_features, "out_features": module.out_features,
                "cldc_hadamard": module.cldc_hadamard,
                "has_scale": module.cldc_scale is not None,
                "permutation_size": module.cldc_permutation.numel(),
            })
        elif isinstance(module, _RotatedRMSNorm):
            norms.append({"path": name, "rotation_shape": list(module.rotation.shape)})
        # RoPE uses non-persistent buffers and plain tensor attributes (e.g.
        # original_inv_freq). state_dict alone omits both. Save exact values so
        # restoration never leaves meta buffers or resets an expanded RoPE cache.
        extra = {key: value for key, value in module._buffers.items()
                 if key in module._non_persistent_buffers_set and value is not None}
        extra.update({key: value for key, value in vars(module).items()
                      if isinstance(value, torch.Tensor) and key != "_loopquant_rotation"})
        for attribute, tensor in extra.items():
            key = f"__runtime_tensor_{len(runtime_tensors)}"
            runtime_tensors.append({"key": key, "module": name, "attribute": attribute})
            state[key] = tensor
        for attribute in ("max_seq_len_cached", "original_max_seq_len", "attention_scaling"):
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float, bool)):
                rope_scalars.append({"module": name, "attribute": attribute, "value": value})
    if not linears:
        raise ValueError("model has no QuantizedLinear modules")
    aliases, seen = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            aliases[name] = seen[id(parameter)]
        else:
            seen[id(parameter)] = name
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "fake_quantization": True,
        "num_loops": loops, "allocation": list(allocation),
        "linears": linears, "rotated_norms": norms, "parameter_aliases": aliases,
        "runtime_tensors": runtime_tensors,
        "rope_scalars": rope_scalars,
        "has_lctc": hasattr(model, "lctc_cgb"),
        "lctc_enabled": getattr(getattr(model, "_loopquant_lctc_runtime", None), "enabled", False),
        "has_rotation": hasattr(model, "_loopquant_rotation"),
        "rotation_seed": getattr(model, "_loopquant_rotation_seed", None),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "quantization_config": quantization_config,
        "model_source": getattr(model, "_loopquant_model_path", None),
    }
    if manifest["has_rotation"]:
        state["__loopquant_rotation"] = model._loopquant_rotation
    return _publish_checkpoint(model, tokenizer, target, state, manifest, max_shard_bytes)


def _runtime_state(model):
    """Capture non-persistent runtime state shared by both checkpoint formats."""
    state, runtime_tensors, rope_scalars = dict(model.state_dict()), [], []
    for name, module in model.named_modules():
        extra = {key: value for key, value in module._buffers.items()
                 if key in module._non_persistent_buffers_set and value is not None}
        extra.update({key: value for key, value in vars(module).items() if isinstance(value, torch.Tensor)})
        for attribute, tensor in extra.items():
            key = f"__runtime_tensor_{len(runtime_tensors)}"
            runtime_tensors.append({"key": key, "module": name, "attribute": attribute})
            state[key] = tensor
        for attribute in ("max_seq_len_cached", "original_max_seq_len", "attention_scaling"):
            value = getattr(module, attribute, None)
            if isinstance(value, (int, float, bool)):
                rope_scalars.append({"module": name, "attribute": attribute, "value": value})
    aliases, seen = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            aliases[name] = seen[id(parameter)]
        else:
            seen[id(parameter)] = name
    return state, {"runtime_tensors": runtime_tensors, "rope_scalars": rope_scalars,
                   "parameter_aliases": aliases}


def _publish_checkpoint(model, tokenizer, target, state, manifest, max_shard_bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        directory = Path(temporary)
        config = deepcopy(model.config)
        _copy_model_code(config, directory, config, "AutoConfig")
        _copy_model_code(model, directory, config, "AutoModelForCausalLM")
        config.save_pretrained(directory)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        manifest["weight_map"] = _write_weights(state, directory, max_shard_bytes)
        manifest["files"] = {file.relative_to(directory).as_posix(): _checksum(file)
                             for file in directory.rglob("*") if file.is_file()}
        save_json(directory / MANIFEST_NAME, manifest)
        # TemporaryDirectory only owns the staging path, never the published one.
        directory.rename(target)
    return target


def _restore_structure(model, manifest):
    """Create wrappers only: saved weights already contain every transformation."""
    for spec in manifest["rotated_norms"]:
        path = spec["path"]
        model.set_submodule(path, _RotatedRMSNorm(
            model.get_submodule(path), torch.empty(spec["rotation_shape"], device="meta")))
    for spec in manifest["linears"]:
        path = spec["path"]
        original = model.get_submodule(path)
        if not isinstance(original, nn.Linear) or (
            original.in_features != spec["in_features"] or original.out_features != spec["out_features"]
        ):
            raise ValueError(f"checkpoint linear structure mismatch: {path}")
        replacement = QuantizedLinear(original, spec["name"], spec["group_size"])
        replacement.cldc_hadamard = spec["cldc_hadamard"]
        if spec["has_scale"]:
            replacement.cldc_scale = nn.Parameter(torch.empty(
                original.in_features, device="meta", dtype=original.weight.dtype), requires_grad=False)
        replacement.cldc_permutation = torch.empty(spec["permutation_size"], dtype=torch.long, device="meta")
        model.set_submodule(path, replacement)
    if manifest["has_lctc"]:
        model.lctc_cgb = LoopConditionedCGB.from_model(model, manifest["num_loops"])


def read_checkpoint_manifest(path, *, num_loops=None):
    """Validate a snapshot before either HF or vLLM constructs its modules."""
    directory = Path(path).expanduser().resolve()
    manifest = json.loads(_local_file(directory, MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") not in {FORMAT, "low-rank-fake-quantized"} or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported LoopQuant checkpoint format/version")
    if num_loops is not None and num_loops != manifest["num_loops"]:
        raise ValueError(f"num_loops={num_loops} conflicts with saved num_loops={manifest['num_loops']}")
    if not set(manifest["weight_map"].values()).issubset(manifest["files"]):
        raise ValueError("weight index references an unverified shard")
    for name, expected in manifest["files"].items():
        if _checksum(_local_file(directory, name)) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return manifest


def restore_checkpoint_tensors(model, path, manifest, *, device):
    """Restore exact tensors/dtypes into an already reconstructed HF skeleton."""
    directory = Path(path).expanduser().resolve()
    expected_keys = set(model.state_dict())
    weights = manifest["weight_map"]
    if manifest["has_rotation"] != ("__loopquant_rotation" in weights):
        raise ValueError("checkpoint rotation metadata does not match tensors")
    extras = {spec["key"]: spec for spec in manifest["runtime_tensors"]}
    if not set(extras).issubset(weights):
        raise ValueError("checkpoint is missing runtime tensors")
    model_keys = set(weights) - {"__loopquant_rotation"} - set(extras)
    if expected_keys != model_keys:
        raise ValueError(f"checkpoint tensor keys differ: missing={expected_keys - model_keys}, "
                         f"unexpected={model_keys - expected_keys}")
    for filename in sorted(set(weights.values())):
        if filename not in manifest["files"]:
            raise ValueError(f"unverified weight file: {filename}")
        tensors = load_file(_local_file(directory, filename), device="cpu")
        if set(tensors) != {name for name, file in weights.items() if file == filename}:
            raise ValueError(f"weight index does not match shard: {filename}")
        rotation = tensors.pop("__loopquant_rotation", None)
        if rotation is not None:
            model._loopquant_rotation = rotation.to(device)
        for key in list(tensors):
            if key in extras:
                spec = extras[key]
                setattr(model.get_submodule(spec["module"]), spec["attribute"], tensors.pop(key).to(device))
        model.load_state_dict(tensors, strict=False, assign=True)
        del tensors
    for name, canonical in manifest["parameter_aliases"].items():
        owner, _, leaf = name.rpartition(".")
        model.get_submodule(owner)._parameters[leaf] = model.get_parameter(canonical)
    for spec in manifest["rope_scalars"]:
        setattr(model.get_submodule(spec["module"]), spec["attribute"], spec["value"])
    model.to(device).requires_grad_(False).eval()


def load_quantized_checkpoint(path, *, device="cuda:0", num_loops=None):
    """Restore a snapshot without original weights, teacher, data or PTQ."""
    from accelerate import init_empty_weights

    directory = Path(path).expanduser().resolve()
    manifest = read_checkpoint_manifest(directory, num_loops=num_loops)
    if manifest["format"] == "low-rank-fake-quantized":
        from .low_rank.checkpoint import load_checkpoint
        return load_checkpoint(directory, manifest, device=device)
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    if config.model_type != "ouro":
        raise ValueError("checkpoint format v1 supports Ouro only")
    loops, allocation = int(manifest["num_loops"]), manifest["allocation"]
    if int(config.total_ut_steps) != loops:
        raise ValueError("checkpoint loop count conflicts with model config")
    # Restore buffers to the correct loop shape before creating QuantizedLinear.
    previous = (QUANTIZATION_CONTEXT.num_loops, QUANTIZATION_CONTEXT.current_loop,
                QUANTIZATION_CONTEXT.activation_bits, QUANTIZATION_CONTEXT.enabled)
    try:
        QUANTIZATION_CONTEXT.configure(loops, allocation)
        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True,
                attn_implementation=manifest["attn_implementation"],
            )
            _restore_structure(model, manifest)
        restore_checkpoint_tensors(model, directory, manifest, device=device)
        model._loopquant_rotation_seed = manifest["rotation_seed"]
        model._loopquant_model_path = str(directory)
        model._loopquant_allocation = tuple(allocation)
        configure_looped_model(model, loops)
        if manifest["has_lctc"]:
            cgb = model.lctc_cgb
            del model.lctc_cgb
            runtime = apply_lctc_calibration(model, cgb)
            runtime.enabled = manifest["lctc_enabled"]
        if (directory / "generation_config.json").exists():
            model.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
        QUANTIZATION_CONTEXT.configure(loops, allocation)
        QUANTIZATION_CONTEXT.enabled = True
    except Exception:
        (QUANTIZATION_CONTEXT.num_loops, QUANTIZATION_CONTEXT.current_loop,
         QUANTIZATION_CONTEXT.activation_bits, QUANTIZATION_CONTEXT.enabled) = previous
        raise
    return model, tokenizer
