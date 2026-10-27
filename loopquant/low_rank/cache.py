"""Immutable teacher trajectories and separately keyed calibration statistics."""

import hashlib
import inspect
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file

from ..evaluation_results import save_json
from ..data import tokenizer_fingerprint
from .adapter import run_block
from .modules import TARGETS
from .quantization import activation_bits, fit_clip

CACHE_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(digest, tensor):
    digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(raw))


def original_linears(block):
    return {name: module for name, module in block.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS}


class TeacherCache:
    def __init__(self, directory, manifest, hit):
        self.directory, self.manifest, self.hit = Path(directory), manifest, hit

    def read(self, block, loop, sample):
        name = f"teacher/block_{block:03d}/loop_{loop:02d}/shard_{sample:05d}.safetensors"
        path = self.directory / name
        if name not in self.manifest["files"] or checksum(path) != self.manifest["files"][name]:
            raise ValueError(f"teacher cache is missing or corrupt: {path}")
        return load_file(path)

    def weights(self, block):
        name = f"teacher/block_{block:03d}/weights.safetensors"
        path = self.directory / name
        if checksum(path) != self.manifest["files"][name]:
            raise ValueError(f"teacher weight cache is corrupt: {path}")
        return load_file(path)


@torch.no_grad()
def prepare_teacher_cache(model, tokenizer, sequences, config, device, logger):
    logger.info("[LowRank] Fingerprinting FP teacher weights and calibration tokens")
    batch_size = min(config.calibration["batch_size"], len(sequences))
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        _hash_tensor(digest, tensor)
    tokens_digest = hashlib.sha256()
    _hash_tensor(tokens_digest, sequences)
    source_file = inspect.getsourcefile(type(model))
    identity = {
        "version": CACHE_VERSION, "weights": digest.hexdigest(), "tokens": tokens_digest.hexdigest(),
        "config": model.config.to_dict(), "num_loops": config.num_loops,
        "dtype": str(next(model.parameters()).dtype), "source_code": checksum(source_file) if source_file else None,
        "torch": torch.__version__,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    # Batched GEMMs can change rounding. Keep their caches distinct, while
    # retaining the identity of existing single-sample teacher caches.
    if batch_size > 1:
        identity["collection_batch_size"] = batch_size
    key = fingerprint(identity)
    model_key = re.sub(r"[^\w.-]", "_", config.model_key or config.model_variant or "ouro")
    directory = Path(config.calibration["cache_dir"]).expanduser().resolve() / model_key / key
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != json.loads(json.dumps(identity, default=str)):
            raise ValueError("teacher cache identity mismatch")
        logger.info("[LowRank] Reusing teacher cache: %s", directory)
        return TeacherCache(directory, manifest, True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[LowRank] Collecting FP teacher trajectories: %s", directory)
    with tempfile.TemporaryDirectory(prefix=".teacher-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        manifest = {"identity": identity, "files": {}, "num_samples": len(sequences),
                    "num_blocks": len(model.model.layers), "num_loops": config.num_loops,
                    "collection_batch_size": batch_size}

        def write(name, tensors):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, path)
            manifest["files"][name] = checksum(path)

        def write_batch(block, loop, start, hidden, output):
            for offset in range(hidden.shape[0]):
                write(f"teacher/block_{block:03d}/loop_{loop:02d}/shard_{start + offset:05d}.safetensors",
                      {"input": hidden[offset:offset + 1], "output": output[offset:offset + 1]})

        write("tokens.safetensors", {"input_ids": sequences})
        pending, counters, handles = {}, {}, []
        writes = deque()
        sample = 0
        logger.info("[LowRank] Caching original projection weights: %d blocks", len(model.model.layers))
        for index, block in enumerate(model.model.layers):
            linears = original_linears(block)
            if not linears:
                raise ValueError(f"block {index} has no supported projection linears")
            write(f"teacher/block_{index:03d}/weights.safetensors",
                  {name: module.weight for name, module in linears.items()})

            def pre_hook(_block, args, kwargs, block_id=index):
                # At most two batches of block states wait for disk, plus the
                # current block. A slow filesystem cannot consume unbounded RAM.
                if len(writes) >= 2:
                    writes.popleft().result()
                loop = int(kwargs.get("current_ut", counters.get(block_id, 0)))
                counters[block_id] = loop + 1
                hidden = args[0] if args else kwargs["hidden_states"]
                pending[block_id] = (loop, hidden.detach().to("cpu", copy=True))

            def post_hook(_block, _args, _kwargs, output, block_id=index):
                loop, hidden = pending.pop(block_id)
                value = output[0] if isinstance(output, tuple) else output
                # Take immutable snapshots before returning to model code. Disk
                # serialization/checksums run off the forward path; D2H stays
                # synchronous so the writer never reads an unfinished transfer.
                value = value.detach().to("cpu", copy=True).contiguous()
                writes.append(writer.submit(write_batch, block_id, loop, sample, hidden, value))

            handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(block.register_forward_hook(post_hook, with_kwargs=True))
        try:
            # Decoder hooks need every token's state, but not vocabulary logits.
            # Ouro exposes this option on the outer CausalLM forward.
            forward_options = {"use_cache": False}
            if "logits_to_keep" in inspect.signature(model.forward).parameters:
                forward_options["logits_to_keep"] = 1
            started = time.perf_counter()
            logger.info("[LowRank] FP collection: samples=%d batch_size=%d", len(sequences), batch_size)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-cache") as writer:
                for sample in range(0, len(sequences), batch_size):
                    counters.clear()
                    ids = sequences[sample:sample + batch_size].to(device)
                    model(input_ids=ids, **forward_options)
                    if any(counters.get(index) != config.num_loops for index in range(len(model.model.layers))):
                        raise ValueError("teacher did not execute every configured loop/block")
                    completed = sample + len(ids)
                    logger.info("[LowRank] FP samples=%d/%d elapsed=%.1fs (disk writes may be pending)",
                                completed, len(sequences), time.perf_counter() - started)
                while writes:
                    writes.popleft().result()
            logger.info("[LowRank] FP cache written: samples=%d elapsed=%.1fs", len(sequences),
                        time.perf_counter() - started)
        finally:
            for handle in handles:
                handle.remove()
        save_json(staging / "manifest.json", manifest)
        try:
            staging.rename(directory)
        except FileExistsError:
            # A concurrent run may have completed the same immutable cache.
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing["identity"] != json.loads(json.dumps(identity, default=str)):
                raise ValueError("concurrent teacher cache identity mismatch")
            return TeacherCache(directory, existing, True)
    return TeacherCache(directory, manifest, False)


def _load_stat(path, description):
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file() or not path.is_file():
        return None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata["description"] != description or checksum(path) != metadata["checksum"]:
        raise ValueError(f"invalid calibration statistics cache: {path}")
    return load_file(path)


def _save_stat(path, tensors, description):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".statistics-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        save_file({key: value.cpu().contiguous() for key, value in tensors.items()}, staged)
        staged.replace(path)
    save_json(path.with_suffix(".json"), {"description": description, "checksum": checksum(path)})


@torch.no_grad()
def block_statistics(model, block_index, teacher, config, device):
    options, act = config.method["gptq"], config.method["activation"]
    bits = activation_bits(config)
    cap, mode = options["statistics_tokens_per_loop"], options["hessian_mode"]
    hdesc = {"version": CACHE_VERSION, "mode": mode, "tokens_per_loop": cap}
    cdesc = {"version": CACHE_VERSION, "bits": bits, "group_size": config.quantization["group_size"],
             "grid": act, "tokens_per_loop": cap}
    hpath = teacher.directory / "hessian" / mode / fingerprint(hdesc) / f"block_{block_index:03d}.safetensors"
    cpath = teacher.directory / "activation_clip" / fingerprint(cdesc) / f"block_{block_index:03d}.safetensors"
    hessians, clips = _load_stat(hpath, hdesc), _load_stat(cpath, cdesc)
    if hessians is not None and clips is not None:
        return hessians, clips
    block = model.model.layers[block_index]
    modules = original_linears(block)
    matrices = {name: torch.zeros(module.in_features, module.in_features, device=device)
                for name, module in modules.items()} if hessians is None else None
    count = int(round((act["clip_max"] - act["clip_min"]) / act["clip_step"])) + 1
    ratios = [min(act["clip_max"], act["clip_min"] + i * act["clip_step"]) for i in range(count)]
    if ratios[-1] < act["clip_max"] - 1e-8:
        ratios.append(act["clip_max"])
    errors = {name: torch.zeros(config.num_loops, len(ratios), device=device) for name in modules} if clips is None else None
    values, handles = {}, []
    for name, module in modules.items():
        def hook(_module, args, key=name):
            values[key] = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        handles.append(module.register_forward_pre_hook(hook))
    total = 0
    try:
        for sample in range(teacher.manifest["num_samples"]):
            if cap is not None and total >= cap:
                break
            loops = []
            for loop in range(config.num_loops):
                hidden = teacher.read(block_index, loop, sample)["input"].to(device)
                run_block(model, block, hidden, loop)
                selected = {name: value[:max(0, cap - total)] if cap is not None else value
                            for name, value in values.items()}
                loops.append(selected)
                if errors is not None:
                    for name, value in selected.items():
                        errors[name][loop] += fit_clip(value, bits, config.quantization["group_size"], ratios)
            tokens = next(iter(loops[0].values())).shape[0]
            total += tokens
            if matrices is not None:
                for name in modules:
                    if mode == "mean_hessian":
                        for current in loops:
                            matrices[name].addmm_(current[name].T, current[name], alpha=1 / config.num_loops)
                    else:
                        average = torch.stack([current[name] for current in loops]).mean(0)
                        matrices[name].addmm_(average.T, average)
    finally:
        for handle in handles:
            handle.remove()
    if matrices is not None:
        hessians = {name: value.mul_(2 / max(total, 1)).cpu() for name, value in matrices.items()}
        _save_stat(hpath, hessians, hdesc)
    if errors is not None:
        grid = torch.tensor(ratios, device=device)
        clips = {name: grid[value.argmin(-1)].cpu() for name, value in errors.items()}
        _save_stat(cpath, clips, cdesc)
    return hessians, clips
