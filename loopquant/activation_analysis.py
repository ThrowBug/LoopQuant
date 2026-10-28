"""Analyze aligned FP decoder outputs without running a quantization pipeline."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .evaluation_results import save_json
from .utils import resolve_device, set_random_seed, setup_logging


def token_positions(length, count, sample, seed):
    """The same deterministic positions are used in every layer and loop."""
    generator = torch.Generator().manual_seed(seed + sample)
    return torch.randperm(length, generator=generator)[:min(count, length)].sort().values


@torch.no_grad()
def increment_r90(increment):
    """Exact uncentered 90%-energy rank, via the smaller FP32 Gram matrix."""
    matrix = increment.reshape(-1, increment.shape[-1]).float()
    scale = matrix.abs().max()
    if scale == 0:
        return 0
    matrix = matrix / scale
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    energy = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
    threshold = energy.sum() * 0.9
    return min(int(torch.searchsorted(energy.cumsum(0), threshold)) + 1, len(energy))


@torch.no_grad()
def layer_metrics(states, device="cpu"):
    """states: [loop, sample, aligned token, feature]. No centering/rescaling."""
    if states.ndim != 4 or states.shape[0] < 2 or any(size == 0 for size in states.shape):
        raise ValueError("expected nonempty [loop>=2, sample, token, feature] activations")
    states = states.to(device=device, dtype=torch.float32)
    if not torch.isfinite(states).all():
        raise ValueError("FP activations contain non-finite values")
    loops, samples, tokens, width = states.shape
    norms = torch.linalg.vector_norm(states, dim=-1)
    normalized = states / norms.clamp_min(1e-12).unsqueeze(-1)
    cosine, counts = [], []
    for first in range(loops):
        row, valid_counts = [], []
        for second in range(loops):
            valid = (norms[first] > 1e-12) & (norms[second] > 1e-12)
            count = int(valid.sum())
            similarity = (normalized[first] * normalized[second]).sum(-1).clamp(-1, 1)
            row.append(float(similarity[valid].mean()) if count else None)
            valid_counts.append(count)
        cosine.append(row)
        counts.append(valid_counts)
    adjacent = []
    for loop in range(1, loops):
        delta = states[loop] - states[loop - 1]
        numerator = torch.linalg.vector_norm(delta.flatten(1), dim=1)
        denominator = torch.linalg.vector_norm(states[loop - 1].flatten(1), dim=1)
        per_sample = [float(change / reference) if reference > 0 else (0.0 if change == 0 else None)
                      for change, reference in zip(numerator, denominator)]
        defined = [value for value in per_sample if value is not None]
        adjacent.append({
            "from_loop": loop, "to_loop": loop + 1,
            "relative_change": sum(defined) / len(defined) if defined else None,
            "relative_change_by_sample": per_sample,
            "relative_change_valid_samples": len(defined),
            "r90": increment_r90(delta), "max_rank": min(samples * tokens, width),
        })
    return {"shape": list(states.shape), "cosine_similarity": cosine,
            "cosine_valid_tokens": counts, "adjacent": adjacent}


def _checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FPTeacherOutputs:
    """Read only output tensors from the existing immutable FP teacher cache."""

    def __init__(self, directory, num_samples, tokens_per_sample, seed, num_loops=None):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        identity = manifest.get("identity", {})
        if identity.get("version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("expected a version-1 FP teacher cache, not a quantized checkpoint")
        for key in ("num_blocks", "num_loops", "num_samples"):
            if type(manifest.get(key)) is not int or manifest[key] <= 0:
                raise ValueError(f"invalid teacher cache {key}")
        self.num_loops, self.num_blocks = manifest["num_loops"], manifest["num_blocks"]
        if self.num_loops < 2 or identity.get("num_loops") != self.num_loops:
            raise ValueError("teacher cache must consistently describe at least two loops")
        if num_loops is not None and num_loops != self.num_loops:
            raise ValueError("--num-loops conflicts with the FP teacher cache")
        generator = torch.Generator().manual_seed(seed)
        self.sample_ids = torch.randperm(manifest["num_samples"], generator=generator)[
            :min(num_samples, manifest["num_samples"])].sort().values.tolist()
        with safe_open(self._path(0, 0, self.sample_ids[0]), framework="pt", device="cpu") as tensors:
            shape = tensors.get_slice("output").get_shape()
        if len(shape) != 3 or shape[0] != 1 or min(shape) <= 0:
            raise ValueError("teacher output must have shape [1, sequence, feature]")
        self.sequence_length = shape[1]
        self.positions = [token_positions(shape[1], tokens_per_sample, sample, seed)
                          for sample in self.sample_ids]

    def _path(self, block, loop, sample):
        name = f"teacher/block_{block:03d}/loop_{loop:02d}/shard_{sample:05d}.safetensors"
        path = self.directory / name
        if name not in self.manifest["files"] or _checksum(path) != self.manifest["files"][name]:
            raise ValueError(f"teacher cache is missing or corrupt: {path}")
        return path

    def read_layer(self, block):
        loops = []
        for loop in range(self.num_loops):
            samples = []
            for sample, positions in zip(self.sample_ids, self.positions):
                with safe_open(self._path(block, loop, sample), framework="pt", device="cpu") as tensors:
                    value = tensors.get_tensor("output")
                if not value.is_floating_point() or value.ndim != 3 or value.shape[:2] != (1, self.sequence_length):
                    raise ValueError("teacher outputs are not aligned across samples/loops")
                samples.append(value[0].index_select(0, positions))
            loops.append(torch.stack(samples))
        return torch.stack(loops)


def _batch_path(directory, block, loop, start):
    return directory / f"block_{block:03d}" / f"loop_{loop:02d}" / f"batch_{start:05d}.safetensors"


@torch.inference_mode()
def collect_fp_outputs(model, sequences, directory, *, num_loops, positions, batch_size, device, logger):
    """Spool sampled FP outputs only; memory does not grow with trajectory length."""
    blocks = model.model.layers
    counts, handles = {}, []
    start = 0

    def hook_for(block):
        def hook(_module, _args, kwargs, output):
            loop = int(kwargs.get("current_ut", counts.get(block, 0)))
            if loop != counts.get(block, 0) or not 0 <= loop < num_loops:
                raise ValueError("FP model did not execute loops in the expected order")
            counts[block] = loop + 1
            value = output[0] if isinstance(output, tuple) else output
            batch_positions = torch.stack(positions[start:start + len(value)]).to(value.device)
            if value.ndim != 3 or value.shape[1] != sequences.shape[1]:
                raise ValueError("decoder output is not aligned with input tokens")
            selected = value.gather(1, batch_positions.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
            path = _batch_path(directory, block, loop, start)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file({"output": selected.detach().cpu().contiguous()}, path)
        return hook

    try:
        for block, module in enumerate(blocks):
            handles.append(module.register_forward_hook(hook_for(block), with_kwargs=True))
        options = {"use_cache": False}
        if "logits_to_keep" in inspect.signature(model.forward).parameters:
            options["logits_to_keep"] = 1
        for start in range(0, len(sequences), batch_size):
            counts.clear()
            model(input_ids=sequences[start:start + batch_size].to(device), **options)
            if any(counts.get(block) != num_loops for block in range(len(blocks))):
                raise ValueError("FP model did not execute every configured loop/block")
            logger.info("[ActivationAnalysis] FP samples=%d/%d", min(start + batch_size, len(sequences)), len(sequences))
    finally:
        for handle in handles:
            handle.remove()
    return len(blocks)


def _read_collected_layer(directory, block, loops, samples, batch_size):
    return torch.stack([torch.cat([
        load_file(_batch_path(directory, block, loop, start))["output"]
        for start in range(0, samples, batch_size)]) for loop in range(loops)])


def write_tables(result, output):
    with (output / "cosine_similarity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "loop_s", "loop_t", "cosine_similarity", "valid_tokens"])
        for layer in result["layers"]:
            for first, row in enumerate(layer["cosine_similarity"]):
                for second, value in enumerate(row):
                    writer.writerow([layer["layer"], first + 1, second + 1, value,
                                     layer["cosine_valid_tokens"][first][second]])
    with (output / "adjacent_changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "from_loop", "to_loop", "relative_change", "r90", "max_rank"])
        for layer in result["layers"]:
            for entry in layer["adjacent"]:
                writer.writerow([layer["layer"], *[entry[key] for key in (
                    "from_loop", "to_loop", "relative_change", "r90", "max_rank")]])


def _plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib. Install with: python -m pip install "
                           "-i https://mirrors.aliyun.com/pypi/simple/ matplotlib; "
                           "or pass --no-plots to save JSON/CSV only") from error


def plot_results(result, output):
    import numpy as np
    plt = _plotting()
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    layers, loops = result["layers"], result["num_loops"]
    cosines = [value for layer in layers for row in layer["cosine_similarity"]
               for value in row if value is not None]
    cosine_min = min(0.99, min(cosines)) if cosines else 0
    columns = min(4, len(layers))
    rows = (len(layers) + columns - 1) // columns
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
        figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3 * rows),
                                    squeeze=False, constrained_layout=True)
        for axis, layer in zip(axes.flat, layers):
            image = axis.imshow(np.array(layer["cosine_similarity"], dtype=float),
                                vmin=cosine_min, vmax=1, cmap="viridis", origin="upper")
            axis.set(title=f"Layer {layer['layer']}", xlabel="Loop", ylabel="Loop",
                     xticks=range(loops), xticklabels=range(1, loops + 1),
                     yticks=range(loops), yticklabels=range(1, loops + 1))
        for axis in list(axes.flat)[len(layers):]:
            axis.set_visible(False)
        figure.colorbar(image, ax=list(axes.flat)[:len(layers)], label="Mean token cosine")
        figure.suptitle("FP decoder outputs: pairwise loop cosine similarity")
        _save_figure(figure, directory / "cosine_similarity")
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(13, max(4, len(layers) * 0.25)),
                                    constrained_layout=True)
        values = [
            [[layer["cosine_similarity"][loop - 1][loop] for loop in range(1, loops)] for layer in layers],
            [[entry["relative_change"] for entry in layer["adjacent"]] for layer in layers],
            [[entry["r90"] for entry in layer["adjacent"]] for layer in layers],
        ]
        for axis, data, title in zip(axes, values, ("Adjacent-loop cosine", "Relative change (sample mean)", "Increment r90")):
            limits = {"vmin": cosine_min, "vmax": 1} if title == "Adjacent-loop cosine" else {"vmin": 0}
            image = axis.imshow(np.array(data, dtype=float), aspect="auto", cmap="viridis", **limits)
            axis.set(title=title, xlabel="Loop transition", ylabel="Layer",
                     xticks=range(loops - 1), xticklabels=[f"{loop}→{loop + 1}" for loop in range(1, loops)],
                     yticks=range(len(layers)), yticklabels=[layer["layer"] for layer in layers])
            figure.colorbar(image, ax=axis)
        _save_figure(figure, directory / "adjacent_changes")
        plt.close(figure)


def _save_figure(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".svg"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Original HF model ID/directory; quantized checkpoints are rejected")
    source.add_argument("--teacher-cache", help="Existing FP cache directory containing manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-loops", type=int, help="Required for --model; otherwise read from the cache")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048, help="FP model mode only")
    parser.add_argument("--batch-size", type=int, default=4, help="FP model mode only")
    parser.add_argument("--tokens-per-sample", type=int, default=128, help="Aligned random positions; capped by sequence length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Original model dtype; statistics always use FP32")
    parser.add_argument("--no-plots", action="store_true", help="Save JSON/CSV without requiring matplotlib")
    return parser


def _analyze_layers(reader, num_blocks, result, output, device, logger):
    for block in range(num_blocks):
        started = time.perf_counter()
        metrics = layer_metrics(reader(block), device)
        result["layers"].append({"layer": block + 1, **metrics})
        save_json(output / "analysis.json", result)
        logger.info("[ActivationAnalysis] layer=%d/%d elapsed=%.1fs", block + 1, num_blocks, time.perf_counter() - started)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model and args.num_loops is None:
        parser.error("--model requires --num-loops")
    if args.num_loops is not None and args.num_loops < 2:
        parser.error("at least two loops are required")
    for name in ("num_samples", "sequence_length", "batch_size", "tokens_per_sample"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if args.teacher_cache:
        cache_dir = Path(args.teacher_cache).expanduser().resolve()
        if output == cache_dir or cache_dir in output.parents:
            parser.error("analysis output must be outside the immutable FP teacher cache")
    elif (Path(args.model).expanduser() / "checkpoint_manifest.json").exists():
        parser.error("--model must be an original FP model, not a quantized checkpoint")
    if not args.no_plots:
        _plotting()  # Check the optional dependency before an expensive FP run.
    logger = setup_logging(output)
    device = resolve_device(args.device)
    set_random_seed(args.seed)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    result = {"status": "running", "source": {}, "num_loops": args.num_loops, "layers": [],
              "location": "decoder_block_output_before_outer_loop_normalization",
              "settings": vars(args), "statistics_dtype": "float32", "device": device,
              "statistics_matmul_precision": "highest",
              "torch_version": str(torch.__version__),
              "definitions": {"cosine": "mean aligned-token cosine; zero-norm pairs excluded",
                              "relative_change": "mean of per-sample Frobenius ratios; zero-to-zero=0, zero-to-nonzero=undefined",
                              "r90": "minimum rank explaining >=90% energy of pooled uncentered adjacent-loop increments; zero increment=0",
                              "indices": "layers and loops are 1-based; sample IDs and token positions are 0-based"}}
    try:
        if args.teacher_cache:
            cache = FPTeacherOutputs(args.teacher_cache, args.num_samples, args.tokens_per_sample, args.seed, args.num_loops)
            result["num_loops"] = cache.num_loops
            result["source"] = {"mode": "teacher_cache", "path": str(cache.directory),
                                "identity": cache.manifest["identity"], "sequence_length": cache.sequence_length}
            sample_ids, positions = cache.sample_ids, cache.positions
            result["sampling"] = {"sample_ids": sample_ids, "token_positions": [p.tolist() for p in positions]}
            logger.info("[ActivationAnalysis] Read-only FP cache: %s; samples=%d/%d", cache.directory, len(sample_ids), cache.manifest["num_samples"])
            _analyze_layers(cache.read_layer, cache.num_blocks, result, output, device, logger)
        else:
            from .models import load_looped_transformer
            from .data import load_calibration_data
            # Non-PTQ mode leaves quantization wrappers and global context untouched.
            model, tokenizer = load_looped_transformer(
                args.model, num_loops=args.num_loops, device=device, dtype=getattr(torch, args.dtype),
                algorithm="activation_analysis", configure_context=False)
            sequences = load_calibration_data(tokenizer, num_samples=args.num_samples,
                                              sequence_length=args.sequence_length, seed=args.seed,
                                              cache_dir=Path("cache/calibration/tokens"))
            positions = [token_positions(sequences.shape[1], args.tokens_per_sample, sample, args.seed)
                         for sample in range(len(sequences))]
            result["sampling"] = {"sample_ids": list(range(len(sequences))), "token_positions": [p.tolist() for p in positions]}
            result["source"] = {"mode": "original_model", "model": args.model, "dtype": str(next(model.parameters()).dtype),
                                "dataset": "wikitext2", "split": "train", "sequence_length": sequences.shape[1],
                                "token_ids_sha256": hashlib.sha256(memoryview(sequences.contiguous().numpy())).hexdigest()}
            with tempfile.TemporaryDirectory(prefix=".fp-outputs-", dir=output) as temporary:
                directory = Path(temporary)
                blocks = collect_fp_outputs(model, sequences, directory, num_loops=args.num_loops,
                                            positions=positions, batch_size=args.batch_size, device=device, logger=logger)
                del model, tokenizer
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                _analyze_layers(lambda block: _read_collected_layer(directory, block, args.num_loops,
                                len(sequences), args.batch_size), blocks, result, output, device, logger)
        write_tables(result, output)
        if not args.no_plots:
            plot_results(result, output)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        result["runtime_seconds"] = time.perf_counter() - started
        save_json(output / "analysis.json", result)
    logger.info("[ActivationAnalysis] DONE output=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
