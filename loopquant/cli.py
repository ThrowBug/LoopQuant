"""Quantize (optionally save/evaluate), or load original/quantized weights and evaluate."""

import argparse
from pathlib import Path
import time

import torch

from .config import LoopQuantConfig
from .data import load_calibration_data
from .evaluation import preflight_evaluation, run_evaluation
from .evaluation_config import load_evaluation_config
from .models import load_looped_transformer, load_model_for_evaluation
from .utils import resolve_device, runtime_metadata, set_random_seed, setup_logging, write_json, write_yaml


def _add_quantize_arguments(parser):
    parser.add_argument("--config", required=True, help="Quantization/calibration YAML")
    parser.add_argument("--save", action="store_true", help="Save output-dir/loopquant before evaluation")
    parser.add_argument("--model-family", choices=("ouro", "nanbeige", "huginn"))
    parser.add_argument("--weight-bits", type=int)
    parser.add_argument("--activation-setting", choices=("w4a4", "w4a6"))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--num-calibration-samples", type=int)
    parser.add_argument("--calibration-seqlen", type=int)
    parser.add_argument("--calibration-seed", type=int)


def _add_evaluate_arguments(parser):
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                        help="Original HF models only; default bfloat16")
    parser.add_argument("--num-loops", type=int, required=True,
                        help="Recurrent loop count; must match saved checkpoints")


def build_parser():
    parser = argparse.ArgumentParser(prog="loopquant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="HF model ID, HF directory, or LoopQuant checkpoint")
    common.add_argument("--output-dir", required=True)
    common.add_argument("--device", default="auto")
    common.add_argument("--eval-config", default=str(
        Path(__file__).resolve().parents[1] / "configs/evaluation_generative.yaml"),
        help="Protocol parameters; switches below select which suites run")
    common.add_argument("--wikitext2", action="store_true")
    common.add_argument("--c4", action="store_true")
    common.add_argument("--zero-shot", action="store_true")
    common.add_argument("--generative", action="store_true")
    quantize = subparsers.add_parser("quantize", parents=[common], help="Quantize, optionally save/evaluate")
    _add_quantize_arguments(quantize)
    evaluate = subparsers.add_parser("evaluate", parents=[common], help="Load original/quantized weights and evaluate")
    _add_evaluate_arguments(evaluate)
    return parser


def _evaluation_requested(args):
    return bool(_evaluation_suites(args))


def _evaluation_suites(args):
    return [name for name in ("wikitext2", "c4", "zero_shot", "generative")
            if getattr(args, name)]


def _run_quantize(args):
    from .checkpoint import save_quantized_checkpoint
    from .pipeline import LoopQuantPipeline

    started = time.time()
    config = LoopQuantConfig.from_yaml(args.config)
    config.apply_overrides(
        model_family=args.model_family, output_dir=args.output_dir,
        quantization__weight_bits=args.weight_bits,
        quantization__activation_setting=args.activation_setting,
        quantization__group_size=args.group_size,
        calibration__num_samples=args.num_calibration_samples,
        calibration__sequence_length=args.calibration_seqlen,
        calibration__seed=args.calibration_seed,
    )
    output = Path(args.output_dir).expanduser().resolve()
    checkpoint = output / "loopquant"
    if args.save and checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint}; choose a new output directory")
    logger = setup_logging(output)
    device = resolve_device(args.device)
    set_random_seed(int(config.calibration["seed"]))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    write_yaml(output / "config.yaml", config.to_dict())
    model, tokenizer = load_looped_transformer(
        args.model, model_family=config.model_family, device=device, num_loops=config.num_loops,
    )
    data = load_calibration_data(
        tokenizer, dataset=config.calibration["dataset"], split=config.calibration["split"],
        num_samples=config.calibration["num_samples"], sequence_length=config.calibration["sequence_length"],
        seed=config.calibration["seed"],
    )
    if tuple(data.shape) != (config.calibration["num_samples"], config.calibration["sequence_length"]):
        raise RuntimeError(f"unexpected calibration shape {tuple(data.shape)}")
    result = LoopQuantPipeline(config, device=device, logger=logger).run(model, tokenizer, data)
    for name, value in (("cldc_summary", result.cldc), ("ilpa_allocation", result.ilpa),
                        ("lctc_summary", result.lctc)):
        write_json(output / f"{name}.json", value)
    summary = {
        "model": str(args.model), "model_family": config.model_family,
        "model_variant": config.model_variant, "num_loops": config.num_loops,
        "weight_bits": config.quantization["weight_bits"],
        "weight_group_size": config.quantization["group_size"],
        "activation_setting": config.quantization["activation_setting"],
        "activation_allocation": result.allocation, "fake_quantization": True,
        "calibration": config.calibration, "stage_order": list(result.stage_order),
        "runtime": runtime_metadata(started), "checkpoint": None,
    }
    write_json(output / "quantization_summary.json", summary)
    if args.save:
        save_quantized_checkpoint(result.model, tokenizer, checkpoint,
                                  quantization_config=config.to_dict())
        summary["checkpoint"] = str(checkpoint)
        write_json(output / "quantization_summary.json", summary)
        logger.info("[Checkpoint] Saved %s", checkpoint)
    if _evaluation_requested(args):
        # Dataset/MBPP failures must not prevent a completed snapshot being saved.
        run_evaluation(result.model, tokenizer, load_evaluation_config(args.eval_config), output,
                       suites=_evaluation_suites(args), device=device,
                       run_metadata={"mode": "quantized", "model_source": str(args.model),
                                     "quantization": summary})
    logger.info("[LoopQuant] Completed: %s", output)
    return 0


def _run_evaluate(args):
    settings = load_evaluation_config(args.eval_config)
    suites = _evaluation_suites(args)
    output = Path(args.output_dir).expanduser().resolve()
    setup_logging(output)
    device = resolve_device(args.device)
    set_random_seed(settings["seed"])
    write_yaml(output / "evaluation_config.yaml", {"evaluation": settings})
    manager = preflight_evaluation(settings, suites=suites)
    dtype = None if args.dtype is None else getattr(torch, args.dtype)
    model, tokenizer, metadata = load_model_for_evaluation(
        args.model, device=device, dtype=dtype, num_loops=args.num_loops,
    )
    run_evaluation(model, tokenizer, settings, output, suites=suites, device=device,
                   task_manager=manager, run_metadata=metadata)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate" and not _evaluation_requested(args):
        parser.error("evaluate requires --wikitext2, --c4, --zero-shot or --generative")
    if getattr(args, "num_loops", None) is not None and args.num_loops <= 0:
        parser.error("--num-loops must be positive")
    if args.command == "quantize":
        return _run_quantize(args)
    return _run_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
