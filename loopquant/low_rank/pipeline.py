"""FP teacher statistics -> alternating shared Q and low-rank corrections."""

import logging

import torch

from ..methods import QuantizationResult
from ..grouping import validate_input_width
from .cache import prepare_teacher_cache, block_statistics, original_linears
from .modules import wrap_linears, iter_linears, prepare_inference
from .quantization import activation_bits
from .rotation import rotate_weight, validate_block_hadamard
from .solver import solve_linear


class LowRankPipeline:
    def __init__(self, config, *, device="cuda:0", logger=None):
        self.config, self.device = config, device
        self.logger = logger or logging.getLogger("loopquant")

    def run(self, model, tokenizer, calibration_data):
        config = self.config
        options = config.method["solver"]
        sharing = config.method["sharing"]
        rotation = config.method.get("rotation", {})
        rotation_enabled = bool(rotation.get("enabled", False))
        rotation_block_size = int(rotation.get("block_size", 32))
        rotation_seed = int(rotation.get("seed", 42))
        active_rank = config.method["rank_step"] * (config.num_loops if sharing == "prefix" else 1)
        model.eval().requires_grad_(False)
        # Validate the complete structure before publishing an expensive cache.
        for block in model.model.layers:
            for module in original_linears(block).values():
                if active_rank > min(module.weight.shape):
                    raise ValueError("active low-rank schedule exceeds a target projection's dimensions")
                validate_input_width(
                    module.in_features,
                    config.quantization["group_size"],
                    name="target input width",
                )
                if rotation_enabled:
                    validate_block_hadamard(module.in_features, rotation_block_size)
        teacher = prepare_teacher_cache(model, tokenizer, calibration_data, config, self.device, self.logger)
        fitted, history = [], []
        rank = config.method["rank_step"]
        for index, block in enumerate(model.model.layers):
            self.logger.info("[LowRank] Collecting linear statistics block=%d/%d", index + 1, len(model.model.layers))
            statistics, clips = block_statistics(model, index, teacher, config, self.device)
            block_factors = {}
            with torch.no_grad():
                for name, module in original_linears(block).items():
                    label = f"block={index + 1}/{len(model.model.layers)} module={name}"
                    original = module.weight
                    if rotation_enabled:
                        original = rotate_weight(original, rotation_block_size, rotation_seed)
                    q, a, b, report = solve_linear(
                        original, statistics.pop(name), rank=rank, options=options, sharing=sharing,
                        bits=config.quantization["weight_bits"], group_size=config.quantization["group_size"],
                        logger=self.logger, label=label)
                    module.weight.copy_(q)
                    block_factors[name] = (a, b)
                    history.append({"block": index, "module": name, **report})
            fitted.append((clips, block_factors))
        wrap_linears(model, config)
        with torch.no_grad():
            for block, (clips, factors) in zip(model.model.layers, fitted):
                for name, module in iter_linears(block):
                    module.clip_ratios.copy_(clips[name].to(module.clip_ratios))
                    a, b = factors[name]
                    for loop in range(len(a)):
                        module.A[loop].copy_(a[loop])
                        module.B[loop].copy_(b[loop])
        model._quantization_algorithm = "low_rank"
        model._low_rank_config = config.to_dict()
        model._loopquant_allocation = (activation_bits(config),) * config.num_loops
        prepare_inference(model)
        linears = [module for block in model.model.layers for _, module in iter_linears(block)]
        parameters = [parameter for module in linears for bank in (module.A, module.B) for parameter in bank]
        summary = {
            "algorithm": "low_rank", "fake_quantization": True,
            "activation_allocation": list(model._loopquant_allocation),
            "weight_update": options["weight_update"], "iterations": options["iterations"],
            "stage_order": ["teacher_cache", "linear_statistics",
                            (f"alternating_{options['weight_update']}_weighted_svd" if rank
                             else f"quantization_{options['weight_update']}")],
            "sharing": sharing, "rank_step": rank,
            "rotation": {"enabled": rotation_enabled,
                          "block_size": rotation_block_size,
                          "seed": rotation_seed},
            "low_rank_enabled": rank > 0,
            "rank_schedule": [(t + 1) * rank if sharing == "prefix" else rank for t in range(config.num_loops)],
            "low_rank_parameters": sum(parameter.numel() for parameter in parameters),
            "low_rank_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
            "damp_percent": options["damp_percent"], "loop_weights": options["loop_weights"],
            "quantized_damping_target": options["quantized_damping_target"],
            "low_rank_damping_target": options["low_rank_damping_target"],
            "teacher_cache": str(teacher.directory),
            "teacher_cache_hit": teacher.hit,
        }
        return QuantizationResult(model, summary, {"low_rank_solver": {"linears": history}})
