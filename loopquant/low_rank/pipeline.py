"""FP teacher statistics -> alternating shared GPTQ and independent low rank."""

import logging

import torch

from ..methods import QuantizationResult
from .cache import prepare_teacher_cache, block_statistics, original_linears
from .modules import wrap_linears, iter_linears, prepare_inference
from .quantization import activation_bits
from .solver import solve_linear


class LowRankPipeline:
    def __init__(self, config, *, device="cuda:0", logger=None):
        self.config, self.device = config, device
        self.logger = logger or logging.getLogger("loopquant")

    def run(self, model, tokenizer, calibration_data):
        config = self.config
        options = config.method["solver"]
        if config.method["sharing"] != "independent":
            raise NotImplementedError("sharing=prefix is reserved; the alternating solver implements independent only")
        if options["weight_update"] != "regression_gptq":
            raise NotImplementedError("solver.weight_update=gptaq is reserved and not implemented")
        model.eval().requires_grad_(False)
        # Validate the complete structure before publishing an expensive cache.
        for block in model.model.layers:
            for module in original_linears(block).values():
                if config.method["rank_step"] > min(module.weight.shape):
                    raise ValueError("rank_step exceeds a target projection's dimensions")
                if module.in_features % config.quantization["group_size"]:
                    raise ValueError("target input width is not divisible by group_size")
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
                    q, a, b, report = solve_linear(
                        module.weight, statistics.pop(name), rank=rank, options=options,
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
                    for loop in range(config.num_loops):
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
            "stage_order": ["teacher_cache", "linear_statistics", "alternating_regression_gptq_weighted_svd"],
            "sharing": config.method["sharing"], "rank_step": rank, "rank_schedule": [rank] * config.num_loops,
            "low_rank_parameters": sum(parameter.numel() for parameter in parameters),
            "low_rank_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
            "damp_percent": options["damp_percent"], "loop_weights": options["loop_weights"],
            "quantized_damping_target": options["quantized_damping_target"],
            "low_rank_damping_target": options["low_rank_damping_target"],
            "teacher_cache": str(teacher.directory),
            "teacher_cache_hit": teacher.hit,
        }
        return QuantizationResult(model, summary, {"low_rank_solver": {"linears": history}})
