"""FP teacher cache -> GPTQ -> loop-first incremental block reconstruction."""

import logging

import torch

from ..methods import QuantizationResult
from .cache import prepare_teacher_cache, block_statistics, original_linears
from .modules import wrap_linears, iter_linears, prepare_inference
from .quantization import activation_bits, gptq
from .reconstruction import reconstruct


class LowRankPipeline:
    def __init__(self, config, *, device="cuda:0", logger=None):
        self.config, self.device = config, device
        self.logger = logger or logging.getLogger("loopquant")

    def run(self, model, tokenizer, calibration_data):
        config = self.config
        model.eval().requires_grad_(False)
        # Validate the complete structure before publishing an expensive cache.
        for block in model.model.layers:
            for module in original_linears(block).values():
                if config.num_loops * config.method["rank_step"] > min(module.weight.shape):
                    raise ValueError("rank_step*num_loops exceeds a target projection's dimensions")
                if module.in_features % config.quantization["group_size"]:
                    raise ValueError("target input width is not divisible by group_size")
        teacher = prepare_teacher_cache(model, tokenizer, calibration_data, config, self.device, self.logger)
        clip_banks = []
        options = config.method["gptq"]
        for index, block in enumerate(model.model.layers):
            self.logger.info("[LowRank] GPTQ block=%d/%d mode=%s", index + 1, len(model.model.layers), options["hessian_mode"])
            hessians, clips = block_statistics(model, index, teacher, config, self.device)
            clip_banks.append(clips)
            with torch.no_grad():
                for name, module in original_linears(block).items():
                    module.weight.copy_(gptq(module.weight, hessians[name],
                                            bits=config.quantization["weight_bits"], group_size=config.quantization["group_size"],
                                            block_size=options["block_size"], damp_percent=options["damp_percent"]))
        wrap_linears(model, config)
        for block, clips in zip(model.model.layers, clip_banks):
            for name, module in iter_linears(block):
                module.clip_ratios.copy_(clips[name].to(module.clip_ratios))
        history = reconstruct(model, teacher, config, self.device, self.logger)
        model._quantization_algorithm = "low_rank"
        model._low_rank_config = config.to_dict()
        model._loopquant_allocation = (activation_bits(config),) * config.num_loops
        prepare_inference(model)
        linears = [module for block in model.model.layers for _, module in iter_linears(block)]
        parameters = [parameter for module in linears for bank in (module.A, module.B) for parameter in bank]
        rank = config.method["rank_step"]
        ranks = ([rank * (loop + 1) for loop in range(config.num_loops)]
                 if config.method["sharing"] == "prefix" else [rank] * config.num_loops)
        summary = {
            "algorithm": "low_rank", "fake_quantization": True,
            "activation_allocation": list(model._loopquant_allocation),
            "stage_order": ["teacher_cache", "GPTQ", "incremental_block_reconstruction"],
            "sharing": config.method["sharing"], "rank_step": rank, "rank_schedule": ranks,
            "low_rank_parameters": sum(parameter.numel() for parameter in parameters),
            "low_rank_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
            "hessian_mode": options["hessian_mode"], "teacher_cache": str(teacher.directory),
            "teacher_cache_hit": teacher.hit,
        }
        return QuantizationResult(model, summary, {"low_rank_reconstruction": {"blocks": history}})
