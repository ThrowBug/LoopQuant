"""Small algorithm dispatcher; command-line orchestration stays method neutral."""

from dataclasses import dataclass
from typing import Any


@dataclass
class QuantizationResult:
    model: Any
    summary: dict
    stage_summaries: dict


def run_quantization(config, model, tokenizer, data, *, device, logger):
    if config.algorithm == "low_rank":
        from .low_rank.pipeline import LowRankPipeline
        return LowRankPipeline(config, device=device, logger=logger).run(model, tokenizer, data)
    from .pipeline import LoopQuantPipeline
    result = LoopQuantPipeline(config, device=device, logger=logger).run(model, tokenizer, data)
    return QuantizationResult(result.model,
        {"algorithm": "loopquant", "activation_allocation": result.allocation,
         "fake_quantization": True, "stage_order": list(result.stage_order)},
        {"cldc_summary": result.cldc, "ilpa_allocation": result.ilpa, "lctc_summary": result.lctc})
