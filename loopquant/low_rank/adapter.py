"""HF Ouro block execution with the original masks, rotary module and norms."""

import importlib
import inspect

import torch


def run_block(model, block, hidden, loop):
    core = model.model
    length = hidden.shape[1]
    position = torch.arange(length, device=hidden.device)
    kwargs = {"current_ut": int(loop), "use_cache": False, "past_key_value": None,
              "position_ids": position.unsqueeze(0), "cache_position": position}
    if hasattr(core, "rotary_emb"):
        kwargs["position_embeddings"] = core.rotary_emb(hidden, position.unsqueeze(0))
    implementation = importlib.import_module(type(core).__module__)
    if hasattr(implementation, "create_causal_mask"):
        name = ("create_sliding_window_causal_mask" if getattr(block, "attention_type", "") == "sliding_attention"
                else "create_causal_mask")
        kwargs["attention_mask"] = getattr(implementation, name)(
            config=model.config, input_embeds=hidden, attention_mask=None,
            cache_position=position, past_key_values=None, position_ids=position.unsqueeze(0))
    parameters = inspect.signature(block.forward).parameters
    if not any(value.kind == value.VAR_KEYWORD for value in parameters.values()):
        kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    result = block(hidden, **kwargs)
    return result[0] if isinstance(result, tuple) else result
