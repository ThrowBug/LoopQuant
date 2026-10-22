"""HF harness adapter with explicit cache policy and no silent prompt truncation.

This module only knows Hugging Face interfaces, not LoopQuant algorithms.
"""

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria


class EvaluationHFLM(HFLM):
    def __init__(self, *args, generation_use_cache=True, allow_truncation=False, **kwargs):
        self.generation_use_cache = generation_use_cache
        self.allow_prompt_truncation = allow_truncation
        self.generation_receipts = []
        super().__init__(*args, **kwargs)

    def tok_batch_encode(self, strings, padding_side="left", left_truncate_len=None,
                         truncation=False):
        # Encode without truncating first, including the backend's BOS policy.
        encoded = super().tok_batch_encode(strings, padding_side=padding_side,
                                           left_truncate_len=None, truncation=False)
        if left_truncate_len is not None and encoded[0].shape[1] > left_truncate_len:
            if not self.allow_prompt_truncation:
                raise ValueError(
                    f"Prompt requires {encoded[0].shape[1]} tokens but only "
                    f"{left_truncate_len} fit after reserving generation tokens. "
                    "Increase max_length within the model's supported context, reduce "
                    "the protocol's few-shot/output budget, or explicitly allow_truncation."
                )
            return super().tok_batch_encode(strings, padding_side=padding_side,
                                            left_truncate_len=left_truncate_len,
                                            truncation=truncation)
        return encoded

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # Upstream HFLM hardcodes use_cache=True. Handle it here instead of passing
        # another use_cache through **kwargs (which causes duplicate arguments).
        generation_kwargs.pop("use_cache", None)
        temperature = generation_kwargs.get("temperature", 0.0)
        if temperature == 0.0 and generation_kwargs.get("do_sample") is None:
            generation_kwargs["do_sample"] = False
        if generation_kwargs.get("do_sample") is False and temperature == 0.0:
            generation_kwargs.pop("temperature", None)
        stopping = stop_sequences_criteria(self.tokenizer, stop, context.shape[1], context.shape[0])
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            output = self.model.generate(
                input_ids=context, max_length=max_length,
                stopping_criteria=stopping, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=self.generation_use_cache, **generation_kwargs,
            )
        self.generation_receipts.append({
            "batch_size": int(context.shape[0]),
            "padded_prompt_tokens": int(context.shape[1]),
            "generated_columns": int(output.shape[1] - context.shape[1]),
            "batch_reached_length_cap": bool(output.shape[1] >= max_length),
            "use_cache": self.generation_use_cache,
        })
        return output
