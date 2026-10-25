"""lm-eval's vLLM backend with the same prompt-budget policy as our HF path."""

from lm_eval.models.vllm_causallms import VLLM


class EvaluationVLLM(VLLM):
    def __init__(self, *args, allow_truncation=False, **kwargs):
        self.allow_prompt_truncation = allow_truncation
        self.generation_receipts = []
        super().__init__(*args, **kwargs)

    def generate_until(self, requests, disable_tqdm=False):
        if not requests:
            return []
        for request in requests:
            prompt, kwargs = request.args
            budget = int(kwargs.get("max_gen_toks", self.max_gen_toks))
            if not 0 < budget < self.max_length:
                raise ValueError("max_gen_toks must be positive and less than max_model_len")
            length = len(self.tok_encode(prompt))
            if length + budget > self.max_length and not self.allow_prompt_truncation:
                raise ValueError(f"Prompt requires {length} tokens but only {self.max_length - budget} fit")
        return super().generate_until(requests, disable_tqdm=disable_tqdm)

    def _model_generate(self, requests, generate=False, sampling_params=None):
        outputs = super()._model_generate(requests, generate=generate, sampling_params=sampling_params)
        if generate:
            parameters = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(outputs)
            for prompt, output, params in zip(requests, outputs, parameters, strict=True):
                result = output.outputs[0]
                self.generation_receipts.append({
                    "prompt_tokens": len(prompt), "generated_tokens": len(result.token_ids),
                    # Harness 0.4.11 passes one SamplingParams per prompt.
                    # Its fallback when params is None generates one token.
                    "max_gen_toks": params.max_tokens if params is not None else 1,
                    "finish_reason": result.finish_reason,
                    "use_cache": True, "backend": "vllm",
                })
        return outputs
