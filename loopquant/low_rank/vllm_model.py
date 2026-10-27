"""Static low-rank loop views using shared Ouro paged-attention execution."""

from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from ..checkpoint import read_checkpoint_manifest, restore_checkpoint_tensors
from ..vllm_model import build_paged_attention, InferenceOuroModel
from .checkpoint import restore_structure


class LowRankInferenceLinear(nn.Module):
    def __init__(self, linear, bits, loop, cgb=None, fast=False):
        super().__init__()
        self.linear, self.loop = linear, int(loop)

    def prepare_for_inference(self):
        if self.linear._inference_cache is None:
            self.linear.prepare_inference()

    def forward(self, value):
        return self.linear.forward_at_loop(value, self.loop)


class LowRankOuroForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("low_rank vLLM supports one GPU only")
        if vllm_config.quant_config is not None:
            raise ValueError("low_rank snapshots already contain fake-quantized weights")
        self.directory = vllm_config.model_config.model
        self.manifest = read_checkpoint_manifest(self.directory)
        if self.manifest["format"] != "low-rank-fake-quantized":
            raise ValueError("not a low_rank checkpoint")
        self.device, self.dtype = vllm_config.device_config.device, vllm_config.model_config.dtype
        config = AutoConfig.from_pretrained(self.directory, trust_remote_code=True, local_files_only=True)
        if config.model_type != "ouro" or config.total_ut_steps != self.manifest["num_loops"]:
            raise ValueError("low_rank checkpoint and Ouro loop count disagree")
        self.config = config
        with init_empty_weights(include_buffers=False):
            self.hf_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
            restore_structure(self.hf_model, self.manifest)
        attentions = build_paged_attention(self.hf_model, vllm_config.cache_config,
                                           prefix=f"{prefix}." if prefix else "")
        self.model = InferenceOuroModel(
            vllm_config=vllm_config, core=self.hf_model.model, config=config,
            attentions=attentions, allocation=self.manifest["allocation"], cgb=None,
            mode="cached", linear_factory=LowRankInferenceLinear)

    def load_weights(self, weights):
        restore_checkpoint_tensors(self.hf_model, self.directory, self.manifest, device=self.device)
        if self.hf_model.model.embed_tokens.weight.dtype != self.dtype:
            raise ValueError("vLLM dtype must match the saved embedding dtype")
        self.model.prepare()
        return {name for name, _ in self.named_parameters()}

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("pipeline parallelism is unsupported")
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.hf_model.lm_head(hidden_states).float()
