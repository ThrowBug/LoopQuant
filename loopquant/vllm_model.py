"""Single-GPU Ouro execution with native vLLM decoder control flow and KV cache.

The HF skeleton is only the unchanged checkpoint tensor carrier. Inference uses
static per-loop views of its weights, without Python quantization context state.
"""

import inspect

import torch
from torch import nn
from torch.nn import functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models.ouro import OuroDecoderLayer

from .checkpoint import read_checkpoint_manifest, restore_checkpoint_tensors, _restore_structure
from .quantization import QUANTIZATION_CONTEXT
from .vllm_ops import EXECUTION_MODES, InferenceLinear, InferenceResidualNorm, TransformCache


def rotary_query_key(query, key, cos, sin):
    """Retain the HF RoPE convention and the checkpoint's exact rotary state."""
    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin)


def build_paged_attention(model, cache_config, *, prefix=""):
    core, config = model.model, model.config
    if not hasattr(core, "rotary_emb"):
        raise ValueError("vLLM LoopQuant requires Ouro's rotary_emb decoder architecture")
    layers = nn.ModuleList()
    count = len(core.layers)
    for index, decoder in enumerate(core.layers):
        attention = decoder.self_attn
        if "position_embeddings" not in inspect.signature(attention.forward).parameters:
            raise ValueError("unsupported Ouro attention signature for vLLM")
        head_dim = getattr(attention, "head_dim", config.hidden_size // config.num_attention_heads)
        for loop in range(model._loopquant_num_loops):
            layers.append(Attention(
                config.num_attention_heads, head_dim, getattr(attention, "scaling", head_dim ** -0.5),
                num_kv_heads=config.num_key_value_heads, cache_config=cache_config,
                per_layer_sliding_window=getattr(attention, "sliding_window", None),
                prefix=f"{prefix}model.layers.{loop * count + index}.self_attn.attn",
            ))
    return layers


class InferenceAttention(nn.Module):
    def __init__(self, attention, config, paged_attention, bits, loop, cgb, fast, linear_factory=None):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.dim = getattr(attention, "head_dim", config.hidden_size // self.heads)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, (linear_factory or InferenceLinear)(getattr(attention, name), bits, loop, cgb, fast=fast))
        self.attn = paged_attention

    def forward(self, positions, hidden_states, current_ut):
        # Native Ouro passes positions through unchanged; here it is the shared
        # (cos, sin) pair computed once by the saved HF rotary module.
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.heads, self.dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        value = self.v_proj(hidden_states).reshape(length, self.kv_heads * self.dim)
        query, key = rotary_query_key(query, key, *positions)
        output = self.attn(
            query.transpose(1, 2).reshape(length, self.heads * self.dim),
            key.transpose(1, 2).reshape(length, self.kv_heads * self.dim), value,
        )
        return self.o_proj(output.reshape(1, length, self.heads * self.dim))


class InferenceMLP(nn.Module):
    def __init__(self, mlp, bits, loop, cgb, fast, linear_factory=None):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, (linear_factory or InferenceLinear)(getattr(mlp, name), bits, loop, cgb, fast=fast))
        self.act_fn = getattr(mlp, "act_fn", F.silu)

    def forward(self, hidden):
        return self.down_proj(self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden))


class InferenceDecoder(OuroDecoderLayer):
    """Reuse native Ouro forward, supplying independent LoopQuant projections."""

    def __init__(self, decoder, config, paged_attention, bits, loop, cgb, fast, linear_factory=None):
        # Do not allocate native fused QKV/gate-up weights: their inputs differ.
        nn.Module.__init__(self)
        self.self_attn = InferenceAttention(decoder.self_attn, config, paged_attention, bits, loop, cgb, fast, linear_factory)
        self.mlp = InferenceMLP(decoder.mlp, bits, loop, cgb, fast, linear_factory)
        for name in ("input_layernorm", "input_layernorm_2",
                     "post_attention_layernorm", "post_attention_layernorm_2"):
            setattr(self, name, InferenceResidualNorm(getattr(decoder, name)))


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0})
class InferenceOuroModel(nn.Module):
    def __init__(self, *, vllm_config, core, config, attentions, allocation, cgb, mode, linear_factory=None):
        super().__init__()
        self.embed_tokens, self.rotary_emb = core.embed_tokens, core.rotary_emb
        self.norm = InferenceResidualNorm(core.norm)
        self.loops = nn.ModuleList([
            nn.ModuleList([
                InferenceDecoder(layer, config, attentions[index * len(allocation) + loop],
                                 bits, loop, cgb, mode == "fast", linear_factory)
                for index, layer in enumerate(core.layers)
            ]) for loop, bits in enumerate(allocation)
        ])
        self.prepared = False

    def prepare(self):
        cache = TransformCache()
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, InferenceLinear):
                    module.prepare(cache)
                elif hasattr(module, "prepare_for_inference"):
                    module.prepare_for_inference()
        self.prepared = True

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors=None, inputs_embeds: torch.Tensor | None = None):
        if not self.prepared:
            raise RuntimeError("checkpoint tensors have not been prepared for inference")
        hidden = (self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, positions.unsqueeze(0))
        for loop, layers in enumerate(self.loops):
            residual = None
            for layer in layers:
                hidden, residual = layer(position_embeddings, hidden, loop, residual)
            hidden, _ = self.norm(hidden, residual)
        return hidden.squeeze(0)


class LoopQuantOuroForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("LoopQuant vLLM currently supports one GPU only")
        if vllm_config.quant_config is not None:
            raise ValueError("LoopQuant snapshots already contain fake-quantized weights")
        self.directory = vllm_config.model_config.model
        self.manifest = read_checkpoint_manifest(self.directory)
        self.device, self.dtype = vllm_config.device_config.device, vllm_config.model_config.dtype
        self.num_loops = int(self.manifest["num_loops"])
        config = AutoConfig.from_pretrained(self.directory, trust_remote_code=True, local_files_only=True)
        if config.model_type != "ouro" or config.total_ut_steps != self.num_loops:
            raise ValueError("checkpoint config and Ouro loop count must match")
        self.config = config
        engine_config = getattr(vllm_config.model_config, "hf_config", config)
        self.execution_mode = getattr(engine_config, "loopquant_execution", "cached")
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"unknown LoopQuant execution mode: {self.execution_mode}")
        previous = vars(QUANTIZATION_CONTEXT).copy()
        try:
            QUANTIZATION_CONTEXT.configure(self.num_loops, self.manifest["allocation"])
            with init_empty_weights(include_buffers=False):
                self.hf_model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, attn_implementation="eager")
                _restore_structure(self.hf_model, self.manifest)
        finally:
            vars(QUANTIZATION_CONTEXT).update(previous)
        self.hf_model._loopquant_num_loops = self.num_loops
        self.attentions = build_paged_attention(
            self.hf_model, vllm_config.cache_config, prefix=f"{prefix}." if prefix else "")
        cgb = (self.hf_model.lctc_cgb
               if self.manifest["has_lctc"] and self.manifest["lctc_enabled"] else None)
        self.model = InferenceOuroModel(
            vllm_config=vllm_config, core=self.hf_model.model, config=config,
            attentions=self.attentions, allocation=self.manifest["allocation"],
            cgb=cgb, mode=self.execution_mode)

    def load_weights(self, weights):
        # Restore into the original HF structure. Execution views reference its
        # modules, so assign=True cannot leave stale/meta weight references.
        restore_checkpoint_tensors(self.hf_model, self.directory, self.manifest, device=self.device)
        if self.hf_model.model.embed_tokens.weight.dtype != self.dtype:
            raise ValueError("vLLM dtype must match the saved checkpoint embedding dtype")
        if self.execution_mode != "cached" and any(
            word in getattr(self.hf_model.model.rotary_emb, "rope_type", "default")
            for word in ("dynamic", "longrope")
        ):
            raise ValueError("dynamic RoPE requires LoopQuant cached execution")
        self.model.prepare()
        return {name for name, _ in self.named_parameters()}

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("pipeline parallelism is unsupported by LoopQuant")
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.hf_model.lm_head(hidden_states).float()
