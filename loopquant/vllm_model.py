"""Single-GPU, eager vLLM bridge for existing LoopQuant Ouro snapshots.

Keep the checkpoint's HF decoder/MLP/norm implementations and unfused projections.
Only attention/cache execution is replaced. No weights are re-quantized or rotated.
The module is imported lazily by vLLM workers, never on the default HF path.
"""

import inspect
from types import MethodType

import torch
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.model_executor.layers.attention import Attention

from .checkpoint import read_checkpoint_manifest, restore_checkpoint_tensors, _restore_structure
from .lctc import ACTIVE_LCTC_RUNTIME, _LCTCRuntime
from .quantization import QUANTIZATION_CONTEXT


def rotary_query_key(query, key, cos, sin):
    """Ouro's HF RoPE convention, retaining the saved rotary embedding state."""
    def rotate_half(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (query * cos + rotate_half(query) * sin,
            key * cos + rotate_half(key) * sin)


def attach_paged_attention(model, cache_config, *, prefix=""):
    """Return registered vLLM attention layers; HF state_dict stays unchanged."""
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
        per_loop = []
        for loop in range(model._loopquant_num_loops):
            per_loop.append(Attention(
                config.num_attention_heads, head_dim, head_dim ** -0.5,
                num_kv_heads=config.num_key_value_heads, cache_config=cache_config,
                per_layer_sliding_window=getattr(attention, "sliding_window", None),
                prefix=f"{prefix}model.layers.{loop * count + index}.self_attn.attn",
            ))
        layers.extend(per_loop)

        def make_forward(caches, heads, kv_heads, dimension):
            def forward(module, hidden_states, position_embeddings, attention_mask=None,
                        current_ut=0, **kwargs):
                batch, length, _ = hidden_states.shape
                if batch != 1:
                    raise ValueError("vLLM bridge expects one flattened token stream")
                query = module.q_proj(hidden_states).view(batch, length, heads, dimension).transpose(1, 2)
                key = module.k_proj(hidden_states).view(batch, length, kv_heads, dimension).transpose(1, 2)
                value = module.v_proj(hidden_states).reshape(length, kv_heads * dimension)
                query, key = rotary_query_key(query, key, *position_embeddings)
                output = caches[current_ut](
                    query.transpose(1, 2).reshape(length, heads * dimension),
                    key.transpose(1, 2).reshape(length, kv_heads * dimension), value,
                )
                return module.o_proj(output.reshape(1, length, heads * dimension)), None
            return forward

        attention.forward = MethodType(make_forward(
            per_loop, config.num_attention_heads, config.num_key_value_heads, head_dim,
        ), attention)
    return layers


class LoopQuantOuroForCausalLM(nn.Module):
    """vLLM model interface with exact saved fake-quantization operators."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("LoopQuant vLLM currently supports one GPU only")
        if vllm_config.quant_config is not None:
            raise ValueError("LoopQuant snapshots already contain fake-quantized weights")
        source = vllm_config.model_config.model
        self.directory = source
        self.manifest = read_checkpoint_manifest(source)
        self.device = vllm_config.device_config.device
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        if config.model_type != "ouro" or config.total_ut_steps != self.manifest["num_loops"]:
            raise ValueError("checkpoint config and Ouro loop count must match")
        self.config = config
        QUANTIZATION_CONTEXT.configure(self.manifest["num_loops"], self.manifest["allocation"])
        with init_empty_weights(include_buffers=False):
            self.hf_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="eager",
            )
            _restore_structure(self.hf_model, self.manifest)
        self.hf_model._loopquant_num_loops = self.manifest["num_loops"]
        self.attentions = attach_paged_attention(
            self.hf_model, vllm_config.cache_config, prefix=f"{prefix}." if prefix else "",
        )
        self.runtime = (_LCTCRuntime(self.hf_model.lctc_cgb, self.manifest["lctc_enabled"])
                        if self.manifest["has_lctc"] else None)

    def load_weights(self, weights):
        # Use the verified manifest/index, not vLLM's standard HF name mapping.
        # The supplied iterator is lazy and is intentionally not consumed.
        restore_checkpoint_tensors(self.hf_model, self.directory, self.manifest, device=self.device)
        return {name for name, _ in self.named_parameters()}

    def embed_input_ids(self, input_ids):
        return self.hf_model.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None:
            raise ValueError("pipeline parallelism is unsupported by the LoopQuant bridge")
        core = self.hf_model.model
        hidden = (self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds).unsqueeze(0)
        position_ids = positions.unsqueeze(0)
        position_embeddings = core.rotary_emb(hidden, position_ids)
        QUANTIZATION_CONTEXT.configure(self.manifest["num_loops"], self.manifest["allocation"])
        QUANTIZATION_CONTEXT.enabled = True
        token = ACTIVE_LCTC_RUNTIME.set(self.runtime)
        try:
            for loop in range(self.manifest["num_loops"]):
                QUANTIZATION_CONTEXT.current_loop = loop
                for layer in core.layers:
                    hidden = layer(hidden, attention_mask=None, position_ids=position_ids,
                                   past_key_value=None, use_cache=False, cache_position=positions,
                                   position_embeddings=position_embeddings, current_ut=loop)
                hidden = core.norm(hidden)
        finally:
            ACTIVE_LCTC_RUNTIME.reset(token)
        return hidden.squeeze(0)

    def compute_logits(self, hidden_states):
        return self.hf_model.lm_head(hidden_states).float()
