"""Small HF recurrent model for offline checkpoint tests; not a benchmark model."""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class TinyOuroConfig(PretrainedConfig):
    model_type = "ouro"

    def __init__(self, vocab_size=6, hidden_size=32, intermediate_size=64,
                 num_hidden_layers=1, total_ut_steps=4, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 0)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.total_ut_steps = total_ut_steps


class TinyRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * self.weight


class TinyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, intermediate = config.hidden_size, config.intermediate_size
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(width, width, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.mlp.up_proj = nn.Linear(width, intermediate, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate, width, bias=False)
        self.input_layernorm = TinyRMSNorm(width)
        self.post_attention_layernorm = TinyRMSNorm(width)

    def forward(self, value, current_ut=0):
        normalized = self.input_layernorm(value)
        attention = self.self_attn
        projected = attention.q_proj(normalized) + attention.k_proj(normalized) + attention.v_proj(normalized)
        value = value + attention.o_proj(projected) * 0.1
        normalized = self.post_attention_layernorm(value)
        return value + self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(normalized)) *
                                         self.mlp.up_proj(normalized)) * 0.1


class TinyBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TinyRMSNorm(config.hidden_size)
        self.register_buffer("inv_freq", torch.arange(2, dtype=torch.float32), persistent=False)
        self.original_inv_freq = self.inv_freq
        self.max_seq_len_cached = 16

    def forward(self, input_ids, loops):
        hidden = self.embed_tokens(input_ids)
        hidden = hidden + (self.inv_freq[0] + self.original_inv_freq[0]).to(hidden) * 0.001
        for step in range(loops):
            for layer in self.layers:
                hidden = layer(hidden, current_ut=step)
            hidden = self.norm(hidden)
        return hidden


class TinyOuroForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TinyOuroConfig
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        self.model = TinyBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                exit_at_step=None, exit_threshold=None, use_weighted_exit=False, **kwargs):
        loops = self.config.total_ut_steps if exit_at_step is None else exit_at_step + 1
        return CausalLMOutputWithPast(logits=self.lm_head(self.model(input_ids, loops)))
