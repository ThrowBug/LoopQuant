"""Small HF Ouro-style decoder with real causal attention and RoPE for bridge tests."""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class PagedOuroConfig(PretrainedConfig):
    model_type = "ouro"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size, self.intermediate_size, self.vocab_size = 32, 64, 6
        self.num_hidden_layers, self.total_ut_steps = 1, 4
        self.num_attention_heads, self.num_key_value_heads = 4, 2
        self.max_position_embeddings = 64


class TestRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        result = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return self.weight * result.to(value.dtype)


class Rotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / 10000 ** (torch.arange(0, 8, 2).float() / 8), persistent=False)

    def forward(self, hidden, positions):
        phases = positions.float().unsqueeze(-1) * self.inv_freq
        phases = torch.cat((phases, phases), -1)
        return phases.cos().to(hidden.dtype), phases.sin().to(hidden.dtype)


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(32, 32, bias=False), nn.Linear(32, 16, bias=False)
        self.v_proj, self.o_proj = nn.Linear(32, 16, bias=False), nn.Linear(32, 32, bias=False)
        self.head_dim, self.sliding_window = 8, None

    def forward(self, hidden_states, position_embeddings, attention_mask=None, current_ut=0, **kwargs):
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, 4, 8).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, 2, 8).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, 2, 8).transpose(1, 2)
        def rotate_half(x):
            first, second = x.chunk(2, -1)
            return torch.cat((-second, first), -1)
        cos, sin = (part.unsqueeze(1) for part in position_embeddings)
        query, key = query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin
        key, value = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
        scores = query @ key.transpose(-1, -2) / 8 ** 0.5
        mask = torch.ones(length, length, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, -torch.inf)
        result = torch.softmax(scores.float(), -1).to(query.dtype) @ value
        return self.o_proj(result.transpose(1, 2).reshape(batch, length, 32)), None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = CausalAttention()
        self.input_layernorm, self.input_layernorm_2 = TestRMSNorm(32), TestRMSNorm(32)
        self.post_attention_layernorm, self.post_attention_layernorm_2 = TestRMSNorm(32), TestRMSNorm(32)
        self.mlp = MLP()

    def forward(self, hidden, position_embeddings, current_ut=0, **kwargs):
        attended, _ = self.self_attn(self.input_layernorm(hidden), position_embeddings, current_ut=current_ut)
        hidden = hidden + self.input_layernorm_2(attended)
        return hidden + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(hidden)))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (
            nn.Linear(32, 64, bias=False), nn.Linear(32, 64, bias=False), nn.Linear(64, 32, bias=False))

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens, self.rotary_emb = nn.Embedding(6, 32), Rotary()
        self.layers, self.norm = nn.ModuleList([Decoder()]), TestRMSNorm(32)

    def forward(self, input_ids, loops):
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        embedding = self.rotary_emb(hidden, positions)
        for loop in range(loops):
            for layer in self.layers:
                hidden = layer(hidden, position_embeddings=embedding, current_ut=loop)
            hidden = self.norm(hidden)
        return hidden


class PagedOuroForCausalLM(PreTrainedModel):
    config_class = PagedOuroConfig

    def __init__(self, config):
        super().__init__(config)
        self.model, self.lm_head = Backbone(), nn.Linear(32, 6, bias=False)
        self.post_init()

    def forward(self, input_ids, exit_at_step=None, **kwargs):
        return CausalLMOutputWithPast(logits=self.lm_head(self.model(input_ids, self.config.total_ut_steps)))
