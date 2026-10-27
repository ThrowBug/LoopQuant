"""A fixed fake-quantized base plus separately trainable rank-k increments."""

from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import activation_bits, quantize_activation

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LowRankLinear(nn.Module):
    def __init__(self, linear, *, num_loops, rank_step, sharing, activation_bits, group_size=32):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if num_loops * rank_step > min(self.in_features, self.out_features):
            raise ValueError(f"rank_step*num_loops exceeds linear dimensions: {self.in_features}, {self.out_features}")
        self.weight, self.bias = linear.weight, linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.num_loops, self.rank_step, self.sharing = int(num_loops), int(rank_step), sharing
        if isinstance(activation_bits, bool) or not isinstance(activation_bits, int) or activation_bits not in (4, 8):
            raise ValueError("low_rank requires a uniform activation precision of 4 or 8 bits")
        self.activation_bits, self.group_size = activation_bits, int(group_size)
        self.loop_id = 0
        self.register_buffer("clip_ratios", torch.ones(num_loops, device=linear.weight.device, dtype=torch.float32))
        self.A = nn.ParameterList([nn.Parameter(linear.weight.new_zeros(rank_step, self.in_features), requires_grad=False)
                                   for _ in range(num_loops)])
        self.B = nn.ParameterList([nn.Parameter(linear.weight.new_zeros(self.out_features, rank_step), requires_grad=False)
                                   for _ in range(num_loops)])
        self._inference_cache = None

    def _apply(self, function, recurse=True):
        self._inference_cache = None
        return super()._apply(function, recurse)

    def factors(self, loop):
        if self.sharing == "independent":
            return self.A[loop], self.B[loop]
        if self._inference_cache is not None:
            return self._inference_cache[loop]
        matrices = (torch.cat(tuple(self.A), 0), torch.cat(tuple(self.B), 1))
        rank = (loop + 1) * self.rank_step
        return matrices[0][:rank], matrices[1][:, :rank]

    def forward_at_loop(self, value, loop):
        value = quantize_activation(value, self.activation_bits, self.group_size, self.clip_ratios[loop])
        a, b = self.factors(loop)
        # Normalize GEMM layouts in training and inference, including mixed-dtype
        # FP32 masters. Frozen inference prefixes are prepared contiguously once.
        correction = F.linear(F.linear(value, a.to(value.dtype).contiguous()), b.to(value.dtype).contiguous())
        return F.linear(value, self.weight, self.bias) + correction

    def forward(self, value):
        return self.forward_at_loop(value, self.loop_id)

    @torch.no_grad()
    def prepare_inference(self):
        if self.sharing == "prefix":
            a, b = torch.cat(tuple(self.A), 0), torch.cat(tuple(self.B), 1)
            self._inference_cache = {
                loop: (a[:(loop + 1) * self.rank_step], b[:, :(loop + 1) * self.rank_step].contiguous())
                for loop in range(self.num_loops)}


def iter_linears(block):
    return ((name, module) for name, module in block.named_modules() if isinstance(module, LowRankLinear))


def bind_loop_indices(model):
    """Bind Ouro current_ut locally, without the original quantization context."""
    for block in model.model.layers:
        if getattr(block, "_low_rank_bound", False):
            continue
        original = block.forward
        modules = tuple(module for _, module in iter_linears(block))

        def bind(function, linears):
            @wraps(function)
            def forward(*args, **kwargs):
                loop = int(kwargs.get("current_ut", 0))
                for linear in linears:
                    if not 0 <= loop < linear.num_loops:
                        raise ValueError("current_ut is outside the saved loop range")
                    linear.loop_id = loop
                return function(*args, **kwargs)
            return forward

        block.forward = bind(original, modules)
        block._low_rank_bound = True


def wrap_linears(model, config):
    bits = activation_bits(config)
    for block in model.model.layers:
        for name, module in list(block.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS:
                block.set_submodule(name, LowRankLinear(
                    module, num_loops=config.num_loops, rank_step=config.method["rank_step"],
                    sharing=config.method["sharing"], activation_bits=bits,
                    group_size=config.quantization["group_size"]))
    bind_loop_indices(model)


def prepare_inference(model):
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            module.prepare_inference()
