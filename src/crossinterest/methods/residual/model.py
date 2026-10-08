"""Keep CIR updates fixed while a fresh, equally sized LoRA learns N."""

from __future__ import annotations

import math

import torch
from peft.tuners.lora.layer import Linear as LoraLinear
from torch.nn import functional as F


class FrozenCirLinear(torch.nn.Linear):
    def __init__(self, linear, source_a, source_b, scaling):
        torch.nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.register_buffer("cir_a", source_a.detach().clone())
        self.register_buffer("cir_b", source_b.detach().clone())
        self.cir_scaling = float(scaling)

    def forward(self, inputs):
        base = F.linear(inputs, self.weight, self.bias)
        update = F.linear(F.linear(inputs.to(self.cir_a.dtype), self.cir_a), self.cir_b)
        return (base + update * self.cir_scaling).to(base.dtype)


def freeze_cir_and_reset_n(model):
    before = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    modules = []
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLinear):
            continue
        if (set(layer.lora_A) != {"default"} or layer.merged
                or layer.use_dora.get("default", False)
                or type(layer.base_layer) is not torch.nn.Linear):
            raise ValueError(f"unsupported CIR LoRA layer: {name}")
        source_a = layer.lora_A["default"].weight
        source_b = layer.lora_B["default"].weight
        layer.base_layer = FrozenCirLinear(
            layer.base_layer, source_a, source_b, layer.scaling["default"])
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(source_a, a=math.sqrt(5))
            source_b.zero_()
        modules.append(name)
    if not modules:
        raise ValueError("no CIR LoRA layers were found")
    after = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if before != after:
        raise ValueError("residual transfer changed the trainable parameter count")
    return {"method": "frozen-cir-fresh-n-lora-v1", "modules": modules,
            "trainable_parameters": after, "cir_branch_dropout": 0.0,
            "input_id_rows": "continue_from_cir", "n_lora_initial_update": 0.0}