"""Full-class residual updates preserving the original ID-set probability mass."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from crossinterest.evaluation.common import write_json_atomic


class MassReadout(nn.Module):
    def __init__(self, hidden_size: int, class_count: int):
        super().__init__()
        if hidden_size < 1 or class_count < 2:
            raise ValueError("invalid readout dimensions")
        self.weight = nn.Parameter(torch.zeros(class_count, hidden_size, dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            residual = F.linear(hidden.float(), self.weight.float())
            return residual - residual.mean(dim=-1, keepdim=True)


def preserve_mass(label_logits: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if label_logits.shape != residual.shape:
        raise ValueError("readout residual must match label logits")
    original = label_logits.float()
    residual = residual.float()
    normalization = torch.logsumexp(original, dim=-1, keepdim=True)
    normalization = normalization - torch.logsumexp(original + residual, dim=-1, keepdim=True)
    return original + (residual + normalization)


class MassReadoutController:
    def __init__(self, model, tokenizer, mapping: dict, directory: Path | None = None):
        self.label_ids = [tokenizer.convert_tokens_to_ids(token) for token in mapping["special_tokens"]]
        if len(set(self.label_ids)) != len(self.label_ids) or len(self.label_ids) < 2:
            raise ValueError("readout requires distinct label IDs")
        normalizers = [module.norm for module in model.modules()
                       if hasattr(module, "layers") and isinstance(getattr(module, "norm", None), nn.Module)]
        if len(normalizers) != 1:
            raise ValueError("expected one language decoder final norm")
        self.normalizer = normalizers[0]
        output_weight = model.get_output_embeddings().weight
        if min(self.label_ids) < 0 or max(self.label_ids) >= output_weight.shape[0]:
            raise ValueError("readout label outside model vocabulary")
        self.head = MassReadout(output_weight.shape[1], len(self.label_ids)).to(
            next(self.normalizer.parameters()).device)
        self.metadata = {"architecture": "mass-preserving-readout-v1", "schema_version": 1,
                         "hidden_size": output_weight.shape[1], "class_count": len(self.label_ids),
                         "label_tokens": mapping["special_tokens"], "label_ids": self.label_ids,
                         "mapping_fingerprint": mapping["mapping_fingerprint"], "first_id_only": True,
                         "initialization": "zero", "residual_centering": "uniform_classes"}
        if directory is not None:
            self.load(directory)
        model.add_module("interest_readout", self.head)
        self.frames, self.pending = [], None
        self.norm_calls, self.applied_calls = 0, 0
        self.handles = [self.normalizer.register_forward_hook(self.capture),
                        model.get_output_embeddings().register_forward_hook(self.fuse)]

    def set_frames(self, frames: list[dict]) -> None:
        if not frames or any(not isinstance(frame.get("prediction_position"), int)
                             or frame["prediction_position"] < 1 for frame in frames):
            raise ValueError("readout requires explicit answer-boundary positions")
        self.frames, self.pending = frames, None

    def capture(self, module, inputs, hidden):
        self.pending = None
        if not self.frames or len(self.frames) not in (1, hidden.shape[0]):
            raise ValueError("readout frames do not match forward batch")
        self.norm_calls += 1
        frames = self.frames * hidden.shape[0] if len(self.frames) == 1 else self.frames
        positions = [frame["prediction_position"] for frame in frames]
        if hidden.shape[1] <= max(positions):
            return
        selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        self.pending = (self.head(selected), hidden.shape[1], positions)

    def fuse(self, module, inputs, logits):
        if self.pending is None:
            return logits
        residual, full_length, positions = self.pending
        self.pending = None
        positions = [position - (full_length - logits.shape[1]) for position in positions]
        if not all(0 <= position < logits.shape[1] for position in positions):
            return logits
        batch_ids = torch.arange(logits.shape[0], device=logits.device)[:, None]
        position_ids = torch.tensor(positions, device=logits.device)[:, None]
        label_ids = torch.tensor(self.label_ids, device=logits.device)[None, :]
        fused = logits.float().clone()
        fused[batch_ids, position_ids, label_ids] = preserve_mass(
            logits[batch_ids, position_ids, label_ids], residual.to(logits.device))
        self.applied_calls += 1
        return fused

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                  str(directory / "interest_readout.safetensors"))
        write_json_atomic(directory / "interest_readout.json", self.metadata)

    def load(self, directory: Path) -> None:
        if json.loads((directory / "interest_readout.json").read_text()) != self.metadata:
            raise ValueError("readout checkpoint metadata mismatch")
        state = load_file(str(directory / "interest_readout.safetensors"))
        if any(not value.isfinite().all() for value in state.values()):
            raise ValueError("nonfinite readout checkpoint")
        self.head.load_state_dict(state, strict=True)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()