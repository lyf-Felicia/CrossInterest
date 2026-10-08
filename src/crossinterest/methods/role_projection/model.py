"""Role-projected residuals for the unchanged full-history causal language model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from crossinterest.methods.mechanism.model import RelationController


class TrilinearBranch(nn.Module):
    def __init__(self, hidden_size: int, dimension: int):
        super().__init__()
        self.source = nn.Linear(hidden_size, dimension, bias=False)
        self.target = nn.Linear(hidden_size, dimension, bias=False)
        self.context = nn.Linear(hidden_size, dimension, bias=False)
        nn.init.zeros_(self.context.weight)

    def forward(self, hidden: torch.Tensor, prototypes: torch.Tensor,
                visible: torch.Tensor, projection: str,
                history_prototypes: torch.Tensor | None = None) -> torch.Tensor:
        source = F.linear(prototypes.float(), self.source.weight.float())
        target = F.linear(prototypes.float(), self.target.weight.float())
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        source = source - source_mean
        target = target - target_mean
        history_source, history_target = source, target
        if history_prototypes is not None:
            history_source = F.linear(history_prototypes.float(), self.source.weight.float()) - source_mean
            history_target = F.linear(history_prototypes.float(), self.target.weight.float()) - target_mean
        context = F.linear(F.normalize(hidden.float(), dim=-1), self.context.weight.float())
        membership = visible.float()
        source_sum = membership @ history_source
        target_sum = membership @ history_target
        pair_sum = (source_sum.square() - membership @ history_source.square()) / 2
        cross_sum = source_sum * target_sum - membership @ (history_source * history_target)
        raw = (context * pair_sum) @ target.T
        exchanged = (context * cross_sum) @ source.T
        if projection == "compatible":
            scores = (raw + exchanged) / 3
        elif projection == "directional":
            scores = (2 * raw - exchanged) / 3
        elif projection == "unconstrained":
            scores = raw
        else:
            raise ValueError(f"unknown role projection: {projection}")
        count = membership.sum(dim=-1, keepdim=True)
        denominator = count * (count - 1) / 2
        return torch.where(count >= 2, scores / denominator.clamp_min(1), 0)


class RoleProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, prototypes: torch.Tensor,
                 dimension: int = 64, mode: str = "projected", stage: str = "auxiliary"):
        super().__init__()
        if (prototypes.ndim != 2 or prototypes.shape[1] != hidden_size
                or prototypes.shape[0] < 2 or dimension < 1):
            raise ValueError("invalid taxonomy prototype or projection dimensions")
        if mode not in ("projected", "unconstrained"):
            raise ValueError(f"unknown role head mode: {mode}")
        self.register_buffer("target_prototypes", prototypes.float().detach().clone())
        self.compatible = TrilinearBranch(hidden_size, dimension)
        self.directional = TrilinearBranch(hidden_size, dimension)
        self.mode = mode
        self.dimension = dimension
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        if stage not in ("auxiliary", "novel"):
            raise ValueError(f"unknown role head stage: {stage}")
        self.stage = stage
        self.directional.requires_grad_(stage == "novel")

    def forward(self, hidden: torch.Tensor, visible: torch.Tensor,
                history_prototypes: torch.Tensor | None = None) -> torch.Tensor:
        category_count = (self.target_prototypes if history_prototypes is None else history_prototypes).shape[0]
        if (hidden.ndim != 2 or visible.shape != (hidden.shape[0], category_count)
                or visible.dtype != torch.bool):
            raise ValueError("expected batched hidden states and a boolean visible-category set")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            taxonomy_scale = self.target_prototypes.float().square().mean().sqrt().clamp_min(1e-12)
            prototypes = self.target_prototypes.float() / taxonomy_scale
            history = history_prototypes.float() / taxonomy_scale if history_prototypes is not None else None
            scores = self.compatible(hidden, prototypes, visible,
                "compatible" if self.mode == "projected" else "unconstrained", history)
            if self.stage == "novel":
                scores = scores + self.directional(hidden, prototypes, visible,
                    "directional" if self.mode == "projected" else "unconstrained", history)
        return scores


class RoleProjectionController(RelationController):
    def __init__(self, model, tokenizer, mapping: dict, directory: Path | None = None,
                 dimension: int = 64, mode: str = "projected", stage: str = "auxiliary",
                 allow_auxiliary_handoff: bool = False):
        self.model, self.tokenizer, self.mapping = model, tokenizer, mapping
        self.label_ids = [tokenizer.convert_tokens_to_ids(token) for token in mapping["special_tokens"]]
        self.categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
        if len(set(self.label_ids)) != len(self.categories):
            raise ValueError("role head label IDs must be distinct")
        self.category_indices = {category: index for index, category in enumerate(self.categories)}
        normalizers = [module.norm for module in model.modules()
                       if hasattr(module, "layers") and isinstance(getattr(module, "norm", None), nn.Module)]
        if len(normalizers) != 1:
            raise ValueError(f"expected one language decoder final norm, found {len(normalizers)}")
        self.normalizer = normalizers[0]
        device = next(self.normalizer.parameters()).device
        self.prototype_cache = {}
        prototypes = torch.stack([self.prototype(category) for category in self.categories]).to(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260825)
            self.head = RoleProjectionHead(prototypes.shape[1], prototypes, dimension, mode, stage).to(device)
        self.metadata = {"schema_version": 1, "dimension": dimension,
                         "hidden_size": prototypes.shape[1], "mapping_fingerprint": mapping["mapping_fingerprint"],
                         "label_tokens": mapping["special_tokens"], "architecture": "role-projection-v2",
                         "mode": mode, "stage": stage, "first_id_only": True,
                         "context_initialization": "zero", "category_centering": "uniform_output_taxonomy",
                         "taxonomy_scaling": "global_output_rms",
                         "history_taxonomy": "all_visible_categories"}
        if directory is not None:
            self.load(directory, allow_auxiliary_handoff=allow_auxiliary_handoff)
        model.add_module("interest_relation", self.head)
        self.frames, self.pending = [], None
        self.norm_calls, self.applied_calls = 0, 0
        self.handles = [self.normalizer.register_forward_hook(self.capture),
                        model.get_output_embeddings().register_forward_hook(self.fuse)]

    def set_frames(self, frames: list[dict]) -> None:
        if not frames:
            raise ValueError("role projection requires explicit history frames")
        for frame in frames:
            if frame["prediction_position"] < 1 or not frame["categories"]:
                raise ValueError("invalid role history frame")
            if any(not isinstance(category, str) or not category for category in frame["categories"]):
                raise ValueError("invalid visible category")
        self.frames = frames
        self.pending = None

    def capture(self, module, inputs, hidden):
        self.pending = None
        if not self.frames:
            raise ValueError("role projection used without history frames")
        self.norm_calls += 1
        if len(self.frames) not in (1, hidden.shape[0]):
            raise ValueError("history frames do not match forward batch")
        frames = self.frames * hidden.shape[0] if len(self.frames) == 1 else self.frames
        positions = [frame["prediction_position"] for frame in frames]
        if hidden.shape[1] <= max(positions):
            return
        categories = sorted({category for frame in frames for category in frame["categories"]})
        indices_by_category = {category: index for index, category in enumerate(categories)}
        prototypes = torch.stack([self.prototype(category) for category in categories]).to(hidden.device)
        membership = torch.zeros(hidden.shape[0], len(categories), dtype=torch.bool, device=hidden.device)
        for batch_index, frame in enumerate(frames):
            indices = [indices_by_category[category] for category in set(frame["categories"])]
            membership[batch_index, indices] = True
        selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        self.pending = (self.head(selected, membership, prototypes), hidden.shape[1])

    def load(self, directory: Path, allow_auxiliary_handoff: bool = False) -> None:
        stored = json.loads((directory / "interest_relation.json").read_text())
        expected = dict(self.metadata)
        if allow_auxiliary_handoff:
            if expected["stage"] != "novel":
                raise ValueError("auxiliary handoff only applies when entering novel training")
            expected["stage"] = "auxiliary"
        if stored != expected:
            raise ValueError("role projection checkpoint metadata mismatch")
        self.head.load_state_dict(load_file(str(directory / "interest_relation.safetensors")), strict=True)