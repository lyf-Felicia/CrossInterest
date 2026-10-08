"""Category-group relation residuals inside the original causal LM forward."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training.prompts import render_history


def history_frame(tokenizer, input_ids: list[int], row: dict, prediction_position: int) -> dict:
    text = tokenizer.decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != input_ids:
        raise ValueError("history span tokenization does not round-trip exactly")
    normalizer = tokenizer.backend_tokenizer.normalizer
    normalize = normalizer.normalize_str if normalizer is not None else lambda text: text
    history = normalize(render_history(row, "behavior_interest"))
    beginning = text.find(history)
    if beginning < 0 or text.find(history, beginning + 1) >= 0:
        raise ValueError("history must occur exactly once in rendered input")
    positions = []
    cursor = beginning
    for number, (title, category) in enumerate(zip(
            row["historical_titles"], row["historical_interests"], strict=True), 1):
        line = normalize(f"{number}. [{category}] {title}")
        if text[cursor:cursor + len(line)] != line:
            raise ValueError("history line span mismatch")
        end = cursor + len(line)
        overlaps = [index for index, (start_offset, end_offset) in enumerate(encoded["offset_mapping"])
                    if start_offset < end and end_offset > cursor]
        if not overlaps or overlaps[-1] >= prediction_position:
            raise ValueError("history span reaches the prediction position")
        positions.append(overlaps[-1])
        cursor = end + 1
    return {"positions": positions, "categories": list(row["historical_interests"]),
            "prediction_position": prediction_position}


class RelationHead(nn.Module):
    def __init__(self, hidden_size: int, prototypes: torch.Tensor, dimension: int = 64):
        super().__init__()
        self.register_buffer("target_prototypes", prototypes.float().clone())
        self.semantic = nn.Linear(hidden_size, dimension, bias=False)
        self.history = nn.Linear(hidden_size, dimension, bias=False)
        self.pool = nn.Linear(dimension, 1, bias=False)
        self.importance = nn.Linear(dimension, 1, bias=False)
        self.source = nn.Linear(dimension, dimension, bias=False)
        self.target = nn.Linear(dimension, dimension, bias=False)
        self.interaction = nn.Linear(dimension, dimension, bias=False)
        self.readout = nn.Linear(dimension, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, item_states: torch.Tensor, groups: list[int],
                source_prototypes: torch.Tensor) -> torch.Tensor:
        item_states = self.history(F.normalize(item_states.float(), dim=-1))
        pooled = []
        group_ids = torch.tensor(groups, device=item_states.device)
        for group in range(source_prototypes.shape[0]):
            members = item_states[group_ids == group]
            weights = self.pool(members).squeeze(-1).softmax(dim=0)
            pooled.append((weights[:, None] * members).sum(dim=0))
        clusters = torch.stack(pooled)
        source = self.semantic(F.normalize(source_prototypes.float(), dim=-1))
        target = self.semantic(F.normalize(self.target_prototypes.float(), dim=-1))
        interactions = self.interaction(source[:, None, :] * target[None, :, :])
        pair = clusters[:, None, :] + self.source(source)[:, None, :] + self.target(target)[None, :, :] + interactions
        scores = self.readout(pair.tanh()).squeeze(-1)
        log_weights = self.importance(clusters).squeeze(-1).log_softmax(dim=0)
        relations = torch.logsumexp(log_weights[:, None] + scores, dim=0)
        return self.beta * (relations - relations.mean())


class RelationController:
    def __init__(self, model, tokenizer, mapping: dict, directory: Path | None = None,
                 dimension: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.label_ids = [tokenizer.convert_tokens_to_ids(token) for token in mapping["special_tokens"]]
        self.categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
        normalizers = [module.norm for module in model.modules()
                       if hasattr(module, "layers") and isinstance(getattr(module, "norm", None), nn.Module)]
        if len(normalizers) != 1:
            raise ValueError(f"expected one language decoder final norm, found {len(normalizers)}")
        self.normalizer = normalizers[0]
        device = next(self.normalizer.parameters()).device
        self.prototype_cache = {}
        prototypes = torch.stack([self.prototype(category) for category in self.categories]).to(device)
        self.head = RelationHead(prototypes.shape[1], prototypes, dimension).to(device)
        self.metadata = {"schema_version": 1, "dimension": dimension,
                         "hidden_size": prototypes.shape[1], "mapping_fingerprint": mapping["mapping_fingerprint"],
                         "label_tokens": mapping["special_tokens"], "architecture": "category-relation-v1",
                         "first_id_only": True, "initial_beta": 0.0}
        model.add_module("interest_relation", self.head)
        self.frames = []
        self.pending = None
        self.norm_calls = 0
        self.applied_calls = 0
        self.handles = [self.normalizer.register_forward_hook(self.capture),
                        model.get_output_embeddings().register_forward_hook(self.fuse)]
        if directory is not None:
            self.load(directory)

    def prototype(self, category: str) -> torch.Tensor:
        if category not in self.prototype_cache:
            embedding = self.model.get_input_embeddings()
            tokens = self.tokenizer.encode(category, add_special_tokens=False)
            if not tokens or set(tokens) & set(self.label_ids):
                raise ValueError("taxonomy prototypes require base text tokens")
            with torch.no_grad():
                values = embedding(torch.tensor(tokens, device=embedding.weight.device))
                self.prototype_cache[category] = values.float().mean(dim=0).detach().cpu()
        return self.prototype_cache[category]

    def set_frames(self, frames: list[dict]) -> None:
        if not frames or any(not frame["positions"] for frame in frames):
            raise ValueError("relation forward requires explicit nonempty history frames")
        self.frames = frames
        self.pending = None

    def capture(self, module, inputs, hidden):
        self.pending = None
        if not self.frames:
            raise ValueError("relation model used without history frames")
        self.norm_calls += 1
        if hidden.shape[1] <= max(max(frame["positions"]) for frame in self.frames):
            return
        if len(self.frames) not in (1, hidden.shape[0]):
            raise ValueError("history frames do not match forward batch")
        residuals = []
        for batch_index, frame in enumerate(self.frames):
            categories = list(dict.fromkeys(frame["categories"]))
            groups = [categories.index(category) for category in frame["categories"]]
            prototypes = torch.stack([self.prototype(category) for category in categories]).to(hidden.device)
            residuals.append(self.head(hidden[batch_index, frame["positions"]], groups, prototypes))
        residual = torch.stack(residuals)
        if len(self.frames) == 1:
            residual = residual.expand(hidden.shape[0], -1)
        self.pending = (residual, hidden.shape[1])

    def fuse(self, module, inputs, logits):
        if self.pending is None:
            return logits
        residual, full_length = self.pending
        self.pending = None
        positions = [frame["prediction_position"] - (full_length - logits.shape[1]) for frame in self.frames]
        if len(positions) == 1:
            positions *= logits.shape[0]
        if not all(0 <= position < logits.shape[1] for position in positions):
            return logits
        fused = logits.float().clone()
        batch_ids = torch.arange(logits.shape[0], device=logits.device)[:, None]
        position_ids = torch.tensor(positions, device=logits.device)[:, None]
        label_ids = torch.tensor(self.label_ids, device=logits.device)[None, :]
        fused[batch_ids, position_ids, label_ids] = (
            logits[batch_ids, position_ids, label_ids].float() + residual.to(device=logits.device, dtype=torch.float32))
        self.applied_calls += 1
        return fused

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                  str(directory / "interest_relation.safetensors"))
        write_json_atomic(directory / "interest_relation.json", self.metadata)

    def load(self, directory: Path) -> None:
        if json.loads((directory / "interest_relation.json").read_text()) != self.metadata:
            raise ValueError("relation checkpoint metadata mismatch")
        self.head.load_state_dict(load_file(str(directory / "interest_relation.safetensors")), strict=True)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()