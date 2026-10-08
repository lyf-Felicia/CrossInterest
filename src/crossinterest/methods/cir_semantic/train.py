"""Auxiliary-only hidden-history semantic prediction without changing CIR inputs."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from safetensors.numpy import load_file as load_arrays
from safetensors.torch import load_file, save_file

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.cir_semantic.data import DATA, ROOT, text_hash
from crossinterest.methods.cir.train import validate_stage
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismCollator, MechanismTrainer, positive_set_loss


class SemanticDataset(MechanismDataset):
    def __init__(self, *args, shuffled: bool, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage != "auxiliary":
            raise ValueError("semantic targets are auxiliary-only")
        manifest = json.loads((ROOT / "MANIFEST.json").read_text())
        if sha256_file(ROOT / "targets.safetensors") != manifest["targets_sha256"]:
            raise ValueError("semantic target cache changed")
        source_ids = pq.read_table(DATA / "train.parquet", columns=["id"])["id"].to_pylist()
        if text_hash(source_ids) != manifest["audit"]["row_order_sha256"]:
            raise ValueError("semantic target row order mismatch")
        lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
        indices = np.asarray([lookup[sample_id] for sample_id in self.table["id"].to_pylist()])
        arrays = load_arrays(ROOT / "targets.safetensors")
        target_indices = arrays["permutation"][indices] if shuffled else indices
        self.targets = arrays["targets"][target_indices].copy()

    def __getitem__(self, index):
        encoded = super().__getitem__(index)
        encoded["semantic_target"] = self.targets[index]
        return encoded


class SemanticCollator(MechanismCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["semantic_target"] = torch.as_tensor(np.stack([feature["semantic_target"] for feature in features]))
        return batch


class SemanticController:
    def __init__(self, model, dimension: int, identity: dict):
        normalizers = [module.norm for module in model.modules()
                       if hasattr(module, "layers") and isinstance(getattr(module, "norm", None), nn.Module)]
        if len(normalizers) != 1:
            raise ValueError("semantic prediction requires exactly one decoder final norm")
        normalizer = normalizers[0]
        parameter = next(normalizer.parameters())
        hidden_size = parameter.numel()
        self.head = nn.Sequential(nn.LayerNorm(hidden_size, elementwise_affine=False),
                                  nn.Linear(hidden_size, dimension, bias=False)).to(parameter.device)
        nn.init.zeros_(self.head[1].weight)
        model.add_module("cir_semantic_projection", self.head)
        self.metadata = {"hidden_size": hidden_size, "dimension": dimension, "identity": identity,
                         "architecture": "nonaffine_layernorm_biasfree_linear", "initialization": "zero",
                         "prediction_position": "before_first_interest_id", "auxiliary_only": True}
        self.positions = None
        self.prediction = None
        self.handle = normalizer.register_forward_hook(self.capture)

    def capture(self, module, inputs, hidden):
        if self.positions is None or len(self.positions) != hidden.shape[0]:
            raise ValueError("semantic projection requires explicit prediction positions")
        if (self.positions < 0).any() or (self.positions >= hidden.shape[1]).any():
            raise ValueError("semantic prediction position outside hidden sequence")
        states = hidden[torch.arange(hidden.shape[0], device=hidden.device), self.positions]
        self.prediction = self.head(states.float())

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                  str(directory / "cir_semantic_projection.safetensors"))
        write_json_atomic(directory / "cir_semantic_projection.json", self.metadata)

    def load(self, directory: Path):
        if json.loads((directory / "cir_semantic_projection.json").read_text()) != self.metadata:
            raise ValueError("semantic projection checkpoint identity mismatch")
        self.head.load_state_dict(load_file(str(directory / "cir_semantic_projection.safetensors")), strict=True)

    def close(self):
        self.handle.remove()


class SemanticTrainer(MechanismTrainer):
    def __init__(self, *args, semantic_dimension: int, semantic_identity: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage != "auxiliary" or self.controller is not None:
            raise ValueError("semantic prediction only extends headless CIR pretraining")
        self.data_collator = SemanticCollator(self.processing_class.pad_token_id)
        self.semantic_controller = SemanticController(self.model, semantic_dimension, semantic_identity)
        if self.is_world_process_zero():
            write_json_atomic(Path(self.args.output_dir) / "SEMANTIC_RUNTIME.json", {
                **self.semantic_controller.metadata, "loss_weight": 1.0,
                "distance_reduction": "sum_dimensions_mean_examples", "target_detached": True,
                "trainable_projection_parameters": sum(parameter.numel() for parameter in self.semantic_controller.head.parameters()),
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
                "targets_passed_to_backbone": False})

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        targets = inputs.pop("semantic_target").detach().float()
        positives, labels = inputs.pop("positive_mask"), inputs.pop("labels")
        positions = (labels != -100).long().argmax(dim=-1) - 1
        self.semantic_controller.positions = positions
        self.semantic_controller.prediction = None
        outputs = model(**inputs)
        logits = outputs.logits[torch.arange(labels.shape[0], device=labels.device), positions]
        classification = positive_set_loss(logits[:, self.label_token_ids], positives)
        prediction = self.semantic_controller.prediction
        if prediction is None or prediction.shape != targets.shape:
            raise ValueError("semantic prediction missing or misaligned")
        distance = (prediction.float() - targets).square().sum(dim=-1).mean()
        loss = classification + distance
        self.semantic_controller.prediction = None
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.semantic_controller.save(Path(output_dir or self.args.output_dir))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.semantic_controller.load(Path(resume_from_checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=("auxiliary",))
    parser.add_argument("--cir-preflight", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    prepared = json.loads(options.cir_preflight.read_text())
    metadata = validate_stage(args, prepared, options.stage, options.smoke, options.cir_preflight)
    semantic = prepared["semantic"]
    if sha256_file(ROOT / "targets.safetensors") != semantic["targets_sha256"]:
        raise ValueError("semantic target freeze differs")
    metadata.update(auxiliary_objective="cir_id_ce_plus_class_centered_hidden_semantic_squared_distance", semantic=semantic)
    baseline.main(args=args, experiment_metadata=metadata,
        dataset_class=partial(SemanticDataset, stage="auxiliary", relation=False, shuffled=semantic["shuffled"]),
        trainer_class=partial(SemanticTrainer, stage="auxiliary", relation=False, init_adapter=None,
                              semantic_dimension=semantic["dimension"], semantic_identity=semantic))


if __name__ == "__main__":
    main()