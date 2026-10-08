"""Auxiliary-only factor loss; the original novel-stage implementation is unchanged."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.cir.train import validate_stage
from crossinterest.methods.factorization.loss import ARMS, factorized_loss
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismCollator, MechanismTrainer


ROOT = HERE / "cir_factorization"


class FactorDataset(MechanismDataset):
    def __init__(self, *args, factor_arm: str, factor_masks: Path, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage != "auxiliary":
            raise ValueError("factor dataset is auxiliary-only")
        arrays = np.load(factor_masks, allow_pickle=False)
        if self.category_order != arrays["categories"].tolist():
            raise ValueError("factor mask category order differs")
        lookup = {sample_id: index for index, sample_id in enumerate(arrays["ids"].tolist())}
        indices = [lookup[sample_id] for sample_id in self.table["id"].to_pylist()]
        self.seen_masks = arrays["pseudo" if ARMS[factor_arm][2] else "seen"][indices]
        targets = arrays["targets"][indices]
        categories = self.table["target_interest"].to_pylist()
        if any(self.category_order[target] != category for target, category in zip(targets, categories)):
            raise ValueError("factor target alignment differs")
        if self.seen_masks[np.arange(len(indices)), targets].any():
            raise ValueError("factor target occurs in selected seen set")

    def __getitem__(self, index):
        encoded = super().__getitem__(index)
        encoded["seen_mask"] = self.seen_masks[index].tolist()
        return encoded


class FactorCollator(MechanismCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["seen_mask"] = torch.tensor([feature["seen_mask"] for feature in features], dtype=torch.bool)
        return batch


class FactorTrainer(MechanismTrainer):
    def __init__(self, *args, factor_arm: str, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage != "auxiliary" or self.controller is not None:
            raise ValueError("factor trainer only accepts the auxiliary stage without a relation head")
        self.factor_arm = factor_arm
        self.data_collator = FactorCollator(self.processing_class.pad_token_id)
        self.factor_totals = None
        self.factor_rows = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        positives = inputs.pop("positive_mask")
        seen = inputs.pop("seen_mask")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        positions = (labels != -100).long().argmax(dim=-1) - 1
        logits = outputs.logits[torch.arange(labels.shape[0], device=labels.device), positions]
        loss, diagnostics = factorized_loss(logits[:, self.label_token_ids], positives, seen, self.factor_arm)
        names = ("relation_loss", "exclusion_loss", "seen_mass", "direction_delta", "gradient_cosine", "scale", "fallback")
        totals = torch.stack([diagnostics[name].detach().float().sum() for name in names])
        self.factor_totals = totals if self.factor_totals is None else self.factor_totals + totals
        self.factor_rows += len(labels)
        self.factor_stat_names = names
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.factor_totals is not None and self.is_world_process_zero():
            write_json_atomic(Path(output_dir or self.args.output_dir) / "FACTOR_RUNTIME.json", {
                "arm": self.factor_arm, "rows_since_process_start_on_rank_zero": self.factor_rows,
                "scope": "rank_zero_training_trajectory_not_fixed_checkpoint_probe",
                "mean": dict(zip(self.factor_stat_names, (self.factor_totals / self.factor_rows).cpu().tolist())),
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
            })


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=("auxiliary", "novel"))
    parser.add_argument("--factor-arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--cir-preflight", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    if sha256_file(Path(baseline.__file__)) != "e23f154bbc1b09ff96b00de849a6a321691960602812835739d75d80df08736a":
        raise ValueError("factor training requires the verified isolated baseline")
    prepared = json.loads(options.cir_preflight.read_text())
    metadata = validate_stage(args, prepared, options.stage, options.smoke, options.cir_preflight)
    manifest = ROOT / "data/MANIFEST.json"
    expected_masks = json.loads(manifest.read_text())["files"]["masks.npz"]
    masks = ROOT / "data/masks.npz"
    if sha256_file(masks) != expected_masks:
        raise ValueError("frozen factor masks changed")
    identity = {"factor_arm": options.factor_arm, "factor_manifest_sha256": sha256_file(manifest),
                "factor_masks_sha256": expected_masks, "factor_loss_sha256": sha256_file(HERE / "cir_factor_loss.py")}
    if args.init_adapter is not None:
        previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())["experiment_metadata"]
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("factor stage handoff belongs to a different arm or contract")
    metadata.update(identity)
    if options.stage == "auxiliary":
        metadata["auxiliary_objective"] = "norm_matched_factorized_single_target_ce_112"
        dataset = partial(FactorDataset, stage="auxiliary", relation=False,
                          factor_arm=options.factor_arm, factor_masks=masks)
        trainer = partial(FactorTrainer, stage="auxiliary", relation=False,
                          init_adapter=None, factor_arm=options.factor_arm)
    else:
        dataset = partial(MechanismDataset, stage="novel", relation=False)
        trainer = partial(MechanismTrainer, stage="novel", relation=False, init_adapter=args.init_adapter)
    baseline.main(args=args, experiment_metadata=metadata, dataset_class=dataset, trainer_class=trainer)


if __name__ == "__main__":
    main()