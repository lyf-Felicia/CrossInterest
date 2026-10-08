"""Keep complete categorical quadruples together during auxiliary pretraining."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.prompts import render_history
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.consistency.loss import ARMS, VIEW_ORDER, dual_loss
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismTrainer


def dual_messages(row: dict, mapping: dict, view: str) -> list[dict]:
    if view not in VIEW_ORDER:
        raise ValueError("unknown dual view")
    historical = view.startswith("a_")
    target = row["target_interest" if historical else "future_interest"]
    evidence = None
    if view == "a_given_rb":
        evidence = "Observed next new category: " + row["future_interest"]
    elif view == "b_given_ra":
        evidence = "Removed historical category: " + row["target_interest"]
    task = ("Infer the removed historical category." if historical else
            "Predict the next newly observed category in the original history.")
    content = ("All entries from one historical category have been removed. "
               "The next new category is defined relative to the complete original history.\n\n"
               + render_history(row, "behavior_interest") + "\n\n"
               + ((evidence + "\n\n") if evidence is not None else "")
               + task + " Respond with only its single-token Interest ID in the exact format "
               "<interest_NNN>. Do not output a category name or explanation.")
    return [{"role": "user", "content": content},
            {"role": "assistant", "content": mapping["category_to_id"][target]}]


class DualDataset(baseline.AmazonSftDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.selection_weights is not None or len(self.table) != self.size:
            raise ValueError("dual pairs cannot be selected or reweighted")
        self.table = self.table.append_column(
            "future_interest", pq.read_table(args[0], columns=["future_interest"])["future_interest"])
        self.category_order = [self.mapping["id_to_category"][token]
                               for token in self.mapping["special_tokens"]]
        self.category_indices = {label: index for index, label in enumerate(self.category_order)}

    def __getitem__(self, index):
        row = self.table.slice(index, 1).to_pylist()[0]
        views = [baseline.tokenize_assistant_only(self.tokenizer, dual_messages(row, self.mapping, view))
                 for view in VIEW_ORDER]
        if any(len(view["input_ids"]) > self.max_length for view in views):
            raise ValueError("dual view truncation is forbidden")
        return {"views": views, "dual_targets": [self.category_indices[row["target_interest"]],
                                                 self.category_indices[row["future_interest"]]]}


class DualCollator(baseline.AssistantOnlyCollator):
    def __call__(self, features):
        if any(len(feature["views"]) != 4 for feature in features):
            raise ValueError("each collated pair must contain all four views")
        batch = super().__call__([view for feature in features for view in feature["views"]])
        batch["dual_targets"] = torch.tensor([feature["dual_targets"] for feature in features],
                                               dtype=torch.long)
        return batch


class DualTrainer(MechanismTrainer):
    def __init__(self, *args, dual_arm: str, **kwargs):
        super().__init__(*args, stage="auxiliary", relation=False, init_adapter=None, **kwargs)
        self.dual_arm = dual_arm
        self.data_collator = DualCollator(self.processing_class.pad_token_id)
        self.dual_totals = None
        self.dual_pairs = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        targets = inputs.pop("dual_targets")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        positions = (labels != -100).long().argmax(dim=-1) - 1
        if (positions < 0).any() or not (labels != -100).any(dim=-1).all():
            raise ValueError("dual views have invalid target positions")
        selected = outputs.logits[torch.arange(len(labels), device=labels.device), positions]
        selected = selected[:, self.label_token_ids].reshape(len(targets), 4, -1)
        loss, diagnostics = dual_loss(selected, targets, self.dual_arm)
        self.dual_names = tuple(diagnostics)
        totals = torch.stack([value.sum() for value in diagnostics.values()])
        self.dual_totals = totals if self.dual_totals is None else self.dual_totals + totals
        self.dual_pairs += len(targets)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.is_world_process_zero() and self.dual_totals is not None:
            write_json_atomic(Path(output_dir or self.args.output_dir) / "DUAL_RUNTIME.json", {
                "arm": self.dual_arm, "coefficient": ARMS[self.dual_arm],
                "view_order": VIEW_ORDER, "pairs_since_process_start_on_rank_zero": self.dual_pairs,
                "mean": dict(zip(self.dual_names, (self.dual_totals / self.dual_pairs).cpu().tolist())),
                "scope": "training_trajectory_not_heldout_knowledge_evidence",
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=("auxiliary", "novel"))
    parser.add_argument("--dual-arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--dual-preflight", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    prepared = json.loads(options.dual_preflight.read_text())
    stage_key = ("smoke_" if options.smoke else "") + options.stage
    expected = prepared["files"][stage_key]
    if sha256_file(Path(baseline.__file__)) != prepared["baseline_sha256"]:
        raise ValueError("dual training requires the pinned isolated baseline")
    if (args.view, args.label_mode, args.loss_mode) != ("behavior_interest", "interest_id", "causal_lm"):
        raise ValueError("original B+I causal protocol required")
    if args.selection_file is not None or args.limit is not None or args.max_steps != -1:
        raise ValueError("complete frozen stage required")
    effective_batch = 8 if options.stage == "auxiliary" else 32
    if (args.epochs, args.learning_rate, args.expected_effective_batch_size) != (1, 1e-4, effective_batch):
        raise ValueError("dual stage budget differs")
    if (args.expected_train_rows != expected["rows"]
            or sha256_file(args.train_file) != expected["sha256"]
            or sha256_file(args.label_map) != prepared["mapping_sha256"]
            or Path(args.model).name != prepared["model_revision"]):
        raise ValueError("dual stage data/model differs")
    if (options.stage == "novel") != (args.init_adapter is not None):
        raise ValueError("fresh auxiliary and complete auxiliary-to-N handoff required")
    identity = {"dual_arm": options.dual_arm, "dual_coefficient": ARMS[options.dual_arm],
                "smoke": options.smoke, "preflight_sha256": sha256_file(options.dual_preflight),
                "dual_loss_sha256": sha256_file(Path(__file__).with_name("cir_dual_loss.py"))}
    if args.init_adapter is not None:
        previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())
        auxiliary = prepared["files"][("smoke_" if options.smoke else "") + "auxiliary"]
        require_fields(previous, {"global_step": auxiliary["steps"], "sample_count": auxiliary["rows"],
                                  "train_sha256": auxiliary["sha256"], "epochs": 1.0}, "dual handoff")
        require_fields(previous["experiment_metadata"], {**identity, "stage": "auxiliary"}, "dual identity")
    metadata = {**identity, "task": "amazon_dual_consistency_sft", "stage": options.stage,
                "prompt_schema_version": "interest-dual-v1" if options.stage == "auxiliary" else "interest-mechanism-v1",
                "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True,
                "relation_head": False, "effective_view_batch_size": 32,
                "sample_unit": "four_view_pair" if options.stage == "auxiliary" else "N_row"}
    if options.stage == "auxiliary":
        dataset = DualDataset
        trainer = partial(DualTrainer, dual_arm=options.dual_arm)
    else:
        dataset = partial(MechanismDataset, stage="novel", relation=False)
        trainer = partial(MechanismTrainer, stage="novel", relation=False, init_adapter=args.init_adapter)
    baseline.main(args=args, experiment_metadata=metadata, dataset_class=dataset, trainer_class=trainer)


if __name__ == "__main__":
    main()