"""Run the reconstruction or full novel stage without modifying the baseline trainer."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.prompts import render_history, render_training_messages
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.model import RelationController, history_frame


ARMS = ("structure", "reconstruction", "combined")


def auxiliary_messages(row: dict, mapping: dict) -> list[dict]:
    return [{"role": "user", "content": (
        "The following product history has had all entries from one topic category removed.\n\n"
        f"{render_history(row, 'behavior_interest')}\n\n"
        "Infer the removed topic category from the remaining observed interests and products. "
        "This is historical interest reconstruction, not a prediction of the next purchase. "
        "Respond with only its single-token Interest ID in the exact format <interest_NNN>. "
        "Do not output the category name or any explanation."
    )}, {"role": "assistant", "content": mapping["category_to_id"][row["target_interest"]]}]


def positive_set_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != positive_mask.shape or not positive_mask.any(dim=-1).all():
        raise ValueError("each auxiliary row must have a nonempty aligned positive set")
    distribution = positive_mask.float() / positive_mask.sum(dim=-1, keepdim=True)
    return -(distribution * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


class MechanismDataset(baseline.AmazonSftDataset):
    def __init__(self, *args, stage: str, relation: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        self.relation = relation
        if stage == "auxiliary":
            positive = pq.read_table(args[0], columns=["positive_interests"])
            self.table = self.table.append_column("positive_interests", positive["positive_interests"])
        self.category_order = [self.mapping["id_to_category"][token] for token in self.mapping["special_tokens"]]

    def __getitem__(self, index):
        row = self.table.slice(index, 1).to_pylist()[0]
        messages = (auxiliary_messages(row, self.mapping) if self.stage == "auxiliary" else
                    render_training_messages(row, "behavior_interest", "interest_id", self.mapping))
        encoded = baseline.tokenize_assistant_only(self.tokenizer, messages)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(f"{row['id']}: truncation is forbidden")
        position = next(index for index, label in enumerate(encoded["labels"]) if label != -100) - 1
        if self.relation:
            encoded["history_frame"] = history_frame(self.tokenizer, encoded["input_ids"], row, position)
        if self.stage == "auxiliary":
            encoded["positive_mask"] = [category in row["positive_interests"] for category in self.category_order]
        return encoded


class MechanismCollator(baseline.AssistantOnlyCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        if "history_frame" in features[0]:
            batch["history_frames"] = [feature["history_frame"] for feature in features]
        if "positive_mask" in features[0]:
            batch["positive_mask"] = torch.tensor([feature["positive_mask"] for feature in features], dtype=torch.bool)
        return batch


class MechanismTrainer(baseline.AmazonSftTrainer):
    def __init__(self, *args, stage: str, relation: bool, init_adapter: Path | None, **kwargs):
        kwargs["data_collator"] = MechanismCollator(kwargs["processing_class"].pad_token_id)
        super().__init__(*args, **kwargs)
        self.stage = stage
        if stage == "auxiliary":
            self.model_accepts_loss_kwargs = False
        self.controller = (RelationController(self.model, self.processing_class, self.train_dataset.mapping,
                                              directory=init_adapter) if relation else None)
        if self.controller is not None and self.is_world_process_zero():
            initial = Path(self.args.output_dir) / "initial_relation"
            if not (initial / "interest_relation.safetensors").exists():
                self.controller.save(initial)
        if self.is_world_process_zero():
            write_json_atomic(Path(self.args.output_dir) / "MECHANISM_RUNTIME.json", {
                "stage": stage, "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs,
                "auxiliary_mean_loss_scaled_by_trainer": stage == "auxiliary" and not self.model_accepts_loss_kwargs,
                "relation_trainable_parameters": sum(parameter.numel() for parameter in self.controller.head.parameters()) if self.controller else 0,
                "total_trainable_parameters": sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
            })

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        frames = inputs.pop("history_frames", None)
        if self.controller is not None:
            self.controller.set_frames(frames)
        if self.stage == "novel":
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        positives = inputs.pop("positive_mask")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        positions = (labels != -100).long().argmax(dim=-1) - 1
        target_logits = outputs.logits[torch.arange(labels.shape[0], device=labels.device), positions]
        loss = positive_set_loss(target_logits[:, self.label_token_ids], positives)
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.controller is not None:
            self.controller.save(Path(output_dir or self.args.output_dir))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.controller is not None:
            self.controller.load(Path(resume_from_checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=("auxiliary", "novel"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--mechanism-preflight", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    prepared = json.loads(options.mechanism_preflight.read_text())
    relation = options.arm != "reconstruction"
    stage_key = ("smoke_" if options.smoke else "") + options.stage
    expected = prepared["files"][stage_key]
    if args.view != "behavior_interest" or args.label_mode != "interest_id" or args.loss_mode != "causal_lm":
        raise ValueError("mechanism experiment requires the frozen B+I causal protocol")
    if args.selection_file is not None or args.limit is not None or args.max_steps != -1:
        raise ValueError("all stages use their complete frozen files")
    if args.epochs != 1 or args.learning_rate != 1e-4 or args.expected_effective_batch_size != 32:
        raise ValueError("stage training budget differs from the frozen contract")
    if args.expected_train_rows != expected["rows"] or sha256_file(args.train_file) != expected["sha256"]:
        raise ValueError("stage data contract mismatch")
    if sha256_file(args.label_map) != prepared["mapping_sha256"]:
        raise ValueError("label mapping mismatch")
    if options.stage == "auxiliary" and (options.arm == "structure" or args.init_adapter is not None):
        raise ValueError("auxiliary stage must start fresh and belong to an auxiliary arm")
    if (options.stage == "novel" and options.arm != "structure") != (args.init_adapter is not None):
        raise ValueError("novel stage warm-start does not match the arm")
    if args.init_adapter is not None:
        previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())
        if (previous["global_step"] != prepared["files"][("smoke_" if options.smoke else "") + "auxiliary"]["steps"]
                or previous["experiment_metadata"]["arm"] != options.arm
                or previous["experiment_metadata"]["smoke"] != options.smoke):
            raise ValueError("auxiliary handoff is incomplete or belongs to another arm")
    metadata = {"task": "amazon_interest_mechanism_sft", "prompt_schema_version": "interest-mechanism-v1",
                "arm": options.arm, "stage": options.stage, "smoke": options.smoke, "relation_head": relation,
                "preflight_sha256": sha256_file(options.mechanism_preflight),
                "auxiliary_objective": "uniform_positive_set_ce_112" if options.stage == "auxiliary" else None,
                "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True,
                "relation_dimension": 64 if relation else None}
    baseline.main(args=args, experiment_metadata=metadata,
        dataset_class=partial(MechanismDataset, stage=options.stage, relation=relation),
        trainer_class=partial(MechanismTrainer, stage=options.stage, relation=relation, init_adapter=args.init_adapter))


if __name__ == "__main__":
    main()