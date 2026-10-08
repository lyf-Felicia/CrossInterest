#!/usr/bin/env python3
"""Run one audited stage with the existing semantic Interest-ID SFT trainer."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.training import train as baseline
from crossinterest.methods.alignment.data import ALIGNMENT_TASKS, RECOMMENDATION_TASKS, VIEWS, render_messages, sha256_file


class TaskDataset(baseline.AmazonSftDataset):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        extra = pq.read_table(path, columns=["task_type", "source_sample_id"])
        if len(extra) != len(self.table):
            raise ValueError("task dataset does not support selection filtering")
        for name in extra.column_names:
            self.table = self.table.append_column(name, extra[name])

    def __getitem__(self, index: int) -> dict:
        row = self.table.slice(index, 1).to_pylist()[0]
        encoded = baseline.tokenize_assistant_only(
            self.tokenizer, render_messages(row, self.mapping, self.view)
        )
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(f"{row['id']}: sequence exceeds {self.max_length}; truncation forbidden")
        return encoded


def stage_metadata(args, stage: str, manifest_path: Path, smoke: bool) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected = manifest[stage]
    if args.view not in VIEWS or args.view != manifest["view"] or args.label_mode != "interest_id":
        raise ValueError("input view must match the manifest and use semantic Interest IDs")
    if args.loss_mode != "causal_lm" or args.semantic_label_token_weight != 1:
        raise ValueError("mixed tasks require unweighted assistant-only causal LM loss")
    if args.selection_file is not None or args.freeze_backbone:
        raise ValueError("selection and backbone freezing are outside this protocol")
    if args.expected_train_rows != expected["rows"] or args.epochs != expected["epochs"]:
        raise ValueError("stage row count or epoch budget differs from manifest")
    if args.expected_label_count != 112 or args.learning_rate != 1e-4:
        raise ValueError("label count or learning rate differs from the frozen protocol")
    if args.expected_effective_batch_size != 32:
        raise ValueError("effective batch must remain 32")
    if not smoke and (args.limit is not None or args.max_steps != -1):
        raise ValueError("formal stages cannot limit samples or optimizer steps")
    if smoke and (args.max_steps <= 0 or args.max_steps > 10):
        raise ValueError("smoke requires a positive limit of at most ten steps")
    if sha256_file(args.train_file) != expected["sha256"]:
        raise ValueError("stage dataset SHA mismatch")
    if sha256_file(args.label_map) != manifest["mapping_sha256"]:
        raise ValueError("label mapping SHA mismatch")
    tasks = Counter(pq.read_table(args.train_file, columns=["task_type"])["task_type"].to_pylist())
    allowed = ALIGNMENT_TASKS if stage == "warmup" else RECOMMENDATION_TASKS
    if set(tasks) != set(allowed) or dict(tasks) != expected["task_counts"]:
        raise ValueError("stage tasks or mixture counts differ from manifest")
    handoff = None
    if stage == "warmup":
        if args.init_adapter is not None:
            raise ValueError("warmup must start from the base model")
    else:
        if args.init_adapter is None:
            raise ValueError("recommendation must start from the completed warmup adapter")
        previous = args.init_adapter.parent
        contract = json.loads((previous / "train_contract.json").read_text())
        metrics = json.loads((previous / "train_metrics.json").read_text())
        for key, expected_value in {
            "model": args.model,
            "view": args.view,
            "label_mode": args.label_mode,
            "label_map_sha256": manifest["mapping_sha256"],
            "train_sha256": manifest["warmup"]["sha256"],
        }.items():
            if contract.get(key) != expected_value or metrics.get(key) != expected_value:
                raise ValueError(f"warmup handoff mismatch: {key}")
        previous_metadata = contract["experiment_metadata"]
        if previous_metadata["stage"] != "warmup" or previous_metadata["smoke"] != smoke:
            raise ValueError("warmup handoff has wrong stage or smoke/formal identity")
        if metrics.get("experiment_metadata") != previous_metadata:
            raise ValueError("warmup metrics and contract metadata differ")
        if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
            raise ValueError("warmup loss must be finite and positive")
        if not smoke and metrics.get("epoch", 0) < manifest["warmup"]["epochs"]:
            raise ValueError("warmup has not completed its epoch budget")
        handoff = {
            "train_contract_sha256": sha256_file(previous / "train_contract.json"),
            "adapter_model_sha256": sha256_file(args.init_adapter / "adapter_model.safetensors"),
            "optimizer_reset": True,
            "scheduler_reset": True,
        }
    return {
        "task": f"amazon_minionerec_{stage}",
        "prompt_schema_version": f"minionerec-two-stage-{stage}-v1",
        "stage": stage,
        "smoke": smoke,
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "manifest_sha256": sha256_file(manifest_path),
        "task_counts": dict(tasks),
        "handoff": handoff,
    }


def audit_stage(args, metadata: dict) -> dict:
    from safetensors import safe_open

    output = args.output_dir
    contract = json.loads((output / "train_contract.json").read_text())
    metrics = json.loads((output / "train_metrics.json").read_text())
    for record in (contract, metrics):
        if record.get("experiment_metadata") != metadata:
            raise ValueError("completed stage metadata differs from requested stage")
        if record.get("model") != args.model or record.get("train_sha256") != sha256_file(args.train_file):
            raise ValueError("completed stage model or dataset differs")
        if record.get("label_map_sha256") != sha256_file(args.label_map):
            raise ValueError("completed stage mapping differs")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("training loss must be finite and positive")
    if not metadata["smoke"] and metrics.get("epoch", 0) < args.epochs:
        raise ValueError("stage did not complete its epoch budget")
    adapter = output / "final_adapter"
    weights = adapter / "adapter_model.safetensors"
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise ValueError("saved LoRA configuration differs from protocol")
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        token_keys = [key for key in keys if "trainable_tokens" in key]
        output_keys = [key for key in keys if "lm_head" in key and "lora_B" in key]
        if not token_keys or not output_keys:
            raise ValueError("adapter lacks sparse semantic embeddings or lm_head LoRA")
        for key in token_keys + output_keys:
            tensor = tensors.get_tensor(key)
            if not tensor.isfinite().all() or not tensor.count_nonzero():
                raise ValueError(f"nonfinite or zero trained tensor: {key}")
    digest = sha256_file(weights)
    if metadata["handoff"] and digest == metadata["handoff"]["adapter_model_sha256"]:
        raise ValueError("recommendation adapter did not change after warmup")
    audit = {
        "stage": metadata["stage"], "smoke": metadata["smoke"],
        "adapter_model_sha256": digest, "epoch": metrics["epoch"],
        "train_loss": metrics["train_loss"], "sparse_token_keys": token_keys,
        "output_lora_keys": output_keys, "passed": True,
    }
    baseline.write_json(output / "stage_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("warmup", "recommendation"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    stage_args, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    metadata = stage_metadata(args, stage_args.stage, stage_args.manifest, stage_args.smoke)
    if stage_args.validate_only:
        print(json.dumps(audit_stage(args, metadata), indent=2))
    else:
        baseline.main(args=args, dataset_class=TaskDataset, experiment_metadata=metadata)


if __name__ == "__main__":
    main()