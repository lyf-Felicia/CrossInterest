"""Train the frozen seen/novel curriculum without reshuffling its row order."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import SequentialSampler

from crossinterest.training import train as baseline
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.curriculum.model import render_messages


class CurriculumDataset(baseline.AmazonSftDataset):
    def __init__(self, path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        extra = pq.read_table(path, columns=["task_type"])
        if len(extra) != len(self.table) or self.selection_weights is not None:
            raise ValueError("curriculum does not permit selection or weighting")
        self.table = self.table.append_column("task_type", extra["task_type"])

    def __getitem__(self, index: int) -> dict:
        row = self.table.slice(index, 1).to_pylist()[0]
        encoded = baseline.tokenize_assistant_only(self.tokenizer, render_messages(row, self.mapping))
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(f"{row['id']}: curriculum truncation is forbidden")
        return encoded


class CurriculumTrainer(baseline.AmazonSftTrainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return SequentialSampler(dataset)


def training_metadata(args, arm: str, manifest_path: Path, smoke: bool) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if arm not in manifest["arms"] or arm not in ("seen_then_novel", "shuffle"):
        raise ValueError("unsupported curriculum arm")
    if not manifest["formal_pool_frozen"] or args.init_adapter is not None:
        raise ValueError("curriculum requires frozen data and a fresh base adapter")
    require_fields(vars(args), {"view": "behavior_interest", "label_mode": "interest_id",
        "expected_label_count": 112, "epochs": 1.0, "learning_rate": 1e-4,
        "max_seq_length": 8192, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8, "expected_effective_batch_size": 32,
        "gradient_checkpointing": True, "ddp_find_unused_parameters": False,
        "loss_mode": "causal_lm", "semantic_label_token_weight": 1.0,
        "selection_file": None, "limit": None, "max_steps": -1,
        "freeze_backbone": False, "exclude_vision_lora": True,
        "require_qwen_fast_path": True}, "curriculum training settings")
    if sha256_file(args.label_map) != manifest["source"]["mapping_sha256"]:
        raise ValueError("curriculum must retain all112 semantic IDs")
    seen_labels = manifest.get("seen_labels", manifest["labels"])
    expected_rows = len(seen_labels) * 4 if smoke else manifest["files"][arm]["rows"]
    if args.expected_train_rows != expected_rows:
        raise ValueError("curriculum row budget mismatch")
    if not smoke and sha256_file(args.train_file) != manifest["files"][arm]["sha256"]:
        raise ValueError("formal curriculum order or data changed")
    rows = pq.read_table(args.train_file, columns=["id", "task_type", "target_interest"]).to_pylist()
    counts = Counter((row["task_type"], row["target_interest"]) for row in rows)
    if smoke:
        expected = {(task, label): 2 for task in ("seen", "novel") for label in seen_labels}
    elif manifest.get("full_novel"):
        expected = {(task, label): count for task, labels in manifest["task_class_counts"].items()
                    for label, count in labels.items()}
    else:
        expected = {(task, label): manifest["quota_per_class_per_task"]
                    for task in ("seen", "novel") for label in manifest["labels"]}
    if dict(counts) != expected or len({row["id"] for row in rows}) != expected_rows:
        raise ValueError("curriculum task/class quotas or unique IDs differ")
    if expected_rows % 4:
        raise ValueError("curriculum would require duplicate DDP padding")
    seen_count = sum(count for (task, label), count in expected.items() if task == "seen")
    if arm == "seen_then_novel" and [row["task_type"] for row in rows] != (
        ["seen"] * seen_count + ["novel"] * (expected_rows - seen_count)
    ):
        raise ValueError("seen-then-novel data order was changed")
    return {"task": "amazon_seen_novel_curriculum_sft",
            "prompt_schema_version": "seen-novel-bi-single-id-v1", "arm": arm,
            "smoke": smoke, "manifest_sha256": sha256_file(manifest_path),
            "training_label_count": len({row["target_interest"] for row in rows}),
            "semantic_label_count": 112,
            "sampler": "sequential_rows_sharded_by_accelerate_without_padding",
            "optimizer_reset_between_tasks": False, "scheduler_reset_between_tasks": False}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=("seen_then_novel", "shuffle"), required=True)
    parser.add_argument("--course-manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    course, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    metadata = training_metadata(args, course.arm, course.course_manifest, course.smoke)
    baseline.main(args=args, dataset_class=CurriculumDataset, experiment_metadata=metadata,
                  trainer_class=CurriculumTrainer, allow_label_subset=True)


if __name__ == "__main__":
    main()