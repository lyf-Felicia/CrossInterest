#!/usr/bin/env python3
"""Build disjoint alignment-warmup and novel-interest SFT task mixtures."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace
from crossinterest.paths import dataset_dir

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.prompts import render_training_messages


HERE = _source_workspace('eval_interest_id_minionerec')
BASE = dataset_dir('home112')
VIEWS = ("interest", "behavior_interest")
ALIGNMENT_TASKS = ("name_to_id", "id_to_name")
RECOMMENDATION_TASKS = ("history_to_id", "history_to_name")
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_sample_id", pa.string()),
        ("task_type", pa.string()),
        ("historical_titles", pa.list_(pa.string())),
        ("historical_interests", pa.list_(pa.string())),
        ("full_seen_interests", pa.list_(pa.string())),
        ("target_interest", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_dir(view: str) -> Path:
    if view not in VIEWS:
        raise ValueError(f"unsupported two-stage input view: {view}")
    return HERE if view == "interest" else HERE / view


def alignment_rows(mapping: dict) -> list[dict]:
    return [
        {
            "id": f"{task}:{interest_id}",
            "source_sample_id": "",
            "task_type": task,
            "historical_titles": [],
            "historical_interests": [],
            "full_seen_interests": [],
            "target_interest": category,
        }
        for category, interest_id in mapping["category_to_id"].items()
        for task in ALIGNMENT_TASKS
    ]


def recommendation_rows(row: dict, mapping: dict) -> list[dict]:
    target = row["target_interest"]
    if target not in mapping["category_to_id"]:
        raise ValueError(f"unknown target: {target}")
    if target in set(row["full_seen_interests"]):
        raise ValueError(f"{row['id']}: recommendation target is not novel")
    if not row["historical_interests"]:
        raise ValueError(f"{row['id']}: empty history")
    return [
        {
            "id": f"{row['id']}:{task}",
            "source_sample_id": row["id"],
            "task_type": task,
            "historical_titles": row["historical_titles"],
            "historical_interests": row["historical_interests"],
            "full_seen_interests": row["full_seen_interests"],
            "target_interest": target,
        }
        for task in RECOMMENDATION_TASKS
    ]


def render_messages(row: dict, mapping: dict, view: str = "interest") -> list[dict[str, str]]:
    if view not in VIEWS:
        raise ValueError(f"unsupported two-stage input view: {view}")
    task = row["task_type"]
    category = row["target_interest"]
    interest_id = mapping["category_to_id"][category]
    if task in ALIGNMENT_TASKS:
        if row["source_sample_id"] or any(
            row[field]
            for field in ("historical_titles", "historical_interests", "full_seen_interests")
        ):
            raise ValueError("alignment may not contain user histories or future targets")
        if task == "name_to_id":
            prompt = (
                "Which Interest ID represents the following category?\n\n"
                f"Category: {category}\n\n"
                "Respond with only the single-token Interest ID in the exact format "
                "<interest_NNN>. Do not output any explanation."
            )
            response = interest_id
        else:
            prompt = (
                "What is the exact category name represented by the following Interest ID?\n\n"
                f"Interest ID: {interest_id}\n\n"
                "Respond with only the exact category name. Do not output any explanation."
            )
            response = category
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    if task in RECOMMENDATION_TASKS:
        mode = "interest_id" if task == "history_to_id" else "category"
        return render_training_messages(row, view, mode, mapping)
    raise ValueError(f"unsupported task: {task}")


def build(train: Path, mapping_path: Path, output: Path) -> dict:
    mapping = load_label_mapping(mapping_path, expected_count=112)
    source = pq.ParquetFile(train)
    if source.metadata.num_rows != 112_000:
        raise ValueError("expected the frozen 112,000-row Home training set")
    output.mkdir(parents=True, exist_ok=True)
    warmup = alignment_rows(mapping)
    pq.write_table(pa.Table.from_pylist(warmup, schema=SCHEMA), output / "warmup.parquet")
    labels: Counter[str] = Counter()
    sample_ids: set[str] = set()
    preview = {row["task_type"]: render_messages(row, mapping) for row in warmup[:2]}
    source_columns = ["id", "historical_titles", "historical_interests", "full_seen_interests", "target_interest"]
    with pq.ParquetWriter(output / "recommendation.parquet", SCHEMA) as writer:
        for batch in source.iter_batches(batch_size=2048, columns=source_columns):
            records = []
            for row in batch.to_pylist():
                if row["id"] in sample_ids:
                    raise ValueError(f"duplicate source ID: {row['id']}")
                sample_ids.add(row["id"])
                labels[row["target_interest"]] += 1
                tasks = recommendation_rows(row, mapping)
                for task in tasks:
                    if task["task_type"] not in preview:
                        preview[task["task_type"]] = render_messages(task, mapping)
                records.extend(tasks)
            writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
    if set(labels) != set(mapping["category_to_id"]) or set(labels.values()) != {1000}:
        raise ValueError("source training labels or per-label quota do not match Home 112")
    manifest = {
        "schema_version": 1,
        "protocol": "home112-minionerec-alignment-then-novel-interest-v1",
        "seed": 20260825,
        "train_source": str(train.resolve()),
        "train_sha256": sha256_file(train),
        "mapping_sha256": sha256_file(mapping_path),
        "mapping_fingerprint": mapping["mapping_fingerprint"],
        "label_count": 112,
        "view": "interest",
        "warmup": {
            "rows": len(warmup),
            "task_counts": dict(Counter(row["task_type"] for row in warmup)),
            "epochs": 10,
            "user_history_used": False,
            "future_target_used": False,
            "sha256": sha256_file(output / "warmup.parquet"),
        },
        "recommendation": {
            "rows": len(sample_ids) * 2,
            "unique_source_samples": len(sample_ids),
            "task_counts": {task: len(sample_ids) for task in RECOMMENDATION_TASKS},
            "epochs": 1,
            "target_absent_from_full_history": True,
            "sha256": sha256_file(output / "recommendation.parquet"),
        },
        "evaluation_data_used_in_training": False,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SAMPLE_PREVIEW.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_behavior_interest(output: Path) -> dict:
    source = HERE / "data"
    if output.resolve() == source.resolve():
        raise ValueError("Behavior+Interest must not overwrite the Interest data contract")
    manifest = json.loads((source / "MANIFEST.json").read_text())
    if manifest["view"] != "interest":
        raise ValueError("expected the frozen Interest source contract")
    mapping_path = BASE / "interest_id_semantic_mapping.json"
    if sha256_file(BASE / "train.parquet") != manifest["train_sha256"] or sha256_file(mapping_path) != manifest["mapping_sha256"]:
        raise ValueError("frozen training data or label mapping changed")
    mapping = load_label_mapping(mapping_path, expected_count=112)
    preview = {}
    for stage in ("warmup", "recommendation"):
        path = source / f"{stage}.parquet"
        if sha256_file(path) != manifest[stage]["sha256"]:
            raise ValueError(f"frozen {stage} dataset changed")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if stage == "recommendation" and (
                    not row["historical_titles"]
                    or len(row["historical_titles"]) != len(row["historical_interests"])
                    or any(not title for title in row["historical_titles"])
                ):
                    raise ValueError(f"{row['id']}: missing or unpaired behavior history")
                if row["task_type"] not in preview:
                    preview[row["task_type"]] = render_messages(row, mapping, "behavior_interest")
    manifest["view"] = "behavior_interest"
    output.mkdir(parents=True, exist_ok=True)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SAMPLE_PREVIEW.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=BASE / "train.parquet")
    parser.add_argument("--mapping", type=Path, default=BASE / "interest_id_semantic_mapping.json")
    parser.add_argument("--view", choices=VIEWS, default="interest")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or experiment_dir(args.view) / "data"
    if args.view == "behavior_interest":
        if args.train != BASE / "train.parquet" or args.mapping != BASE / "interest_id_semantic_mapping.json":
            raise ValueError("Behavior+Interest must reuse the frozen source data and mapping")
        manifest = prepare_behavior_interest(output)
    else:
        manifest = build(args.train, args.mapping, output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()