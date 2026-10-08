#!/usr/bin/env python3
"""Convert the frozen Home benchmark to the official MeRA data contract."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from crossinterest.baselines.mera.core import build_cooccurrence_matrix, write_json
from crossinterest.baselines.mera.data import stable_digest


HERE = _source_workspace('eval_baseline_mera')
DATA_ROOT = HERE.parent / 'crossinterest.training' / "data"
INPUT_PREFIX = (
    "The user has given high ratings to the following categories of home goods before: "
)
INSTRUCTION = (
    "Given a list of categories of home goods the user has bought before, please "
    "predict what new category of home goods this user will be interested in in the "
    "future. The predicted categorie should be different from the history categories."
)
VALIDATION_SEED = 20260902
VALIDATION_ROWS = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_name(label: str) -> str:
    parent, separator, child = label.rpartition(" > ")
    if not separator or not parent or not child:
        raise ValueError(f"Home label lacks a parent/child path: {label!r}")
    return child


def user_from_source_id(source_id: str) -> str:
    parts = source_id.split(":", 2)
    if len(parts) != 3 or parts[0] != "dev":
        raise ValueError(f"unrecognized Home Dev source ID: {source_id!r}")
    return parts[1]


def select_validation_rows(
    dev_rows: list[dict],
    labels: set[str],
    prediction_labels: list[str],
    blocked_users: set[str],
) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in dev_rows:
        if (
            row["user_id"] not in blocked_users
            and set(row["historical_interests"]) <= labels
            and row["target_interest"] in prediction_labels
        ):
            by_label[row["target_interest"]].append(row)
    base, remainder = divmod(VALIDATION_ROWS, len(prediction_labels))
    selected = []
    for label_index, label in enumerate(prediction_labels):
        quota = base + int(label_index < remainder)
        candidates = sorted(
            by_label[label],
            key=lambda row: stable_digest(
                VALIDATION_SEED, "official-home-validation", row["id"]
            ),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"not enough validation rows for {label!r}: {len(candidates)} < {quota}"
            )
        selected.extend(candidates[:quota])
    selected.sort(
        key=lambda row: stable_digest(
            VALIDATION_SEED, "official-home-validation-order", row["id"]
        )
    )
    return selected


def to_official_record(row: dict, full_to_child: dict[str, str]) -> dict[str, str]:
    history = [full_to_child[label] for label in row["historical_interests"]]
    target = full_to_child[row["target_interest"]]
    history_text = ", ".join(json.dumps(label, ensure_ascii=False) for label in history)
    return {
        "instruction": INSTRUCTION,
        "input": f"{INPUT_PREFIX}{history_text}\n",
        "output": json.dumps(target, ensure_ascii=False),
    }


def write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_ROOT / "train.parquet")
    parser.add_argument("--dev", type=Path, default=DATA_ROOT / "dev.parquet")
    parser.add_argument(
        "--test",
        type=Path,
        default=DATA_ROOT / "eval" / "dev_2k_interest.parquet",
    )
    parser.add_argument(
        "--label-space", type=Path, default=HERE / "data" / "home" / "label_space.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    dev_path = args.dev.resolve()
    test_path = args.test.resolve()
    label_space_path = args.label_space.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_space = json.loads(label_space_path.read_text(encoding="utf-8"))
    labels = list(label_space["labels"])
    prediction_labels = list(label_space["prediction_labels"])
    full_to_child = {label: child_name(label) for label in labels}
    duplicate_children = [
        child for child, count in Counter(full_to_child.values()).items() if count > 1
    ]
    if duplicate_children:
        raise ValueError(f"Home child names are not unique: {duplicate_children}")
    child_to_full = {child: full for full, child in full_to_child.items()}

    train_columns = [
        "id",
        "user_id",
        "historical_interests",
        "full_seen_interests",
        "target_interest",
    ]
    train_rows = pq.read_table(train_path, columns=train_columns).to_pylist()
    dev_rows = pq.read_table(dev_path, columns=train_columns).to_pylist()
    test_columns = [
        "sample_id",
        "source_sample_id",
        "historical_interests",
        "full_seen_interests",
        "target_interest",
        "candidate_interests",
        "gold_index",
    ]
    test_rows = pq.read_table(test_path, columns=test_columns).to_pylist()
    blocked_users = {user_from_source_id(row["source_sample_id"]) for row in test_rows}
    validation_rows = select_validation_rows(
        dev_rows, set(labels), prediction_labels, blocked_users
    )

    label_set = set(labels)
    prediction_set = set(prediction_labels)
    for split, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        for row in rows:
            visible = set(row["historical_interests"])
            target = row["target_interest"]
            if not visible <= label_set or target not in prediction_set:
                raise ValueError(f"{split} row is outside the frozen label spaces")
            if target in set(row["full_seen_interests"]):
                raise ValueError(f"{split} row violates novel-interest supervision")

    if len(train_rows) != 112_000 or len(validation_rows) != VALIDATION_ROWS:
        raise ValueError("Home Train/validation row count contract failed")
    if len(test_rows) != 2_000 or len({row["sample_id"] for row in test_rows}) != 2_000:
        raise ValueError("Home formal evaluation sample contract failed")
    validation_users = {row["user_id"] for row in validation_rows}
    if validation_users & blocked_users:
        raise ValueError("Home validation users overlap the formal Dev-2K users")

    train_records = [to_official_record(row, full_to_child) for row in train_rows]
    validation_records = [
        to_official_record(row, full_to_child) for row in validation_rows
    ]
    test_records = [to_official_record(row, full_to_child) for row in test_rows]
    for split, records in (
        ("train", train_records),
        ("validation", validation_records),
        ("test", test_records),
    ):
        for index, record in enumerate(records):
            parsed = ast.literal_eval(record["input"][81:].strip("\n"))
            parsed_categories = [parsed] if isinstance(parsed, str) else list(parsed)
            if not parsed_categories or not set(parsed_categories) <= set(child_to_full):
                raise ValueError(
                    f"{split}:{index} is incompatible with official inference parsing"
                )
    write_records(output_dir / "train.json", train_records)
    write_records(output_dir / "valid.json", validation_records)
    write_records(output_dir / "test.json", test_records)

    category_tree = {
        parent: [full_to_child[label] for label in children]
        for parent, children in label_space["category_tree"].items()
    }
    write_json(output_dir / "category_tree.json", category_tree)
    (output_dir / "layer2_map.txt").write_text(
        "".join(f"{full_to_child[label]}\t{index}\n" for index, label in enumerate(labels)),
        encoding="utf-8",
    )
    cooccurrence = build_cooccurrence_matrix(
        (
            set(row["historical_interests"]) | {row["target_interest"]}
            for row in train_rows
        ),
        labels,
    ).astype(np.float32)
    np.save(output_dir / "category_cooccurrence.npy", cooccurrence)

    test_metadata = [
        {
            "sample_id": row["sample_id"],
            "source_sample_id": row["source_sample_id"],
            "target": row["target_interest"],
            "candidates": row["candidate_interests"],
            "gold_index": row["gold_index"],
        }
        for row in test_rows
    ]
    write_json(output_dir / "test_metadata.json", test_metadata)
    write_json(
        output_dir / "label_mapping.json",
        {
            "labels": labels,
            "prediction_labels": prediction_labels,
            "full_to_official_category": full_to_child,
            "official_category_to_full": child_to_full,
        },
    )

    generated = [
        "train.json",
        "valid.json",
        "test.json",
        "category_tree.json",
        "layer2_map.txt",
        "category_cooccurrence.npy",
        "test_metadata.json",
        "label_mapping.json",
    ]
    manifest = {
        "schema_version": 1,
        "dataset": "home",
        "conversion": "frozen Home Interest view to official MeRA JSON contract",
        "official_input_prefix_length": len(INPUT_PREFIX),
        "official_inference_slice_offset": 81,
        "official_inference_parse_roundtrip": True,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "context_label_count": len(labels),
        "prediction_label_count": len(prediction_labels),
        "validation_seed": VALIDATION_SEED,
        "validation_policy": (
            "stratified Dev sample after excluding every formal Dev-2K user"
        ),
        "validation_user_count": len(validation_users),
        "formal_test_user_count": len(blocked_users),
        "validation_test_user_overlap": 0,
        "cooccurrence_dtype": str(cooccurrence.dtype),
        "cooccurrence_source": (
            "unique visible official-JSON training history plus target; Train only"
        ),
        "source_files": {
            str(path): sha256_file(path)
            for path in (train_path, dev_path, test_path, label_space_path)
        },
        "generated_files": {
            name: sha256_file(output_dir / name) for name in generated
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()