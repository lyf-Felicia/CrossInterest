#!/usr/bin/env python3
"""Build deterministic nested selections from the frozen 112K Amazon Train set."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file, write_json_atomic
from crossinterest.training.prompts import render_training_messages


HERE = _source_workspace('eval_amazon_datasize')
DEFAULT_SOURCE = HERE.parent / 'crossinterest.training' / "data" / "train.parquet"
DEFAULT_EVAL = HERE.parent / 'crossinterest.training' / "data" / "eval" / "dev_2k_behavior_interest.parquet"
SIZES = (5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 112_000)
EXPECTED_LABEL_COUNT = 112
EXPECTED_PER_LABEL = 1_000
SEED = 20260825


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def stable_order(seed: int, round_index: int, label: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0amazon-datasize-label-order\0{round_index}\0{label}".encode()
    ).digest()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    eval_file = args.eval_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or not eval_file.is_file():
        raise FileNotFoundError(source if not source.is_file() else eval_file)

    table = pq.read_table(source)
    rows = table.to_pylist()
    if len(rows) != SIZES[-1]:
        raise ValueError(f"expected {SIZES[-1]} source rows, found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("source Train contains duplicate IDs")
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["target_interest"]].append(row)
    counts = {label: len(label_rows) for label, label_rows in by_label.items()}
    if len(counts) != EXPECTED_LABEL_COUNT or set(counts.values()) != {EXPECTED_PER_LABEL}:
        raise ValueError("source Train is not 112 labels x 1,000 rows")

    ordered = []
    labels = sorted(by_label)
    for round_index in range(EXPECTED_PER_LABEL):
        round_labels = sorted(
            labels,
            key=lambda label: stable_order(args.seed, round_index, label),
        )
        ordered.extend(by_label[label][round_index] for label in round_labels)
    selection = pa.table(
        {
            "id": [row["id"] for row in ordered],
            "target_interest": [row["target_interest"] for row in ordered],
            "nested_rank": list(range(1, len(ordered) + 1)),
        }
    )
    selection_path = output_dir / "train_selection.parquet"
    pq.write_table(selection, selection_path, compression="zstd")

    subsets = {}
    ids = selection["id"].to_pylist()
    targets = selection["target_interest"].to_pylist()
    for size in SIZES:
        subset_counts = Counter(targets[:size])
        if len(set(ids[:size])) != size:
            raise ValueError(f"{size}: nested prefix contains duplicate IDs")
        if len(subset_counts) != EXPECTED_LABEL_COUNT:
            raise ValueError(f"{size}: nested prefix does not cover every label")
        if max(subset_counts.values()) - min(subset_counts.values()) > 1:
            raise ValueError(f"{size}: per-label counts differ by more than one")
        subsets[str(size)] = {
            "rows": size,
            "label_count": len(subset_counts),
            "count_min": min(subset_counts.values()),
            "count_max": max(subset_counts.values()),
            "id_prefix_sha256": hashlib.sha256("\n".join(ids[:size]).encode()).hexdigest(),
        }

    eval_rows = pq.read_table(eval_file).to_pylist()
    if len(eval_rows) != 2_000 or len({row["target_interest"] for row in eval_rows}) != 112:
        raise ValueError("frozen B+I evaluation set must contain 2,000 rows and 112 labels")
    preview_row = ordered[0]
    preview = {
        "source_id": preview_row["id"],
        "view": "behavior_interest",
        "label_mode": "category",
        "messages": render_training_messages(preview_row, "behavior_interest", "category"),
    }
    write_json_atomic(output_dir / "PROMPT_PREVIEW.json", preview)
    manifest = {
        "schema_version": 1,
        "task": "amazon_112_label_training_data_saturation",
        "seed": args.seed,
        "sizes": list(SIZES),
        "nested": True,
        "selection_policy": (
            "preserve each label's frozen 112K source order; emit one row per label per "
            "round; deterministically SHA-256 shuffle label order within each round"
        ),
        "source_train": {
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": len(rows),
            "label_count": len(counts),
            "count_min": min(counts.values()),
            "count_max": max(counts.values()),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "rows": len(selection),
        },
        "evaluation": {
            "view": "behavior_interest",
            "label_mode": "category",
            "path": str(eval_file),
            "sha256": sha256_file(eval_file),
            "rows": len(eval_rows),
            "label_count": len({row["target_interest"] for row in eval_rows}),
        },
        "subsets": subsets,
    }
    write_json_atomic(output_dir / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()