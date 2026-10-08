#!/usr/bin/env python3
"""Build exact-quota nested selections from the frozen 99-label Amazon Train set."""

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
from crossinterest.legacy.training_99.prompts import render_training_messages


HERE = _source_workspace('eval_amazon_datasize_99')
DEFAULT_SOURCE = HERE.parent / "train.parquet"
DEFAULT_EVAL = HERE.parent / 'crossinterest.evaluation' / "data" / "dev_2k_behavior_interest.parquet"
SIZES = (
    10_000,
    20_000,
    40_000,
    80_000,
    120_000,
    160_000,
    200_000,
    240_000,
    280_000,
    320_000,
    360_000,
    396_000,
)
EXPECTED_LABEL_COUNT = 99
EXPECTED_PER_LABEL = 4000
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
        f"{seed}\0amazon-datasize-99-label-order\0{round_index}\0{label}".encode()
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
    if len(rows) != SIZES[-1] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("source Train must contain 396,000 unique rows")
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["target_interest"]].append(row)
    counts = Counter({label: len(label_rows) for label, label_rows in by_label.items()})
    if len(counts) != EXPECTED_LABEL_COUNT or set(counts.values()) != {EXPECTED_PER_LABEL}:
        raise ValueError("source Train is not 99 labels x 4,000 rows")

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

    ids = selection["id"].to_pylist()
    targets = selection["target_interest"].to_pylist()
    subsets = {}
    for size in SIZES:
        subset_counts = Counter(targets[:size])
        quotient, remainder = divmod(size, EXPECTED_LABEL_COUNT)
        if len(set(ids[:size])) != size:
            raise ValueError(f"{size}: selection prefix contains duplicate IDs")
        if len(subset_counts) != EXPECTED_LABEL_COUNT:
            raise ValueError(f"{size}: selection prefix does not cover all 99 labels")
        if min(subset_counts.values()) != quotient:
            raise ValueError(f"{size}: minimum class count is not {quotient}")
        if max(subset_counts.values()) != quotient + bool(remainder):
            raise ValueError(f"{size}: maximum class count is invalid")
        if sum(value == quotient + 1 for value in subset_counts.values()) != remainder:
            raise ValueError(f"{size}: quotient/remainder allocation is invalid")
        if size % 4:
            raise ValueError(f"{size}: four-rank distributed sampler would pad")
        subsets[str(size)] = {
            "rows": size,
            "label_count": EXPECTED_LABEL_COUNT,
            "count_min": quotient,
            "count_max": quotient + bool(remainder),
            "remainder_classes": remainder,
            "optimizer_steps": (size + 31) // 32,
            "partial_final_optimizer_step": bool(size % 32),
            "id_prefix_sha256": hashlib.sha256("\n".join(ids[:size]).encode()).hexdigest(),
        }

    eval_rows = pq.read_table(eval_file).to_pylist()
    if len(eval_rows) != 2000 or len({row["target_interest"] for row in eval_rows}) != 99:
        raise ValueError("frozen B+I evaluation set must contain 2,000 rows and 99 labels")
    write_json_atomic(
        output_dir / "PROMPT_PREVIEW.json",
        {
            "source_id": ordered[0]["id"],
            "view": "behavior_interest",
            "label_mode": "category",
            "messages": render_training_messages(ordered[0], "behavior_interest", "category"),
        },
    )
    manifest = {
        "schema_version": 1,
        "task": "amazon_99_label_training_data_saturation",
        "seed": args.seed,
        "sizes": list(SIZES),
        "new_training_sizes": list(SIZES[:-1]),
        "nested": True,
        "selection_policy": (
            "preserve each label's frozen 396K source order; emit one row per label per "
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
