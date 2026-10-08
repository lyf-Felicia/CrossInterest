#!/usr/bin/env python3
"""Build the independent quota-1000, 112-label Amazon SFT dataset."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file, write_json_atomic
from crossinterest.datasets.amazon.scripts.build import DEFAULT_SEED, JsonlWriter, build_balanced_train
from crossinterest.datasets.amazon.scripts.export_parquet import convert


HERE = _source_workspace('eval_amazon_sft_112')
DEFAULT_MOTHER_POOL = HERE.parent / "prep" / "artifacts" / "home_and_kitchen" / "data" / "unbalanced"
DEFAULT_PREP_MANIFEST = HERE.parent / "prep" / "artifacts" / "home_and_kitchen" / "MANIFEST.json"
EXPECTED_LABEL_COUNT = 112
TRAIN_QUOTA = 1_000
EXPECTED_TRAIN_ROWS = EXPECTED_LABEL_COUNT * TRAIN_QUOTA
EXPECTED_DEV_ROWS = 596_449
EVAL_SIZE = 2_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mother-pool", type=Path, default=DEFAULT_MOTHER_POOL)
    parser.add_argument("--prep-manifest", type=Path, default=DEFAULT_PREP_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_closed_dev(source: Path, destination: Path, support: set[str]) -> dict:
    writer = JsonlWriter(destination)
    labels = Counter()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = record["target_interest"]
            if label in support:
                writer.write(record)
                labels[label] += 1
    writer.close()
    if set(labels) != support:
        raise ValueError("closed-set Dev does not cover every Train label")
    return {
        "sample_count": writer.count,
        "label_count": len(labels),
        "count_min": min(labels.values()),
        "count_max": max(labels.values()),
        "resampled": False,
    }


def main() -> None:
    args = parse_args()
    mother_pool = args.mother_pool.resolve()
    output_dir = args.output_dir.resolve()
    prep_manifest = args.prep_manifest.resolve()
    unbalanced_train = mother_pool / "train.jsonl"
    unbalanced_dev = mother_pool / "dev.jsonl"
    for path in (unbalanced_train, unbalanced_dev, prep_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="amazon-sft-112-") as temporary:
        stage = Path(temporary)
        balance = build_balanced_train(
            unbalanced_train,
            stage / "train.jsonl",
            args.seed,
            min_label_support=TRAIN_QUOTA,
            requested_quota=TRAIN_QUOTA,
        )
        if balance["support_size"] != EXPECTED_LABEL_COUNT:
            raise ValueError(
                f"expected {EXPECTED_LABEL_COUNT} retained labels, "
                f"found {balance['support_size']}"
            )
        if balance["sample_count"] != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"expected {EXPECTED_TRAIN_ROWS} Train rows, "
                f"found {balance['sample_count']}"
            )
        support = set(balance["support"])
        dev_stats = build_closed_dev(
            unbalanced_dev,
            stage / "dev.jsonl",
            support,
        )
        if dev_stats["sample_count"] != EXPECTED_DEV_ROWS:
            raise ValueError(
                f"expected {EXPECTED_DEV_ROWS} closed-set Dev rows, "
                f"found {dev_stats['sample_count']}"
            )
        train_export = convert(stage / "train.jsonl", output_dir / "train.parquet", 10_000)
        dev_export = convert(stage / "dev.jsonl", output_dir / "dev.parquet", 10_000)

    eval_dir = output_dir / "eval"
    subprocess.run(
        [
            sys.executable,
            "-m",
            'crossinterest.evaluation.build_views',
            "--train",
            str(output_dir / "train.parquet"),
            "--dev",
            str(output_dir / "dev.parquet"),
            "--output-dir",
            str(eval_dir),
            "--size",
            str(EVAL_SIZE),
            "--seed",
            str(args.seed),
            "--expected-label-count",
            str(EXPECTED_LABEL_COUNT),
        ],
        check=True,
    )

    train_labels = Counter(
        pq.read_table(output_dir / "train.parquet", columns=["target_interest"])[
            "target_interest"
        ].to_pylist()
    )
    dev_labels = Counter(
        pq.read_table(output_dir / "dev.parquet", columns=["target_interest"])[
            "target_interest"
        ].to_pylist()
    )
    if set(train_labels.values()) != {TRAIN_QUOTA}:
        raise ValueError("exported Train quota mismatch")
    if set(train_labels) != set(dev_labels):
        raise ValueError("exported Train and Dev support mismatch")

    manifest = {
        "schema_version": 1,
        "task": "amazon_new_interest_quota_balanced_dataset",
        "seed": args.seed,
        "policy": (
            "retain every mother-pool Train target with support >= quota; "
            "sample exactly quota records per retained target by stable SHA-256 order"
        ),
        "quota": TRAIN_QUOTA,
        "label_count": EXPECTED_LABEL_COUNT,
        "source": {
            "mother_pool": str(mother_pool),
            "prep_manifest": str(prep_manifest),
            "prep_manifest_sha256": sha256_file(prep_manifest),
            "train_bytes": unbalanced_train.stat().st_size,
            "dev_bytes": unbalanced_dev.stat().st_size,
        },
        "balance": balance,
        "train": {
            **train_export,
            "path": str(output_dir / "train.parquet"),
            "sha256": sha256_file(output_dir / "train.parquet"),
            "label_count": len(train_labels),
            "count_min": min(train_labels.values()),
            "count_max": max(train_labels.values()),
        },
        "dev": {
            **dev_export,
            **dev_stats,
            "path": str(output_dir / "dev.parquet"),
            "sha256": sha256_file(output_dir / "dev.parquet"),
        },
        "evaluation": json.loads((eval_dir / "MANIFEST.json").read_text(encoding="utf-8")),
    }
    write_json_atomic(output_dir / "DATASET_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
