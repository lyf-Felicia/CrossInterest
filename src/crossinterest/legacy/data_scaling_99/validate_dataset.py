#!/usr/bin/env python3
"""Validate exact-quota nested selections for the 99-label saturation curve."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file
from crossinterest.legacy.training_99.train import load_selection_ids, training_label_counts

from crossinterest.legacy.data_scaling_99.build_dataset import DEFAULT_SOURCE, EXPECTED_LABEL_COUNT, SIZES


HERE = _source_workspace('eval_amazon_datasize_99')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    source = args.source.resolve()
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    selection_path = data_dir / "train_selection.parquet"
    if sha256_file(source) != manifest["source_train"]["sha256"]:
        raise ValueError("source Train SHA-256 differs from manifest")
    if sha256_file(selection_path) != manifest["selection"]["sha256"]:
        raise ValueError("selection SHA-256 differs from manifest")

    selection = pq.read_table(selection_path)
    ids = selection["id"].to_pylist()
    targets = selection["target_interest"].to_pylist()
    source_table = pq.read_table(source, columns=["id", "target_interest"])
    source_labels = dict(
        zip(
            source_table["id"].to_pylist(),
            source_table["target_interest"].to_pylist(),
            strict=True,
        )
    )
    if len(source_labels) != len(source_table):
        raise ValueError("source Train contains duplicate IDs")
    if len(ids) != len(source_labels) or set(ids) != set(source_labels):
        raise ValueError("selection is not a permutation of the source Train IDs")
    if any(
        source_labels.get(sample_id) != target
        for sample_id, target in zip(ids, targets, strict=True)
    ):
        raise ValueError("selection labels differ from source Train labels")

    previous: set[str] = set()
    for size in SIZES:
        selected = load_selection_ids(selection_path, size)
        if previous - selected:
            raise ValueError(f"{size}: selections are not nested")
        counts = training_label_counts(source, selected)
        quotient, remainder = divmod(size, EXPECTED_LABEL_COUNT)
        if len(counts) != EXPECTED_LABEL_COUNT:
            raise ValueError(f"{size}: label support mismatch")
        if min(counts.values()) != quotient or max(counts.values()) != quotient + bool(remainder):
            raise ValueError(f"{size}: quotient/remainder balance mismatch")
        if sum(value == quotient + 1 for value in counts.values()) != remainder:
            raise ValueError(f"{size}: remainder class count mismatch")
        if size % 4:
            raise ValueError(f"{size}: four-rank distributed sampler would pad")
        expected = manifest["subsets"][str(size)]
        if expected["count_min"] != quotient or expected["count_max"] != quotient + bool(remainder):
            raise ValueError(f"{size}: manifest balance mismatch")
        if expected["optimizer_steps"] != (size + 31) // 32:
            raise ValueError(f"{size}: manifest optimizer steps mismatch")
        previous = selected
    print(
        f"Validated {len(SIZES)} nested rounded-size selections through {SIZES[-1]} rows; "
        "all cover 99 labels with class-count gap <= 1"
    )


if __name__ == "__main__":
    main()
