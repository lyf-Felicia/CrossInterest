#!/usr/bin/env python3
"""Validate nested Amazon data-size selections and their frozen evaluation contract."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file
from crossinterest.training.train import load_selection_ids

from crossinterest.experiments.data_scaling.build_dataset import DEFAULT_SOURCE, EXPECTED_LABEL_COUNT, SIZES


HERE = _source_workspace('eval_amazon_datasize')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    source = args.source.resolve()
    manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if sha256_file(source) != manifest["source_train"]["sha256"]:
        raise ValueError("source Train SHA-256 differs from manifest")
    selection_path = data_dir / "train_selection.parquet"
    if sha256_file(selection_path) != manifest["selection"]["sha256"]:
        raise ValueError("selection SHA-256 differs from manifest")
    table = pq.read_table(selection_path)
    ids = table["id"].to_pylist()
    targets = table["target_interest"].to_pylist()
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
    if any(source_labels.get(sample_id) != target for sample_id, target in zip(ids, targets, strict=True)):
        raise ValueError("selection labels differ from source Train labels")
    previous: set[str] = set()
    for size in SIZES:
        selected = load_selection_ids(selection_path, size)
        if previous - selected:
            raise ValueError(f"{size}: selections are not nested")
        counts = Counter(source_labels[sample_id] for sample_id in selected)
        if len(counts) != EXPECTED_LABEL_COUNT:
            raise ValueError(f"{size}: label support mismatch")
        quotient, remainder = divmod(size, EXPECTED_LABEL_COUNT)
        if min(counts.values()) != quotient:
            raise ValueError(f"{size}: class balance mismatch")
        if max(counts.values()) != quotient + bool(remainder):
            raise ValueError(f"{size}: class balance mismatch")
        if sum(count == quotient + 1 for count in counts.values()) != remainder:
            raise ValueError(f"{size}: remainder allocation mismatch")
        expected = manifest["subsets"][str(size)]
        if min(counts.values()) != expected["count_min"] or max(counts.values()) != expected["count_max"]:
            raise ValueError(f"{size}: manifest count range mismatch")
        previous = selected
    print(f"Validated {len(SIZES)} nested selections through {SIZES[-1]} rows")


if __name__ == "__main__":
    main()