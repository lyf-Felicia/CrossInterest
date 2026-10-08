#!/usr/bin/env python3
"""Validate one Amazon label-space dataset and freeze its SFT artifacts."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file, write_json_atomic

from crossinterest.training.label_space import LABEL_MODES, build_label_mapping, validate_label_mapping
from crossinterest.training.prompts import INPUT_TYPES, render_training_messages


HERE = _source_workspace('eval_amazon_sft_112')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--expected-label-count", type=int, required=True)
    parser.add_argument("--expected-quota", type=int, required=True)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-eval-rows", type=int, default=2_000)
    return parser.parse_args()


def validate_no_label_ids(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    columns = ("historical_titles", "historical_interests", "full_seen_interests")
    for batch in parquet.iter_batches(columns=columns, batch_size=10_000):
        for row in batch.to_pylist():
            for column in columns:
                if any(
                    "<interest_" in value or "<sid_" in value
                    for value in row[column]
                ):
                    raise ValueError(f"{path}: concrete label ID found in {column}")


def load_eval_rows(eval_dir: Path, expected_rows: int, support: set[str]) -> dict[str, list[dict]]:
    views = {}
    for view in INPUT_TYPES:
        path = eval_dir / f"dev_2k_{view}.parquet"
        rows = pq.read_table(path).to_pylist()
        if len(rows) != expected_rows:
            raise ValueError(f"{view}: expected {expected_rows} eval rows, found {len(rows)}")
        if len({row["sample_id"] for row in rows}) != expected_rows:
            raise ValueError(f"{view}: duplicate sample IDs")
        if any(row["target_interest"] not in support for row in rows):
            raise ValueError(f"{view}: eval target outside Train support")
        views[view] = rows
    reference = views[INPUT_TYPES[0]]
    for view in INPUT_TYPES[1:]:
        for first, other in zip(reference, views[view], strict=True):
            for field in ("source_sample_id", "target_interest", "candidate_interests", "gold_index"):
                if first[field] != other[field]:
                    raise ValueError(f"paired view mismatch: {first['source_sample_id']}")
    return views


def main() -> None:
    args = parse_args()
    train = args.train.resolve()
    dev = args.dev.resolve()
    eval_dir = args.eval_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_table = pq.read_table(train, columns=["target_interest"])
    if len(train_table) != args.expected_train_rows:
        raise ValueError(
            f"expected {args.expected_train_rows} Train rows, found {len(train_table)}"
        )
    train_counts = Counter(train_table["target_interest"].to_pylist())
    if len(train_counts) != args.expected_label_count:
        raise ValueError(
            f"expected {args.expected_label_count} Train labels, found {len(train_counts)}"
        )
    if set(train_counts.values()) != {args.expected_quota}:
        raise ValueError("Train labels do not all have the expected quota")
    support = set(train_counts)

    dev_table = pq.read_table(dev, columns=["target_interest"])
    dev_counts = Counter(dev_table["target_interest"].to_pylist())
    if set(dev_counts) != support:
        raise ValueError("closed-set Dev support differs from Train support")
    validate_no_label_ids(train)
    validate_no_label_ids(dev)
    eval_views = load_eval_rows(eval_dir, args.expected_eval_rows, support)

    mapping = validate_label_mapping(
        build_label_mapping(support),
        expected_categories=support,
        expected_count=args.expected_label_count,
    )
    mapping_path = output_dir / "interest_id_semantic_mapping.json"
    write_json_atomic(mapping_path, mapping)

    preview_source = pq.read_table(
        train,
        columns=[
            "id",
            "historical_titles",
            "historical_interests",
            "full_seen_interests",
            "target_interest",
        ],
    ).slice(0, 1).to_pylist()[0]
    previews = []
    for label_mode in LABEL_MODES:
        for view in INPUT_TYPES:
            messages = render_training_messages(preview_source, view, label_mode, mapping)
            if "Candidates:" in messages[0]["content"]:
                raise ValueError("training prompt contains candidates")
            previews.append(
                {
                    "view": view,
                    "label_mode": label_mode,
                    "source_id": preview_source["id"],
                    "target_category": preview_source["target_interest"],
                    "messages": messages,
                }
            )
    write_json_atomic(output_dir / "TRAINING_SAMPLE_PREVIEW.json", previews)

    manifest = {
        "schema_version": 1,
        "task": "amazon_new_interest_sft_label_space",
        "label_count": len(support),
        "per_label_train_quota": args.expected_quota,
        "train": {
            "path": str(train),
            "rows": len(train_table),
            "sha256": sha256_file(train),
            "label_count": len(train_counts),
            "count_min": min(train_counts.values()),
            "count_max": max(train_counts.values()),
        },
        "dev": {
            "path": str(dev),
            "rows": len(dev_table),
            "sha256": sha256_file(dev),
            "label_count": len(dev_counts),
            "resampled": False,
        },
        "evaluation": {
            "rows_per_view": args.expected_eval_rows,
            "views": {
                view: {
                    "path": str(eval_dir / f"dev_2k_{view}.parquet"),
                    "sha256": sha256_file(eval_dir / f"dev_2k_{view}.parquet"),
                    "rows": len(rows),
                }
                for view, rows in eval_views.items()
            },
        },
        "label_modes": list(LABEL_MODES),
        "interest_id_mapping": {
            "path": str(mapping_path),
            "sha256": sha256_file(mapping_path),
            "mapping_fingerprint": mapping["mapping_fingerprint"],
            "method": mapping["method"],
            "semantic_token_count": mapping["semantic_token_count"],
            "registered_as_special_tokens": True,
            "token_initialization": "mean base-model rows of child-label text tokens only",
        },
        "input_contract": {
            "history_uses_natural_language_categories": True,
            "history_contains_concrete_interest_ids": False,
            "training_candidates_in_prompt": False,
            "views": list(INPUT_TYPES),
        },
    }
    write_json_atomic(output_dir / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
