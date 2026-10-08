#!/usr/bin/env python3
"""Validate and score full rankings produced by the pinned official MeRA code."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    from crossinterest.baselines.mera_strict.prepare_official import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, QWEN2_MODEL, QWEN2_REVISION
except ImportError:
    from prepare_official import (  # type: ignore[no-redef]
        OFFICIAL_COMMIT,
        OFFICIAL_REPOSITORY,
        QWEN2_MODEL,
        QWEN2_REVISION,
    )


OFFICIAL_INPUT_PREFIX = "The user has given high ratings to the following categories"


def normalize_prediction(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def parse_category_map(path: Path) -> list[str]:
    labels = [line.rstrip().rsplit("\t", 1)[0] for line in path.read_text().splitlines()]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("official category map must contain unique, non-empty labels")
    return labels


def parse_official_rows(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        input_text = record["input"]
        if OFFICIAL_INPUT_PREFIX not in input_text:
            raise ValueError(f"{path}:{index} has an unrecognized input prompt")
        _, separator, history_literal = input_text.partition(":")
        if not separator:
            raise ValueError(f"{path}:{index} has no history separator")
        history = ast.literal_eval(history_literal.strip())
        if isinstance(history, str):
            history = [history]
        if not isinstance(history, (list, tuple)) or not all(
            isinstance(category, str) for category in history
        ):
            raise ValueError(f"{path}:{index} history is not a category sequence")
        targets = re.findall(r'"(.*?)"', record["output"])
        if not targets:
            raise ValueError(f"{path}:{index} has no target categories")
        rows.append(
            {
                "sample_id": f"{path.stem}:{index}",
                "history": list(history),
                "targets": targets,
            }
        )
    return rows


def official_formula_metrics(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
    cutoffs: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, float]:
    """Mirror the loops in the official ``analyze_output.py`` implementation."""
    if len(rankings) != len(targets) or not rankings:
        raise ValueError("rankings and targets must have the same non-zero length")
    precision = {cutoff: 0.0 for cutoff in cutoffs}
    recall = {cutoff: 0.0 for cutoff in cutoffs}
    for ranking, golds in zip(rankings, targets, strict=True):
        gold_set = set(golds)
        if not gold_set:
            raise ValueError("target set cannot be empty")
        for cutoff in cutoffs:
            hits = len(set(ranking[:cutoff]) & gold_set)
            precision[cutoff] += hits / cutoff
            recall[cutoff] += hits / len(gold_set)
    count = len(rankings)
    return {
        **{
            f"precision_at_{cutoff}": precision[cutoff] / count
            for cutoff in cutoffs
        },
        **{
            f"recall_at_{cutoff}": recall[cutoff] / count
            for cutoff in cutoffs
        },
    }


def validate_rankings(records: list[dict], labels: list[str]) -> list[list[str]]:
    label_set = set(labels)
    rankings = [
        [normalize_prediction(token) for token in record["predict"]]
        for record in records
    ]
    for index, ranking in enumerate(rankings):
        if len(ranking) != len(labels):
            raise ValueError(
                f"row {index} ranking length {len(ranking)} != {len(labels)}"
            )
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"row {index} ranking contains duplicate labels")
        if set(ranking) != label_set:
            missing = sorted(label_set - set(ranking))[:3]
            extra = sorted(set(ranking) - label_set)[:3]
            raise ValueError(f"row {index} label mismatch: missing={missing}, extra={extra}")
    return rankings


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("arts", "grocery"), required=True)
    parser.add_argument("--official-result", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--category-map", type=Path, required=True)
    parser.add_argument("--train-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_records = json.loads(args.official_result.read_text(encoding="utf-8"))
    source_rows = parse_official_rows(args.test_file)
    labels = parse_category_map(args.category_map)
    train_contract = json.loads(args.train_contract.read_text(encoding="utf-8"))
    if len(result_records) != len(source_rows):
        raise ValueError(
            f"result/test row mismatch: {len(result_records)} != {len(source_rows)}"
        )
    rankings = validate_rankings(result_records, labels)
    targets = [row["targets"] for row in source_rows]
    metrics = {
        "schema_version": 1,
        "method": "MeRA official-code strict execution reproduction",
        "dataset": args.dataset,
        "model": QWEN2_MODEL,
        "model_revision": QWEN2_REVISION,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "source_sample_count": len(source_rows),
        "ranking_label_count": len(labels),
        "official_metrics": official_formula_metrics(rankings, targets),
        "train_contract": train_contract,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_dir = args.output_dir.resolve()
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "source_rankings.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "targets": row["targets"],
                "ranking": ranking,
            }
            for row, ranking in zip(source_rows, rankings, strict=True)
        ],
    )
    write_json(
        output_dir / "official_top4_predictions.json",
        [
            {
                **{key: value for key, value in record.items() if key != "predict"},
                "predict": [f'"{label}"' for label in ranking[:4]],
            }
            for record, ranking in zip(result_records, rankings, strict=True)
        ],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()