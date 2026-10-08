#!/usr/bin/env python3
"""Validate and score full rankings produced by the pinned official MeRA code."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from crossinterest.baselines.mera.core import paper_metrics, validate_label_space, write_json
from crossinterest.baselines.mera.data import load_official_rows
from crossinterest.baselines.mera.evaluate import unified_metrics, write_jsonl


OFFICIAL_REPOSITORY = "https://github.com/GabyUSTC/MeRA.git"
OFFICIAL_COMMIT = "c07bdbe57bf82e5c938b478035baee78d630b17c"
QWEN2_REVISION = "8a16abf2848eda07cc5253dec660bf1ce007ad7a"


def validated_strict_method(method: str) -> str:
    allowed_methods = {
        "MeRA official-code strict execution reproduction",
        "MeRA author-clarified reproduction",
        "MeRA official method with model-architecture compatibility only",
        "MeRA author-clarified cross-model reproduction",
    }
    if method not in allowed_methods:
        raise ValueError(f"unexpected strict training contract: {method}")
    return method


def normalize_prediction(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def official_formula_metrics(
    rankings: list[list[str]],
    targets: list[list[str]],
    cutoffs: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, float]:
    """Mirror the loops in the official ``analyze_output.py`` implementation."""
    precision = {cutoff: 0.0 for cutoff in cutoffs}
    recall = {cutoff: 0.0 for cutoff in cutoffs}
    for ranking, golds in zip(rankings, targets, strict=True):
        gold_set = set(golds)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("arts", "grocery"), required=True)
    parser.add_argument("--official-result", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--label-space", type=Path, required=True)
    parser.add_argument("--train-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_records = json.loads(args.official_result.read_text(encoding="utf-8"))
    source_rows = load_official_rows(args.test_file)
    label_space = validate_label_space(
        json.loads(args.label_space.read_text(encoding="utf-8"))
    )
    train_contract = json.loads(args.train_contract.read_text(encoding="utf-8"))
    if len(result_records) != len(source_rows):
        raise ValueError(
            f"result/test row mismatch: {len(result_records)} != {len(source_rows)}"
        )
    labels = set(label_space["prediction_labels"])
    rankings = [
        [normalize_prediction(token) for token in record["predict"]]
        for record in result_records
    ]
    for index, ranking in enumerate(rankings):
        if len(ranking) != len(labels):
            raise ValueError(
                f"row {index} ranking length {len(ranking)} != {len(labels)}"
            )
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"row {index} ranking contains duplicate labels")
        if set(ranking) != labels:
            missing = sorted(labels - set(ranking))[:3]
            extra = sorted(set(ranking) - labels)[:3]
            raise ValueError(f"row {index} label mismatch: missing={missing}, extra={extra}")

    targets = [row["targets"] for row in source_rows]
    paper = paper_metrics(rankings, targets, cutoffs=(1, 2, 3, 4))
    independent_paper = official_formula_metrics(rankings, targets)
    for key, value in paper.items():
        if abs(value - independent_paper[key]) > 1e-12:
            raise AssertionError(
                f"official/project metric mismatch for {key}: "
                f"{independent_paper[key]} != {value}"
            )
    unified, unified_details = unified_metrics(source_rows, rankings, label_space)
    history_top1 = sum(
        ranking[0] in set(row["history"])
        for row, ranking in zip(source_rows, rankings, strict=True)
    ) / len(source_rows)
    history_top4_slots = sum(
        prediction in set(row["history"])
        for row, ranking in zip(source_rows, rankings, strict=True)
        for prediction in ranking[:4]
    ) / (4 * len(source_rows))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 1,
        "method": validated_strict_method(train_contract["method"]),
        "dataset": args.dataset,
        "model": train_contract["model"],
        "model_revision": train_contract["model_revision"],
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "source_sample_count": len(source_rows),
        "ranking_label_count": len(labels),
        "paper_metrics": paper,
        "official_formula_metrics": independent_paper,
        "unified_metrics": unified,
        "history_copy": {
            "top1_rate": history_top1,
            "top4_slot_rate": history_top4_slots,
        },
        "train_contract": train_contract,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
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
    write_jsonl(output_dir / "unified_details.jsonl", unified_details)
    official_top4 = []
    for record, ranking in zip(result_records, rankings, strict=True):
        official_top4.append(
            {
                **{key: value for key, value in record.items() if key != "predict"},
                "predict": [f'"{label}"' for label in ranking[:4]],
            }
        )
    write_json(output_dir / "official_top4_predictions.json", official_top4)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()