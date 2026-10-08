#!/usr/bin/env python3
"""Validate and score official-code MeRA rankings on frozen Home Dev-2K."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from crossinterest.baselines.mera.core import home_metrics, query_set_metrics, write_json
from crossinterest.baselines.mera.evaluate import write_jsonl


def validated_home_method(method: str) -> str:
    allowed_methods = {
        "official-code MeRA Home extension",
        "MeRA author-clarified Home extension",
    }
    if method not in allowed_methods:
        raise ValueError(f"unexpected Home training contract: {method}")
    return method


def normalize_prediction(token: str) -> str:
    if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-result", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--train-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_records = json.loads(args.official_result.read_text(encoding="utf-8"))
    metadata = json.loads(args.test_metadata.read_text(encoding="utf-8"))
    mapping = json.loads(args.label_mapping.read_text(encoding="utf-8"))
    train_contract = json.loads(args.train_contract.read_text(encoding="utf-8"))
    if len(result_records) != 2_000 or len(metadata) != 2_000:
        raise ValueError("Home official result must contain exactly 2,000 rows")

    official_to_full = mapping["official_category_to_full"]
    all_labels = set(mapping["labels"])
    prediction_labels = set(mapping["prediction_labels"])
    full_rankings = []
    rankings = []
    for index, record in enumerate(result_records):
        official_ranking = [normalize_prediction(token) for token in record["predict"]]
        if len(official_ranking) != 136 or len(set(official_ranking)) != 136:
            raise ValueError(f"Home row {index} is not a complete 136-category ranking")
        if set(official_ranking) != set(official_to_full):
            raise ValueError(f"Home row {index} official category space mismatch")
        full_ranking = [official_to_full[label] for label in official_ranking]
        if set(full_ranking) != all_labels:
            raise ValueError(f"Home row {index} full category mapping mismatch")
        ranking = [label for label in full_ranking if label in prediction_labels]
        if len(ranking) != 112 or set(ranking) != prediction_labels:
            raise ValueError(f"Home row {index} prediction category filter mismatch")
        full_rankings.append(full_ranking)
        rankings.append(ranking)

    targets = [row["target"] for row in metadata]
    candidates = [row["candidates"] for row in metadata]
    if any(
        row["candidates"][row["gold_index"]] != row["target"] for row in metadata
    ):
        raise ValueError("Home frozen candidate/gold contract failed")
    if set(targets) != prediction_labels:
        raise ValueError("Home formal evaluation does not cover all prediction labels")

    category_macro = home_metrics(rankings, targets, candidates)
    query_mean = query_set_metrics(
        rankings, [[target] for target in targets], cutoffs=(10, 20)
    )
    accuracy = sum(
        next(label for label in ranking if label in set(candidate_set)) == target
        for ranking, target, candidate_set in zip(
            rankings, targets, candidates, strict=True
        )
    ) / len(targets)
    metrics = {
        "schema_version": 1,
        "method": validated_home_method(train_contract["method"]),
        "dataset": "home",
        "model": train_contract["model"],
        "model_revision": train_contract["model_revision"],
        "loss_condition": train_contract["loss_condition"],
        "source_sample_count": len(metadata),
        "context_label_count": len(all_labels),
        "prediction_label_count": len(prediction_labels),
        "query_accuracy_10way": accuracy,
        "query_mean": query_mean,
        "category_macro": category_macro,
        "train_contract": train_contract,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "source_rankings.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "target": row["target"],
                "ranking": ranking,
                "full_context_ranking": full_ranking,
            }
            for row, ranking, full_ranking in zip(
                metadata, rankings, full_rankings, strict=True
            )
        ],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()