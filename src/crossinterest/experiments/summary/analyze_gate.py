#!/usr/bin/env python3
"""Compare a summary-augmented ranking against the frozen direct-logit baseline."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_BASELINE = {
    "recall_at_10": 0.2266865079365079,
    "recall_at_20": 0.36855158730158716,
    "ndcg_at_10": 0.11600623798961322,
    "ndcg_at_20": 0.15181008809944949,
}
METRICS = tuple(EXPECTED_BASELINE)
TARGETS = {"recall_at_20": 0.45, "ndcg_at_20": 0.25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--candidate-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def load_details(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"duplicate sample ID in {path}: {sample_id}")
            target_rank = row.get("target_rank")
            if not isinstance(target_rank, int) or not 1 <= target_rank <= 112:
                raise ValueError(f"{sample_id}: invalid target_rank={target_rank}")
            rows[sample_id] = row
    if len(rows) != 2000:
        raise ValueError(f"{path}: expected 2000 details, found {len(rows)}")
    return rows


def values_from_rank(rank: int) -> np.ndarray:
    values = []
    for cutoff in (10, 20):
        values.append(float(rank <= cutoff))
    for cutoff in (10, 20):
        values.append(1 / math.log2(rank + 1) if rank <= cutoff else 0.0)
    return np.asarray(values, dtype=np.float64)


def metric_ordered_values(rank: int) -> np.ndarray:
    raw = values_from_rank(rank)
    return raw[[0, 1, 2, 3]]


def macro_metrics(values: np.ndarray, class_indices: list[np.ndarray]) -> np.ndarray:
    return np.mean([values[indices].mean(axis=0) for indices in class_indices], axis=0)


def paired_bootstrap(
    deltas: np.ndarray,
    class_indices: list[np.ndarray],
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    generator = np.random.default_rng(seed)
    distribution = np.zeros((samples, len(METRICS)), dtype=np.float64)
    for indices in class_indices:
        draws = generator.integers(0, len(indices), size=(samples, len(indices)))
        distribution += deltas[indices][draws].mean(axis=1)
    return distribution / len(class_indices)


def metric_dict(values: np.ndarray) -> dict[str, float]:
    return {metric: float(value) for metric, value in zip(METRICS, values, strict=True)}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    baseline = load_details(args.baseline_details)
    candidate = load_details(args.candidate_details)
    if set(baseline) != set(candidate):
        raise ValueError(
            f"sample ID mismatch: baseline_only={len(set(baseline) - set(candidate))}, "
            f"candidate_only={len(set(candidate) - set(baseline))}"
        )
    sample_ids = sorted(baseline)
    labels = []
    baseline_values = []
    candidate_values = []
    baseline_ranks = []
    candidate_ranks = []
    for sample_id in sample_ids:
        baseline_row = baseline[sample_id]
        candidate_row = candidate[sample_id]
        if baseline_row["target_interest"] != candidate_row["target_interest"]:
            raise ValueError(f"{sample_id}: target differs between paired details")
        labels.append(baseline_row["target_interest"])
        baseline_ranks.append(baseline_row["target_rank"])
        candidate_ranks.append(candidate_row["target_rank"])
        baseline_values.append(metric_ordered_values(baseline_row["target_rank"]))
        candidate_values.append(metric_ordered_values(candidate_row["target_rank"]))
    baseline_values_array = np.stack(baseline_values)
    candidate_values_array = np.stack(candidate_values)
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    if len(by_class) != 112:
        raise ValueError(f"expected 112 classes, found {len(by_class)}")
    class_indices = [np.asarray(by_class[label], dtype=np.int64) for label in sorted(by_class)]
    baseline_macro = macro_metrics(baseline_values_array, class_indices)
    candidate_macro = macro_metrics(candidate_values_array, class_indices)
    expected = np.asarray([EXPECTED_BASELINE[metric] for metric in METRICS])
    if not np.allclose(baseline_macro, expected, rtol=0, atol=1e-12):
        raise ValueError(
            "baseline does not reproduce the frozen direct metrics: "
            f"actual={metric_dict(baseline_macro)}"
        )
    deltas = candidate_values_array - baseline_values_array
    delta_macro = candidate_macro - baseline_macro
    bootstrap = paired_bootstrap(
        deltas,
        class_indices,
        args.bootstrap_samples,
        args.seed,
    )
    lower = np.quantile(bootstrap, 0.025, axis=0)
    upper = np.quantile(bootstrap, 0.975, axis=0)
    confidence_intervals = {
        metric: {
            "lower_95": float(low),
            "upper_95": float(high),
            "probability_positive": float(np.mean(bootstrap[:, index] > 0)),
        }
        for index, (metric, low, high) in enumerate(
            zip(METRICS, lower, upper, strict=True)
        )
    }
    baseline_rank_array = np.asarray(baseline_ranks)
    candidate_rank_array = np.asarray(candidate_ranks)
    target_status = {
        metric: {
            "target": target,
            "actual": float(candidate_macro[METRICS.index(metric)]),
            "gap": float(candidate_macro[METRICS.index(metric)] - target),
            "met": bool(candidate_macro[METRICS.index(metric)] >= target),
        }
        for metric, target in TARGETS.items()
    }
    stable_primary_improvement = all(
        confidence_intervals[metric]["lower_95"] > 0
        for metric in ("recall_at_20", "ndcg_at_20")
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "paired_samples": len(sample_ids),
        "class_count": len(by_class),
        "baseline": metric_dict(baseline_macro),
        "candidate": metric_dict(candidate_macro),
        "delta": metric_dict(delta_macro),
        "delta_confidence_intervals": confidence_intervals,
        "rank_comparison": {
            "improved": int(np.sum(candidate_rank_array < baseline_rank_array)),
            "unchanged": int(np.sum(candidate_rank_array == baseline_rank_array)),
            "worsened": int(np.sum(candidate_rank_array > baseline_rank_array)),
            "median_baseline_rank": float(np.median(baseline_rank_array)),
            "median_candidate_rank": float(np.median(candidate_rank_array)),
        },
        "target_status": target_status,
        "gate": {
            "baseline_reproduced": True,
            "stable_primary_improvement": stable_primary_improvement,
            "both_targets_met": all(status["met"] for status in target_status.values()),
            "recommend_training_canary": stable_primary_improvement,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()