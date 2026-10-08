#!/usr/bin/env python3
"""Compare aligned base, SFT, and SuperLoss predictions with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VIEWS = ("behavior", "interest")
CUTOFFS = (10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for view in VIEWS:
        parser.add_argument(f"--{view}-base-choice", type=Path, required=True)
        parser.add_argument(f"--{view}-sft-choice", type=Path, required=True)
        parser.add_argument(f"--{view}-superloss-choice", type=Path, required=True)
        parser.add_argument(f"--{view}-sft-ranking", type=Path, required=True)
        parser.add_argument(f"--{view}-superloss-ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in rows:
                raise ValueError(f"{path}: duplicate sample {row['sample_id']}")
            rows[row["sample_id"]] = row
    return rows


def macro_mean(values: np.ndarray, labels: np.ndarray) -> float:
    values = [values[labels == label].mean() for label in np.unique(labels)]
    return float(np.mean(values))


def comparison(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
    binary: bool,
) -> dict:
    macro_delta = macro_mean(right, labels) - macro_mean(left, labels)
    macro_bootstrap = np.empty(replicates)
    label_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for index in range(replicates):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in label_indices]
        )
        macro_bootstrap[index] = macro_mean(right[sample], labels[sample]) - macro_mean(
            left[sample], labels[sample]
        )
    result = {
        "macro_delta": macro_delta,
        "macro_ci95": np.quantile(macro_bootstrap, [0.025, 0.975]).tolist(),
    }
    if binary:
        result.update(
            right_only_positive=int(np.sum((right == 1) & (left == 0))),
            left_only_positive=int(np.sum((left == 1) & (right == 0))),
        )
    return result


def aligned_values(
    left: dict[str, dict], right: dict[str, dict], metric: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if set(left) != set(right):
        raise ValueError("prediction files do not contain identical sample IDs")
    sample_ids = sorted(left)
    left_labels = [left[sample_id]["target_subcategory"] for sample_id in sample_ids]
    right_labels = [right[sample_id]["target_subcategory"] for sample_id in sample_ids]
    if left_labels != right_labels:
        raise ValueError("prediction files contain different target labels")
    labels = np.array(left_labels)
    left_values = np.array([float(left[sample_id][metric]) for sample_id in sample_ids])
    right_values = np.array([float(right[sample_id][metric]) for sample_id in sample_ids])
    return sample_ids, labels, left_values, right_values


def analyze_view(args: argparse.Namespace, view: str, rng: np.random.Generator) -> dict:
    choice = {
        "base": load(getattr(args, f"{view}_base_choice").resolve()),
        "sft": load(getattr(args, f"{view}_sft_choice").resolve()),
        "superloss": load(getattr(args, f"{view}_superloss_choice").resolve()),
    }
    sample_sets = {name: set(rows) for name, rows in choice.items()}
    if len({frozenset(values) for values in sample_sets.values()}) != 1:
        raise ValueError("choice prediction files do not contain identical sample IDs")
    sample_ids = sorted(sample_sets["base"])
    labels = np.array([choice["base"][sample_id]["target_subcategory"] for sample_id in sample_ids])
    correct = {
        name: np.array([int(rows[sample_id]["correct"]) for sample_id in sample_ids], dtype=float)
        for name, rows in choice.items()
    }
    for name, rows in choice.items():
        observed = [rows[sample_id]["target_subcategory"] for sample_id in sample_ids]
        if observed != labels.tolist():
            raise ValueError(f"{name}: target labels differ from base")
    sft_ranking = load(getattr(args, f"{view}_sft_ranking").resolve())
    superloss_ranking = load(getattr(args, f"{view}_superloss_ranking").resolve())
    ranking_metrics = {}
    for cutoff in CUTOFFS:
        for name in ("recall", "ndcg"):
            metric = f"{name}@{cutoff}"
            ranking_ids, ranking_labels, left, right = aligned_values(
                sft_ranking, superloss_ranking, metric
            )
            ranking_metrics[f"macro_{metric}"] = {
                "sft": macro_mean(left, ranking_labels),
                "superloss": macro_mean(right, ranking_labels),
                "superloss_minus_sft": comparison(
                    left,
                    right,
                    ranking_labels,
                    args.replicates,
                    rng,
                    binary=name == "recall",
                ),
            }
    return {
        "choice_sample_count": len(sample_ids),
        "ranking_sample_count": len(ranking_ids),
        "choice_macro_accuracy@10": {
            name: macro_mean(values, labels)
            for name, values in correct.items()
        },
        "comparisons": {
            "choice_sft_minus_base": comparison(
                correct["base"], correct["sft"], labels, args.replicates, rng, binary=True
            ),
            "choice_superloss_minus_sft": comparison(
                correct["sft"],
                correct["superloss"],
                labels,
                args.replicates,
                rng,
                binary=True,
            ),
        },
        "ranking": ranking_metrics,
    }


def main() -> None:
    args = parse_args()
    output = {
        "replicates": args.replicates,
        "seed": args.seed,
        "views": {
            view: analyze_view(
                args,
                view,
                np.random.default_rng(args.seed + view_index),
            )
            for view_index, view in enumerate(VIEWS)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()