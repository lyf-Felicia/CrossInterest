#!/usr/bin/env python3
"""Build a class-balanced temporal OOF selection with reliability weights."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB

from crossinterest.experiments.denoising.build_selection import nested_ranks, sha256_file, validate_source
from crossinterest.experiments.denoising.proxy_pilot import SEED, VIEWS
from crossinterest.experiments.denoising.proxy_teacher_search import teacher_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, default="behavior_interest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--keep-per-label", type=int, default=714)
    parser.add_argument("--expected-rows", type=int, default=112000)
    parser.add_argument("--expected-labels", type=int, default=112)
    parser.add_argument("--expected-rows-per-label", type=int, default=1000)
    return parser.parse_args()


def cross_user_oof_loss(
    rows: list[dict],
    features,
    targets: np.ndarray,
    labels: list[str],
    folds: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray([row["user_id"] for row in rows], dtype=object)
    probabilities = np.zeros(len(rows), dtype=np.float64)
    fold_assignments = np.zeros(len(rows), dtype=np.int8)
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=SEED,
    )
    dummy = np.zeros(len(rows), dtype=np.int8)
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(dummy, targets, groups), 1
    ):
        train_users = set(groups[train_indices])
        heldout_users = set(groups[heldout_indices])
        if train_users & heldout_users:
            raise AssertionError(f"fold {fold}: train/heldout users overlap")
        model = MultinomialNB(alpha=alpha)
        model.fit(features[train_indices], targets[train_indices])
        fold_probabilities = model.predict_proba(features[heldout_indices])
        class_columns = {
            category: index for index, category in enumerate(model.classes_)
        }
        if set(class_columns) != set(labels):
            raise AssertionError(f"fold {fold}: incomplete training label support")
        probabilities[heldout_indices] = fold_probabilities[
            np.arange(len(heldout_indices)),
            [class_columns[targets[index]] for index in heldout_indices],
        ]
        fold_assignments[heldout_indices] = fold
        print(
            f"fold={fold}/{folds} train={len(train_indices)} "
            f"heldout={len(heldout_indices)}",
            flush=True,
        )
    if np.any(fold_assignments == 0) or np.any(probabilities <= 0):
        raise AssertionError("OOF scoring did not cover every row")
    return -np.log(np.clip(probabilities, 1e-12, 1.0)), fold_assignments


def main() -> None:
    args = parse_args()
    if args.alpha <= 0 or args.temperature <= 0:
        raise ValueError("--alpha and --temperature must be positive")
    if not 0 < args.keep_per_label <= args.expected_rows_per_label:
        raise ValueError("--keep-per-label is outside the class size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "user_id",
        "historical_titles",
        "historical_interests",
        "target_interest",
    ]
    rows = pq.read_table(args.train_file, columns=columns).to_pylist()
    rows = sorted(rows, key=lambda row: row["id"])
    labels = validate_source(rows, args)
    targets = np.asarray([row["target_interest"] for row in rows], dtype=object)
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    features = vectorizer.transform(
        [teacher_features(row, args.view, "temporal_combined") for row in rows]
    )
    losses, folds = cross_user_oof_loss(
        rows,
        features,
        targets,
        labels,
        args.folds,
        args.alpha,
    )
    within_class, nested = nested_ranks(
        rows,
        labels,
        lambda index: float(losses[index]),
    )
    rank_percentile = (within_class.astype(np.float64) - 1.0) / max(
        1, args.expected_rows_per_label - 1
    )
    sample_weight = np.exp(-args.temperature * rank_percentile)
    selected_mask = within_class <= args.keep_per_label
    sample_weight /= sample_weight[selected_mask].mean()
    selected_rows = int(selected_mask.sum())
    expected_selected = args.expected_labels * args.keep_per_label
    if selected_rows != expected_selected:
        raise AssertionError(
            f"expected {expected_selected} selected rows, found {selected_rows}"
        )
    selected_counts = Counter(targets[selected_mask])
    if set(selected_counts.values()) != {args.keep_per_label}:
        raise AssertionError("selected classes are not exactly balanced")
    selected_weights = sample_weight[selected_mask]
    if not np.isclose(selected_weights.mean(), 1.0, atol=1e-12):
        raise AssertionError("selected sample weights do not have mean 1")
    effective_sample_size = float(
        selected_weights.sum() ** 2 / np.square(selected_weights).sum()
    )
    output_path = args.output_dir / f"temporal_mnb_weighted_{args.view}.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [row["id"] for row in rows],
                "target_interest": targets,
                "selection_method": ["temporal_mnb_oof_weighted"] * len(rows),
                "oof_loss": losses,
                "oof_fold": folds,
                "within_class_rank": within_class,
                "nested_rank": nested,
                "sample_weight": sample_weight,
            }
        ),
        output_path,
        compression="zstd",
    )
    manifest = {
        "schema_version": 1,
        "method": "class-balanced temporal MultinomialNB OOF selection with rank weights",
        "seed": SEED,
        "view": args.view,
        "folds": args.folds,
        "rows": len(rows),
        "labels": len(labels),
        "rows_per_label": args.expected_rows_per_label,
        "keep_per_label": args.keep_per_label,
        "selected_rows": selected_rows,
        "teacher": {
            "features": "temporal_combined HashingVectorizer",
            "model": "MultinomialNB",
            "alpha": args.alpha,
            "n_features": args.n_features,
            "ngram_range": [1, 2],
        },
        "weighting": {
            "formula": "exp(-temperature * (within_class_rank - 1) / (rows_per_label - 1))",
            "temperature": args.temperature,
            "selected_mean": float(selected_weights.mean()),
            "selected_min": float(selected_weights.min()),
            "selected_max": float(selected_weights.max()),
            "selected_effective_sample_size": effective_sample_size,
        },
        "source": {
            "path": str(args.train_file.resolve()),
            "sha256": sha256_file(args.train_file),
        },
        "selection": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    manifest_path = args.output_dir / f"manifest_temporal_weighted_{args.view}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()