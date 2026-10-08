#!/usr/bin/env python3
"""Build class-balanced selections from cross-user OOF prediction loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from crossinterest.experiments.denoising.proxy_pilot import SEED, VIEWS, classifier, render_features, stable_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument("--expected-rows", type=int, default=112000)
    parser.add_argument("--expected-labels", type=int, default=112)
    parser.add_argument("--expected-rows-per-label", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(rows: list[dict], args: argparse.Namespace) -> list[str]:
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("source contains duplicate sample IDs")
    labels = sorted({row["target_interest"] for row in rows})
    if len(labels) != args.expected_labels:
        raise ValueError(f"expected {args.expected_labels} labels, found {len(labels)}")
    counts = Counter(row["target_interest"] for row in rows)
    if set(counts.values()) != {args.expected_rows_per_label}:
        raise ValueError(f"source class counts differ: {sorted(set(counts.values()))}")
    return labels


def cross_user_oof_loss(
    rows: list[dict],
    features,
    targets: np.ndarray,
    labels: list[str],
    folds: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.array([row["user_id"] for row in rows], dtype=object)
    label_to_index = {label: index for index, label in enumerate(labels)}
    target_indices = np.array([label_to_index[target] for target in targets])
    probabilities = np.zeros(len(rows), dtype=np.float32)
    fold_assignments = np.zeros(len(rows), dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    dummy = np.zeros(len(rows), dtype=np.int8)
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(dummy, targets, groups), 1
    ):
        train_users = set(groups[train_indices])
        heldout_users = set(groups[heldout_indices])
        if train_users & heldout_users:
            raise AssertionError(f"fold {fold}: train/heldout users overlap")
        model = classifier(max_iter)
        model.fit(features[train_indices], targets[train_indices])
        fold_probabilities = model.predict_proba(features[heldout_indices])
        class_columns = {category: index for index, category in enumerate(model.classes_)}
        if set(class_columns) != set(labels):
            raise AssertionError(f"fold {fold}: training label support is incomplete")
        probabilities[heldout_indices] = fold_probabilities[
            np.arange(len(heldout_indices)),
            [class_columns[targets[index]] for index in heldout_indices],
        ]
        fold_assignments[heldout_indices] = fold
        print(
            f"fold={fold}/{folds} train={len(train_indices)} "
            f"heldout={len(heldout_indices)} train_users={len(train_users)} "
            f"heldout_users={len(heldout_users)}",
            flush=True,
        )
    if np.any(fold_assignments == 0) or np.any(probabilities <= 0):
        raise AssertionError("OOF scoring did not cover every row")
    losses = -np.log(np.clip(probabilities, 1e-8, 1.0))
    return losses, fold_assignments


def nested_ranks(
    rows: list[dict], labels: list[str], score_key
) -> tuple[np.ndarray, np.ndarray]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row["target_interest"]].append(index)
    within_class = np.zeros(len(rows), dtype=np.int32)
    nested = np.zeros(len(rows), dtype=np.int32)
    label_order = {label: index for index, label in enumerate(labels)}
    for label, indices in sorted(by_class.items()):
        ordered = sorted(indices, key=lambda index: (score_key(index), rows[index]["id"]))
        for rank, index in enumerate(ordered, 1):
            within_class[index] = rank
            nested[index] = (rank - 1) * len(labels) + label_order[label] + 1
    if sorted(nested.tolist()) != list(range(1, len(rows) + 1)):
        raise AssertionError("nested ranks must be exactly 1..N")
    return within_class, nested


def write_selection(
    path: Path,
    rows: list[dict],
    within_class: np.ndarray,
    nested: np.ndarray,
    method: str,
    losses: np.ndarray | None = None,
    folds: np.ndarray | None = None,
) -> None:
    columns: dict[str, object] = {
        "id": [row["id"] for row in rows],
        "target_interest": [row["target_interest"] for row in rows],
        "selection_method": [method] * len(rows),
        "within_class_rank": within_class,
        "nested_rank": nested,
    }
    if losses is not None:
        columns["oof_loss"] = losses
    if folds is not None:
        columns["oof_fold"] = folds
    pq.write_table(pa.table(columns), path, compression="zstd")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_columns = [
        "id",
        "user_id",
        "historical_titles",
        "historical_interests",
        "target_interest",
    ]
    rows = pq.read_table(args.train_file, columns=source_columns).to_pylist()
    rows = sorted(rows, key=lambda row: row["id"])
    labels = validate_source(rows, args)
    targets = np.array([row["target_interest"] for row in rows], dtype=object)
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    features = vectorizer.transform([render_features(row, args.view) for row in rows])
    losses, folds = cross_user_oof_loss(
        rows, features, targets, labels, args.folds, args.max_iter
    )
    oof_within, oof_nested = nested_ranks(
        rows, labels, lambda index: float(losses[index])
    )
    random_within, random_nested = nested_ranks(
        rows,
        labels,
        lambda index: stable_key("formal-random-selection", rows[index]["id"]),
    )
    oof_path = args.output_dir / f"oof_low_loss_{args.view}.parquet"
    random_path = args.output_dir / "random_balanced.parquet"
    write_selection(
        oof_path,
        rows,
        oof_within,
        oof_nested,
        "cross_user_oof_low_loss",
        losses,
        folds,
    )
    if random_path.exists():
        existing_random = pq.read_table(random_path, columns=["id", "nested_rank"])
        expected_random = pa.table(
            {"id": [row["id"] for row in rows], "nested_rank": random_nested}
        )
        if not existing_random.equals(expected_random):
            raise ValueError(f"existing random selection differs: {random_path}")
    else:
        write_selection(
            random_path,
            rows,
            random_within,
            random_nested,
            "stable_hash_random_balanced",
        )
    manifest = {
        "schema_version": 1,
        "method": "class-balanced cross-user OOF low-loss selection",
        "seed": SEED,
        "view": args.view,
        "folds": args.folds,
        "rows": len(rows),
        "labels": len(labels),
        "rows_per_label": args.expected_rows_per_label,
        "feature_model": {
            "type": "HashingVectorizer + SGDClassifier(log_loss)",
            "n_features": args.n_features,
            "ngram_range": [1, 2],
            "max_iter": args.max_iter,
        },
        "source": {
            "path": str(args.train_file.resolve()),
            "sha256": sha256_file(args.train_file),
        },
        "oof_selection": {
            "path": str(oof_path.resolve()),
            "sha256": sha256_file(oof_path),
            "mean_loss": float(losses.mean()),
            "median_loss": float(np.median(losses)),
        },
        "random_selection": {
            "path": str(random_path.resolve()),
            "sha256": sha256_file(random_path),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    manifest_path = args.output_dir / f"manifest_{args.view}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()