#!/usr/bin/env python3
"""Falsify OOF learnability scoring before launching LLM-scale experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260901
VIEWS = ("behavior", "interest", "behavior_interest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, default="behavior_interest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--current-fraction", type=float, default=0.1)
    parser.add_argument("--noise-rate", type=float, default=0.1)
    parser.add_argument(
        "--selection-method",
        choices=("oof_low_loss", "learning_progress"),
        default="oof_low_loss",
    )
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=2**18)
    return parser.parse_args()


def stable_key(namespace: str, *values: object) -> bytes:
    payload = ":".join((str(SEED), namespace, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode()).digest()


def select_balanced(rows: list[dict], per_class: int) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["target_interest"]].append(row)
    selected = []
    for category, class_rows in sorted(by_class.items()):
        if len(class_rows) < per_class:
            raise ValueError(
                f"{category}: requested {per_class} rows, found {len(class_rows)}"
            )
        selected.extend(
            sorted(
                class_rows,
                key=lambda row: stable_key("pilot-row", row["id"]),
            )[:per_class]
        )
    return sorted(selected, key=lambda row: row["id"])


def render_features(row: dict, view: str) -> str:
    parts = []
    if view in {"behavior", "behavior_interest"}:
        parts.append(
            " ".join(f"title_{title}" for title in row["historical_titles"])
        )
    if view in {"interest", "behavior_interest"}:
        parts.append(
            " ".join(
                f"interest_{interest.replace(' ', '_')}"
                for interest in row["historical_interests"]
            )
        )
    return " view_break ".join(parts)


def corrupt_labels(
    rows: list[dict], labels: list[str], noise_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.array([row["target_interest"] for row in rows], dtype=object)
    observed = clean.copy()
    corrupted = np.zeros(len(rows), dtype=bool)
    if noise_rate == 0:
        return observed, corrupted
    if not 0 < noise_rate < 1:
        raise ValueError("--noise-rate must be in [0, 1)")
    label_index = {label: index for index, label in enumerate(labels)}
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(clean):
        by_class[label].append(index)
    count = round(len(next(iter(by_class.values()))) * noise_rate)
    for label, indices in sorted(by_class.items()):
        chosen = sorted(
            indices,
            key=lambda index: stable_key("noise-row", rows[index]["id"]),
        )[:count]
        replacement = labels[(label_index[label] + 1) % len(labels)]
        observed[chosen] = replacement
        corrupted[chosen] = True
    if Counter(observed) != Counter(clean):
        raise AssertionError("cyclic synthetic corruption must preserve class counts")
    return observed, corrupted


def classifier(max_iter: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=max_iter,
        tol=1e-4,
        random_state=SEED,
        average=True,
        n_jobs=-1,
    )


def oof_learning_probabilities(
    rows: list[dict],
    features,
    labels: np.ndarray,
    groups: np.ndarray,
    label_to_index: dict[str, int],
    folds: int,
    low_resource_fraction: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < low_resource_fraction < 1:
        raise ValueError("--current-fraction must be in (0, 1)")
    high_probabilities = np.zeros((len(labels), len(label_to_index)), dtype=np.float32)
    low_probabilities = np.zeros_like(high_probabilities)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    dummy = np.zeros(len(labels), dtype=np.int8)
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(dummy, labels, groups), 1
    ):
        by_class: dict[str, list[int]] = defaultdict(list)
        for index in train_indices:
            by_class[labels[index]].append(int(index))
        low_indices = []
        for indices in by_class.values():
            keep = max(1, round(len(indices) * low_resource_fraction))
            low_indices.extend(
                sorted(
                    indices,
                    key=lambda index: stable_key(
                        f"low-resource-fold-{fold}", rows[index]["id"]
                    ),
                )[:keep]
            )
        for destination, fit_indices in (
            (low_probabilities, np.array(low_indices, dtype=np.int64)),
            (high_probabilities, train_indices),
        ):
            model = classifier(max_iter)
            model.fit(features[fit_indices], labels[fit_indices])
            fold_probabilities = model.predict_proba(features[heldout_indices])
            for source_column, category in enumerate(model.classes_):
                destination[heldout_indices, label_to_index[category]] = (
                    fold_probabilities[:, source_column]
                )
        print(
            f"fold={fold}/{folds} low={len(low_indices)} high={len(train_indices)} "
            f"heldout={len(heldout_indices)}",
            flush=True,
        )
    for name, probabilities in (
        ("low-resource", low_probabilities),
        ("high-resource", high_probabilities),
    ):
        if np.any(probabilities.sum(axis=1) < 0.999):
            raise AssertionError(f"{name} OOF probability rows are incomplete")
    return low_probabilities, high_probabilities


def ranking_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    target_indices = np.array([labels.index(target) for target in targets])
    order = np.argsort(-probabilities, axis=1)
    ranks = np.empty(len(targets), dtype=np.int32)
    for index, target_index in enumerate(target_indices):
        ranks[index] = int(np.flatnonzero(order[index] == target_index)[0]) + 1
    per_class: dict[str, list[int]] = defaultdict(list)
    for target, rank in zip(targets, ranks, strict=True):
        per_class[target].append(int(rank))

    def macro(metric) -> float:
        return float(
            np.mean(
                [np.mean([metric(rank) for rank in class_ranks]) for class_ranks in per_class.values()]
            )
        )

    return {
        "macro_accuracy": macro(lambda rank: rank == 1),
        "macro_recall_at_10": macro(lambda rank: rank <= 10),
        "macro_recall_at_20": macro(lambda rank: rank <= 20),
        "macro_ndcg_at_10": macro(
            lambda rank: 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
        ),
        "macro_ndcg_at_20": macro(
            lambda rank: 1.0 / math.log2(rank + 1) if rank <= 20 else 0.0
        ),
    }


def fit_and_evaluate(
    train_features,
    train_labels: np.ndarray,
    selected_indices: np.ndarray,
    dev_features,
    dev_targets: np.ndarray,
    labels: list[str],
    max_iter: int,
    sample_weights: np.ndarray | None = None,
) -> dict[str, float]:
    model = classifier(max_iter)
    model.fit(
        train_features[selected_indices],
        train_labels[selected_indices],
        sample_weight=sample_weights,
    )
    raw = model.predict_proba(dev_features)
    probabilities = np.zeros((len(dev_targets), len(labels)), dtype=np.float32)
    for source_column, category in enumerate(model.classes_):
        probabilities[:, labels.index(category)] = raw[:, source_column]
    return ranking_metrics(probabilities, dev_targets, labels)


def selection_indices(
    rows: list[dict],
    observed: np.ndarray,
    irreducible_loss: np.ndarray,
    rho_loss: np.ndarray,
    keep_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("--keep-fraction must be in (0, 1]")
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(observed):
        by_class[category].append(index)
    selected_quality = []
    selected_rho = []
    selected_random = []
    quality_within_class_rank = np.zeros(len(rows), dtype=np.int32)
    rho_within_class_rank = np.zeros(len(rows), dtype=np.int32)
    for category, indices in sorted(by_class.items()):
        keep = max(1, round(len(indices) * keep_fraction))
        quality_order = sorted(indices, key=lambda index: (irreducible_loss[index], rows[index]["id"]))
        rho_order = sorted(indices, key=lambda index: (-rho_loss[index], rows[index]["id"]))
        random_order = sorted(indices, key=lambda index: stable_key("random-select", rows[index]["id"]))
        selected_quality.extend(quality_order[:keep])
        selected_rho.extend(rho_order[:keep])
        selected_random.extend(random_order[:keep])
        for rank, index in enumerate(quality_order, 1):
            quality_within_class_rank[index] = rank
        for rank, index in enumerate(rho_order, 1):
            rho_within_class_rank[index] = rank
    return (
        np.array(sorted(selected_quality), dtype=np.int64),
        np.array(sorted(selected_rho), dtype=np.int64),
        np.array(sorted(selected_random), dtype=np.int64),
        quality_within_class_rank,
        rho_within_class_rank,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_columns = [
        "id",
        "user_id",
        "historical_titles",
        "historical_interests",
        "target_interest",
    ]
    train_rows = pq.read_table(args.train_file, columns=train_columns).to_pylist()
    train_rows = select_balanced(train_rows, args.per_class)
    dev_rows = pq.read_table(
        args.dev_file,
        columns=[
            "source_sample_id",
            "historical_titles",
            "historical_interests",
            "target_interest",
        ],
    ).to_pylist()
    labels = sorted({row["target_interest"] for row in train_rows})
    if set(labels) != {row["target_interest"] for row in dev_rows}:
        raise ValueError("train and dev target supports differ")
    label_to_index = {label: index for index, label in enumerate(labels)}
    observed, corrupted = corrupt_labels(train_rows, labels, args.noise_rate)
    groups = np.array([row["user_id"] for row in train_rows], dtype=object)
    clean = np.array([row["target_interest"] for row in train_rows], dtype=object)
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    train_features = vectorizer.transform(
        [render_features(row, args.view) for row in train_rows]
    )
    dev_features = vectorizer.transform(
        [render_features(row, args.view) for row in dev_rows]
    )
    low_oof, high_oof = oof_learning_probabilities(
        train_rows,
        train_features,
        observed,
        groups,
        label_to_index,
        args.folds,
        args.current_fraction,
        args.max_iter,
    )
    observed_indices = np.array([label_to_index[label] for label in observed])
    observed_probability = np.clip(
        high_oof[np.arange(len(observed)), observed_indices], 1e-8, 1.0
    )
    irreducible_loss = -np.log(observed_probability)
    current_target_probability = np.clip(
        low_oof[np.arange(len(observed)), observed_indices],
        1e-8,
        1.0,
    )
    current_loss = -np.log(current_target_probability)
    rho_loss = current_loss - irreducible_loss
    (
        quality_indices,
        rho_indices,
        random_indices,
        quality_within_class_rank,
        rho_within_class_rank,
    ) = selection_indices(
        train_rows,
        observed,
        irreducible_loss,
        rho_loss,
        args.keep_fraction,
    )
    all_indices = np.arange(len(train_rows), dtype=np.int64)
    dev_targets = np.array([row["target_interest"] for row in dev_rows], dtype=object)
    random_metrics = fit_and_evaluate(
        train_features,
        observed,
        random_indices,
        dev_features,
        dev_targets,
        labels,
        args.max_iter,
    )
    quality_metrics = fit_and_evaluate(
        train_features,
        observed,
        quality_indices,
        dev_features,
        dev_targets,
        labels,
        args.max_iter,
    )
    rho_metrics = fit_and_evaluate(
        train_features,
        observed,
        rho_indices,
        dev_features,
        dev_targets,
        labels,
        args.max_iter,
    )
    noise_rank = np.argsort(-irreducible_loss)
    predicted_noise = noise_rank[: int(corrupted.sum())]
    metrics = {
        "schema_version": 1,
        "seed": SEED,
        "view": args.view,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "label_count": len(labels),
        "per_class": args.per_class,
        "folds": args.folds,
        "noise_rate": args.noise_rate,
        "synthetic_corrupted_rows": int(corrupted.sum()),
        "keep_fraction": args.keep_fraction,
        "selection_method": args.selection_method,
        "low_resource_fraction": args.current_fraction,
        "mean_oof_probability_observed": float(observed_probability.mean()),
        "mean_low_resource_probability_observed": float(
            current_target_probability.mean()
        ),
        "mean_learning_progress": float(rho_loss.mean()),
        "noise_detection": (
            {
                "clean_auroc": float(roc_auc_score(~corrupted, -irreducible_loss)),
                "noise_average_precision": float(
                    average_precision_score(corrupted, irreducible_loss)
                ),
                "noise_precision_at_true_count": float(corrupted[predicted_noise].mean()),
                "mean_loss_clean": float(irreducible_loss[~corrupted].mean()),
                "mean_loss_corrupted": float(irreducible_loss[corrupted].mean()),
                "random_selected_noise_rate": float(corrupted[random_indices].mean()),
                "oof_selected_noise_rate": float(corrupted[quality_indices].mean()),
                "rho_selected_noise_rate": float(corrupted[rho_indices].mean()),
            }
            if corrupted.any()
            else None
        ),
        "downstream": {
            "full": fit_and_evaluate(
                train_features,
                observed,
                all_indices,
                dev_features,
                dev_targets,
                labels,
                args.max_iter,
            ),
            "random_balanced": random_metrics,
            "oof_low_loss_balanced": quality_metrics,
            "cross_fitted_learning_progress_balanced": rho_metrics,
            "oof_minus_random": {
                name: quality_metrics[name] - random_metrics[name]
                for name in random_metrics
            },
            "learning_progress_minus_random": {
                name: rho_metrics[name] - random_metrics[name]
                for name in random_metrics
            },
            "learning_progress_minus_oof_low_loss": {
                name: rho_metrics[name] - quality_metrics[name]
                for name in random_metrics
            },
        },
    }
    selection_rows = []
    class_order = {category: index for index, category in enumerate(labels)}
    selected_within_class_rank = (
        quality_within_class_rank
        if args.selection_method == "oof_low_loss"
        else rho_within_class_rank
    )
    for index, row in enumerate(train_rows):
        nested_rank = (
            (selected_within_class_rank[index] - 1) * len(labels)
            + class_order[observed[index]]
            + 1
        )
        selection_rows.append(
            {
                "id": row["id"],
                "clean_target_interest": clean[index],
                "observed_target_interest": observed[index],
                "synthetic_corrupted": bool(corrupted[index]),
                "oof_target_probability": float(observed_probability[index]),
                "irreducible_loss": float(irreducible_loss[index]),
                "low_resource_target_probability": float(
                    current_target_probability[index]
                ),
                "low_resource_loss": float(current_loss[index]),
                "learning_progress": float(rho_loss[index]),
                "oof_low_loss_within_class_rank": int(
                    quality_within_class_rank[index]
                ),
                "learning_progress_within_class_rank": int(
                    rho_within_class_rank[index]
                ),
                "selection_method": args.selection_method,
                "within_class_rank": int(selected_within_class_rank[index]),
                "nested_rank": int(nested_rank),
            }
        )
    pq.write_table(
        pa.Table.from_pylist(selection_rows),
        args.output_dir / "selection_scores.parquet",
        compression="zstd",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()