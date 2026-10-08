#!/usr/bin/env python3
"""Compare class-balanced selection policies using cached OOF scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer

from crossinterest.experiments.denoising.proxy_pilot import VIEWS, fit_and_evaluate, render_features, select_balanced, stable_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--keep-fraction", type=float, default=0.714)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument(
        "--methods",
        help="Comma-separated candidate names; random is always included",
    )
    return parser.parse_args()


def class_indices(observed: np.ndarray) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(observed):
        grouped[str(label)].append(index)
    return grouped


def exact_allocation(
    sizes: dict[str, int],
    total: int,
    power: float,
) -> dict[str, int]:
    if total > sum(sizes.values()):
        raise ValueError("allocation exceeds available rows")
    weights = {key: size**power for key, size in sizes.items()}
    weight_sum = sum(weights.values())
    raw = {key: total * weight / weight_sum for key, weight in weights.items()}
    allocation = {
        key: min(size, int(np.floor(raw[key]))) for key, size in sizes.items()
    }
    while sum(allocation.values()) < total:
        candidates = [
            key for key, size in sizes.items() if allocation[key] < size
        ]
        key = max(
            candidates,
            key=lambda item: (
                raw[item] - allocation[item],
                sizes[item] - allocation[item],
                item,
            ),
        )
        allocation[key] += 1
    return allocation


def anchor(row: dict, mode: str) -> str:
    interests = row["historical_interests"]
    parents = [value.split(" > ", 1)[0] for value in interests]
    if mode == "last_interest":
        return interests[-1] if interests else "<empty>"
    if mode == "last_parent":
        return parents[-1] if parents else "<empty>"
    if mode == "dominant_parent":
        if not parents:
            return "<empty>"
        counts = {parent: parents.count(parent) for parent in set(parents)}
        return max(counts, key=lambda parent: (counts[parent], parents[::-1].index(parent)))
    if mode == "last_parent_length":
        length = len(interests)
        length_bin = "5-6" if length <= 6 else "7-10" if length <= 10 else "11-15" if length <= 15 else "16+"
        return f"{parents[-1] if parents else '<empty>'}|{length_bin}"
    if mode == "last_dominant_parent":
        dominant = anchor(row, "dominant_parent")
        return f"{parents[-1] if parents else '<empty>'}|{dominant}"
    raise ValueError(f"unknown anchor mode: {mode}")


def select_by_loss_window(
    grouped: dict[str, list[int]],
    losses: np.ndarray,
    row_ids: list[str],
    keep: int,
    start_fraction: float,
) -> np.ndarray:
    selected = []
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: (losses[index], row_ids[index]))
        start = min(round(len(indices) * start_fraction), len(indices) - keep)
        selected.extend(ordered[start : start + keep])
    return np.asarray(sorted(selected), dtype=np.int64)


def select_quality_random_mix(
    grouped: dict[str, list[int]],
    losses: np.ndarray,
    row_ids: list[str],
    keep: int,
    quality_fraction: float,
) -> np.ndarray:
    selected = []
    quality_count = round(keep * quality_fraction)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: (losses[index], row_ids[index]))
        quality = ordered[:quality_count]
        quality_set = set(quality)
        remainder = sorted(
            (index for index in indices if index not in quality_set),
            key=lambda index: stable_key("iterate-random-tail", row_ids[index]),
        )[: keep - quality_count]
        selected.extend(quality + remainder)
    return np.asarray(sorted(selected), dtype=np.int64)


def select_anchor_coverage(
    rows: list[dict],
    grouped: dict[str, list[int]],
    losses: np.ndarray,
    row_ids: list[str],
    keep: int,
    anchor_mode: str,
    allocation_power: float,
) -> np.ndarray:
    selected = []
    for indices in grouped.values():
        by_anchor: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_anchor[anchor(rows[index], anchor_mode)].append(index)
        allocation = exact_allocation(
            {key: len(values) for key, values in by_anchor.items()},
            keep,
            allocation_power,
        )
        for key, anchor_indices in by_anchor.items():
            ordered = sorted(
                anchor_indices,
                key=lambda index: (losses[index], row_ids[index]),
            )
            selected.extend(ordered[: allocation[key]])
    return np.asarray(sorted(selected), dtype=np.int64)


def select_cluster_coverage(
    train_features,
    grouped: dict[str, list[int]],
    losses: np.ndarray,
    row_ids: list[str],
    keep: int,
    clusters: int,
    allocation_power: float,
) -> np.ndarray:
    selected = []
    for class_number, indices in enumerate(grouped.values()):
        model = MiniBatchKMeans(
            n_clusters=min(clusters, len(indices)),
            batch_size=len(indices),
            n_init=5,
            max_iter=100,
            random_state=20260901 + class_number,
        )
        assignments = model.fit_predict(train_features[indices])
        by_cluster: dict[str, list[int]] = defaultdict(list)
        for index, assignment in zip(indices, assignments, strict=True):
            by_cluster[str(assignment)].append(index)
        allocation = exact_allocation(
            {key: len(values) for key, values in by_cluster.items()},
            keep,
            allocation_power,
        )
        for key, cluster_indices in by_cluster.items():
            ordered = sorted(
                cluster_indices,
                key=lambda index: (losses[index], row_ids[index]),
            )
            selected.extend(ordered[: allocation[key]])
    return np.asarray(sorted(selected), dtype=np.int64)


def select_mmr(
    train_features,
    grouped: dict[str, list[int]],
    losses: np.ndarray,
    row_ids: list[str],
    keep: int,
    quality_weight: float,
) -> np.ndarray:
    if not 0 <= quality_weight <= 1:
        raise ValueError("quality_weight must be in [0, 1]")
    selected = []
    for indices in grouped.values():
        ordered_positions = sorted(
            range(len(indices)),
            key=lambda position: (losses[indices[position]], row_ids[indices[position]]),
        )
        quality = np.empty(len(indices), dtype=np.float32)
        quality[ordered_positions] = np.linspace(1.0, 0.0, len(indices), dtype=np.float32)
        similarity = (train_features[indices] @ train_features[indices].T).toarray()
        chosen = [ordered_positions[0]]
        chosen_mask = np.zeros(len(indices), dtype=bool)
        chosen_mask[chosen[0]] = True
        max_similarity = similarity[:, chosen[0]].copy()
        while len(chosen) < keep:
            utility = quality_weight * quality + (1.0 - quality_weight) * (
                1.0 - max_similarity
            )
            utility[chosen_mask] = -np.inf
            position = int(np.argmax(utility))
            chosen.append(position)
            chosen_mask[position] = True
            np.maximum(max_similarity, similarity[:, position], out=max_similarity)
        selected.extend(indices[position] for position in chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def relative_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict:
    return {
        key: 100.0 * (metrics[key] - baseline[key]) / baseline[key]
        for key in baseline
    }


def main() -> None:
    args = parse_args()
    requested = set(args.methods.split(",")) if args.methods else None
    if not 0 < args.keep_fraction <= 1:
        raise ValueError("--keep-fraction must be in (0, 1]")
    train_columns = [
        "id",
        "historical_titles",
        "historical_interests",
        "target_interest",
    ]
    rows = pq.read_table(args.train_file, columns=train_columns).to_pylist()
    rows = select_balanced(rows, args.per_class)
    scores = {
        row["id"]: row
        for row in pq.read_table(
            args.score_file,
            columns=[
                "id",
                "observed_target_interest",
                "synthetic_corrupted",
                "irreducible_loss",
            ],
        ).to_pylist()
    }
    row_ids = [row["id"] for row in rows]
    if set(row_ids) != set(scores):
        raise ValueError("cached score IDs do not match the selected proxy pool")
    observed = np.asarray(
        [scores[row_id]["observed_target_interest"] for row_id in row_ids],
        dtype=object,
    )
    losses = np.asarray(
        [scores[row_id]["irreducible_loss"] for row_id in row_ids],
        dtype=np.float64,
    )
    corrupted = np.asarray(
        [scores[row_id]["synthetic_corrupted"] for row_id in row_ids],
        dtype=bool,
    )
    labels = sorted(set(observed))
    grouped = class_indices(observed)
    class_sizes = {len(indices) for indices in grouped.values()}
    if class_sizes != {args.per_class}:
        raise ValueError(f"observed class sizes differ: {sorted(class_sizes)}")
    keep = round(args.per_class * args.keep_fraction)

    dev_rows = pq.read_table(
        args.dev_file,
        columns=[
            "historical_titles",
            "historical_interests",
            "target_interest",
        ],
    ).to_pylist()
    dev_targets = np.asarray(
        [row["target_interest"] for row in dev_rows], dtype=object
    )
    if set(labels) != set(dev_targets):
        raise ValueError("train and dev target supports differ")
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    train_features = vectorizer.transform(
        [render_features(row, args.view) for row in rows]
    )
    dev_features = vectorizer.transform(
        [render_features(row, args.view) for row in dev_rows]
    )

    methods = {
        "cogs": select_by_loss_window(grouped, losses, row_ids, keep, 0.0),
        "loss_window_05": select_by_loss_window(
            grouped, losses, row_ids, keep, 0.05
        ),
        "loss_window_10": select_by_loss_window(
            grouped, losses, row_ids, keep, 0.10
        ),
        "loss_window_15": select_by_loss_window(
            grouped, losses, row_ids, keep, 0.15
        ),
        "loss_window_20": select_by_loss_window(
            grouped, losses, row_ids, keep, 0.20
        ),
        "loss_window_286": select_by_loss_window(
            grouped, losses, row_ids, keep, 1.0 - args.keep_fraction
        ),
        "quality50_random50": select_quality_random_mix(
            grouped, losses, row_ids, keep, 0.50
        ),
        "quality75_random25": select_quality_random_mix(
            grouped, losses, row_ids, keep, 0.75
        ),
        "last_interest_uniform": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_interest", 0.0
        ),
        "last_interest_sqrt": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_interest", 0.5
        ),
        "last_parent_uniform": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent", 0.0
        ),
        "last_parent_quarter": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent", 0.25
        ),
        "last_parent_sqrt": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent", 0.5
        ),
        "last_parent_three_quarter": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent", 0.75
        ),
        "last_parent_proportional": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent", 1.0
        ),
        "dominant_parent_quarter": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "dominant_parent", 0.25
        ),
        "dominant_parent_sqrt": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "dominant_parent", 0.5
        ),
        "last_parent_length_sqrt": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent_length", 0.5
        ),
        "last_parent_length_three_quarter": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_parent_length", 0.75
        ),
        "last_dominant_parent_sqrt": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_dominant_parent", 0.5
        ),
        "last_dominant_parent_three_quarter": select_anchor_coverage(
            rows, grouped, losses, row_ids, keep, "last_dominant_parent", 0.75
        ),
    }
    for clusters in (8, 16, 32):
        for power_name, power in (
            ("quarter", 0.25),
            ("sqrt", 0.5),
            ("three_quarter", 0.75),
        ):
            name = f"cluster{clusters}_{power_name}"
            if requested is not None and name not in requested:
                continue
            methods[name] = select_cluster_coverage(
                train_features,
                grouped,
                losses,
                row_ids,
                keep,
                clusters,
                power,
            )
    for quality_percent in (50, 65, 75, 85, 90, 95):
        name = f"mmr_quality{quality_percent}"
        if requested is not None and name not in requested:
            continue
        methods[name] = select_mmr(
            train_features,
            grouped,
            losses,
            row_ids,
            keep,
            quality_percent / 100.0,
        )
    methods["random"] = np.asarray(
        sorted(
            index
            for indices in grouped.values()
            for index in sorted(
                indices,
                key=lambda item: stable_key("random-select", row_ids[item]),
            )[:keep]
        ),
        dtype=np.int64,
    )
    if requested is not None:
        unknown = requested - set(methods)
        if unknown:
            raise ValueError(f"unknown --methods: {sorted(unknown)}")
        methods = {
            name: indices
            for name, indices in methods.items()
            if name == "random" or name in requested
        }
    methods = {"random": methods.pop("random"), **methods}
    expected_selected = len(labels) * keep
    for name, indices in methods.items():
        if len(indices) != expected_selected or len(set(indices.tolist())) != len(indices):
            raise AssertionError(f"{name}: invalid selected row count")

    results = {}
    for name, indices in methods.items():
        print(f"evaluating={name} rows={len(indices)}", flush=True)
        metrics = fit_and_evaluate(
            train_features,
            observed,
            indices,
            dev_features,
            dev_targets,
            labels,
            args.max_iter,
        )
        results[name] = {
            "metrics": metrics,
            "selected_noise_rate": (
                float(corrupted[indices].mean()) if corrupted.any() else None
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": 1,
            "status": "partial",
            "view": args.view,
            "noise_rate": float(corrupted.mean()),
            "completed_methods": list(results),
            "methods": results,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(args.output)
    baseline = results["random"]["metrics"]
    for result in results.values():
        result["relative_delta_percent_vs_random"] = relative_delta(
            result["metrics"], baseline
        )
    ranked = sorted(
        results,
        key=lambda name: (
            results[name]["relative_delta_percent_vs_random"][
                "macro_recall_at_20"
            ]
            + results[name]["relative_delta_percent_vs_random"][
                "macro_ndcg_at_20"
            ],
            name,
        ),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "status": "complete",
        "view": args.view,
        "noise_rate": float(corrupted.mean()),
        "train_rows": len(rows),
        "selected_rows": expected_selected,
        "keep_fraction": args.keep_fraction,
        "target_relative_improvement_percent": 10.0,
        "ranking": ranked,
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for name in ranked:
        delta = results[name]["relative_delta_percent_vs_random"]
        print(
            f"{name}: R20={delta['macro_recall_at_20']:+.2f}% "
            f"nDCG20={delta['macro_ndcg_at_20']:+.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()