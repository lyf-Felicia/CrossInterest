#!/usr/bin/env python3
"""Search stronger cross-user OOF teachers for fixed-budget denoising."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from crossinterest.experiments.denoising.proxy_iterate_selection import class_indices, relative_delta, select_anchor_coverage, select_by_loss_window
from crossinterest.experiments.denoising.proxy_pilot import SEED, VIEWS, classifier, fit_and_evaluate, render_features, select_balanced, stable_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-score-file",
        type=Path,
        help="Cached proxy_pilot selection_scores.parquet",
    )
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--keep-fraction", type=float, default=0.714)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--teacher-max-iter", type=int)
    parser.add_argument(
        "--teacher-model",
        choices=("sgd", "multinomial_nb", "complement_nb"),
        default="sgd",
    )
    parser.add_argument("--teacher-alpha", type=float, default=1.0)
    parser.add_argument(
        "--weight-temperatures",
        default="0.5,1,2,3",
        help="Comma-separated positive rank-weight temperatures",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("full", "weighted_cogs"),
        default="full",
        help="Limit teacher sweeps to the surviving weighted COGS branch",
    )
    parser.add_argument(
        "--margin-blends",
        default="",
        help="Comma-separated margin-rank fractions for hybrid reliability weights",
    )
    parser.add_argument(
        "--hierarchy-blends",
        default="",
        help="Comma-separated target-parent rank fractions for hierarchical scores",
    )
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument(
        "--variants",
        default="temporal_parent,temporal_full,temporal_combined",
        help="Comma-separated teacher feature variants",
    )
    return parser.parse_args()


def token(value: str) -> str:
    return value.lower().replace(" ", "_")


def temporal_interest_features(row: dict, include_full: bool) -> str:
    interests = row["historical_interests"]
    parents = [interest.split(" > ", 1)[0] for interest in interests]
    parts = []
    parent_counts = Counter(parents)
    for parent, count in parent_counts.items():
        parts.extend([f"parent={token(parent)}"] * min(count, 4))
    for reverse_position, (interest, parent) in enumerate(
        zip(reversed(interests), reversed(parents), strict=True), 1
    ):
        if reverse_position <= 5:
            parts.append(f"r{reverse_position}_parent={token(parent)}")
            if include_full:
                parts.append(f"r{reverse_position}_interest={token(interest)}")
        if include_full:
            parts.append(f"interest={token(interest)}")
    if len(parents) >= 2:
        parts.append(f"last_transition={token(parents[-2])}__{token(parents[-1])}")
    if parents:
        dominant = max(
            parent_counts,
            key=lambda parent: (parent_counts[parent], parents[::-1].index(parent)),
        )
        parts.append(f"dominant_parent={token(dominant)}")
        parts.append(f"last_dominant={token(parents[-1])}__{token(dominant)}")
    length = len(interests)
    length_bin = "5-6" if length <= 6 else "7-10" if length <= 10 else "11-15" if length <= 15 else "16+"
    parts.append(f"history_length={length_bin}")
    return " ".join(parts)


def teacher_features(row: dict, view: str, variant: str) -> str:
    if variant == "baseline":
        return render_features(row, view)
    temporal = temporal_interest_features(
        row, include_full=variant != "temporal_parent"
    )
    if variant in {"temporal_parent", "temporal_full"}:
        return temporal
    if variant == "temporal_combined":
        return f"{render_features(row, view)} temporal_break {temporal}"
    raise ValueError(f"unknown teacher variant: {variant}")


def within_class_percentile(
    grouped: dict[str, list[int]], losses: np.ndarray, row_ids: list[str]
) -> np.ndarray:
    percentiles = np.zeros(len(losses), dtype=np.float64)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: (losses[index], row_ids[index]))
        denominator = max(1, len(ordered) - 1)
        for rank, index in enumerate(ordered):
            percentiles[index] = rank / denominator
    return percentiles


def selected_rank_weights(
    selected_indices: np.ndarray,
    rank_percentiles: np.ndarray,
    temperature: float,
) -> np.ndarray:
    weights = np.exp(-temperature * rank_percentiles[selected_indices])
    return weights / weights.mean()


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def cross_user_oof_scores(
    rows: list[dict],
    features,
    targets: np.ndarray,
    labels: list[str],
    folds: int,
    max_iter: int,
    model_name: str,
    alpha: float,
) -> dict[str, np.ndarray]:
    groups = np.asarray([row["user_id"] for row in rows], dtype=object)
    label_to_index = {label: index for index, label in enumerate(labels)}
    target_indices = np.asarray([label_to_index[target] for target in targets])
    target_probability = np.zeros(len(rows), dtype=np.float32)
    target_rank = np.zeros(len(rows), dtype=np.int16)
    competitor_margin = np.zeros(len(rows), dtype=np.float32)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=SEED
    )
    dummy = np.zeros(len(rows), dtype=np.int8)
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(dummy, targets, groups), 1
    ):
        if model_name == "sgd":
            model = classifier(max_iter)
        elif model_name == "multinomial_nb":
            model = MultinomialNB(alpha=alpha)
        elif model_name == "complement_nb":
            model = ComplementNB(alpha=alpha)
        else:
            raise ValueError(f"unknown teacher model: {model_name}")
        model.fit(features[train_indices], targets[train_indices])
        probabilities = model.predict_proba(features[heldout_indices])
        class_columns = {
            category: index for index, category in enumerate(model.classes_)
        }
        columns = np.asarray(
            [class_columns[targets[index]] for index in heldout_indices]
        )
        row_numbers = np.arange(len(heldout_indices))
        target_values = probabilities[row_numbers, columns]
        competitors = probabilities.copy()
        competitors[row_numbers, columns] = -np.inf
        target_probability[heldout_indices] = target_values
        target_rank[heldout_indices] = 1 + np.sum(
            probabilities > target_values[:, None], axis=1
        )
        competitor_margin[heldout_indices] = (
            np.max(competitors, axis=1) - target_values
        )
        print(
            f"fold={fold}/{folds} train={len(train_indices)} "
            f"heldout={len(heldout_indices)}",
            flush=True,
        )
    if np.any(target_probability <= 0) or np.any(target_rank <= 0):
        raise AssertionError("OOF scoring did not cover every row")
    loss = -np.log(np.clip(target_probability, 1e-8, 1.0))
    rank_loss = target_rank.astype(np.float64) + within_class_percentile(
        class_indices(targets), loss, [row["id"] for row in rows]
    )
    return {
        "loss": loss,
        "target_rank": target_rank.astype(np.float64),
        "competitor_margin": competitor_margin.astype(np.float64),
        "rank_loss": rank_loss,
    }


def main() -> None:
    args = parse_args()
    weight_temperatures = [
        float(value) for value in args.weight_temperatures.split(",") if value
    ]
    if not weight_temperatures or any(value <= 0 for value in weight_temperatures):
        raise ValueError("--weight-temperatures must contain positive values")
    margin_blends = [
        float(value) for value in args.margin_blends.split(",") if value
    ]
    if any(not 0 < value < 1 for value in margin_blends):
        raise ValueError("--margin-blends values must be in (0, 1)")
    hierarchy_blends = [
        float(value) for value in args.hierarchy_blends.split(",") if value
    ]
    if any(not 0 < value < 1 for value in hierarchy_blends):
        raise ValueError("--hierarchy-blends values must be in (0, 1)")
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    allowed = {"baseline", "temporal_parent", "temporal_full", "temporal_combined"}
    unknown = set(variants) - allowed
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    train_columns = [
        "id",
        "user_id",
        "historical_titles",
        "historical_interests",
        "target_interest",
    ]
    rows = pq.read_table(args.train_file, columns=train_columns).to_pylist()
    rows = select_balanced(rows, args.per_class)
    labels = sorted({row["target_interest"] for row in rows})
    targets = np.asarray([row["target_interest"] for row in rows], dtype=object)
    row_ids = [row["id"] for row in rows]
    grouped = class_indices(targets)
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
    downstream_vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    train_downstream = downstream_vectorizer.transform(
        [render_features(row, args.view) for row in rows]
    )
    dev_downstream = downstream_vectorizer.transform(
        [render_features(row, args.view) for row in dev_rows]
    )
    random_indices = np.asarray(
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
    random_metrics = fit_and_evaluate(
        train_downstream,
        targets,
        random_indices,
        dev_downstream,
        dev_targets,
        labels,
        args.max_iter,
    )
    results = {
        "random": {
            "metrics": random_metrics,
            "relative_delta_percent_vs_random": relative_delta(
                random_metrics, random_metrics
            ),
        }
    }
    teacher_losses: dict[str, np.ndarray] = {}
    if args.baseline_score_file:
        cached = {
            row["id"]: row["irreducible_loss"]
            for row in pq.read_table(
                args.baseline_score_file, columns=["id", "irreducible_loss"]
            ).to_pylist()
        }
        if set(cached) != set(row_ids):
            raise ValueError("cached baseline score IDs do not match proxy rows")
        teacher_losses["baseline"] = np.asarray(
            [cached[row_id] for row_id in row_ids], dtype=np.float64
        )
    for variant in variants:
        teacher_tag = (
            variant
            if args.teacher_model == "sgd"
            else f"{variant}_{args.teacher_model}_a{args.teacher_alpha:g}"
        )
        print(f"scoring teacher={teacher_tag}", flush=True)
        teacher_vectorizer = HashingVectorizer(
            n_features=args.n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        teacher_matrix = teacher_vectorizer.transform(
            [teacher_features(row, args.view, variant) for row in rows]
        )
        score_variants = cross_user_oof_scores(
            rows,
            teacher_matrix,
            targets,
            labels,
            args.folds,
            args.teacher_max_iter or args.max_iter,
            args.teacher_model,
            args.teacher_alpha,
        )
        hierarchy_scores = None
        if hierarchy_blends:
            parent_targets = np.asarray(
                [target.split(" > ", 1)[0] for target in targets], dtype=object
            )
            parent_labels = sorted(set(parent_targets))
            print(
                f"scoring hierarchy teacher={teacher_tag} "
                f"parents={len(parent_labels)}",
                flush=True,
            )
            hierarchy_scores = cross_user_oof_scores(
                rows,
                teacher_matrix,
                parent_targets,
                parent_labels,
                args.folds,
                args.teacher_max_iter or args.max_iter,
                args.teacher_model,
                args.teacher_alpha,
            )
        losses = score_variants["loss"]
        teacher_losses[teacher_tag] = losses
        rank_percentiles = within_class_percentile(grouped, losses, row_ids)
        selections = {
            f"{teacher_tag}_cogs": select_by_loss_window(
                grouped, losses, row_ids, keep, 0.0
            )
        }
        if args.evaluation_mode == "full":
            selections[f"{teacher_tag}_parent_quarter"] = select_anchor_coverage(
                rows,
                grouped,
                losses,
                row_ids,
                keep,
                "last_parent",
                0.25,
            )
            for score_name, score_values in score_variants.items():
                if score_name == "loss":
                    continue
                selections[f"{teacher_tag}_{score_name}"] = select_by_loss_window(
                    grouped, score_values, row_ids, keep, 0.0
                )
            for window_percent in (2, 5):
                selections[f"{teacher_tag}_window{window_percent:02d}"] = (
                    select_by_loss_window(
                        grouped,
                        losses,
                        row_ids,
                        keep,
                        window_percent / 100.0,
                    )
                )
        for name, indices in selections.items():
            print(f"evaluating={name}", flush=True)
            metrics = fit_and_evaluate(
                train_downstream,
                targets,
                indices,
                dev_downstream,
                dev_targets,
                labels,
                args.max_iter,
            )
            results[name] = {
                "metrics": metrics,
                "relative_delta_percent_vs_random": relative_delta(
                    metrics, random_metrics
                ),
            }
            partial = {
                "schema_version": 1,
                "status": "partial",
                "view": args.view,
                "teacher_variants": variants,
                "methods": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(
                json.dumps(partial, indent=2, sort_keys=True) + "\n"
            )
            temporary.replace(args.output)
        cogs_indices = selections[f"{teacher_tag}_cogs"]
        for temperature in weight_temperatures:
            weights = selected_rank_weights(
                cogs_indices, rank_percentiles, temperature
            )
            name = f"{teacher_tag}_cogs_weight{temperature:g}"
            print(f"evaluating={name}", flush=True)
            metrics = fit_and_evaluate(
                train_downstream,
                targets,
                cogs_indices,
                dev_downstream,
                dev_targets,
                labels,
                args.max_iter,
                sample_weights=weights,
            )
            results[name] = {
                "metrics": metrics,
                "relative_delta_percent_vs_random": relative_delta(
                    metrics, random_metrics
                ),
                "weight_mean": float(weights.mean()),
                "effective_sample_size": effective_sample_size(weights),
            }
        if margin_blends:
            margin_percentiles = within_class_percentile(
                grouped,
                score_variants["competitor_margin"],
                row_ids,
            )
            for margin_fraction in margin_blends:
                hybrid_rank = (
                    (1.0 - margin_fraction) * rank_percentiles
                    + margin_fraction * margin_percentiles
                )
                margin_name = round(100 * margin_fraction)
                for temperature in weight_temperatures:
                    weights = selected_rank_weights(
                        cogs_indices, hybrid_rank, temperature
                    )
                    name = (
                        f"{teacher_tag}_cogs_hybrid_margin{margin_name}"
                        f"_weight{temperature:g}"
                    )
                    print(f"evaluating={name}", flush=True)
                    metrics = fit_and_evaluate(
                        train_downstream,
                        targets,
                        cogs_indices,
                        dev_downstream,
                        dev_targets,
                        labels,
                        args.max_iter,
                        sample_weights=weights,
                    )
                    results[name] = {
                        "metrics": metrics,
                        "relative_delta_percent_vs_random": relative_delta(
                            metrics, random_metrics
                        ),
                        "weight_mean": float(weights.mean()),
                        "effective_sample_size": effective_sample_size(weights),
                    }
        if hierarchy_scores is not None:
            parent_percentiles = within_class_percentile(
                grouped, hierarchy_scores["loss"], row_ids
            )
            for parent_fraction in hierarchy_blends:
                hierarchical_rank = (
                    (1.0 - parent_fraction) * rank_percentiles
                    + parent_fraction * parent_percentiles
                )
                parent_name = round(100 * parent_fraction)
                indices = select_by_loss_window(
                    grouped, hierarchical_rank, row_ids, keep, 0.0
                )
                name = f"{teacher_tag}_hierarchy_parent{parent_name}"
                print(f"evaluating={name}", flush=True)
                metrics = fit_and_evaluate(
                    train_downstream,
                    targets,
                    indices,
                    dev_downstream,
                    dev_targets,
                    labels,
                    args.max_iter,
                )
                results[name] = {
                    "metrics": metrics,
                    "relative_delta_percent_vs_random": relative_delta(
                        metrics, random_metrics
                    ),
                }
                for temperature in weight_temperatures:
                    weights = selected_rank_weights(
                        indices, hierarchical_rank, temperature
                    )
                    weighted_name = f"{name}_weight{temperature:g}"
                    print(f"evaluating={weighted_name}", flush=True)
                    weighted_metrics = fit_and_evaluate(
                        train_downstream,
                        targets,
                        indices,
                        dev_downstream,
                        dev_targets,
                        labels,
                        args.max_iter,
                        sample_weights=weights,
                    )
                    results[weighted_name] = {
                        "metrics": weighted_metrics,
                        "relative_delta_percent_vs_random": relative_delta(
                            weighted_metrics, random_metrics
                        ),
                        "weight_mean": float(weights.mean()),
                        "effective_sample_size": effective_sample_size(weights),
                    }
    if "baseline" in teacher_losses and "temporal_combined" in teacher_losses:
        baseline_rank = within_class_percentile(
            grouped, teacher_losses["baseline"], row_ids
        )
        temporal_rank = within_class_percentile(
            grouped, teacher_losses["temporal_combined"], row_ids
        )
        for baseline_weight in (0.2, 0.4, 0.6, 0.8):
            blend = (
                baseline_weight * baseline_rank
                + (1.0 - baseline_weight) * temporal_rank
            )
            weight_name = round(100 * baseline_weight)
            for window_percent in (0, 2, 5):
                name = f"consensus_b{weight_name}_t{100-weight_name}_window{window_percent:02d}"
                indices = select_by_loss_window(
                    grouped,
                    blend,
                    row_ids,
                    keep,
                    window_percent / 100.0,
                )
                print(f"evaluating={name}", flush=True)
                metrics = fit_and_evaluate(
                    train_downstream,
                    targets,
                    indices,
                    dev_downstream,
                    dev_targets,
                    labels,
                    args.max_iter,
                )
                results[name] = {
                    "metrics": metrics,
                    "relative_delta_percent_vs_random": relative_delta(
                        metrics, random_metrics
                    ),
                }
                if window_percent == 0 and baseline_weight in (0.2, 0.4):
                    for temperature in weight_temperatures:
                        weights = selected_rank_weights(
                            indices, blend, temperature
                        )
                        weighted_name = f"{name}_weight{temperature:g}"
                        print(f"evaluating={weighted_name}", flush=True)
                        weighted_metrics = fit_and_evaluate(
                            train_downstream,
                            targets,
                            indices,
                            dev_downstream,
                            dev_targets,
                            labels,
                            args.max_iter,
                            sample_weights=weights,
                        )
                        results[weighted_name] = {
                            "metrics": weighted_metrics,
                            "relative_delta_percent_vs_random": relative_delta(
                                weighted_metrics, random_metrics
                            ),
                            "weight_mean": float(weights.mean()),
                            "effective_sample_size": effective_sample_size(weights),
                        }
    ranked = sorted(
        results,
        key=lambda name: (
            results[name]["relative_delta_percent_vs_random"]["macro_recall_at_20"]
            + results[name]["relative_delta_percent_vs_random"]["macro_ndcg_at_20"]
        ),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "status": "complete",
        "view": args.view,
        "train_rows": len(rows),
        "selected_rows": len(random_indices),
        "keep_fraction": args.keep_fraction,
        "teacher_variants": variants,
        "ranking": ranked,
        "methods": results,
    }
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