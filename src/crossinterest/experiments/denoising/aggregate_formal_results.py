#!/usr/bin/env python3
"""Validate and aggregate the matched 12-job COGS formal experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_SAMPLES = 2_000
EXPECTED_CLASSES = 112
EXPECTED_SELECTION_SIZE = 79_968
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_902
METRIC_FIELDS = (
    "macro_accuracy",
    "macro_recall_at_10",
    "macro_recall_at_20",
    "macro_ndcg_at_10",
    "macro_ndcg_at_20",
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    model_key: str
    model_label: str
    view: str
    view_label: str
    method: str
    selection_name: str


def build_specs() -> tuple[JobSpec, ...]:
    specs = []
    models = (
        ("qwen35", "qwen35_9b", "Qwen3.5-9B"),
        ("llama31", "llama31_8b", "Llama-3.1-8B-Instruct"),
    )
    views = (
        ("behavior", "behavior", "Behavior"),
        ("interest", "interest", "Interest"),
        ("bi", "behavior_interest", "B+I"),
    )
    methods = (
        ("random", "Random", "random_balanced.parquet"),
        ("oof-low-loss", "COGS", None),
    )
    for name_prefix, model_key, model_label in models:
        for job_view, view, view_label in views:
            for name_method, method, selection_name in methods:
                if selection_name is None:
                    selection_name = f"oof_low_loss_{view}.parquet"
                specs.append(
                    JobSpec(
                        name=f"{name_prefix}-{job_view}-{name_method}-79968",
                        model_key=model_key,
                        model_label=model_label,
                        view=view,
                        view_label=view_label,
                        method=method,
                        selection_name=selection_name,
                    )
                )
    return tuple(specs)


SPECS = build_specs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def find_job_roots(results_root: Path) -> dict[str, Path]:
    candidates = {spec.name: [] for spec in SPECS}
    for metrics_path in results_root.rglob("classification/metrics.json"):
        for spec in SPECS:
            if spec.name in metrics_path.parts:
                candidates[spec.name].append(metrics_path.parent.parent)
    problems = []
    roots = {}
    for name, matches in candidates.items():
        unique_matches = sorted(set(matches))
        if len(unique_matches) != 1:
            problems.append(f"{name}: expected one result root, found {len(unique_matches)}")
        else:
            roots[name] = unique_matches[0]
    if problems:
        raise ValueError("invalid downloaded result layout:\n" + "\n".join(problems))
    return roots


def require_equal(data: dict, field: str, expected: object, path: Path) -> None:
    if data.get(field) != expected:
        raise ValueError(
            f"{path}: {field}={data.get(field)!r}, expected {expected!r}"
        )


def validate_training(contract: dict, spec: JobSpec, path: Path) -> None:
    require_equal(contract, "task", "amazon_new_interest_candidate_free_sft", path)
    require_equal(contract, "candidate_free", True, path)
    require_equal(contract, "assistant_only", True, path)
    require_equal(contract, "label_mode", "interest_id", path)
    require_equal(contract, "label_count", EXPECTED_CLASSES, path)
    require_equal(contract, "view", spec.view, path)
    require_equal(contract, "selection_size", EXPECTED_SELECTION_SIZE, path)
    require_equal(contract, "selected_train_rows", EXPECTED_SELECTION_SIZE, path)
    require_equal(contract, "selected_label_count_min", 714, path)
    require_equal(contract, "selected_label_count_max", 714, path)
    require_equal(contract, "selected_label_count_max_gap", 0, path)
    require_equal(contract, "formal_train_rows", 112_000, path)
    require_equal(contract, "epochs", 1, path)
    require_equal(contract, "effective_batch_size", 32, path)
    selection_file = Path(str(contract.get("selection_file", ""))).name
    if selection_file != spec.selection_name:
        raise ValueError(
            f"{path}: selection file {selection_file!r}, expected {spec.selection_name!r}"
        )


def validate_common_metrics(metrics: dict, spec: JobSpec, path: Path) -> None:
    require_equal(metrics, "requested_sample_count", EXPECTED_SAMPLES, path)
    require_equal(metrics, "recorded_sample_count", EXPECTED_SAMPLES, path)
    require_equal(metrics, "missing_sample_count", 0, path)
    require_equal(metrics, "class_count", EXPECTED_CLASSES, path)
    require_equal(metrics, "complete", True, path)
    require_equal(metrics, "model_key", spec.model_key, path)
    require_equal(metrics, "input_type", spec.view, path)
    require_equal(metrics, "label_mode", "interest_id", path)


def validate_classification(metrics: dict, spec: JobSpec, path: Path) -> None:
    validate_common_metrics(metrics, spec, path)
    require_equal(metrics, "successful_sample_count", EXPECTED_SAMPLES, path)
    require_equal(metrics, "request_error_count", 0, path)
    require_equal(metrics, "strict_format_rate", 1.0, path)
    require_equal(metrics, "resolved_prediction_rate", 1.0, path)
    require_equal(metrics, "decoding", "candidate_constrained_beam_search", path)
    require_equal(metrics, "num_beams", 10, path)


def validate_ranking(metrics: dict, spec: JobSpec, path: Path) -> None:
    validate_common_metrics(metrics, spec, path)
    require_equal(metrics, "decoding", "candidate_free_direct_beam_search", path)
    require_equal(metrics, "num_beams", 20, path)
    for field in METRIC_FIELDS[1:]:
        if not isinstance(metrics.get(field), (int, float)):
            raise ValueError(f"{path}: missing numeric metric {field}")


def per_class_metrics(classification: dict, ranking: dict) -> dict[str, dict[str, float]]:
    values = {"macro_accuracy": classification["per_class_accuracy"]}
    for field in METRIC_FIELDS[1:]:
        class_field = field.removeprefix("macro_")
        values[field] = {
            label: metrics[class_field]
            for label, metrics in ranking["per_class_metrics"].items()
        }
    expected_labels = set(values["macro_accuracy"])
    if len(expected_labels) != EXPECTED_CLASSES:
        raise ValueError(
            f"per-class metrics contain {len(expected_labels)} labels, "
            f"expected {EXPECTED_CLASSES}"
        )
    for field, field_values in values.items():
        if set(field_values) != expected_labels:
            raise ValueError(f"per-class label set differs for {field}")
    return values


def paired_class_bootstrap(
    cogs: dict[str, float],
    random: dict[str, float],
    rng: np.random.Generator,
) -> dict:
    if set(cogs) != set(random):
        raise ValueError("COGS and Random per-class label sets differ")
    labels = sorted(cogs)
    differences = np.asarray(
        [float(cogs[label]) - float(random[label]) for label in labels],
        dtype=np.float64,
    )
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_RESAMPLES, len(differences)),
    )
    means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    nonpositive = int(np.count_nonzero(means <= 0.0))
    return {
        "resampling_unit": "target_class",
        "class_count": len(differences),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "mean_delta": float(differences.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_probability_nonpositive": (nonpositive + 1)
        / (BOOTSTRAP_RESAMPLES + 1),
        "positive_class_count": int(np.count_nonzero(differences > 0.0)),
        "negative_class_count": int(np.count_nonzero(differences < 0.0)),
        "tied_class_count": int(np.count_nonzero(differences == 0.0)),
    }


def collect(results_root: Path) -> dict:
    job_roots = find_job_roots(results_root)
    rows = []
    class_values = {}
    for spec in SPECS:
        job_root = job_roots[spec.name]
        training_path = job_root / "train" / "train_contract.json"
        classification_path = job_root / "classification" / "metrics.json"
        ranking_path = job_root / "ranking" / "metrics.json"
        training = load_json(training_path)
        classification = load_json(classification_path)
        ranking = load_json(ranking_path)
        validate_training(training, spec, training_path)
        validate_classification(classification, spec, classification_path)
        validate_ranking(ranking, spec, ranking_path)
        class_values[(spec.model_key, spec.view, spec.method)] = per_class_metrics(
            classification,
            ranking,
        )
        rows.append(
            {
                "job_name": spec.name,
                "model_key": spec.model_key,
                "model": spec.model_label,
                "view": spec.view,
                "view_label": spec.view_label,
                "method": spec.method,
                "selection_file_sha256": training["selection_file_sha256"],
                "macro_accuracy": classification["macro_accuracy"],
                "macro_recall_at_10": ranking["macro_recall_at_10"],
                "macro_recall_at_20": ranking["macro_recall_at_20"],
                "macro_ndcg_at_10": ranking["macro_ndcg_at_10"],
                "macro_ndcg_at_20": ranking["macro_ndcg_at_20"],
                "classification_strict_format_rate": classification[
                    "strict_format_rate"
                ],
                "ranking_valid_beam_rate": ranking["valid_beam_rate"],
                "ranking_strict_beam_format_rate": ranking[
                    "strict_beam_format_rate"
                ],
                "ranking_duplicate_valid_beam_rate": ranking[
                    "duplicate_valid_beam_rate"
                ],
            }
        )

    deltas = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for model_key in ("qwen35_9b", "llama31_8b"):
        for view in ("behavior", "interest", "behavior_interest"):
            pair = {
                row["method"]: row
                for row in rows
                if row["model_key"] == model_key and row["view"] == view
            }
            if set(pair) != {"COGS", "Random"}:
                raise ValueError(f"incomplete pair for {model_key}/{view}")
            bootstrap = {
                field: paired_class_bootstrap(
                    class_values[(model_key, view, "COGS")][field],
                    class_values[(model_key, view, "Random")][field],
                    rng,
                )
                for field in METRIC_FIELDS
            }
            deltas.append(
                {
                    "model_key": model_key,
                    "model": pair["COGS"]["model"],
                    "view": view,
                    "view_label": pair["COGS"]["view_label"],
                    **{
                        field: pair["COGS"][field] - pair["Random"][field]
                        for field in METRIC_FIELDS
                    },
                    "paired_class_bootstrap": bootstrap,
                }
            )
    return {
        "schema_version": 1,
        "expected_job_count": len(SPECS),
        "validated_job_count": len(rows),
        "jobs": rows,
        "cogs_minus_random": deltas,
        "positive_delta_counts": {
            field: sum(delta[field] > 0 for delta in deltas)
            for field in METRIC_FIELDS
        },
    }


def format_number(value: float, percentage: bool = False) -> str:
    scaled = value * 100 if percentage else value
    return f"{scaled:.2f}" if percentage else f"{scaled:.4f}"


def render_markdown(summary: dict) -> str:
    lines = [
        "| Model | View | Method | Macro Accuracy (%) | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["jobs"]:
        lines.append(
            "| {model} | {view_label} | {method} | {accuracy} | {r10} | {r20} | {n10} | {n20} |".format(
                model=row["model"],
                view_label=row["view_label"],
                method=row["method"],
                accuracy=format_number(row["macro_accuracy"], percentage=True),
                r10=format_number(row["macro_recall_at_10"]),
                r20=format_number(row["macro_recall_at_20"]),
                n10=format_number(row["macro_ndcg_at_10"]),
                n20=format_number(row["macro_ndcg_at_20"]),
            )
        )
    lines.extend(
        [
            "",
            "| Model | View | Accuracy delta (pp) | Recall@10 delta (pp) | Recall@20 delta (pp) | nDCG@10 delta (pp) | nDCG@20 delta (pp) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cogs_minus_random"]:
        lines.append(
            "| {model} | {view_label} | {accuracy} | {r10} | {r20} | {n10} | {n20} |".format(
                model=row["model"],
                view_label=row["view_label"],
                accuracy=f"{row['macro_accuracy'] * 100:+.2f}",
                r10=f"{row['macro_recall_at_10'] * 100:+.2f}",
                r20=f"{row['macro_recall_at_20'] * 100:+.2f}",
                n10=f"{row['macro_ndcg_at_10'] * 100:+.2f}",
                n20=f"{row['macro_ndcg_at_20'] * 100:+.2f}",
            )
        )
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise ValueError(f"results root is not a directory: {results_root}")
    summary = collect(results_root)
    output_json = args.output_json or results_root / "formal_summary.json"
    output_markdown = args.output_markdown or results_root / "formal_summary.md"
    write_atomic(output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_atomic(output_markdown, render_markdown(summary))
    print(
        json.dumps(
            {
                "validated_job_count": summary["validated_job_count"],
                "positive_delta_counts": summary["positive_delta_counts"],
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()