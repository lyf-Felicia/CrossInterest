#!/usr/bin/env python3
"""Summarize formal Amazon evaluation progress and completed metrics."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
from pathlib import Path

try:
    from crossinterest.evaluation.common import INPUT_TYPES, latest_details, write_json_atomic
except ImportError:
    from common import INPUT_TYPES, latest_details, write_json_atomic


HERE = _source_workspace('eval_amazon_notrain')
REGISTRY_PATH = HERE / "models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs" / "formal")
    parser.add_argument("--output", type=Path, default=HERE / "PROGRESS.json")
    return parser.parse_args()


def summarize_view(run_dir: Path) -> dict:
    details = latest_details(run_dir / "details.jsonl")
    successful = sum(row.get("error") is None for row in details.values())
    errors = len(details) - successful
    retries = sum(row.get("attempts", 1) > 1 for row in details.values())
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    requested = int(metrics.get("requested_sample_count", 2_000))
    return {
        "covered": len(details),
        "requested": requested,
        "progress": len(details) / requested,
        "successful": successful,
        "errors": errors,
        "retried_successes": retries,
        "complete": bool(metrics.get("complete", False)),
        "macro_accuracy": metrics.get("macro_accuracy"),
        "micro_accuracy": metrics.get("micro_accuracy"),
        "strict_format_rate": metrics.get("strict_format_rate"),
        "resolved_prediction_rate": metrics.get("resolved_prediction_rate"),
    }


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 1,
        "run_root": str(args.run_root),
        "models": {},
    }
    total_covered = 0
    total_requested = 0
    for model_key, model in registry["models"].items():
        views = {
            input_type: summarize_view(args.run_root / model_key / input_type)
            for input_type in INPUT_TYPES
        }
        summary["models"][model_key] = {
            "display_name": model["display_name"],
            "backend": model["backend"],
            "views": views,
        }
        total_covered += sum(view["covered"] for view in views.values())
        total_requested += sum(view["requested"] for view in views.values())
    summary["total"] = {
        "covered": total_covered,
        "requested": total_requested,
        "progress": total_covered / total_requested,
        "complete_views": sum(
            view["complete"]
            for model in summary["models"].values()
            for view in model["views"].values()
        ),
        "view_count": len(summary["models"]) * len(INPUT_TYPES),
    }
    write_json_atomic(args.output, summary)
    print(
        "Model | Input | Progress | Errors | Retries | Macro | Strict format | Complete"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---")
    for model in summary["models"].values():
        for input_type, view in model["views"].items():
            macro = "-" if view["macro_accuracy"] is None else f"{view['macro_accuracy']:.4f}"
            strict = (
                "-"
                if view["strict_format_rate"] is None
                else f"{view['strict_format_rate']:.4f}"
            )
            print(
                f"{model['display_name']} | {input_type} | "
                f"{view['covered']}/{view['requested']} | {view['errors']} | "
                f"{view['retried_successes']} | {macro} | {strict} | {view['complete']}"
            )
    print(
        f"TOTAL {total_covered}/{total_requested} "
        f"({summary['total']['progress']:.2%}), "
        f"complete views {summary['total']['complete_views']}/{summary['total']['view_count']}"
    )


if __name__ == "__main__":
    main()