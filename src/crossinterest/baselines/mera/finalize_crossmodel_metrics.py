#!/usr/bin/env python3
"""Attach the actual base-model identity to validated official-formula metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossinterest.baselines.mera.core import write_json


def validated_crossmodel_method(method: str) -> str:
    allowed_methods = {
        "MeRA official method with model-architecture compatibility only",
        "MeRA author-clarified cross-model reproduction",
    }
    if method not in allowed_methods:
        raise ValueError(f"unexpected cross-model training contract: {method}")
    return method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--train-contract", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    contract = json.loads(args.train_contract.read_text(encoding="utf-8"))
    if metrics["paper_metrics"] != metrics["official_formula_metrics"]:
        raise ValueError("official and independent paper metrics do not match")
    metrics["method"] = validated_crossmodel_method(contract["method"])
    metrics["model"] = contract["model"]
    metrics["model_revision"] = contract["model_revision"]
    metrics["equivalence_scope"] = contract["equivalence_scope"]
    write_json(args.metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()