#!/usr/bin/env python3
"""Validate a completed MeRA training artifact before evaluation or reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossinterest.baselines.mera.core import validate_label_space
from crossinterest.baselines.mera.train import MODELS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--model-key", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=("arts", "grocery", "home"), required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--expected-context-labels", type=int, required=True)
    parser.add_argument("--expected-prediction-labels", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.resolve()
    adapter = train_dir / "final_adapter"
    required = [
        train_dir / "train_contract.json",
        train_dir / "train_metrics.json",
        train_dir / "semantic_token_setup.json",
        adapter / "adapter_config.json",
        adapter / "tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    weights = [
        path
        for path in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
        if path.is_file() and path.stat().st_size
    ]
    if missing or len(weights) != 1:
        raise ValueError(
            f"incomplete artifact: missing={missing}, adapter_weight_files={len(weights)}"
        )
    contract = json.loads((train_dir / "train_contract.json").read_text(encoding="utf-8"))
    metrics = json.loads((train_dir / "train_metrics.json").read_text(encoding="utf-8"))
    label_space = validate_label_space(
        json.loads((args.assets_dir / "label_space.json").read_text(encoding="utf-8"))
    )
    expected = {
        "model_key": args.model_key,
        "dataset": args.dataset,
        "label_space_fingerprint": label_space["fingerprint"],
        "label_count": args.expected_context_labels,
        "prediction_label_count": args.expected_prediction_labels,
        "history_representation": "same atomic category tokens as targets",
        "language_loss": "full-vocabulary next-token cross entropy",
        "precision": "bf16",
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"training contract mismatch: {mismatches}")
    if metrics.get("label_space_fingerprint") != contract["label_space_fingerprint"]:
        raise ValueError("train metrics and contract have different label-space fingerprints")
    print(
        json.dumps(
            {
                "valid": True,
                "model_key": args.model_key,
                "dataset": args.dataset,
                "adapter_weight": str(weights[0]),
                "adapter_weight_sha256": sha256_file(weights[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
