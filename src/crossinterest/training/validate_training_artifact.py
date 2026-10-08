#!/usr/bin/env python3
"""Validate that a completed training artifact matches the requested run identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.training.label_space import INTEREST_ID_LABEL_MODES, LABEL_MODES, load_label_mapping, prompt_schema_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--selection-size", type=int)
    parser.add_argument("--selection-weight-column")
    parser.add_argument("--semantic-label-token-weight", type=float, default=1.0)
    parser.add_argument("--expected-train-rows", type=int, required=True)
    parser.add_argument("--expected-label-count", type=int, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    contract_path = args.train_dir / "train_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    has_history_summary = "history_summary" in pq.ParquetFile(args.train_file).schema.names
    has_target_rationale = "target_rationale" in pq.ParquetFile(
        args.train_file
    ).schema.names
    has_target_reasoning = "target_reasoning" in pq.ParquetFile(
        args.train_file
    ).schema.names
    if has_target_rationale and has_target_reasoning:
        raise ValueError("training data cannot contain both rationale order columns")
    expected = {
        "task": "amazon_new_interest_candidate_free_sft",
        "prompt_schema_version": prompt_schema_version(
            args.label_mode,
            history_summary=has_history_summary,
            target_rationale=has_target_rationale,
            target_reasoning=has_target_reasoning,
        ),
        "history_summary": has_history_summary,
        "target_rationale_auxiliary": has_target_rationale,
        "reasoning_first": has_target_reasoning,
        "assistant_turn_count": (
            2 if has_target_rationale or has_target_reasoning else 1
        ),
        "candidate_free": True,
        "assistant_only": True,
        "label_mode": args.label_mode,
        "label_count": args.expected_label_count,
        "model": args.model,
        "view": args.view,
        "train_sha256": sha256_file(args.train_file),
        "selection_file": str(args.selection_file.resolve()) if args.selection_file else None,
        "selection_file_sha256": (
            sha256_file(args.selection_file) if args.selection_file else None
        ),
        "selection_size": args.selection_size,
        "selection_weight_column": args.selection_weight_column,
        "formal_train_rows": args.expected_train_rows,
        "sample_limit": None,
        "epochs": 1.0,
        "learning_rate": 1e-4,
        "semantic_label_token_weight": args.semantic_label_token_weight,
        "effective_batch_size": 32,
    }
    if args.label_mode in INTEREST_ID_LABEL_MODES:
        mapping = load_label_mapping(args.label_map, expected_count=args.expected_label_count)
        expected.update(
            {
                "label_map_sha256": sha256_file(args.label_map),
                "label_mapping_fingerprint": mapping["mapping_fingerprint"],
                "semantic_token_initialization": (
                    "mean base-model rows of child-label text tokens only"
                ),
                "semantic_output_training": "lm_head LoRA",
            }
        )
        registration = contract.get("semantic_token_registration")
        if not isinstance(registration, dict):
            raise ValueError("training contract lacks semantic token registration")
        if registration.get("semantic_token_count") != args.expected_label_count:
            raise ValueError("training contract has the wrong semantic token count")
    mismatches = {
        field: {"expected": value, "actual": contract.get(field)}
        for field, value in expected.items()
        if contract.get(field) != value
    }
    if mismatches:
        raise ValueError(
            "completed training artifact contract differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    print(f"Validated reusable training artifact: {args.train_dir}")


if __name__ == "__main__":
    main()