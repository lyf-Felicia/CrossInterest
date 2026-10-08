#!/usr/bin/env python3
"""Evaluate a base model on the exact Amazon 112-label SFT classification path."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, sha256_file, write_json_atomic, write_predictions
from crossinterest.training.evaluate import predict, render_prompt_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=112)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    return parser.parse_args()


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    model.eval()
    return model


def establish_contract(args: argparse.Namespace, dataset: Path) -> dict:
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_choice_10way_no_training_112",
        "prompt_schema_version": 3,
        "model_key": args.model_key,
        "model": args.model,
        "adapter": None,
        "view": args.view,
        "label_mode": "category",
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "sample_limit": args.limit,
        "decoding": "candidate_constrained_beam_search",
        "num_beams": 10,
        "do_sample": False,
        "candidate_count": 10,
        "expected_label_count": args.expected_label_count,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    path = args.run_dir / "run_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        details = args.run_dir / "details.jsonl"
        if details.exists() and details.stat().st_size:
            raise ValueError(f"{args.run_dir}: existing run contract differs")
    write_json_atomic(path, contract)
    return contract


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / f"dev_2k_{args.view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    contract = establish_contract(args, dataset)
    details_path = args.run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model)
    for index, row in enumerate(rows, 1):
        if row["sample_id"] in completed:
            continue
        prompt_ids = render_prompt_ids(
            tokenizer,
            row,
            args.model_key,
            args.view,
            "category",
        )
        if len(prompt_ids) > args.max_prompt_length:
            raise ValueError(f"{row['sample_id']}: prompt has {len(prompt_ids)} tokens")
        started = time.monotonic()
        prediction, sequence_score = predict(
            model,
            tokenizer,
            prompt_ids,
            row["candidate_interests"],
        )
        detail = {
            "sample_id": row["sample_id"],
            "source_sample_id": row["source_sample_id"],
            "target_interest": row["target_interest"],
            "prediction": prediction,
            "predicted_output": prediction,
            "target_output": row["target_interest"],
            "correct": prediction == row["target_interest"],
            "format_valid": True,
            "resolved": True,
            "extraction_method": "candidate_constrained_beam",
            "sequence_score": sequence_score,
            "latency_seconds": time.monotonic() - started,
            "attempts": 1,
            "finish_reason": "candidate_eos",
            "error": None,
            "status_code": None,
            "run_fingerprint": contract["fingerprint"],
        }
        append_jsonl(details_path, detail)
        completed[row["sample_id"]] = detail
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(
                f"{args.model_key}/{args.view} completed={len(completed)}/{len(rows)}",
                flush=True,
            )
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_details(ordered, rows)
    metrics.update(
        model_key=args.model_key,
        input_type=args.view,
        adapter=None,
        decoding="candidate_constrained_beam_search",
        label_mode="category",
        num_beams=10,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    if metrics["complete"] and metrics["class_count"] != args.expected_label_count:
        raise ValueError(
            f"expected {args.expected_label_count} classes, found {metrics['class_count']}"
        )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()