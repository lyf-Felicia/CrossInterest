#!/usr/bin/env python3
"""Evaluate the Google K=2 baseline on the frozen Amazon 10-way task."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from crossinterest.evaluation.common import append_jsonl, latest_details, metrics_from_details, sha256_file, write_json_atomic, write_predictions
from crossinterest.training.evaluate import load_model as load_sft_model, predict

from crossinterest.baselines.google.prompts import PROMPT_SCHEMA_VERSION, cluster_output, render_classification_prompt


HERE = _source_workspace('eval_baseline_google')


def load_category_model(model_key: str, model_name: str, adapter: Path, tokenizer):
    return load_sft_model(model_key, model_name, adapter, tokenizer, None, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=HERE / "data" / "dev_2k_pair.parquet")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=112)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    return parser.parse_args()


def render_prompt_ids(tokenizer, row: dict, model_key: str) -> list[int]:
    prompt = render_classification_prompt(row["cluster_pair"], row["candidate_interests"])
    options = {"tokenize": True, "add_generation_prompt": True, "return_dict": True}
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **options)
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def establish_contract(args: argparse.Namespace, dataset: Path) -> dict:
    contract = {
        "schema_version": 1,
        "task": "google_interest_exploration_amazon_choice_10way",
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "model_key": args.model_key,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "sample_limit": args.limit,
        "decoding": "candidate_constrained_beam_search",
        "candidate_count": 10,
        "num_beams": 10,
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
    args.adapter = args.adapter.resolve()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.dataset).to_pylist()
    if len({row["target_cluster"] for row in rows}) != args.expected_label_count:
        raise ValueError("evaluation label count mismatch")
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    expected_rows = [{**row, "target_interest": row["target_cluster"]} for row in rows]
    contract = establish_contract(args, args.dataset)
    details_path = args.run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_category_model(args.model_key, args.model, args.adapter, tokenizer)
    for index, row in enumerate(rows, 1):
        if row["sample_id"] in completed:
            continue
        prompt_ids = render_prompt_ids(tokenizer, row, args.model_key)
        if len(prompt_ids) > args.max_prompt_length:
            raise ValueError(f"{row['sample_id']}: prompt has {len(prompt_ids)} tokens")
        output_candidates = [cluster_output(candidate) for candidate in row["candidate_interests"]]
        started = time.monotonic()
        predicted_output, sequence_score = predict(model, tokenizer, prompt_ids, output_candidates)
        prediction = row["candidate_interests"][output_candidates.index(predicted_output)]
        detail = {
            "sample_id": row["sample_id"],
            "source_sample_id": row["source_sample_id"],
            "target_interest": row["target_cluster"],
            "prediction": prediction,
            "predicted_output": predicted_output,
            "target_output": cluster_output(row["target_cluster"]),
            "correct": prediction == row["target_cluster"],
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
            print(f"{args.model_key} classification completed={len(completed)}/{len(rows)}", flush=True)
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_details(ordered, expected_rows)
    metrics.update(
        model_key=args.model_key,
        adapter=str(args.adapter),
        decoding="candidate_constrained_beam_search",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered, expected_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()