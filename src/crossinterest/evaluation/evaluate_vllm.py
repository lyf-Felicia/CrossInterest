#!/usr/bin/env python3
"""Run resumable Amazon 10-way evaluation with base Hugging Face models in vLLM."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

try:
    from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, score_response, sha256_file, write_json_atomic, write_predictions
except ImportError:
    from common import (
        INPUT_TYPES,
        append_jsonl,
        latest_details,
        metrics_from_details,
        score_response,
        sha256_file,
        write_json_atomic,
        write_predictions,
    )


HERE = _source_workspace('eval_amazon_notrain')
REGISTRY_PATH = HERE / "models.json"
DEFAULT_DATA_DIR = HERE / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input-type", choices=(*INPUT_TYPES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument("--run-name", default="formal")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def establish_contract(
    args: argparse.Namespace,
    model: dict,
    dataset: Path,
    input_type: str,
    run_dir: Path,
) -> dict:
    contract = {
        "schema_version": 2,
        "task": "amazon_new_interest_choice_10way",
        "prompt_schema_version": 3,
        "model_key": args.model_key,
        "backend": "vllm",
        "model": model["model"],
        "input_type": input_type,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "sample_limit": args.limit,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "max_model_len": args.max_model_len,
        "enable_thinking": model.get("enable_thinking"),
        "adapter": None,
        "strict_output": "<exact candidate name only>",
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    path = run_dir / "run_contract.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
        details = run_dir / "details.jsonl"
        if details.exists() and details.stat().st_size:
            raise ValueError(f"{run_dir}: existing run contract differs")
    write_json_atomic(path, contract)
    return contract


def render_prompt(tokenizer, row: dict, model: dict) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(row["prompt"], **kwargs)


def run_view(args, model, tokenizer, engine, sampling_params, input_type: str) -> dict:
    dataset = args.data_dir / f"dev_2k_{input_type}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    run_dir = args.run_root / args.run_name / args.model_key / input_type
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = establish_contract(args, model, dataset, input_type, run_dir)
    details_path = run_dir / "details.jsonl"
    existing = latest_details(details_path, contract["fingerprint"])
    pending = [row for row in rows if row["sample_id"] not in existing]
    batch_size = min(args.initial_batch_size, args.max_batch_size)
    cursor = 0
    while cursor < len(pending):
        batch = pending[cursor : cursor + batch_size]
        prompts = [render_prompt(tokenizer, row, model) for row in batch]
        prompt_lengths = [
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompt in prompts
        ]
        if max(prompt_lengths) + 128 > args.max_model_len:
            raise ValueError(
                f"prompt plus output budget exceeds max-model-len: {max(prompt_lengths)} + 128"
            )
        started = time.monotonic()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        batch_latency = time.monotonic() - started
        for row, output in zip(batch, outputs, strict=True):
            generated = output.outputs[0]
            truncated = generated.finish_reason == "length"
            scored = (
                {
                    "prediction": None,
                    "correct": False,
                    "format_valid": False,
                    "resolved": False,
                    "extraction_method": "truncated",
                }
                if truncated
                else score_response(
                    generated.text,
                    row["candidate_interests"],
                    row["target_interest"],
                )
            )
            detail = {
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "target_interest": row["target_interest"],
                **scored,
                "raw_response": generated.text,
                "latency_seconds": batch_latency / len(batch),
                "attempts": 1,
                "finish_reason": generated.finish_reason,
                "output_tokens": len(generated.token_ids),
                "truncated": truncated,
                "error": None,
                "status_code": None,
                "run_fingerprint": contract["fingerprint"],
            }
            append_jsonl(details_path, detail)
            existing[row["sample_id"]] = detail
        cursor += len(batch)
        if batch_size < args.max_batch_size:
            batch_size = min(args.max_batch_size, batch_size * 2)
        print(
            f"{args.model_key}/{input_type} completed={cursor}/{len(pending)} "
            f"next_batch_size={batch_size}",
            flush=True,
        )
    ordered = [existing[row["sample_id"]] for row in rows if row["sample_id"] in existing]
    metrics = metrics_from_details(ordered, rows)
    metrics.update(
        model_key=args.model_key,
        display_name=model["display_name"],
        input_type=input_type,
        final_batch_size=batch_size,
        truncation_count=sum(bool(row.get("truncated")) for row in ordered),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(run_dir / "metrics.json", metrics)
    write_predictions(run_dir / "predictions.jsonl", ordered, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    if args.initial_batch_size <= 0 or args.max_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    model = registry["models"].get(args.model_key)
    if model is None or model.get("backend") != "vllm":
        raise ValueError(f"{args.model_key}: expected a registered vLLM model")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model["model"], trust_remote_code=True)
    engine = LLM(
        model=model["model"],
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=128, seed=42)
    input_types = INPUT_TYPES if args.input_type == "all" else (args.input_type,)
    metrics = [
        run_view(args, model, tokenizer, engine, sampling_params, input_type)
        for input_type in input_types
    ]
    if any(not item["complete"] for item in metrics):
        raise RuntimeError("one or more views are incomplete")


if __name__ == "__main__":
    main()