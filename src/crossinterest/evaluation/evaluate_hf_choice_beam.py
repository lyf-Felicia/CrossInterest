#!/usr/bin/env python3
"""Evaluate base HF models on Amazon prompt-v3 with constrained 10-way beam search."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

try:
    from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, sha256_file, write_json_atomic, write_predictions
except ImportError:
    from common import (
        INPUT_TYPES,
        append_jsonl,
        latest_details,
        metrics_from_details,
        sha256_file,
        write_json_atomic,
        write_predictions,
    )


HERE = _source_workspace('eval_amazon_notrain')
REGISTRY_PATH = HERE / "models.json"
DEFAULT_DATA_DIR = HERE / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--input-type", choices=(*INPUT_TYPES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument("--run-name", default="formal")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    return parser.parse_args()


class TokenTrie:
    def __init__(self, sequences: dict[tuple[int, ...], str]) -> None:
        self.root: dict[int, dict] = {}
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})

    def allowed(self, generated: list[int]) -> list[int]:
        node = self.root
        for token in generated:
            if token not in node:
                return []
            node = node[token]
        return list(node)


def directory_contract(args: argparse.Namespace, model: dict, dataset: Path, view: str) -> dict:
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_choice_10way",
        "prompt_schema_version": 3,
        "model_key": args.model_key,
        "model": model["model"],
        "adapter": None,
        "input_type": view,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "sample_limit": args.limit,
        "decoding": "candidate_constrained_beam_search",
        "num_beams": 10,
        "do_sample": False,
        "candidate_count": 10,
        "strict_output": "<exact candidate name only>",
        "enable_thinking": model.get("enable_thinking"),
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    return contract


def render_prompt_ids(tokenizer, row: dict, model: dict) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
    }
    if model.get("enable_thinking") is False:
        kwargs["enable_thinking"] = False
    value = tokenizer.apply_chat_template(row["prompt"], **kwargs)
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def candidate_sequences(tokenizer, candidates: list[str]) -> dict[tuple[int, ...], str]:
    sequences = {}
    for candidate in candidates:
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        sequence = tuple(token_ids + [tokenizer.eos_token_id])
        if sequence in sequences:
            raise ValueError(f"token collision: {candidate} and {sequences[sequence]}")
        sequences[sequence] = candidate
    return sequences


def predict(model, tokenizer, prompt_ids: list[int], candidates: list[str]) -> tuple[str, float]:
    sequences = candidate_sequences(tokenizer, candidates)
    trie = TokenTrie(sequences)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    def allowed_tokens(_batch_id: int, current_ids: torch.Tensor) -> list[int]:
        generated = current_ids.tolist()[len(prompt_ids) :]
        allowed = trie.allowed(generated)
        if not allowed:
            raise RuntimeError(f"beam escaped candidate trie after tokens {generated}")
        return allowed

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            num_beams=10,
            num_return_sequences=1,
            do_sample=False,
            max_new_tokens=max(len(sequence) for sequence in sequences),
            prefix_allowed_tokens_fn=allowed_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    suffix = output.sequences[0].tolist()[len(prompt_ids) :]
    if tokenizer.eos_token_id in suffix:
        suffix = suffix[: suffix.index(tokenizer.eos_token_id) + 1]
    prediction = sequences.get(tuple(suffix))
    if prediction is None:
        raise RuntimeError(f"constrained beam did not resolve to a candidate: {suffix}")
    return prediction, float(output.sequences_scores[0])


def load_model(model_key: str, model_name: str):
    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    if model_key == "qwen35_9b":
        model = AutoModelForImageTextToText.from_pretrained(model_name, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common)
    model.eval()
    return model


def run_view(args, model_entry: dict, tokenizer, model, view: str) -> dict:
    dataset = args.data_dir / f"dev_2k_{view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    run_dir = args.run_root / args.run_name / args.model_key / view
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = directory_contract(args, model_entry, dataset, view)
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        details_path = run_dir / "details.jsonl"
        if details_path.exists() and details_path.stat().st_size:
            raise ValueError(f"{run_dir}: existing run contract differs")
    write_json_atomic(contract_path, contract)
    details_path = run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    for index, row in enumerate(rows, 1):
        if row["sample_id"] in completed:
            continue
        prompt_ids = render_prompt_ids(tokenizer, row, model_entry)
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
            print(f"{args.model_key}/{view} completed={len(completed)}/{len(rows)}", flush=True)
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_details(ordered, rows)
    metrics.update(
        model_key=args.model_key,
        display_name=model_entry["display_name"],
        input_type=view,
        decoding="candidate_constrained_beam_search",
        num_beams=10,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(run_dir / "metrics.json", metrics)
    write_predictions(run_dir / "predictions.jsonl", ordered, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    model_entry = registry["models"][args.model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_entry["model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model_key, model_entry["model"])
    views = INPUT_TYPES if args.input_type == "all" else (args.input_type,)
    metrics = [run_view(args, model_entry, tokenizer, model, view) for view in views]
    if any(not item["complete"] for item in metrics):
        raise RuntimeError("one or more views are incomplete")


if __name__ == "__main__":
    main()