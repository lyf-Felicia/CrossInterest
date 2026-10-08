#!/usr/bin/env python3
"""Evaluate an Amazon SFT LoRA with prompt-v3 candidate-constrained beam search."""

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
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, sha256_file, write_json_atomic, write_predictions
from crossinterest.legacy.training_99.label_space import LABEL_MODES, category_from_output, load_label_mapping, output_label
from crossinterest.legacy.training_99.prompts import render_classification_prompt


HERE = _source_workspace('eval_amazon_sft_99')
DEFAULT_DATA_DIR = HERE.parent / 'crossinterest.evaluation' / "data"
DEFAULT_LABEL_MAP = HERE / "data" / "interest_id_mapping.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="category")
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=99)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--reset-on-contract-mismatch", action="store_true")
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


def candidate_sequences(tokenizer, candidates: list[str]) -> dict[tuple[int, ...], str]:
    sequences = {}
    for candidate in candidates:
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        sequence = tuple(token_ids + [tokenizer.eos_token_id])
        if sequence in sequences:
            raise ValueError(f"candidate token collision: {candidate}")
        sequences[sequence] = candidate
    return sequences


def render_prompt_ids(
    tokenizer,
    row: dict,
    model_key: str,
    view: str,
    label_mode: str,
) -> list[int]:
    source = {**row, "id": row["source_sample_id"]}
    prompt = render_classification_prompt(
        source,
        row["candidate_interests"],
        view,
        label_mode,
    )
    if label_mode == "category" and prompt != row["prompt"][0]["content"]:
        raise ValueError(f"{row['sample_id']}: category prompt differs from frozen v3")
    options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
    }
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **options,
    )
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def predict(model, tokenizer, prompt_ids: list[int], candidates: list[str]) -> tuple[str, float]:
    sequences = candidate_sequences(tokenizer, candidates)
    trie = TokenTrie(sequences)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    def allowed_tokens(_batch_id: int, current_ids: torch.Tensor) -> list[int]:
        generated = current_ids.tolist()[len(prompt_ids) :]
        allowed = trie.allowed(generated)
        if not allowed:
            generated_tokens = tuple(generated)
            completed = any(
                generated_tokens[: len(sequence)] == sequence
                and all(
                    token == tokenizer.eos_token_id
                    for token in generated_tokens[len(sequence) :]
                )
                for sequence in sequences
            )
            if completed:
                return [tokenizer.eos_token_id]
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


def load_model(model_key: str, model_name: str, adapter: Path):
    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    base = AutoModelForCausalLM.from_pretrained(model_name, **common)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model


def adapter_hashes(adapter: Path) -> tuple[str, str]:
    config = json.loads((adapter / "adapter_config.json").read_text())
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_sha256 = hashlib.sha256(canonical_config.encode()).hexdigest()
    model_files = [
        path
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path := adapter / name).is_file()
    ]
    if len(model_files) != 1:
        raise ValueError(f"{adapter}: expected exactly one adapter model file")
    return config_sha256, sha256_file(model_files[0])


def establish_contract(
    args: argparse.Namespace,
    dataset: Path,
    mapping: dict | None,
) -> dict:
    adapter_config_sha256, adapter_model_sha256 = adapter_hashes(args.adapter)
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_choice_10way_sft",
        "prompt_schema_version": 3 if args.label_mode == "category" else "4-interest-id",
        "model_key": args.model_key,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_model_sha256": adapter_model_sha256,
        "view": args.view,
        "label_mode": args.label_mode,
        "label_map": str(args.label_map.resolve()) if mapping else None,
        "label_map_sha256": sha256_file(args.label_map) if mapping else None,
        "label_mapping_fingerprint": mapping["mapping_fingerprint"] if mapping else None,
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
            if not args.reset_on_contract_mismatch:
                raise ValueError(f"{args.run_dir}: existing run contract differs")
            stale_dir = args.run_dir / "stale" / json.loads(path.read_text())["fingerprint"]
            stale_dir.mkdir(parents=True, exist_ok=True)
            for name in ("run_contract.json", "details.jsonl", "metrics.json", "predictions.jsonl"):
                source = args.run_dir / name
                if source.exists():
                    source.replace(stale_dir / name)
    write_json_atomic(path, contract)
    return contract


def main() -> None:
    args = parse_args()
    args.adapter = args.adapter.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / f"dev_2k_{args.view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    mapping = None
    if args.label_mode == "interest_id":
        categories = {
            category
            for row in rows
            for category in row["candidate_interests"]
        }
        mapping = load_label_mapping(
            args.label_map.resolve(),
            expected_count=args.expected_label_count,
        )
        if not categories <= set(mapping["category_to_id"]):
            raise ValueError("evaluation candidates are absent from the label mapping")
    contract = establish_contract(args, dataset, mapping)
    details_path = args.run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model_key, args.model, args.adapter)
    for index, row in enumerate(rows, 1):
        if row["sample_id"] in completed:
            continue
        prompt_ids = render_prompt_ids(
            tokenizer,
            row,
            args.model_key,
            args.view,
            args.label_mode,
        )
        if len(prompt_ids) > args.max_prompt_length:
            raise ValueError(f"{row['sample_id']}: prompt has {len(prompt_ids)} tokens")
        started = time.monotonic()
        output_candidates = [
            output_label(category, args.label_mode, mapping)
            for category in row["candidate_interests"]
        ]
        predicted_output, sequence_score = predict(
            model, tokenizer, prompt_ids, output_candidates
        )
        prediction = category_from_output(predicted_output, args.label_mode, mapping)
        detail = {
            "sample_id": row["sample_id"],
            "source_sample_id": row["source_sample_id"],
            "target_interest": row["target_interest"],
            "prediction": prediction,
            "predicted_output": predicted_output,
            "target_output": output_label(
                row["target_interest"], args.label_mode, mapping
            ),
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
            print(f"{args.model_key}/{args.view} completed={len(completed)}/{len(rows)}", flush=True)
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_details(ordered, rows)
    metrics.update(
        model_key=args.model_key,
        input_type=args.view,
        adapter=str(args.adapter),
        decoding="candidate_constrained_beam_search",
        label_mode=args.label_mode,
        num_beams=10,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()