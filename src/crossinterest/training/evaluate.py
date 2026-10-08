#!/usr/bin/env python3
"""Evaluate an Amazon SFT LoRA with prompt-v3 candidate-constrained beam search."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace
from crossinterest.paths import dataset_dir

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, sha256_file, write_json_atomic, write_predictions
from crossinterest.training.label_space import INTEREST_ID_LABEL_MODES, LABEL_MODES, category_from_output, load_label_mapping, output_label, prompt_schema_version
from crossinterest.training.prompts import render_classification_prompt
from crossinterest.training.semantic_tokens import initialize_semantic_token_embeddings, register_semantic_tokens


HERE = _source_workspace('eval_amazon_sft_112')
DEFAULT_DATA_DIR = dataset_dir() / "eval"
DEFAULT_LABEL_MAP = dataset_dir() / "interest_id_semantic_mapping.json"


def normalized_adapter_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    return config


def resolve_contract(path: Path, contract: dict, adapter: Path) -> dict:
    if not path.exists():
        write_json_atomic(path, contract)
        return contract
    existing = json.loads(path.read_text())
    if existing == contract:
        return existing
    details = path.parent / "details.jsonl"
    if not details.exists() or not details.stat().st_size:
        write_json_atomic(path, contract)
        return contract
    ignored = {"adapter", "adapter_config_sha256", "fingerprint"}
    existing_identity = {key: value for key, value in existing.items() if key not in ignored}
    current_identity = {key: value for key, value in contract.items() if key not in ignored}
    current_config = normalized_adapter_config(adapter / "adapter_config.json")
    compatible_checkpoint = any(
        sha256_file(checkpoint_config) == existing.get("adapter_config_sha256")
        and normalized_adapter_config(checkpoint_config) == current_config
        for checkpoint_config in adapter.parent.glob("checkpoint-*/adapter_config.json")
    )
    if existing_identity == current_identity and compatible_checkpoint:
        print(f"Reusing compatible existing run contract: {path}", flush=True)
        return existing
    raise ValueError(f"{path.parent}: existing run contract differs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="category")
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=112)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--device-map", default="cuda")
    return parser.parse_args(argv)


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


def load_model(
    model_key: str,
    model_name: str,
    adapter: Path,
    tokenizer,
    mapping: dict | None,
    semantic_registration: dict | None,
    *,
    device_map: str = "cuda",
):
    from peft import PeftModel

    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    base = AutoModelForCausalLM.from_pretrained(model_name, **common)
    if mapping is not None:
        initialize_semantic_token_embeddings(
            base,
            tokenizer,
            mapping,
            semantic_registration,
        )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model


def establish_contract(
    args: argparse.Namespace,
    dataset: Path,
    mapping: dict | None,
    semantic_registration: dict | None,
) -> dict:
    has_history_summary = "history_summary" in pq.ParquetFile(dataset).schema.names
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_choice_10way_sft",
        "prompt_schema_version": prompt_schema_version(
            args.label_mode,
            history_summary=has_history_summary,
        ),
        "history_summary": has_history_summary,
        "model_key": args.model_key,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "view": args.view,
        "label_mode": args.label_mode,
        "label_map": str(args.label_map.resolve()) if mapping else None,
        "label_map_sha256": sha256_file(args.label_map) if mapping else None,
        "label_mapping_fingerprint": mapping["mapping_fingerprint"] if mapping else None,
        "semantic_token_registration": semantic_registration,
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
    return resolve_contract(path, contract, args.adapter)


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
    if args.label_mode in INTEREST_ID_LABEL_MODES:
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    semantic_registration = (
        register_semantic_tokens(tokenizer, mapping) if mapping is not None else None
    )
    contract = establish_contract(args, dataset, mapping, semantic_registration)
    details_path = args.run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    model = load_model(
        args.model_key,
        args.model,
        args.adapter,
        tokenizer,
        mapping,
        semantic_registration,
        device_map=args.device_map,
    )
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