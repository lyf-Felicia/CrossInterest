#!/usr/bin/env python3
"""Evaluate an Amazon SFT LoRA with candidate-free direct beam generation."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, normalize_candidate_text, sha256_file, write_json_atomic
from crossinterest.legacy.training_99.evaluate import adapter_hashes, load_model
from crossinterest.legacy.training_99.label_space import LABEL_MODES, load_label_mapping, output_label
from crossinterest.legacy.training_99.prompts import render_training_prompt


HERE = _source_workspace('eval_amazon_sft_99')
DEFAULT_DATA_DIR = HERE.parent / 'crossinterest.evaluation' / "data"
DEFAULT_LABEL_FILE = HERE.parent / "train.parquet"
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
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=99)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--reset-on-contract-mismatch", action="store_true")
    return parser.parse_args()


def load_labels(path: Path, expected_label_count: int = 99) -> list[str]:
    labels = sorted(set(pq.read_table(path, columns=["target_interest"])["target_interest"].to_pylist()))
    if len(labels) != expected_label_count:
        raise ValueError(
            f"expected {expected_label_count} formal Train labels, found {len(labels)}"
        )
    return labels


def render_prompt_ids(
    tokenizer,
    row: dict,
    model_key: str,
    view: str,
    label_mode: str,
) -> list[int]:
    source = {**row, "id": row["source_sample_id"]}
    prompt = render_training_prompt(source, view, label_mode)
    if "Candidates:" in prompt:
        raise AssertionError("candidate-free ranking prompt contains a candidate list")
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


def direct_beam_rank(
    model,
    tokenizer,
    prompt_ids: list[int],
    output_to_category: dict[str, str],
    num_beams: int,
    max_new_tokens: int,
) -> list[dict]:
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    label_by_normalized = {
        label.casefold(): label for label in output_to_category
    }
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ranked = []
    for rank, (sequence, score) in enumerate(
        zip(output.sequences, output.sequences_scores, strict=True),
        1,
    ):
        suffix = sequence.tolist()[len(prompt_ids) :]
        raw_text = tokenizer.decode(suffix, skip_special_tokens=True).strip()
        normalized = normalize_candidate_text(raw_text)
        predicted_output = label_by_normalized.get(normalized.casefold())
        prediction = output_to_category.get(predicted_output) if predicted_output else None
        ranked.append(
            {
                "rank": rank,
                "raw_text": raw_text,
                "predicted_output": predicted_output,
                "prediction": prediction,
                "format_valid": predicted_output is not None and raw_text == predicted_output,
                "sequence_score": float(score),
            }
        )
    return ranked


def score_ranking(ranking: list[dict], target: str) -> dict:
    target_rank = next(
        (item["rank"] for item in ranking if item["prediction"] == target),
        None,
    )
    metrics = {}
    for cutoff in (10, 20):
        hit = target_rank is not None and target_rank <= cutoff
        metrics[f"recall_at_{cutoff}"] = float(hit)
        metrics[f"ndcg_at_{cutoff}"] = (
            1.0 / math.log2(target_rank + 1) if hit else 0.0
        )
    return {"target_rank": target_rank, **metrics}


def metrics_from_rankings(rows: list[dict], expected_rows: list[dict], num_beams: int) -> dict:
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    actual_by_id = {row["sample_id"]: row for row in rows}
    unknown = set(actual_by_id) - set(expected_by_id)
    if unknown:
        raise ValueError(f"details contain unknown sample IDs: {sorted(unknown)[:3]}")
    per_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid_beams = 0
    strict_beams = 0
    duplicate_valid_beams = 0
    for expected in expected_rows:
        detail = actual_by_id.get(expected["sample_id"])
        if detail is None:
            continue
        ranking = detail["ranking"]
        if len(ranking) != num_beams:
            raise ValueError(f"{expected['sample_id']}: expected {num_beams} beams")
        valid_predictions = [item["prediction"] for item in ranking if item["prediction"]]
        valid_beams += len(valid_predictions)
        strict_beams += sum(bool(item["format_valid"]) for item in ranking)
        duplicate_valid_beams += len(valid_predictions) - len(set(valid_predictions))
        for metric in ("recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20"):
            per_label[expected["target_interest"]][metric].append(float(detail[metric]))
    per_class = {
        label: {
            metric: sum(values) / len(values)
            for metric, values in sorted(label_metrics.items())
        }
        for label, label_metrics in sorted(per_label.items())
    }
    macro = {
        f"macro_{metric}": sum(values[metric] for values in per_class.values()) / len(per_class)
        for metric in ("recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20")
    }
    requested = len(expected_rows)
    total_beams = requested * num_beams
    return {
        "requested_sample_count": requested,
        "recorded_sample_count": len(actual_by_id),
        "missing_sample_count": requested - len(actual_by_id),
        **macro,
        "class_count": len(per_class),
        "per_class_metrics": per_class,
        "valid_beam_rate": valid_beams / total_beams if total_beams else 0.0,
        "strict_beam_format_rate": strict_beams / total_beams if total_beams else 0.0,
        "duplicate_valid_beam_rate": duplicate_valid_beams / total_beams if total_beams else 0.0,
        "complete": len(actual_by_id) == requested,
    }


def establish_contract(
    args: argparse.Namespace,
    dataset: Path,
    labels: list[str],
    mapping: dict | None,
) -> dict:
    adapter_config_sha256, adapter_model_sha256 = adapter_hashes(args.adapter)
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_candidate_free_ranking",
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
        "label_file": str(args.label_file.resolve()),
        "label_file_sha256": sha256_file(args.label_file),
        "label_count": len(labels),
        "expected_label_count": args.expected_label_count,
        "sample_limit": args.limit,
        "decoding": "candidate_free_direct_beam_search",
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "candidates_in_prompt": False,
        "decoder_label_constraint": False,
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


def write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "ranking": [item["prediction"] for item in row["ranking"]],
                        "target_rank": row["target_rank"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_beams < 20:
        raise ValueError("--num-beams must be at least 20 for Recall/nDCG@20")
    args.adapter = args.adapter.resolve()
    args.label_file = args.label_file.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / f"dev_2k_{args.view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    labels = load_labels(args.label_file, args.expected_label_count)
    mapping = None
    if args.label_mode == "interest_id":
        mapping = load_label_mapping(
            args.label_map.resolve(),
            expected_categories=labels,
            expected_count=args.expected_label_count,
        )
    output_to_category = {
        output_label(category, args.label_mode, mapping): category
        for category in labels
    }
    contract = establish_contract(args, dataset, labels, mapping)
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
        ranking = direct_beam_rank(
            model,
            tokenizer,
            prompt_ids,
            output_to_category,
            args.num_beams,
            args.max_new_tokens,
        )
        detail = {
            "sample_id": row["sample_id"],
            "source_sample_id": row["source_sample_id"],
            "target_interest": row["target_interest"],
            "ranking": ranking,
            **score_ranking(ranking, row["target_interest"]),
            "latency_seconds": time.monotonic() - started,
            "run_fingerprint": contract["fingerprint"],
        }
        append_jsonl(details_path, detail)
        completed[row["sample_id"]] = detail
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(
                f"{args.model_key}/{args.view} ranking completed={len(completed)}/{len(rows)}",
                flush=True,
            )
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_rankings(ordered, rows, args.num_beams)
    metrics.update(
        model_key=args.model_key,
        input_type=args.view,
        adapter=str(args.adapter),
        decoding="candidate_free_direct_beam_search",
        label_mode=args.label_mode,
        num_beams=args.num_beams,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()