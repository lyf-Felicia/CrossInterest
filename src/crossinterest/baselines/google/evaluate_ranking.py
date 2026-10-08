#!/usr/bin/env python3
"""Evaluate the K=2-trained Google baseline on complete histories with beam ranking."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from crossinterest.evaluation.common import append_jsonl, latest_details, sha256_file, write_json_atomic
from crossinterest.training.evaluate_ranking import direct_beam_rank, metrics_from_rankings, score_ranking, write_predictions

from crossinterest.baselines.google.evaluate import load_category_model
from crossinterest.baselines.google.prompts import FULL_HISTORY_RANKING_PROMPT_SCHEMA_VERSION, cluster_output, render_full_history_generation_prompt


HERE = _source_workspace('eval_baseline_google')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            HERE.parent
            / 'crossinterest.training'
            / "data"
            / "eval"
            / "dev_2k_behavior_interest.parquet"
        ),
    )
    parser.add_argument("--label-file", type=Path, default=HERE / "data" / "train_top10_pairs.parquet")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--expected-resume-rows", type=int)
    parser.add_argument("--resume-adapter-sha256")
    parser.add_argument("--expected-label-count", type=int, default=112)
    parser.add_argument("--expected-eval-rows", type=int, default=2_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    return parser.parse_args()


def load_labels(path: Path, expected_count: int) -> list[str]:
    schema = pq.read_schema(path)
    label_column = "target_cluster" if "target_cluster" in schema.names else "target_interest"
    labels = sorted(set(pq.read_table(path, columns=[label_column])[label_column].to_pylist()))
    if len(labels) != expected_count:
        raise ValueError(f"expected {expected_count} labels, found {len(labels)}")
    return labels


def validate_full_history_rows(rows: list[dict], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} full-history rows, found {len(rows)}")
    if len({row["sample_id"] for row in rows}) != expected_rows:
        raise ValueError("full-history ranking rows contain duplicate sample IDs")
    for row in rows:
        titles = row.get("historical_titles")
        interests = row.get("historical_interests")
        if row.get("input_type") != "behavior_interest":
            raise ValueError(f"{row['sample_id']}: expected behavior_interest input")
        if not titles or len(titles) != len(interests) or len(titles) != row["history_length"]:
            raise ValueError(f"{row['sample_id']}: incomplete or unaligned full history")


def render_prompt_ids(tokenizer, row: dict, model_key: str) -> list[int]:
    prompt = render_full_history_generation_prompt(
        row["historical_titles"],
        row["historical_interests"],
    )
    options = {"tokenize": True, "add_generation_prompt": True, "return_dict": True}
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **options)
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def establish_contract(args: argparse.Namespace, labels: list[str]) -> dict:
    contract = {
        "schema_version": 1,
        "task": "google_interest_exploration_amazon_candidate_free_ranking",
        "prompt_schema_version": FULL_HISTORY_RANKING_PROMPT_SCHEMA_VERSION,
        "model_key": args.model_key,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "label_file": str(args.label_file.resolve()),
        "label_file_sha256": sha256_file(args.label_file),
        "label_count": len(labels),
        "sample_limit": args.limit,
        "decoding": "candidate_free_direct_beam_search",
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "decoder_label_constraint": False,
        "input_history_policy": (
            "complete paired historical_titles and historical_interests sequences "
            "in chronological order"
        ),
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


def seed_resume_details(
    resume_run_dir: Path | None,
    run_dir: Path,
    contract: dict,
    expected_resume_rows: int | None,
    resume_adapter_sha256: str | None,
) -> None:
    if resume_run_dir is None:
        if expected_resume_rows is not None or resume_adapter_sha256 is not None:
            raise ValueError("resume checks require --resume-run-dir")
        return
    source_contract_path = resume_run_dir / "run_contract.json"
    source_details_path = resume_run_dir / "details.jsonl"
    if not source_contract_path.exists() or not source_details_path.exists():
        raise ValueError(f"{resume_run_dir}: resume contract or details are missing")
    source_contract = json.loads(source_contract_path.read_text())
    contracts_match = source_contract == contract
    if not contracts_match:
        ignored_fields = {"adapter", "fingerprint"}
        source_semantics = {
            key: value for key, value in source_contract.items() if key not in ignored_fields
        }
        current_semantics = {
            key: value for key, value in contract.items() if key not in ignored_fields
        }
        if source_semantics != current_semantics:
            raise ValueError(f"{resume_run_dir}: resume run contract differs")
        if resume_adapter_sha256 is None:
            raise ValueError("--resume-adapter-sha256 is required for an adapter path alias")
        adapter_sha256 = sha256_file(Path(contract["adapter"]) / "adapter_model.safetensors")
        if adapter_sha256 != resume_adapter_sha256:
            raise ValueError(
                f"resume adapter SHA differs: expected {resume_adapter_sha256}, "
                f"found {adapter_sha256}"
            )
    source_completed = latest_details(
        source_details_path,
        source_contract["fingerprint"],
    )
    if not contracts_match:
        source_completed = {
            sample_id: {**detail, "run_fingerprint": contract["fingerprint"]}
            for sample_id, detail in source_completed.items()
        }
    if expected_resume_rows is not None and len(source_completed) != expected_resume_rows:
        raise ValueError(
            f"{resume_run_dir}: expected {expected_resume_rows} resume rows, "
            f"found {len(source_completed)}"
        )
    details_path = run_dir / "details.jsonl"
    if not details_path.exists():
        if contracts_match:
            shutil.copyfile(source_details_path, details_path)
        else:
            temporary = details_path.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for detail in source_completed.values():
                    handle.write(json.dumps(detail, ensure_ascii=False, separators=(",", ":")) + "\n")
            temporary.replace(details_path)
    else:
        completed = latest_details(details_path, contract["fingerprint"])
        if any(completed.get(sample_id) != detail for sample_id, detail in source_completed.items()):
            raise ValueError(f"{run_dir}: existing details do not contain the resume run")
    write_json_atomic(
        run_dir / "resume_provenance.json",
        {
            "source_run_dir": str(resume_run_dir),
            "source_fingerprint": source_contract["fingerprint"],
            "target_fingerprint": contract["fingerprint"],
            "adapter_model_sha256": resume_adapter_sha256,
            "imported_rows": len(source_completed),
        },
    )


def main() -> None:
    args = parse_args()
    if args.num_beams < 20:
        raise ValueError("--num-beams must be at least 20 for Recall/nDCG@20")
    args.adapter = args.adapter.resolve()
    args.dataset = args.dataset.resolve()
    args.label_file = args.label_file.resolve()
    args.run_dir = args.run_dir.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.dataset).to_pylist()
    validate_full_history_rows(rows, args.expected_eval_rows)
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    labels = load_labels(args.label_file, args.expected_label_count)
    output_to_category = {cluster_output(label): label for label in labels}
    expected_rows = rows
    contract = establish_contract(args, labels)
    seed_resume_details(
        args.resume_run_dir,
        args.run_dir,
        contract,
        args.expected_resume_rows,
        args.resume_adapter_sha256,
    )
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
        started = time.monotonic()
        ranking = direct_beam_rank(
            model, tokenizer, prompt_ids, output_to_category, args.num_beams, args.max_new_tokens
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
            print(f"{args.model_key} ranking completed={len(completed)}/{len(rows)}", flush=True)
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_rankings(ordered, expected_rows, args.num_beams)
    metrics.update(
        model_key=args.model_key,
        adapter=str(args.adapter),
        decoding="candidate_free_direct_beam_search",
        num_beams=args.num_beams,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()