#!/usr/bin/env python3
"""Run resumable Amazon 10-way evaluation through TRAPI or Copilot API."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI

try:
    from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, metrics_from_details, prompt_text, score_response, sha256_file, write_json_atomic, write_predictions
except ImportError:
    from common import (
        INPUT_TYPES,
        append_jsonl,
        latest_details,
        metrics_from_details,
        prompt_text,
        score_response,
        sha256_file,
        write_json_atomic,
        write_predictions,
    )


HERE = _source_workspace('eval_amazon_notrain')
REGISTRY_PATH = HERE / "models.json"
DEFAULT_DATA_DIR = HERE / "data"
API_VERSION = "2024-12-01-preview"
RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRY_AFTER_PATTERN = re.compile(r"(?:in|after)\s+(\d+)\s+seconds?", re.IGNORECASE)
WRITE_LOCK = threading.Lock()
RETRY_PRESSURE_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input-type", choices=(*INPUT_TYPES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument("--run-name", default="formal")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-concurrency", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume-rounds", type=int, default=3)
    return parser.parse_args()


def make_client(registry: dict, model: dict) -> AzureOpenAI | OpenAI:
    if model["backend"] == "openai_compatible":
        return OpenAI(
            base_url=model["base_url"],
            api_key="copilot-proxy",
            timeout=600.0,
            max_retries=0,
        )
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "api://trapi/.default",
    )
    return AzureOpenAI(
        azure_endpoint=(
            "https://trapi.research.microsoft.com/"
            f"{registry['trapi_instance']}/"
        ),
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
        timeout=180.0,
        max_retries=0,
    )


def request_once(
    client: AzureOpenAI | OpenAI,
    model: dict,
    prompt: str,
) -> tuple[str, dict]:
    parameters: dict[str, Any] = {
        "model": model["deployment"],
        "messages": [{"role": "user", "content": prompt}],
    }
    if model["backend"] == "openai_compatible":
        parameters.update(
            reasoning_effort=model["reasoning_effort"],
            stream=True,
        )
        stream = client.chat.completions.create(**parameters)
        parts = []
        finish_reason = None
        usage = None
        for chunk in stream:
            usage = chunk.usage or usage
            for choice in chunk.choices:
                finish_reason = choice.finish_reason or finish_reason
                parts.append(choice.delta.content or "")
        content = "".join(parts)
        if finish_reason is None or not content.strip():
            raise ValueError("stream ended without a complete nonempty response")
        if finish_reason == "length":
            raise ValueError("stream response was truncated at its output limit")
        return content, {
            "finish_reason": finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    parameters.update(
        max_completion_tokens=2048,
        reasoning_effort=model["reasoning_effort"],
    )
    response = client.chat.completions.create(**parameters)
    choice = response.choices[0]
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty model response")
    usage = response.usage
    return content, {
        "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def evaluate_row(
    client: AzureOpenAI | OpenAI,
    model: dict,
    row: dict,
    fingerprint: str,
    max_retries: int,
) -> dict:
    started = time.monotonic()
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            response, usage = request_once(client, model, prompt_text(row))
            scored = score_response(
                response,
                row["candidate_interests"],
                row["target_interest"],
            )
            return {
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "target_interest": row["target_interest"],
                **scored,
                "raw_response": response,
                "latency_seconds": time.monotonic() - started,
                "attempts": attempts,
                "finish_reason": usage["finish_reason"],
                "usage": usage,
                "error": None,
                "status_code": None,
                "run_fingerprint": fingerprint,
            }
        except Exception as error:
            last_error = error
            code = status_code(error)
            retryable = code in RETRYABLE_CODES or code is None
            if attempt >= max_retries or not retryable:
                break
            retry_after = RETRY_AFTER_PATTERN.search(str(error))
            delay = int(retry_after.group(1)) + 1 if retry_after else min(2**attempt, 16)
            time.sleep(delay + random.random())
    assert last_error is not None
    return {
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "target_interest": row["target_interest"],
        "prediction": None,
        "correct": False,
        "format_valid": False,
        "resolved": False,
        "extraction_method": "request_error",
        "raw_response": None,
        "latency_seconds": time.monotonic() - started,
        "attempts": attempts,
        "finish_reason": None,
        "usage": {},
        "error": f"{type(last_error).__name__}: {last_error}",
        "status_code": status_code(last_error),
        "run_fingerprint": fingerprint,
    }


def next_concurrency(workers: int, maximum: int, results: list[dict]) -> int:
    retry_count = sum(
        result["attempts"] > 1 or result["status_code"] in RETRYABLE_CODES
        for result in results
    )
    retry_ratio = retry_count / len(results)
    if retry_ratio >= RETRY_PRESSURE_THRESHOLD:
        return max(1, workers // 2)
    if retry_count == 0 and workers < maximum:
        return min(maximum, workers + max(1, workers // 4))
    return workers


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
        "backend": model["backend"],
        "deployment": model["deployment"],
        "reasoning_effort": model.get("reasoning_effort"),
        "input_type": input_type,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "sample_limit": args.limit,
        "temperature": None,
        "max_completion_tokens": None if model["backend"] == "openai_compatible" else 2048,
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


def run_view(
    args: argparse.Namespace,
    client: AzureOpenAI | OpenAI,
    model: dict,
    input_type: str,
) -> dict:
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
    initial = args.initial_concurrency or model["initial_concurrency"]
    maximum = args.max_concurrency or model["max_concurrency"]
    workers = max(1, min(initial, maximum))
    row_by_id = {row["sample_id"]: row for row in rows}
    for round_index in range(args.resume_rounds):
        existing = latest_details(details_path, contract["fingerprint"])
        pending = [
            row
            for row in rows
            if row["sample_id"] not in existing or existing[row["sample_id"]].get("error") is not None
        ]
        if not pending:
            break
        print(
            f"{args.model_key}/{input_type} round={round_index + 1} "
            f"recovered={len(rows) - len(pending)}/{len(rows)} pending={len(pending)}",
            flush=True,
        )
        cursor = 0
        while cursor < len(pending):
            batch = pending[cursor : cursor + workers]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_row,
                        client,
                        model,
                        row,
                        contract["fingerprint"],
                        args.max_retries,
                    )
                    for row in batch
                ]
                results = [future.result() for future in as_completed(futures)]
            for result in results:
                with WRITE_LOCK:
                    append_jsonl(details_path, result)
            cursor += len(batch)
            workers = next_concurrency(workers, maximum, results)
            print(
                f"{args.model_key}/{input_type} round={round_index + 1} "
                f"completed={cursor}/{len(pending)} concurrency={workers}",
                flush=True,
            )
    latest = latest_details(details_path, contract["fingerprint"])
    ordered_details = [latest[row_id] for row_id in row_by_id if row_id in latest]
    metrics = metrics_from_details(ordered_details, rows)
    metrics.update(
        model_key=args.model_key,
        display_name=model["display_name"],
        input_type=input_type,
        final_concurrency=workers,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(run_dir / "metrics.json", metrics)
    write_predictions(run_dir / "predictions.jsonl", ordered_details, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    model = registry["models"].get(args.model_key)
    if model is None or model.get("backend") not in {"trapi", "openai_compatible"}:
        raise ValueError(f"{args.model_key}: expected a registered API model")
    if args.resume_rounds <= 0:
        raise ValueError("--resume-rounds must be positive")
    client = make_client(registry, model)
    input_types = INPUT_TYPES if args.input_type == "all" else (args.input_type,)
    metrics = [run_view(args, client, model, input_type) for input_type in input_types]
    if any(not item["complete"] for item in metrics):
        raise RuntimeError("one or more views remain incomplete after resume rounds")


if __name__ == "__main__":
    main()