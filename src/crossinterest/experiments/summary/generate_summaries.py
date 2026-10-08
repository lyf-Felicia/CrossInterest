#!/usr/bin/env python3
"""Generate target-blind, grounded history summaries through TRAPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI


API_VERSION = "2024-12-01-preview"
DEFAULT_INSTANCE = "redmond/interactive"
DEFAULT_DEPLOYMENT = "gpt-5.6-sol_2026-07-09"
RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}
INTEREST_ID_PATTERN = re.compile(r"<interest_\d{3}>", re.IGNORECASE)
WRITE_LOCK = threading.Lock()

SUMMARY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "target_blind_history_summary",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "stable_needs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "need": {"type": "string"},
                            "positions": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["need", "positions"],
                    },
                },
                "recent_focus": {"type": "string"},
                "recent_positions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer"},
                },
                "cross_category_pattern": {"type": "string"},
                "diversity": {
                    "type": "string",
                    "enum": ["focused", "mixed", "broad"],
                },
            },
            "required": [
                "stable_needs",
                "recent_focus",
                "recent_positions",
                "cross_category_pattern",
                "diversity",
            ],
        },
    },
}

SYSTEM_PROMPT = """You summarize a user's observed shopping history for a downstream recommender.
Use only the supplied historical items. You never see or infer a held-out target, candidates, labels, or Interest IDs.
Describe grounded needs and transitions, not a prediction. Do not name any taxonomy category absent from the history.
Keep each text field concise. Every need and recent focus must cite valid 1-based history positions.
For cross_category_pattern, write "none" unless at least two observed categories support a concrete shared use case."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--id-column")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()


def select_balanced(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    if per_class <= 0:
        raise ValueError("--per-class must be positive")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["target_interest"], []).append(row)
    selected = []
    for label in sorted(grouped):
        ordered = sorted(grouped[label], key=lambda row: stable_key(seed, row["sample_id"]))
        if len(ordered) < per_class:
            raise ValueError(f"{label}: only {len(ordered)} rows for quota {per_class}")
        selected.extend(ordered[:per_class])
    return sorted(selected, key=lambda row: row["sample_id"])


def history_payload(row: dict) -> list[dict[str, Any]]:
    titles = row["historical_titles"]
    interests = row["historical_interests"]
    if not titles or len(titles) != len(interests):
        raise ValueError(f"{row['sample_id']}: invalid aligned history")
    return [
        {"position": index, "observed_category": interest, "product_title": title}
        for index, (interest, title) in enumerate(zip(interests, titles, strict=True), 1)
    ]


def render_user_prompt(row: dict) -> str:
    payload = history_payload(row)
    prompt = (
        "Summarize the following chronological observed history. The final items are the most recent.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if row["target_interest"] in prompt or INTEREST_ID_PATTERN.search(prompt):
        raise ValueError(f"{row['sample_id']}: target or Interest ID leaked into request")
    return prompt


def make_client(instance: str) -> AzureOpenAI:
    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "api://trapi/.default",
    )
    return AzureOpenAI(
        azure_endpoint=f"https://trapi.research.microsoft.com/{instance.strip('/')}/",
        azure_ad_token_provider=token_provider,
        api_version=API_VERSION,
        timeout=180.0,
        max_retries=0,
    )


def validate_summary(row: dict, summary: dict) -> tuple[str, list[str]]:
    history_size = len(row["historical_titles"])
    positions = list(summary["recent_positions"])
    for need in summary["stable_needs"]:
        positions.extend(need["positions"])
    errors = []
    if not positions or any(not 1 <= position <= history_size for position in positions):
        errors.append("invalid_evidence_position")
    rendered = (
        "Stable needs: "
        + "; ".join(
            f"{need['need']} [items {','.join(map(str, need['positions']))}]"
            for need in summary["stable_needs"]
        )
        + f"\nRecent focus: {summary['recent_focus']} "
        + f"[items {','.join(map(str, summary['recent_positions']))}]"
        + f"\nCross-category pattern: {summary['cross_category_pattern']}"
        + f"\nHistory diversity: {summary['diversity']}"
    )
    if INTEREST_ID_PATTERN.search(rendered):
        errors.append("interest_id_leak")
    if row["target_interest"].casefold() in rendered.casefold():
        errors.append("exact_target_mention")
    unseen_categories = {
        category.casefold()
        for category in row["all_categories"]
        if category not in set(row["historical_interests"])
    }
    mentioned_unseen = sorted(
        category for category in unseen_categories if category in rendered.casefold()
    )
    if mentioned_unseen:
        errors.append("unseen_taxonomy_mention")
    return rendered, errors


def request_summary(
    client: AzureOpenAI,
    args: argparse.Namespace,
    row: dict,
) -> dict:
    prompt = render_user_prompt(row)
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=SUMMARY_SCHEMA,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=1600,
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                raise ValueError("empty model response")
            summary = json.loads(content)
            rendered, validation_errors = validate_summary(row, summary)
            usage = response.usage
            return {
                "schema_version": 1,
                "sample_id": row["sample_id"],
                "split": args.split,
                "target_interest": row["target_interest"],
                "history_length": len(row["historical_titles"]),
                "summary": summary,
                "rendered_summary": rendered,
                "validation_errors": validation_errors,
                "valid": not validation_errors,
                "attempts": attempt + 1,
                "latency_seconds": time.monotonic() - started,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                "error": None,
            }
        except Exception as error:
            last_error = error
            code = getattr(error, "status_code", None)
            if attempt >= args.max_retries or (code is not None and code not in RETRYABLE_CODES):
                break
            time.sleep(min(2**attempt, 16) + random.random())
    assert last_error is not None
    return {
        "schema_version": 1,
        "sample_id": row["sample_id"],
        "split": args.split,
        "target_interest": row["target_interest"],
        "valid": False,
        "validation_errors": [],
        "attempts": args.max_retries + 1,
        "latency_seconds": time.monotonic() - started,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_existing(path: Path) -> dict[str, dict]:
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                existing[row["sample_id"]] = row
    return existing


def write_manifest(args: argparse.Namespace, selected: list[dict], results: list[dict]) -> None:
    valid = [row for row in results if row.get("valid")]
    summaries = [" ".join(row["rendered_summary"].casefold().split()) for row in valid]
    errors = Counter(error for row in results for error in row.get("validation_errors", []))
    manifest = {
        "schema_version": 1,
        "status": "complete" if len(results) == len(selected) else "incomplete",
        "split": args.split,
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "instance": args.instance,
        "deployment": args.deployment,
        "reasoning_effort": args.reasoning_effort,
        "per_class": args.per_class,
        "class_count": len({row["target_interest"] for row in selected}),
        "requested": len(selected),
        "recorded": len(results),
        "valid": len(valid),
        "request_errors": sum(row.get("error") is not None for row in results),
        "validation_error_counts": dict(sorted(errors.items())),
        "exact_duplicate_count": len(summaries) - len(set(summaries)),
        "mean_summary_characters": (
            sum(len(row["rendered_summary"]) for row in valid) / len(valid) if valid else None
        ),
        "usage": {
            key: sum((row.get("usage") or {}).get(key) or 0 for row in results)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "summary_schema_sha256": hashlib.sha256(
            json.dumps(SUMMARY_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    id_column = args.id_column or ("id" if args.split == "train" else "sample_id")
    table = pq.read_table(
        args.input,
        columns=[
            id_column,
            "historical_titles",
            "historical_interests",
            "target_interest",
        ],
        memory_map=True,
    )
    all_categories = set(table["target_interest"].to_pylist())
    rows = []
    for raw in table.to_pylist():
        raw["sample_id"] = raw.pop(id_column)
        raw["all_categories"] = all_categories
        rows.append(raw)
    selected = (
        sorted(rows, key=lambda row: row["sample_id"])
        if args.all_rows
        else select_balanced(rows, args.per_class, args.seed)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(args.output)
    pending = [row for row in selected if not existing.get(row["sample_id"], {}).get("valid")]
    print(
        json.dumps(
            {
                "selected": len(selected),
                "recovered_valid": len(selected) - len(pending),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if pending:
        client = make_client(args.instance)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(request_summary, client, args, row): row["sample_id"]
                for row in pending
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                append_jsonl(args.output, result)
                existing[result["sample_id"]] = result
                completed += 1
                if completed % 16 == 0 or completed == len(pending):
                    print(f"completed={completed}/{len(pending)}", flush=True)
    results = [existing[row["sample_id"]] for row in selected if row["sample_id"] in existing]
    write_manifest(args, selected, results)


if __name__ == "__main__":
    main()