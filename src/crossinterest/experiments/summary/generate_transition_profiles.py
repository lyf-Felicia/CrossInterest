#!/usr/bin/env python3
"""Generate target-aware category transition profiles from aggregate Train evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI


API_VERSION = "2024-12-01-preview"
DEFAULT_INSTANCE = "redmond/interactive"
DEFAULT_DEPLOYMENT = "gpt-5.6-sol_2026-07-09"
RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRY_AFTER_PATTERN = re.compile(r"retry(?: after| in) (\d+) seconds?", re.IGNORECASE)
INTEREST_ID_PATTERN = re.compile(r"<interest_\d{3}>", re.IGNORECASE)
WRITE_LOCK = threading.Lock()

SYSTEM_PROMPT = """You design auxiliary supervision for a next-interest recommender.
Given one held-out target category and aggregate statistics from labeled training histories, describe grounded transition mechanisms that can make that category follow observed shopping interests.
Use only source categories present in the supplied statistics. Do not invent causality or mention Interest IDs.
Prefer specific complements, project progressions, seasonal needs, replacements, and shared room or event contexts.
Each cue must be useful only when at least one named source category is actually observed. Keep all text concise."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-sources", type=int, default=24)
    parser.add_argument("--instance", default=DEFAULT_INSTANCE)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_schema(categories: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "category_transition_profile",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category_profile": {"type": "string"},
                    "transition_cues": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "relation_type": {
                                    "type": "string",
                                    "enum": [
                                        "complement",
                                        "project_progression",
                                        "seasonal_event",
                                        "replacement_upgrade",
                                        "shared_context",
                                    ],
                                },
                                "source_categories": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 3,
                                    "items": {"type": "string", "enum": categories},
                                },
                                "explanation": {"type": "string"},
                            },
                            "required": [
                                "relation_type",
                                "source_categories",
                                "explanation",
                            ],
                        },
                    },
                },
                "required": ["category_profile", "transition_cues"],
            },
        },
    }


def aggregate_statistics(rows: list[dict], top_sources: int) -> dict[str, list[dict]]:
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_target[row["target_interest"]].append(row)
    output = {}
    for target, target_rows in sorted(by_target.items()):
        users = Counter()
        recent = Counter()
        latest_titles: dict[str, Counter] = defaultdict(Counter)
        for row in target_rows:
            interests = row["historical_interests"]
            titles = row["historical_titles"]
            users.update(set(interests))
            recent.update(set(interests[-5:]))
            for category, title in zip(interests[-5:], titles[-5:], strict=True):
                latest_titles[category][title] += 1
        ordered = sorted(
            users,
            key=lambda category: (-recent[category], -users[category], category),
        )[:top_sources]
        output[target] = [
            {
                "source_category": category,
                "history_support": users[category],
                "recent_support": recent[category],
                "representative_titles": [
                    title for title, _ in latest_titles[category].most_common(3)
                ],
            }
            for category in ordered
        ]
    return output


def make_client(instance: str) -> AzureOpenAI:
    provider = get_bearer_token_provider(AzureCliCredential(), "api://trapi/.default")
    return AzureOpenAI(
        azure_endpoint=f"https://trapi.research.microsoft.com/{instance.strip('/')}/",
        azure_ad_token_provider=provider,
        api_version=API_VERSION,
        timeout=180.0,
        max_retries=0,
    )


def request_profile(
    client: AzureOpenAI,
    args: argparse.Namespace,
    target: str,
    statistics: list[dict],
    schema: dict,
) -> dict:
    prompt = json.dumps(
        {"target_category": target, "aggregate_predecessor_evidence": statistics},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    allowed_sources = {row["source_category"] for row in statistics}
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
                response_format=schema,
                reasoning_effort=args.reasoning_effort,
                max_completion_tokens=2400,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty model response")
            profile = json.loads(content)
            source_categories = [
                category
                for cue in profile["transition_cues"]
                for category in cue["source_categories"]
            ]
            if any(category not in allowed_sources for category in source_categories):
                raise ValueError("profile cites a source absent from aggregate evidence")
            if INTEREST_ID_PATTERN.search(content):
                raise ValueError("profile contains an Interest ID")
            usage = response.usage
            return {
                "schema_version": 1,
                "target_interest": target,
                "profile": profile,
                "valid": True,
                "attempts": attempt + 1,
                "latency_seconds": time.monotonic() - started,
                "usage": {
                    key: getattr(usage, key, None)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "error": None,
            }
        except Exception as error:
            last_error = error
            code = getattr(error, "status_code", None)
            if attempt >= args.max_retries or (code is not None and code not in RETRYABLE_CODES):
                break
            match = RETRY_AFTER_PATTERN.search(str(error))
            delay = int(match.group(1)) if match else min(2**attempt, 16)
            time.sleep(delay + random.random())
    assert last_error is not None
    return {
        "schema_version": 1,
        "target_interest": target,
        "valid": False,
        "attempts": args.max_retries + 1,
        "latency_seconds": time.monotonic() - started,
        "error": f"{type(last_error).__name__}: {last_error}",
    }


def load_existing(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows[row["target_interest"]] = row
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    rows = pq.read_table(
        args.train,
        columns=["historical_titles", "historical_interests", "target_interest"],
        memory_map=True,
    ).to_pylist()
    statistics = aggregate_statistics(rows, args.top_sources)
    mapping = json.loads(args.label_map.read_text())
    categories = sorted(mapping["category_to_id"])
    if set(statistics) != set(categories) or len(categories) != 112:
        raise ValueError("Train targets and the 112-label mapping differ")
    schema = response_schema(categories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(args.output)
    pending = [target for target in categories if not existing.get(target, {}).get("valid")]
    if pending:
        client = make_client(args.instance)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    request_profile,
                    client,
                    args,
                    target,
                    statistics[target],
                    schema,
                ): target
                for target in pending
            }
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                append_jsonl(args.output, result)
                print(
                    f"[{completed}/{len(pending)}] {result['target_interest']}: "
                    f"{'ok' if result.get('valid') else result['error']}",
                    flush=True,
                )
    latest = load_existing(args.output)
    valid = [latest[target] for target in categories if latest.get(target, {}).get("valid")]
    manifest = {
        "schema_version": 1,
        "status": "complete" if len(valid) == len(categories) else "incomplete",
        "train": str(args.train.resolve()),
        "train_sha256": sha256_file(args.train),
        "label_map": str(args.label_map.resolve()),
        "label_map_sha256": sha256_file(args.label_map),
        "output": str(args.output.resolve()),
        "target_count": len(categories),
        "valid": len(valid),
        "aggregate_rows_per_target": len(rows) // len(categories),
        "top_sources": args.top_sources,
        "instance": args.instance,
        "deployment": args.deployment,
        "reasoning_effort": args.reasoning_effort,
        "usage": {
            key: sum((row.get("usage") or {}).get(key) or 0 for row in valid)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()