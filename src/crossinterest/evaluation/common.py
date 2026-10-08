"""Shared I/O, parsing, and metrics for Amazon 10-way evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


INPUT_TYPES = ("behavior", "interest", "behavior_interest")
ANSWER_PATTERN = re.compile(r"New\s+topic\s*:\s*(?P<label>[^\r\n]+)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_details(path: Path, fingerprint: str | None = None) -> dict[str, dict]:
    latest = {}
    for row in load_jsonl(path):
        if fingerprint is not None and row.get("run_fingerprint") != fingerprint:
            raise ValueError(f"{path}: cached result fingerprint mismatch")
        latest[row["sample_id"]] = row
    return latest


def prompt_text(row: dict) -> str:
    messages = row["prompt"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError(f"{row['sample_id']}: malformed prompt")
    return messages[0]["content"]


def normalize_candidate_text(value: str) -> str:
    previous = None
    normalized = value.strip()
    while normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip("`*_ ").rstrip(".! ")
    return normalized


def score_response(raw_response: str, candidates: list[str], target: str) -> dict:
    stripped = raw_response.strip()
    matches = list(ANSWER_PATTERN.finditer(stripped))
    exact_candidate = next(
        (candidate for candidate in candidates if candidate == stripped),
        None,
    )
    predicted = None
    extraction_method = "unresolved"
    if exact_candidate is not None:
        predicted = exact_candidate
        extraction_method = "exact_candidate"
    elif matches:
        raw_label = normalize_candidate_text(matches[-1].group("label"))
        predicted = next(
            (candidate for candidate in candidates if candidate.casefold() == raw_label.casefold()),
            None,
        )
        extraction_method = "new_topic_marker"
    if predicted is None:
        final_line = normalize_candidate_text(stripped.splitlines()[-1]) if stripped else ""
        predicted = next(
            (candidate for candidate in candidates if candidate.casefold() == final_line.casefold()),
            None,
        )
        if predicted is not None:
            extraction_method = "final_line"
    return {
        "prediction": predicted,
        "correct": predicted == target,
        "format_valid": exact_candidate is not None,
        "resolved": predicted is not None,
        "extraction_method": extraction_method,
    }


def metrics_from_details(rows: list[dict], expected_rows: list[dict]) -> dict:
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    actual_by_id = {row["sample_id"]: row for row in rows}
    unknown = set(actual_by_id) - set(expected_by_id)
    if unknown:
        raise ValueError(f"details contain unknown sample IDs: {sorted(unknown)[:3]}")
    per_label_total = Counter(row["target_interest"] for row in expected_rows)
    per_label_correct = Counter()
    successful = 0
    correct = 0
    format_valid = 0
    resolved = 0
    errors = 0
    latencies = []
    finish_reasons = Counter()
    for expected in expected_rows:
        detail = actual_by_id.get(expected["sample_id"])
        if detail is None:
            continue
        if detail.get("error") is not None:
            errors += 1
            continue
        successful += 1
        is_correct = bool(detail["correct"])
        correct += is_correct
        per_label_correct[expected["target_interest"]] += is_correct
        format_valid += bool(detail["format_valid"])
        resolved += bool(detail["resolved"])
        if detail.get("latency_seconds") is not None:
            latencies.append(detail["latency_seconds"])
        if detail.get("finish_reason") is not None:
            finish_reasons[str(detail["finish_reason"])] += 1
    requested = len(expected_rows)
    per_class = {
        label: per_label_correct[label] / count
        for label, count in sorted(per_label_total.items())
    }
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "requested_sample_count": requested,
        "recorded_sample_count": len(actual_by_id),
        "successful_sample_count": successful,
        "request_error_count": errors,
        "missing_sample_count": requested - len(actual_by_id),
        "macro_accuracy": macro,
        "micro_accuracy": correct / requested if requested else 0.0,
        "correct_count": correct,
        "class_count": len(per_class),
        "per_class_accuracy": per_class,
        "strict_format_rate": format_valid / requested if requested else 0.0,
        "resolved_prediction_rate": resolved / requested if requested else 0.0,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "complete": len(actual_by_id) == requested and errors == 0,
    }


def write_predictions(path: Path, rows: list[dict], expected_rows: list[dict]) -> None:
    expected_ids = {row["sample_id"] for row in expected_rows}
    successful = {
        row["sample_id"]: row
        for row in rows
        if row["sample_id"] in expected_ids and row.get("error") is None
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for expected in expected_rows:
            row = successful.get(expected["sample_id"])
            if row is not None:
                handle.write(
                    json.dumps(
                        {"sample_id": row["sample_id"], "prediction": row["prediction"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    temporary.replace(path)