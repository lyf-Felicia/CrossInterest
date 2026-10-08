#!/usr/bin/env python3
"""Verify the Amazon new-interest build contract from emitted artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--formal-dir",
        type=Path,
        required=True,
        help="Verify formal Train/Dev JSONL staging files in this directory.",
    )
    return parser.parse_args()


def inspect(path: Path, expected_split: str, minimum_history: int) -> dict:
    ids = set()
    users = set()
    labels = Counter()
    events_by_user: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            record_id = record["id"]
            if record_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id}")
            ids.add(record_id)
            if record["split"] != expected_split:
                raise ValueError(f"{record_id}: unexpected split")
            user_id = record["user_id"]
            users.add(user_id)
            history_lengths = {
                len(record["history_parent_asins"]),
                len(record["historical_titles"]),
                len(record["historical_level_2"]),
                len(record["historical_interests"]),
                len(record["history_timestamps"]),
            }
            if len(history_lengths) != 1:
                raise ValueError(f"{record_id}: history fields have different lengths")
            visible_length = history_lengths.pop()
            if not minimum_history <= record["full_history_length"]:
                raise ValueError(f"{record_id}: history shorter than minimum")
            if visible_length != min(record["full_history_length"], record["history_limit"]):
                raise ValueError(f"{record_id}: visible history length mismatch")
            if any(value >= record["timestamp"] for value in record["history_timestamps"]):
                raise ValueError(f"{record_id}: non-causal history timestamp")
            if record["target_interest"] in record["full_seen_interests"]:
                raise ValueError(f"{record_id}: target occurs in complete history")
            if not set(record["historical_interests"]) <= set(record["full_seen_interests"]):
                raise ValueError(f"{record_id}: visible labels missing from complete history")
            current = (record["timestamp"], record["full_history_length"])
            events_by_user[user_id].append(current)
            labels[record["target_interest"]] += 1
    for user_id, events in events_by_user.items():
        ordered = sorted(events)
        for previous, current in zip(ordered, ordered[1:]):
            if current[0] <= previous[0] or current[1] <= previous[1]:
                raise ValueError(f"{user_id}: user history does not advance")
    return {"ids": ids, "users": users, "labels": labels}


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.work_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    minimum_history = manifest["contract"]["minimum_history"]
    data = args.work_dir / "data"
    train = inspect(data / "unbalanced" / "train.jsonl", "train", minimum_history)
    dev = inspect(data / "unbalanced" / "dev.jsonl", "dev", minimum_history)
    balanced = inspect(args.formal_dir / "train.jsonl", "train", minimum_history)
    overlap = train["users"] & dev["users"]
    if overlap:
        raise ValueError(f"train/dev user leakage: {len(overlap)}")
    if not balanced["ids"] <= train["ids"]:
        raise ValueError("balanced train is not a subset of unbalanced train")
    expected_quota = manifest["balanced_train"]["per_label_count"]
    if set(balanced["labels"].values()) != {expected_quota}:
        raise ValueError("balanced train label counts are not equal")
    expected_dev = Counter(manifest["unbalanced"]["dev"]["target_counts"])
    if dev["labels"] != expected_dev:
        raise ValueError("dev distribution differs from the natural mother pool")
    evaluation_dir = data / "evaluation"
    closed = inspect(args.formal_dir / "dev.jsonl", "dev", minimum_history)
    oov = inspect(evaluation_dir / "dev_oov.jsonl", "dev", minimum_history)
    if closed["ids"] & oov["ids"]:
        raise ValueError("closed-set and OOV Dev views overlap")
    if closed["ids"] | oov["ids"] != dev["ids"]:
        raise ValueError("evaluation views do not partition the natural Dev pool")
    support = set(manifest["balanced_train"]["support"])
    if set(closed["labels"]) != support:
        raise ValueError("closed-set Dev labels differ from frozen Train support")
    if set(oov["labels"]) & support:
        raise ValueError("OOV Dev contains a frozen Train label")
    result = {
        "train_samples": len(train["ids"]),
        "dev_samples": len(dev["ids"]),
        "balanced_train_samples": len(balanced["ids"]),
        "train_dev_user_overlap": 0,
        "balanced_train_labels": len(balanced["labels"]),
        "dev_is_unbalanced_mother_pool": True,
        "closed_dev_samples": len(closed["ids"]),
        "oov_dev_samples": len(oov["ids"]),
        "evaluation_views_partition_dev": True,
        "published_dataset_verified": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()