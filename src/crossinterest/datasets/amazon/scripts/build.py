#!/usr/bin/env python3
"""Build a user-disjoint Amazon Reviews'23 new-interest dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import sqlite3
import sys
import urllib.request
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


DATASET_REPO = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
DOMAIN = "Home_and_Kitchen"
ROOT_CATEGORY = "Home & Kitchen"
DEFAULT_SEED = 20260806
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_MIN_HISTORY = 5
DEFAULT_DEV_FRACTION = 0.10
DEFAULT_MIN_RATING = 4.0
DEFAULT_MIN_LABEL_SUPPORT = 200
SCHEMA_VERSION = 1

BASE_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET_REPO}/resolve/{DATASET_REVISION}"
)
DEFAULT_METADATA_URL = f"{BASE_URL}/raw/meta_categories/meta_{DOMAIN}.jsonl"
DEFAULT_RATINGS_URL = (
    f"{BASE_URL}/benchmark/0core/rating_only/{DOMAIN}.csv"
)

NON_TAXONOMY_TERMS = (
    "deal",
    "markdown",
    "featured",
    "seller storefront",
    "special offer",
    "prime day",
)
ALLOWED_LEVEL_2 = frozenset(
    {
        "Bath",
        "Bedding",
        "Event & Party Supplies",
        "Furniture",
        "Heating, Cooling & Air Quality",
        "Home Décor Products",
        "Irons & Steamers",
        "Kitchen & Dining",
        "Seasonal Décor",
        "Storage & Organization",
        "Vacuums & Floor Care",
        "Wall Art",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--formal-dir",
        type=Path,
        required=True,
        help="Write the formal balanced Train JSONL staging file in this directory.",
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA_URL)
    parser.add_argument("--ratings", default=DEFAULT_RATINGS_URL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--min-history", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--dev-fraction", type=float, default=DEFAULT_DEV_FRACTION)
    parser.add_argument("--min-rating", type=float, default=DEFAULT_MIN_RATING)
    parser.add_argument(
        "--min-label-support", type=int, default=DEFAULT_MIN_LABEL_SUPPORT
    )
    parser.add_argument(
        "--train-quota",
        type=int,
        help="Optional equal per-label quota; default is the minimum actual support among retained labels.",
    )
    parser.add_argument(
        "--balance-only",
        action="store_true",
        help="Rebuild balanced Train from an existing unbalanced Train mother pool.",
    )
    parser.add_argument("--rebuild-metadata", action="store_true")
    return parser.parse_args()


def compact(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_digest(seed: int, *parts: str) -> bytes:
    payload = ":".join((str(seed), *parts)).encode()
    return hashlib.sha256(payload).digest()


def stable_rng(seed: int, *parts: str) -> random.Random:
    return random.Random(int.from_bytes(stable_digest(seed, *parts)[:8], "big"))


def user_split(user_id: str, seed: int, dev_fraction: float) -> str:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    bucket = int.from_bytes(stable_digest(seed, "user-split", user_id)[:8], "big")
    threshold = int(dev_fraction * 10_000)
    return "dev" if bucket % 10_000 < threshold else "train"


def open_binary(source: str):
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            source, headers={"User-Agent": "crossinterest-amazon-builder/1.0"}
        )
        return closing(urllib.request.urlopen(request))
    return Path(source).open("rb")


def iter_text_lines(source: str) -> Iterator[str]:
    with open_binary(source) as handle:
        for line in handle:
            yield line.decode("utf-8")


def clean_taxonomy(row: dict) -> tuple[str, str] | None:
    path = [str(value).strip() for value in row.get("categories") or []]
    path = [value for value in path if value]
    if len(path) < 3 or path[0] != ROOT_CATEGORY:
        return None
    level_2, level_3 = path[1], path[2]
    if level_2 not in ALLOWED_LEVEL_2:
        return None
    normalized = f"{level_2} {level_3}".lower()
    if any(term in normalized for term in NON_TAXONOMY_TERMS):
        return None
    if level_2 == ROOT_CATEGORY or level_3 in {ROOT_CATEGORY, level_2}:
        return None
    return level_2, f"{level_2} > {level_3}"


def connect_metadata(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            parent_asin TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            level_2 TEXT NOT NULL,
            label TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def build_metadata_index(source: str, path: Path) -> dict:
    temporary = path.with_suffix(".tmp.sqlite")
    temporary.unlink(missing_ok=True)
    connection = connect_metadata(temporary)
    stats = Counter()
    batch = []
    for line in iter_text_lines(source):
        stats["rows"] += 1
        row = json.loads(line)
        taxonomy = clean_taxonomy(row)
        title = str(row.get("title") or "").strip()
        parent_asin = str(row.get("parent_asin") or "").strip()
        if not taxonomy or not title or not parent_asin:
            stats["rejected"] += 1
            continue
        level_2, label = taxonomy
        batch.append((parent_asin, title, level_2, label))
        if len(batch) >= 10_000:
            connection.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?)", batch)
            connection.commit()
            batch.clear()
        if stats["rows"] % 250_000 == 0:
            print(f"metadata: {stats['rows']:,} rows", file=sys.stderr, flush=True)
    if batch:
        connection.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?)", batch)
    connection.commit()
    stats["accepted"] = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    connection.close()
    temporary.replace(path)
    return dict(stats)


def load_products(path: Path) -> dict[str, tuple[str, str, str]]:
    connection = sqlite3.connect(path)
    products = {
        parent_asin: (title, level_2, label)
        for parent_asin, title, level_2, label in connection.execute(
            "SELECT parent_asin,title,level_2,label FROM products"
        )
    }
    connection.close()
    return products


def iter_positive_users(
    source: str, min_rating: float
) -> Iterator[tuple[str, list[tuple[int, str, float]]]]:
    lines = iter_text_lines(source)
    reader = csv.DictReader(lines)
    expected = ["user_id", "parent_asin", "rating", "timestamp"]
    if reader.fieldnames != expected:
        raise ValueError(f"unexpected rating columns: {reader.fieldnames}")
    current_user = None
    interactions: list[tuple[int, str, float]] = []
    for row in reader:
        user_id = row["user_id"]
        if current_user is not None and user_id != current_user:
            yield current_user, interactions
            interactions = []
        current_user = user_id
        rating = float(row["rating"])
        if rating >= min_rating:
            interactions.append((int(row["timestamp"]), row["parent_asin"], rating))
    if current_user is not None:
        yield current_user, interactions


def event_groups(
    interactions: Iterable[tuple[int, str, float]],
) -> Iterator[tuple[int, list[tuple[int, str, float]]]]:
    ordered = sorted(interactions, key=lambda value: (value[0], value[1]))
    start = 0
    while start < len(ordered):
        timestamp = ordered[start][0]
        end = start + 1
        while end < len(ordered) and ordered[end][0] == timestamp:
            end += 1
        yield timestamp, ordered[start:end]
        start = end


def choose_target(
    user_id: str,
    timestamp: int,
    candidates: dict[str, list[tuple[str, str, str, float]]],
    seed: int,
) -> tuple[str, str, str, str, float]:
    labels = sorted(candidates)
    rng = stable_rng(seed, "target", user_id, str(timestamp))
    label = labels[rng.randrange(len(labels))]
    products = sorted(candidates[label])
    parent_asin, title, level_2, rating = products[rng.randrange(len(products))]
    return parent_asin, title, level_2, label, rating


def build_user_records(
    user_id: str,
    interactions: Iterable[tuple[int, str, float]],
    products: dict[str, tuple[str, str, str]],
    split: str,
    seed: int,
    history_limit: int,
    min_history: int = DEFAULT_MIN_HISTORY,
) -> list[dict]:
    full_seen_labels: set[str] = set()
    seen_products: set[str] = set()
    history: list[tuple[int, str, str, str, str, float]] = []
    records = []
    for timestamp, group in event_groups(interactions):
        valid = []
        for _, parent_asin, rating in group:
            product = products.get(parent_asin)
            if product is None or parent_asin in seen_products:
                continue
            title, level_2, label = product
            valid.append((parent_asin, title, level_2, label, rating))
        new_by_label: dict[str, list[tuple[str, str, str, float]]] = {}
        for parent_asin, title, level_2, label, rating in valid:
            if label not in full_seen_labels:
                new_by_label.setdefault(label, []).append(
                    (parent_asin, title, level_2, rating)
                )
        if len(history) >= min_history and new_by_label:
            target_asin, target_title, target_level_2, target_label, rating = choose_target(
                user_id, timestamp, new_by_label, seed
            )
            visible = history[-history_limit:]
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": f"{split}:{user_id}:{timestamp}",
                    "split": split,
                    "domain": DOMAIN,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "timestamp_iso": datetime.fromtimestamp(
                        timestamp / 1000, tz=timezone.utc
                    ).isoformat(),
                    "history_limit": history_limit,
                    "full_history_length": len(history),
                    "full_seen_interests": sorted(full_seen_labels),
                    "history_parent_asins": [item[1] for item in visible],
                    "historical_titles": [item[2] for item in visible],
                    "historical_level_2": [item[3] for item in visible],
                    "historical_interests": [item[4] for item in visible],
                    "history_timestamps": [item[0] for item in visible],
                    "target_parent_asin": target_asin,
                    "target_title": target_title,
                    "target_level_2": target_level_2,
                    "target_interest": target_label,
                    "target_rating": rating,
                    "same_history_new_interest_count": len(new_by_label),
                    "target_selection": "seeded_uniform_label_then_product",
                }
            )
        for parent_asin, title, level_2, label, rating in valid:
            history.append((timestamp, parent_asin, title, level_2, label, rating))
            seen_products.add(parent_asin)
            full_seen_labels.add(label)
    return records


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle = self.temporary.open("w", encoding="utf-8")
        self.count = 0

    def write(self, record: dict) -> None:
        self.handle.write(compact(record) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.close()
        self.temporary.replace(self.path)


def collect_mother_pool(
    ratings: str,
    products: dict[str, tuple[str, str, str]],
    output_dir: Path,
    seed: int,
    dev_fraction: float,
    min_rating: float,
    history_limit: int,
    min_history: int,
) -> dict:
    writers = {
        split: JsonlWriter(output_dir / "unbalanced" / f"{split}.jsonl")
        for split in ("train", "dev")
    }
    stats = {
        split: {"users": 0, "positive_interactions": 0, "samples": 0, "labels": Counter()}
        for split in writers
    }
    for user_number, (user_id, interactions) in enumerate(
        iter_positive_users(ratings, min_rating), 1
    ):
        split = user_split(user_id, seed, dev_fraction)
        split_stats = stats[split]
        split_stats["users"] += 1
        split_stats["positive_interactions"] += len(interactions)
        records = build_user_records(
            user_id, interactions, products, split, seed, history_limit, min_history
        )
        for record in records:
            writers[split].write(record)
            split_stats["samples"] += 1
            split_stats["labels"][record["target_interest"]] += 1
        if user_number % 250_000 == 0:
            print(f"ratings: {user_number:,} users", file=sys.stderr, flush=True)
    for writer in writers.values():
        writer.close()
    return stats


def build_balanced_train(
    source: Path,
    destination: Path,
    seed: int,
    min_label_support: int,
    requested_quota: int | None = None,
) -> dict:
    counts = Counter()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            counts[json.loads(line)["target_interest"]] += 1
    support = sorted(
        label for label, count in counts.items() if count >= min_label_support
    )
    if not support:
        raise ValueError(
            f"no label has at least {min_label_support} Train samples"
        )
    minimum_actual_support = min(counts[label] for label in support)
    quota = requested_quota or minimum_actual_support
    if quota > minimum_actual_support:
        raise ValueError(
            f"train quota {quota} exceeds minimum retained support "
            f"{minimum_actual_support}"
        )
    support_set = set(support)
    selected: dict[str, list[tuple[int, str, str]]] = {label: [] for label in support}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = record["target_interest"]
            if label not in support_set:
                continue
            key = int.from_bytes(
                stable_digest(seed, "balanced-train", record["id"]), "big"
            )
            entry = (-key, record["id"], line)
            heap = selected[label]
            if len(heap) < quota:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    writer = JsonlWriter(destination)
    for label in support:
        for _, _, line in sorted(selected[label], reverse=True):
            writer.write(json.loads(line))
    writer.close()
    return {
        "support": support,
        "support_size": len(support),
        "min_label_support": min_label_support,
        "minimum_actual_support": minimum_actual_support,
        "quota_source": "explicit" if requested_quota else "minimum_actual_support",
        "per_label_count": quota,
        "sample_count": len(support) * quota,
    }


def serializable_stats(stats: dict) -> dict:
    result = {}
    for split, values in stats.items():
        result[split] = {
            **{key: value for key, value in values.items() if key != "labels"},
            "label_count": len(values["labels"]),
            "target_counts": dict(values["labels"].most_common()),
        }
    return result


def main() -> None:
    args = parse_args()
    positive_values = [args.min_history, args.history_limit, args.min_label_support]
    if args.train_quota is not None:
        positive_values.append(args.train_quota)
    if min(positive_values) <= 0:
        raise ValueError("history limits and label support parameters must be positive")
    if args.min_history > args.history_limit:
        raise ValueError("min_history cannot exceed history_limit")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.work_dir / "data"
    if args.balance_only:
        source = output_dir / "unbalanced" / "train.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"missing Train mother pool: {source}")
        balanced = build_balanced_train(
            source,
            args.formal_dir / "train.jsonl",
            args.seed,
            args.min_label_support,
            args.train_quota,
        )
        manifest_path = args.work_dir / "MANIFEST.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["created_at"] = datetime.now(timezone.utc).isoformat()
            manifest["contract"]["taxonomy"] = {
                "level_1": "categories[1]",
                "target": "categories[1] > categories[2]",
                "allowed_level_1": sorted(ALLOWED_LEVEL_2),
                "storefront_and_campaign_nodes_excluded": True,
            }
            manifest["contract"]["train_balance"] = (
                "retain every label meeting min_label_support; equal quota is the "
                "minimum actual support unless explicitly overridden"
            )
            manifest["balanced_train"] = balanced
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(balanced, ensure_ascii=False, indent=2))
        return
    metadata_db = args.work_dir / "metadata.sqlite"
    if args.rebuild_metadata or not metadata_db.exists():
        metadata_stats = build_metadata_index(args.metadata, metadata_db)
    else:
        metadata_stats = {"reused": True}
    products = load_products(metadata_db)
    stats = collect_mother_pool(
        args.ratings,
        products,
        output_dir,
        args.seed,
        args.dev_fraction,
        args.min_rating,
        args.history_limit,
        args.min_history,
    )
    balanced = build_balanced_train(
        output_dir / "unbalanced" / "train.jsonl",
        args.formal_dir / "train.jsonl",
        args.seed,
        args.min_label_support,
        args.train_quota,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "domain": DOMAIN,
        "sources": {"metadata": args.metadata, "ratings": args.ratings},
        "seed": args.seed,
        "contract": {
            "positive_interaction": f"rating >= {args.min_rating}",
            "user_split": f"stable SHA-256, train/dev={1-args.dev_fraction:.0%}/{args.dev_fraction:.0%}",
            "user_disjoint": True,
            "new_interest": "target L2>L3 absent from the complete strict history prefix",
            "same_timestamp": "one event group; all interactions share the same pre-event history",
            "multi_target": "seeded uniform choice over new labels, then products",
            "minimum_history": args.min_history,
            "visible_history": f"most recent {args.history_limit} positive interactions",
            "history_update": "append the complete timestamp group after target generation",
            "taxonomy": {
                "level_1": "categories[1]",
                "target": "categories[1] > categories[2]",
                "allowed_level_1": sorted(ALLOWED_LEVEL_2),
                "storefront_and_campaign_nodes_excluded": True,
            },
            "train_balance": (
                "retain every label meeting min_label_support; equal quota is the "
                "minimum actual support unless explicitly overridden"
            ),
            "dev_balance": "none; preserve the natural eligible distribution",
        },
        "metadata": {**metadata_stats, "loaded_products": len(products)},
        "unbalanced": serializable_stats(stats),
        "balanced_train": balanced,
    }
    (args.work_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()