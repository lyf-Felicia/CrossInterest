#!/usr/bin/env python3
"""Freeze deterministic minimum-support train and natural-dev MIND views."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "mind")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "mind/balanced",
    )
    parser.add_argument("--samples-per-label", type=int, default=200)
    parser.add_argument("--dev-size", type=int, default=5_000)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_of(row: dict) -> str:
    return row["target_interest"]["subcategory"]


def hash_key(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size < 1 or size > len(rows):
        raise ValueError(f"invalid sample size {size} for {len(rows)} rows")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[label_of(row)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (hash_key(seed, row["sample_id"]), row["sample_id"]))

    total = len(rows)
    allocations = {
        label: min(len(group), max(1, int(size * len(group) / total)))
        for label, group in groups.items()
    }
    if sum(allocations.values()) > size:
        raise ValueError("sample size is too small to retain every label")
    candidates = []
    for label, group in groups.items():
        for rank, row in enumerate(group[allocations[label] :], start=allocations[label]):
            candidates.append(
                (
                    rank / len(group),
                    hash_key(seed + 1, row["sample_id"]),
                    label,
                    row,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _, _, label, _ in candidates[: size - sum(allocations.values())]:
        allocations[label] += 1

    selected = [row for label, group in groups.items() for row in group[: allocations[label]]]
    selected.sort(key=lambda row: (hash_key(seed + 2, row["sample_id"]), row["sample_id"]))
    if len(selected) != size or len({row["sample_id"] for row in selected}) != size:
        raise AssertionError("stratified sampling produced an invalid sample")
    return selected


def fixed_support_sample(
    rows: list[dict], samples_per_label: int, seed: int
) -> tuple[list[dict], dict]:
    if samples_per_label < 1:
        raise ValueError("samples per label must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[label_of(row)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (hash_key(seed, row["sample_id"]), row["sample_id"]))

    selected = []
    label_stats = {}
    for label in sorted(groups):
        group = groups[label]
        if len(group) < samples_per_label:
            continue
        selected.extend(group[:samples_per_label])
        label_stats[label] = {
            "source_rows": len(group),
            "selected_rows": samples_per_label,
        }

    selected.sort(key=lambda row: (hash_key(seed + 2, row["sample_id"]), row["sample_id"]))
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise AssertionError("fixed-support sampling produced duplicate output IDs")
    counts = Counter(label_of(row) for row in selected)
    if set(counts.values()) != {samples_per_label}:
        raise AssertionError("fixed-support sampling did not produce equal label counts")
    dropped_labels = {
        label: len(group)
        for label, group in sorted(groups.items())
        if len(group) < samples_per_label
    }
    return selected, {
        "method": "filter_below_threshold_then_deterministic_fixed_count",
        "samples_per_label": samples_per_label,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "unique_source_rows": len(selected),
        "repeated_rows": 0,
        "kept_label_count": len(label_stats),
        "dropped_label_count": len(dropped_labels),
        "dropped_label_rows": sum(dropped_labels.values()),
        "downsampled_rows": sum(len(groups[label]) for label in label_stats) - len(selected),
        "dropped_labels": dropped_labels,
        "per_label": dict(sorted(label_stats.items())),
    }


def convert(row: dict, max_history: int) -> dict:
    history = list(row["history"])
    retained = history[-max_history:]
    target = label_of(row)
    historical_labels = {item["subcategory"] for item in retained}
    if target in historical_labels:
        raise ValueError(
            f"{row['sample_id']}: target {target!r} occurs in retained history"
        )
    converted = {
        "sample_id": row["sample_id"],
        "user_id": row["user_id"],
        "target_subcategory": target,
        "history_original_count": len(history),
        "history_retained_count": len(retained),
        "history": retained,
    }
    if "source_sample_id" in row:
        converted["source_sample_id"] = row["source_sample_id"]
        converted["balance_replica_index"] = row["balance_replica_index"]
    return converted


def add_choice_fields(rows: list[dict], labels: list[str], seed: int) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_subcategory"]].append(row)
    for target, group in grouped.items():
        group.sort(key=lambda row: (hash_key(seed + 10, row["sample_id"]), row["sample_id"]))
        position_offset = int.from_bytes(hash_key(seed + 11, target), "big") % 10
        for rank, row in enumerate(group):
            negatives = sorted(
                (label for label in labels if label != target),
                key=lambda label: (hash_key(seed + 12, f"{row['sample_id']}:{label}"), label),
            )[:9]
            target_index = (position_offset + rank) % 10
            candidates = list(negatives)
            candidates.insert(target_index, target)
            row["candidate_subcategories"] = candidates
            row["target_index"] = target_index


def write_parquet(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def write_converted_train(
    path: Path, rows: list[dict], max_history: int, batch_size: int = 2_048
) -> dict:
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer = None
    history_original_max = 0
    history_retained_max = 0
    history_cropped_rows = 0
    try:
        for start in range(0, len(rows), batch_size):
            batch = [convert(row, max_history) for row in rows[start : start + batch_size]]
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            history_original_max = max(
                history_original_max,
                max(row["history_original_count"] for row in batch),
            )
            history_retained_max = max(
                history_retained_max,
                max(row["history_retained_count"] for row in batch),
            )
            history_cropped_rows += sum(
                row["history_original_count"] > row["history_retained_count"]
                for row in batch
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("cannot write an empty training set")
    temporary.replace(path)
    return {
        "history_original_max": history_original_max,
        "history_retained_max": history_retained_max,
        "history_cropped_rows": history_cropped_rows,
    }


def write_fixed_support_train(
    source_path: Path,
    output_path: Path,
    samples_per_label: int,
    seed: int,
    max_history: int,
    batch_size: int = 2_048,
) -> tuple[dict, dict]:
    label_counts = Counter(
        row["target_interest"]["subcategory"]
        for row in pq.read_table(source_path, columns=["target_interest"]).to_pylist()
    )
    if not label_counts:
        raise ValueError("source training set is empty")
    kept_labels = {
        label for label, count in label_counts.items() if count >= samples_per_label
    }
    dropped_labels = {
        label: count for label, count in label_counts.items() if count < samples_per_label
    }
    selected_heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    parquet = pq.ParquetFile(source_path)
    for record_batch in parquet.iter_batches(
        batch_size=batch_size, columns=["sample_id", "target_interest"]
    ):
        for row in record_batch.to_pylist():
            label = label_of(row)
            if label not in kept_labels:
                continue
            priority = int.from_bytes(hash_key(seed, row["sample_id"]), "big")
            candidate = (-priority, row["sample_id"])
            heap = selected_heaps[label]
            if len(heap) < samples_per_label:
                heapq.heappush(heap, candidate)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
    selected_ids = {
        sample_id for heap in selected_heaps.values() for _, sample_id in heap
    }
    expected_rows = len(kept_labels) * samples_per_label
    if len(selected_heaps) != len(kept_labels) or len(selected_ids) != expected_rows:
        raise AssertionError("failed to select the fixed number of rows per retained label")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = None
    history_original_max = 0
    history_retained_max = 0
    history_cropped_rows = 0

    def write_rows(rows: list[dict]) -> None:
        nonlocal writer, history_original_max, history_retained_max, history_cropped_rows
        converted = [convert(row, max_history) for row in rows]
        table = pa.Table.from_pylist(converted)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        history_original_max = max(
            history_original_max,
            max(row["history_original_count"] for row in converted),
        )
        history_retained_max = max(
            history_retained_max,
            max(row["history_retained_count"] for row in converted),
        )
        history_cropped_rows += sum(
            row["history_original_count"] > row["history_retained_count"]
            for row in converted
        )

    try:
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            selected_rows = [
                row for row in record_batch.to_pylist() if row["sample_id"] in selected_ids
            ]
            if selected_rows:
                write_rows(selected_rows)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("cannot write an empty training set")
    temporary.replace(output_path)

    per_label = {}
    for label, source_rows in sorted(label_counts.items()):
        if label not in kept_labels:
            continue
        per_label[label] = {
            "source_rows": source_rows,
            "selected_rows": samples_per_label,
        }
    balance_report = {
        "method": "filter_below_threshold_then_deterministic_fixed_count",
        "samples_per_label": samples_per_label,
        "source_rows": sum(label_counts.values()),
        "selected_rows": expected_rows,
        "unique_source_rows": expected_rows,
        "repeated_rows": 0,
        "kept_label_count": len(kept_labels),
        "dropped_label_count": len(dropped_labels),
        "dropped_label_rows": sum(dropped_labels.values()),
        "downsampled_rows": sum(label_counts[label] for label in kept_labels) - expected_rows,
        "dropped_labels": dict(sorted(dropped_labels.items())),
        "per_label": per_label,
    }
    history_report = {
        "history_original_max": history_original_max,
        "history_retained_max": history_retained_max,
        "history_cropped_rows": history_cropped_rows,
    }
    return balance_report, history_report


def summarize(rows: list[dict]) -> dict:
    counts = Counter(row["target_subcategory"] for row in rows)
    source_ids = [row.get("source_sample_id", row["sample_id"]) for row in rows]
    return {
        "rows": len(rows),
        "unique_source_rows": len(set(source_ids)),
        "repeated_rows": len(rows) - len(set(source_ids)),
        "labels": len(counts),
        "label_counts": dict(sorted(counts.items())),
        "history_original_max": max(row["history_original_count"] for row in rows),
        "history_retained_max": max(row["history_retained_count"] for row in rows),
        "history_cropped_rows": sum(
            row["history_original_count"] > row["history_retained_count"] for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {split: data_dir / f"{split}.parquet" for split in ("train", "dev")}
    train_path = output_dir / "train.parquet"
    balance_report, train_history = write_fixed_support_train(
        source_paths["train"],
        train_path,
        args.samples_per_label,
        args.seed,
        args.max_history,
    )
    train_labels = sorted(balance_report["per_label"])

    source_dev_rows = [
        row
        for row in pq.read_table(source_paths["dev"]).to_pylist()
        if label_of(row) in set(train_labels)
    ]
    sampled_dev = stratified_sample(source_dev_rows, args.dev_size, args.seed)
    dev_rows = [convert(row, args.max_history) for row in sampled_dev]
    dev_labels = {row["target_subcategory"] for row in dev_rows}
    if not dev_labels <= set(train_labels):
        raise ValueError(f"dev contains unseen labels: {sorted(dev_labels - set(train_labels))}")
    add_choice_fields(dev_rows, train_labels, args.seed)

    train_counts = {
        label: statistics["selected_rows"]
        for label, statistics in balance_report["per_label"].items()
    }
    outputs = {
        "train": {
            "path": train_path.name,
            "sha256": file_sha256(train_path),
            "rows": balance_report["selected_rows"],
            "unique_source_rows": balance_report["unique_source_rows"],
            "repeated_rows": balance_report["repeated_rows"],
            "labels": len(train_counts),
            "label_counts": train_counts,
            **train_history,
        }
    }
    dev_path = output_dir / "dev.parquet"
    write_parquet(dev_path, dev_rows)
    outputs["dev"] = {
        "path": dev_path.name,
        "sha256": file_sha256(dev_path),
        **summarize(dev_rows),
    }
    labels_path = output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(train_labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 3,
        "seed": args.seed,
        "sampling": {
            "train": balance_report,
            "dev": "per-label proportional allocation with one-sample floor and SHA-256 order",
        },
        "history_policy": f"last_{args.max_history}_chronological_items",
        "views": {
            "behavior": ["history.title"],
            "interest": ["history.subcategory"],
        },
        "assistant_target": "New topic: <subcategory>",
        "choice_evaluation": {
            "candidate_count": 10,
            "negative_count": 9,
            "negative_source": "full frozen train taxonomy excluding gold",
            "negative_selection": "lowest per-sample SHA-256 priorities",
            "target_position": "per-label SHA-256 offset plus within-label rank modulo 10",
        },
        "sources": {
            split: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for split, path in source_paths.items()
        },
        "outputs": outputs,
        "labels": {"path": labels_path.name, "sha256": file_sha256(labels_path), "count": len(train_labels)},
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()