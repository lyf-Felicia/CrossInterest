#!/usr/bin/env python3
"""Build paired 10-way Amazon evaluation views from the formal closed-set Dev."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


SEED = 20260806
EVAL_SIZE = 2_000
INPUT_TYPES = ("behavior", "interest", "behavior_interest")
PROMPT_SCHEMA_VERSION = 3
SOURCE_COLUMNS = (
    "id",
    "full_history_length",
    "full_seen_interests",
    "historical_titles",
    "historical_interests",
    "target_level_2",
    "target_interest",
)
HISTORY_INTRODUCTIONS = {
    "behavior": "The titles of the products I recently rated positively, in chronological order, are listed below:",
    "interest": "The topic categories associated with the products I recently rated positively, in chronological order, are listed below:",
    "behavior_interest": "The titles and corresponding topic categories of the products I recently rated positively, in chronological order, are listed below. Each item is formatted as \"[Topic category] Product title\":",
}
PREDICTION_INSTRUCTIONS = {
    "behavior": "Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should represent a new topic that is not reflected in my product history.",
    "interest": "Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should not appear in the history sequence above.",
    "behavior_interest": "Select the one topic category from the candidates below that I am most likely to be interested in next. The selected category should represent a new topic that does not appear among the topic categories in my product history.",
}
OUTPUT_INSTRUCTION = "Respond with only the exact category name from the candidate list."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--expected-label-count", type=int, default=99)
    return parser.parse_args()


def stable_digest(seed: int, namespace: str, *parts: str) -> bytes:
    payload = ":".join((str(seed), namespace, *parts)).encode()
    return hashlib.sha256(payload).digest()


def stable_order(seed: int, namespace: str, values: Iterable[str], *parts: str) -> list[str]:
    return sorted(values, key=lambda value: stable_digest(seed, namespace, *parts, value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist(rows),
        temporary,
        compression="zstd",
        row_group_size=1_000,
    )
    temporary.replace(path)


def load_label_to_level_2(
    train_path: Path,
    expected_label_count: int = 99,
) -> dict[str, str]:
    table = pq.read_table(train_path, columns=["target_interest", "target_level_2"])
    pairs = set(zip(table["target_interest"].to_pylist(), table["target_level_2"].to_pylist()))
    labels = {label for label, _ in pairs}
    if len(labels) != len(pairs):
        raise ValueError("a target interest maps to more than one L2 category")
    if len(labels) != expected_label_count:
        raise ValueError(
            f"expected {expected_label_count} formal Train labels, found {len(labels)}"
        )
    return dict(sorted(pairs))


def stratified_quotas(labels: list[str], size: int, seed: int) -> dict[str, int]:
    if size < len(labels):
        raise ValueError("evaluation size must be at least the label count")
    base, remainder = divmod(size, len(labels))
    extra_labels = set(stable_order(seed, "quota-extra", labels)[:remainder])
    return {label: base + int(label in extra_labels) for label in labels}


def select_stratified_rows(
    dev_path: Path,
    quotas: dict[str, int],
    seed: int,
) -> list[dict]:
    heaps: dict[str, list[tuple[int, str, dict]]] = {label: [] for label in quotas}
    counts = Counter()
    parquet = pq.ParquetFile(dev_path)
    for batch in parquet.iter_batches(columns=list(SOURCE_COLUMNS), batch_size=10_000):
        for row in batch.to_pylist():
            label = row["target_interest"]
            if label not in quotas:
                raise ValueError(f"Dev contains a target outside formal Train support: {label}")
            counts[label] += 1
            source_id = row["id"]
            priority = int.from_bytes(stable_digest(seed, "eval-sample", source_id), "big")
            item = (-priority, source_id, row)
            heap = heaps[label]
            if len(heap) < quotas[label]:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)
    insufficient = {
        label: {"available": counts[label], "required": quota}
        for label, quota in quotas.items()
        if counts[label] < quota
    }
    if insufficient:
        raise ValueError(f"insufficient Dev support: {insufficient}")
    selected = [item[2] for heap in heaps.values() for item in heap]
    return sorted(
        selected,
        key=lambda row: stable_digest(seed, "eval-output-order", row["id"]),
    )


def build_candidates(
    row: dict,
    label_to_level_2: dict[str, str],
    seed: int,
) -> tuple[list[str], int, int]:
    source_id = row["id"]
    target = row["target_interest"]
    target_level_2 = label_to_level_2[target]
    if row["target_level_2"] != target_level_2:
        raise ValueError(f"{source_id}: target L2 disagrees with formal Train mapping")
    seen = set(row["full_seen_interests"])
    if target in seen:
        raise ValueError(f"{source_id}: target appears in complete history")
    available = [
        label
        for label in label_to_level_2
        if label != target and label not in seen
    ]
    same_level_2 = stable_order(
        seed,
        "same-l2-negative",
        (label for label in available if label_to_level_2[label] == target_level_2),
        source_id,
    )
    cross_level_2 = stable_order(
        seed,
        "cross-l2-negative",
        (label for label in available if label_to_level_2[label] != target_level_2),
        source_id,
    )
    negatives = same_level_2[:3]
    negatives.extend(cross_level_2[: 9 - len(negatives)])
    if len(negatives) != 9:
        raise ValueError(f"{source_id}: only {len(available)} unseen negative labels available")
    candidates = [target, *negatives]
    random.Random(
        int.from_bytes(stable_digest(seed, "candidate-order", source_id)[:8], "big")
    ).shuffle(candidates)
    return candidates, len(same_level_2[:3]), len(available)


def render_history(row: dict, input_type: str) -> str:
    titles = row["historical_titles"]
    interests = row["historical_interests"]
    if len(titles) != len(interests):
        raise ValueError(f"{row['id']}: title and interest histories are not aligned")
    if input_type == "behavior":
        values = titles
    elif input_type == "interest":
        values = interests
    elif input_type == "behavior_interest":
        values = [
            f"[{interest}] {title}"
            for title, interest in zip(titles, interests, strict=True)
        ]
    else:
        raise ValueError(f"unsupported input type: {input_type}")
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))


def render_prompt(row: dict, candidates: list[str], input_type: str) -> str:
    candidate_text = "\n".join(
        f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1)
    )
    return (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}\n\n"
        f"{PREDICTION_INSTRUCTIONS[input_type]}\n\n"
        f"Candidates:\n\n{candidate_text}\n\n"
        f"{OUTPUT_INSTRUCTION}"
    )


def build_view_row(
    source: dict,
    candidates: list[str],
    input_type: str,
    same_level_2_negatives: int,
    available_negative_count: int,
) -> dict:
    target = source["target_interest"]
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "sample_id": f"{source['id']}:{input_type}",
        "source_sample_id": source["id"],
        "input_type": input_type,
        "history_length": len(source["historical_titles"]),
        "full_history_length": source["full_history_length"],
        "historical_titles": source["historical_titles"],
        "historical_interests": source["historical_interests"],
        "full_seen_interests": source["full_seen_interests"],
        "target_interest": target,
        "target_level_2": source["target_level_2"],
        "candidate_interests": candidates,
        "gold_index": candidates.index(target),
        "same_level_2_negative_count": same_level_2_negatives,
        "available_negative_count": available_negative_count,
        "prompt": [{"role": "user", "content": render_prompt(source, candidates, input_type)}],
    }


def validate_views(views: dict[str, list[dict]], expected_size: int) -> dict:
    if set(views) != set(INPUT_TYPES):
        raise ValueError("all three input views are required")
    for input_type, rows in views.items():
        if len(rows) != expected_size:
            raise ValueError(f"{input_type}: expected {expected_size} rows, found {len(rows)}")
        if len({row["source_sample_id"] for row in rows}) != expected_size:
            raise ValueError(f"{input_type}: duplicate source IDs")
        for row in rows:
            candidates = row["candidate_interests"]
            if len(candidates) != 10 or len(set(candidates)) != 10:
                raise ValueError(f"{row['sample_id']}: candidates are not 10 unique labels")
            if row["target_interest"] not in candidates:
                raise ValueError(f"{row['sample_id']}: target is absent from candidates")
            if set(candidates) & set(row["full_seen_interests"]):
                raise ValueError(f"{row['sample_id']}: a candidate appears in complete history")
    reference = views[INPUT_TYPES[0]]
    paired_fields = ("source_sample_id", "target_interest", "candidate_interests", "gold_index")
    for input_type in INPUT_TYPES[1:]:
        for first, other in zip(reference, views[input_type], strict=True):
            if any(first[field] != other[field] for field in paired_fields):
                raise ValueError(f"paired view mismatch for {first['source_sample_id']}")
    label_counts = Counter(row["target_interest"] for row in reference)
    gold_positions = Counter(row["gold_index"] for row in reference)
    return {
        "label_count": len(label_counts),
        "label_count_min": min(label_counts.values()),
        "label_count_max": max(label_counts.values()),
        "gold_position_counts": {str(index): gold_positions[index] for index in range(10)},
        "minimum_available_negative_count": min(
            row["available_negative_count"] for row in reference
        ),
        "same_level_2_negative_counts": dict(
            sorted(Counter(row["same_level_2_negative_count"] for row in reference).items())
        ),
    }


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    label_to_level_2 = load_label_to_level_2(
        args.train,
        args.expected_label_count,
    )
    labels = sorted(label_to_level_2)
    quotas = stratified_quotas(labels, args.size, args.seed)
    selected = select_stratified_rows(args.dev, quotas, args.seed)
    views = {input_type: [] for input_type in INPUT_TYPES}
    for source in selected:
        candidates, same_level_2_count, available_count = build_candidates(
            source,
            label_to_level_2,
            args.seed,
        )
        for input_type in INPUT_TYPES:
            views[input_type].append(
                build_view_row(
                    source,
                    candidates,
                    input_type,
                    same_level_2_count,
                    available_count,
                )
            )
    validation = validate_views(views, args.size)
    output_paths = {}
    for input_type, rows in views.items():
        output_path = args.output_dir / f"dev_2k_{input_type}.parquet"
        write_parquet(output_path, rows)
        output_paths[input_type] = {
            "path": str(output_path),
            "rows": len(rows),
            "sha256": file_sha256(output_path),
        }
    manifest = {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "task": "amazon_new_interest_choice_10way",
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "seed": args.seed,
        "sample_count": args.size,
        "input_types": list(INPUT_TYPES),
        "source": {
            "train": str(args.train),
            "train_sha256": file_sha256(args.train),
            "dev": str(args.dev),
            "dev_sha256": file_sha256(args.dev),
        },
        "sampling": {
            "method": "deterministic stratified minimum-hash sampling without replacement",
            "label_count": len(labels),
            "per_label_min": min(quotas.values()),
            "per_label_max": max(quotas.values()),
        },
        "candidates": {
            "count": 10,
            "gold": 1,
            "same_l2_negatives": "up to 3 deterministic unseen formal labels",
            "remaining_negatives": "deterministic unseen cross-L2 formal labels",
            "order": "deterministic per-source shuffle shared by all input views",
        },
        "validation": validation,
        "outputs": output_paths,
    }
    write_json(args.output_dir / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
