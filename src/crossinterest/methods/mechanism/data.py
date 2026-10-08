"""Frozen whole-category holdout views for novel-interest mechanism experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.prompts import render_history
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file


ROOT = HERE / "mechanism"
SEED = 20260825
TRAIN_SHA = "43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd"
DEV_SHA = "aebd86d67243022c2dd76a77dccf30146124d54b437baf72788ac1ca98192cb3"
MAPPING_SHA = "5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da"
SCHEMA = pa.schema([
    ("id", pa.string()), ("source_sample_ids", pa.list_(pa.string())),
    ("historical_titles", pa.list_(pa.string())),
    ("historical_interests", pa.list_(pa.string())),
    ("full_seen_interests", pa.list_(pa.string())),
    ("target_interest", pa.string()), ("positive_interests", pa.list_(pa.string())),
])


def history_signature(row: dict) -> str:
    text = unicodedata.normalize("NFKC", render_history(row, "behavior_interest"))
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def holdout_view(row: dict, categories: set[str], seed: int = SEED) -> dict | None:
    titles, interests = row["historical_titles"], row["historical_interests"]
    if len(titles) != len(interests) or not all(titles):
        raise ValueError("missing or unpaired history")
    counts = Counter(interests)
    choices = [category for category in counts if category in categories
               and len(interests) - counts[category] >= 5 and len(counts) >= 2]
    if not choices:
        return None
    signature = history_signature(row)
    hidden = min(choices, key=lambda category: hashlib.sha256(
        f"{seed}:{signature}:{category}".encode()).hexdigest())
    retained = [(title, category) for title, category in zip(titles, interests, strict=True)
                if category != hidden]
    return {
        "source_sample_ids": [row["id"]],
        "historical_titles": [title for title, _ in retained],
        "historical_interests": [category for _, category in retained],
        "full_seen_interests": sorted({category for _, category in retained}),
        "target_interest": hidden, "positive_interests": [hidden],
    }


def construct_views(rows: list[dict], categories: set[str], blocked: set[str],
                    seed: int = SEED) -> tuple[list[dict], dict]:
    merged: dict[str, dict] = {}
    counts: Counter = Counter()
    for row in rows:
        counts["source_rows"] += 1
        view = holdout_view(row, categories, seed)
        if view is None:
            counts["ineligible_rows"] += 1
            continue
        counts["eligible_rows"] += 1
        signature = history_signature(view)
        if signature in blocked:
            counts["isolated_rows"] += 1
            continue
        if signature in merged:
            counts["merged_rows"] += 1
            previous = merged[signature]
            previous["source_sample_ids"].extend(view["source_sample_ids"])
            previous["positive_interests"] = sorted(set(
                previous["positive_interests"] + view["positive_interests"]))
            previous["target_interest"] = previous["positive_interests"][0]
        else:
            view["id"] = f"reconstruct:{signature}"
            merged[signature] = view
    views = sorted(merged.values(), key=lambda row: hashlib.sha256(
        f"{seed}:order:{row['id']}".encode()).hexdigest())
    counts["output_rows"] = len(views)
    counts["multi_positive_rows"] = sum(len(row["positive_interests"]) > 1 for row in views)
    for view in views:
        if set(view["positive_interests"]) & set(view["historical_interests"]):
            raise ValueError("hidden category remains visible after merging")
        if history_signature(view) in blocked:
            raise ValueError("cross-task input overlap")
    return views, dict(counts)


def build(output: Path = ROOT / "data") -> dict:
    train_path = BASE / "train.parquet"
    dev_path = BASE / "eval/dev_2k_behavior_interest.parquet"
    mapping_path = BASE / "interest_id_semantic_mapping.json"
    old_path = HERE / "curriculum/seen500-novel112k/seen.parquet"
    for path, expected in ((train_path, TRAIN_SHA), (dev_path, DEV_SHA),
                           (mapping_path, MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    if (output / "MANIFEST.json").exists():
        manifest = json.loads((output / "MANIFEST.json").read_text())
        if (manifest["builder_sha256"] != sha256_file(Path(__file__))
                or manifest["aux_sha256"] != sha256_file(output / "reconstruction.parquet")):
            raise ValueError("existing mechanism freeze differs; refusing overwrite")
        return manifest
    mapping = load_label_mapping(mapping_path, expected_count=112)
    categories = set(mapping["category_to_id"])
    columns = ["id", "historical_titles", "historical_interests"]
    rows = pq.read_table(train_path, columns=columns).to_pylist()
    if len(rows) != 112000 or len({row["id"] for row in rows}) != 112000:
        raise ValueError("original N112K source identity mismatch")
    blocked = {history_signature(row) for row in rows}
    for path in (dev_path, old_path):
        for batch in pq.ParquetFile(path).iter_batches(columns=[
                "historical_titles", "historical_interests"]):
            blocked.update(history_signature(row) for row in batch.to_pylist())
    views, counts = construct_views(rows, categories, blocked)
    label_counts = Counter(category for row in views for category in row["positive_interests"])
    if set(label_counts) != categories:
        raise ValueError("auxiliary views do not cover all 112 labels")
    output.mkdir(parents=True, exist_ok=True)
    aux_path = output / "reconstruction.parquet"
    pq.write_table(pa.Table.from_pylist(views, schema=SCHEMA), aux_path)
    manifest = {
        "schema_version": 1, "protocol": "home112-category-holdout-relation-v1",
        "seed": SEED, "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA,
        "mapping_sha256": MAPPING_SHA, "mapping_fingerprint": mapping["mapping_fingerprint"],
        "old_block_sha256": sha256_file(old_path), "builder_sha256": sha256_file(Path(__file__)),
        "aux_sha256": sha256_file(aux_path), "counts": counts,
        "positive_class_counts": dict(sorted(label_counts.items())),
        "novel_rows": 112000, "novel_epochs": 1, "aux_epochs": 1,
        "effective_batch_size": 32, "novel_steps": 3500,
        "aux_steps": (len(views) + 31) // 32,
        "future_target_used_for_view_selection": False,
        "full_history_used": False, "cross_task_input_overlap": 0,
        "aux_multi_positive_policy": "merge_and_uniform_distribution_ce",
        "class_balancing": "none; one hash-selected eligible category per source history",
        "passed": True,
    }
    write_json_atomic(output / "MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()