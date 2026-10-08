"""Audit balanced regular/seen-interest pools without changing frozen novel data."""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.build_views import HISTORY_INTRODUCTIONS, render_history
from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training.label_space import load_label_mapping, output_label
from crossinterest.training.prompts import INTEREST_ID_OUTPUT_INSTRUCTION, render_training_messages
from crossinterest.methods.backbones.runner import TRAIN, TRAIN_SHA256, DEV, MAPPING
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.crossfit.train import input_signature


ROOT = HERE / "curriculum"
PREP = HERE.parent / "prep/artifacts/home_and_kitchen"
SEED = 20260825


def render_messages(row: dict, mapping: dict) -> list[dict]:
    task = row["task_type"]
    if task == "novel":
        return render_training_messages(row, "behavior_interest", "interest_id", mapping)
    if task not in ("regular", "seen"):
        raise ValueError("unknown curriculum task")
    was_seen = row["target_interest"] in row["full_seen_interests"]
    if task == "seen" and not was_seen:
        raise ValueError("seen-interest target is absent from complete history")
    instruction = (
        "Predict the Interest ID for the one topic category that I am most likely "
        "to be interested in next. The category may be previously observed or new."
        if task == "regular" else
        "Predict the Interest ID for the one previously observed topic category "
        "that I am most likely to be interested in next."
    )
    return [{"role": "user", "content": (
        f"{HISTORY_INTRODUCTIONS['behavior_interest']}\n\n"
        f"{render_history(row, 'behavior_interest')}\n\n"
        f"{instruction}\n\n{INTEREST_ID_OUTPUT_INSTRUCTION}"
    )}, {"role": "assistant", "content": output_label(row["target_interest"], "interest_id", mapping)}]


def user_records(user_id: str, interactions, products: dict, labels: set, stats: Counter):
    from crossinterest.datasets.amazon.scripts.build import choose_target, event_groups, DEFAULT_SEED

    seen_labels = set()
    seen_products = set()
    history = []
    for timestamp, group in event_groups(interactions):
        unique = {}
        for _, asin, rating in group:
            if asin in unique:
                stats["duplicate_source_events"] += 1
            unique[asin] = max(rating, unique.get(asin, rating))
        valid = [(asin, *products[asin], rating) for asin, rating in sorted(unique.items())
                 if asin in products and asin not in seen_products]
        if len(history) >= 5 and valid:
            visible = history[-50:]
            common = {
                "schema_version": 1, "id": f"train:{user_id}:{timestamp}", "split": "train",
                "domain": "Home_and_Kitchen", "user_id": user_id, "timestamp": timestamp,
                "timestamp_iso": datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat(),
                "history_limit": 50, "full_history_length": len(history),
                "full_seen_interests": sorted(seen_labels),
                "history_parent_asins": [item[1] for item in visible],
                "historical_titles": [item[2] for item in visible],
                "historical_level_2": [item[3] for item in visible],
                "historical_interests": [item[4] for item in visible],
                "history_timestamps": [item[0] for item in visible],
                "same_history_new_interest_count": len({item[3] for item in valid} - seen_labels),
                "target_selection": "seeded_uniform_label_then_product",
            }
            signature = input_signature(common)
            for task in ("regular", "seen"):
                candidates = {}
                for asin, title, level_2, label, rating in valid:
                    if label in labels and (task == "regular" or label in seen_labels):
                        candidates.setdefault(label, []).append((asin, title, level_2, rating))
                if candidates:
                    asin, title, level_2, label, rating = choose_target(
                        user_id, timestamp, candidates, DEFAULT_SEED)
                    yield {**common, "target_parent_asin": asin, "target_title": title,
                           "target_level_2": level_2, "target_interest": label, "target_rating": rating,
                           "task_type": task, "target_was_seen": label in seen_labels,
                           "input_sha256": signature}
        for asin, title, level_2, label, rating in valid:
            history.append((timestamp, asin, title, level_2, label, rating))
            seen_products.add(asin)
            seen_labels.add(label)


def audit_pools() -> dict:
    from crossinterest.datasets.amazon.scripts.build import load_products, iter_positive_users, user_split, stable_digest, DEFAULT_SEED

    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise ValueError("frozen novel source changed")
    mapping = load_label_mapping(MAPPING, expected_count=112)
    labels = set(mapping["category_to_id"])
    novel = pq.read_table(TRAIN, columns=["id", "historical_titles", "historical_interests"]).to_pylist()
    blocked_ids = {row["id"] for row in novel}
    novel_inputs = {input_signature(row) for row in novel}
    dev_inputs = {input_signature(row) for row in pq.read_table(DEV).to_pylist()}
    del novel
    quota = 1000
    stats = Counter()
    counts = {task: {label: Counter() for label in labels} for task in ("regular", "seen")}
    heaps = {task: {label: [] for label in labels} for task in counts}
    seen_inputs = {task: set() for task in counts}
    print("Loading existing product metadata", flush=True)
    products = load_products(PREP / "metadata.sqlite")
    print(f"Loaded {len(products)} products; auditing all training users", flush=True)
    for user_number, (user_id, interactions) in enumerate(
        iter_positive_users(str(PREP / "raw/Home_and_Kitchen.csv"), 4.0), 1
    ):
        if user_split(user_id, DEFAULT_SEED, 0.10) == "train":
            stats["train_users"] += 1
            for row in user_records(user_id, interactions, products, labels, stats):
                task, label, signature = row["task_type"], row["target_interest"], row["input_sha256"]
                class_stats = counts[task][label]
                class_stats["eligible_before_exclusions"] += 1
                if row["id"] in blocked_ids:
                    class_stats["excluded_novel_event_group"] += 1
                    continue
                if signature in novel_inputs:
                    class_stats["excluded_novel_input"] += 1
                    continue
                if signature in dev_inputs:
                    class_stats["excluded_dev_input"] += 1
                    continue
                if signature in seen_inputs[task]:
                    class_stats["excluded_duplicate_input"] += 1
                    continue
                seen_inputs[task].add(signature)
                class_stats["available"] += 1
                class_stats["available_seen"] += int(row["target_was_seen"])
                key = int.from_bytes(stable_digest(SEED, "curriculum-balanced", row["id"]), "big")
                heap = heaps[task][label]
                entry = (-key, row["id"], json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                if len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
        if user_number % 250000 == 0:
            print(json.dumps({"users_scanned": user_number, "available": {
                task: sum(count["available"] for count in counts[task].values()) for task in counts
            }}, sort_keys=True), flush=True)
    del products, seen_inputs
    ROOT.mkdir(exist_ok=True)
    schema = pq.read_schema(TRAIN).append(pa.field("task_type", pa.string(), nullable=False))
    schema = schema.append(pa.field("target_was_seen", pa.bool_(), nullable=False))
    schema = schema.append(pa.field("input_sha256", pa.string(), nullable=False))
    pools = {}
    for task in counts:
        rows = [json.loads(entry[2]) for label in sorted(labels) for entry in sorted(heaps[task][label], reverse=True)]
        if len({row["id"] for row in rows}) != len(rows) or len({row["input_sha256"] for row in rows}) != len(rows):
            raise ValueError("candidate pool has duplicate source or content")
        if {row["id"] for row in rows} & blocked_ids:
            raise ValueError("candidate pool overlaps frozen novel groups")
        if {row["input_sha256"] for row in rows} & (novel_inputs | dev_inputs):
            raise ValueError("candidate pool overlaps novel or Dev inputs")
        if any(any(timestamp >= row["timestamp"] for timestamp in row["history_timestamps"]) for row in rows):
            raise ValueError("future events leaked into history")
        for row in rows:
            render_messages(row, mapping)
        selected_counts = Counter(row["target_interest"] for row in rows)
        deficit = {label: counts[task][label]["available"] for label in sorted(labels)
                   if counts[task][label]["available"] < quota}
        selected_seen = Counter(row["target_interest"] for row in rows if row["target_was_seen"])
        path = ROOT / f"{task}_candidates.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
        pools[task] = {"candidate_rows": len(rows), "sha256": sha256_file(path),
                       "supports_full_quota": not deficit, "insufficient_classes": deficit,
                       "selected_seen_rows": sum(selected_seen.values()),
                       "selected_users": len({row["user_id"] for row in rows}),
                       "class_counts": {label: {**dict(counts[task][label]),
                                                "selected": selected_counts[label],
                                                "selected_seen": selected_seen[label]} for label in sorted(labels)}}
    manifest = {"schema_version": 1, "seed": SEED, "source_seed": DEFAULT_SEED,
                "frozen_novel_sha256": TRAIN_SHA256, "source_manifest_sha256": sha256_file(PREP / "MANIFEST.json"),
                "mapping_sha256": sha256_file(MAPPING), "dev_sha256": sha256_file(DEV),
                "requested_quota": quota, "label_count": 112, "frozen_novel_dev_shared_inputs": len(novel_inputs & dev_inputs),
                "stats": dict(stats), "pools": pools, "selected_mode": None,
                "formal_pool_frozen": False, "source_and_input_overlap": 0}
    write_json_atomic(ROOT / "SUPPORT_AUDIT.json", manifest)
    print(json.dumps({task: {key: value for key, value in pool.items() if key != "class_counts"}
                      for task, pool in pools.items()}, ensure_ascii=False, indent=2), flush=True)
    return manifest


def balanced_pools(seen_rows: list[dict], novel_rows: list[dict], quota: int,
                   full_novel: bool = False) -> dict:
    from crossinterest.datasets.amazon.scripts.build import stable_digest

    if quota <= 0:
        raise ValueError("quota must be positive")
    isolation = Counter()
    if full_novel:
        novel_inputs = {}
        for row in novel_rows:
            signature = input_signature(row)
            if signature in novel_inputs:
                raise ValueError("frozen novel input is duplicated; full novel data cannot be changed")
            novel_inputs[signature] = row["target_interest"]
        novel_ids = {row["id"] for row in novel_rows}
        retained = []
        seen_inputs = set()
        for row in seen_rows:
            signature = input_signature(row)
            if signature in novel_inputs:
                isolation["excluded_shared_history"] += 1
                isolation["excluded_conflicting_label"] += int(
                    row["target_interest"] != novel_inputs[signature])
                continue
            if row["id"] in novel_ids:
                isolation["excluded_shared_source"] += 1
                continue
            if signature in seen_inputs:
                raise ValueError("duplicate input within seen candidates")
            seen_inputs.add(signature)
            retained.append(row)
        seen_rows = retained
    sources = {"seen": seen_rows, "novel": novel_rows}
    counts = {task: Counter(row["target_interest"] for row in rows) for task, rows in sources.items()}
    labels = sorted(label for label in counts["novel"]
                    if counts["novel"][label] >= quota and counts["seen"][label] >= quota)
    if not labels:
        raise ValueError("no common classes meet quota")
    selected = {}
    for task, rows in sources.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate source IDs")
        task_labels = sorted(counts[task]) if full_novel and task == "novel" else labels
        grouped = {label: [] for label in task_labels}
        for row in rows:
            if row["target_interest"] in grouped:
                grouped[row["target_interest"]].append(row)
        subset = []
        for label in task_labels:
            ordered = sorted(grouped[label], key=lambda row: (
                stable_digest(SEED, "curriculum-balanced", row["id"]), row["id"]))
            selected_rows = ordered if full_novel and task == "novel" else ordered[:quota]
            for row in selected_rows:
                was_seen = row["target_interest"] in row["full_seen_interests"]
                if was_seen != (task == "seen"):
                    raise ValueError("target violates task history condition")
                subset.append({**row, "task_type": task, "target_was_seen": was_seen,
                               "input_sha256": input_signature(row)})
        selected[task] = sorted(subset, key=lambda row: (
            stable_digest(SEED, "curriculum-within-pool", row["id"]), row["id"]))
    union = selected["seen"] + selected["novel"]
    if len({row["id"] for row in union}) != len(union):
        raise ValueError("source overlap between tasks")
    if len({row["input_sha256"] for row in union}) != len(union):
        raise ValueError("duplicate input within or between tasks")
    result = {"labels": sorted(counts["novel"]) if full_novel else labels,
              "pools": selected, "orders": {
        "seen_then_novel": union,
        "shuffle": sorted(union, key=lambda row: (
            stable_digest(SEED, "curriculum-shuffle", row["id"]), row["id"])),
    }}
    if full_novel:
        if {row["id"] for row in selected["novel"]} != {row["id"] for row in novel_rows}:
            raise ValueError("full novel source was reduced")
        result["seen_labels"] = labels
        result["isolation"] = {key: isolation[key] for key in (
            "excluded_shared_history", "excluded_conflicting_label", "excluded_shared_source")}
    return result


def prepare(quota: int = 500, full_novel: bool = False) -> dict:
    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise ValueError("frozen novel source changed")
    support_path = ROOT / "SUPPORT_AUDIT.json"
    support = json.loads(support_path.read_text())
    candidate_path = ROOT / "seen_candidates.parquet"
    if sha256_file(candidate_path) != support["pools"]["seen"]["sha256"]:
        raise ValueError("audited seen candidates changed")
    if quota > support["requested_quota"]:
        raise ValueError("requested quota exceeds audited candidate sample")
    mapping = load_label_mapping(MAPPING, expected_count=112)
    require = {"frozen_novel_sha256": TRAIN_SHA256, "mapping_sha256": sha256_file(MAPPING),
               "dev_sha256": sha256_file(DEV)}
    if any(support[key] != value for key, value in require.items()):
        raise ValueError("source, mapping or Dev differs from support audit")
    novel_rows = pq.read_table(TRAIN).to_pylist()
    seen_rows = pq.read_table(candidate_path).to_pylist()
    built = balanced_pools(seen_rows, novel_rows, quota, full_novel=full_novel)
    if full_novel:
        novel_counts = Counter(row["target_interest"] for row in novel_rows)
        if len(novel_rows) != 112000 or dict(novel_counts) != {
            label: 1000 for label in mapping["category_to_id"]
        }:
            raise ValueError("full novel must retain112000 rows and112 classes of1000")
        original = {row["id"]: row for row in novel_rows}
        if any({key: row[key] for key in original[row["id"]]} != original[row["id"]]
               for row in built["pools"]["novel"]):
            raise ValueError("full novel content changed")
    novel_ids = {row["id"] for row in novel_rows}
    blocked_inputs = {input_signature(row) for row in novel_rows}
    dev_inputs = {input_signature(row) for row in pq.read_table(DEV).to_pylist()}
    if any(row["id"] in novel_ids or row["input_sha256"] in blocked_inputs | dev_inputs
           for row in built["pools"]["seen"]):
        raise ValueError("seen pool overlaps frozen novel or Dev")
    for rows in built["pools"].values():
        for row in rows:
            if row["input_sha256"] in dev_inputs:
                raise ValueError("course input overlaps Dev")
            if any(timestamp >= row["timestamp"] for timestamp in row["history_timestamps"]):
                raise ValueError("future event in course history")
            render_messages(row, mapping)
    output = ROOT / (f"seen{quota}-novel112k" if full_novel else f"seen{quota}-novel{quota}")
    output.mkdir(exist_ok=True)
    schema = pq.read_schema(candidate_path)
    files = {}
    for name, rows in (built["pools"] | built["orders"]).items():
        path = output / f"{name}.parquet"
        table = pa.Table.from_pylist(rows, schema=schema)
        if path.exists():
            if not pq.read_table(path).equals(table):
                raise ValueError(f"frozen curriculum changed: {path}")
        else:
            pq.write_table(table, path, compression="zstd")
        files[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    manifest = {"schema_version": 1, "seed": SEED, "quota_per_class_per_task": quota,
                "labels": built["labels"], "label_count": len(built["labels"]),
                "source": require, "support_audit_sha256": sha256_file(support_path),
                "seen_candidates_sha256": sha256_file(candidate_path), "files": files,
                "arms": list(built["orders"]), "input_type": "behavior_interest",
                "seen_definition": "complete_positive_history", "visible_history_limit": 50,
                "seen_target_absent_from_visible_history": sum(
                    row["target_interest"] not in row["historical_interests"]
                    for row in built["pools"]["seen"]),
                "evaluation": {"path": str(DEV), "sha256": sha256_file(DEV),
                               "task": "novel", "rows": 2000, "label_count": 112},
                "formal_pool_frozen": True, "training_submitted": False,
                "source_and_input_overlap": 0}
    if full_novel:
        manifest.pop("quota_per_class_per_task")
        manifest.update({"schema_version": 2, "full_novel": True,
                         "seen_quota_per_class": quota, "novel_quota_per_class": 1000,
                         "seen_labels": built["seen_labels"],
                         "task_class_counts": {task: dict(sorted(Counter(
                             row["target_interest"] for row in rows).items()))
                             for task, rows in built["pools"].items()},
                         "isolation": {"policy": "exclude_all_shared_history_from_seen",
                                       "signature": "normalized_ordered_behavior_interest",
                                       "novel_source_rows_unchanged": True,
                                       "cross_task_shared_histories": 0,
                                       "cross_task_conflicting_labels": 0,
                                       "excluded_from_audited_candidates": built["isolation"]},
                         "formal_global_step": (files["seen_then_novel"]["rows"] + 31) // 32})
    path = output / "MANIFEST.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("frozen course manifest changed")
    write_json_atomic(path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit-pools", "prepare"))
    parser.add_argument("--quota", type=int, default=500)
    parser.add_argument("--full-novel", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare(arguments.quota, full_novel=arguments.full_novel)
    else:
        audit_pools()


if __name__ == "__main__":
    main()