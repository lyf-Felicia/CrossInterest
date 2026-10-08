"""New-interest training subsets selected by observed future recurrence."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.datasets.amazon.scripts.build import build_user_records, event_groups, iter_positive_users, load_products, user_split

from crossinterest.methods.alignment.data import BASE, HERE, sha256_file


DAY_MS = 86_400_000
SEED = 20260825
FROZEN_FILES = {
    "train.parquet": "43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd",
    "eval/dev_2k_behavior_interest.parquet": "aebd86d67243022c2dd76a77dccf30146124d54b437baf72788ac1ca98192cb3",
    "interest_id_semantic_mapping.json": "5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da",
}


def counted_user_records(
    user_id, interactions, products, split, seed, history_limit=50, min_history=5
):
    interactions = list(interactions)
    records = build_user_records(
        user_id, interactions, products, split, seed, history_limit, min_history
    )
    if not records:
        return []
    seen_products = set()
    category_events = defaultdict(list)
    accepted_timestamps = []
    for timestamp, group in event_groups(interactions):
        valid = [
            parent_asin for _, parent_asin, _ in group
            if parent_asin in products and parent_asin not in seen_products
        ]
        for parent_asin in valid:
            category_events[products[parent_asin][2]].append((timestamp, parent_asin))
            accepted_timestamps.append(timestamp)
        seen_products.update(valid)
    last_positive_timestamp = max(timestamp for timestamp, _, _ in interactions)
    for row in records:
        timestamp = row["timestamp"]
        events = category_events[row["target_interest"]]
        if any(event_timestamp < timestamp for event_timestamp, _ in events):
            raise ValueError("target is not new in the complete strict prefix")
        row.update({
            "future_target_count_including_first": len(events),
            "future_target_count_excluding_selected_event": len(events) - 1,
            "future_target_count_strictly_after_timestamp": sum(
                event_timestamp > timestamp for event_timestamp, _ in events
            ),
            "future_target_distinct_products": len({parent_asin for _, parent_asin in events}),
            "future_target_distinct_days": len({event_timestamp // DAY_MS for event_timestamp, _ in events}),
            "future_target_span_days": (events[-1][0] - timestamp) / DAY_MS,
            "future_observed_positive_events": sum(value >= timestamp for value in accepted_timestamps),
            "future_followup_days": (accepted_timestamps[-1] - timestamp) / DAY_MS,
            "future_last_positive_followup_days": (last_positive_timestamp - timestamp) / DAY_MS,
        })
    return records


def frequency_arm(count):
    return "once" if count == 1 else "at_least_five" if count >= 5 else None


class BalancedPool:
    def __init__(self, categories, original_ids, cap=1000):
        if not 1 <= cap <= 1000:
            raise ValueError("cap must be between 1 and 1000")
        self.categories = sorted(categories)
        self.original_ids = original_ids
        self.cap = cap
        self.heaps = {category: [] for category in self.categories}
        self.support = Counter({category: 0 for category in self.categories})
        self.original_support = Counter({category: 0 for category in self.categories})

    def add(self, row):
        category = row["target_interest"]
        if category not in self.heaps:
            return
        self.support[category] += 1
        original = row["id"] in self.original_ids
        self.original_support[category] += int(original)
        digest = hashlib.sha256(f"{SEED}:future-frequency:{row['id']}".encode()).digest()
        priority = int.from_bytes(digest, "big") + (0 if original else 1 << 256)
        item = (-priority, row["id"], row)
        heap = self.heaps[category]
        if len(heap) < self.cap:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    def rows(self):
        quota = min(self.cap, min(self.support.values()))
        selected = [
            item[2] for category in self.categories
            for item in sorted(self.heaps[category], key=lambda item: (-item[0], item[1]))[:quota]
        ]
        selected.sort(key=lambda row: hashlib.sha256(f"{SEED}:order:{row['id']}".encode()).digest())
        if len({row["id"] for row in selected}) != len(selected):
            raise ValueError("duplicate selected sample IDs")
        return selected, quota


def selected_diagnostics(rows):
    import numpy as np

    fields = [
        "full_history_length", "future_target_count_including_first",
        "future_target_count_strictly_after_timestamp", "future_target_distinct_days",
        "future_target_span_days", "future_observed_positive_events", "future_followup_days",
    ]
    return {
        "unique_users": len({row["user_id"] for row in rows}),
        "zero_followup_rows": sum(row["future_followup_days"] == 0 for row in rows),
        "single_day_target_rows": sum(row["future_target_distinct_days"] == 1 for row in rows),
        "quantiles_min_p25_p50_p75_max": {
            field: np.quantile([row[field] for row in rows], [0, .25, .5, .75, 1]).tolist()
            for field in fields
        } if rows else {},
    }


def build(output: Path, cap=1000):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "MANIFEST.json").exists():
        raise FileExistsError("a completed frequency build already exists")
    for relative, expected in FROZEN_FILES.items():
        if sha256_file(BASE / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    original = pq.read_table(BASE / "train.parquet", columns=["id"])
    original_ids = set(original["id"].to_pylist())
    mapping = json.loads((BASE / "interest_id_semantic_mapping.json").read_text())
    categories = sorted(mapping["category_to_id"])
    if len(categories) != 112:
        raise ValueError("expected the fixed 112-category mapping")
    prep = HERE.parent / "prep" / "artifacts" / "home_and_kitchen"
    prep_manifest = json.loads((prep / "MANIFEST.json").read_text())
    ratings = prep / "raw" / "Home_and_Kitchen.csv"
    metadata = prep / "metadata.sqlite"
    print("loading frozen product taxonomy", flush=True)
    products = load_products(metadata)
    pools = {arm: BalancedPool(categories, original_ids, cap) for arm in ("once", "at_least_five")}
    strict_support = {arm: Counter({category: 0 for category in categories}) for arm in pools}
    original_reconstructed = []
    stats = Counter()
    target_counts = Counter()
    print("scanning complete raw user timelines", flush=True)
    for user_number, (user_id, interactions) in enumerate(iter_positive_users(str(ratings), 4.0), 1):
        split = user_split(user_id, prep_manifest["seed"], .1)
        if split == "train":
            stats["train_users"] += 1
            stats["train_positive_interactions"] += len(interactions)
            if len(interactions) >= 6:
                rows = counted_user_records(user_id, interactions, products, split, prep_manifest["seed"])
                stats["reconstructed_train_rows"] += len(rows)
                for row in rows:
                    category = row["target_interest"]
                    target_counts[category] += 1
                    if category not in pools["once"].heaps:
                        continue
                    if row["id"] in original_ids:
                        original_reconstructed.append(row)
                    arm = frequency_arm(row["future_target_count_including_first"])
                    if arm:
                        pools[arm].add(row)
                    strict_arm = frequency_arm(row["future_target_count_strictly_after_timestamp"])
                    if strict_arm:
                        strict_support[strict_arm][category] += 1
        if user_number % 2_000_000 == 0:
            print(json.dumps({"users_scanned": user_number, **stats}), flush=True)
    expected = prep_manifest["unbalanced"]["train"]
    if target_counts != Counter(expected["target_counts"]) or stats["reconstructed_train_rows"] != expected["samples"]:
        raise ValueError("raw reconstruction does not match the frozen mother-pool counts")
    if {row["id"] for row in original_reconstructed} != original_ids or len(original_reconstructed) != len(original_ids):
        raise ValueError("original N112K rows not recovered exactly once")
    schema = pq.read_schema(BASE / "train.parquet")
    rebuilt = pa.Table.from_pylist(original_reconstructed, schema=schema).sort_by("id")
    original_full = pq.read_table(BASE / "train.parquet").sort_by("id")
    if not rebuilt.equals(original_full):
        raise ValueError("raw reconstruction differs from original N content")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "new-interest-future-frequency-inclusive-v1",
        "count_definition": "all observed accepted positive target-category events, including the first target timestamp group",
        "selection": "original N eligible rows first; supplement from all raw Train users; stable SHA order; no user cap or upsampling",
        "observation_window": "complete observed positive timeline; no minimum followup filter",
        "training": "Qwen3.5-9B N-only; method proposals not authorized",
        "frozen_inputs": FROZEN_FILES,
        "source": {"ratings": str(ratings), "ratings_sha256": sha256_file(ratings),
                   "metadata_sha256": sha256_file(metadata), "prep_manifest_sha256": sha256_file(prep / "MANIFEST.json")},
        "raw_reconstruction": {**stats, "mother_pool_counts_exact": True, "original_n112k_content_exact": True},
        "strictly_after_timestamp_alternative_support": strict_support,
        "arms": {},
    }
    count_fields = sorted(key for key in original_reconstructed[0] if key.startswith("future_"))
    original_stats = [{"id": row["id"], "target_interest": row["target_interest"],
                       **{key: row[key] for key in count_fields}} for row in original_reconstructed]
    pq.write_table(pa.Table.from_pylist(original_stats), output / "original_n_frequency.parquet", compression="zstd")
    for arm, pool in pools.items():
        rows, quota = pool.rows()
        retained = [item[2] for heap in pool.heaps.values() for item in heap]
        retained.sort(key=lambda row: row["id"])
        directory = output / arm
        directory.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(retained), directory / "retained_candidates.parquet", compression="zstd")
        arm_result = {
            "quota": quota, "rows": len(rows), "per_class_support": dict(pool.support),
            "original_n_eligible_per_class": dict(pool.original_support),
            "zero_support_categories": [category for category in categories if pool.support[category] == 0],
            "retained_candidates": len(retained),
            "discarded_for_strict_balance": len(retained) - len(rows),
            "selected_original_n": sum(row["id"] in original_ids for row in rows),
            "diagnostics": selected_diagnostics(rows),
        }
        if quota:
            if Counter(row["target_interest"] for row in rows) != Counter({category: quota for category in categories}):
                raise ValueError("selected data not strictly 112-class balanced")
            path = directory / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
            pq.write_table(pa.Table.from_pylist([
                {"id": row["id"], **{key: row[key] for key in count_fields}} for row in rows
            ]), directory / "selection_metadata.parquet", compression="zstd")
            arm_result.update({"train_path": str(path.resolve()), "train_sha256": sha256_file(path)})
        result["arms"][arm] = arm_result
    result["both_112_class_arms_feasible"] = all(arm["quota"] > 0 for arm in result["arms"].values())
    write_json_atomic(output / "MANIFEST.json", result)
    print(json.dumps({"both_feasible": result["both_112_class_arms_feasible"], "arms": {
        arm: {key: value[key] for key in ("quota", "rows", "zero_support_categories")}
        for arm, value in result["arms"].items()
    }}, indent=2), flush=True)
    return result


def paired_rows(original, once, five, categories, cap=1000):
    if not 1 <= cap <= 1000:
        raise ValueError("cap must be between 1 and 1000")
    grouped = []
    for source in (original, once, five):
        if len({row["id"] for row in source}) != len(source):
            raise ValueError("duplicate source IDs")
        by_category = defaultdict(list)
        for row in source:
            by_category[row["target_interest"]].append(row)
        grouped.append(by_category)
    original_by, once_by, five_by = grouped
    def order(row):
        return hashlib.sha256(f"{SEED}:paired-selection:{row['id']}".encode()).digest()
    slots = []
    for category in sorted(categories):
        base = sorted(original_by[category], key=order)
        if len(base) != cap:
            raise ValueError("original data must have the exact per-class quota")
        base_ids = {row["id"] for row in base}
        repeated = sorted(five_by[category], key=lambda row: (row["id"] not in base_ids, order(row)))[:cap]
        count = len(repeated)
        repeated_ids = {row["id"] for row in repeated}
        if any(row["future_target_count_including_first"] < 5 for row in repeated):
            raise ValueError("repeated replacement does not satisfy count >=5")
        candidates = sorted(once_by[category], key=order)
        if any(row["future_target_count_including_first"] != 1 for row in candidates):
            raise ValueError("once replacement does not satisfy count =1")
        original_budget = count - len(repeated_ids & base_ids)
        single = [row for row in candidates if row["id"] in base_ids][:original_budget]
        single += [row for row in candidates if row["id"] not in base_ids][:count - len(single)]
        if len(single) != count:
            raise ValueError("insufficient once candidates after reserving shared original background")
        single_ids = {row["id"] for row in single}
        if single_ids & repeated_ids:
            raise ValueError("replacement arms overlap")
        background = [row for row in base if row["id"] not in single_ids | repeated_ids][:cap-count]
        if len(background) != cap-count:
            raise ValueError("insufficient disjoint original background")
        for index, row in enumerate(background):
            slots.append((f"{category}:background:{index}", row, row, "background"))
        for index, (single_row, repeated_row) in enumerate(zip(single, repeated)):
            slots.append((f"{category}:replacement:{index}", single_row, repeated_row, "replacement"))
    slots.sort(key=lambda slot: hashlib.sha256(f"{SEED}:paired-order:{slot[0]}".encode()).digest())
    arms = {"once": [slot[1] for slot in slots], "at_least_five": [slot[2] for slot in slots]}
    metadata = [{"slot": slot[0], "role": slot[3], "target_interest": slot[1]["target_interest"],
                 "once_id": slot[1]["id"], "at_least_five_id": slot[2]["id"]} for slot in slots]
    for rows in arms.values():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate IDs within a paired arm")
    common = {row["once_id"] for row in metadata if row["role"] == "background"}
    if {row["id"] for row in arms["once"]} & {row["id"] for row in arms["at_least_five"]} != common:
        raise ValueError("unexpected overlap outside shared background")
    return arms, metadata


def build_paired(source: Path, output: Path):
    if output.exists():
        raise FileExistsError("paired output must be a new directory")
    for relative, expected in FROZEN_FILES.items():
        if sha256_file(BASE / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    manifest = json.loads((source / "MANIFEST.json").read_text())
    audit = json.loads((source / "INDEPENDENT_AUDIT.json").read_text())
    if not audit["passed"] or audit["manifest_sha256"] != sha256_file(source / "MANIFEST.json"):
        raise ValueError("frequency source audit mismatch")
    table = pq.read_table(BASE / "train.parquet")
    original = table.to_pylist()
    categories = set(manifest["arms"]["once"]["per_class_support"])
    candidates = {arm: pq.read_table(source / arm / "retained_candidates.parquet").to_pylist()
                  for arm in ("once", "at_least_five")}
    for arm, rows in candidates.items():
        expected = {category: min(1000, count) for category, count in manifest["arms"][arm]["per_class_support"].items()}
        if Counter(row["target_interest"] for row in rows) != Counter(expected):
            raise ValueError("retained candidate support mismatch")
    arms, slots = paired_rows(original, candidates["once"], candidates["at_least_five"], categories)
    output.mkdir(parents=True)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": "home112-paired-frequency-N-only-v1",
              "approved": True, "count_includes_first_target": True, "seed": SEED,
              "source_manifest_sha256": sha256_file(source / "MANIFEST.json"),
              "source_audit_sha256": sha256_file(source / "INDEPENDENT_AUDIT.json"),
              "source_candidate_sha256": {arm: sha256_file(source / arm / "retained_candidates.parquet") for arm in arms},
              "frozen_inputs": FROZEN_FILES, "common_rows": sum(slot["role"] == "background" for slot in slots),
              "replacement_rows": sum(slot["role"] == "replacement" for slot in slots),
              "per_class_replacements": dict(Counter(slot["target_interest"] for slot in slots if slot["role"] == "replacement")),
              "row_order": "same class and role per slot; same background row at same position; training shuffle same seed",
              "arms": {}}
    original_ids = {row["id"] for row in original}
    for arm, rows in arms.items():
        if Counter(row["target_interest"] for row in rows) != Counter({category: 1000 for category in categories}):
            raise ValueError("paired data not strictly balanced")
        directory = output / arm
        directory.mkdir()
        path = directory / "train.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path, compression="zstd")
        replacements = [row for row, slot in zip(rows, slots) if slot["role"] == "replacement"]
        result["arms"][arm] = {"rows": len(rows), "quota": 1000, "sha256": sha256_file(path),
                              "path": str(path.resolve()), "original_n_rows": sum(row["id"] in original_ids for row in rows),
                              "replacement_diagnostics": selected_diagnostics(replacements)}
    pq.write_table(pa.Table.from_pylist(slots), output / "PAIRED_SLOTS.parquet", compression="zstd")
    result["paired_slots_sha256"] = sha256_file(output / "PAIRED_SLOTS.parquet")
    write_json_atomic(output / "MANIFEST.json", result)
    print(json.dumps({"common": result["common_rows"], "replacement": result["replacement_rows"],
                      "arms": result["arms"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "frequency" / "data")
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--paired-source", type=Path)
    arguments = parser.parse_args()
    if arguments.paired_source:
        build_paired(arguments.paired_source, arguments.output)
    else:
        build(arguments.output, arguments.cap)