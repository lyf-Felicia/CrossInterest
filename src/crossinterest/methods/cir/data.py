"""Build strictly balanced external-user CIR without changing the original data."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.datasets.amazon.scripts.build import user_split
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.mechanism.data import SEED, history_signature


def stable_key(*parts: str) -> str:
    return hashlib.sha256(":".join((str(SEED), *parts)).encode()).hexdigest()


def eligible_categories(row: dict, categories: set[str]) -> list[str]:
    titles, interests = row["historical_titles"], row["historical_interests"]
    if len(titles) != len(interests) or not all(titles):
        raise ValueError("missing or unpaired history")
    counts = Counter(interests)
    return sorted(category for category in counts if category in categories
                  and len(counts) >= 2 and len(interests) - counts[category] >= 5)


def category_view(row: dict, category: str, categories: set[str]) -> dict:
    if category not in eligible_categories(row, categories):
        raise ValueError("ineligible masked category")
    retained = [index for index, value in enumerate(row["historical_interests"]) if value != category]
    hidden = [index for index, value in enumerate(row["historical_interests"]) if value == category]
    timestamps = row["history_timestamps"]
    if len(timestamps) != len(row["historical_titles"]) or any(value >= row["timestamp"] for value in timestamps):
        raise ValueError("history must strictly precede the source event")
    view = {
        "source_sample_ids": [row["id"]], "user_id": row["user_id"],
        "source_timestamp": row["timestamp"],
        "historical_titles": [row["historical_titles"][index] for index in retained],
        "historical_interests": [row["historical_interests"][index] for index in retained],
        "history_timestamps": [timestamps[index] for index in retained],
        "hidden_titles": [row["historical_titles"][index] for index in hidden],
        "hidden_timestamps": [timestamps[index] for index in hidden],
        "full_seen_interests": sorted({row["historical_interests"][index] for index in retained}),
        "target_interest": category, "positive_interests": [category],
    }
    view["id"] = f"cir:{history_signature(view)}"
    return view


def balanced_assignment(candidates: list[dict], categories: set[str], max_per_class: int = 1000) -> tuple[list[dict], dict]:
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow

    if not categories or not 1 <= max_per_class <= 1000:
        raise ValueError("require output categories and a quota in [1, 1000]")
    by_signature = defaultdict(set)
    for row in candidates:
        if row["target_interest"] not in categories:
            raise ValueError("candidate target outside frozen label space")
        by_signature[row["id"]].add(row["target_interest"])
    conflicts = {signature for signature, labels in by_signature.items() if len(labels) != 1}
    edges = {}
    for row in candidates:
        if row["id"] in conflicts:
            continue
        pair = (row["user_id"], row["id"])
        previous = edges.get(pair)
        if previous is None or stable_key(row["source_sample_ids"][0]) < stable_key(previous["source_sample_ids"][0]):
            edges[pair] = row
    users = sorted({user for user, _ in edges})
    signatures = sorted({signature for _, signature in edges})
    labels = sorted(categories)
    user_nodes = {user: index + 1 for index, user in enumerate(users)}
    signature_nodes = {signature: len(users) + index + 1 for index, signature in enumerate(signatures)}
    label_nodes = {label: len(users) + len(signatures) + index + 1 for index, label in enumerate(labels)}
    sink = len(users) + len(signatures) + len(labels) + 1
    sources, destinations, capacities = [], [], []
    for node in user_nodes.values():
        sources.append(0)
        destinations.append(node)
        capacities.append(1)
    for user, signature in sorted(edges):
        sources.append(user_nodes[user])
        destinations.append(signature_nodes[signature])
        capacities.append(1)
    for signature in signatures:
        sources.append(signature_nodes[signature])
        destinations.append(label_nodes[next(iter(by_signature[signature]))])
        capacities.append(1)
    supports = Counter(next(iter(by_signature[signature])) for signature in signatures)
    upper = min(max_per_class, *(supports[label] for label in labels))
    if upper == 0:
        raise ValueError("at least one frozen category has no valid candidate")
    base_count = len(capacities)
    for node in label_nodes.values():
        sources.append(node)
        destinations.append(sink)
        capacities.append(upper)

    def solve(quota):
        values = np.asarray(capacities, dtype=np.int32)
        values[base_count:] = quota
        graph = coo_matrix((values, (sources, destinations)), shape=(sink + 1, sink + 1)).tocsr()
        return maximum_flow(graph, 0, sink, method="dinic")

    result = solve(upper)
    if result.flow_value != upper * len(labels):
        lower, higher = 0, upper - 1
        while lower < higher:
            middle = (lower + higher + 1) // 2
            trial = solve(middle)
            if trial.flow_value == middle * len(labels):
                lower = middle
            else:
                higher = middle - 1
        upper = lower
        if not upper:
            raise ValueError("no jointly feasible positive quota")
        result = solve(upper)
    selected = [row for (user, signature), row in edges.items()
                if result.flow[user_nodes[user], signature_nodes[signature]] == 1]
    selected.sort(key=lambda row: stable_key("order", row["id"]))
    counts = Counter(row["target_interest"] for row in selected)
    if set(counts) != categories or set(counts.values()) != {upper}:
        raise ValueError("flow failed strict class balance")
    if len({row["id"] for row in selected}) != len(selected) or len({row["user_id"] for row in selected}) != len(selected):
        raise ValueError("flow repeated an input or user")
    return selected, {"quota": upper, "rows": len(selected), "classes": len(labels),
                      "class_counts": dict(sorted(counts.items())), "conflicting_inputs_excluded": len(conflicts),
                      "candidate_edges": len(edges), "unique_users": len(users),
                      "quota_scope": "maximum_feasible_within_retained_candidate_pool"}


def build(output: Path, max_per_class: int = 1000, candidate_users_per_class: int = 4000) -> dict:
    from crossinterest.methods.mechanism.data import TRAIN_SHA, DEV_SHA, MAPPING_SHA

    if candidate_users_per_class < max_per_class:
        raise ValueError("candidate capacity must cover requested quota")
    train_path = BASE / "train.parquet"
    dev_path = BASE / "eval/dev_2k_behavior_interest.parquet"
    mapping_path = BASE / "interest_id_semantic_mapping.json"
    for path, expected in ((train_path, TRAIN_SHA), (dev_path, DEV_SHA), (mapping_path, MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    source_manifest_path = BASE / "DATASET_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source = Path(source_manifest["source"]["mother_pool"]) / "train.jsonl"
    prep = json.loads(Path(source_manifest["source"]["prep_manifest"]).read_text())
    if source.stat().st_size != source_manifest["source"]["train_bytes"]:
        raise ValueError("mother pool size changed")
    labels = set(source_manifest["balance"]["support"])
    if len(labels) != 112:
        raise ValueError("expected all 112 classes")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text())
        if result["builder_sha256"] != sha256_file(Path(__file__)) or result["train_sha256"] != sha256_file(output / "train.parquet"):
            raise ValueError("existing balanced freeze differs")
        return result
    forbidden_users = set(pq.read_table(train_path, columns=["user_id"])["user_id"].to_pylist())
    blocked = set()
    old_path = HERE / "curriculum/seen500-novel112k/seen.parquet"
    for path in (train_path, old_path, dev_path):
        for batch in pq.ParquetFile(path).iter_batches(columns=["historical_titles", "historical_interests"]):
            blocked.update(history_signature(row) for row in batch.to_pylist())
    pools = {category: [] for category in labels}
    counters = Counter()
    per_user = {}
    visited_users = set()
    current_user = None
    current_allowed = False
    raw_digest = hashlib.sha256()

    def finish_user():
        for category, (_, row) in per_user.items():
            priority = int(stable_key("candidate-user", current_user, category), 16)
            pool = pools[category]
            if len(pool) >= candidate_users_per_class and priority >= -pool[0][0]:
                continue
            view = category_view(row, category, labels)
            if history_signature(view) in blocked:
                counters["blocked_candidate_views"] += 1
                continue
            entry = (-priority, current_user, view)
            if len(pool) < candidate_users_per_class:
                heapq.heappush(pool, entry)
            else:
                heapq.heapreplace(pool, entry)

    with source.open("rb") as handle:
        for line in handle:
            raw_digest.update(line)
            row = json.loads(line)
            counters["mother_rows"] += 1
            if row["user_id"] != current_user:
                if current_user is not None:
                    finish_user()
                current_user = row["user_id"]
                if current_user in visited_users:
                    raise ValueError("mother pool user groups are not contiguous")
                visited_users.add(current_user)
                per_user = {}
                current_allowed = current_user not in forbidden_users
                if user_split(current_user, prep["seed"], 0.1) != "train":
                    raise ValueError("Dev user in auxiliary source")
            if row["split"] != "train":
                raise ValueError("non-Train source")
            if not current_allowed:
                counters["excluded_n_user_rows"] += 1
                continue
            counters["external_rows"] += 1
            eligible = eligible_categories(row, labels)
            if eligible:
                counters["eligible_external_rows"] += 1
            for category in eligible:
                priority = stable_key("candidate-prefix", row["id"], category)
                if category not in per_user or priority < per_user[category][0]:
                    per_user[category] = (priority, row)
            if counters["mother_rows"] % 500000 == 0:
                print(json.dumps({"event": "external_scan", **counters}), flush=True)
    finish_user()
    if counters["mother_rows"] != prep["unbalanced"]["train"]["samples"]:
        raise ValueError("mother pool row count differs")
    candidates = [entry[2] for category in sorted(labels) for entry in pools[category]]
    print(json.dumps({"event": "matching", "candidate_rows": len(candidates)}), flush=True)
    selected, assignment = balanced_assignment(candidates, labels, max_per_class)
    if len(selected) % 4 or assignment["quota"] < 2:
        raise ValueError("balanced data cannot support four ranks and complete smoke")
    if any(row["user_id"] in forbidden_users or history_signature(row) in blocked for row in selected):
        raise ValueError("final auxiliary isolation failed")
    table = pa.Table.from_pylist(selected)
    train_output = output / "train.parquet"
    pq.write_table(table, train_output, compression="zstd")
    result = {"passed": True, "schema_version": 1, "method": "external_train_balanced_cir",
              "seed": SEED, "max_per_class": max_per_class, "candidate_users_per_class": candidate_users_per_class,
              "builder_sha256": sha256_file(Path(__file__)), "train_sha256": sha256_file(train_output),
              "original_n_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
              "mother_sha256": raw_digest.hexdigest(), "source_manifest_sha256": sha256_file(source_manifest_path),
              "old_isolation_sha256": sha256_file(old_path), "counts": dict(counters), "assignment": assignment,
              "rows": len(selected), "steps": (len(selected) + 31) // 32, "epochs": 1,
              "user_overlap_with_n": 0, "user_overlap_with_dev": 0, "cross_task_input_overlap": 0,
              "max_rows_per_user": 1, "duplicate_inputs": 0, "distributed_padding_duplicates": 0,
              "future_targets_used": False, "natural_distribution_arm": False,
              "hidden_fields_are_supervision_only": True}
    write_json_atomic(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "cir/data/external_balanced")
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--candidate-users-per-class", type=int, default=4000)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.max_per_class, args.candidate_users_per_class), indent=2), flush=True)


if __name__ == "__main__":
    main()