"""Freeze balanced, source-paired forward and inverse interest controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, maximum_bipartite_matching

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.cir.data import eligible_categories, stable_key
from crossinterest.methods.mechanism.data import TRAIN_SHA, DEV_SHA, MAPPING_SHA, history_signature


ARMS = ("forward", "ordinary", "paired", "shuffled")
FIELDS = ("id", "user_id", "timestamp", "historical_titles", "historical_interests",
          "history_timestamps", "full_seen_interests", "target_interest", "target_title")


def assign_pairs(rows: list[dict], categories: set[str], cap: int = 500,
                 allowed: dict[str, list[str]] | None = None) -> tuple[list[tuple[dict, str]], int]:
    labels = sorted(categories)
    support = Counter(row["target_interest"] for row in rows)
    upper = min(cap, min(support[label] // 2 for label in labels))
    if upper < 2:
        raise ValueError("insufficient forward support for distinct control rows")
    forward_nodes = {label: index + 1 for index, label in enumerate(labels)}
    row_offset = len(labels) + 1
    reverse_offset = row_offset + len(rows)
    reverse_nodes = {label: reverse_offset + index for index, label in enumerate(labels)}
    sink = reverse_offset + len(labels)
    starts, ends, capacities = [], [], []
    for label in labels:
        starts.extend((0, reverse_nodes[label]))
        ends.extend((forward_nodes[label], sink))
        capacities.extend((upper, upper))
    quota_edges = len(capacities)
    eligible = []
    for index, row in enumerate(rows):
        choices = eligible_categories(row, categories) if allowed is None else allowed[row["id"]]
        eligible.append(choices)
        starts.append(forward_nodes[row["target_interest"]])
        ends.append(row_offset + index)
        capacities.append(1)
        for label in choices:
            starts.append(row_offset + index)
            ends.append(reverse_nodes[label])
            capacities.append(1)

    def solve(quota):
        values = np.asarray(capacities, dtype=np.int32)
        values[:quota_edges] = quota
        graph = coo_matrix((values, (starts, ends)), shape=(sink + 1, sink + 1)).tocsr()
        return maximum_flow(graph, 0, sink, method="dinic")

    result = solve(upper)
    if result.flow_value != len(labels) * upper:
        lower, higher = 0, upper - 1
        while lower < higher:
            middle = (lower + higher + 1) // 2
            trial = solve(middle)
            if trial.flow_value == len(labels) * middle:
                lower = middle
            else:
                higher = middle - 1
        upper = lower
        if upper < 2:
            raise ValueError("no balanced joint A/B quota with smoke support")
        result = solve(upper)
    selected = []
    for index, row in enumerate(rows):
        for label in eligible[index]:
            if result.flow[row_offset + index, reverse_nodes[label]] == 1:
                selected.append((row, label))
    assert len(selected) == upper * len(labels)
    assert set(Counter(row["target_interest"] for row, _ in selected).values()) == {upper}
    assert set(Counter(label for _, label in selected).values()) == {upper}
    return selected, upper


def donor_permutation(pairs: list[tuple[dict, str]], categories: set[str]) -> list[int]:
    labels = sorted(categories)
    grouped = defaultdict(list)
    for index, (row, _) in enumerate(pairs):
        grouped[row["target_interest"]].append(index)
    offset = len(labels) + 1
    sink = offset + len(pairs)
    starts, ends, capacities = [], [], []
    for label_index, label in enumerate(labels):
        starts.append(0)
        ends.append(label_index + 1)
        capacities.append(len(grouped[label]))
        for index, (row, target) in enumerate(pairs):
            if label not in set(row["full_seen_interests"]) | {target, row["target_interest"]}:
                starts.append(label_index + 1)
                ends.append(offset + index)
                capacities.append(1)
    for index in range(len(pairs)):
        starts.append(offset + index)
        ends.append(sink)
        capacities.append(1)
    graph = coo_matrix((np.asarray(capacities, dtype=np.int32), (starts, ends)), shape=(sink + 1, sink + 1)).tocsr()
    result = maximum_flow(graph, 0, sink, method="dinic")
    if result.flow_value != len(pairs):
        raise ValueError("cannot shuffle B while preserving margins and preventing target/seen-label leakage")
    permutation = [-1] * len(pairs)
    for label_index, label in enumerate(labels):
        recipients = [index for index in range(len(pairs)) if result.flow[label_index + 1, offset + index] == 1]
        donors = sorted(grouped[label], key=lambda index: stable_key("donor", pairs[index][0]["id"]))
        allowed = np.asarray([[pairs[recipient][0]["user_id"] != pairs[donor][0]["user_id"]
                               for donor in donors] for recipient in recipients], dtype=np.int8)
        matching = maximum_bipartite_matching(coo_matrix(allowed).tocsr(), perm_type="column")
        if (matching < 0).any():
            raise ValueError("cannot exclude same-user shuffled donors")
        for recipient, donor_index in zip(recipients, matching):
            permutation[recipient] = donors[donor_index]
    assert sorted(permutation) == list(range(len(pairs)))
    return permutation


def make_view(row: dict, hidden: str | None = None, donor: dict | None = None) -> dict:
    target = row["target_interest"] if hidden is None else hidden
    keep = [index for index, label in enumerate(row["historical_interests"]) if label != hidden]
    titles = [row["historical_titles"][index] for index in keep]
    labels = [row["historical_interests"][index] for index in keep]
    timestamps = [row["history_timestamps"][index] for index in keep]
    if donor is not None:
        if donor["target_interest"] in set(row["full_seen_interests"]):
            raise ValueError("inverse evidence leaks a seen or target category")
        titles.append(donor["target_title"])
        labels.append(donor["target_interest"])
        timestamps.append(row["timestamp"])
    if target in labels or not titles or any(value >= row["timestamp"] for value in row["history_timestamps"]):
        raise ValueError("target leakage or noncausal source history")
    view = {"id": "", "source_id": row["id"], "source_sample_ids": [row["id"]],
            "user_id": row["user_id"], "historical_titles": titles, "historical_interests": labels,
            "history_timestamps": timestamps, "full_seen_interests": sorted(set(labels)),
            "target_interest": target, "positive_interests": [target],
            "direction": "forward" if hidden is None else "inverse",
            "donor_source_id": donor["id"] if donor else None,
            "donor_user_id": donor["user_id"] if donor else None,
            "evidence_interest": donor["target_interest"] if donor else None,
            "source_timestamp": row["timestamp"]}
    view["id"] = "bidirectional:" + stable_key(view["direction"], history_signature(view), target)
    return view


def freeze(rows: list[dict], categories: set[str], output: Path) -> dict:
    rows = sorted(rows, key=lambda row: stable_key("pair-order", row["id"]))
    unique = {}
    for row in rows:
        unique.setdefault(history_signature(row), row)
    rows = list(unique.values())
    occupied = set(unique)
    for source in (BASE / "train.parquet", BASE / "eval/dev_2k_behavior_interest.parquet"):
        for batch in pq.ParquetFile(source).iter_batches(columns=["historical_titles", "historical_interests"]):
            occupied.update(history_signature(row) for row in batch.to_pylist())
    allowed = {}
    for row in rows:
        allowed[row["id"]] = []
        for label in eligible_categories(row, categories):
            signature = history_signature(make_view(row, label))
            if signature not in occupied:
                allowed[row["id"]].append(label)
                occupied.add(signature)
    pairs, quota = assign_pairs(rows, categories, allowed=allowed)
    used = {row["id"] for row, _ in pairs}
    extras, counts = [], Counter()
    for row in rows:
        label = row["target_interest"]
        if row["id"] not in used and counts[label] < quota:
            extras.append(row)
            counts[label] += 1
    assert set(counts) == categories and set(counts.values()) == {quota}
    permutation = donor_permutation(pairs, categories)
    forward = [make_view(row) for row, _ in pairs]
    extra_by_label = defaultdict(list)
    for row in extras:
        extra_by_label[row["target_interest"]].append(row)
    arms = {arm: [] for arm in ARMS}
    trace = []
    for index, (row, hidden) in enumerate(pairs):
        donor = pairs[permutation[index]][0]
        replacement = extra_by_label[hidden].pop()
        inverse = {"forward": make_view(replacement), "ordinary": make_view(row, hidden),
                   "paired": make_view(row, hidden, row), "shuffled": make_view(row, hidden, donor)}
        for arm in ARMS:
            arms[arm].extend((forward[index], inverse[arm]))
        trace.append({"source_id": row["id"], "user_id": row["user_id"], "A": hidden,
                      "B": row["target_interest"], "donor_id": donor["id"],
                      "extra_forward_id": replacement["id"]})
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for arm, views in arms.items():
        count = Counter(view["target_interest"] for view in views)
        if set(count) != categories or set(count.values()) != {2 * quota}:
            raise ValueError("unbalanced auxiliary arm")
        signatures = [history_signature(view) for view in views]
        if len(set(signatures)) != len(views) or len({view["id"] for view in views}) != len(views):
            raise ValueError("duplicate input within auxiliary arm")
        path = output / f"{arm}.parquet"
        pq.write_table(pa.Table.from_pylist(views), path, compression="zstd")
        files[arm] = {"rows": len(views), "sha256": sha256_file(path), "class_counts": dict(sorted(count.items())),
                      "unique_users": len({view["user_id"] for view in views}),
                      "direction_counts": dict(Counter(view["direction"] for view in views))}
    pq.write_table(pa.Table.from_pylist(trace), output / "PAIRS.parquet", compression="zstd")
    return {"passed": True, "quota_per_direction": quota, "quota_per_class": quota * 2,
            "pairs": len(pairs), "files": files, "pairs_sha256": sha256_file(output / "PAIRS.parquet"),
            "no_upsampling": True, "same_ordered_forward_half": True, "same_ordered_inverse_targets": True,
            "shuffle_preserves_B_marginal": True, "shuffle_different_user_and_label": True,
            "quota_scope": "maximum_feasible_within_retained_source_pool", "equal_token_budget_claimed": False}


def build(output: Path, pool_cap: int = 4000) -> dict:
    from crossinterest.datasets.amazon.scripts.build import user_split

    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        raise ValueError("refusing to overwrite frozen bidirectional data")
    for name, expected in (("train.parquet", TRAIN_SHA), ("eval/dev_2k_behavior_interest.parquet", DEV_SHA),
                           ("interest_id_semantic_mapping.json", MAPPING_SHA)):
        if sha256_file(BASE / name) != expected:
            raise ValueError("original input changed")
    original = pq.read_table(BASE / "train.parquet").to_pylist()
    dev = pq.read_table(BASE / "eval/dev_2k_behavior_interest.parquet").to_pylist()
    forbidden = {row["user_id"] for row in original}
    forbidden.update(row["source_sample_id"].split(":")[1] for row in dev)
    blocked = {history_signature(row) for row in original + dev}
    original.clear()
    dev.clear()
    manifest = json.loads((BASE / "DATASET_MANIFEST.json").read_text())
    categories = set(manifest["balance"]["support"])
    source = Path(manifest["source"]["mother_pool"]) / "train.jsonl"
    prep = json.loads(Path(manifest["source"]["prep_manifest"]).read_text())
    if source.stat().st_size != manifest["source"]["train_bytes"]:
        raise ValueError("mother pool size changed")
    pools = {(direction, label): [] for direction in ("A", "B") for label in categories}
    counts = Counter()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            counts["source_rows"] += 1
            raw = json.loads(line)
            if raw["user_id"] in forbidden or raw["target_interest"] not in categories:
                continue
            if raw["split"] != "train" or user_split(raw["user_id"], prep["seed"], 0.1) != "train":
                raise ValueError("non-Train user in auxiliary source")
            choices = eligible_categories(raw, categories)
            if not choices or raw["target_interest"] in raw["full_seen_interests"]:
                continue
            row = {key: raw[key] for key in FIELDS}
            if history_signature(row) in blocked:
                continue
            priority = int(stable_key("source-pool", row["id"]), 16)
            for key in [("B", row["target_interest"])] + [("A", label) for label in choices]:
                pool = pools[key]
                if len(pool) < pool_cap:
                    heapq.heappush(pool, (-priority, row["id"], row))
                elif priority < -pool[0][0]:
                    heapq.heapreplace(pool, (-priority, row["id"], row))
            counts["eligible_rows"] += 1
            if counts["source_rows"] % 500000 == 0:
                print(json.dumps(dict(counts)), flush=True)
    if counts["source_rows"] != prep["unbalanced"]["train"]["samples"]:
        raise ValueError("incomplete source scan")
    selected = {item[1]: item[2] for pool in pools.values() for item in pool}
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(selected.values())), output / "CANDIDATES.parquet", compression="zstd")
    scan = {"source_sha256": digest.hexdigest(), "counts": dict(counts), "pool_cap": pool_cap,
            "excluded_users": len(forbidden), "original_n_sha256": TRAIN_SHA,
            "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA}
    write_json_atomic(output / "SCAN.json", scan)
    print(json.dumps({"event": "matching", "candidates": len(selected)}), flush=True)
    result = freeze(list(selected.values()), categories, output)
    result.update(builder_sha256=sha256_file(Path(__file__)), **scan)
    write_json_atomic(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "bidirectional/data")
    parser.add_argument("--pool-cap", type=int, default=4000)
    args = parser.parse_args()
    result = build(args.output_dir, args.pool_cap)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()