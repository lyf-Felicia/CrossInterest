"""Freeze factor masks and an independent-user CIR diagnostic sample."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import HERE, BASE, sha256_file
from crossinterest.methods.cir.data import balanced_assignment, category_view, eligible_categories
from crossinterest.methods.mechanism.data import history_signature, TRAIN_SHA, DEV_SHA, MAPPING_SHA


ROOT = HERE / "cir_factorization"
SEED = 20260919


def randomized_masks(seen: np.ndarray, targets: np.ndarray, rounds: int = 20) -> tuple[np.ndarray, dict]:
    if seen.ndim != 2 or seen.dtype != np.bool_ or targets.shape != (len(seen),):
        raise ValueError("invalid factor mask arrays")
    if seen[np.arange(len(seen)), targets].any():
        raise ValueError("target occurs in visible history")
    result = seen.copy()
    row_indices, column_indices = np.nonzero(result)
    generator = np.random.default_rng(SEED)
    accepted = 0
    proposals = rounds * len(row_indices)
    if len(row_indices) < 2:
        raise ValueError("mask randomization requires at least two memberships")
    for _ in range(rounds):
        pairs = generator.integers(0, len(row_indices), size=(len(row_indices), 2))
        for first, second in pairs:
            first_row, second_row = row_indices[first], row_indices[second]
            first_label, second_label = column_indices[first], column_indices[second]
            if (first_row == second_row or first_label == second_label or
                    targets[first_row] == second_label or targets[second_row] == first_label or
                    result[first_row, second_label] or result[second_row, first_label]):
                continue
            result[first_row, first_label] = result[second_row, second_label] = False
            result[first_row, second_label] = result[second_row, first_label] = True
            column_indices[first], column_indices[second] = second_label, first_label
            accepted += 1
    if not (np.array_equal(result.sum(0), seen.sum(0)) and np.array_equal(result.sum(1), seen.sum(1))):
        raise AssertionError("degree-preserving randomization changed margins")
    if result[np.arange(len(result)), targets].any():
        raise AssertionError("randomization exposed a target")
    return result, {"algorithm": "fixed_seed_bipartite_double_edge_swaps_not_uniform_sampling",
                    "seed": SEED, "rounds": rounds, "proposals": proposals, "accepted": accepted,
                    "unchanged_row_fraction": float(np.all(result == seen, axis=1).mean()),
                    "membership_overlap_fraction": float((result & seen).sum() / max(1, seen.sum())),
                    "row_margins_equal": True, "column_margins_equal": True}


def mask_arrays(rows: list[dict], categories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {category: index for index, category in enumerate(categories)}
    seen = np.zeros((len(rows), len(categories)), dtype=bool)
    targets = np.asarray([lookup[row["target_interest"]] for row in rows], dtype=np.int64)
    for index, row in enumerate(rows):
        for category in set(row["historical_interests"]):
            if category in lookup:
                seen[index, lookup[category]] = True
    if seen[np.arange(len(rows)), targets].any():
        raise ValueError("CIR target is not absent")
    return seen, targets


def match_donors(rows: list[dict], categories: list[str]) -> tuple[list[int], dict]:
    seen, targets = mask_arrays(rows, categories)
    lengths = np.asarray([len(row["historical_interests"]) for row in rows])
    cardinalities = np.asarray([len(set(row["historical_interests"])) for row in rows])
    costs = (np.abs(lengths[:, None] - lengths[None, :]) +
             10 * np.abs(cardinalities[:, None] - cardinalities[None, :])).astype(float)
    for index, row in enumerate(rows):
        forbidden = seen[:, targets[index]].copy()
        forbidden |= np.asarray([donor["user_id"] == row["user_id"] for donor in rows])
        costs[index, forbidden] = np.inf
    selected_rows, donors = linear_sum_assignment(costs)
    if not np.array_equal(selected_rows, np.arange(len(rows))) or not np.isfinite(costs[selected_rows, donors]).all():
        raise ValueError("independent probe donor matching is infeasible")
    return donors.tolist(), {
        "algorithm": "minimum_cost_different_user_target_absent_permutation",
        "length_absolute_difference_mean": float(np.abs(lengths - lengths[donors]).mean()),
        "category_count_absolute_difference_mean": float(np.abs(cardinalities - cardinalities[donors]).mean()),
        "exact_length_and_category_match_fraction": float(((lengths == lengths[donors]) &
                                                           (cardinalities == cardinalities[donors])).mean()),
    }


def freeze() -> dict:
    output = ROOT / "data"
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        return audit()
    for path, expected in ((BASE / "train.parquet", TRAIN_SHA),
                           (BASE / "eval/dev_2k_behavior_interest.parquet", DEV_SHA),
                           (BASE / "interest_id_semantic_mapping.json", MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"original source changed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((BASE / "interest_id_semantic_mapping.json").read_text())
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    auxiliary = HERE / "cir/data/external_balanced/train.parquet"
    rows = pq.read_table(auxiliary).to_pylist()
    if len(rows) != 83888 or set(Counter(row["target_interest"] for row in rows).values()) != {749}:
        raise ValueError("expected the original balanced 83,888 CIR rows")
    seen, targets = mask_arrays(rows, categories)
    pseudo, randomization = randomized_masks(seen, targets)
    np.savez_compressed(output / "masks.npz", ids=np.asarray([row["id"] for row in rows]),
                        categories=np.asarray(categories), seen=seen, pseudo=pseudo, targets=targets)
    blocked_users = set(pq.read_table(BASE / "train.parquet", columns=["user_id"])["user_id"].to_pylist())
    blocked_users.update(row["user_id"] for row in rows)
    blocked_inputs = {history_signature(row) for row in rows}
    for path in (BASE / "train.parquet", BASE / "eval/dev_2k_behavior_interest.parquet"):
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                blocked_inputs.add(history_signature(row))
                if "user_id" in row:
                    blocked_users.add(row["user_id"])
                else:
                    blocked_users.add(row["source_sample_id"].split(":")[1])
    for arm in ("forward", "ordinary", "paired", "shuffled"):
        path = HERE / "bidirectional/data" / f"{arm}.parquet"
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                blocked_users.add(row["user_id"])
                blocked_inputs.add(history_signature(row))
    old_cir = HERE / "curriculum/seen500-novel112k/seen.parquet"
    for batch in pq.ParquetFile(old_cir).iter_batches():
        for row in batch.to_pylist():
            blocked_users.add(row["user_id"])
            blocked_inputs.add(history_signature(row))
    source_manifest = json.loads((BASE / "DATASET_MANIFEST.json").read_text())
    source = Path(source_manifest["source"]["mother_pool"]) / "train.jsonl"
    original_cir_manifest = json.loads((auxiliary.parent / "MANIFEST.json").read_text())
    pools = {category: [] for category in categories}
    label_set = set(categories)
    raw_digest = hashlib.sha256()
    scanned = 0
    current_user = None
    per_user = {}
    visited_users = set()

    def finish_user():
        for category, (_, row) in per_user.items():
            priority = int(hashlib.sha256(f"{SEED}:{current_user}:{category}".encode()).hexdigest(), 16)
            pool = pools[category]
            if len(pool) >= 200 and priority >= -pool[0][0]:
                continue
            view = category_view(row, category, label_set)
            if history_signature(view) in blocked_inputs:
                continue
            view["probe_source"] = row
            entry = (-priority, current_user, view)
            if len(pool) < 200:
                heapq.heappush(pool, entry)
            else:
                heapq.heapreplace(pool, entry)

    with source.open("rb") as handle:
        for line in handle:
            raw_digest.update(line)
            row = json.loads(line)
            scanned += 1
            if row["split"] != "train":
                raise ValueError("probe source contains a non-Train row")
            if row["user_id"] != current_user:
                finish_user()
                current_user = row["user_id"]
                if current_user in visited_users:
                    raise ValueError("mother pool user groups are not contiguous")
                visited_users.add(current_user)
                per_user = {}
            if scanned % 500000 == 0:
                print(json.dumps({"event": "probe_mother_scan", "rows": scanned}), flush=True)
            if row["user_id"] in blocked_users:
                continue
            for category in eligible_categories(row, label_set):
                priority = int(hashlib.sha256(f"{SEED}:{row['id']}:{category}".encode()).hexdigest(), 16)
                if category not in per_user or priority < per_user[category][0]:
                    per_user[category] = (priority, row)
    finish_user()
    if raw_digest.hexdigest() != original_cir_manifest["mother_sha256"]:
        raise ValueError("original Train mother pool changed")
    candidate_counts = {category: len(pool) for category, pool in sorted(pools.items())}
    write_json_atomic(output / "PROBE_CANDIDATE_COUNTS.json", {
        "mother_sha256": raw_digest.hexdigest(), "mother_rows": scanned,
        "blocked_users": len(blocked_users), "counts": candidate_counts})
    supported = {category for category, count in candidate_counts.items() if count}
    if not supported:
        raise ValueError("no independent probe candidates in any category")
    candidates = [entry[2] for pool in pools.values() for entry in pool]
    probes, assignment = balanced_assignment(candidates, supported, max_per_class=20)
    if any(row["user_id"] in blocked_users or history_signature(row) in blocked_inputs for row in probes):
        raise ValueError("probe isolation failed")
    donors, donor_audit = match_donors(probes, categories)
    for index, row in enumerate(probes):
        row["donor_index"] = donors[index]
    pq.write_table(pa.Table.from_pylist(probes), output / "probe.parquet", compression="zstd")
    per_class_gate = Counter()
    gate_ids = []
    for row in sorted(rows, key=lambda row: hashlib.sha256(f"gate:{SEED}:{row['id']}".encode()).hexdigest()):
        if per_class_gate[row["target_interest"]] < 2:
            gate_ids.append(row["id"])
            per_class_gate[row["target_interest"]] += 1
    write_json_atomic(output / "gate_ids.json", gate_ids)
    result = {"passed": True, "seed": SEED, "auxiliary_sha256": sha256_file(auxiliary),
              "builder_sha256": sha256_file(Path(__file__)), "categories": categories,
              "rows": len(rows), "per_class": 749, "randomization": randomization,
              "probe_candidate_sha256": raw_digest.hexdigest(), "probe_assignment": assignment,
              "probe_missing_categories": sorted(label_set - supported),
              "probe_requested_classes": len(categories), "probe_requested_per_class": 20,
              "probe_support_scope": "supported_classes_only_not_112_class_generalization",
              "probe_donors": donor_audit, "blocked_users": len(blocked_users),
              "probe_user_overlap": 0, "probe_input_overlap": 0,
              "gate_rows": len(gate_ids), "gate_scope": "fixed_train_224_two_per_class_not_performance_selection",
              "files": {name: sha256_file(output / name) for name in ("masks.npz", "probe.parquet", "gate_ids.json")}}
    write_json_atomic(manifest_path, result)
    return audit()


def audit() -> dict:
    output = ROOT / "data"
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"factor data changed: {name}")
    auxiliary = HERE / "cir/data/external_balanced/train.parquet"
    if sha256_file(auxiliary) != manifest["auxiliary_sha256"]:
        raise ValueError("original CIR changed")
    rows = pq.read_table(auxiliary).to_pylist()
    arrays = np.load(output / "masks.npz", allow_pickle=False)
    seen, targets = mask_arrays(rows, manifest["categories"])
    if not (np.array_equal(seen, arrays["seen"]) and np.array_equal(targets, arrays["targets"]) and
            np.array_equal(arrays["ids"], np.asarray([row["id"] for row in rows]))):
        raise ValueError("factor rows do not match original histories")
    pseudo = arrays["pseudo"]
    if pseudo[np.arange(len(pseudo)), targets].any() or not (
            np.array_equal(pseudo.sum(0), seen.sum(0)) and np.array_equal(pseudo.sum(1), seen.sum(1))):
        raise ValueError("pseudo masks changed target absence or margins")
    probes = pq.read_table(output / "probe.parquet").to_pylist()
    if len({row["user_id"] for row in probes}) != len(probes):
        raise ValueError("probe user repeated")
    if sorted(row["donor_index"] for row in probes) != list(range(len(probes))):
        raise ValueError("probe donors are not a permutation")
    for row in probes:
        source = row["probe_source"]
        retained = [index for index, category in enumerate(source["historical_interests"])
                    if category != row["target_interest"]]
        for field in ("historical_titles", "historical_interests", "history_timestamps"):
            if row[field] != [source[field][index] for index in retained]:
                raise ValueError("probe reconstruction differs from source")
        if len(retained) < 5 or max(row["history_timestamps"]) >= row["source_timestamp"]:
            raise ValueError("probe temporal or minimum-history contract failed")
        donor = probes[row["donor_index"]]
        if row["user_id"] == donor["user_id"] or row["target_interest"] in donor["historical_interests"]:
            raise ValueError("invalid mismatched history")
    result = {"passed": True, "rows": len(rows), "probe_rows": len(probes),
              "probe_class_counts": dict(sorted(Counter(row["target_interest"] for row in probes).items())),
              "true_masks_reconstructed": True, "pseudo_margins_and_targets_verified": True,
              "probe_source_reconstruction_verified": True, "manifest_sha256": sha256_file(output / "MANIFEST.json")}
    write_json_atomic(output / "INDEPENDENT_AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "audit"))
    arguments = parser.parse_args()
    print(json.dumps(freeze() if arguments.action == "freeze" else audit(), ensure_ascii=False))