"""Independently verify directional views against original Train records."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.mechanism.data import history_signature


def audit() -> dict:
    root = HERE / "bidirectional/data"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    trace = pq.read_table(root / "PAIRS.parquet").to_pylist()
    wanted = {row[key] for row in trace for key in ("source_id", "donor_id", "extra_forward_id")}
    source_manifest = json.loads((BASE / "DATASET_MANIFEST.json").read_text())
    source = Path(source_manifest["source"]["mother_pool"]) / "train.jsonl"
    originals, digest = {}, hashlib.sha256()
    for line in source.open("rb"):
        digest.update(line)
        row = json.loads(line)
        if row["id"] in wanted:
            assert row["id"] not in originals
            originals[row["id"]] = row
    assert set(originals) == wanted
    assert digest.hexdigest() == manifest["source_sha256"]
    train = pq.read_table(BASE / "train.parquet").to_pylist()
    dev = pq.read_table(BASE / "eval/dev_2k_behavior_interest.parquet").to_pylist()
    forbidden = {row["user_id"] for row in train} | {row["source_sample_id"].split(":")[1] for row in dev}
    blocked = {history_signature(row) for row in train + dev}
    labels = set(source_manifest["balance"]["support"])
    arms = {arm: pq.read_table(root / f"{arm}.parquet").to_pylist() for arm in manifest["files"]}
    quota = manifest["quota_per_direction"]
    for arm, rows in arms.items():
        assert sha256_file(root / f"{arm}.parquet") == manifest["files"][arm]["sha256"]
        assert len(rows) == len(trace) * 2 == 112 * 2 * quota
        counts = Counter(row["target_interest"] for row in rows)
        assert set(counts) == labels and set(counts.values()) == {2 * quota}
        signatures = {history_signature(row) for row in rows}
        assert len(signatures) == len(rows) and not signatures & blocked
        assert len({row["id"] for row in rows}) == len(rows)
        assert not {row["user_id"] for row in rows} & forbidden
    for index, item in enumerate(trace):
        source_row = originals[item["source_id"]]
        assert source_row["split"] == "train"
        assert source_row["target_interest"] == item["B"]
        assert item["B"] not in source_row["full_seen_interests"]
        assert all(value < source_row["timestamp"] for value in source_row["history_timestamps"])
        forward = arms["forward"][index * 2]
        for arm in arms:
            assert arms[arm][index * 2] == forward
        for view, original in ((forward, source_row),
                               (arms["forward"][index * 2 + 1], originals[item["extra_forward_id"]])):
            for field in ("historical_titles", "historical_interests", "history_timestamps", "target_interest", "user_id"):
                assert view[field] == original[field]
            assert view["direction"] == "forward" and view["evidence_interest"] is None
        hidden = item["A"]
        keep = [position for position, label in enumerate(source_row["historical_interests"]) if label != hidden]
        assert len(keep) >= 5 and len(keep) < len(source_row["historical_interests"])
        ordinary = arms["ordinary"][index * 2 + 1]
        for field in ("historical_titles", "historical_interests", "history_timestamps"):
            assert ordinary[field] == [source_row[field][position] for position in keep]
        donor = originals[item["donor_id"]]
        assert donor["user_id"] != source_row["user_id"]
        assert donor["target_interest"] not in set(source_row["full_seen_interests"]) | {item["B"]}
        assert donor["user_id"] not in forbidden
        for arm, evidence in (("paired", source_row), ("shuffled", donor)):
            view = arms[arm][index * 2 + 1]
            for field in ("historical_titles", "historical_interests", "history_timestamps"):
                assert view[field][:-1] == ordinary[field]
            assert view["historical_titles"][-1] == evidence["target_title"]
            assert view["historical_interests"][-1] == evidence["target_interest"]
            assert view["history_timestamps"][-1] == source_row["timestamp"]
            assert view["donor_source_id"] == evidence["id"]
        for arm in ("ordinary", "paired", "shuffled"):
            view = arms[arm][index * 2 + 1]
            assert view["target_interest"] == hidden and hidden not in view["historical_interests"]
            assert view["direction"] == "inverse" and view["positive_interests"] == [hidden]
    assert Counter(item["B"] for item in trace) == Counter(arms["shuffled"][index * 2 + 1]["evidence_interest"]
                                                          for index in range(len(trace)))
    assert len({item["donor_id"] for item in trace}) == len(trace)
    assert not {item["source_id"] for item in trace} & {item["extra_forward_id"] for item in trace}
    result = {"passed": True, "manifest_sha256": sha256_file(root / "MANIFEST.json"),
              "auditor_sha256": sha256_file(Path(__file__)), "source_sha256": digest.hexdigest(),
              "source_records_verified": len(originals), "rows_per_arm": 224 * quota,
              "classes": 112, "per_class": 2 * quota, "per_direction_class": quota,
              "target_leakage": 0, "duplicate_inputs_per_arm": 0, "n_dev_user_overlap": 0,
              "n_dev_input_overlap": 0, "same_forward_rows": True, "same_inverse_R_and_A": True,
              "shuffled_B_multiset_exact": True, "inverse_evidence_not_in_forward": True}
    write_json_atomic(root / "INDEPENDENT_AUDIT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    audit()