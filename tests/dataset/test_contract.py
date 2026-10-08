import json
from collections import Counter
from pathlib import Path

from scripts.build import (
    build_balanced_train,
    build_user_records,
    clean_taxonomy,
    user_split,
)


PRODUCTS = {
    "A": ("A title", "L2-A", "L2-A > L3-A"),
    "B": ("B title", "L2-B", "L2-B > L3-B"),
    "C": ("C title", "L2-C", "L2-C > L3-C"),
    "D": ("D title", "L2-D", "L2-D > L3-D"),
}


def test_history_is_strict_dynamic_prefix_and_multi_target_is_single() -> None:
    interactions = [
        (1000, "A", 5.0),
        (2000, "B", 5.0),
        (2000, "C", 4.0),
        (3000, "D", 5.0),
    ]
    records = build_user_records("U1", interactions, PRODUCTS, "train", 7, 50, 1)
    assert len(records) == 2
    second_event, third_event = records
    assert second_event["history_parent_asins"] == ["A"]
    assert second_event["same_history_new_interest_count"] == 2
    assert second_event["target_parent_asin"] in {"B", "C"}
    assert third_event["history_parent_asins"] == ["A", "B", "C"]
    assert third_event["full_seen_interests"] == [
        "L2-A > L3-A",
        "L2-B > L3-B",
        "L2-C > L3-C",
    ]
    assert all(value < third_event["timestamp"] for value in third_event["history_timestamps"])


def test_complete_prefix_not_h50_determines_novelty() -> None:
    products = {
        str(index): (str(index), f"L2-{index}", f"L2-{index} > L3-{index}")
        for index in range(52)
    }
    interactions = [(index, str(index), 5.0) for index in range(51)]
    interactions.append((100, "0", 5.0))
    records = build_user_records("U2", interactions, products, "train", 7, 5, 1)
    assert len(records) == 50
    assert records[-1]["target_parent_asin"] == "50"
    assert all(record["timestamp"] != 100 for record in records)
    assert len(records[-1]["history_parent_asins"]) == 5


def test_user_split_is_deterministic_and_exclusive() -> None:
    first = {user: user_split(user, 17, 0.1) for user in ("U1", "U2", "U3")}
    second = {user: user_split(user, 17, 0.1) for user in ("U1", "U2", "U3")}
    assert first == second
    assert set(first.values()) <= {"train", "dev"}


def test_minimum_history_is_applied_before_sampling() -> None:
    interactions = [(index, parent_asin, 5.0) for index, parent_asin in enumerate(PRODUCTS)]
    records = build_user_records("U3", interactions, PRODUCTS, "train", 7, 50, 3)
    assert len(records) == 1
    assert records[0]["full_history_length"] == 3


def test_taxonomy_allows_core_departments_and_rejects_storefronts() -> None:
    assert clean_taxonomy(
        {"categories": ["Home & Kitchen", "Bedding", "Sheets & Pillowcases"]}
    ) == ("Bedding", "Bedding > Sheets & Pillowcases")
    assert clean_taxonomy(
        {"categories": ["Home & Kitchen", "Umbra", "Designers"]}
    ) is None
    assert clean_taxonomy(
        {"categories": ["Home & Kitchen", "Dorm Room HQ", "Cooling & Air Quality"]}
    ) is None


def test_only_train_is_balanced(tmp_path: Path) -> None:
    train = tmp_path / "unbalanced" / "train.jsonl"
    dev = tmp_path / "unbalanced" / "dev.jsonl"
    train.parent.mkdir(parents=True)
    train_rows = [
        {"id": f"train:A:{index}", "target_interest": "A"}
        for index in range(4)
    ] + [
        {"id": f"train:B:{index}", "target_interest": "B"}
        for index in range(3)
    ]
    dev_rows = [
        {"id": f"dev:A:{index}", "target_interest": "A"}
        for index in range(5)
    ] + [{"id": "dev:B:0", "target_interest": "B"}]
    train.write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    dev.write_text("".join(json.dumps(row) + "\n" for row in dev_rows))
    before = dev.read_bytes()
    stats = build_balanced_train(train, tmp_path / "balanced" / "train.jsonl", 3, 2)
    assert stats["sample_count"] == 6
    assert stats["support"] == ["A", "B"]
    assert stats["quota_source"] == "minimum_actual_support"
    assert stats["per_label_count"] == 3
    counts = Counter(
        json.loads(line)["target_interest"]
        for line in (tmp_path / "balanced" / "train.jsonl").read_text().splitlines()
    )
    assert counts == {"A": 3, "B": 3}
    assert dev.read_bytes() == before