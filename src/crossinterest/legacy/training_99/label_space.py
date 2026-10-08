"""Frozen output-label mappings for Amazon SFT variants."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


LABEL_MODES = ("category", "interest_id")
MAPPING_NAMESPACE = "amazon-interest-id-v1"
INTEREST_ID_PATTERN = re.compile(r"<interest_(\d{3})>")


def _mapping_fingerprint(category_to_id: dict[str, str]) -> str:
    payload = json.dumps(
        category_to_id,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_label_mapping(categories: Iterable[str]) -> dict:
    unique = set(categories)
    ordered = sorted(
        unique,
        key=lambda category: hashlib.sha256(
            f"{MAPPING_NAMESPACE}:{category}".encode()
        ).digest(),
    )
    category_to_id = {
        category: f"<interest_{index:03d}>"
        for index, category in enumerate(ordered)
    }
    id_to_category = {
        interest_id: category for category, interest_id in category_to_id.items()
    }
    return {
        "schema_version": 1,
        "namespace": MAPPING_NAMESPACE,
        "assignment": "sort by SHA256(namespace:category), then assign contiguous zero-based IDs",
        "label_count": len(category_to_id),
        "mapping_fingerprint": _mapping_fingerprint(category_to_id),
        "category_to_id": dict(sorted(category_to_id.items())),
        "id_to_category": dict(sorted(id_to_category.items())),
    }


def validate_label_mapping(
    mapping: dict,
    expected_categories: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> dict:
    category_to_id = mapping.get("category_to_id")
    id_to_category = mapping.get("id_to_category")
    if not isinstance(category_to_id, dict) or not isinstance(id_to_category, dict):
        raise ValueError("mapping must contain category_to_id and id_to_category objects")
    if len(category_to_id) != len(id_to_category):
        raise ValueError("mapping directions have different sizes")
    if mapping.get("label_count") != len(category_to_id):
        raise ValueError("mapping label_count is inconsistent")
    if expected_count is not None and len(category_to_id) != expected_count:
        raise ValueError(
            f"expected {expected_count} mapped labels, found {len(category_to_id)}"
        )
    expected_ids = {f"<interest_{index:03d}>" for index in range(len(category_to_id))}
    if set(category_to_id.values()) != expected_ids:
        raise ValueError("interest IDs must be contiguous from zero")
    if any(INTEREST_ID_PATTERN.fullmatch(value) is None for value in category_to_id.values()):
        raise ValueError("mapping contains a malformed interest ID")
    inverse = {interest_id: category for category, interest_id in category_to_id.items()}
    if inverse != id_to_category:
        raise ValueError("id_to_category is not the inverse of category_to_id")
    if expected_categories is not None and set(category_to_id) != set(expected_categories):
        raise ValueError("mapping categories differ from the dataset target support")
    fingerprint = _mapping_fingerprint(category_to_id)
    if mapping.get("mapping_fingerprint") != fingerprint:
        raise ValueError("mapping fingerprint mismatch")
    return mapping


def load_label_mapping(
    path: Path,
    expected_categories: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> dict:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    return validate_label_mapping(mapping, expected_categories, expected_count)


def output_label(category: str, label_mode: str, mapping: dict | None) -> str:
    if label_mode == "category":
        return category
    if label_mode == "interest_id":
        if mapping is None:
            raise ValueError("interest_id mode requires a label mapping")
        try:
            return mapping["category_to_id"][category]
        except KeyError as error:
            raise ValueError(f"category is absent from the label mapping: {category}") from error
    raise ValueError(f"unsupported label mode: {label_mode}")


def category_from_output(value: str, label_mode: str, mapping: dict | None) -> str | None:
    if label_mode == "category":
        return value
    if label_mode == "interest_id":
        if mapping is None:
            raise ValueError("interest_id mode requires a label mapping")
        return mapping["id_to_category"].get(value)
    raise ValueError(f"unsupported label mode: {label_mode}")
