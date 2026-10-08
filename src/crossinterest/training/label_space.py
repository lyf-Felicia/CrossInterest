"""Frozen output-label mappings for Amazon SFT variants."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


LABEL_MODES = ("category", "interest_id", "interest_id_category")
INTEREST_ID_LABEL_MODES = ("interest_id", "interest_id_category")
MAPPING_NAMESPACE = "amazon-child-interest-id-semantic-v1"
INTEREST_ID_PATTERN = re.compile(r"<interest_(\d{3})>")
INTEREST_ID_CATEGORY_PATTERN = re.compile(r"\[(<interest_\d{3}>)\] (.+)")


def prompt_schema_version(
    label_mode: str,
    history_summary: bool = False,
    target_rationale: bool = False,
    target_reasoning: bool = False,
) -> int | str:
    versions = {
        "category": 3,
        "interest_id": "7-single-interest-id",
        "interest_id_category": "8-interest-id-category",
    }
    try:
        version = versions[label_mode]
    except KeyError as error:
        raise ValueError(f"unsupported label mode: {label_mode}") from error
    suffixes = []
    if history_summary:
        suffixes.append("target-blind-history-summary-v1")
    if target_rationale:
        suffixes.append("target-rationale-aux-v1")
    if target_reasoning:
        suffixes.append("reasoning-first-v1")
    return f"{version}-{'-'.join(suffixes)}" if suffixes else version


def _mapping_fingerprint(identity: dict) -> str:
    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _child_label(category: str) -> str:
    _, separator, child = category.rpartition(" > ")
    if not separator or not child:
        raise ValueError(f"Interest IDs require a child category label: {category}")
    return child


def build_label_mapping(categories: Iterable[str]) -> dict:
    unique_categories = set(categories)
    if not unique_categories:
        raise ValueError("cannot build an Interest ID mapping without categories")
    category_to_child_label = {
        category: _child_label(category) for category in unique_categories
    }
    if len(set(category_to_child_label.values())) != len(category_to_child_label):
        raise ValueError("Interest IDs require unique child category labels")
    ordered_categories = sorted(
        unique_categories,
        key=lambda category: category_to_child_label[category],
    )
    category_to_token = {
        category: f"<interest_{index:03d}>"
        for index, category in enumerate(ordered_categories)
    }
    category_to_tokens = {
        category: [token] for category, token in category_to_token.items()
    }
    category_to_id = dict(category_to_token)
    id_to_category = {
        interest_id: category for category, interest_id in category_to_id.items()
    }
    token_to_text = {
        token: category_to_child_label[category]
        for category, token in category_to_token.items()
    }
    special_tokens = list(category_to_token.values())
    identity = {
        "namespace": MAPPING_NAMESPACE,
        "category_to_child_label": category_to_child_label,
        "category_to_tokens": category_to_tokens,
        "token_to_text": token_to_text,
    }
    return {
        "schema_version": 4,
        "namespace": MAPPING_NAMESPACE,
        "method": "single-level Amazon child-category Interest ID with semantic initialization",
        "assignment": (
            "lexically index the 112 unique child category labels; encode each child "
            "category as one unique token initialized only from that child label's "
            "base-model embeddings"
        ),
        "label_count": len(category_to_id),
        "semantic_token_count": len(special_tokens),
        "mapping_fingerprint": _mapping_fingerprint(identity),
        "special_tokens": special_tokens,
        "category_to_child_label": dict(sorted(category_to_child_label.items())),
        "category_to_token": category_to_token,
        "category_to_tokens": category_to_tokens,
        "token_to_text": token_to_text,
        "category_to_id": dict(sorted(category_to_id.items())),
        "id_to_category": dict(sorted(id_to_category.items())),
    }


def validate_label_mapping(
    mapping: dict,
    expected_categories: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> dict:
    if mapping.get("schema_version") != 4:
        raise ValueError("Interest ID mapping schema_version must be 4")
    if mapping.get("namespace") != MAPPING_NAMESPACE:
        raise ValueError("Interest ID mapping namespace mismatch")
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
    expected = build_label_mapping(category_to_id)
    structural_fields = (
        "semantic_token_count",
        "special_tokens",
        "category_to_child_label",
        "category_to_token",
        "category_to_tokens",
        "token_to_text",
        "category_to_id",
    )
    for field in structural_fields:
        if mapping.get(field) != expected[field]:
            raise ValueError(f"Interest ID mapping {field} is inconsistent")
    if any(INTEREST_ID_PATTERN.fullmatch(value) is None for value in category_to_id.values()):
        raise ValueError("mapping contains a malformed Interest ID")
    inverse = {interest_id: category for category, interest_id in category_to_id.items()}
    if inverse != id_to_category:
        raise ValueError("id_to_category is not the inverse of category_to_id")
    if expected_categories is not None and set(category_to_id) != set(expected_categories):
        raise ValueError("mapping categories differ from the dataset target support")
    identity = {
        "namespace": mapping["namespace"],
        "category_to_child_label": mapping["category_to_child_label"],
        "category_to_tokens": mapping["category_to_tokens"],
        "token_to_text": mapping["token_to_text"],
    }
    fingerprint = _mapping_fingerprint(identity)
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
    if label_mode in INTEREST_ID_LABEL_MODES:
        if mapping is None:
            raise ValueError(f"{label_mode} mode requires a label mapping")
        try:
            interest_id = mapping["category_to_id"][category]
        except KeyError as error:
            raise ValueError(f"category is absent from the label mapping: {category}") from error
        if label_mode == "interest_id_category":
            return f"[{interest_id}] {category}"
        return interest_id
    raise ValueError(f"unsupported label mode: {label_mode}")


def category_from_output(value: str, label_mode: str, mapping: dict | None) -> str | None:
    if label_mode == "category":
        return value
    if label_mode in INTEREST_ID_LABEL_MODES:
        if mapping is None:
            raise ValueError(f"{label_mode} mode requires a label mapping")
        if label_mode == "interest_id":
            return mapping["id_to_category"].get(value)
        match = INTEREST_ID_CATEGORY_PATTERN.fullmatch(value)
        if match is None:
            return None
        interest_id, category = match.groups()
        return category if mapping["id_to_category"].get(interest_id) == category else None
    raise ValueError(f"unsupported label mode: {label_mode}")
