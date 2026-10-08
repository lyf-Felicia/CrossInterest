"""Dataset adapters and prompts shared by MeRA training and evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from torch.utils.data import Dataset


OFFICIAL_INPUT_PREFIX = "The user has given high ratings to the following categories"
UNIFIED_EVAL_SEED = 20260806
QWEN2_OFFICIAL_PROMPT_PREFIX = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request. \n\n### Instruction:\n"
)


def parse_quoted_categories(value: str) -> list[str]:
    return re.findall(r'"(.*?)"', value)


def parse_official_history(value: str) -> list[str]:
    if OFFICIAL_INPUT_PREFIX not in value:
        raise ValueError("unrecognized official MeRA input prompt")
    _, separator, literal = value.partition(":")
    if not separator:
        raise ValueError("official MeRA input lacks a history separator")
    history = ast.literal_eval(literal.strip())
    if isinstance(history, str):
        history = [history]
    if not isinstance(history, (list, tuple)) or not all(
        isinstance(category, str) for category in history
    ):
        raise ValueError("official MeRA history is not a category sequence")
    return list(history)


def load_official_rows(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for index, record in enumerate(records):
        history = parse_official_history(record["input"])
        targets = parse_quoted_categories(record["output"])
        if not targets:
            raise ValueError(f"{path}:{index} has no target categories")
        rows.append(
            {
                "sample_id": f"{path.stem}:{index}",
                "history": history,
                "full_seen": sorted(set(history)),
                "targets": targets,
                "official_instruction": record["instruction"],
                "official_input": record["input"],
            }
        )
    return rows


def load_home_rows(path: Path, training: bool) -> list[dict]:
    columns = [
        "id" if training else "sample_id",
        "historical_interests",
        "full_seen_interests",
        "target_interest",
    ]
    if not training:
        columns.append("candidate_interests")
    records = pq.read_table(path, columns=columns, memory_map=True).to_pylist()
    return [
        {
            "sample_id": record["id" if training else "sample_id"],
            "history": list(record["historical_interests"]),
            "full_seen": list(record["full_seen_interests"]),
            "targets": [record["target_interest"]],
            **(
                {"candidates": list(record["candidate_interests"])}
                if not training
                else {}
            ),
        }
        for record in records
    ]


def validate_rows(
    rows: Iterable[dict],
    labels: set[str],
    single_target: bool,
    prediction_labels: set[str] | None = None,
) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("dataset split is empty")
    unknown = set()
    novelty_violations = []
    identifiers = set()
    duplicate_ids = 0
    target_counts = []
    prediction_labels = prediction_labels or labels
    for row in rows:
        if row["sample_id"] in identifiers:
            duplicate_ids += 1
        identifiers.add(row["sample_id"])
        target_counts.append(len(row["targets"]))
        if single_target and len(row["targets"]) != 1:
            raise ValueError(f"{row['sample_id']} must contain exactly one target")
        unknown.update((set(row["history"]) | set(row["targets"])) - labels)
        unknown_targets = set(row["targets"]) - prediction_labels
        if unknown_targets:
            raise ValueError(
                f"{row['sample_id']} has targets outside the prediction space: "
                f"{sorted(unknown_targets)}"
            )
        if set(row["targets"]) & set(row["full_seen"]):
            novelty_violations.append(row["sample_id"])
    if unknown:
        raise ValueError(f"rows contain labels outside the label space: {sorted(unknown)[:3]}")
    if novelty_violations:
        raise ValueError(f"novelty violations: {novelty_violations[:3]}")
    if duplicate_ids:
        raise ValueError(f"split contains {duplicate_ids} duplicate sample IDs")
    return {
        "rows": len(rows),
        "target_count_min": min(target_counts),
        "target_count_max": max(target_counts),
        "target_count_mean": sum(target_counts) / len(target_counts),
    }


def stable_digest(seed: int, namespace: str, *parts: str) -> bytes:
    payload = "\x1f".join((str(seed), namespace, *parts))
    return hashlib.sha256(payload.encode()).digest()


def category_to_parent(label_space: dict) -> dict[str, str]:
    return {
        category: parent
        for parent, children in label_space["category_tree"].items()
        for category in children
    }


def stable_order(
    values: Iterable[str],
    seed: int,
    namespace: str,
    sample_id: str,
) -> list[str]:
    return sorted(
        values,
        key=lambda value: stable_digest(seed, namespace, sample_id, value),
    )


def uniform_target_resample(rows: Iterable[dict], seed: int) -> list[dict]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot balance an empty training split")
    by_target: dict[str, list[dict]] = {}
    for row in rows:
        targets = list(dict.fromkeys(row["targets"]))
        if len(targets) != 1:
            raise ValueError("uniform target resampling requires single-target rows")
        by_target.setdefault(targets[0], []).append(row)
    labels = sorted(by_target)
    base_quota, remainder = divmod(len(rows), len(labels))
    balanced = []
    for label_index, label in enumerate(labels):
        quota = base_quota + int(label_index < remainder)
        pool = by_target[label]
        label_rng = random.Random(
            int.from_bytes(
                stable_digest(seed, "uniform-target-resample", label)[:8],
                "big",
            )
        )
        selected = []
        while len(selected) < quota:
            cycle = list(pool)
            label_rng.shuffle(cycle)
            selected.extend(cycle[: quota - len(selected)])
        balanced.extend(selected)
    random.Random(
        int.from_bytes(
            stable_digest(seed, "uniform-target-resample-order")[:8],
            "big",
        )
    ).shuffle(balanced)
    if len(balanced) != len(rows):
        raise AssertionError("uniform target resampling changed the training row count")
    return balanced


def build_candidates(
    row: dict,
    target: str,
    label_space: dict,
    seed: int = UNIFIED_EVAL_SEED,
) -> tuple[list[str], int, int]:
    parent_by_category = category_to_parent(label_space)
    excluded = set(row["full_seen"]) | set(row["targets"])
    excluded.remove(target)
    available = [
        category
        for category in label_space["prediction_labels"]
        if category != target and category not in excluded
    ]
    target_parent = parent_by_category[target]
    same_parent = stable_order(
        (
            category
            for category in available
            if parent_by_category[category] == target_parent
        ),
        seed,
        "same-parent-negative",
        row["sample_id"],
    )
    cross_parent = stable_order(
        (
            category
            for category in available
            if parent_by_category[category] != target_parent
        ),
        seed,
        "cross-parent-negative",
        row["sample_id"],
    )
    negatives = same_parent[:3]
    negatives.extend(cross_parent[: 9 - len(negatives)])
    if len(negatives) != 9:
        raise ValueError(
            f"{row['sample_id']}: only {len(available)} valid unseen negatives"
        )
    candidates = [target, *negatives]
    target_instance_id = f"{row['sample_id']}::{target}"
    random.Random(
        int.from_bytes(
            stable_digest(seed, "candidate-order", target_instance_id)[:8],
            "big",
        )
    ).shuffle(candidates)
    return candidates, min(3, len(same_parent)), len(available)


def expand_unified_eval(
    rows: Iterable[dict],
    label_space: dict,
    seed: int = UNIFIED_EVAL_SEED,
) -> list[dict]:
    expanded = []
    for row in rows:
        if "candidates" in row:
            if len(row["targets"]) != 1:
                raise ValueError("a frozen candidate row must have exactly one target")
            target = row["targets"][0]
            candidates = list(row["candidates"])
            if len(candidates) != 10 or len(set(candidates)) != 10:
                raise ValueError(f"{row['sample_id']}: frozen candidates must be 10 unique labels")
            if target not in candidates:
                raise ValueError(f"{row['sample_id']}: frozen candidates omit the target")
            expanded.append(
                {
                    **row,
                    "source_targets": [target],
                    "target_index": 0,
                    "gold_index": candidates.index(target),
                    "unified_sample_id": row["sample_id"],
                    "candidate_source": "frozen",
                }
            )
            continue
        for target_index, target in enumerate(dict.fromkeys(row["targets"])):
            candidates, same_parent_count, available_count = build_candidates(
                row,
                target,
                label_space,
                seed,
            )
            expanded.append(
                {
                    **row,
                    "source_targets": list(row["targets"]),
                    "targets": [target],
                    "target_index": target_index,
                    "candidates": candidates,
                    "gold_index": candidates.index(target),
                    "same_parent_negative_count": same_parent_count,
                    "available_negative_count": available_count,
                    "unified_sample_id": f"{row['sample_id']}::{target_index}",
                    "candidate_source": "deterministic_same_protocol",
                }
            )
    if len({row["unified_sample_id"] for row in expanded}) != len(expanded):
        raise ValueError("expanded unified evaluation contains duplicate IDs")
    return expanded


def render_prompt(
    row: dict,
    category_to_token: dict[str, str],
    model_key: str | None = None,
    training: bool = False,
) -> str:
    if model_key == "qwen2_1_5b":
        if training:
            if "official_instruction" not in row or "official_input" not in row:
                raise ValueError("official Qwen2 training requires released prompt fields")
            return (
                f"{QWEN2_OFFICIAL_PROMPT_PREFIX}{row['official_instruction']}\n\n"
                f"### Input:\n{row['official_input']}\n### Response:\n"
            )
        return "".join(category_to_token[category] for category in row["history"])
    history = [category_to_token[category] for category in row["history"]]
    numbered = "\n".join(
        f"{index}. {token}" for index, token in enumerate(history, 1)
    )
    return (
        "The user's positively rated interest categories, in chronological order, are:\n\n"
        f"{numbered}\n\n"
        "Predict the one future novel interest category most likely to occur next. "
        "It must not repeat a category from the history."
    )


def prompt_token_ids(tokenizer, prompt: str, model_key: str) -> list[int]:
    if model_key == "qwen2_1_5b":
        value = tokenizer(prompt)
        input_ids = value["input_ids"] if hasattr(value, "keys") else value
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return list(input_ids)
    options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
    }
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **options,
    )
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


class MeRADataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        model_key: str,
        label_space: dict,
        category_to_token: dict[str, str],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.label_space = label_space
        self.category_to_token = category_to_token
        self.max_length = max_length
        self.category_to_index = {
            category: index for index, category in enumerate(label_space["labels"])
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        prompt = render_prompt(
            row,
            self.category_to_token,
            model_key=self.model_key,
            training=True,
        )
        input_ids = prompt_token_ids(self.tokenizer, prompt, self.model_key)
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"{row['sample_id']}: {len(input_ids)} tokens exceed max_length="
                f"{self.max_length}; truncation is forbidden"
            )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": self.category_to_index[row["targets"][0]],
        }


class MeRACollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
