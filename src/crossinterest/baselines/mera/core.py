"""Model-independent MeRA label, relation, loss, and metric utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as functional


def category_token(index: int) -> str:
    return f"<category_{index:03d}>"


def build_label_space(
    categories: Iterable[str],
    category_tree: dict[str, list[str]] | None = None,
    preserve_order: bool = False,
    prediction_categories: Iterable[str] | None = None,
    semantic_texts: dict[str, str] | None = None,
) -> dict:
    source_labels = list(categories)
    labels = list(dict.fromkeys(source_labels)) if preserve_order else sorted(set(source_labels))
    if not labels:
        raise ValueError("label space cannot be empty")
    prediction_set = (
        set(labels) if prediction_categories is None else set(prediction_categories)
    )
    prediction_labels = [label for label in labels if label in prediction_set]
    if not prediction_labels or not set(prediction_labels) <= set(labels):
        raise ValueError("prediction labels must be a non-empty subset of all labels")
    category_to_token = {
        category: category_token(index) for index, category in enumerate(labels)
    }
    category_to_semantic_text = (
        dict(semantic_texts)
        if semantic_texts is not None
        else {category: category for category in labels}
    )
    if set(category_to_semantic_text) != set(labels) or any(
        not text.strip() for text in category_to_semantic_text.values()
    ):
        raise ValueError("semantic texts must cover the label space with non-empty text")
    if category_tree is None:
        parents = {}
        for category in labels:
            parent, separator, child = category.partition(" > ")
            if not separator or not parent or not child:
                raise ValueError(f"expected 'parent > child' category: {category}")
            parents.setdefault(parent, []).append(category)
    else:
        parents = {parent: list(children) for parent, children in category_tree.items()}
        children = [category for group in parents.values() for category in group]
        if len(children) != len(set(children)):
            raise ValueError("a category appears more than once in the category tree")
        if set(children) != set(labels):
            raise ValueError("category tree must cover the label space exactly")
    identity = json.dumps(
        {
            "category_to_token": category_to_token,
            "category_to_semantic_text": category_to_semantic_text,
            "prediction_labels": prediction_labels,
            "category_tree": dict(sorted(parents.items())),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "label_count": len(labels),
        "labels": labels,
        "prediction_label_count": len(prediction_labels),
        "prediction_labels": prediction_labels,
        "category_to_token": category_to_token,
        "category_to_semantic_text": category_to_semantic_text,
        "token_to_category": {
            token: category for category, token in category_to_token.items()
        },
        "category_tree": dict(sorted(parents.items())),
        "fingerprint": hashlib.sha256(identity.encode()).hexdigest(),
    }


def validate_label_space(label_space: dict, expected_count: int | None = None) -> dict:
    rebuilt = build_label_space(
        label_space["labels"],
        category_tree=label_space["category_tree"],
        preserve_order=True,
        prediction_categories=label_space["prediction_labels"],
        semantic_texts=label_space["category_to_semantic_text"],
    )
    for field in (
        "schema_version",
        "label_count",
        "labels",
        "prediction_label_count",
        "prediction_labels",
        "category_to_token",
        "category_to_semantic_text",
        "token_to_category",
        "category_tree",
        "fingerprint",
    ):
        if label_space.get(field) != rebuilt[field]:
            raise ValueError(f"invalid label-space field: {field}")
    if expected_count is not None and label_space["label_count"] != expected_count:
        raise ValueError(
            f"expected {expected_count} labels, found {label_space['label_count']}"
        )
    return label_space


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def target_frequency_summary(rows: Iterable[dict], labels: Sequence[str]) -> dict:
    label_set = set(labels)
    if not labels or len(label_set) != len(labels):
        raise ValueError("frequency labels must be non-empty and unique")
    counts = Counter(
        target
        for row in rows
        for target in dict.fromkeys(row["targets"])
    )
    unknown = set(counts) - label_set
    if unknown:
        raise ValueError(f"frequency rows contain unknown targets: {sorted(unknown)[:3]}")
    all_counts = [counts[label] for label in labels]
    nonzero = sorted(value for value in all_counts if value)
    total = sum(all_counts)
    if not total:
        raise ValueError("frequency rows contain no targets")
    mean = total / len(labels)
    variance = sum((value - mean) ** 2 for value in all_counts) / len(labels)
    probabilities = [value / total for value in nonzero]
    normalized_entropy = (
        -sum(probability * math.log(probability) for probability in probabilities)
        / math.log(len(labels))
        if len(labels) > 1
        else 1.0
    )
    middle = len(nonzero) // 2
    median_nonzero = (
        nonzero[middle]
        if len(nonzero) % 2
        else (nonzero[middle - 1] + nonzero[middle]) / 2
    )
    return {
        "gold_count": total,
        "label_count": len(labels),
        "observed_label_count": len(nonzero),
        "zero_count_label_count": len(labels) - len(nonzero),
        "min_nonzero_count": nonzero[0],
        "median_nonzero_count": median_nonzero,
        "max_count": nonzero[-1],
        "max_to_min_nonzero_ratio": nonzero[-1] / nonzero[0],
        "coefficient_of_variation_all_labels": math.sqrt(variance) / mean,
        "normalized_entropy_all_labels": normalized_entropy,
        "count_by_label": {label: counts[label] for label in labels},
    }


def build_cooccurrence_matrix(
    category_sets: Iterable[Iterable[str]],
    labels: Sequence[str],
) -> np.ndarray:
    category_to_index = {category: index for index, category in enumerate(labels)}
    if len(category_to_index) != len(labels):
        raise ValueError("labels must be unique")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for categories in category_sets:
        unique = set(categories)
        unknown = unique - set(category_to_index)
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)[:3]}")
        indices = [category_to_index[category] for category in unique]
        matrix[np.ix_(indices, indices)] += 1
    return matrix


def threshold_cooccurrence(
    counts: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    """Return directed conditional co-occurrence weights P(j present | i present)."""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("co-occurrence matrix must be square")
    diagonal = np.diag(counts).astype(np.float64)
    probabilities = np.divide(
        counts,
        diagonal[:, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=diagonal[:, None] > 0,
    )
    np.fill_diagonal(probabilities, 0.0)
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError("co-occurrence threshold must be in [0, 1]")
        probabilities[probabilities < threshold] = 0.0
    return probabilities


def sibling_mask(label_space: dict) -> np.ndarray:
    validate_label_space(label_space)
    labels = label_space["labels"]
    indices = {category: index for index, category in enumerate(labels)}
    mask = np.zeros((len(labels), len(labels)), dtype=bool)
    for children in label_space["category_tree"].values():
        child_indices = [indices[category] for category in children]
        mask[np.ix_(child_indices, child_indices)] = True
    np.fill_diagonal(mask, False)
    return mask


def sample_relation_rows(
    positive_weights: torch.Tensor,
    negative_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample [anchor, positive, negatives...] for every valid anchor."""
    if positive_weights.ndim != 2 or positive_weights.shape[0] != positive_weights.shape[1]:
        raise ValueError("positive weights must be square")
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    size = positive_weights.shape[0]
    rows = []
    for anchor in range(size):
        positive = positive_weights[anchor].clone().float().cpu()
        positive[anchor] = 0
        positive_candidates = torch.nonzero(
            positive > 0,
            as_tuple=False,
        ).flatten()
        negative_candidates = torch.nonzero(
            positive == 0,
            as_tuple=False,
        ).flatten()
        negative_candidates = negative_candidates[negative_candidates != anchor]
        if not len(positive_candidates):
            continue
        if len(negative_candidates) < negative_count:
            raise ValueError(
                f"anchor {anchor} has {len(negative_candidates)} negatives, "
                f"need {negative_count}"
            )
        positive_index = torch.multinomial(
            positive,
            1,
            replacement=False,
            generator=generator,
        )
        negative_order = torch.randperm(
            len(negative_candidates),
            generator=generator,
        )[:negative_count]
        negatives = negative_candidates[negative_order]
        rows.append(torch.cat((torch.tensor([anchor]), positive_index, negatives)))
    if not rows:
        raise ValueError("relation graph has no valid positive pairs")
    return torch.stack(rows)


def relation_infonce(
    representations: torch.Tensor,
    relation_rows: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    selected = representations[relation_rows]
    anchors = functional.normalize(selected[:, 0], dim=-1)
    candidates = functional.normalize(selected[:, 1:], dim=-1)
    logits = torch.einsum("bd,bkd->bk", anchors, candidates) / temperature
    return functional.cross_entropy(
        logits,
        torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device),
    )


def paper_metrics(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
    cutoffs: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have the same non-zero length")
    metrics = {}
    for cutoff in cutoffs:
        precision = []
        recall = []
        for predicted, target in zip(predictions, targets, strict=True):
            target_set = set(target)
            if not target_set:
                raise ValueError("paper metric target set cannot be empty")
            hits = len(set(predicted[:cutoff]) & target_set)
            precision.append(hits / cutoff)
            recall.append(hits / len(target_set))
        metrics[f"precision_at_{cutoff}"] = sum(precision) / len(precision)
        metrics[f"recall_at_{cutoff}"] = sum(recall) / len(recall)
    return metrics


def set_ranking_score(
    ranking: Sequence[str],
    targets: Sequence[str],
    cutoffs: Sequence[int] = (10, 20),
) -> dict[str, float]:
    """Score one query with binary relevance over all known relevant labels."""
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking contains duplicate labels")
    target_set = set(targets)
    if not target_set:
        raise ValueError("target set cannot be empty")
    missing = target_set - set(ranking)
    if missing:
        raise ValueError(f"ranking omits target labels: {sorted(missing)[:3]}")
    scores = {}
    for cutoff in cutoffs:
        if cutoff < 1:
            raise ValueError("metric cutoffs must be positive")
        hits = [
            rank
            for rank, label in enumerate(ranking[:cutoff], 1)
            if label in target_set
        ]
        scores[f"recall_at_{cutoff}"] = len(hits) / len(target_set)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_count = min(len(target_set), cutoff)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        scores[f"ndcg_at_{cutoff}"] = dcg / ideal_dcg
    return scores


def query_set_metrics(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
    cutoffs: Sequence[int] = (10, 20),
) -> dict[str, float | int]:
    """Average set-aware ranking scores with one equal vote per source query."""
    if len(rankings) != len(targets) or not rankings:
        raise ValueError("rankings and targets must have the same non-zero length")
    per_query = [
        set_ranking_score(ranking, target, cutoffs)
        for ranking, target in zip(rankings, targets, strict=True)
    ]
    metric_names = per_query[0]
    unique_target_counts = [len(set(target)) for target in targets]
    return {
        **{
            f"mean_{name}": sum(row[name] for row in per_query) / len(per_query)
            for name in metric_names
        },
        "source_sample_count": len(per_query),
        "target_count_min": min(unique_target_counts),
        "target_count_max": max(unique_target_counts),
        "target_count_mean": sum(unique_target_counts) / len(unique_target_counts),
    }


def home_metrics(
    rankings: Sequence[Sequence[str]],
    targets: Sequence[str],
    candidate_sets: Sequence[Sequence[str]] | None = None,
) -> dict:
    if len(rankings) != len(targets) or not rankings:
        raise ValueError("rankings and targets must have the same non-zero length")
    if candidate_sets is not None and len(candidate_sets) != len(targets):
        raise ValueError("candidate sets and targets must have the same length")
    per_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (ranking, target) in enumerate(zip(rankings, targets, strict=True)):
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"ranking {index} contains duplicate labels")
        try:
            rank = ranking.index(target) + 1
        except ValueError:
            rank = None
        values = {}
        for cutoff in (10, 20):
            hit = rank is not None and rank <= cutoff
            values[f"recall_at_{cutoff}"] = float(hit)
            values[f"ndcg_at_{cutoff}"] = (
                1.0 / math.log2(rank + 1) if hit else 0.0
            )
        if candidate_sets is not None:
            candidates = candidate_sets[index]
            if target not in candidates:
                raise ValueError(f"candidate set {index} omits its target")
            candidate_ranking = [label for label in ranking if label in set(candidates)]
            values["accuracy_10way"] = float(
                bool(candidate_ranking) and candidate_ranking[0] == target
            )
        for name, value in values.items():
            per_class[target][name].append(value)
    class_metrics = {
        label: {
            name: sum(values) / len(values)
            for name, values in sorted(group.items())
        }
        for label, group in sorted(per_class.items())
    }
    metric_names = next(iter(class_metrics.values()))
    return {
        **{
            f"macro_{name}": sum(row[name] for row in class_metrics.values())
            / len(class_metrics)
            for name in metric_names
        },
        "sample_count": len(targets),
        "class_count": len(class_metrics),
        "per_class_metrics": class_metrics,
    }
