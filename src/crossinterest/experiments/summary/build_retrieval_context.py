#!/usr/bin/env python3
"""Build target-blind retrieval evidence from labeled Amazon Train histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--exemplars", type=int, default=8)
    parser.add_argument("--priors", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n-features", type=int, default=2**18)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()


def select_balanced(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_interest"]].append(row)
    selected = []
    for label in sorted(grouped):
        ordered = sorted(
            grouped[label],
            key=lambda row: stable_key(seed, row["sample_id"]),
        )
        if len(ordered) < per_class:
            raise ValueError(f"{label}: insufficient rows for balanced quota")
        selected.extend(ordered[:per_class])
    return sorted(selected, key=lambda row: row["sample_id"])


def feature_token(value: str) -> str:
    return value.lower().replace(" ", "_")


def render_features(row: dict) -> str:
    lexical = " ".join(f"title_{title}" for title in row["historical_titles"])
    lexical += " view_break " + " ".join(
        f"interest_{interest.replace(' ', '_')}"
        for interest in row["historical_interests"]
    )
    interests = row["historical_interests"]
    parents = [interest.split(" > ", 1)[0] for interest in interests]
    temporal = []
    parent_counts = Counter(parents)
    for parent, count in parent_counts.items():
        temporal.extend([f"parent={feature_token(parent)}"] * min(count, 4))
    for reverse_position, (interest, parent) in enumerate(
        zip(reversed(interests), reversed(parents), strict=True),
        1,
    ):
        if reverse_position <= 5:
            temporal.append(f"r{reverse_position}_parent={feature_token(parent)}")
            temporal.append(f"r{reverse_position}_interest={feature_token(interest)}")
        temporal.append(f"interest={feature_token(interest)}")
    if len(parents) >= 2:
        temporal.append(
            f"last_transition={feature_token(parents[-2])}__{feature_token(parents[-1])}"
        )
    return lexical + " temporal_break " + " ".join(temporal)


def main() -> None:
    args = parse_args()
    if not 0 < args.exemplars <= args.neighbors:
        raise ValueError("--exemplars must be between 1 and --neighbors")
    if not 0 < args.priors <= 112:
        raise ValueError("--priors must be between 1 and 112")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    mapping = json.loads(args.label_map.read_text())
    category_to_id = mapping["category_to_id"]
    train = pq.read_table(
        args.train,
        columns=[
            "id",
            "historical_titles",
            "historical_interests",
            "target_interest",
        ],
        memory_map=True,
    ).to_pylist()
    query_rows = pq.read_table(
        args.queries,
        columns=[
            "sample_id",
            "historical_titles",
            "historical_interests",
            "full_seen_interests",
            "target_interest",
        ],
        memory_map=True,
    ).to_pylist()
    selected = select_balanced(query_rows, args.per_class, args.seed)
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    train_features = vectorizer.transform(render_features(row) for row in train)
    query_features = vectorizer.transform(render_features(row) for row in selected)
    index = NearestNeighbors(
        n_neighbors=args.neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(train_features)
    distances, indices = index.kneighbors(query_features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for query, query_distances, query_indices in zip(
            selected,
            distances,
            indices,
            strict=True,
        ):
            seen = set(query["full_seen_interests"])
            support: dict[str, float] = defaultdict(float)
            exemplars = []
            exemplar_labels = set()
            for distance, index_value in zip(query_distances, query_indices, strict=True):
                neighbor = train[int(index_value)]
                target = neighbor["target_interest"]
                if target in seen:
                    continue
                similarity = 1.0 - float(distance)
                support[target] += float(np.exp(-distance / args.temperature))
                if len(exemplars) < args.exemplars and target not in exemplar_labels:
                    exemplar_labels.add(target)
                    exemplars.append(
                        {
                            "similarity": similarity,
                            "recent_observed_categories": neighbor["historical_interests"][-5:],
                            "recent_product_titles": neighbor["historical_titles"][-5:],
                            "next_interest_id": category_to_id[target],
                            "next_category": target,
                        }
                    )
            priors = [
                {
                    "interest_id": category_to_id[category],
                    "category": category,
                    "retrieval_support": score,
                }
                for category, score in sorted(
                    support.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: args.priors]
            ]
            record = {
                "schema_version": 1,
                "sample_id": query["sample_id"],
                "retrieval_target_blind": True,
                "query_target_used_for_retrieval": False,
                "exemplars": exemplars,
                "empirical_prior": priors,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "train": str(args.train.resolve()),
        "train_sha256": sha256_file(args.train),
        "queries": str(args.queries.resolve()),
        "queries_sha256": sha256_file(args.queries),
        "label_map": str(args.label_map.resolve()),
        "label_map_sha256": sha256_file(args.label_map),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": len(selected),
        "class_count": len({row["target_interest"] for row in selected}),
        "target_blind": True,
        "query_target_used_for_retrieval": False,
        "feature_view": "behavior_interest_temporal_combined",
        "neighbors": args.neighbors,
        "exemplars": args.exemplars,
        "priors": args.priors,
        "temperature": args.temperature,
        "n_features": args.n_features,
        "seed": args.seed,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()