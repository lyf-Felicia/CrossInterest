#!/usr/bin/env python3
"""Materialize matched plain/rationale balanced canaries from transition profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--plain-output", type=Path, required=True)
    parser.add_argument("--rationale-output", type=Path, required=True)
    parser.add_argument("--reasoning-output", type=Path)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()


def load_profiles(path: Path) -> dict[str, dict]:
    profiles = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            profiles[row["target_interest"]] = row
    invalid = [target for target, row in profiles.items() if not row.get("valid")]
    if invalid:
        raise ValueError(f"latest profiles contain {len(invalid)} invalid targets")
    return profiles


def matched_rationale(row: dict, profile_record: dict) -> tuple[float, str] | None:
    interests = row["historical_interests"]
    positions_by_category: dict[str, list[int]] = defaultdict(list)
    for position, category in enumerate(interests, 1):
        positions_by_category[category].append(position)
    matches = []
    for cue in profile_record["profile"]["transition_cues"]:
        categories = [
            category
            for category in cue["source_categories"]
            if category in positions_by_category
        ]
        if not categories:
            continue
        latest_position = max(positions_by_category[category][-1] for category in categories)
        recency = latest_position / len(interests)
        coverage = len(categories) / len(cue["source_categories"])
        matches.append((recency + 0.25 * coverage, latest_position, cue, categories))
    if not matches:
        return None
    score, _, cue, categories = max(
        matches,
        key=lambda value: (value[0], value[1], value[2]["explanation"]),
    )
    evidence = "; ".join(
        f"{category} [items {','.join(map(str, positions_by_category[category]))}]"
        for category in categories
    )
    rationale = (
        f"Observed evidence: {evidence}. "
        f"Transition mechanism ({cue['relation_type']}): {cue['explanation']} "
        f"Category role: {profile_record['profile']['category_profile']}"
    )
    return score, rationale


def main() -> None:
    args = parse_args()
    source = pq.read_table(args.train, memory_map=True)
    profiles = load_profiles(args.profiles)
    rows = source.to_pylist()
    candidates: dict[str, list[tuple[float, bytes, int, str]]] = defaultdict(list)
    unmatched = defaultdict(int)
    for index, row in enumerate(rows):
        target = row["target_interest"]
        if target not in profiles:
            raise ValueError(f"missing transition profile for {target}")
        match = matched_rationale(row, profiles[target])
        if match is None:
            unmatched[target] += 1
            continue
        score, rationale = match
        candidates[target].append(
            (-score, stable_key(args.seed, row["id"]), index, rationale)
        )
    if len(candidates) != 112:
        raise ValueError(f"expected 112 matched target classes, found {len(candidates)}")
    selected = []
    rationales = []
    for target in sorted(candidates):
        ordered = sorted(candidates[target])
        if len(ordered) < args.per_class:
            raise ValueError(
                f"{target}: only {len(ordered)} cue-matched rows for quota {args.per_class}"
            )
        for _, _, index, rationale in ordered[: args.per_class]:
            selected.append(index)
            rationales.append(rationale)
    order = sorted(range(len(selected)), key=lambda offset: rows[selected[offset]]["id"])
    selected = [selected[offset] for offset in order]
    rationales = [rationales[offset] for offset in order]
    plain = source.take(pa.array(selected, type=pa.int64()))
    rationale = plain.append_column("target_rationale", pa.array(rationales, pa.string()))
    reasoning = plain.append_column("target_reasoning", pa.array(rationales, pa.string()))
    args.plain_output.parent.mkdir(parents=True, exist_ok=True)
    args.rationale_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(plain, args.plain_output, compression="zstd")
    pq.write_table(rationale, args.rationale_output, compression="zstd")
    if args.reasoning_output is not None:
        args.reasoning_output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(reasoning, args.reasoning_output, compression="zstd")
    restored_plain = pq.read_table(args.plain_output, memory_map=True)
    restored_rationale = pq.read_table(args.rationale_output, memory_map=True)
    if not restored_rationale.select(restored_plain.schema.names).equals(restored_plain):
        raise AssertionError("rationale output changed a source column")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "train": str(args.train.resolve()),
        "train_sha256": sha256_file(args.train),
        "profiles": str(args.profiles.resolve()),
        "profiles_sha256": sha256_file(args.profiles),
        "plain_output": str(args.plain_output.resolve()),
        "plain_output_sha256": sha256_file(args.plain_output),
        "rationale_output": str(args.rationale_output.resolve()),
        "rationale_output_sha256": sha256_file(args.rationale_output),
        "reasoning_output": (
            str(args.reasoning_output.resolve()) if args.reasoning_output else None
        ),
        "reasoning_output_sha256": (
            sha256_file(args.reasoning_output) if args.reasoning_output else None
        ),
        "rows": len(selected),
        "class_count": len(candidates),
        "per_class": args.per_class,
        "original_columns_unchanged": True,
        "selection": "highest cue match recency then coverage then stable hash",
        "unmatched_rows": sum(unmatched.values()),
        "minimum_matched_rows_per_class": min(map(len, candidates.values())),
        "mean_rationale_characters": sum(map(len, rationales)) / len(rationales),
        "seed": args.seed,
    }
    manifest_path = args.rationale_output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()