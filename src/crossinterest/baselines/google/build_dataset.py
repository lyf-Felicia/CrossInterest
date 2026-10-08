#!/usr/bin/env python3
"""Build the public Amazon adaptation of Google's interest-exploration data."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import sha256_file, write_json_atomic

from crossinterest.baselines.google.prompts import PROMPT_SCHEMA_VERSION, cluster_description, render_generation_prompt


HERE = _source_workspace('eval_baseline_google')
AMAZON_ROOT = HERE.parent
DEFAULT_SOURCE_TRAIN = (
    AMAZON_ROOT
    / "prep"
    / "artifacts"
    / "home_and_kitchen"
    / "data"
    / "unbalanced"
    / "train.jsonl"
)
DEFAULT_EVAL = AMAZON_ROOT / 'crossinterest.training' / "data" / "eval" / "dev_2k_interest.parquet"
DEFAULT_LABEL_SOURCE = AMAZON_ROOT / 'crossinterest.training' / "data" / "train.parquet"
DEFAULT_OUTPUT_DIR = HERE / "data"
EXPECTED_LABEL_COUNT = 112
PAIRS_PER_TARGET = 10
LEGACY_PROMPT_SCHEMA_VERSION = "google-interest-v1"
OUTPUT_FILENAMES = (
    "train_top10_pairs.parquet",
    "dev_2k_pair.parquet",
    "pair_grid.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--label-source", type=Path, default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--prompt-schema",
        choices=(LEGACY_PROMPT_SCHEMA_VERSION, PROMPT_SCHEMA_VERSION),
        default=PROMPT_SCHEMA_VERSION,
    )
    parser.add_argument(
        "--verify-frozen",
        type=Path,
        help="build in a temporary directory and compare with an existing data directory",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        help="development-only source row limit; disables formal count assertions",
    )
    return parser.parse_args()


def legacy_render_generation_prompt(cluster_pair: tuple[str, str] | list[str]) -> str:
    if len(cluster_pair) != 2:
        raise ValueError("the paper-faithful prompt requires exactly two clusters")
    first, second = (cluster_description(cluster) for cluster in cluster_pair)
    return (
        "The product categories I interacted with most recently are in the following "
        "clusters:\n"
        f"Cluster 1: {{{first}}}\n"
        f"Cluster 2: {{{second}}}.\n"
        "Each cluster is described with salient category or entity names. With less "
        "than 30 words, generate a new and different product category cluster I will "
        "interact with next with highly specific category or entity names, with a "
        'prefix that says "New cluster: "'
    )


def renderer(prompt_schema: str):
    if prompt_schema == LEGACY_PROMPT_SCHEMA_VERSION:
        return legacy_render_generation_prompt
    if prompt_schema == PROMPT_SCHEMA_VERSION:
        return render_generation_prompt
    raise ValueError(f"unsupported prompt schema: {prompt_schema}")


def recent_unique_pair(history: Iterable[str], support: set[str]) -> tuple[str, str] | None:
    selected_reversed: list[str] = []
    seen: set[str] = set()
    for cluster in reversed(list(history)):
        if cluster not in support or cluster in seen:
            continue
        selected_reversed.append(cluster)
        seen.add(cluster)
        if len(selected_reversed) == 2:
            return selected_reversed[1], selected_reversed[0]
    return None


def load_support(path: Path) -> list[str]:
    labels = sorted(set(pq.read_table(path, columns=["target_interest"])["target_interest"].to_pylist()))
    if len(labels) != EXPECTED_LABEL_COUNT:
        raise ValueError(f"expected {EXPECTED_LABEL_COUNT} labels, found {len(labels)}")
    return labels


def iter_jsonl(path: Path, max_rows: int | None = None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                return
            yield json.loads(line)


def scan_training_transitions(
    source: Path,
    support: set[str],
    max_rows: int | None,
) -> tuple[dict[str, Counter], dict[tuple[str, tuple[str, str]], str], dict]:
    pair_counts: dict[str, Counter] = defaultdict(Counter)
    representatives: dict[tuple[str, tuple[str, str]], str] = {}
    target_counts = Counter()
    scanned_rows = supported_target_rows = eligible_transition_rows = 0
    insufficient_pair_rows = 0

    for row in iter_jsonl(source, max_rows):
        scanned_rows += 1
        target = row["target_interest"]
        if target not in support:
            continue
        supported_target_rows += 1
        target_counts[target] += 1
        pair = recent_unique_pair(row["historical_interests"], support)
        if pair is None:
            insufficient_pair_rows += 1
            continue
        if target in row["full_seen_interests"]:
            raise ValueError(f"target novelty violation: {row['id']}")
        eligible_transition_rows += 1
        pair_counts[target][pair] += 1
        representatives.setdefault((target, pair), row["id"])

    unique_counts = [len(pair_counts[target]) for target in support]
    eligible_counts = [sum(pair_counts[target].values()) for target in support]
    stats = {
        "scanned_rows": scanned_rows,
        "supported_target_rows": supported_target_rows,
        "eligible_transition_rows": eligible_transition_rows,
        "insufficient_pair_rows": insufficient_pair_rows,
        "unique_target_pair_transitions": sum(unique_counts),
        "target_count_min": min(target_counts.values()),
        "target_count_max": max(target_counts.values()),
        "eligible_target_count_min": min(eligible_counts),
        "eligible_target_count_max": max(eligible_counts),
        "unique_pairs_per_target_min": min(unique_counts),
        "unique_pairs_per_target_max": max(unique_counts),
    }
    return pair_counts, representatives, stats


def build_train_rows(
    labels: list[str],
    pair_counts: dict[str, Counter],
    representatives: dict[tuple[str, tuple[str, str]], str],
    prompt_schema: str,
) -> tuple[list[dict], dict]:
    render_prompt = renderer(prompt_schema)
    rows = []
    for target in labels:
        ranked = sorted(pair_counts[target].items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) < PAIRS_PER_TARGET:
            raise ValueError(f"{target}: fewer than {PAIRS_PER_TARGET} eligible pairs")
        for rank, (pair, frequency) in enumerate(ranked[:PAIRS_PER_TARGET], 1):
            rows.append(
                {
                    "schema_version": 1,
                    "source_id": representatives[(target, pair)],
                    "cluster_pair": list(pair),
                    "target_cluster": target,
                    "pair_frequency": frequency,
                    "target_pair_rank": rank,
                    "prompt": render_prompt(pair),
                    "assistant": f"New cluster: {cluster_description(target)}",
                }
            )
    frequencies = [row["pair_frequency"] for row in rows]
    return rows, {
        "selected_pair_frequency_min": min(frequencies),
        "selected_pair_frequency_max": max(frequencies),
    }


def build_dev_rows(
    path: Path,
    support: set[str],
    prompt_schema: str,
) -> tuple[list[dict], dict]:
    render_prompt = renderer(prompt_schema)
    rows = []
    skipped = Counter()
    source_rows = pq.read_table(path).to_pylist()
    for source in source_rows:
        pair = recent_unique_pair(source["historical_interests"], support)
        if pair is None:
            skipped["fewer_than_two_supported_unique_clusters"] += 1
            continue
        target = source["target_interest"]
        if target in source["full_seen_interests"]:
            raise ValueError(f"target novelty violation: {source['source_sample_id']}")
        rows.append(
            {
                "schema_version": 1,
                "sample_id": f"{source['source_sample_id']}:google-pair",
                "source_sample_id": source["source_sample_id"],
                "cluster_pair": list(pair),
                "target_cluster": target,
                "full_seen_interests": source["full_seen_interests"],
                "candidate_interests": source["candidate_interests"],
                "gold_index": source["gold_index"],
                "prompt": render_prompt(pair),
            }
        )
    target_counts = Counter(row["target_cluster"] for row in rows)
    return rows, {
        "source_rows": len(source_rows),
        "eligible_rows": len(rows),
        "skipped": dict(skipped),
        "label_count": len(target_counts),
        "label_count_min": min(target_counts.values()),
        "label_count_max": max(target_counts.values()),
    }


def build_pair_grid(labels: list[str], prompt_schema: str) -> list[dict]:
    render_prompt = renderer(prompt_schema)
    return [
        {
            "schema_version": 1,
            "pair_id": f"pair-{first_index:03d}-{second_index:03d}",
            "cluster_pair": [first, second],
            "same_cluster_pair": first == second,
            "prompt": render_prompt((first, second)),
        }
        for first_index, first in enumerate(labels)
        for second_index, second in enumerate(labels)
    ]


def write_parquet(rows: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def semantic_digest(path: Path, ignored_columns: set[str] | None = None) -> str:
    ignored_columns = ignored_columns or set()
    digest = hashlib.sha256()
    for row in pq.read_table(path).to_pylist():
        for column in ignored_columns:
            row.pop(column, None)
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def column_mismatches(built: Path, frozen: Path) -> dict:
    built_rows = pq.read_table(built).to_pylist()
    frozen_rows = pq.read_table(frozen).to_pylist()
    if len(built_rows) != len(frozen_rows):
        return {"row_count": {"built": len(built_rows), "frozen": len(frozen_rows)}}
    columns = built_rows[0].keys() | frozen_rows[0].keys()
    result = {}
    for column in sorted(columns):
        mismatched = [
            index
            for index, (built_row, frozen_row) in enumerate(zip(built_rows, frozen_rows, strict=True))
            if built_row.get(column) != frozen_row.get(column)
        ]
        if mismatched:
            index = mismatched[0]
            result[column] = {
                "count": len(mismatched),
                "first_row": index,
                "built": built_rows[index].get(column),
                "frozen": frozen_rows[index].get(column),
            }
    return result


def compare_frozen(built_dir: Path, frozen_dir: Path) -> dict:
    comparisons = {}
    for name in OUTPUT_FILENAMES:
        built = built_dir / name
        frozen = frozen_dir / name
        comparisons[name] = {
            "byte_sha256_match": sha256_file(built) == sha256_file(frozen),
            "full_semantic_match": semantic_digest(built) == semantic_digest(frozen),
            "semantic_without_prompt_match": semantic_digest(built, {"prompt"})
            == semantic_digest(frozen, {"prompt"}),
            "semantic_without_prompt_or_source_id_match": semantic_digest(
                built, {"prompt", "source_id"}
            )
            == semantic_digest(frozen, {"prompt", "source_id"}),
            "column_mismatches": column_mismatches(built, frozen),
            "built_sha256": sha256_file(built),
            "frozen_sha256": sha256_file(frozen),
        }
    built_manifest = json.loads((built_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads((frozen_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    for manifest in (built_manifest, frozen_manifest):
        for output in manifest["outputs"].values():
            output.pop("path", None)
    comparisons["MANIFEST.json"] = {
        "stable_fields_match": built_manifest == frozen_manifest,
    }
    return comparisons


def frozen_rebuild_matches(comparison: dict) -> bool:
    return comparison["MANIFEST.json"]["stable_fields_match"] and all(
        comparison[name]["byte_sha256_match"] for name in OUTPUT_FILENAMES
    )


def build(args: argparse.Namespace, output_dir: Path) -> dict:
    source_train = args.source_train.resolve()
    eval_path = args.eval.resolve()
    label_source = args.label_source.resolve()
    for path in (source_train, eval_path, label_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = load_support(label_source)
    support = set(labels)
    pair_counts, representatives, train_stats = scan_training_transitions(
        source_train, support, args.max_source_rows
    )
    train_rows, selected_stats = build_train_rows(
        labels, pair_counts, representatives, args.prompt_schema
    )
    train_stats.update(selected_stats)
    dev_rows, dev_stats = build_dev_rows(eval_path, support, args.prompt_schema)
    pair_grid = build_pair_grid(labels, args.prompt_schema)

    if args.max_source_rows is None:
        if len(train_rows) != EXPECTED_LABEL_COUNT * PAIRS_PER_TARGET:
            raise ValueError("formal Train row count mismatch")
        if len(dev_rows) != 1997:
            raise ValueError("formal Dev row count mismatch")
        if len(pair_grid) != EXPECTED_LABEL_COUNT**2:
            raise ValueError("formal pair-grid row count mismatch")

    outputs = {
        "train": write_parquet(train_rows, output_dir / "train_top10_pairs.parquet"),
        "dev": write_parquet(dev_rows, output_dir / "dev_2k_pair.parquet"),
        "pair_grid": write_parquet(pair_grid, output_dir / "pair_grid.parquet"),
    }
    manifest = {
        "schema_version": 2,
        "task": "google_llm_interest_exploration_public_amazon_reproduction",
        "reproduction_scope": "public-data mechanism reproduction; not an industrial full reproduction",
        "paper": {
            "title": "LLMs for User Interest Exploration in Large-scale Recommendation Systems",
            "arxiv": "2405.16363v2",
            "model": "Gemini-Pro",
            "official_code_or_data_found": False,
        },
        "seed": args.seed,
        "prompt_schema_version": args.prompt_schema,
        "cluster_contract": {
            "space": "112 frozen Amazon L2 > L3 taxonomy labels",
            "description": "two taxonomy entities (L2, L3) as public metadata substitute for paper keyword sets",
            "history_policy": "two most recent distinct supported clusters, chronological order",
            "target_novelty": "target absent from the complete eligible pre-target history",
            "pair_order": "ordered; oldest to newest among the two selected clusters",
        },
        "training_curation": {
            "policy": "top-frequency ordered cluster pairs independently within each target label",
            "pairs_per_label": PAIRS_PER_TARGET,
            "label_count": len(labels),
            "rows": len(train_rows),
            "balanced_by_target": True,
            "one_representative_per_target_pair": True,
            "representative_policy": "first source row in immutable JSONL order",
            "tie_break": "pair frequency descending, then ordered pair lexicographic ascending",
            "source": str(source_train),
            "source_sha256": sha256_file(source_train),
            "stats": train_stats,
        },
        "evaluation": {
            "paper_pair_dev": {
                "source": str(eval_path),
                "source_sha256": sha256_file(eval_path),
                **dev_stats,
            },
            "exhaustive_ordered_pair_grid": {
                "rows": len(pair_grid),
                "expected_rows": len(labels) ** 2,
                "includes_same_cluster_pairs": True,
            },
            "amazon_aligned": {
                "source_directory": str(eval_path.parent),
                "protocols": [
                    "frozen 10-way constrained Macro Accuracy",
                    "candidate-free Macro Recall/nDCG@10/20",
                ],
            },
        },
        "outputs": outputs,
    }
    write_json_atomic(output_dir / "MANIFEST.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.verify_frozen:
        with tempfile.TemporaryDirectory(prefix="google-baseline-rebuild-") as temporary:
            manifest = build(args, Path(temporary))
            result = compare_frozen(Path(temporary), args.verify_frozen.resolve())
            print(json.dumps({"manifest": manifest, "comparison": result}, ensure_ascii=False, indent=2))
            if not frozen_rebuild_matches(result):
                raise SystemExit("frozen byte-level reconstruction or manifest mismatch")
        return
    manifest = build(args, args.output_dir.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()