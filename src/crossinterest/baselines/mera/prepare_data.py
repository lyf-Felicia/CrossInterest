#!/usr/bin/env python3
"""Build auditable label and Train-only relation assets for MeRA."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from crossinterest.baselines.mera.core import build_cooccurrence_matrix, build_label_space, sibling_mask, target_frequency_summary, threshold_cooccurrence, validate_label_space, write_json
from crossinterest.baselines.mera.data import expand_unified_eval, load_home_rows, load_official_rows, validate_rows


HERE = _source_workspace('eval_baseline_mera')
HOME_TRAIN = HERE.parent / 'crossinterest.training' / "data" / "train.parquet"
HOME_EVAL = (
    HERE.parent
    / 'crossinterest.training'
    / "data"
    / "eval"
    / "dev_2k_interest.parquet"
)
OFFICIAL_NAMES = {
    "arts": "Arts",
    "grocery": "Grocery_and_Gourmet_Food",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("arts", "grocery", "home"), required=True)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--home-train", type=Path, default=HOME_TRAIN)
    parser.add_argument("--home-eval", type=Path, default=HOME_EVAL)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--cooccurrence-threshold", type=float, default=0.05)
    return parser.parse_args()


def official_label_space(root: Path, dataset: str) -> tuple[dict, Path, Path]:
    data_root = root / "data" if (root / "data").is_dir() else root
    source = data_root / OFFICIAL_NAMES[dataset]
    map_path = source / "layer2_map.txt"
    labels_by_id = []
    for line in map_path.read_text(encoding="utf-8").splitlines():
        category, index = line.rsplit("\t", 1)
        labels_by_id.append((int(index), category))
    labels = [category for _, category in sorted(labels_by_id)]
    tree_paths = list(source.glob("category_tree*.json"))
    if len(tree_paths) != 1:
        raise ValueError(f"expected one taxonomy tree under {source}")
    tree_path = tree_paths[0]
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    return (
        build_label_space(labels, category_tree=tree, preserve_order=True),
        source,
        tree_path,
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve() / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "home":
        train_path = args.home_train.resolve()
        eval_path = args.home_eval.resolve()
        train_rows = load_home_rows(train_path, training=True)
        eval_rows = load_home_rows(eval_path, training=False)
        prediction_labels = sorted(
            set(
                pq.read_table(train_path, columns=["target_interest"])[
                    "target_interest"
                ].to_pylist()
            )
        )
        labels = sorted(
            {
                category
                for row in train_rows
                for category in (*row["history"], *row["targets"])
            }
        )
        label_space = build_label_space(
            labels,
            prediction_categories=prediction_labels,
            semantic_texts={
                category: category.rpartition(" > ")[2]
                for category in labels
            },
        )
        validation = {
            "train": validate_rows(
                train_rows,
                set(labels),
                single_target=True,
                prediction_labels=set(prediction_labels),
            ),
            "eval": validate_rows(
                eval_rows,
                set(labels),
                single_target=True,
                prediction_labels=set(prediction_labels),
            ),
        }
        sources = {"train": train_path, "eval": eval_path}
        unified_eval = expand_unified_eval(eval_rows, label_space)
        distribution_rows = {"train": train_rows, "eval": eval_rows}
    else:
        if args.official_root is None:
            raise ValueError("--official-root is required for Arts/Grocery")
        label_space, source, tree_path = official_label_space(
            args.official_root.resolve(),
            args.dataset,
        )
        labels = label_space["labels"]
        train_path = source / "train.json"
        valid_path = source / "valid.json"
        test_path = source / "test.json"
        train_rows = load_official_rows(train_path)
        valid_rows = load_official_rows(valid_path)
        test_rows = load_official_rows(test_path)
        validation = {
            "train": validate_rows(train_rows, set(labels), single_target=True),
            "valid": validate_rows(valid_rows, set(labels), single_target=True),
            "test": validate_rows(test_rows, set(labels), single_target=False),
        }
        sources = {
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
            "taxonomy": tree_path,
        }
        unified_eval = expand_unified_eval(test_rows, label_space)
        distribution_rows = {
            "train": train_rows,
            "valid": valid_rows,
            "test": test_rows,
        }
    validate_label_space(label_space)
    counts = build_cooccurrence_matrix(
        (set(row["full_seen"]) | set(row["targets"]) for row in train_rows),
        label_space["labels"],
    )
    collaborative = threshold_cooccurrence(counts, args.cooccurrence_threshold)
    structure = sibling_mask(label_space)
    np.save(output / "cooccurrence_counts.npy", counts)
    np.save(output / "collaborative_weights.npy", collaborative)
    np.save(output / "structure_mask.npy", structure)
    write_json(output / "label_space.json", label_space)
    target_distribution = {
        split: target_frequency_summary(rows, label_space["prediction_labels"])
        for split, rows in distribution_rows.items()
    }
    write_json(output / "target_distribution.json", target_distribution)
    train_distribution = target_distribution["train"]
    train_is_exactly_balanced = (
        train_distribution["zero_count_label_count"] == 0
        and train_distribution["min_nonzero_count"]
        == train_distribution["max_count"]
    )
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "label_count": len(label_space["labels"]),
        "prediction_label_count": len(label_space["prediction_labels"]),
        "label_space_fingerprint": label_space["fingerprint"],
        "cooccurrence_source": "train-only unique history categories plus target",
        "cooccurrence_threshold": args.cooccurrence_threshold,
        "collaborative_directed_edge_count": int(np.count_nonzero(collaborative)),
        "structure_directed_edge_count": int(np.count_nonzero(structure)),
        "target_balance": {
            "training_row_policy": (
                "frozen equal per-label quota"
                if args.dataset == "home"
                else "natural frequency in the official released Train split"
            ),
            "resampling_applied_by_asset_builder": False,
            "class_weighting_applied_by_trainer": False,
            "train_is_exactly_class_balanced": train_is_exactly_balanced,
            "distribution_file": "target_distribution.json",
        },
        "validation": validation,
        "unified_evaluation": {
            "source_row_count": len(eval_rows if args.dataset == "home" else test_rows),
            "gold_evaluation_count": len(unified_eval),
            "primary_aggregation": "equal mean over source samples",
            "multi_gold_recall": "set recall",
            "multi_gold_ndcg": "binary-relevance DCG normalized by ideal DCG",
            "ten_way_accuracy": "mean over golds within source, then mean over sources",
            "random_gold_selection": False,
            "candidate_count": 10,
            "same_parent_negative_limit": 3,
            "other_gold_labels_excluded_from_negatives": True,
            "history_labels_excluded_from_negatives": True,
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
