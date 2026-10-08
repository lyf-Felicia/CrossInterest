#!/usr/bin/env python3
"""Plot taxonomy, positive-target, and complete-history distributions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from crossinterest.datasets.amazon.scripts.build import ALLOWED_LEVEL_2, ROOT_CATEGORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=("all", "categories", "history"),
        default="all",
        help="Limit work when only one analysis family needs to be refreshed.",
    )
    return parser.parse_args()


HISTORY_BINS = (
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, 50, "50"),
    (51, 99, "51-99"),
    (100, 199, "100-199"),
    (200, 499, "200-499"),
    (500, 999, "500-999"),
    (1_000, None, "1000+"),
)


def metadata_level_2_counts(path: Path) -> tuple[Counter[str], Counter[str]]:
    allowed: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            categories = [str(value).strip() for value in row.get("categories") or []]
            categories = [value for value in categories if value]
            if len(categories) < 2 or categories[0] != ROOT_CATEGORY:
                continue
            level_2 = categories[1]
            destination = allowed if level_2 in ALLOWED_LEVEL_2 else excluded
            destination[level_2] += 1
    return allowed, excluded


def write_metadata_counts(
    path: Path, allowed: Counter[str], excluded: Counter[str]
) -> None:
    rows = []
    for prefix, group, counts in (
        ("W", "whitelist", allowed),
        ("X", "non_whitelist", excluded),
    ):
        for rank, (category, count) in enumerate(counts.most_common(), start=1):
            rows.append((f"{prefix}{rank:03d}", group, rank, category, count))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "group", "rank_in_group", "level_2", "metadata_rows"])
        writer.writerows(rows)


def plot_metadata_counts(
    path: Path, allowed: Counter[str], excluded: Counter[str]
) -> None:
    formatter = FuncFormatter(
        lambda value, _: (
            f"{value / 1_000_000:g}M"
            if value >= 1_000_000
            else f"{value / 1_000:g}k"
            if value >= 1_000
            else f"{value:g}"
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, prefix, title, color, counts in (
        (axes[0], "W", "Whitelist L2", "#1976A3", allowed),
        (axes[1], "X", "Non-whitelist L2", "#D26A3A", excluded),
    ):
        ranked = counts.most_common()
        x_values = range(1, len(ranked) + 1)
        axis.bar(x_values, [count for _, count in ranked], width=0.9, color=color)
        axis.set_yscale("log")
        axis.set_title(f"{title} (n={len(ranked)})")
        axis.set_xlabel("Category rank (full code/name in CSV)")
        axis.set_ylabel("Metadata rows (log scale)")
        axis.grid(axis="y", alpha=0.22)
        tick_step = 1 if len(ranked) <= 15 else max(10, len(ranked) // 8)
        ticks = list(range(1, len(ranked) + 1, tick_step))
        if ticks[-1] != len(ranked):
            ticks.append(len(ranked))
        axis.set_xticks(ticks, [f"{prefix}{rank:03d}" for rank in ticks], rotation=45)
        axis.yaxis.set_major_formatter(formatter)
    figure.suptitle("Raw metadata L2 distribution by whitelist membership")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_target_counts(path: Path, train: dict[str, int], dev: dict[str, int]) -> None:
    categories = sorted(set(train) | set(dev), key=lambda item: (-train.get(item, 0), item))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["code", "rank_by_train", "target_category", "train_positive_targets", "dev_positive_targets", "in_formal_99"]
        )
        for rank, category in enumerate(categories, start=1):
            writer.writerow(
                [
                    f"T{rank:03d}",
                    rank,
                    category,
                    train.get(category, 0),
                    dev.get(category, 0),
                    train.get(category, 0) >= 4_000,
                ]
            )


def plot_target_counts(path: Path, train: dict[str, int], dev: dict[str, int]) -> None:
    categories = sorted(set(train) | set(dev), key=lambda item: (-train.get(item, 0), item))
    ranks = list(range(1, len(categories) + 1))
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    axis.plot(ranks, [train.get(item, 0) for item in categories], color="#1976A3", label="Train targets")
    axis.plot(ranks, [dev.get(item, 0) for item in categories], color="#D26A3A", label="Dev targets")
    axis.axhline(4_000, color="#333333", linestyle="--", linewidth=1, label="Train quota threshold = 4,000")
    axis.set_yscale("log")
    axis.set_xlabel("Target-category rank by Train support (full code/name in CSV)")
    axis.set_ylabel("Selected positive targets (log scale)")
    axis.set_title("Clean L2 > L3 positive-target support (all categories)")
    ticks = list(range(1, len(categories) + 1, 10))
    if ticks[-1] != len(categories):
        ticks.append(len(categories))
    axis.set_xticks(ticks, [f"T{rank:03d}" for rank in ticks], rotation=45)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def history_length_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            counts[json.loads(line)["full_history_length"]] += 1
    return counts


def weighted_percentile(counts: Counter[int], percentile: float) -> int:
    target = max(1, int((sum(counts.values()) * percentile + 99) // 100))
    cumulative = 0
    for value, count in sorted(counts.items()):
        cumulative += count
        if cumulative >= target:
            return value
    raise ValueError("cannot compute percentile of an empty distribution")


def summarize_history_lengths(counts: Counter[int]) -> dict:
    total = sum(counts.values())
    return {
        "samples": total,
        "minimum": min(counts),
        "mean": sum(length * count for length, count in counts.items()) / total,
        "p50": weighted_percentile(counts, 50),
        "p75": weighted_percentile(counts, 75),
        "p90": weighted_percentile(counts, 90),
        "p95": weighted_percentile(counts, 95),
        "p99": weighted_percentile(counts, 99),
        "maximum": max(counts),
        "at_most_50_ratio": sum(
            count for length, count in counts.items() if length <= 50
        )
        / total,
        "over_50_ratio": sum(
            count for length, count in counts.items() if length > 50
        )
        / total,
        "over_100_ratio": sum(
            count for length, count in counts.items() if length > 100
        )
        / total,
        "over_500_ratio": sum(
            count for length, count in counts.items() if length > 500
        )
        / total,
    }


def write_history_length_counts(
    path: Path, train: Counter[int], dev: Counter[int]
) -> None:
    lengths = sorted(set(train) | set(dev))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["full_history_length", "train_positive_samples", "dev_positive_samples"]
        )
        for length in lengths:
            writer.writerow([length, train[length], dev[length]])


def binned_history_ratios(counts: Counter[int]) -> list[float]:
    total = sum(counts.values())
    return [
        sum(
            count
            for length, count in counts.items()
            if length >= lower and (upper is None or length <= upper)
        )
        / total
        for lower, upper, _ in HISTORY_BINS
    ]


def plot_history_lengths(
    path: Path, train: Counter[int], dev: Counter[int]
) -> None:
    labels = [label for _, _, label in HISTORY_BINS]
    train_ratios = binned_history_ratios(train)
    dev_ratios = binned_history_ratios(dev)
    positions = list(range(len(labels)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    axis.bar(
        [position - width / 2 for position in positions],
        train_ratios,
        width,
        color="#1976A3",
        label="Train",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        dev_ratios,
        width,
        color="#D26A3A",
        label="Dev",
    )
    axis.axvline(3.5, color="#333333", linestyle="--", linewidth=1)
    axis.text(3.55, max(train_ratios + dev_ratios) * 0.72, "H50 truncation boundary")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Complete eligible-history length before target")
    axis.set_yscale("log")
    axis.set_ylabel("Share of positive samples (log scale)")
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value * 100:g}%")
    )
    axis.set_title("Untruncated complete eligible-history length distribution")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    raw = args.work_dir / "raw" / "meta_Home_and_Kitchen.jsonl"
    manifest = json.loads((args.work_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    output = args.work_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    result = {}
    if args.only in {"all", "categories"}:
        allowed, excluded = metadata_level_2_counts(raw)
        write_metadata_counts(output / "metadata_level_2_counts.csv", allowed, excluded)
        plot_metadata_counts(output / "metadata_level_2_distribution.png", allowed, excluded)

        train = manifest["unbalanced"]["train"]["target_counts"]
        dev = manifest["unbalanced"]["dev"]["target_counts"]
        write_target_counts(output / "positive_target_counts.csv", train, dev)
        plot_target_counts(output / "positive_target_distribution.png", train, dev)
        result["categories"] = {
                "whitelist_level_2": len(allowed),
                "non_whitelist_level_2": len(excluded),
                "whitelist_metadata_rows": sum(allowed.values()),
                "non_whitelist_metadata_rows": sum(excluded.values()),
                "positive_target_categories": len(set(train) | set(dev)),
        }
    if args.only in {"all", "history"}:
        data = args.work_dir / "data" / "unbalanced"
        train_history = history_length_counts(data / "train.jsonl")
        dev_history = history_length_counts(data / "dev.jsonl")
        write_history_length_counts(
            output / "full_history_length_counts.csv", train_history, dev_history
        )
        history_summary = {
            "definition": (
                "all earlier eligible in-domain positive interactions before target; "
                "never H50-truncated, but not the user's complete Amazon-wide behavior"
            ),
            "train": summarize_history_lengths(train_history),
            "dev": summarize_history_lengths(dev_history),
        }
        (output / "full_history_length_summary.json").write_text(
            json.dumps(history_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plot_history_lengths(
            output / "full_history_length_distribution.png",
            train_history,
            dev_history,
        )
        result["history"] = history_summary
    result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()