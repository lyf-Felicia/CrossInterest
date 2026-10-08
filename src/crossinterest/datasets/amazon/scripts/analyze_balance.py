#!/usr/bin/env python3
"""Compare data-driven balance policies on an emitted Train/Dev mother pool."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--quotas",
        type=int,
        nargs="+",
        default=[100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
    )
    parser.add_argument("--minimum-quota", type=int, default=500)
    parser.add_argument("--minimum-labels", type=int, default=50)
    parser.add_argument("--minimum-dev-coverage", type=float, default=0.90)
    return parser.parse_args()


def label_counts(path: Path) -> Counter[str]:
    counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            counts[json.loads(line)["target_interest"]] += 1
    return counts


def main() -> None:
    args = parse_args()
    data = args.work_dir / "data" / "unbalanced"
    train = label_counts(data / "train.jsonl")
    dev = label_counts(data / "dev.jsonl")
    rows = []
    for quota in sorted(set(args.quotas)):
        support = sorted(label for label, count in train.items() if count >= quota)
        balanced_size = len(support) * quota
        dev_samples_in_support = sum(dev[label] for label in support)
        rows.append(
            {
                "quota": quota,
                "retained_labels": len(support),
                "balanced_train_samples": balanced_size,
                "train_mother_pool_samples": sum(train.values()),
                "balanced_to_train_ratio": (
                    balanced_size / sum(train.values()) if train else 0.0
                ),
                "dev_labels_total": len(dev),
                "dev_labels_covered": sum(label in support for label in dev),
                "dev_samples_total": sum(dev.values()),
                "dev_samples_covered": dev_samples_in_support,
                "dev_sample_coverage": (
                    dev_samples_in_support / sum(dev.values()) if dev else 0.0
                ),
            }
        )
    for row in rows:
        row["feasible"] = (
            row["quota"] >= args.minimum_quota
            and row["retained_labels"] >= args.minimum_labels
            and row["dev_sample_coverage"] >= args.minimum_dev_coverage
        )
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        max_log_quota = max(math.log(row["quota"]) for row in feasible)
        max_labels = max(row["retained_labels"] for row in feasible)
        max_log_size = max(math.log(row["balanced_train_samples"]) for row in feasible)
        for row in feasible:
            row["selection_score"] = (
                math.log(row["quota"]) / max_log_quota
                * row["retained_labels"] / max_labels
                * math.log(row["balanced_train_samples"]) / max_log_size
                * row["dev_sample_coverage"]
            ) ** 0.25
        recommended = max(feasible, key=lambda row: (row["selection_score"], row["quota"]))
    else:
        recommended = None
    result = {
        "policy": (
            "for each quota q, retain all Train labels with support >=q and sample "
            "exactly q records per retained label"
        ),
        "selection_constraints": {
            "minimum_quota": args.minimum_quota,
            "minimum_labels": args.minimum_labels,
            "minimum_dev_sample_coverage": args.minimum_dev_coverage,
        },
        "selection_score": (
            "geometric mean of normalized log(quota), retained label count, "
            "log(balanced size), and Dev sample coverage among feasible policies"
        ),
        "quota_comparison": rows,
        "recommended": recommended,
    }
    output = args.work_dir / "BALANCE_ANALYSIS.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()