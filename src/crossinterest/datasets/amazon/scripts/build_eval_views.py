#!/usr/bin/env python3
"""Build unresampled closed-set and OOV Dev views from the natural Dev pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from crossinterest.datasets.amazon.scripts.build import JsonlWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--formal-dir",
        type=Path,
        required=True,
        help="Write the formal closed-set Dev JSONL staging file in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.work_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = set(manifest["balanced_train"]["support"])
    data = args.work_dir / "data"
    writers = {
        "dev_in_support": JsonlWriter(args.formal_dir / "dev.jsonl"),
        "dev_oov": JsonlWriter(data / "evaluation" / "dev_oov.jsonl"),
    }
    labels = {name: Counter() for name in writers}
    with (data / "unbalanced" / "dev.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            name = (
                "dev_in_support"
                if record["target_interest"] in support
                else "dev_oov"
            )
            writers[name].write(record)
            labels[name][record["target_interest"]] += 1
    stats = {}
    for name, writer in writers.items():
        writer.close()
        stats[name] = {
            "sample_count": writer.count,
            "label_count": len(labels[name]),
            "resampled": False,
        }
    manifest["evaluation_views"] = stats
    manifest["contract"]["primary_dev"] = (
        "natural Dev records whose target is in the frozen Train support; no resampling"
    )
    manifest["contract"]["open_world_dev"] = (
        "complete natural Dev, with OOV records separately emitted for robustness analysis"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()