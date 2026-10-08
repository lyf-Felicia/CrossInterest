#!/usr/bin/env python3
"""Attach validated target-blind summaries to an existing Amazon parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INTEREST_ID_PATTERN = re.compile(r"<interest_\d{3}>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_valid_summaries(path: Path) -> dict[str, dict]:
    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            latest[row["sample_id"]] = row
    invalid = [sample_id for sample_id, row in latest.items() if not row.get("valid")]
    if invalid:
        raise ValueError(f"latest summary records include {len(invalid)} invalid rows")
    return latest


def render_grounded_summary(record: dict) -> str:
    summary = record["summary"]
    rendered = (
        "Stable needs: "
        + "; ".join(
            f"{need['need']} [items {','.join(map(str, need['positions']))}]"
            for need in summary["stable_needs"]
        )
        + f"\nRecent focus: {summary['recent_focus']} "
        + f"[items {','.join(map(str, summary['recent_positions']))}]"
        + f"\nHistory diversity: {summary['diversity']}"
    )
    if INTEREST_ID_PATTERN.search(rendered):
        raise ValueError(f"{record['sample_id']}: Interest ID leaked into summary")
    if record["target_interest"].casefold() in rendered.casefold():
        raise ValueError(f"{record['sample_id']}: exact target leaked into summary")
    return rendered


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source = pq.read_table(args.source, memory_map=True)
    if args.id_column not in source.schema.names:
        raise ValueError(f"source lacks ID column {args.id_column}")
    if "history_summary" in source.schema.names:
        raise ValueError("source already contains history_summary")
    records = latest_valid_summaries(args.summaries)
    ids = source[args.id_column].to_pylist()
    targets = source["target_interest"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("source sample IDs are not unique")
    if set(ids) != set(records):
        raise ValueError(
            f"summary/source ID mismatch: missing={len(set(ids) - set(records))}, "
            f"extra={len(set(records) - set(ids))}"
        )
    summaries = []
    for sample_id, target in zip(ids, targets, strict=True):
        record = records[sample_id]
        if record["target_interest"] != target:
            raise ValueError(f"{sample_id}: summary target audit field differs")
        summaries.append(render_grounded_summary(record))
    augmented = source.append_column("history_summary", pa.array(summaries, pa.string()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(augmented, args.output, compression="zstd")
    restored = pq.read_table(args.output, memory_map=True)
    original_columns = source.schema.names
    if restored.num_rows != source.num_rows:
        raise AssertionError("output row count changed")
    for column in original_columns:
        if not restored[column].equals(source[column]):
            raise AssertionError(f"source column changed: {column}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "summaries": str(args.summaries.resolve()),
        "summaries_sha256": sha256_file(args.summaries),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "id_column": args.id_column,
        "rows": source.num_rows,
        "original_columns_unchanged": True,
        "summary_field": "history_summary",
        "included_summary_components": [
            "stable_needs",
            "recent_focus",
            "diversity",
        ],
        "excluded_summary_components": ["cross_category_pattern"],
        "mean_summary_characters": sum(map(len, summaries)) / len(summaries),
    }
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()