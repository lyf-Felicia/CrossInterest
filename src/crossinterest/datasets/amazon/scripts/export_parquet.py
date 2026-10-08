#!/usr/bin/env python3
"""Stream formal Amazon JSONL records into typed Parquet datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("timestamp_iso", pa.string(), nullable=False),
        pa.field("history_limit", pa.int16(), nullable=False),
        pa.field("full_history_length", pa.int32(), nullable=False),
        pa.field("full_seen_interests", pa.list_(pa.string()), nullable=False),
        pa.field("history_parent_asins", pa.list_(pa.string()), nullable=False),
        pa.field("historical_titles", pa.list_(pa.string()), nullable=False),
        pa.field("historical_level_2", pa.list_(pa.string()), nullable=False),
        pa.field("historical_interests", pa.list_(pa.string()), nullable=False),
        pa.field(
            "history_timestamps",
            pa.list_(pa.timestamp("ms", tz="UTC")),
            nullable=False,
        ),
        pa.field("target_parent_asin", pa.string(), nullable=False),
        pa.field("target_title", pa.string(), nullable=False),
        pa.field("target_level_2", pa.string(), nullable=False),
        pa.field("target_interest", pa.string(), nullable=False),
        pa.field("target_rating", pa.float32(), nullable=False),
        pa.field("same_history_new_interest_count", pa.int16(), nullable=False),
        pa.field("target_selection", pa.string(), nullable=False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args()


def prepare_record(record: dict) -> dict:
    record["timestamp"] = pa.scalar(record["timestamp"], type=SCHEMA.field("timestamp").type).as_py()
    history_type = SCHEMA.field("history_timestamps").type.value_type
    record["history_timestamps"] = [
        pa.scalar(value, type=history_type).as_py()
        for value in record["history_timestamps"]
    ]
    return record


def convert(source: Path, destination: Path, batch_size: int) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    rows = []
    count = 0
    first_id = None
    last_id = None
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = prepare_record(json.loads(line))
                first_id = first_id or record["id"]
                last_id = record["id"]
                rows.append(record)
                if len(rows) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                    count += len(rows)
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                count += len(rows)
    finally:
        writer.close()
    temporary.replace(destination)
    parquet = pq.ParquetFile(destination)
    if parquet.metadata.num_rows != count or parquet.schema_arrow != SCHEMA:
        raise ValueError(f"Parquet verification failed: {destination}")
    return {
        "rows": count,
        "row_groups": parquet.metadata.num_row_groups,
        "first_id": first_id,
        "last_id": last_id,
        "bytes": destination.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    result = {}
    for split in ("train", "dev"):
        result[split] = convert(
            args.input_dir / f"{split}.jsonl",
            args.dataset_dir / f"{split}.parquet",
            args.batch_size,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()