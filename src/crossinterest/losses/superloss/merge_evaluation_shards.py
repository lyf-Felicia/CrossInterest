#!/usr/bin/env python3
"""Validate and merge deterministic evaluation shards in original dev order."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pyarrow.parquet as pq

from evaluate import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=Path(__file__).parents[1] / "mind/balanced/dev.parquet",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=Path(__file__).parents[1] / "mind/balanced/labels.json",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank prediction line")
            rows.append(json.loads(line))
    return rows


def normalized_contract(contract: dict) -> dict:
    return {name: value for name, value in contract.items() if name != "shard_index"}


def validate_contracts(contracts: list[dict], shard_count: int) -> dict:
    if len(contracts) != shard_count:
        raise ValueError(f"expected {shard_count} shard contracts, got {len(contracts)}")
    reference = normalized_contract(contracts[0])
    for shard_index, contract in enumerate(contracts):
        if contract.get("shard_count") != shard_count:
            raise ValueError(f"shard {shard_index}: inconsistent shard_count")
        if contract.get("shard_index") != shard_index:
            raise ValueError(f"shard {shard_index}: inconsistent shard_index")
        if normalized_contract(contract) != reference:
            raise ValueError(f"shard {shard_index}: evaluation contract differs")
    return reference


def merge_shard_rows(shards: list[list[dict]], expected_ids: list[str]) -> list[dict]:
    if not shards:
        raise ValueError("at least one shard is required")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("dev data contains duplicate sample IDs")
    by_id = {}
    for shard_index, rows in enumerate(shards):
        observed_ids = [row["sample_id"] for row in rows]
        expected_shard_ids = expected_ids[shard_index:: len(shards)]
        if observed_ids != expected_shard_ids:
            raise ValueError(
                f"shard {shard_index}: prediction IDs do not match its dev partition"
            )
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in by_id:
                raise ValueError(f"duplicate prediction for {sample_id}")
            by_id[sample_id] = row
    if set(by_id) != set(expected_ids):
        raise ValueError("merged predictions do not exactly cover the dev data")
    return [by_id[sample_id] for sample_id in expected_ids]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("shard-count must be positive")
    shard_dirs = [args.shards_root.resolve() / f"shard-{index}" for index in range(args.shard_count)]
    contracts = [read_json(path / "evaluation_config.json") for path in shard_dirs]
    contract = validate_contracts(contracts, args.shard_count)
    limit = contract.get("limit")
    dev_rows = pq.read_table(args.dev_file.resolve(), columns=["sample_id"]).to_pylist()
    if limit is not None:
        dev_rows = dev_rows[:limit]
    expected_ids = [row["sample_id"] for row in dev_rows]
    shards = [read_jsonl(path / "predictions.jsonl") for path in shard_dirs]
    rows = merge_shard_rows(shards, expected_ids)
    labels = json.loads(args.labels_file.resolve().read_text(encoding="utf-8"))
    summary_args = Namespace(
        model=contract["model"],
        adapter=Path(contract["adapter"]) if contract["adapter"] else None,
        view=contract["view"],
        task=contract["task"],
        num_beams=contract["num_beams"],
        return_count=contract["return_count"],
    )
    metrics = summarize(rows, labels, summary_args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_contract = {
        name: value
        for name, value in contract.items()
        if name not in {"shard_count", "shard_index"}
    }
    merged_contract.update(
        {
            "merged_shard_count": args.shard_count,
            "sample_count": len(rows),
            "shard_prediction_sha256": [
                file_sha256(path / "predictions.jsonl") for path in shard_dirs
            ],
        }
    )
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "evaluation_config.json", merged_contract)
    write_json(output_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                "task": contract["task"],
                "view": contract["view"],
                "sample_count": len(rows),
                "merged_shard_count": args.shard_count,
            }
        )
    )


if __name__ == "__main__":
    main()