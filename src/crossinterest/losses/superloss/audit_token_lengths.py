#!/usr/bin/env python3
"""Audit exact Qwen chat-template token lengths without truncation."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset as ArrowDataset
from transformers import AutoTokenizer

from prompts import VIEWS, render_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(__file__).parents[1] / "mind/balanced/train.parquet",
    )
    parser.add_argument("--view", choices=(*VIEWS, "all"), default="all")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ArrowDataset.from_parquet(
        str(args.train_file),
        columns=["sample_id", "history", "target_subcategory"],
        keep_in_memory=False,
    )
    if args.sample_id:
        requested = set(args.sample_id)
        dataset = dataset.filter(lambda sample_id: sample_id in requested, input_columns="sample_id")
        found = set(dataset["sample_id"])
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"sample IDs not found: {missing}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    views = VIEWS if args.view == "all" else (args.view,)
    maxima = {view: (0, "") for view in views}

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : min(start + args.batch_size, len(dataset))]
        for view in views:
            rendered = [
                tokenizer.apply_chat_template(
                    render_messages(history, view, target=target),
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                for history, target in zip(batch["history"], batch["target_subcategory"])
            ]
            lengths = [
                len(input_ids)
                for input_ids in tokenizer(
                    rendered,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
            ]
            index = max(range(len(lengths)), key=lengths.__getitem__)
            if lengths[index] > maxima[view][0]:
                maxima[view] = (lengths[index], batch["sample_id"][index])
        if start == 0 or start + args.batch_size >= len(dataset):
            print(f"progress={min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)

    exceeded = False
    for view, (length, sample_id) in maxima.items():
        print(f"view={view} max_tokens={length} sample_id={sample_id}")
        exceeded |= length > args.max_seq_length
    if exceeded:
        raise SystemExit(f"at least one prompt exceeds max_seq_length={args.max_seq_length}")


if __name__ == "__main__":
    main()