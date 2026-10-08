#!/usr/bin/env python3
"""Evaluate a base model or LoRA adapter over the frozen full taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from prompts import VIEWS, render_messages

CUTOFFS = (10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--task", choices=("choice", "ranking"), required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--adapter", type=Path)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--return-count", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(file_sha256(file_path).encode())
    return digest.hexdigest()


def shard_rows(rows: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1:
        raise ValueError("shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    return rows[shard_index::shard_count]


def template_ids(value) -> list[int]:
    return list(value["input_ids"] if hasattr(value, "keys") else value)


class TokenTrie:
    def __init__(self, sequences: dict[tuple[int, ...], str]) -> None:
        self.root: dict = {}
        self.labels = sequences
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})

    def allowed(self, generated: list[int]) -> list[int]:
        node = self.root
        for token in generated:
            if token not in node:
                return []
            node = node[token]
        return list(node)


def candidate_sequences(
    tokenizer, messages: list[dict], prompt_ids: list[int], labels: list[str]
) -> dict[tuple[int, ...], str]:
    sequences = {}
    for label in labels:
        full_ids = template_ids(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": f"New topic: {label}"}],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                enable_thinking=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("full chat does not begin with the generation prompt")
        suffix = full_ids[len(prompt_ids) :]
        if tokenizer.eos_token_id not in suffix:
            raise ValueError(f"chat template produced no EOS for {label!r}")
        sequence = tuple(suffix[: suffix.index(tokenizer.eos_token_id) + 1])
        if sequence in sequences:
            raise ValueError(f"token collision between {label!r} and {sequences[sequence]!r}")
        sequences[sequence] = label
    return sequences


def render_prompt(tokenizer, messages: list[dict]) -> list[int]:
    return template_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
        )
    )


def rank_labels(
    model, prompt_ids: list[int], trie: TokenTrie, eos_token_id: int, beams: int, count: int
) -> tuple[list[str], list[float]]:
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    def allowed_tokens(_batch_id: int, current_ids: torch.Tensor) -> list[int]:
        generated = current_ids.tolist()[len(prompt_ids) :]
        return trie.allowed(generated) or [eos_token_id]

    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            num_beams=beams,
            num_return_sequences=count,
            do_sample=False,
            max_new_tokens=max(len(sequence) for sequence in trie.labels),
            prefix_allowed_tokens_fn=allowed_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )
    ranked = []
    for output in generated.sequences:
        suffix = output.tolist()[len(prompt_ids) :]
        if eos_token_id in suffix:
            suffix = suffix[: suffix.index(eos_token_id) + 1]
        label = trie.labels.get(tuple(suffix))
        if label is None:
            raise ValueError(f"constrained decoder emitted an unknown sequence: {suffix}")
        ranked.append(label)
    if len(ranked) != count or len(set(ranked)) != count:
        raise ValueError(f"expected {count} unique ranked labels, got {ranked}")
    return ranked, generated.sequences_scores.detach().float().cpu().tolist()


def ranking_metrics(ranked: list[str], target: str) -> dict[str, float]:
    metrics = {}
    for cutoff in CUTOFFS:
        top = ranked[:cutoff]
        metrics[f"recall@{cutoff}"] = float(target in top)
        metrics[f"ndcg@{cutoff}"] = (
            1.0 / math.log2(ranked.index(target) + 2) if target in top else 0.0
        )
    return metrics


def macro_mean(rows: list[dict], metric: str) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["target_subcategory"]].append(float(row[metric]))
    per_label = {
        label: sum(values) / len(values) for label, values in sorted(grouped.items())
    }
    return sum(per_label.values()) / len(per_label), per_label


def summarize(rows: list[dict], labels: list[str], args: argparse.Namespace) -> dict:
    common = {
        "model": args.model,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "view": args.view,
        "task": args.task,
        "sample_count": len(rows),
        "taxonomy_size": len(labels),
        "num_beams": args.num_beams,
        "return_count": args.return_count,
        "prediction_counts": dict(sorted(Counter(row["predicted_label"] for row in rows).items())),
    }
    if args.task == "choice":
        macro, per_label = macro_mean(rows, "correct")
        return {
            **common,
            "candidate_count": 10,
            "macro_accuracy@10": macro,
            "per_label_accuracy@10": per_label,
        }
    metrics = {**common, "cutoffs": list(CUTOFFS)}
    for cutoff in CUTOFFS:
        for name in ("recall", "ndcg"):
            metric = f"{name}@{cutoff}"
            macro, per_label = macro_mean(rows, metric)
            metrics[f"macro_{metric}"] = macro
            metrics[f"per_label_{metric}"] = per_label
    return metrics


def main() -> None:
    args = parse_args()
    args.num_beams = args.num_beams or (10 if args.task == "choice" else 40)
    args.return_count = args.return_count or (1 if args.task == "choice" else 20)
    if args.return_count > args.num_beams:
        raise ValueError("return-count cannot exceed num-beams")
    labels = json.loads(args.labels_file.read_text(encoding="utf-8"))
    rows = pq.read_table(args.dev_file).to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = shard_rows(rows, args.shard_count, args.shard_index)
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter.resolve() if args.adapter else args.model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter.resolve())
    model.eval()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "model": args.model,
        "task": args.task,
        "view": args.view,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "adapter_sha256": directory_sha256(args.adapter.resolve()) if args.adapter else None,
        "dev_sha256": file_sha256(args.dev_file.resolve()),
        "labels_sha256": file_sha256(args.labels_file.resolve()),
        "num_beams": args.num_beams,
        "return_count": args.return_count,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "prompt_candidates": args.task == "choice",
        "decoding": "candidate_trie_constrained_beam_search",
    }
    contract_path = output_dir / "evaluation_config.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise ValueError(f"{output_dir}: existing evaluation configuration differs")
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path = output_dir / "predictions.jsonl"
    cached = {}
    if prediction_path.exists():
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cached[row["sample_id"]] = row
    with prediction_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["sample_id"] in cached:
                continue
            task_labels = row["candidate_subcategories"] if args.task == "choice" else labels
            messages = render_messages(
                row["history"],
                args.view,
                candidates=task_labels if args.task == "choice" else None,
            )
            prompt_ids = render_prompt(tokenizer, messages)
            sequences = candidate_sequences(tokenizer, messages, prompt_ids, task_labels)
            if row["target_subcategory"] not in sequences.values():
                raise ValueError(f"{row['sample_id']}: target is absent from candidates")
            ranked, scores = rank_labels(
                model,
                prompt_ids,
                TokenTrie(sequences),
                tokenizer.eos_token_id,
                args.num_beams,
                args.return_count,
            )
            prediction = {
                "sample_id": row["sample_id"],
                "target_subcategory": row["target_subcategory"],
                "predicted_label": ranked[0],
                "ranked_labels": ranked,
                "sequence_scores": scores,
                "correct": ranked[0] == row["target_subcategory"],
            }
            if args.task == "ranking":
                prediction.update(ranking_metrics(ranked, row["target_subcategory"]))
            cached[row["sample_id"]] = prediction
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if (index + 1) % 25 == 0:
                print(f"evaluated {index + 1}/{len(rows)}", flush=True)
    ordered = [cached[row["sample_id"]] for row in rows]
    metrics = summarize(ordered, labels, args)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()