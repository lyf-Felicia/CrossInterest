#!/usr/bin/env python3
"""Evaluate MeRA with paper metrics and unified project metrics from one forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from crossinterest.baselines.mera.core import home_metrics, paper_metrics, query_set_metrics, set_ranking_score, validate_label_space, write_json
from crossinterest.baselines.mera.data import expand_unified_eval, load_home_rows, load_official_rows, prompt_token_ids, render_prompt, validate_rows
from crossinterest.baselines.mera.modeling import category_token_mapping, initialize_category_tokens, register_category_tokens
from crossinterest.baselines.mera.train import MODELS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODELS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", choices=("arts", "grocery", "home"), required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--use-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def fingerprint(value: dict) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def load_rows(path: Path, dataset: str) -> list[dict]:
    return (
        load_home_rows(path, training=False)
        if dataset == "home"
        else load_official_rows(path)
    )


def load_model(
    model_name: str,
    model_revision: str,
    adapter: Path,
    tokenizer,
    label_space: dict,
    registration: dict,
    use_kernels: bool,
):
    model_options = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "cuda",
    }
    if use_kernels:
        model_options["use_kernels"] = True
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        **model_options,
    )
    initialize_category_tokens(
        base,
        tokenizer,
        label_space,
        registration,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model


def collate_prompt_ids(
    prompt_ids: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, prompt_ids))
    padded = [[pad_token_id] * (maximum - len(ids)) + ids for ids in prompt_ids]
    masks = [[0] * (maximum - len(ids)) + [1] * len(ids) for ids in prompt_ids]
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def rank_rows(
    model,
    tokenizer,
    rows: list[dict],
    model_key: str,
    label_space: dict,
    category_to_token: dict[str, str],
    token_ids: list[int],
    batch_size: int,
    max_length: int,
) -> list[list[str]]:
    rankings = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = [
            prompt_token_ids(
                tokenizer,
                render_prompt(
                    row,
                    category_to_token,
                    model_key=model_key,
                    training=False,
                ),
                model_key,
            )
            for row in batch
        ]
        too_long = [
            (row["sample_id"], len(ids))
            for row, ids in zip(batch, encoded, strict=True)
            if len(ids) > max_length
        ]
        if too_long:
            raise ValueError(
                f"evaluation truncation is forbidden; first overlength row: {too_long[0]}"
            )
        input_ids, attention_mask = collate_prompt_ids(
            encoded,
            tokenizer.pad_token_id,
            model.device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, -1, token_ids]
        orders = torch.argsort(logits, dim=-1, descending=True).cpu().tolist()
        rankings.extend(
            [
                [label_space["prediction_labels"][index] for index in order]
                for order in orders
            ]
        )
        print(f"ranked={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return rankings


def unified_metrics(
    source_rows: list[dict],
    source_rankings: list[list[str]],
    label_space: dict,
) -> tuple[dict, list[dict]]:
    ranking_by_id = {
        row["sample_id"]: ranking
        for row, ranking in zip(source_rows, source_rankings, strict=True)
    }
    expanded = expand_unified_eval(source_rows, label_space)
    gold_details_by_source = defaultdict(list)
    for row in expanded:
        ranking = ranking_by_id[row["sample_id"]]
        target = row["targets"][0]
        candidate_set = set(row["candidates"])
        candidate_ranking = [label for label in ranking if label in candidate_set]
        gold_details_by_source[row["sample_id"]].append(
            {
                "target": target,
                "target_rank": ranking.index(target) + 1,
                "candidate_prediction": candidate_ranking[0],
                "candidate_correct": candidate_ranking[0] == target,
                "candidates": row["candidates"],
            }
        )
    details = []
    source_accuracies = []
    for row in source_rows:
        ranking = ranking_by_id[row["sample_id"]]
        gold_details = gold_details_by_source[row["sample_id"]]
        if len(gold_details) != len(set(row["targets"])):
            raise ValueError(f"{row['sample_id']}: incomplete per-gold 10-way scores")
        mean_accuracy = sum(
            float(detail["candidate_correct"]) for detail in gold_details
        ) / len(gold_details)
        source_accuracies.append(mean_accuracy)
        details.append(
            {
                "source_sample_id": row["sample_id"],
                "targets": list(dict.fromkeys(row["targets"])),
                "target_count": len(set(row["targets"])),
                "mean_accuracy_10way": mean_accuracy,
                **set_ranking_score(ranking, row["targets"]),
                "gold_details": gold_details,
            }
        )
    query_metrics = query_set_metrics(
        source_rankings,
        [row["targets"] for row in source_rows],
    )
    query_metrics["mean_accuracy_10way"] = sum(source_accuracies) / len(
        source_accuracies
    )
    category_macro = None
    if all(
        len(set(row["targets"])) == 1 and "candidates" in row
        for row in source_rows
    ):
        category_macro = home_metrics(
            source_rankings,
            [row["targets"][0] for row in source_rows],
            [row["candidates"] for row in source_rows],
        )
    return {
        "aggregation": "equal mean over source samples",
        "multi_gold_recall": "set recall",
        "multi_gold_ndcg": "binary-relevance DCG normalized by ideal DCG",
        "ten_way_accuracy": "mean over golds within source, then mean over sources",
        "query_mean": query_metrics,
        "category_macro_single_gold_only": category_macro,
        "gold_evaluation_count": len(expanded),
    }, details


def write_jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model_name = args.model or MODELS[args.model_key]
    if args.use_kernels and args.model_key != "qwen35_9b":
        raise ValueError("--use-kernels is reserved for the Qwen3.5 main model")
    if args.model_key == "qwen2_1_5b" and args.dataset == "home":
        raise ValueError("the Qwen2 paper-model reference is limited to official data")
    eval_file = args.eval_file.resolve()
    adapter = args.adapter.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_space_path = args.assets_dir / "label_space.json"
    label_space = validate_label_space(
        json.loads(label_space_path.read_text(encoding="utf-8"))
    )
    rows = load_rows(eval_file, args.dataset)
    validate_rows(
        rows,
        set(label_space["labels"]),
        single_target=args.dataset == "home",
        prediction_labels=set(label_space["prediction_labels"]),
    )
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and the evaluation size")
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    token_style = (
        "official_quoted_name"
        if args.model_key == "qwen2_1_5b"
        else "indexed"
    )
    runtime_category_tokens = category_token_mapping(label_space, token_style)
    registration = register_category_tokens(
        tokenizer,
        label_space,
        runtime_category_tokens,
    )
    contract = {
        "schema_version": 1,
        "task": "MeRA paper and unified evaluation",
        "model_key": args.model_key,
        "model": model_name,
        "model_revision": args.model_revision,
        "dataset": args.dataset,
        "eval_file": str(eval_file),
        "eval_sha256": sha256_file(eval_file),
        "adapter": str(adapter),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "label_space_fingerprint": label_space["fingerprint"],
        "category_token_ids": registration["token_ids"],
        "category_token_style": token_style,
        "prompt_style": (
            "official concatenated quoted-category-token inference"
            if args.model_key == "qwen2_1_5b"
            else "model-native chat template with unified project instruction"
        ),
        "forward_passes_per_source_sample": 1,
        "paper_metrics": args.dataset != "home",
        "unified_metrics": {
            "primary": [
                "mean_accuracy_10way",
                "mean_recall_at_10",
                "mean_recall_at_20",
                "mean_ndcg_at_10",
                "mean_ndcg_at_20",
            ],
            "aggregation": "equal mean over source samples",
            "multi_gold_recall": "set recall",
            "multi_gold_ndcg": "binary-relevance normalized DCG",
            "random_gold_selection": False,
        },
        "candidate_count": 10,
        "same_parent_negative_limit": 3,
        "context_label_count": label_space["label_count"],
        "candidate_free_label_count": label_space["prediction_label_count"],
        "history_masking_for_global_ranking": False,
        "decoding": "direct category-token logits",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "use_kernels": args.use_kernels,
        "sample_limit": args.limit,
    }
    contract["fingerprint"] = fingerprint(contract)
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"{output_dir}: existing evaluation contract differs")
    write_json(contract_path, contract)
    model = load_model(
        model_name,
        args.model_revision,
        adapter,
        tokenizer,
        label_space,
        registration,
        args.use_kernels,
    )
    rankings = rank_rows(
        model,
        tokenizer,
        rows,
        args.model_key,
        label_space,
        runtime_category_tokens,
        [
            registration["token_ids"][label_space["labels"].index(category)]
            for category in label_space["prediction_labels"]
        ],
        args.batch_size,
        args.max_length,
    )
    unified, unified_details = unified_metrics(rows, rankings, label_space)
    paper = (
        paper_metrics(rankings, [row["targets"] for row in rows], cutoffs=(1, 2, 3, 4))
        if args.dataset != "home"
        else None
    )
    if paper is not None:
        for cutoff in (10, 20):
            paper_at_cutoff = paper_metrics(
                rankings,
                [row["targets"] for row in rows],
                cutoffs=(cutoff,),
            )[f"recall_at_{cutoff}"]
            unified_at_cutoff = unified["query_mean"][f"mean_recall_at_{cutoff}"]
            if abs(paper_at_cutoff - unified_at_cutoff) > 1e-12:
                raise AssertionError(
                    f"set Recall@{cutoff} differs between paper and unified paths"
                )
    metrics = {
        "model_key": args.model_key,
        "dataset": args.dataset,
        "source_sample_count": len(rows),
        "gold_evaluation_count": unified["gold_evaluation_count"],
        "paper_metrics": paper,
        "unified_metrics": unified,
        "run_fingerprint": contract["fingerprint"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(
        output_dir / "source_rankings.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "targets": row["targets"],
                "ranking": ranking,
            }
            for row, ranking in zip(rows, rankings, strict=True)
        ],
    )
    write_jsonl(output_dir / "unified_details.jsonl", unified_details)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
