#!/usr/bin/env python3
"""LoRA SFT for the public Amazon adaptation of Google's K=2 method."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, set_seed

from crossinterest.training.train import AssistantOnlyCollator, latest_checkpoint, load_model, qwen_fast_path_status, sha256_file, tokenize_assistant_only, write_json

from crossinterest.baselines.google.prompts import PROMPT_SCHEMA_VERSION, render_training_messages


HERE = _source_workspace('eval_baseline_google')
SEED = 20260825
EXPECTED_TRAIN_ROWS = 1_120
EXPECTED_LABEL_COUNT = 112
SCALED_PAPER_STEPS = 442


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, default=HERE / "data" / "train_top10_pairs.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--expected-label-count", type=int, default=EXPECTED_LABEL_COUNT)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=SCALED_PAPER_STEPS)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--expected-effective-batch-size", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exclude-vision-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-qwen-fast-path", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class GooglePairDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, limit: int | None, expected_rows: int):
        parquet = pq.ParquetFile(path)
        if limit is None and parquet.metadata.num_rows != expected_rows:
            raise ValueError(f"formal training requires {expected_rows} rows, found {parquet.metadata.num_rows}")
        self.rows = pq.read_table(path).to_pylist()
        self.size = min(limit, len(self.rows)) if limit is not None else len(self.rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        messages = render_training_messages(row)
        if row["prompt"] != messages[0]["content"] or row["assistant"] != messages[1]["content"]:
            raise ValueError(f"{row['source_id']}: frozen prompt or assistant differs from {PROMPT_SCHEMA_VERSION}")
        encoded = tokenize_assistant_only(self.tokenizer, messages)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"{row['source_id']}: {len(encoded['input_ids'])} tokens exceed "
                f"max_seq_length={self.max_length}; truncation is forbidden"
            )
        return encoded


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    if effective_batch != args.expected_effective_batch_size:
        raise ValueError(
            f"effective batch must be {args.expected_effective_batch_size}, got {effective_batch}"
        )
    set_seed(SEED)
    train_file = args.train_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_labels = set(
        pq.read_table(train_file, columns=["target_cluster"])["target_cluster"].to_pylist()
    )
    if len(target_labels) != args.expected_label_count:
        raise ValueError(f"expected {args.expected_label_count} labels, found {len(target_labels)}")

    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    qwen_fast_path = None
    if model_config.model_type == "qwen3_5":
        qwen_fast_path = qwen_fast_path_status()
        if args.require_qwen_fast_path and not qwen_fast_path["available"]:
            raise RuntimeError("Qwen3.5 fast path is required but unavailable: " + json.dumps(qwen_fast_path))
    elif args.require_qwen_fast_path:
        raise ValueError("--require-qwen-fast-path is only valid for Qwen3.5 models")

    contract = {
        "schema_version": 1,
        "task": "google_interest_exploration_public_amazon_sft",
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "candidate_free": True,
        "assistant_only": True,
        "model": args.model,
        "train_file": str(train_file),
        "train_sha256": sha256_file(train_file),
        "formal_train_rows": args.expected_train_rows,
        "label_count": len(target_labels),
        "sample_limit": args.limit,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "effective_batch_size": effective_batch,
        "paper_reference_batch_size": 16,
        "paper_reference_steps": 3000,
        "scaled_reference_steps": SCALED_PAPER_STEPS,
        "seed": SEED,
    }
    contract_path = output_dir / "train_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"{output_dir}: existing training contract differs")
    if is_main_process:
        write_json(contract_path, contract)
        write_json(
            output_dir / "runtime_config.json",
            {
                "world_size": world_size,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch,
                "gradient_checkpointing": args.gradient_checkpointing,
                "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
                "qwen_fast_path": qwen_fast_path,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = GooglePairDataset(
        train_file, tokenizer, args.max_seq_length, args.limit, args.expected_train_rows
    )
    model = load_model(args.model, args.use_kernels)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    exclude_modules = r"(?:.*\.)?visual\..*" if args.exclude_vision_lora else None
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_visual = [name for name in trainable if ".visual." in name or name.startswith("visual.")]
    if args.exclude_vision_lora and trainable_visual:
        raise ValueError("exclude_vision_lora left trainable visual parameters")
    if not args.ddp_find_unused_parameters and trainable_visual:
        raise ValueError("ddp_find_unused_parameters=False requires excluding visual LoRA modules")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    checkpoint = latest_checkpoint(output_dir) if args.resume else None
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    if is_main_process:
        tokenizer.save_pretrained(final_adapter)
        write_json(
            output_dir / "train_metrics.json",
            {
                **result.metrics,
                **contract,
                "sample_count": len(dataset),
                "resumed_from": str(checkpoint) if checkpoint else None,
            },
        )


if __name__ == "__main__":
    main()