#!/usr/bin/env python3
"""Train one Amazon prompt-v3 view with candidate-free assistant-only LoRA SFT."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from crossinterest.legacy.training_99.prompts import INPUT_TYPES, render_training_messages
    from crossinterest.legacy.training_99.label_space import LABEL_MODES, load_label_mapping
except ImportError:
    from prompts import INPUT_TYPES, render_training_messages
    from label_space import LABEL_MODES, load_label_mapping


SEED = 20260825
EXPECTED_TRAIN_ROWS = 396_000
EXPECTED_LABEL_COUNT = 99
HERE = _source_workspace('eval_amazon_sft_99')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="category")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=HERE / "data" / "interest_id_mapping.json",
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--selection-size", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--expected-label-count", type=int, default=EXPECTED_LABEL_COUNT)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--expected-effective-batch-size", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--exclude-vision-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-qwen-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def template_ids(value) -> list[int]:
    return list(value["input_ids"] if hasattr(value, "keys") else value)


def apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    options = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": True,
    }
    try:
        value = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, **options)
    return template_ids(value)


def tokenize_assistant_only(tokenizer, messages: list[dict]) -> dict[str, list[int]]:
    prefix = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    complete = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    if complete[: len(prefix)] != prefix:
        raise ValueError("full chat does not begin with the assistant generation prefix")
    labels = [-100] * len(prefix) + complete[len(prefix) :]
    if not any(label != -100 for label in labels):
        raise ValueError("assistant-only mask contains no trainable tokens")
    return {"input_ids": complete, "labels": labels, "attention_mask": [1] * len(complete)}


class AmazonSftDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        view: str,
        label_mode: str,
        mapping: dict | None,
        max_length: int,
        limit: int | None,
        selection_ids: set[str] | None,
        expected_train_rows: int,
    ):
        parquet = pq.ParquetFile(path)
        if limit is None and parquet.metadata.num_rows != expected_train_rows:
            raise ValueError(
                f"formal training requires {expected_train_rows} rows, found "
                f"{parquet.metadata.num_rows}"
            )
        table = pq.read_table(
            path,
            columns=[
                "id",
                "historical_titles",
                "historical_interests",
                "full_seen_interests",
                "target_interest",
            ],
            memory_map=True,
        )
        if selection_ids is not None:
            selected_indices = [
                index
                for index, sample_id in enumerate(table["id"].to_pylist())
                if sample_id in selection_ids
            ]
            if len(selected_indices) != len(selection_ids):
                raise ValueError("selection contains IDs absent from the training parquet")
            table = table.take(selected_indices)
        self.table = table
        self.size = min(limit, len(table)) if limit is not None else len(table)
        self.tokenizer = tokenizer
        self.view = view
        self.label_mode = label_mode
        self.mapping = mapping
        self.max_length = max_length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.table.slice(index, 1).to_pylist()[0]
        messages = render_training_messages(
            row,
            self.view,
            self.label_mode,
            self.mapping,
        )
        encoded = tokenize_assistant_only(self.tokenizer, messages)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"{row['id']}: {len(encoded['input_ids'])} tokens exceed "
                f"max_seq_length={self.max_length}; truncation is forbidden"
            )
        return encoded


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required_files = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        )
        if all((path / name).is_file() for name in required_files):
            checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def qwen_fast_path_status() -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    def package_version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    symbols = {
        name: getattr(modeling_qwen3_5, name) is not None
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        )
    }
    return {
        "available": bool(modeling_qwen3_5.is_fast_path_available),
        "symbols": symbols,
        "flash_linear_attention_version": package_version("flash-linear-attention"),
        "causal_conv1d_version": package_version("causal-conv1d"),
    }


def load_model(model_name: str, use_kernels: bool):
    common = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_kernels": use_kernels,
    }
    return AutoModelForCausalLM.from_pretrained(model_name, **common)


def load_selection_ids(path: Path, selection_size: int) -> set[str]:
    if selection_size <= 0:
        raise ValueError("--selection-size must be positive")
    table = pq.read_table(path, columns=["id", "nested_rank"])
    ranks = table["nested_rank"].to_pylist()
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("selection nested_rank must be exactly 1..N")
    if selection_size > len(table):
        raise ValueError("--selection-size exceeds the selection file")
    selected = {
        sample_id
        for sample_id, rank in zip(table["id"].to_pylist(), ranks, strict=True)
        if rank <= selection_size
    }
    if len(selected) != selection_size:
        raise ValueError("selection prefix contains duplicate sample IDs")
    return selected


def training_label_counts(
    train_file: Path,
    selection_ids: set[str] | None,
) -> Counter[str]:
    table = pq.read_table(train_file, columns=["id", "target_interest"])
    return Counter(
        category
        for sample_id, category in zip(
            table["id"].to_pylist(),
            table["target_interest"].to_pylist(),
            strict=True,
        )
        if selection_ids is None or sample_id in selection_ids
    )


def main() -> None:
    args = parse_args()
    if (args.selection_file is None) != (args.selection_size is None):
        raise ValueError("--selection-file and --selection-size must be provided together")
    if args.limit is not None and args.selection_file is not None:
        raise ValueError("--limit cannot be combined with --selection-file")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    effective_batch = (
        args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    )
    if effective_batch != args.expected_effective_batch_size:
        raise ValueError(
            f"effective batch must be {args.expected_effective_batch_size}, got {effective_batch}"
        )
    set_seed(SEED)
    train_file = args.train_file.resolve()
    selection_file = args.selection_file.resolve() if args.selection_file else None
    selection_ids = (
        load_selection_ids(selection_file, args.selection_size)
        if selection_file is not None
        else None
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = training_label_counts(train_file, selection_ids)
    target_categories = set(label_counts)
    if len(target_categories) != args.expected_label_count:
        raise ValueError(
            f"expected {args.expected_label_count} target labels, "
            f"found {len(target_categories)}"
        )
    selected_train_rows = sum(label_counts.values())
    label_count_min = min(label_counts.values())
    label_count_max = max(label_counts.values())
    if selection_ids is not None:
        if selected_train_rows != args.selection_size:
            raise ValueError(
                f"expected {args.selection_size} selected rows, found {selected_train_rows}"
            )
        if selected_train_rows % world_size != 0:
            raise ValueError(
                f"selection size {selected_train_rows} is not divisible by world size "
                f"{world_size}; distributed sampling would pad duplicate rows"
            )
        quotient, remainder = divmod(selected_train_rows, args.expected_label_count)
        if label_count_min != quotient or label_count_max != quotient + bool(remainder):
            raise ValueError(
                "selected training classes violate quotient/remainder balance: "
                f"min={label_count_min}, max={label_count_max}"
            )
        if sum(value == quotient + 1 for value in label_counts.values()) != remainder:
            raise ValueError("selected training class remainder allocation is invalid")
    mapping = None
    label_map_path = None
    if args.label_mode == "interest_id":
        label_map_path = args.label_map.resolve()
        mapping = load_label_mapping(
            label_map_path,
            expected_categories=target_categories,
            expected_count=args.expected_label_count,
        )
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    qwen_fast_path = None
    if model_config.model_type == "qwen3_5":
        qwen_fast_path = qwen_fast_path_status()
        if args.require_qwen_fast_path and not qwen_fast_path["available"]:
            raise RuntimeError(
                "Qwen3.5 fast path is required but unavailable: "
                + json.dumps(qwen_fast_path, sort_keys=True)
            )
    elif args.require_qwen_fast_path:
        raise ValueError("--require-qwen-fast-path is only valid for Qwen3.5 models")
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_candidate_free_sft",
        "prompt_schema_version": 3 if args.label_mode == "category" else "4-interest-id",
        "candidate_free": True,
        "assistant_only": True,
        "label_mode": args.label_mode,
        "label_count": len(target_categories),
        "label_map": str(label_map_path) if label_map_path else None,
        "label_map_sha256": sha256_file(label_map_path) if label_map_path else None,
        "label_mapping_fingerprint": mapping["mapping_fingerprint"] if mapping else None,
        "model": args.model,
        "view": args.view,
        "train_file": str(train_file),
        "train_sha256": sha256_file(train_file),
        "selection_file": str(selection_file) if selection_file else None,
        "selection_file_sha256": sha256_file(selection_file) if selection_file else None,
        "selection_size": args.selection_size,
        "selected_train_rows": selected_train_rows,
        "selected_label_count_min": label_count_min,
        "selected_label_count_max": label_count_max,
        "formal_train_rows": args.expected_train_rows,
        "sample_limit": args.limit,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": effective_batch,
        "use_kernels": args.use_kernels,
        "exclude_vision_lora": args.exclude_vision_lora,
        "require_qwen_fast_path": args.require_qwen_fast_path,
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
                "dataloader_num_workers": args.dataloader_num_workers,
                "dataloader_drop_last": False,
                "use_kernels": args.use_kernels,
                "exclude_vision_lora": args.exclude_vision_lora,
                "qwen_fast_path": qwen_fast_path,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = AmazonSftDataset(
        train_file,
        tokenizer,
        args.view,
        args.label_mode,
        mapping,
        args.max_seq_length,
        args.limit,
        selection_ids,
        args.expected_train_rows,
    )
    model = load_model(args.model, args.use_kernels)
    base_model_class = type(model).__name__
    base_parameter_tensor_count = sum(1 for _ in model.parameters())
    base_visual_parameter_tensor_count = sum(
        1
        for name, _ in model.named_parameters()
        if ".visual." in name or name.startswith("visual.")
    )
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
    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_visual_parameters = [
        name for name in trainable_parameters if ".visual." in name or name.startswith("visual.")
    ]
    if args.exclude_vision_lora and trainable_visual_parameters:
        raise ValueError(
            "exclude_vision_lora left trainable visual parameters: "
            + ", ".join(trainable_visual_parameters[:10])
        )
    if not args.ddp_find_unused_parameters and trainable_visual_parameters:
        raise ValueError(
            "ddp_find_unused_parameters=False requires excluding the unused visual LoRA modules"
        )
    if is_main_process:
        write_json(
            output_dir / "trainable_parameters.json",
            {
                "base_model_class": base_model_class,
                "base_parameter_tensor_count": base_parameter_tensor_count,
                "base_visual_parameter_tensor_count": base_visual_parameter_tensor_count,
                "count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "tensor_count": len(trainable_parameters),
                "visual_tensor_count": len(trainable_visual_parameters),
                "names": trainable_parameters,
            },
        )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
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
        dataloader_drop_last=False,
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
    metrics = {
        **result.metrics,
        **contract,
        "sample_count": len(dataset),
        "resumed_from": str(checkpoint) if checkpoint else None,
    }
    if is_main_process:
        write_json(output_dir / "train_metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()