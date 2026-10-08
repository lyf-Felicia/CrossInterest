#!/usr/bin/env python3
"""Train MeRA on official or unified Amazon data and paper-model references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME

from crossinterest.baselines.mera.core import relation_infonce, sample_relation_rows, target_frequency_summary, validate_label_space, write_json
from crossinterest.baselines.mera.data import MeRACollator, MeRADataset, load_home_rows, load_official_rows, uniform_target_resample, validate_rows
from crossinterest.baselines.mera.modeling import category_representations, category_token_mapping, initialize_category_tokens, lora_linear_targets, register_category_tokens


MODELS = {
    "qwen2_1_5b": "Qwen/Qwen2-1.5B",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "llama31_8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}
SEED = 2024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODELS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", choices=("arts", "grocery", "home"), required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--expected-effective-batch-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--structure-weight", type=float, default=0.05)
    parser.add_argument("--collaborative-weight", type=float, default=0.1)
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument(
        "--balance-strategy",
        choices=("none", "uniform_target_resample"),
        default="none",
    )
    parser.add_argument("--relation-temperature", type=float, default=1.0)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument(
        "--use-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_rows(path: Path, dataset: str) -> list[dict]:
    return (
        load_home_rows(path, training=True)
        if dataset == "home"
        else load_official_rows(path)
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if (path / "trainer_state.json").is_file():
            checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


class MeRATrainer(Trainer):
    def __init__(
        self,
        *args,
        category_token_ids: list[int],
        structure_weights: torch.Tensor,
        collaborative_weights: torch.Tensor,
        structure_weight: float,
        collaborative_weight: float,
        negative_count: int,
        relation_temperature: float,
        seed: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.category_token_ids = category_token_ids
        self.structure_weights = structure_weights
        self.collaborative_weights = collaborative_weights
        self.structure_weight = structure_weight
        self.collaborative_weight = collaborative_weight
        self.negative_count = negative_count
        self.relation_temperature = relation_temperature
        self.relation_generator = torch.Generator().manual_seed(seed)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _relation_loss(self, model, weights: torch.Tensor) -> torch.Tensor:
        rows = sample_relation_rows(
            weights,
            self.negative_count,
            self.relation_generator,
        )
        device = next(model.parameters()).device
        token_ids = torch.tensor(self.category_token_ids, dtype=torch.long, device=device)
        rows = rows.to(device)
        input_rows, output_rows = category_representations(model, token_ids)
        return relation_infonce(
            input_rows,
            rows,
            self.relation_temperature,
        ) + relation_infonce(
            output_rows,
            rows,
            self.relation_temperature,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        class_labels = inputs.pop("labels")
        outputs = model(**inputs)
        last_positions = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
        next_logits = outputs.logits[batch_indices, last_positions]
        token_ids = torch.tensor(
            self.category_token_ids,
            dtype=torch.long,
            device=class_labels.device,
        )
        target_ids = token_ids[class_labels]
        language_loss = functional.cross_entropy(next_logits, target_ids)
        loss = language_loss
        if self.structure_weight > 0:
            loss = loss + self.structure_weight * self._relation_loss(
                model,
                self.structure_weights,
            )
        if self.collaborative_weight > 0:
            loss = loss + self.collaborative_weight * self._relation_loss(
                model,
                self.collaborative_weights,
            )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    model_name = args.model or MODELS[args.model_key]
    default_epochs = 1.0 if args.dataset == "home" else 10.0
    epochs = args.epochs if args.epochs is not None else default_epochs
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    effective_batch = (
        world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    )
    if effective_batch != args.expected_effective_batch_size:
        raise ValueError(
            f"expected effective batch {args.expected_effective_batch_size}, "
            f"found {effective_batch}"
        )
    if args.dataset != "home" and args.valid_file is None:
        raise ValueError("Arts/Grocery require a validation file for early stopping")
    if args.model_key == "qwen2_1_5b" and args.dataset == "home":
        raise ValueError("the Qwen2 paper-model reference is limited to official data")
    set_seed(SEED)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_space_path = args.assets_dir / "label_space.json"
    label_space = validate_label_space(
        json.loads(label_space_path.read_text(encoding="utf-8"))
    )
    train_rows = load_rows(args.train_file.resolve(), args.dataset)
    validate_rows(
        train_rows,
        set(label_space["labels"]),
        single_target=True,
        prediction_labels=set(label_space["prediction_labels"]),
    )
    valid_rows = (
        load_rows(args.valid_file.resolve(), args.dataset)
        if args.valid_file is not None
        else None
    )
    if valid_rows is not None:
        validate_rows(
            valid_rows,
            set(label_space["labels"]),
            single_target=True,
            prediction_labels=set(label_space["prediction_labels"]),
        )
    train_distribution_before_balance = target_frequency_summary(
        train_rows,
        label_space["prediction_labels"],
    )
    if args.balance_strategy == "uniform_target_resample":
        train_rows = uniform_target_resample(train_rows, SEED)
    train_distribution_after_balance = target_frequency_summary(
        train_rows,
        label_space["prediction_labels"],
    )
    if args.limit is not None:
        train_rows = train_rows[: args.limit]
        if valid_rows is not None:
            valid_rows = valid_rows[: min(args.limit, len(valid_rows))]
    effective_train_distribution = target_frequency_summary(
        train_rows,
        label_space["prediction_labels"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
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
    if args.use_kernels and args.model_key != "qwen35_9b":
        raise ValueError("--use-kernels is reserved for the Qwen3.5 main model")
    model_options = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.use_kernels:
        model_options["use_kernels"] = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=args.model_revision,
        **model_options,
    )
    semantic_setup = initialize_category_tokens(
        model,
        tokenizer,
        label_space,
        registration,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    target_modules = lora_linear_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            trainable_token_indices=registration["token_ids"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    train_dataset = MeRADataset(
        train_rows,
        tokenizer,
        args.model_key,
        label_space,
        runtime_category_tokens,
        args.max_length,
    )
    valid_dataset = (
        MeRADataset(
            valid_rows,
            tokenizer,
            args.model_key,
            label_space,
            runtime_category_tokens,
            args.max_length,
        )
        if valid_rows is not None
        else None
    )
    structure = torch.tensor(
        np.load(args.assets_dir / "structure_mask.npy"),
        dtype=torch.float32,
    )
    collaborative = torch.tensor(
        np.load(args.assets_dir / "collaborative_weights.npy"),
        dtype=torch.float32,
    )
    contract = {
        "schema_version": 1,
        "method": (
            "MeRA paper-model reference"
            if args.model_key == "qwen2_1_5b"
            else "MeRA adapted to project main models"
        ),
        "dataset": args.dataset,
        "model_key": args.model_key,
        "model": model_name,
        "model_revision": args.model_revision,
        "train_file": str(args.train_file.resolve()),
        "train_sha256": sha256_file(args.train_file.resolve()),
        "valid_file": str(args.valid_file.resolve()) if args.valid_file else None,
        "valid_sha256": sha256_file(args.valid_file.resolve()) if args.valid_file else None,
        "label_space_fingerprint": label_space["fingerprint"],
        "label_count": label_space["label_count"],
        "prediction_label_count": label_space["prediction_label_count"],
        "category_token_registration": registration,
        "category_token_style": token_style,
        "prompt_style": (
            "official Alpaca training prompt and official token-only inference prompt"
            if args.model_key == "qwen2_1_5b"
            else "model-native chat template with unified project instruction"
        ),
        "semantic_initialization": semantic_setup["initialization"],
        "history_representation": "same atomic category tokens as targets",
        "language_loss": "full-vocabulary next-token cross entropy",
        "relation_modules": ["input_embeddings", "output_lm_head"],
        "structure_weight": args.structure_weight,
        "collaborative_weight": args.collaborative_weight,
        "negative_count": args.negative_count,
        "relation_temperature": args.relation_temperature,
        "balance_strategy": args.balance_strategy,
        "balance_preserves_training_row_count": True,
        "train_distribution_before_balance": train_distribution_before_balance,
        "train_distribution_after_balance": train_distribution_after_balance,
        "effective_train_distribution": effective_train_distribution,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": effective_batch,
        "max_length": args.max_length,
        "truncation": False,
        "precision": "bf16",
        "use_kernels": args.use_kernels,
        "lora": {
            "r": 16,
            "alpha": args.lora_alpha,
            "dropout": 0.05,
            "target": "all linear layers including the output LM head",
            "target_module_count": len(target_modules),
        },
        "trainable_embeddings": "new category-token input rows only",
        "trainable_output_head": "LoRA, including exact effective rows in relation loss",
        "checkpoint_embedding_save": "sparse trainable_tokens_delta only",
        "seed": SEED,
        "sample_limit": args.limit,
    }
    contract_path = output_dir / "train_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError(f"{output_dir}: existing training contract differs")
    if rank == 0:
        write_json(contract_path, contract)
        write_json(output_dir / "semantic_token_setup.json", semantic_setup)
    has_eval = valid_dataset is not None
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = MeRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=MeRACollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if has_eval else [],
        category_token_ids=registration["token_ids"],
        structure_weights=structure,
        collaborative_weights=collaborative,
        structure_weight=args.structure_weight,
        collaborative_weight=args.collaborative_weight,
        negative_count=args.negative_count,
        relation_temperature=args.relation_temperature,
        seed=SEED,
    )
    checkpoint = latest_checkpoint(output_dir) if args.resume else None
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(final_adapter)
    if rank == 0:
        tokenizer.save_pretrained(final_adapter)
        write_json(
            output_dir / "train_metrics.json",
            {**contract, **result.metrics, "resumed_from": str(checkpoint) if checkpoint else None},
        )


if __name__ == "__main__":
    main()
