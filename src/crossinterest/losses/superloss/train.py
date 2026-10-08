#!/usr/bin/env python3
"""Train the controlled SFT or sequence-level SuperLoss LoRA condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset as ArrowDataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from prompts import VIEWS, render_messages
from superloss import sequence_answer_loss, superloss_confidence


SEED = 20260825


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sft", "superloss"), required=True)
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path(__file__).parents[1] / "mind/balanced/train.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--expected-effective-batch-size", type=int, default=32)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--superloss-lambda", type=float, default=1.0)
    parser.add_argument("--superloss-ema-beta", type=float, default=0.99)
    parser.add_argument("--superloss-min-weight", type=float, default=0.0)
    parser.add_argument("--uniform-warmup-fraction", type=float, default=0.05)
    parser.add_argument("--uniform-final-fraction", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_or_validate_contract(path: Path, value: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"{path}: existing training configuration differs")
        return
    write_json(path, value)


def template_ids(value) -> list[int]:
    return list(value["input_ids"] if hasattr(value, "keys") else value)


def tokenize_assistant_only(tokenizer, messages: list[dict]) -> dict[str, list[int]]:
    options = {"enable_thinking": False}
    prefix = template_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True, **options
        )
    )
    complete = template_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True, **options
        )
    )
    if complete[: len(prefix)] != prefix:
        raise ValueError("full chat does not begin with the assistant generation prefix")
    labels = [-100] * len(prefix) + complete[len(prefix) :]
    if not any(label != -100 for label in labels):
        raise ValueError("assistant-only mask contains no trainable tokens")
    return {"input_ids": complete, "labels": labels, "attention_mask": [1] * len(complete)}


class SftDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        view: str,
        max_length: int,
        limit: int | None,
    ) -> None:
        self.dataset = ArrowDataset.from_parquet(
            str(path),
            columns=["sample_id", "history", "target_subcategory"],
            keep_in_memory=False,
        )
        self.size = min(limit, len(self.dataset)) if limit is not None else len(self.dataset)
        self.tokenizer = tokenizer
        self.view = view
        self.max_length = max_length

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.dataset[index]
        messages = render_messages(
            row["history"], self.view, target=row["target_subcategory"]
        )
        encoded = tokenize_assistant_only(self.tokenizer, messages)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError(
                f"{row['sample_id']}: {len(encoded['input_ids'])} tokens exceed "
                f"max_seq_length={self.max_length}; tokenizer truncation is forbidden"
            )
        return encoded


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        first_supervised = min(
            next(index for index, label in enumerate(feature["labels"]) if label != -100)
            for feature in features
        )
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        tensors = {
            name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()
        }
        tensors["logits_to_keep"] = min(maximum, maximum - first_supervised + 1)
        return tensors


class CurriculumTrainer(Trainer):
    def __init__(
        self, *args, method: str, view: str, curriculum: dict, metrics_path: Path, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.method = method
        self.view = view
        self.curriculum = curriculum
        self.metrics_path = metrics_path
        self.threshold_ema: float | None = None
        self.weight_ema = 1.0
        self.pending_losses: list[float] = []
        self.pending_weights: list[float] = []
        self.observed_global_step = 0
        self.last_logged_step = -1

    def _distributed_mean(self, values: list[float]) -> float | None:
        statistics = torch.tensor(
            [sum(values), len(values)], dtype=torch.float64, device=self.accelerator.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        return (
            float((statistics[0] / statistics[1]).item())
            if statistics[1].item() > 0
            else None
        )

    def _commit_previous_step(self, new_global_step: int) -> None:
        if new_global_step == self.observed_global_step:
            return
        beta = self.curriculum["ema_beta"]
        mean_loss = self._distributed_mean(self.pending_losses)
        if mean_loss is not None:
            self.threshold_ema = (
                mean_loss
                if self.threshold_ema is None
                else beta * self.threshold_ema + (1 - beta) * mean_loss
            )
        mean_weight = self._distributed_mean(self.pending_weights)
        if mean_weight is not None:
            self.weight_ema = beta * self.weight_ema + (1 - beta) * mean_weight
        self.pending_losses.clear()
        self.pending_weights.clear()
        self.observed_global_step = new_global_step

    def _phase(self) -> str:
        progress = self.state.global_step / max(1, self.state.max_steps)
        if progress < self.curriculum["warmup_fraction"]:
            return "uniform_warmup"
        if progress >= 1 - self.curriculum["final_fraction"]:
            return "uniform_final"
        return "superloss"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        self._commit_previous_step(self.state.global_step)
        labels = inputs.pop("labels")
        logits_to_keep = inputs.pop("logits_to_keep")
        outputs = model(**inputs, logits_to_keep=logits_to_keep)
        sample_losses = sequence_answer_loss(outputs.logits, labels)
        raw_loss = sample_losses.mean()
        phase = "uniform" if self.method == "sft" else self._phase()
        weights = torch.ones_like(sample_losses)
        if phase == "superloss" and self.threshold_ema is not None:
            raw_weights = superloss_confidence(
                sample_losses,
                threshold=self.threshold_ema,
                regularization=self.curriculum["regularization"],
                minimum=self.curriculum["minimum_weight"],
            )
            weights = raw_weights / max(self.weight_ema, 1e-6)
            self.pending_weights.extend(raw_weights.detach().float().cpu().tolist())
        loss = (weights * sample_losses).mean()
        self.pending_losses.extend(sample_losses.detach().float().cpu().tolist())
        if self.last_logged_step != self.state.global_step and self.is_world_process_zero():
            record = {
                "global_step": self.state.global_step,
                "phase": phase,
                "raw_loss": float(raw_loss.detach()),
                "weighted_loss": float(loss.detach()),
                "threshold_ema": self.threshold_ema,
                "weight_ema": self.weight_ema,
                "weight_mean": float(weights.detach().mean()),
                "weight_min": float(weights.detach().min()),
                "weight_max": float(weights.detach().max()),
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.last_logged_step = self.state.global_step
        return (loss, outputs) if return_outputs else loss

    def curriculum_state(self) -> dict:
        return {
            "schema_version": 1,
            "method": self.method,
            "view": self.view,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "threshold_ema": self.threshold_ema,
            "weight_ema": self.weight_ema,
            "pending_losses": self.pending_losses,
            "pending_weights": self.pending_weights,
            "observed_global_step": self.observed_global_step,
            "last_logged_step": self.last_logged_step,
        }

    def restore_curriculum_state(self, state: dict) -> None:
        expected = {
            "method": self.method,
            "view": self.view,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"curriculum checkpoint {name}={state.get(name)!r}, expected {value!r}"
                )
        self.threshold_ema = state["threshold_ema"]
        self.weight_ema = state["weight_ema"]
        self.pending_losses = list(state["pending_losses"])
        self.pending_weights = list(state["pending_weights"])
        self.observed_global_step = state["observed_global_step"]
        self.last_logged_step = state["last_logged_step"]


class CurriculumCheckpointCallback(TrainerCallback):
    def __init__(self, trainer: CurriculumTrainer) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        rank = int(os.environ.get("RANK", "0"))
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / f"curriculum_state_rank{rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, self.trainer.curriculum_state())
        return control


def find_complete_checkpoint(output_dir: Path, world_size: int) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    for _, path in sorted(candidates, reverse=True):
        required = [path / "trainer_state.json"] + [
            path / f"curriculum_state_rank{rank}.json" for rank in range(world_size)
        ]
        if all(item.is_file() for item in required):
            return path
    return None


def truncate_metrics(path: Path, resume_step: int) -> None:
    if not path.exists():
        return
    retained = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["global_step"] < resume_step:
                retained.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(retained), encoding="utf-8")
    temporary.replace(path)


def lora_b_health(model) -> dict:
    parameters = [
        parameter.detach().float()
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]
    if not parameters:
        raise RuntimeError("model contains no LoRA-B parameters")
    nonzero = sum(torch.count_nonzero(parameter).item() for parameter in parameters)
    total = sum(parameter.numel() for parameter in parameters)
    if nonzero == 0:
        raise RuntimeError("all LoRA-B parameters remain zero after training")
    return {"lora_b_nonzero": nonzero, "lora_b_total": total}


def main() -> None:
    args = parse_args()
    effective_batch = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    if effective_batch != args.expected_effective_batch_size:
        raise ValueError(
            f"effective batch must be {args.expected_effective_batch_size}, got {effective_batch}"
        )
    if args.uniform_warmup_fraction + args.uniform_final_fraction >= 1:
        raise ValueError("uniform curriculum fractions must sum to less than one")
    set_seed(SEED)
    train_path = args.train_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SftDataset(
        train_path,
        tokenizer,
        args.view,
        args.max_seq_length,
        args.limit,
    )
    curriculum = {
        "regularization": args.superloss_lambda,
        "ema_beta": args.superloss_ema_beta,
        "minimum_weight": args.superloss_min_weight,
        "warmup_fraction": args.uniform_warmup_fraction,
        "final_fraction": args.uniform_final_fraction,
        "threshold": "previous_optimizer_steps_loss_ema",
        "normalization": "previous_optimizer_steps_raw_weight_ema",
    }
    contract = {
        "schema_version": 1,
        "method": args.method,
        "view": args.view,
        "model": args.model,
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "sample_count": len(dataset),
        "seed": SEED,
        "assistant_only": True,
        "sequence_loss": "mean_answer_token_nll",
        "sequence_loss_sha256": sha256_file(Path(__file__).with_name("superloss.py")),
        "logits_projection": "answer_suffix_only",
        "max_seq_length": args.max_seq_length,
        "prompt_candidates": False,
        "prompt_renderer_sha256": sha256_file(Path(__file__).with_name("prompts.py")),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "effective_batch_size": effective_batch,
        "expected_effective_batch_size": args.expected_effective_batch_size,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "curriculum": curriculum if args.method == "superloss" else None,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "exclude_modules": r".*\.visual\..*",
        },
        "ddp_find_unused_parameters": False,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        write_or_validate_contract(output_dir / "experiment_config.json", contract)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            exclude_modules=r".*\.visual\..*",
            task_type="CAUSAL_LM",
            bias="none",
        ),
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
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=SEED,
        data_seed=SEED,
    )
    metrics_path = output_dir / "curriculum_metrics.jsonl"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resume_checkpoint = find_complete_checkpoint(output_dir, world_size) if args.resume else None
    if int(os.environ.get("RANK", "0")) == 0:
        if resume_checkpoint is None and metrics_path.exists():
            metrics_path.unlink()
        elif resume_checkpoint is not None:
            truncate_metrics(metrics_path, int(resume_checkpoint.name.removeprefix("checkpoint-")))
    trainer = CurriculumTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        method=args.method,
        view=args.view,
        curriculum=curriculum,
        metrics_path=metrics_path,
    )
    trainer.add_callback(CurriculumCheckpointCallback(trainer))
    if resume_checkpoint is not None:
        rank = int(os.environ.get("RANK", "0"))
        state_path = resume_checkpoint / f"curriculum_state_rank{rank}.json"
        trainer.restore_curriculum_state(json.loads(state_path.read_text(encoding="utf-8")))
    result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    health = lora_b_health(model)
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_adapter)
        write_json(output_dir / "train_metrics.json", {**result.metrics, **health, **contract})


if __name__ == "__main__":
    main()