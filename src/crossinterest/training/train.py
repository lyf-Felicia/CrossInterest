#!/usr/bin/env python3
"""Train one Amazon prompt-v3 view with candidate-free assistant-only LoRA SFT."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace
from crossinterest.paths import dataset_dir

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file as save_safetensors
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME

try:
    from crossinterest.training.prompts import INPUT_TYPES, render_training_messages
    from crossinterest.training.label_space import INTEREST_ID_LABEL_MODES, LABEL_MODES, load_label_mapping, prompt_schema_version
    from crossinterest.training.semantic_tokens import initialize_semantic_token_embeddings, register_semantic_tokens, semantic_lora_target_modules
except ImportError:
    from prompts import INPUT_TYPES, render_training_messages
    from label_space import (
        INTEREST_ID_LABEL_MODES,
        LABEL_MODES,
        load_label_mapping,
        prompt_schema_version,
    )
    from semantic_tokens import (
        initialize_semantic_token_embeddings,
        register_semantic_tokens,
        semantic_lora_target_modules,
    )


SEED = 20260825
EXPECTED_TRAIN_ROWS = 112_000
EXPECTED_LABEL_COUNT = 112
HERE = _source_workspace('eval_amazon_sft_112')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="category")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=dataset_dir() / "interest_id_semantic_mapping.json",
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--selection-size", type=int)
    parser.add_argument("--selection-weight-column")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--expected-label-count", type=int, default=EXPECTED_LABEL_COUNT)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode",
        choices=("causal_lm", "label_space", "lambdarank", "hybrid"),
        default="causal_lm",
    )
    parser.add_argument("--label-space-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-label-token-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-space-negative-count",
        type=int,
        default=0,
        help="Use only the top-k hardest wrong labels; 0 uses all labels",
    )
    parser.add_argument("--lambdarank-cutoff", type=int, default=20)
    parser.add_argument(
        "--label-score-head",
        choices=("lm_head", "classifier", "residual_classifier"),
        default="lm_head",
    )
    parser.add_argument(
        "--classifier-initialization",
        choices=("lm_head", "xavier", "zero"),
        default="lm_head",
    )
    parser.add_argument(
        "--classifier-feature-normalization",
        choices=("none", "l2"),
        default="none",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
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


def render_chat_template(tokenizer, messages: list[dict]) -> str:
    options = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **options,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **options)


def tokenize_multiturn_assistant_only(
    tokenizer,
    messages: list[dict],
) -> dict[str, list[int]]:
    rendered = render_chat_template(tokenizer, messages)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    expected_ids = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    if input_ids != expected_ids:
        raise ValueError("rendered chat tokenization differs from apply_chat_template")
    labels = [-100] * len(input_ids)
    previous_end = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        variant = [dict(item) for item in messages]
        variant[index]["content"] = f"@@ASSISTANT_CONTENT_{index}@@"
        sentinel_rendered = render_chat_template(tokenizer, variant)
        start = 0
        shared_limit = min(len(rendered), len(sentinel_rendered))
        while start < shared_limit and rendered[start] == sentinel_rendered[start]:
            start += 1
        shared_suffix = 0
        suffix_limit = min(len(rendered) - start, len(sentinel_rendered) - start)
        while (
            shared_suffix < suffix_limit
            and rendered[-shared_suffix - 1] == sentinel_rendered[-shared_suffix - 1]
        ):
            shared_suffix += 1
        end = len(rendered) - shared_suffix
        if rendered[start:end] != message["content"]:
            raise ValueError("chat template does not preserve assistant content verbatim")
        if start < previous_end:
            raise ValueError("assistant content spans overlap or are out of order")
        selected = [
            position
            for position, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        ]
        if not selected:
            raise ValueError("assistant content maps to no tokens")
        for position in selected:
            labels[position] = input_ids[position]
        eos_position = selected[-1] + 1
        if (
            eos_position < len(input_ids)
            and input_ids[eos_position] == tokenizer.eos_token_id
        ):
            labels[eos_position] = input_ids[eos_position]
        previous_end = end
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def tokenize_assistant_only(tokenizer, messages: list[dict]) -> dict[str, list[int]]:
    assistant_turns = sum(message["role"] == "assistant" for message in messages)
    if assistant_turns > 1:
        return tokenize_multiturn_assistant_only(tokenizer, messages)
    complete = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    labels = [-100] * len(complete)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = apply_chat_template(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
        )
        turn_complete = apply_chat_template(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
        )
        if turn_complete[: len(prefix)] != prefix:
            raise ValueError("assistant turn does not begin with its generation prefix")
        if complete[: len(turn_complete)] != turn_complete:
            raise ValueError("full chat does not preserve an assistant turn prefix")
        labels[len(prefix) : len(turn_complete)] = turn_complete[len(prefix) :]
    if assistant_turns == 0:
        raise ValueError("training conversation contains no assistant turn")
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
        selection_weights: dict[str, float] | None,
        expected_train_rows: int,
    ):
        parquet = pq.ParquetFile(path)
        if limit is None and parquet.metadata.num_rows != expected_train_rows:
            raise ValueError(
                f"formal training requires {expected_train_rows} rows, found "
                f"{parquet.metadata.num_rows}"
            )
        columns = [
                "id",
                "historical_titles",
                "historical_interests",
                "full_seen_interests",
                "target_interest",
            ]
        if "history_summary" in parquet.schema.names:
            columns.append("history_summary")
        if "target_rationale" in parquet.schema.names:
            columns.append("target_rationale")
        if "target_reasoning" in parquet.schema.names:
            columns.append("target_reasoning")
        table = pq.read_table(
            path,
            columns=columns,
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
        self.selection_weights = selection_weights

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
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
        if self.selection_weights is not None:
            encoded["sample_weight"] = self.selection_weights[row["id"]]
        return encoded


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
        tensors = {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }
        weighted = ["sample_weight" in feature for feature in features]
        if any(weighted) and not all(weighted):
            raise ValueError("sample weights must be present for every batch item")
        if all(weighted):
            tensors["sample_weight"] = torch.tensor(
                [feature["sample_weight"] for feature in features],
                dtype=torch.float32,
            )
        return tensors


class AmazonSftTrainer(Trainer):
    def __init__(
        self,
        *args,
        sparse_embedding_save: bool = False,
        loss_mode: str = "causal_lm",
        label_token_ids: list[int] | None = None,
        label_space_loss_weight: float = 1.0,
        semantic_label_token_weight: float = 1.0,
        label_space_negative_count: int = 0,
        lambdarank_cutoff: int = 20,
        label_classifier: torch.nn.Linear | None = None,
        label_score_head: str = "lm_head",
        classifier_feature_normalization: str = "none",
        label_classifier_metadata: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sparse_embedding_save = sparse_embedding_save
        self.loss_mode = loss_mode
        self.label_token_ids = label_token_ids
        self.label_space_loss_weight = label_space_loss_weight
        self.semantic_label_token_weight = semantic_label_token_weight
        self.label_space_negative_count = label_space_negative_count
        self.lambdarank_cutoff = lambdarank_cutoff
        self.label_classifier = label_classifier
        self.label_score_head = label_score_head
        self.classifier_feature_normalization = classifier_feature_normalization
        self.label_classifier_metadata = label_classifier_metadata

    def _causal_lm_loss_per_sample(self, logits, labels) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
        token_mask = shift_labels.ne(-100)
        token_weights = token_mask.to(token_loss.dtype)
        if self.semantic_label_token_weight != 1:
            if not self.label_token_ids:
                raise ValueError("semantic label weighting requires label token IDs")
            label_ids = torch.tensor(
                self.label_token_ids,
                dtype=torch.long,
                device=shift_labels.device,
            )
            semantic_positions = shift_labels.unsqueeze(-1).eq(label_ids).any(dim=-1)
            counts = semantic_positions.sum(dim=1)
            if not torch.all(counts.eq(1)):
                raise ValueError(
                    "semantic label weighting requires exactly one semantic target "
                    f"token per sample; found {counts.tolist()}"
                )
            token_weights = torch.where(
                semantic_positions,
                torch.full_like(token_weights, self.semantic_label_token_weight),
                token_weights,
            )
        return (token_loss * token_weights).sum(dim=1) / token_weights.sum(
            dim=1
        ).clamp_min(1)

    def _label_space_logits_and_targets(
        self,
        logits,
        labels,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.label_token_ids:
            raise ValueError("label-space loss requires semantic label token IDs")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        label_ids = torch.tensor(
            self.label_token_ids,
            dtype=torch.long,
            device=shift_labels.device,
        )
        label_positions = shift_labels.unsqueeze(-1).eq(label_ids).any(dim=-1)
        counts = label_positions.sum(dim=1)
        if not torch.all(counts.eq(1)):
            raise ValueError(
                "label-space loss requires exactly one semantic target token per sample; "
                f"found {counts.tolist()}"
            )
        target_token_ids = shift_labels[label_positions]
        target_matches = target_token_ids.unsqueeze(-1).eq(label_ids)
        if not torch.all(target_matches.sum(dim=1).eq(1)):
            raise AssertionError("semantic target token did not map to exactly one label")
        target_indices = target_matches.to(torch.int64).argmax(dim=1)
        label_logits = shift_logits[label_positions]
        if self.label_classifier is None:
            label_logits = label_logits.index_select(-1, label_ids)
        elif label_logits.shape[1] != len(self.label_token_ids):
            raise ValueError("classifier output size differs from semantic label count")
        return label_logits, target_indices, target_matches

    def _label_space_loss_per_sample(self, logits, labels) -> torch.Tensor:
        label_logits, target_indices, target_matches = (
            self._label_space_logits_and_targets(logits, labels)
        )
        if self.label_space_negative_count:
            if self.label_space_negative_count >= label_logits.shape[1]:
                raise ValueError(
                    "label-space negative count must be smaller than the label count"
                )
            negative_logits = label_logits.masked_fill(target_matches, -torch.inf)
            hard_negatives = negative_logits.topk(
                self.label_space_negative_count,
                dim=1,
            ).values
            target_logits = label_logits.gather(1, target_indices.unsqueeze(1))
            label_logits = torch.cat((target_logits, hard_negatives), dim=1)
            target_indices = torch.zeros_like(target_indices)
        return F.cross_entropy(label_logits, target_indices, reduction="none")

    def _lambdarank_loss_per_sample(self, logits, labels) -> torch.Tensor:
        label_logits, target_indices, target_matches = (
            self._label_space_logits_and_targets(logits, labels)
        )
        label_count = label_logits.shape[1]
        if not 0 < self.lambdarank_cutoff < label_count:
            raise ValueError("LambdaRank cutoff must be between 1 and label_count - 1")
        target_logits = label_logits.gather(1, target_indices.unsqueeze(1))
        negative_logits = label_logits.masked_select(~target_matches).view(
            label_logits.shape[0],
            label_count - 1,
        )
        sorted_indices = label_logits.detach().argsort(dim=1, descending=True)
        ranks = torch.empty_like(sorted_indices)
        rank_values = torch.arange(
            1,
            label_count + 1,
            dtype=sorted_indices.dtype,
            device=sorted_indices.device,
        ).expand_as(sorted_indices)
        ranks.scatter_(1, sorted_indices, rank_values)
        target_ranks = ranks.gather(1, target_indices.unsqueeze(1))
        negative_ranks = ranks.masked_select(~target_matches).view_as(negative_logits)

        def discount(rank: torch.Tensor) -> torch.Tensor:
            value = torch.reciprocal(torch.log2(rank.to(torch.float32) + 1))
            return torch.where(rank <= self.lambdarank_cutoff, value, 0.0)

        delta_ndcg = (discount(target_ranks) - discount(negative_ranks)).abs()
        pairwise_loss = F.softplus(negative_logits - target_logits)
        return (delta_ndcg * pairwise_loss).sum(dim=1) / delta_ndcg.sum(
            dim=1
        ).clamp_min(torch.finfo(delta_ndcg.dtype).eps)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        sample_weight = inputs.pop("sample_weight", None)
        if self.loss_mode == "causal_lm" and sample_weight is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs.pop("labels")
        outputs = model(
            **inputs,
            output_hidden_states=self.label_classifier is not None,
        )
        if self.label_classifier is not None:
            classifier_features = outputs.hidden_states[-1]
            if self.classifier_feature_normalization == "l2":
                classifier_features = F.normalize(
                    classifier_features.to(torch.float32),
                    p=2,
                    dim=-1,
                )
            label_score_logits = self.label_classifier(classifier_features)
        else:
            label_score_logits = outputs.logits
        if self.label_score_head == "residual_classifier":
            label_ids = torch.tensor(
                self.label_token_ids,
                dtype=torch.long,
                device=outputs.logits.device,
            )
            label_score_logits = label_score_logits + outputs.logits.index_select(
                -1,
                label_ids,
            )
        if self.loss_mode == "causal_lm":
            per_sample_loss = self._causal_lm_loss_per_sample(outputs.logits, labels)
        elif self.loss_mode == "label_space":
            per_sample_loss = self._label_space_loss_per_sample(label_score_logits, labels)
        elif self.loss_mode == "lambdarank":
            per_sample_loss = self._lambdarank_loss_per_sample(label_score_logits, labels)
        elif self.loss_mode == "hybrid":
            per_sample_loss = self._causal_lm_loss_per_sample(outputs.logits, labels)
            per_sample_loss = per_sample_loss + self.label_space_loss_weight * (
                self._label_space_loss_per_sample(label_score_logits, labels)
            )
        else:
            raise ValueError(f"unknown loss mode: {self.loss_mode}")
        if sample_weight is not None:
            per_sample_loss = per_sample_loss * sample_weight.to(per_sample_loss.dtype)
        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        if not self.sparse_embedding_save:
            super()._save(output_dir, state_dict)
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            save_embedding_layers=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        if self.label_classifier is not None:
            save_safetensors(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in self.label_classifier.state_dict().items()
                },
                os.path.join(output_dir, "label_classifier.safetensors"),
            )
            write_json(
                Path(output_dir) / "label_classifier.json",
                self.label_classifier_metadata,
            )
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / "trainer_state.json").is_file():
            continue
        try:
            checkpoints.append((int(path.name.removeprefix("checkpoint-")), path))
        except ValueError:
            continue
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


def build_label_classifier(
    output_weight: torch.Tensor,
    label_token_ids: list[int],
    initialization: str,
) -> torch.nn.Linear:
    classifier = torch.nn.Linear(
        output_weight.shape[1],
        len(label_token_ids),
        bias=True,
        dtype=torch.float32,
    )
    with torch.no_grad():
        if initialization == "lm_head":
            classifier.weight.copy_(
                output_weight.index_select(
                    0,
                    torch.tensor(label_token_ids, device=output_weight.device),
                ).to(torch.float32)
            )
        elif initialization == "xavier":
            torch.nn.init.xavier_uniform_(classifier.weight)
        elif initialization == "zero":
            classifier.weight.zero_()
        else:
            raise ValueError(f"unknown classifier initialization: {initialization}")
        classifier.bias.zero_()
    return classifier


def attach_or_load_lora(model, config: LoraConfig, init_adapter: Path | None):
    if init_adapter is None:
        return get_peft_model(model, config)
    return PeftModel.from_pretrained(
        model,
        str(init_adapter),
        is_trainable=True,
    )


def load_selection(
    path: Path,
    selection_size: int,
    weight_column: str | None,
) -> tuple[set[str], dict[str, float] | None]:
    if selection_size <= 0:
        raise ValueError("--selection-size must be positive")
    columns = ["id", "nested_rank"]
    if weight_column is not None:
        columns.append(weight_column)
    table = pq.read_table(path, columns=columns)
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
    if weight_column is None:
        return selected, None
    weights = {
        sample_id: float(weight)
        for sample_id, rank, weight in zip(
            table["id"].to_pylist(),
            ranks,
            table[weight_column].to_pylist(),
            strict=True,
        )
        if rank <= selection_size
    }
    values = torch.tensor(list(weights.values()), dtype=torch.float64)
    if not torch.isfinite(values).all() or torch.any(values <= 0):
        raise ValueError("selected sample weights must be finite and positive")
    if not torch.isclose(values.mean(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6):
        raise ValueError("selected sample weights must have mean 1")
    return selected, weights


def training_label_counts(
    train_file: Path,
    selection_ids: set[str] | None,
) -> Counter[str]:
    label_table = pq.read_table(train_file, columns=["id", "target_interest"])
    return Counter(
        category
        for sample_id, category in zip(
            label_table["id"].to_pylist(),
            label_table["target_interest"].to_pylist(),
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
    if args.selection_weight_column is not None and args.selection_file is None:
        raise ValueError("--selection-weight-column requires --selection-file")
    if args.label_space_loss_weight <= 0:
        raise ValueError("--label-space-loss-weight must be positive")
    if args.semantic_label_token_weight <= 0:
        raise ValueError("--semantic-label-token-weight must be positive")
    if args.semantic_label_token_weight != 1 and args.label_mode != "interest_id":
        raise ValueError("semantic label token weighting requires interest_id mode")
    if args.label_space_negative_count < 0:
        raise ValueError("--label-space-negative-count cannot be negative")
    if args.label_space_negative_count >= args.expected_label_count:
        raise ValueError("--label-space-negative-count must be smaller than label count")
    if not 0 < args.lambdarank_cutoff < args.expected_label_count:
        raise ValueError("--lambdarank-cutoff must be between 1 and label count - 1")
    if args.loss_mode != "causal_lm" and args.label_mode != "interest_id":
        raise ValueError("label-space losses require single-token Interest-ID output")
    if args.label_score_head != "lm_head" and args.loss_mode == "causal_lm":
        raise ValueError("classifier label head requires a label-space loss mode")
    if args.label_score_head == "lm_head" and args.classifier_initialization != "lm_head":
        raise ValueError("--classifier-initialization requires --label-score-head classifier")
    if args.label_score_head == "lm_head" and args.classifier_feature_normalization != "none":
        raise ValueError(
            "--classifier-feature-normalization requires a classifier label head"
        )
    if args.label_score_head == "residual_classifier" and args.classifier_initialization != "zero":
        raise ValueError("residual classifier requires --classifier-initialization zero")
    if args.freeze_backbone and args.label_score_head == "lm_head":
        raise ValueError("--freeze-backbone requires a classifier label head")
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
    train_schema = pq.ParquetFile(train_file).schema.names
    has_history_summary = "history_summary" in train_schema
    has_target_rationale = "target_rationale" in train_schema
    has_target_reasoning = "target_reasoning" in train_schema
    if has_target_rationale and has_target_reasoning:
        raise ValueError("training data cannot contain both rationale order columns")
    init_adapter = args.init_adapter.resolve() if args.init_adapter else None
    if init_adapter is not None:
        for required_file in ("adapter_config.json", "adapter_model.safetensors"):
            if not (init_adapter / required_file).is_file():
                raise ValueError(f"init adapter is missing {required_file}: {init_adapter}")
    selection_file = args.selection_file.resolve() if args.selection_file else None
    selection_ids, selection_weights = (
        load_selection(
            selection_file,
            args.selection_size,
            args.selection_weight_column,
        )
        if selection_file is not None
        else (None, None)
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
    weight_values = (
        torch.tensor(list(selection_weights.values()), dtype=torch.float64)
        if selection_weights is not None
        else None
    )
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
        if label_count_max - label_count_min > 1:
            raise ValueError(
                "selected training classes are imbalanced: "
                f"min={label_count_min}, max={label_count_max}"
            )
    mapping = None
    label_map_path = None
    if args.label_mode in INTEREST_ID_LABEL_MODES:
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    semantic_registration = (
        register_semantic_tokens(tokenizer, mapping) if mapping is not None else None
    )
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_candidate_free_sft",
        "prompt_schema_version": prompt_schema_version(
            args.label_mode,
            history_summary=has_history_summary,
            target_rationale=has_target_rationale,
            target_reasoning=has_target_reasoning,
        ),
        "history_summary": has_history_summary,
        "target_rationale_auxiliary": has_target_rationale,
        "reasoning_first": has_target_reasoning,
        "assistant_turn_count": (
            2 if has_target_rationale or has_target_reasoning else 1
        ),
        "candidate_free": True,
        "assistant_only": True,
        "label_mode": args.label_mode,
        "label_count": len(target_categories),
        "label_map": str(label_map_path) if label_map_path else None,
        "label_map_sha256": sha256_file(label_map_path) if label_map_path else None,
        "label_mapping_fingerprint": mapping["mapping_fingerprint"] if mapping else None,
        "semantic_token_registration": semantic_registration,
        "semantic_token_initialization": (
            "mean base-model rows of child-label text tokens only" if mapping else None
        ),
        "semantic_output_training": "lm_head LoRA" if mapping else None,
        "model": args.model,
        "init_adapter": str(init_adapter) if init_adapter else None,
        "init_adapter_config_sha256": (
            sha256_file(init_adapter / "adapter_config.json") if init_adapter else None
        ),
        "init_adapter_model_sha256": (
            sha256_file(init_adapter / "adapter_model.safetensors") if init_adapter else None
        ),
        "view": args.view,
        "train_file": str(train_file),
        "train_sha256": sha256_file(train_file),
        "selection_file": str(selection_file) if selection_file else None,
        "selection_file_sha256": sha256_file(selection_file) if selection_file else None,
        "selection_size": args.selection_size,
        "selection_weight_column": args.selection_weight_column,
        "selection_weight_mean": (
            float(weight_values.mean()) if weight_values is not None else None
        ),
        "selection_weight_min": (
            float(weight_values.min()) if weight_values is not None else None
        ),
        "selection_weight_max": (
            float(weight_values.max()) if weight_values is not None else None
        ),
        "selection_weight_effective_sample_size": (
            float(weight_values.sum().square() / weight_values.square().sum())
            if weight_values is not None
            else None
        ),
        "selected_train_rows": selected_train_rows,
        "selected_label_count_min": label_count_min,
        "selected_label_count_max": label_count_max,
        "selected_label_count_max_gap": label_count_max - label_count_min,
        "formal_train_rows": args.expected_train_rows,
        "sample_limit": args.limit,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss_mode": args.loss_mode,
        "label_space_loss_weight": args.label_space_loss_weight,
        "semantic_label_token_weight": args.semantic_label_token_weight,
        "label_space_negative_count": args.label_space_negative_count,
        "lambdarank_cutoff": args.lambdarank_cutoff,
        "label_score_head": args.label_score_head,
        "classifier_initialization": (
            args.classifier_initialization
            if args.label_score_head != "lm_head"
            else None
        ),
        "classifier_feature_normalization": (
            args.classifier_feature_normalization
            if args.label_score_head != "lm_head"
            else None
        ),
        "freeze_backbone": args.freeze_backbone,
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

    dataset = AmazonSftDataset(
        train_file,
        tokenizer,
        args.view,
        args.label_mode,
        mapping,
        args.max_seq_length,
        args.limit,
        selection_ids,
        selection_weights,
        args.expected_train_rows,
    )
    model = load_model(args.model, args.use_kernels)
    semantic_metadata = None
    if mapping is not None:
        semantic_metadata = initialize_semantic_token_embeddings(
            model,
            tokenizer,
            mapping,
            semantic_registration,
        )
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
    if mapping is None:
        target_modules = "all-linear"
        exclude_modules = r"(?:.*\.)?visual\..*" if args.exclude_vision_lora else None
        trainable_token_indices = None
    else:
        target_modules = semantic_lora_target_modules(
            model,
            args.exclude_vision_lora,
            include_output_embeddings=args.label_score_head == "lm_head",
        )
        exclude_modules = None
        trainable_token_indices = [
            semantic_registration["token_ids"][token]
            for token in mapping["special_tokens"]
        ]
        if is_main_process:
            write_json(
                output_dir / "semantic_token_setup.json",
                {
                    **semantic_metadata,
                    "lora_target_modules": target_modules,
                    "output_embedding_training": "LoRA",
                },
            )
    model = attach_or_load_lora(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
            bias="none",
            trainable_token_indices=trainable_token_indices,
        ),
        init_adapter,
    )
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    label_classifier = None
    label_classifier_metadata = None
    if args.label_score_head != "lm_head":
        label_token_ids = [
            semantic_registration["token_ids"][token]
            for token in mapping["special_tokens"]
        ]
        output_weight = model.get_output_embeddings().weight.detach()
        label_classifier = build_label_classifier(
            output_weight,
            label_token_ids,
            args.classifier_initialization,
        )
        model.add_module("label_classifier", label_classifier)
        label_classifier_metadata = {
            "schema_version": 1,
            "label_mapping_fingerprint": mapping["mapping_fingerprint"],
            "label_count": len(label_token_ids),
            "label_tokens": mapping["special_tokens"],
            "hidden_size": output_weight.shape[1],
            "head_type": args.label_score_head,
            "initialization": args.classifier_initialization,
            "feature_normalization": args.classifier_feature_normalization,
        }
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
    trainer = AmazonSftTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        sparse_embedding_save=mapping is not None,
        loss_mode=args.loss_mode,
        label_token_ids=(
            [
                semantic_registration["token_ids"][token]
                for token in mapping["special_tokens"]
            ]
            if mapping is not None
            else None
        ),
        label_space_loss_weight=args.label_space_loss_weight,
        semantic_label_token_weight=args.semantic_label_token_weight,
        label_space_negative_count=args.label_space_negative_count,
        lambdarank_cutoff=args.lambdarank_cutoff,
        label_classifier=label_classifier,
        label_score_head=args.label_score_head,
        classifier_feature_normalization=args.classifier_feature_normalization,
        label_classifier_metadata=label_classifier_metadata,
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