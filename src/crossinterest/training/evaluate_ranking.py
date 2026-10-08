#!/usr/bin/env python3
"""Evaluate an Amazon SFT LoRA with candidate-free direct beam generation."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace
from crossinterest.paths import dataset_dir

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

from crossinterest.evaluation.common import INPUT_TYPES, append_jsonl, latest_details, normalize_candidate_text, sha256_file, write_json_atomic
from crossinterest.training.evaluate import load_model, resolve_contract
from crossinterest.training.label_space import INTEREST_ID_PATTERN, INTEREST_ID_LABEL_MODES, LABEL_MODES, load_label_mapping, output_label, prompt_schema_version
from crossinterest.training.prompts import REASONING_PREDICTION_INSTRUCTION, render_reasoning_prompt, render_training_prompt
from crossinterest.training.semantic_tokens import register_semantic_tokens


HERE = _source_workspace('eval_amazon_sft_112')
DEFAULT_DATA_DIR = dataset_dir() / "eval"
DEFAULT_LABEL_FILE = dataset_dir() / "train.parquet"
DEFAULT_LABEL_MAP = dataset_dir() / "interest_id_semantic_mapping.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("qwen35_9b", "llama31_8b"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--view", choices=INPUT_TYPES, required=True)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="category")
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=112)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--decoding",
        choices=(
            "direct_beam_search",
            "direct_label_logits",
            "reasoning_then_label_logits",
            "semantic_classifier",
        ),
        default="direct_beam_search",
    )
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="cuda")
    return parser.parse_args(argv)


def load_labels(path: Path, expected_label_count: int = 112) -> list[str]:
    labels = sorted(set(pq.read_table(path, columns=["target_interest"])["target_interest"].to_pylist()))
    if len(labels) != expected_label_count:
        raise ValueError(
            f"expected {expected_label_count} formal Train labels, found {len(labels)}"
        )
    return labels


def exact_output_sequences(tokenizer, outputs: list[str]) -> dict[tuple[int, ...], str]:
    sequences = {}
    for output in outputs:
        token_ids = tuple(tokenizer(output, add_special_tokens=False)["input_ids"])
        if token_ids in sequences:
            raise ValueError(f"output token collision: {output}")
        sequences[token_ids] = output
    return sequences


def render_prompt_ids(
    tokenizer,
    row: dict,
    model_key: str,
    view: str,
    label_mode: str,
) -> list[int]:
    source = {**row, "id": row["source_sample_id"]}
    prompt = render_training_prompt(source, view, label_mode)
    if "Candidates:" in prompt:
        raise AssertionError("candidate-free ranking prompt contains a candidate list")
    options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
    }
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **options,
    )
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def apply_generation_template(
    tokenizer,
    messages: list[dict[str, str]],
    model_key: str,
) -> list[int]:
    options = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
    }
    if model_key == "qwen35_9b":
        options["enable_thinking"] = False
    value = tokenizer.apply_chat_template(messages, **options)
    input_ids = value["input_ids"] if hasattr(value, "keys") else value
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def render_reasoning_inputs(
    tokenizer,
    row: dict,
    model_key: str,
    view: str,
) -> tuple[str, list[int]]:
    source = {**row, "id": row["source_sample_id"]}
    prompt = render_reasoning_prompt(source, view)
    return prompt, apply_generation_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        model_key,
    )


def render_reasoning_label_prompt_ids(
    tokenizer,
    reasoning_prompt: str,
    generated_reasoning: str,
    model_key: str,
) -> list[int]:
    return apply_generation_template(
        tokenizer,
        [
            {"role": "user", "content": reasoning_prompt},
            {"role": "assistant", "content": generated_reasoning},
            {"role": "user", "content": REASONING_PREDICTION_INSTRUCTION},
        ],
        model_key,
    )


def direct_beam_rank(
    model,
    tokenizer,
    prompt_ids: list[int],
    output_to_category: dict[str, str],
    num_beams: int,
    max_new_tokens: int,
    semantic_output_sequences: dict[tuple[int, ...], str] | None = None,
) -> list[dict]:
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    label_by_normalized = (
        {label.casefold(): label for label in output_to_category}
        if semantic_output_sequences is None
        else None
    )
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ranked = []
    for rank, (sequence, score) in enumerate(
        zip(output.sequences, output.sequences_scores, strict=True),
        1,
    ):
        suffix = sequence.tolist()[len(prompt_ids) :]
        if tokenizer.eos_token_id in suffix:
            content_tokens = suffix[: suffix.index(tokenizer.eos_token_id)]
        else:
            content_tokens = suffix
        if semantic_output_sequences is None:
            raw_text = tokenizer.decode(suffix, skip_special_tokens=True).strip()
            normalized = normalize_candidate_text(raw_text)
            predicted_output = label_by_normalized.get(normalized.casefold())
            format_valid = predicted_output is not None and raw_text == predicted_output
        else:
            raw_text = tokenizer.decode(
                content_tokens,
                skip_special_tokens=False,
            ).strip()
            predicted_output = semantic_output_sequences.get(tuple(content_tokens))
            format_valid = predicted_output is not None
        prediction = output_to_category.get(predicted_output) if predicted_output else None
        ranked.append(
            {
                "rank": rank,
                "raw_text": raw_text,
                "predicted_output": predicted_output,
                "prediction": prediction,
                "format_valid": format_valid,
                "sequence_score": float(score),
            }
        )
    return ranked


def direct_label_logit_rank(
    model,
    prompt_ids: list[int],
    output_to_category: dict[str, str],
    semantic_output_sequences: dict[tuple[int, ...], str],
    num_outputs: int,
) -> list[dict]:
    return direct_label_logit_rank_batch(
        model,
        [prompt_ids],
        output_to_category,
        semantic_output_sequences,
        num_outputs,
        pad_token_id=0,
    )[0]


def padded_prompt_batch(
    prompt_ids_batch: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not prompt_ids_batch or any(not prompt_ids for prompt_ids in prompt_ids_batch):
        raise ValueError("prompt batch must contain non-empty token sequences")
    lengths = torch.tensor(
        [len(prompt_ids) for prompt_ids in prompt_ids_batch],
        dtype=torch.long,
        device=device,
    )
    maximum = int(lengths.max().item())
    inputs = torch.full(
        (len(prompt_ids_batch), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(inputs)
    for index, prompt_ids in enumerate(prompt_ids_batch):
        inputs[index, : len(prompt_ids)] = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, : len(prompt_ids)] = 1
    return inputs, attention_mask, lengths - 1


def left_padded_prompt_batch(
    prompt_ids_batch: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompt_ids_batch or any(not prompt_ids for prompt_ids in prompt_ids_batch):
        raise ValueError("prompt batch must contain non-empty token sequences")
    maximum = max(map(len, prompt_ids_batch))
    inputs = torch.full(
        (len(prompt_ids_batch), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(inputs)
    for index, prompt_ids in enumerate(prompt_ids_batch):
        inputs[index, -len(prompt_ids) :] = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[index, -len(prompt_ids) :] = 1
    return inputs, attention_mask


def rank_batch_with_oom_backoff(
    prompt_ids_batch: list[list[int]],
    rank_batch: Callable[[list[list[int]]], list[list[dict]]],
) -> list[list[dict]]:
    try:
        return rank_batch(prompt_ids_batch)
    except torch.OutOfMemoryError:
        if len(prompt_ids_batch) == 1:
            raise
    torch.cuda.empty_cache()
    midpoint = len(prompt_ids_batch) // 2
    print(
        f"CUDA OOM for batch size {len(prompt_ids_batch)}; retrying as "
        f"{midpoint}+{len(prompt_ids_batch) - midpoint}",
        flush=True,
    )
    return rank_batch_with_oom_backoff(
        prompt_ids_batch[:midpoint],
        rank_batch,
    ) + rank_batch_with_oom_backoff(
        prompt_ids_batch[midpoint:],
        rank_batch,
    )


def rank_label_scores(
    label_scores: torch.Tensor,
    token_ids: torch.Tensor,
    output_by_token_id: dict[int, str],
    output_to_category: dict[str, str],
    num_outputs: int,
) -> list[dict]:
    scores, positions = torch.topk(
        label_scores,
        k=num_outputs,
        largest=True,
        sorted=True,
    )
    ranked = []
    for rank, (position, score) in enumerate(
        zip(positions.tolist(), scores.tolist(), strict=True),
        1,
    ):
        token_id = int(token_ids[position])
        predicted_output = output_by_token_id[token_id]
        ranked.append(
            {
                "rank": rank,
                "raw_text": predicted_output,
                "predicted_output": predicted_output,
                "prediction": output_to_category[predicted_output],
                "format_valid": True,
                "sequence_score": float(score),
            }
        )
    return ranked


def direct_label_logit_rank_batch(
    model,
    prompt_ids_batch: list[list[int]],
    output_to_category: dict[str, str],
    semantic_output_sequences: dict[tuple[int, ...], str],
    num_outputs: int,
    pad_token_id: int,
) -> list[list[dict]]:
    output_by_token_id = {}
    for token_ids, output in semantic_output_sequences.items():
        if len(token_ids) != 1:
            raise ValueError(
                "direct label-logit ranking requires every output label to be one token"
            )
        output_by_token_id[token_ids[0]] = output
    if num_outputs > len(output_by_token_id):
        raise ValueError(
            f"requested {num_outputs} outputs from {len(output_by_token_id)} labels"
        )
    inputs, attention_mask, final_positions = padded_prompt_batch(
        prompt_ids_batch,
        pad_token_id,
        model.device,
    )
    with torch.inference_mode():
        all_logits = model(
            input_ids=inputs,
            attention_mask=attention_mask,
        ).logits
        logits = all_logits[
            torch.arange(len(prompt_ids_batch), device=all_logits.device),
            final_positions,
        ]
    token_ids = torch.tensor(
        list(output_by_token_id),
        dtype=torch.long,
        device=logits.device,
    )
    label_logits = logits.index_select(-1, token_ids)
    return [
        rank_label_scores(
            scores,
            token_ids,
            output_by_token_id,
            output_to_category,
            num_outputs,
        )
        for scores in label_logits
    ]


def reasoning_then_label_logit_rank_batch(
    model,
    tokenizer,
    initial_prompt_ids_batch: list[list[int]],
    reasoning_prompts: list[str],
    model_key: str,
    output_to_category: dict[str, str],
    semantic_output_sequences: dict[tuple[int, ...], str],
    num_outputs: int,
    reasoning_max_new_tokens: int,
    max_prompt_length: int,
) -> tuple[list[list[dict]], list[dict]]:
    inputs, attention_mask = left_padded_prompt_batch(
        initial_prompt_ids_batch,
        tokenizer.pad_token_id,
        model.device,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=reasoning_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    traces = []
    label_prompt_ids_batch = []
    generated_suffixes = generated[:, inputs.shape[1] :].tolist()
    for prompt, suffix in zip(reasoning_prompts, generated_suffixes, strict=True):
        eos_reached = tokenizer.eos_token_id in suffix
        content_tokens = (
            suffix[: suffix.index(tokenizer.eos_token_id)] if eos_reached else suffix
        )
        reasoning = tokenizer.decode(
            content_tokens,
            skip_special_tokens=False,
        ).strip()
        label_prompt_ids = render_reasoning_label_prompt_ids(
            tokenizer,
            prompt,
            reasoning,
            model_key,
        )
        if len(label_prompt_ids) > max_prompt_length:
            raise ValueError(
                f"reasoning label prompt has {len(label_prompt_ids)} tokens"
            )
        label_prompt_ids_batch.append(label_prompt_ids)
        traces.append(
            {
                "generated_reasoning": reasoning,
                "generated_reasoning_token_count": len(content_tokens),
                "generated_reasoning_eos_reached": eos_reached,
                "generated_reasoning_interest_id_leakage": bool(
                    INTEREST_ID_PATTERN.search(reasoning)
                ),
            }
        )
    rankings = direct_label_logit_rank_batch(
        model,
        label_prompt_ids_batch,
        output_to_category,
        semantic_output_sequences,
        num_outputs,
        tokenizer.pad_token_id,
    )
    return rankings, traces


def reasoning_rank_with_oom_backoff(
    model,
    tokenizer,
    initial_prompt_ids_batch: list[list[int]],
    reasoning_prompts: list[str],
    model_key: str,
    output_to_category: dict[str, str],
    semantic_output_sequences: dict[tuple[int, ...], str],
    num_outputs: int,
    reasoning_max_new_tokens: int,
    max_prompt_length: int,
) -> tuple[list[list[dict]], list[dict]]:
    try:
        return reasoning_then_label_logit_rank_batch(
            model,
            tokenizer,
            initial_prompt_ids_batch,
            reasoning_prompts,
            model_key,
            output_to_category,
            semantic_output_sequences,
            num_outputs,
            reasoning_max_new_tokens,
            max_prompt_length,
        )
    except torch.OutOfMemoryError:
        if len(initial_prompt_ids_batch) == 1:
            raise
    torch.cuda.empty_cache()
    midpoint = len(initial_prompt_ids_batch) // 2
    print(
        f"CUDA OOM for reasoning batch size {len(initial_prompt_ids_batch)}; "
        f"retrying as {midpoint}+{len(initial_prompt_ids_batch) - midpoint}",
        flush=True,
    )
    first_rankings, first_traces = reasoning_rank_with_oom_backoff(
        model,
        tokenizer,
        initial_prompt_ids_batch[:midpoint],
        reasoning_prompts[:midpoint],
        model_key,
        output_to_category,
        semantic_output_sequences,
        num_outputs,
        reasoning_max_new_tokens,
        max_prompt_length,
    )
    second_rankings, second_traces = reasoning_rank_with_oom_backoff(
        model,
        tokenizer,
        initial_prompt_ids_batch[midpoint:],
        reasoning_prompts[midpoint:],
        model_key,
        output_to_category,
        semantic_output_sequences,
        num_outputs,
        reasoning_max_new_tokens,
        max_prompt_length,
    )
    return first_rankings + second_rankings, first_traces + second_traces


def load_label_classifier(
    adapter: Path,
    mapping: dict,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str], str, str]:
    metadata_path = adapter / "label_classifier.json"
    weights_path = adapter / "label_classifier.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise ValueError(f"classifier artifacts are missing from {adapter}")
    metadata = json.loads(metadata_path.read_text())
    expected_tokens = mapping["special_tokens"]
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported label classifier schema version")
    if metadata.get("label_mapping_fingerprint") != mapping["mapping_fingerprint"]:
        raise ValueError("label classifier mapping fingerprint mismatch")
    if metadata.get("label_tokens") != expected_tokens:
        raise ValueError("label classifier token order differs from the label mapping")
    state = load_safetensors(str(weights_path), device=str(device))
    if set(state) != {"bias", "weight"}:
        raise ValueError("label classifier must contain exactly weight and bias")
    expected_shape = (len(expected_tokens), metadata.get("hidden_size"))
    if tuple(state["weight"].shape) != expected_shape:
        raise ValueError(
            f"label classifier weight shape must be {expected_shape}, "
            f"found {tuple(state['weight'].shape)}"
        )
    if tuple(state["bias"].shape) != (len(expected_tokens),):
        raise ValueError("label classifier bias shape differs from label count")
    head_type = metadata.get("head_type", "classifier")
    if head_type not in {"classifier", "residual_classifier"}:
        raise ValueError(f"unsupported label classifier head type: {head_type}")
    feature_normalization = metadata.get("feature_normalization", "none")
    if feature_normalization not in {"none", "l2"}:
        raise ValueError(
            f"unsupported classifier feature normalization: {feature_normalization}"
        )
    return state, expected_tokens, head_type, feature_normalization


def semantic_classifier_rank(
    model,
    prompt_ids: list[int],
    output_to_category: dict[str, str],
    classifier_state: dict[str, torch.Tensor],
    label_tokens: list[str],
    num_outputs: int,
    head_type: str = "classifier",
    label_token_ids: list[int] | None = None,
) -> list[dict]:
    return semantic_classifier_rank_batch(
        model,
        [prompt_ids],
        output_to_category,
        classifier_state,
        label_tokens,
        num_outputs,
        pad_token_id=0,
        head_type=head_type,
        label_token_ids=label_token_ids,
    )[0]


def semantic_classifier_rank_batch(
    model,
    prompt_ids_batch: list[list[int]],
    output_to_category: dict[str, str],
    classifier_state: dict[str, torch.Tensor],
    label_tokens: list[str],
    num_outputs: int,
    pad_token_id: int,
    head_type: str = "classifier",
    label_token_ids: list[int] | None = None,
    feature_normalization: str = "none",
) -> list[list[dict]]:
    if num_outputs > len(label_tokens):
        raise ValueError(f"requested {num_outputs} outputs from {len(label_tokens)} labels")
    inputs, attention_mask, final_positions = padded_prompt_batch(
        prompt_ids_batch,
        pad_token_id,
        model.device,
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states[-1]
        hidden = all_hidden[
            torch.arange(len(prompt_ids_batch), device=all_hidden.device),
            final_positions,
        ]
        if feature_normalization == "l2":
            hidden = F.normalize(hidden.to(torch.float32), p=2, dim=-1)
        elif feature_normalization != "none":
            raise ValueError(
                f"unsupported classifier feature normalization: {feature_normalization}"
            )
        scores = F.linear(
            hidden.to(classifier_state["weight"].dtype),
            classifier_state["weight"],
            classifier_state["bias"],
        )
        if head_type == "residual_classifier":
            if label_token_ids is None or len(label_token_ids) != len(label_tokens):
                raise ValueError("residual classifier requires one token ID per label")
            token_ids_for_logits = torch.tensor(
                label_token_ids,
                dtype=torch.long,
                device=outputs.logits.device,
            )
            base_scores = outputs.logits[
                torch.arange(len(prompt_ids_batch), device=outputs.logits.device),
                final_positions,
            ].index_select(-1, token_ids_for_logits)
            scores = scores + base_scores.to(scores.dtype)
        elif head_type != "classifier":
            raise ValueError(f"unsupported label classifier head type: {head_type}")
    token_ids = torch.arange(len(label_tokens), device=scores.device)
    output_by_token_id = dict(enumerate(label_tokens))
    return [
        rank_label_scores(
            row_scores,
            token_ids,
            output_by_token_id,
            output_to_category,
            num_outputs,
        )
        for row_scores in scores
    ]


def score_ranking(ranking: list[dict], target: str) -> dict:
    target_rank = next(
        (item["rank"] for item in ranking if item["prediction"] == target),
        None,
    )
    metrics = {}
    for cutoff in (10, 20):
        hit = target_rank is not None and target_rank <= cutoff
        metrics[f"recall_at_{cutoff}"] = float(hit)
        metrics[f"ndcg_at_{cutoff}"] = (
            1.0 / math.log2(target_rank + 1) if hit else 0.0
        )
    return {"target_rank": target_rank, **metrics}


def metrics_from_rankings(rows: list[dict], expected_rows: list[dict], num_beams: int) -> dict:
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    actual_by_id = {row["sample_id"]: row for row in rows}
    unknown = set(actual_by_id) - set(expected_by_id)
    if unknown:
        raise ValueError(f"details contain unknown sample IDs: {sorted(unknown)[:3]}")
    per_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid_beams = 0
    strict_beams = 0
    duplicate_valid_beams = 0
    for expected in expected_rows:
        detail = actual_by_id.get(expected["sample_id"])
        if detail is None:
            continue
        ranking = detail["ranking"]
        if len(ranking) != num_beams:
            raise ValueError(f"{expected['sample_id']}: expected {num_beams} beams")
        valid_predictions = [item["prediction"] for item in ranking if item["prediction"]]
        valid_beams += len(valid_predictions)
        strict_beams += sum(bool(item["format_valid"]) for item in ranking)
        duplicate_valid_beams += len(valid_predictions) - len(set(valid_predictions))
        for metric in ("recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20"):
            per_label[expected["target_interest"]][metric].append(float(detail[metric]))
    per_class = {
        label: {
            metric: sum(values) / len(values)
            for metric, values in sorted(label_metrics.items())
        }
        for label, label_metrics in sorted(per_label.items())
    }
    macro = {
        f"macro_{metric}": sum(values[metric] for values in per_class.values()) / len(per_class)
        for metric in ("recall_at_10", "recall_at_20", "ndcg_at_10", "ndcg_at_20")
    }
    requested = len(expected_rows)
    total_beams = requested * num_beams
    return {
        "requested_sample_count": requested,
        "recorded_sample_count": len(actual_by_id),
        "missing_sample_count": requested - len(actual_by_id),
        **macro,
        "class_count": len(per_class),
        "per_class_metrics": per_class,
        "valid_beam_rate": valid_beams / total_beams if total_beams else 0.0,
        "strict_beam_format_rate": strict_beams / total_beams if total_beams else 0.0,
        "duplicate_valid_beam_rate": duplicate_valid_beams / total_beams if total_beams else 0.0,
        "complete": len(actual_by_id) == requested,
    }


def establish_contract(
    args: argparse.Namespace,
    dataset: Path,
    labels: list[str],
    mapping: dict | None,
    semantic_registration: dict | None,
) -> dict:
    has_history_summary = "history_summary" in pq.ParquetFile(dataset).schema.names
    contract = {
        "schema_version": 1,
        "task": "amazon_new_interest_candidate_free_ranking",
        "prompt_schema_version": prompt_schema_version(
            args.label_mode,
            history_summary=has_history_summary,
            target_reasoning=args.decoding == "reasoning_then_label_logits",
        ),
        "history_summary": has_history_summary,
        "reasoning_first": args.decoding == "reasoning_then_label_logits",
        "model_key": args.model_key,
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "label_classifier_sha256": (
            sha256_file(args.adapter / "label_classifier.safetensors")
            if args.decoding == "semantic_classifier"
            else None
        ),
        "label_classifier_metadata_sha256": (
            sha256_file(args.adapter / "label_classifier.json")
            if args.decoding == "semantic_classifier"
            else None
        ),
        "view": args.view,
        "label_mode": args.label_mode,
        "label_map": str(args.label_map.resolve()) if mapping else None,
        "label_map_sha256": sha256_file(args.label_map) if mapping else None,
        "label_mapping_fingerprint": mapping["mapping_fingerprint"] if mapping else None,
        "semantic_token_registration": semantic_registration,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "label_file": str(args.label_file.resolve()),
        "label_file_sha256": sha256_file(args.label_file),
        "label_count": len(labels),
        "expected_label_count": args.expected_label_count,
        "sample_limit": args.limit,
        "decoding": f"candidate_free_{args.decoding}",
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_max_new_tokens": (
            args.reasoning_max_new_tokens
            if args.decoding == "reasoning_then_label_logits"
            else None
        ),
        "eval_batch_size": args.eval_batch_size,
        "candidates_in_prompt": False,
        "decoder_label_constraint": args.decoding in {
            "direct_label_logits",
            "reasoning_then_label_logits",
            "semantic_classifier",
        },
        "output_parser": (
            "exact_interest_id_category_sequence"
            if args.label_mode == "interest_id_category"
            else "exact_single_interest_id_token"
            if mapping
            else "normalized_text"
        ),
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    path = args.run_dir / "run_contract.json"
    return resolve_contract(path, contract, args.adapter)


def write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "ranking": [item["prediction"] for item in row["ranking"]],
                        "target_rank": row["target_rank"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_beams < 20:
        raise ValueError("--num-beams must be at least 20 for Recall/nDCG@20")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.reasoning_max_new_tokens <= 0:
        raise ValueError("--reasoning-max-new-tokens must be positive")
    if args.decoding == "direct_beam_search" and args.eval_batch_size != 1:
        raise ValueError("beam search requires --eval-batch-size 1")
    args.adapter = args.adapter.resolve()
    args.label_file = args.label_file.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / f"dev_2k_{args.view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit must be between 1 and dataset size")
        rows = rows[: args.limit]
    labels = load_labels(args.label_file, args.expected_label_count)
    mapping = None
    if args.label_mode in INTEREST_ID_LABEL_MODES:
        mapping = load_label_mapping(
            args.label_map.resolve(),
            expected_categories=labels,
            expected_count=args.expected_label_count,
        )
    output_to_category = {
        output_label(category, args.label_mode, mapping): category
        for category in labels
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    semantic_registration = (
        register_semantic_tokens(tokenizer, mapping) if mapping is not None else None
    )
    semantic_output_sequences = (
        exact_output_sequences(tokenizer, list(output_to_category))
        if mapping is not None
        else None
    )
    if args.decoding in {
        "direct_label_logits",
        "reasoning_then_label_logits",
        "semantic_classifier",
    } and mapping is None:
        raise ValueError("direct label ranking requires an Interest-ID label mode")
    contract = establish_contract(
        args,
        dataset,
        labels,
        mapping,
        semantic_registration,
    )
    details_path = args.run_dir / "details.jsonl"
    completed = latest_details(details_path, contract["fingerprint"])
    model = load_model(
        args.model_key,
        args.model,
        args.adapter,
        tokenizer,
        mapping,
        semantic_registration,
        device_map=args.device_map,
    )
    classifier_state = None
    classifier_tokens = None
    classifier_head_type = None
    classifier_feature_normalization = None
    if args.decoding == "semantic_classifier":
        (
            classifier_state,
            classifier_tokens,
            classifier_head_type,
            classifier_feature_normalization,
        ) = load_label_classifier(args.adapter, mapping, model.device)
    pending_rows = [row for row in rows if row["sample_id"] not in completed]
    for offset in range(0, len(pending_rows), args.eval_batch_size):
        batch_rows = pending_rows[offset : offset + args.eval_batch_size]
        reasoning_prompts = None
        if args.decoding == "reasoning_then_label_logits":
            rendered_reasoning = [
                render_reasoning_inputs(
                    tokenizer,
                    row,
                    args.model_key,
                    args.view,
                )
                for row in batch_rows
            ]
            reasoning_prompts = [item[0] for item in rendered_reasoning]
            prompt_ids_batch = [item[1] for item in rendered_reasoning]
        else:
            prompt_ids_batch = [
                render_prompt_ids(
                    tokenizer,
                    row,
                    args.model_key,
                    args.view,
                    args.label_mode,
                )
                for row in batch_rows
            ]
        for row, prompt_ids in zip(batch_rows, prompt_ids_batch, strict=True):
            if len(prompt_ids) > args.max_prompt_length:
                raise ValueError(f"{row['sample_id']}: prompt has {len(prompt_ids)} tokens")
        started = time.monotonic()
        reasoning_traces = None
        if args.decoding == "direct_label_logits":
            rankings = rank_batch_with_oom_backoff(
                prompt_ids_batch,
                lambda batch: direct_label_logit_rank_batch(
                    model,
                    batch,
                    output_to_category,
                    semantic_output_sequences,
                    args.num_beams,
                    tokenizer.pad_token_id,
                ),
            )
        elif args.decoding == "reasoning_then_label_logits":
            rankings, reasoning_traces = reasoning_rank_with_oom_backoff(
                model,
                tokenizer,
                prompt_ids_batch,
                reasoning_prompts,
                args.model_key,
                output_to_category,
                semantic_output_sequences,
                args.num_beams,
                args.reasoning_max_new_tokens,
                args.max_prompt_length,
            )
        elif args.decoding == "semantic_classifier":
            rankings = rank_batch_with_oom_backoff(
                prompt_ids_batch,
                lambda batch: semantic_classifier_rank_batch(
                    model,
                    batch,
                    output_to_category,
                    classifier_state,
                    classifier_tokens,
                    args.num_beams,
                    tokenizer.pad_token_id,
                    classifier_head_type,
                    [
                        semantic_registration["token_ids"][token]
                        for token in classifier_tokens
                    ],
                    classifier_feature_normalization,
                ),
            )
        else:
            rankings = [
                direct_beam_rank(
                    model,
                    tokenizer,
                    prompt_ids_batch[0],
                    output_to_category,
                    args.num_beams,
                    args.max_new_tokens,
                    semantic_output_sequences,
                )
            ]
        latency = (time.monotonic() - started) / len(batch_rows)
        for index, (row, ranking) in enumerate(zip(batch_rows, rankings, strict=True)):
            detail = {
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "target_interest": row["target_interest"],
                "ranking": ranking,
                **score_ranking(ranking, row["target_interest"]),
                "latency_seconds": latency,
                "run_fingerprint": contract["fingerprint"],
            }
            if reasoning_traces is not None:
                detail.update(reasoning_traces[index])
            append_jsonl(details_path, detail)
            completed[row["sample_id"]] = detail
        if len(completed) == len(batch_rows) or len(completed) % 10 == 0 or not pending_rows[offset + args.eval_batch_size :]:
            print(
                f"{args.model_key}/{args.view} ranking completed={len(completed)}/{len(rows)}",
                flush=True,
            )
    ordered = [completed[row["sample_id"]] for row in rows]
    metrics = metrics_from_rankings(ordered, rows, args.num_beams)
    if args.decoding == "reasoning_then_label_logits":
        metrics.update(
            generated_reasoning_mean_tokens=sum(
                row["generated_reasoning_token_count"] for row in ordered
            )
            / len(ordered),
            generated_reasoning_max_tokens=max(
                row["generated_reasoning_token_count"] for row in ordered
            ),
            generated_reasoning_eos_rate=sum(
                row["generated_reasoning_eos_reached"] for row in ordered
            )
            / len(ordered),
            generated_reasoning_empty_rate=sum(
                not row["generated_reasoning"] for row in ordered
            )
            / len(ordered),
            generated_reasoning_interest_id_leakage_rate=sum(
                row["generated_reasoning_interest_id_leakage"] for row in ordered
            )
            / len(ordered),
        )
    metrics.update(
        model_key=args.model_key,
        input_type=args.view,
        adapter=str(args.adapter),
        decoding=f"candidate_free_{args.decoding}",
        label_mode=args.label_mode,
        num_beams=args.num_beams,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_atomic(args.run_dir / "metrics.json", metrics)
    write_predictions(args.run_dir / "predictions.jsonl", ordered)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()