"""Tokenizer and model setup for semantically initialized Interest IDs."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from crossinterest.training.label_space import validate_label_mapping


def register_semantic_tokens(tokenizer, mapping: dict) -> dict[str, Any]:
    validate_label_mapping(mapping)
    tokens = mapping["special_tokens"]
    vocabulary_size_before = len(tokenizer)
    registration_options = {}
    parameters = inspect.signature(tokenizer.add_special_tokens).parameters
    if "replace_extra_special_tokens" in parameters:
        registration_options["replace_extra_special_tokens"] = False
    elif "replace_additional_special_tokens" in parameters:
        registration_options["replace_additional_special_tokens"] = False
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        **registration_options,
    )
    token_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in tokens
    }
    if len(set(token_ids.values())) != len(tokens):
        raise ValueError("Interest ID tokens do not have unique tokenizer IDs")
    unknown_token_id = tokenizer.unk_token_id
    if unknown_token_id is not None and unknown_token_id in token_ids.values():
        raise ValueError("an Interest ID token resolved to the unknown token")
    for category, interest_id in mapping["category_to_id"].items():
        expected = [token_ids[token] for token in mapping["category_to_tokens"][category]]
        actual = tokenizer(interest_id, add_special_tokens=False)["input_ids"]
        if actual != expected:
            raise ValueError(
                f"{category}: Interest ID must encode as exactly one registered token; "
                f"expected {expected}, found {actual}"
            )
    return {
        "vocabulary_size_before": vocabulary_size_before,
        "vocabulary_size_after": len(tokenizer),
        "added_token_count": added,
        "semantic_token_count": len(tokens),
        "token_ids": token_ids,
    }


def _source_token_ids(tokenizer, text: str, base_tokenizer_size: int) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError(f"semantic initialization text produced no tokens: {text}")
    if any(token_id >= base_tokenizer_size for token_id in token_ids):
        raise ValueError(f"semantic initialization text contains a newly added token: {text}")
    return token_ids


@torch.no_grad()
def initialize_semantic_token_embeddings(
    model,
    tokenizer,
    mapping: dict,
    registration: dict[str, Any],
) -> dict[str, Any]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose both input and output embeddings")
    model_vocabulary_size_before = input_embeddings.num_embeddings
    base_tokenizer_size = registration["vocabulary_size_before"]
    expected_new_tokens = registration["semantic_token_count"]
    if len(tokenizer) != base_tokenizer_size + expected_new_tokens:
        raise ValueError(
            "fresh base tokenizer vocabulary mismatch: expected exactly "
            f"{expected_new_tokens} appended Interest ID tokens"
        )
    if registration["added_token_count"] != expected_new_tokens:
        raise ValueError("Interest ID tokens must be newly appended to the base tokenizer")

    model_vocabulary_size_after = max(model_vocabulary_size_before, len(tokenizer))
    if model_vocabulary_size_after != model_vocabulary_size_before:
        model.resize_token_embeddings(model_vocabulary_size_after, mean_resizing=False)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    tied_embeddings = input_weight.data_ptr() == output_weight.data_ptr()
    source_token_ids = {}
    for token in mapping["special_tokens"]:
        text = mapping["token_to_text"][token]
        source_ids = _source_token_ids(tokenizer, text, base_tokenizer_size)
        token_id = registration["token_ids"][token]
        input_weight[token_id].copy_(input_weight[source_ids].mean(dim=0))
        if not tied_embeddings:
            output_weight[token_id].copy_(output_weight[source_ids].mean(dim=0))
        source_token_ids[token] = source_ids
    return {
        **registration,
        "model_vocabulary_size_before": model_vocabulary_size_before,
        "model_vocabulary_size_after": model_vocabulary_size_after,
        "tied_word_embeddings": tied_embeddings,
        "initialization": "mean base-model rows of taxonomy text tokens",
        "source_token_ids": source_token_ids,
    }


def semantic_lora_target_modules(
    model,
    exclude_visual: bool,
    include_output_embeddings: bool = True,
) -> list[str]:
    output_embeddings = model.get_output_embeddings()
    targets = []
    output_layer_found = False
    for name, module in model.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue
        if exclude_visual and (".visual." in name or name.startswith("visual.")):
            continue
        if module is output_embeddings and not include_output_embeddings:
            continue
        targets.append(name)
        if module is output_embeddings:
            output_layer_found = True
    if include_output_embeddings and not output_layer_found:
        raise ValueError("Interest ID LoRA targets must include the untied output embedding layer")
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate Interest ID LoRA target modules")
    return targets