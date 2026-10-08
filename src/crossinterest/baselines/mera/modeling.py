"""Tokenizer and model helpers for atomic MeRA category tokens."""

from __future__ import annotations

import inspect

import torch

from crossinterest.baselines.mera.core import validate_label_space


def category_token_mapping(label_space: dict, token_style: str) -> dict[str, str]:
    validate_label_space(label_space)
    if token_style == "indexed":
        return dict(label_space["category_to_token"])
    if token_style == "official_quoted_name":
        return {
            category: f'"{category}"'
            for category in label_space["labels"]
        }
    raise ValueError(f"unknown category token style: {token_style}")


def register_category_tokens(
    tokenizer,
    label_space: dict,
    category_to_token: dict[str, str] | None = None,
) -> dict:
    validate_label_space(label_space)
    category_to_token = category_to_token or label_space["category_to_token"]
    if set(category_to_token) != set(label_space["labels"]):
        raise ValueError("runtime category tokens must cover the label space")
    tokens = [
        category_to_token[category]
        for category in label_space["labels"]
    ]
    if len(set(tokens)) != len(tokens):
        raise ValueError("runtime category tokens must be unique")
    vocabulary_size_before = len(tokenizer)
    options = {}
    parameters = inspect.signature(tokenizer.add_special_tokens).parameters
    if "replace_extra_special_tokens" in parameters:
        options["replace_extra_special_tokens"] = False
    elif "replace_additional_special_tokens" in parameters:
        options["replace_additional_special_tokens"] = False
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens},
        **options,
    )
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    if added != len(tokens):
        raise ValueError("category tokens must be newly appended to a fresh tokenizer")
    if len(set(token_ids)) != len(tokens):
        raise ValueError("category tokens did not receive unique token IDs")
    if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in token_ids:
        raise ValueError("a category token resolved to the unknown token")
    for token, token_id in zip(tokens, token_ids, strict=True):
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            raise ValueError(f"{token} is not encoded as exactly one token: {encoded}")
    return {
        "vocabulary_size_before": vocabulary_size_before,
        "vocabulary_size_after": len(tokenizer),
        "added_token_count": added,
        "tokens": tokens,
        "token_ids": token_ids,
        "category_to_token": category_to_token,
    }


@torch.no_grad()
def initialize_category_tokens(
    model,
    tokenizer,
    label_space: dict,
    registration: dict,
) -> dict:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("model must expose input and output embeddings")
    vocabulary_size_before = registration["vocabulary_size_before"]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    tied = input_weight.data_ptr() == output_weight.data_ptr()
    sources = {}
    source_texts = {}
    for category, token_id in zip(
        label_space["labels"],
        registration["token_ids"],
        strict=True,
    ):
        semantic_text = label_space["category_to_semantic_text"][category]
        source_ids = tokenizer(semantic_text, add_special_tokens=False)["input_ids"]
        if not source_ids or any(
            source_id >= vocabulary_size_before for source_id in source_ids
        ):
            raise ValueError(f"invalid semantic initialization source: {category}")
        input_weight[token_id].copy_(input_weight[source_ids].mean(dim=0))
        if not tied:
            output_weight[token_id].copy_(output_weight[source_ids].mean(dim=0))
        sources[category] = source_ids
        source_texts[category] = semantic_text
    return {
        **registration,
        "initialization": "mean base-model token rows of the child category name",
        "tied_word_embeddings": tied,
        "source_token_ids": sources,
        "source_texts": source_texts,
    }


def lora_linear_targets(model, exclude_visual: bool = True) -> list[str]:
    output_embeddings = model.get_output_embeddings()
    targets = []
    output_found = False
    for name, module in model.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue
        if exclude_visual and (".visual." in name or name.startswith("visual.")):
            continue
        targets.append(name)
        if module is output_embeddings:
            output_found = True
    if not output_found:
        raise ValueError("LoRA targets must include the output LM head")
    return targets


def _active_module(module):
    modules_to_save = getattr(module, "modules_to_save", None)
    if modules_to_save is None:
        return module
    active_adapter = getattr(module, "active_adapter", None)
    if isinstance(active_adapter, (list, tuple)):
        active_adapter = active_adapter[0]
    if active_adapter in modules_to_save:
        return modules_to_save[active_adapter]
    if len(modules_to_save) == 1:
        return next(iter(modules_to_save.values()))
    raise ValueError("cannot resolve active modules_to_save adapter")


def _effective_linear_rows(module, token_ids: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "base_layer"):
        if not hasattr(module, "weight"):
            raise ValueError("output LM head does not expose weight rows")
        return module.weight[token_ids]
    rows = module.base_layer.weight[token_ids]
    active_adapters = getattr(module, "active_adapters", [])
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    for adapter in active_adapters:
        if adapter not in module.lora_A:
            continue
        lora_a = module.lora_A[adapter].weight
        lora_b = module.lora_B[adapter].weight[token_ids]
        rows = rows + (lora_b @ lora_a) * module.scaling[adapter]
    return rows


def category_representations(
    model,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    while not (
        hasattr(model, "get_input_embeddings")
        and hasattr(model, "get_output_embeddings")
    ):
        wrapped_model = getattr(model, "module", None)
        if wrapped_model is None:
            raise ValueError("model does not expose input/output embeddings")
        model = wrapped_model
    input_module = _active_module(model.get_input_embeddings())
    output_module = _active_module(model.get_output_embeddings())
    input_rows = input_module(token_ids)
    output_rows = _effective_linear_rows(output_module, token_ids)
    return input_rows, output_rows
