"""Target-blind independent history groups for compositional ID generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


def history_groups(row: dict, grouping: str = "category") -> list[list[int]]:
    titles = list(row["historical_titles"])
    categories = list(row["historical_interests"])
    if not titles or len(titles) != len(categories):
        raise ValueError("history titles and categories must be nonempty and aligned")
    if any(not isinstance(value, str) or not value.strip() for value in titles + categories):
        raise ValueError("history requires nonempty string titles and categories")
    members = {}
    for position, category in enumerate(categories):
        members.setdefault(category, []).append(position)
    groups = sorted(members.values(), key=lambda positions: positions[-1])
    if grouping == "random":
        payload = json.dumps([titles, categories], ensure_ascii=False, separators=(",", ":"))
        seed = hashlib.sha256(payload.encode()).hexdigest()
        permutation = sorted(range(len(titles)), key=lambda position: hashlib.sha256(
            f"interest-composition-v1:42:{seed}:{position}".encode()).digest())
        shuffled = []
        offset = 0
        for group in groups:
            shuffled.append(sorted(permutation[offset:offset + len(group)]))
            offset += len(group)
        groups = sorted(shuffled, key=lambda positions: positions[-1])
    elif grouping != "category":
        raise ValueError(f"unknown grouping: {grouping}")
    if sorted(position for group in groups for position in group) != list(range(len(titles))):
        raise ValueError("grouping must partition every visible event exactly once")
    return groups


def group_texts(row: dict, grouping: str = "category") -> list[str]:
    groups = history_groups(row, grouping)
    titles = row["historical_titles"]
    categories = row["historical_interests"]
    return ["Observed purchase history (original event positions):\n" + "\n".join(
        f"{position + 1}. [{categories[position]}] {titles[position]}" for position in group
    ) + "\nHistory representation:" for group in groups]


class InterestBridge(nn.Module):
    def __init__(self, hidden_size: int, embedding_rms: float):
        super().__init__()
        if embedding_rms <= 0:
            raise ValueError("positive embedding scale required")
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.projection.weight)
        self.register_buffer("embedding_rms", torch.tensor(float(embedding_rms)))

    def forward(self, hidden):
        normalized = F.layer_norm(hidden.float(), (hidden.shape[-1],))
        return self.projection(normalized.to(self.projection.weight.dtype)) * self.embedding_rms


class CompositionController:
    def __init__(self, model, grouping: str = "category", directory: Path | None = None,
                 encoder_token_budget: int = 8192):
        if grouping not in ("category", "random") or encoder_token_budget < 1:
            raise ValueError("invalid composition configuration")
        if hasattr(model, "interest_bridge"):
            raise ValueError("composition controller is already attached")
        decoders = [module for module in model.modules() if hasattr(module, "layers")
                    and hasattr(module, "embed_tokens") and hasattr(module, "norm")]
        if len(decoders) != 1:
            raise ValueError(f"expected exactly one text decoder, found {len(decoders)}")
        self.model = model
        self.decoder = decoders[0]
        self.original_forward = model.forward
        self.encoder_token_budget = encoder_token_budget
        embedding = model.get_input_embeddings().weight
        scale = embedding[:min(1024, len(embedding))].detach().float().square().mean().sqrt().item()
        bridge = InterestBridge(embedding.shape[1], scale).to(embedding.device)
        model.add_module("interest_bridge", bridge)
        self.bridge = bridge
        self.metadata = {"architecture": "independent-interest-composition-v1", "grouping": grouping,
                         "hidden_size": embedding.shape[1], "memory_tokens_per_group": 1,
                         "independent_encoder_batches": True, "raw_history_decoder_bypass": False,
                         "encoder_token_budget": encoder_token_budget}
        self.encoder_calls = 0
        self.decoder_calls = 0
        if directory is not None:
            self.load(directory)

    def encode(self, group_input_ids, group_attention_mask):
        if group_input_ids.ndim != 2 or group_attention_mask.shape != group_input_ids.shape:
            raise ValueError("group tokens and masks must be aligned matrices")
        lengths = group_attention_mask.sum(dim=-1)
        if not len(lengths) or not (lengths > 0).all():
            raise ValueError("empty interest group")
        positions = torch.arange(group_input_ids.shape[1], device=lengths.device)
        if not torch.equal(group_attention_mask.bool(), positions[None, :] < lengths[:, None]):
            raise ValueError("independent groups require contiguous right padding")
        order = lengths.argsort(descending=True, stable=True)
        encoded = []
        offset = 0
        while offset < len(order):
            longest = int(lengths[order[offset]])
            count = max(1, self.encoder_token_budget // longest)
            selected = order[offset:offset + count]
            inputs = group_input_ids[selected, :longest]
            masks = group_attention_mask[selected, :longest]
            states = self.decoder(input_ids=inputs, attention_mask=masks, use_cache=False,
                                  return_dict=True).last_hidden_state
            endpoint = lengths[selected].to(states.device) - 1
            encoded.append(states[torch.arange(len(selected), device=states.device), endpoint])
            self.encoder_calls += 1
            offset += len(selected)
        inverse = order.argsort().to(encoded[0].device)
        return self.bridge(torch.cat(encoded, dim=0)[inverse].to(self.bridge.projection.weight.device))

    def inputs_embeds(self, input_ids, memory_mask, group_input_ids, group_attention_mask):
        if memory_mask.shape != input_ids.shape or memory_mask.dtype != torch.bool:
            raise ValueError("aligned boolean memory mask required")
        if int(memory_mask.sum()) != len(group_input_ids) or not memory_mask.any(dim=-1).all():
            raise ValueError("memory slots and independently encoded groups must correspond")
        memories = self.encode(group_input_ids, group_attention_mask)
        embeddings = self.model.get_input_embeddings()(input_ids).clone()
        embeddings[memory_mask.to(embeddings.device)] = memories.to(embeddings.device, embeddings.dtype)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, memory_mask=None,
                group_input_ids=None, group_attention_mask=None, **kwargs):
        if any(value is None for value in (input_ids, attention_mask, memory_mask,
                                           group_input_ids, group_attention_mask)):
            raise ValueError("composition training requires complete explicit history inputs")
        if kwargs.get("past_key_values") is not None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("external history cache or embeddings are forbidden")
        if labels is not None and (labels[memory_mask] != -100).any():
            raise ValueError("memory slots cannot carry supervised labels")
        embeddings = self.inputs_embeds(input_ids, memory_mask, group_input_ids, group_attention_mask)
        self.decoder_calls += 1
        kwargs["use_cache"] = False
        return self.original_forward(inputs_embeds=embeddings, attention_mask=attention_mask,
                                     labels=labels, **kwargs)

    def attach_training(self):
        from functools import wraps

        @wraps(self.forward)
        def forward(*args, **kwargs):
            return self.forward(*args, **kwargs)

        self.model.forward = forward

    def detach_training(self):
        self.model.forward = self.original_forward

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in self.bridge.state_dict().items()},
                  str(directory / "interest_bridge.safetensors"))
        (directory / "interest_bridge.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def load(self, directory: Path):
        if json.loads((directory / "interest_bridge.json").read_text()) != self.metadata:
            raise ValueError("composition checkpoint configuration mismatch")
        self.bridge.load_state_dict(load_file(str(directory / "interest_bridge.safetensors")), strict=True)


def encode_example(row, tokenizer, mapping, stage, grouping, training=True, max_length=8192):
    from crossinterest.training import train as baseline
    from crossinterest.training.prompts import render_history, render_training_messages, render_training_prompt
    from crossinterest.methods.mechanism.train import auxiliary_messages

    if stage not in ("auxiliary", "novel"):
        raise ValueError("unknown training stage")
    if any(row.get(field) is not None for field in ("history_summary", "target_reasoning", "target_rationale")):
        raise ValueError("extra history or reasoning bypass is forbidden")
    texts = group_texts(row, grouping)
    if tokenizer.pad_token is None:
        raise ValueError("tokenizer requires an explicit memory placeholder token")
    if not training:
        if stage != "novel":
            raise ValueError("formal inference is defined only for novel-interest prediction")
        messages = [{"role": "user", "content": render_training_prompt(row, "behavior_interest", "interest_id")}]
    else:
        messages = (auxiliary_messages(row, mapping) if stage == "auxiliary" else
                    render_training_messages(row, "behavior_interest", "interest_id", mapping))
    history = render_history(row, "behavior_interest")
    if messages[0]["content"].count(history) != 1:
        raise ValueError("expected exactly one original history body")
    messages[0]["content"] = messages[0]["content"].replace(history, tokenizer.pad_token * len(texts))
    if training:
        encoded = baseline.tokenize_assistant_only(tokenizer, messages)
    else:
        prompt = baseline.apply_chat_template(tokenizer, messages[:1], add_generation_prompt=True)
        encoded = {"input_ids": prompt, "attention_mask": [1] * len(prompt), "labels": [-100] * len(prompt)}
    encoded["memory_mask"] = [token == tokenizer.pad_token_id and label == -100
                              for token, label in zip(encoded["input_ids"], encoded["labels"], strict=True)]
    if sum(encoded["memory_mask"]) != len(texts):
        raise ValueError("placeholder token must map one-to-one to interest memory slots")
    encoded["groups"] = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if any(not group or len(group) > max_length for group in encoded["groups"]):
        raise ValueError("empty or overlong encoded interest; truncation is forbidden")
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("decoder context exceeds limit; truncation is forbidden")
    return encoded


def bind_ranking_composition(evaluator, controller, mapping, record_check):
    from crossinterest.methods.composition.train import CompositionCollator

    original_rank = evaluator.direct_beam_rank
    current = {}

    def render(tokenizer, row, model_key, view, label_mode):
        if (model_key, view, label_mode) != ("qwen35_9b", "behavior_interest", "interest_id"):
            raise ValueError("composition evaluation requires the frozen B+I ID protocol")
        feature = encode_example(row, tokenizer, mapping, "novel", controller.metadata["grouping"], training=False)
        current.clear()
        current.update(sample_id=row["sample_id"], feature=feature)
        return feature["input_ids"]

    def rank(model, tokenizer, prompt_ids, output_to_category, num_beams, max_new_tokens,
             semantic_output_sequences=None):
        if current.get("feature", {}).get("input_ids") != prompt_ids:
            raise ValueError("history and generation prefix are not aligned")
        batch = CompositionCollator(tokenizer.pad_token_id)([current["feature"]])
        device = model.get_input_embeddings().weight.device
        batch = {key: value.to(device) for key, value in batch.items()}
        batch.pop("labels")
        batch.pop("attention_mask")
        before = controller.encoder_calls
        with torch.inference_mode():
            embeddings = controller.inputs_embeds(**batch)
        original_generate = model.generate
        generate_calls = 0

        def generate(*args, **kwargs):
            nonlocal generate_calls
            if (args or kwargs.get("inputs_embeds") is not None or kwargs.get("past_key_values") is not None
                    or kwargs.get("logits_processor") or kwargs.get("prefix_allowed_tokens_fn")):
                raise ValueError("unexpected generation constraint or external history")
            if kwargs["input_ids"].tolist() != [prompt_ids]:
                raise ValueError("generation prefix changed")
            generate_calls += 1
            return original_generate(**kwargs, inputs_embeds=embeddings)

        model.generate = generate
        try:
            result = original_rank(model, tokenizer, prompt_ids, output_to_category,
                                   num_beams, max_new_tokens, semantic_output_sequences)
        finally:
            model.generate = original_generate
        if generate_calls != 1:
            raise ValueError("expected one complete beam generation per user")
        record_check({"sample_id": current["sample_id"], "groups": len(current["feature"]["groups"]),
                      "encoder_calls": controller.encoder_calls - before, "generate_calls": generate_calls,
                      "prompt_tokens": len(prompt_ids), "num_beams": num_beams,
                      "grouping": controller.metadata["grouping"], "raw_history_decoder_bypass": False})
        current.clear()
        return result

    evaluator.render_prompt_ids = render
    evaluator.direct_beam_rank = rank