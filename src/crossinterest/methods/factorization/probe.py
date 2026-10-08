"""Fixed independent-user diagnostics, never a replacement for 20-beam evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.evaluate import load_model
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.prompts import render_training_messages
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.methods.alignment.data import HERE, BASE, sha256_file
from crossinterest.methods.mechanism.train import auxiliary_messages


ROOT = HERE / "cir_factorization"


def diagnostic_statistics(logits, donor_logits, target, seen, donor_seen):
    if seen[target] or donor_seen[target]:
        raise ValueError("diagnostic target is visible in one of the histories")
    scores, other = logits.double(), donor_logits.double()
    common = ~(seen | donor_seen)
    target_nll = torch.logsumexp(scores[~seen], 0) - scores[target]
    correct_logp = scores[target] - torch.logsumexp(scores[common], 0)
    donor_logp = other[target] - torch.logsumexp(other[common], 0)
    return {"conditional_nll": float(target_nll), "history_advantage": float(correct_logp - donor_logp),
            "seen_mass": float(scores.softmax(0)[seen].sum()), "common_unseen_count": int(common.sum())}


def macro_statistics(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["target_interest"]].append(row)
    names = ("conditional_nll", "history_advantage", "seen_mass")
    return {name: sum(sum(row[name] for row in rows) / len(rows) for rows in groups.values()) / len(groups)
            for name in names}


def probe_rows(limit=None):
    rows = pq.read_table(ROOT / "data/probe.parquet").to_pylist()
    return rows, rows if limit is None else rows[:limit]


def audit_probe(output: Path, adapter: Path, limit=None):
    result = json.loads(output.read_text())
    identity = {"adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "probe_sha256": sha256_file(ROOT / "data/probe.parquet"), "limit": limit}
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("probe output identity changed")
    raw = output.with_suffix(".npz")
    if sha256_file(raw) != result["logits_sha256"]:
        raise ValueError("probe raw logits changed")
    arrays = np.load(raw, allow_pickle=False)
    mapping = load_label_mapping(BASE / "interest_id_semantic_mapping.json", expected_count=112)
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    indices = {category: index for index, category in enumerate(categories)}
    all_rows, rows = probe_rows(limit)
    records, novel_records = [], []
    for index, row in enumerate(rows):
        donor = all_rows[row["donor_index"]]
        seen = torch.tensor([category in row["historical_interests"] for category in categories])
        donor_seen = torch.tensor([category in donor["historical_interests"] for category in categories])
        values = diagnostic_statistics(torch.from_numpy(arrays["correct"][index]),
            torch.from_numpy(arrays["donor"][index]), indices[row["target_interest"]], seen, donor_seen)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite probe diagnostics")
        records.append({"id": row["id"], "target_interest": row["target_interest"], "donor_id": donor["id"], **values})
        source = row["probe_source"]
        if source["target_interest"] in indices:
            logits = torch.from_numpy(arrays["novel"][index]).double()
            source_seen = torch.tensor([category in source["historical_interests"] for category in categories])
            target = indices[source["target_interest"]]
            if source_seen[target] or not torch.isfinite(logits).all():
                raise ValueError("invalid original novel-interest probe")
            novel_records.append({"id": source["id"], "target_interest": source["target_interest"],
                "target_nll_112": float(torch.logsumexp(logits, 0) - logits[target]),
                "conditional_nll": float(torch.logsumexp(logits[~source_seen], 0) - logits[target]),
                "seen_mass": float(logits.softmax(0)[source_seen].sum())})
    if result["records"] != records or result["macro"] != macro_statistics(records) or result["novel_records"] != novel_records:
        raise ValueError("probe diagnostics disagree with independent raw-logit recomputation")
    return {"passed": True, "rows": len(rows), "classes": len({row["target_interest"] for row in rows}),
            "macro": result["macro"], "novel_rows": len(novel_records), "sha256": sha256_file(output), **identity}


def run(model_path: str, adapter: Path, output: Path, limit=None):
    from transformers import AutoTokenizer

    if output.exists():
        return audit_probe(output, adapter, limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = load_label_mapping(BASE / "interest_id_semantic_mapping.json", expected_count=112)
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    indices = {category: index for index, category in enumerate(categories)}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    registration = register_semantic_tokens(tokenizer, mapping)
    label_ids = [registration["token_ids"][token] for token in mapping["special_tokens"]]
    model = load_model("qwen35_9b", model_path, adapter, tokenizer, mapping, registration, device_map="balanced")
    all_rows, rows = probe_rows(limit)
    correct_logits, donor_logits, novel_logits = [], [], []
    records, novel_records = [], []

    def forward(messages):
        encoded = baseline.tokenize_assistant_only(tokenizer, messages)
        first = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
        if first <= 0 or len(encoded["input_ids"]) > 8192:
            raise ValueError("probe target position or length invalid")
        inputs = torch.tensor([encoded["input_ids"][:first]], device=model.device)
        logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
        return logits[0, -1, label_ids].float().cpu()

    with torch.inference_mode():
        for index, row in enumerate(rows):
            donor = all_rows[row["donor_index"]]
            mismatch = {**donor, "target_interest": row["target_interest"]}
            correct = forward(auxiliary_messages(row, mapping))
            other = forward(auxiliary_messages(mismatch, mapping))
            correct_logits.append(correct.numpy())
            donor_logits.append(other.numpy())
            seen = torch.tensor([category in row["historical_interests"] for category in categories])
            donor_seen = torch.tensor([category in donor["historical_interests"] for category in categories])
            values = diagnostic_statistics(correct, other, indices[row["target_interest"]], seen, donor_seen)
            records.append({"id": row["id"], "target_interest": row["target_interest"], "donor_id": donor["id"], **values})
            source = row["probe_source"]
            if source["target_interest"] in indices:
                novel = forward(render_training_messages(source, "behavior_interest", "interest_id", mapping))
                source_seen = torch.tensor([category in source["historical_interests"] for category in categories])
                target = indices[source["target_interest"]]
                scores = novel.double()
                novel_records.append({"id": source["id"], "target_interest": source["target_interest"],
                    "target_nll_112": float(torch.logsumexp(scores, 0) - scores[target]),
                    "conditional_nll": float(torch.logsumexp(scores[~source_seen], 0) - scores[target]),
                    "seen_mass": float(scores.softmax(0)[source_seen].sum())})
                novel_logits.append(novel.numpy())
            else:
                novel_logits.append(np.full(112, np.nan, dtype=np.float32))
            if (index + 1) % 100 == 0:
                print(json.dumps({"event": "factor_probe", "rows": index + 1, "total": len(rows)}), flush=True)
    raw = output.with_suffix(".npz")
    np.savez_compressed(raw, correct=np.asarray(correct_logits), donor=np.asarray(donor_logits), novel=np.asarray(novel_logits))
    result = {"passed": True, "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
              "probe_sha256": sha256_file(ROOT / "data/probe.parquet"), "limit": limit,
              "logits_sha256": sha256_file(raw), "scope": "111_supported_independent_train_classes_not_official_ranking",
              "normalization": "history_advantage_uses_common_unseen_set_112_minus_union",
              "records": records, "macro": macro_statistics(records), "novel_records": novel_records,
              "novel_scope": "original_source_next_interest_if_in_112_no_target_rebalancing_single_ID_diagnostic"}
    write_json_atomic(output, result)
    return audit_probe(output, adapter, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(4,))
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.model, arguments.adapter, arguments.output, arguments.limit)), flush=True)