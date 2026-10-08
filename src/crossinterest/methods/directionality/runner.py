"""Tokenize and freeze the four equally sized directional training controls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.prompts import render_training_messages
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.mechanism.train import MechanismDataset, auxiliary_messages
from crossinterest.methods.mechanism.runner import immutable_json, save_table, smoke_indices


class BidirectionalDataset(MechanismDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stage == "auxiliary":
            self.table = self.table.append_column("direction", pq.read_table(args[0], columns=["direction"])["direction"])

    def __getitem__(self, index):
        if self.stage != "auxiliary":
            return super().__getitem__(index)
        row = self.table.slice(index, 1).to_pylist()[0]
        if row["direction"] not in ("forward", "inverse"):
            raise ValueError("unknown directional training task")
        messages = (render_training_messages(row, "behavior_interest", "interest_id", self.mapping)
                    if row["direction"] == "forward" else auxiliary_messages(row, self.mapping))
        encoded = baseline.tokenize_assistant_only(self.tokenizer, messages)
        if len(encoded["input_ids"]) > self.max_length:
            raise ValueError("directional training truncation forbidden")
        encoded["positive_mask"] = [category == row["target_interest"] for category in self.category_order]
        return encoded


def preflight(model_key: str, variant: str) -> dict:
    from transformers import AutoTokenizer
    from crossinterest.methods.cir.runner import MODELS, TRAIN, DEV, MAPPING, TRAIN_SHA, DEV_SHA, MAPPING_SHA, sources, preflight_path

    if model_key != "qwen35_9b":
        raise ValueError("directional experiment only authorizes Qwen3.5-9B")
    arm = variant.removeprefix("bidirectional_")
    root = HERE / "bidirectional/data"
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    audit_path = root / "INDEPENDENT_AUDIT.json"
    require_fields(json.loads(audit_path.read_text()), {"passed": True, "manifest_sha256": sha256_file(manifest_path)},
                   "directional independent data audit")
    auxiliary = root / f"{arm}.parquet"
    for source, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA),
                             (auxiliary, manifest["files"][arm]["sha256"])):
        if sha256_file(source) != expected:
            raise ValueError(f"directional source changed: {source}")
    model, revision = MODELS[model_key]
    path = preflight_path(model_key, variant)
    identity = {"model_key": model_key, "model_repo": model, "model_revision": revision,
                "variant": variant, "directional": True, "code_sha256": sources(),
                "data_manifest_sha256": sha256_file(manifest_path), "data_audit_sha256": sha256_file(audit_path)}
    if path.exists():
        prepared = json.loads(path.read_text())
        require_fields(prepared, identity, "directional freeze")
        for info in prepared["files"].values():
            if sha256_file(HERE.parent / info["path"]) != info["sha256"]:
                raise ValueError("directional file changed")
        return prepared
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    files, maxima, totals, digests, direction_totals = {}, {}, {}, {}, {}
    for stage, source in (("auxiliary", auxiliary), ("novel", TRAIN)):
        table = pq.read_table(source)
        counts = Counter(table["target_interest"].to_pylist())
        if set(counts) != set(mapping["category_to_id"]) or len(set(counts.values())) != 1:
            raise ValueError("unbalanced directional stage")
        files[stage] = {"path": str(source.relative_to(HERE.parent)), "sha256": sha256_file(source),
                        "rows": len(table), "steps": (len(table) + 31) // 32}
        dataset = BidirectionalDataset(source, tokenizer, "behavior_interest", "interest_id", mapping,
                                       8192, None, None, None, len(table), stage=stage, relation=False)
        lengths, digest, token_counts = [], hashlib.sha256(), Counter()
        directions = table["direction"].to_pylist() if stage == "auxiliary" else ["novel"] * len(table)
        for index in range(len(dataset)):
            encoded = dataset[index]
            lengths.append(len(encoded["input_ids"]))
            digest.update(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            token_counts[directions[index]] += lengths[-1]
            if (index + 1) % 10000 == 0:
                print(f"tokenizer audit {variant}/{stage} {index+1}/{len(table)}", flush=True)
        longest = table.take(sorted(range(len(table)), key=lambda index: (-lengths[index], index)))
        if stage == "auxiliary" and arm != "forward":
            selected = []
            seen = set()
            for index, row in enumerate(longest.select(["target_interest", "direction"]).to_pylist()):
                key = (row["target_interest"], row["direction"])
                if key not in seen:
                    seen.add(key)
                    selected.append(index)
            if len(selected) != 224:
                raise ValueError("smoke must cover all classes and both directions")
        else:
            selected = smoke_indices(longest)
        files[f"smoke_{stage}"] = save_table(path.parent / f"smoke_{stage}.parquet", longest.take(selected))
        maxima[stage], totals[stage], digests[stage] = max(lengths), sum(lengths), digest.hexdigest()
        direction_totals[stage] = dict(token_counts)
    prepared = {"passed": True, "schema_version": 1, **identity, "files": files,
                "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
                "token_registration": registration, "max_train_tokens": maxima,
                "train_tokens_per_epoch": totals, "tokenized_sha256": digests,
                "tokens_by_direction": direction_totals, "equal_tokens_or_flops": False,
                "distributed_padding_duplicates": 0,
                "smoke_selection": "longest_per_class_and_direction_or_two_per_class",
                "auxiliary_objective": "uniform_positive_set_ce_112_for_both_directions",
                "evaluation": {"rows": 2000, "beams": 20, "candidate_free": True, "do_sample": False}}
    immutable_json(path, prepared)
    return prepared


def inverse_probe(model_path: str, adapter: Path, output: Path) -> None:
    import torch
    from transformers import AutoTokenizer
    from crossinterest.training.evaluate import load_model
    from crossinterest.methods.cir.runner import MAPPING

    identity = {"adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "scope": "seen_training_pairs_diagnostic_not_generalization", "per_class": 2}
    if output.exists():
        require_fields(json.loads(output.read_text()), identity, "inverse diagnostic")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    model = load_model("qwen35_9b", model_path, adapter, tokenizer, mapping, registration, device_map="balanced")
    label_ids = [registration["token_ids"][token] for token in mapping["special_tokens"]]
    label_indices = {mapping["id_to_category"][token]: index for index, token in enumerate(mapping["special_tokens"])}
    tables = {arm: pq.read_table(HERE / f"bidirectional/data/{arm}.parquet")
              for arm in ("ordinary", "paired", "shuffled")}
    counts, indices = Counter(), []
    for index, row in enumerate(tables["ordinary"].select(["direction", "target_interest"]).to_pylist()):
        if row["direction"] == "inverse" and counts[row["target_interest"]] < 2:
            counts[row["target_interest"]] += 1
            indices.append(index)
    if len(indices) != 224:
        raise ValueError("inverse probe must cover 112 labels twice")
    records = []
    with torch.inference_mode():
        for index in indices:
            losses = {}
            for arm, table in tables.items():
                row = table.slice(index, 1).to_pylist()[0]
                encoded = baseline.tokenize_assistant_only(tokenizer, auxiliary_messages(row, mapping))
                position = next(offset for offset, label in enumerate(encoded["labels"]) if label != -100) - 1
                inputs = torch.tensor([encoded["input_ids"]], device=model.device)
                logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
                selected = logits[0, position, label_ids].float()
                losses[arm] = -torch.log_softmax(selected, dim=-1)[label_indices[row["target_interest"]]].item()
                del logits, inputs
            if not all(__import__("math").isfinite(value) for value in losses.values()):
                raise ValueError("nonfinite diagnostic NLL")
            records.append({"source_id": row["source_id"], "target_interest": row["target_interest"], "nll": losses})
    write_json_atomic(output, {"passed": True, **identity, "rows": len(records), "records": records,
                              "mean_nll": {arm: sum(record["nll"][arm] for record in records) / len(records)
                                           for arm in tables}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inverse_probe(args.model, args.adapter, args.output)