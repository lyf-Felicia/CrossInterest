"""Prepare and verify the fixed single-SFT Amazon backbone comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from crossinterest.evaluation.common import latest_details, metrics_from_details, write_json_atomic
from crossinterest.training import evaluate, evaluate_ranking
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.prompts import render_training_messages
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.training.train import tokenize_assistant_only
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.alignment.audit import require_fields


BACKBONE_KEYS = (
    "qwen25_05b", "qwen25_15b", "qwen25_3b", "qwen25_7b",
    "qwen25_14b", "qwen25_32b", "qwen3_8b", "qwen35_9b", "qwen35_27b",
)
ROOT = HERE / "backbone"
TRAIN = BASE / "train.parquet"
DEV = BASE / "eval" / "dev_2k_behavior_interest.parquet"
MAPPING = BASE / "interest_id_semantic_mapping.json"
TRAIN_SHA256 = "43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd"
MAPPING_SHA256 = "5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da"
CODE_FILES = (
    "eval_interest_id_minionerec/backbone.py",
    'crossinterest.training/train.py', 'crossinterest.training/prompts.py',
    'crossinterest.training/label_space.py', 'crossinterest.training/semantic_tokens.py',
    'crossinterest.training/evaluate.py', 'crossinterest.training/evaluate_ranking.py',
    'crossinterest.training/run_train_eval.sh',
    'crossinterest.evaluation/common.py', 'crossinterest.evaluation/build_views.py',
)


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(HERE.parent / name) for name in CODE_FILES}


def validate_code_contract(manifest: dict) -> None:
    actual = code_hashes()
    original = manifest["code_sha256"]
    if actual == original:
        return
    repair = manifest.get("evaluation_repair")
    if not isinstance(repair, dict):
        raise ValueError("code changed without an explicit evaluation repair")
    require_fields(repair, {"original_code_sha256": original, "repaired_code_sha256": actual},
                   "evaluation repair provenance")
    changed = {name for name in actual.keys() | original.keys() if actual.get(name) != original.get(name)}
    if changed != {"eval_interest_id_minionerec/backbone.py", 'crossinterest.training/evaluate.py'}:
        raise ValueError("evaluation repair must not alter training, data, or decoding code")


def tokenized_hash(features: dict) -> str:
    payload = {key: features[key] for key in ("input_ids", "labels")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def longest_balanced_indices(rows: list[dict], lengths: list[int], per_class: int = 2) -> list[int]:
    if len(rows) != len(lengths) or per_class < 1:
        raise ValueError("smoke rows/lengths must align and quota must be positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("smoke source contains duplicate IDs")
    by_class = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row["target_interest"]].append(index)
    selected = []
    for label, indices in sorted(by_class.items()):
        if len(indices) < per_class:
            raise ValueError(f"{label}: insufficient distinct smoke samples")
        selected.extend(sorted(indices, key=lambda index: (-lengths[index], rows[index]["id"]))[:per_class])
    return sorted(selected, key=lambda index: rows[index]["id"])


def prepare(model_key: str) -> dict:
    if model_key not in BACKBONE_KEYS:
        raise ValueError("model is not part of the approved backbone matrix")
    if sha256_file(TRAIN) != TRAIN_SHA256 or sha256_file(MAPPING) != MAPPING_SHA256:
        raise ValueError("frozen Amazon Train or semantic mapping changed")
    table = pq.read_table(TRAIN)
    rows = table.to_pylist()
    counts = Counter(row["target_interest"] for row in rows)
    if len(rows) != 112000 or len(counts) != 112 or set(counts.values()) != {1000}:
        raise ValueError("expected balanced Amazon112K source")
    dev_rows = pq.read_table(DEV).to_pylist()
    if len(dev_rows) != 2000 or len({row["target_interest"] for row in dev_rows}) != 112:
        raise ValueError("expected the complete frozen Dev-2K")
    model_name = evaluate.MODEL_NAMES[model_key]
    manifest_path = ROOT / "data" / model_key / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None:
        require_fields(previous, {"model_repo": model_name}, "existing preparation")
    revision = previous["model_revision"] if previous else HfApi().model_info(model_name).sha
    if not revision or len(revision) != 40:
        raise ValueError("model revision could not be frozen")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    lengths = []
    for index, row in enumerate(rows):
        features = tokenize_assistant_only(
            tokenizer, render_training_messages(row, "behavior_interest", "interest_id", mapping)
        )
        length = len(features["input_ids"])
        if length > 8192 or not any(label != -100 for label in features["labels"]):
            raise ValueError(f"{model_key}/{row['id']}: invalid length or empty assistant mask")
        lengths.append(length)
        if (index + 1) % 10000 == 0:
            print(f"{model_key}: training tokenizer audit {index + 1}/112000", flush=True)
    eval_maxima = {}
    for name, module in (("classification", evaluate), ("ranking", evaluate_ranking)):
        maximum = max(len(module.render_prompt_ids(
            tokenizer, row, model_key, "behavior_interest", "interest_id"
        )) for row in dev_rows)
        if maximum > 8192:
            raise ValueError(f"{model_key}/{name}: prompt length {maximum} exceeds 8192")
        eval_maxima[name] = maximum
    selected = longest_balanced_indices(rows, lengths)
    directory = ROOT / "data" / model_key
    directory.mkdir(parents=True, exist_ok=True)
    smoke_path = directory / "smoke.parquet"
    pq.write_table(table.take(selected), smoke_path, compression="zstd")
    smoke_hashes = {
        rows[index]["id"]: tokenized_hash(tokenize_assistant_only(
            tokenizer, render_training_messages(rows[index], "behavior_interest", "interest_id", mapping)
        )) for index in selected
    }
    manifest = {
        "schema_version": 1, "model_key": model_key, "model_repo": model_name,
        "model_revision": revision, "view": "behavior_interest", "label_mode": "interest_id",
        "seed": 20260825, "train_sha256": TRAIN_SHA256, "mapping_sha256": MAPPING_SHA256,
        "dev_sha256": sha256_file(DEV), "train_rows": len(rows), "label_count": len(counts),
        "train_max_tokens": max(lengths), "eval_max_tokens": eval_maxima,
        "max_seq_length": 8192, "token_registration": registration,
        "smoke_rows": len(selected), "smoke_rows_per_class": 2,
        "smoke_sha256": sha256_file(smoke_path), "smoke_tokenized_sha256": smoke_hashes,
        "code_sha256": code_hashes(), "passed": True,
    }
    write_json_atomic(directory / "MANIFEST.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in ("token_registration", "smoke_tokenized_sha256", "code_sha256")}), flush=True)
    return manifest


def selection_fields(manifest: dict, smoke: bool) -> dict:
    if "weighting" not in manifest:
        return {"selection_file": None}
    stage = manifest["weighting"]["smoke" if smoke else "formal"]
    path = (HERE.parent / stage["file"]).resolve()
    if sha256_file(path) != stage["sha256"]:
        raise ValueError("frozen sample weights changed")
    return {"selection_file": str(path), "selection_file_sha256": stage["sha256"],
            "selection_size": 224 if smoke else 112000,
            "selection_weight_column": "sample_weight"}


def repair_ranking_output(directory: Path) -> dict | None:
    path = directory / "details.jsonl"
    if not path.exists():
        return None
    original = path.read_bytes()
    valid = []
    invalid_lines = []
    for number, line in enumerate(original.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            invalid_lines.append(number)
            continue
        if not isinstance(record, dict) or not {"sample_id", "run_fingerprint", "ranking"} <= record.keys():
            raise ValueError("unexpected ranking record schema; manual review required")
        valid.append(line.rstrip(b"\r\n") + b"\n")
    if not invalid_lines:
        return None
    digest = hashlib.sha256(original).hexdigest()
    backup = directory / f"details.corrupt-{digest}.jsonl"
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError("corrupt output backup differs")
    backup.write_bytes(original)
    temporary = path.with_suffix(".repair.tmp")
    temporary.write_bytes(b"".join(valid))
    temporary.replace(path)
    receipt = {"original_sha256": digest, "backup": str(backup),
               "repaired_sha256": sha256_file(path), "retained_records": len(valid),
               "invalid_lines": invalid_lines, "missing_predictions_must_be_regenerated": True}
    write_json_atomic(directory / "OUTPUT_REPAIR.json", receipt)
    return receipt


def audit_run(directory: Path, manifest: dict, model_path: str, smoke: bool) -> dict:
    import torch
    from safetensors.torch import load_file

    train_dir = directory / "train"
    contract = json.loads((train_dir / "train_contract.json").read_text())
    metrics = json.loads((train_dir / "train_metrics.json").read_text())
    expected_rows = 224 if smoke else 112000
    expected_steps = expected_rows // 32
    expected = {
        "model": model_path, "view": "behavior_interest", "label_mode": "interest_id",
        "task": "amazon_new_interest_candidate_free_sft", "assistant_only": True,
        "train_sha256": manifest["smoke_sha256" if smoke else "train_sha256"],
        "label_map_sha256": MAPPING_SHA256, "init_adapter": None, "selection_file": None,
        "epochs": 1.0, "effective_batch_size": 32, "learning_rate": 1e-4,
        "seed": 20260825, "sample_limit": None, "formal_train_rows": expected_rows,
        "loss_mode": "causal_lm", "semantic_label_token_weight": 1.0,
        **selection_fields(manifest, smoke),
    }
    for record in (contract, metrics):
        require_fields(record, expected, "training")
    require_fields(metrics, {"sample_count": expected_rows, "global_step": expected_steps,
                             "max_steps": expected_steps, "epoch": 1.0}, "training completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("training loss is not finite and positive")
    require_fields(json.loads((train_dir / "runtime_config.json").read_text()), {
        "gradient_checkpointing": True, "world_size": 4, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8, "ddp_find_unused_parameters": False,
    }, "runtime")
    adapter = train_dir / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()), {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    }, "adapter")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("missing or nonfinite adapter weights")
    if not any(torch.count_nonzero(value).item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("LoRA output weights did not update")
    rows = pq.read_table(DEV).to_pylist()
    rows = rows[:4] if smoke else rows
    summaries = {}
    for evaluation, beams, decoding in (
        ("classification", 10, "candidate_constrained_beam_search"),
        ("ranking", 20, "candidate_free_direct_beam_search"),
    ):
        eval_dir = directory / evaluation
        eval_contract = json.loads((eval_dir / "run_contract.json").read_text())
        saved = json.loads((eval_dir / "metrics.json").read_text())
        require_fields(eval_contract, {
            "model_key": manifest["model_key"], "model": model_path,
            "view": "behavior_interest", "label_mode": "interest_id",
            "label_map_sha256": MAPPING_SHA256, "dataset_sha256": manifest["dev_sha256"],
            "sample_limit": 4 if smoke else None, "num_beams": beams, "decoding": decoding,
            "do_sample": False, "model_device_map": "balanced",
            "semantic_token_registration": contract["semantic_token_registration"],
        }, evaluation)
        if evaluation == "ranking":
            require_fields(eval_contract, {"decoder_label_constraint": False,
                                          "candidates_in_prompt": False,
                                          "label_file_sha256": TRAIN_SHA256}, evaluation)
        require_fields(saved, {"complete": True, "missing_sample_count": 0,
                               "recorded_sample_count": len(rows), "requested_sample_count": len(rows),
                               "class_count": len({row["target_interest"] for row in rows})}, evaluation)
        details = latest_details(eval_dir / "details.jsonl", eval_contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows}:
            raise ValueError(f"{evaluation}: raw coverage differs")
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            require_fields(detail, {"source_sample_id": row["source_sample_id"],
                                    "target_interest": row["target_interest"]}, evaluation)
            if evaluation == "ranking":
                detail.update(evaluate_ranking.score_ranking(detail["ranking"], row["target_interest"]))
            else:
                detail["correct"] = detail["prediction"] == row["target_interest"]
            ordered.append(detail)
        recalculated = (evaluate_ranking.metrics_from_rankings(ordered, rows, beams)
                        if evaluation == "ranking" else metrics_from_details(ordered, rows))
        require_fields(saved, recalculated, f"{evaluation} raw recomputation")
        if evaluation == "classification":
            require_fields(saved, {"successful_sample_count": len(rows), "request_error_count": 0}, evaluation)
        summaries[evaluation] = {key: value for key, value in recalculated.items()
                                 if not key.startswith("per_class")}
    result = {"passed": True, "smoke": smoke, "model_key": manifest["model_key"],
              "model_repo": manifest["model_repo"], "model_revision": manifest["model_revision"],
              "global_step": expected_steps, "results": summaries,
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors")}
    if "evaluation_repair" in manifest:
        result["evaluation_repair"] = manifest["evaluation_repair"]
    if "weighting" in manifest:
        result["weighting"] = manifest["weighting"]
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def run_stages(manifest: dict, model_path: str, output: Path, smoke_only: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    identity = {"model_key": manifest["model_key"], "model_repo": manifest["model_repo"],
                "model_revision": manifest["model_revision"], "model_path": model_path,
                "code_sha256": manifest["code_sha256"], "train_sha256": TRAIN_SHA256,
                "smoke_sha256": manifest["smoke_sha256"], "gradient_checkpointing": True}
    if "weighting" in manifest:
        identity["weighting"] = manifest["weighting"]
    identity_path = output / "EXECUTION_CONTRACT.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("existing execution contract differs")
    write_json_atomic(identity_path, identity)
    if "evaluation_repair" in manifest:
        repair_path = output / "EVALUATION_REPAIR.json"
        if repair_path.exists() and json.loads(repair_path.read_text()) != manifest["evaluation_repair"]:
            raise ValueError("existing evaluation repair differs")
        write_json_atomic(repair_path, manifest["evaluation_repair"])
    for smoke in (True,) if smoke_only else (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("TRAIN_SELECTION_", "SEMANTIC_LABEL_")) and key != "EVAL_LIMIT"}
        environment.update({
            "MODEL_KEY": manifest["model_key"], "MODEL_NAME": model_path,
            "VIEW": "behavior_interest", "LABEL_MODE": "interest_id", "NPROC": "4",
            "PER_DEVICE_BATCH_SIZE": "1", "GRADIENT_ACCUMULATION_STEPS": "8",
            "GRADIENT_CHECKPOINTING": "1", "AMLT_OUTPUT_DIR": str(directory),
            "TRAIN_FILE": str(ROOT / "data" / manifest["model_key"] / "smoke.parquet" if smoke else TRAIN),
            "EXPECTED_TRAIN_ROWS": "224" if smoke else "112000", "EXPECTED_LABEL_COUNT": "112",
            "LABEL_MAP": str(MAPPING), "LABEL_FILE": str(TRAIN), "EVAL_DATA_DIR": str(DEV.parent),
            "EVAL_DEVICE_MAP": "balanced", "EVAL_VISIBLE_DEVICES": "0,1,2,3",
            "RANKING_NUM_BEAMS": "20", "RANKING_EVAL_BATCH_SIZE": "1",
            "RANKING_DECODING": "direct_beam_search",
            "REQUIRE_QWEN_FAST_PATH": "1" if manifest["model_key"] in ("qwen35_9b", "qwen35_27b") else "0",
        })
        if smoke:
            environment["EVAL_LIMIT"] = "4"
        selection = selection_fields(manifest, smoke)
        if selection["selection_file"] is not None:
            environment.update({"TRAIN_SELECTION_FILE": selection["selection_file"],
                                "TRAIN_SELECTION_SIZE": str(selection["selection_size"]),
                                "TRAIN_SELECTION_WEIGHT_COLUMN": "sample_weight"})
        if manifest.get("repair_incomplete_ranking"):
            repair_ranking_output(directory / "ranking")
        subprocess.run(["bash", 'crossinterest.training/run_train_eval.sh'],
                       cwd=HERE.parent, env=environment, check=True)
        print(json.dumps(audit_run(directory, manifest, model_path, smoke)), flush=True)


def execute(model_key: str, output: Path, smoke_only: bool = False) -> None:
    manifest = json.loads((ROOT / "data" / model_key / "MANIFEST.json").read_text())
    require_fields(manifest, {"model_key": model_key, "model_repo": evaluate.MODEL_NAMES[model_key],
                              "passed": True}, "preparation")
    validate_code_contract(manifest)
    for path, key in ((TRAIN, "train_sha256"), (DEV, "dev_sha256"), (MAPPING, "mapping_sha256"),
                      (ROOT / "data" / model_key / "smoke.parquet", "smoke_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"preparation artifact changed: {path}")
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_repo"], revision=manifest["model_revision"],
                                            trust_remote_code=True)
    mapping = load_label_mapping(MAPPING, expected_count=112)
    if register_semantic_tokens(tokenizer, mapping) != manifest["token_registration"]:
        raise ValueError("runtime semantic token registration differs")
    for row in pq.read_table(ROOT / "data" / model_key / "smoke.parquet").to_pylist():
        actual = tokenized_hash(tokenize_assistant_only(
            tokenizer, render_training_messages(row, "behavior_interest", "interest_id", mapping)))
        if actual != manifest["smoke_tokenized_sha256"][row["id"]]:
            raise ValueError(f"runtime tokenizer or assistant mask changed: {row['id']}")
    model_path = snapshot_download(manifest["model_repo"], revision=manifest["model_revision"],
                                   cache_dir="/tmp/huggingface/hub")
    run_stages(manifest, model_path, output.resolve(), smoke_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--model-key", choices=BACKBONE_KEYS, action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.action == "execute":
        if not args.output_dir or not args.model_key or len(args.model_key) != 1:
            parser.error("execute requires one --model-key and --output-dir")
        execute(args.model_key[0], args.output_dir, args.smoke_only)
        return
    for model_key in args.model_key or BACKBONE_KEYS:
        prepare(model_key)


if __name__ == "__main__":
    main()