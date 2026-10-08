"""Freeze, smoke, train and independently audit the three mechanism conditions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import latest_details, metrics_from_details, write_json_atomic
from crossinterest.training import evaluate_ranking
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.backbones.runner import code_hashes
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.crossfit.train import teacher_command
from crossinterest.methods.mechanism.data import ROOT, TRAIN_SHA, DEV_SHA, MAPPING_SHA
from crossinterest.methods.mechanism.train import ARMS, MechanismDataset


DATA = ROOT / "data"
PREFLIGHT = DATA / "PREFLIGHT.json"
TRAIN = BASE / "train.parquet"
DEV = BASE / "eval/dev_2k_behavior_interest.parquet"
MAPPING = BASE / "interest_id_semantic_mapping.json"
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def sources() -> dict:
    extra = ["mechanism_data.py", "mechanism_model.py", "mechanism_train.py",
             "mechanism_evaluate.py", "mechanism_run.py", "crossfit.py", "audit_results.py", "build_dataset.py"]
    return {**code_hashes(), **{f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name) for name in extra}}


def immutable_json(path: Path, value: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen contract differs: {path}")
    write_json_atomic(path, value)


def save_table(path: Path, table: pa.Table) -> dict:
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise ValueError(f"frozen rows differ: {path}")
    else:
        pq.write_table(table, path, compression="zstd")
    return {"path": str(path.relative_to(HERE.parent)), "rows": len(table),
            "sha256": sha256_file(path), "steps": math.ceil(len(table) / 32)}


def smoke_indices(table: pa.Table) -> list[int]:
    counts = Counter()
    selected = []
    for index, category in enumerate(table["target_interest"].to_pylist()):
        if counts[category] < 2:
            selected.append(index)
            counts[category] += 1
    if len(counts) != 112 or set(counts.values()) != {2}:
        raise ValueError("smoke must cover every semantic label twice")
    return selected


def preflight() -> dict:
    from transformers import AutoTokenizer

    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"original dataset changed: {path}")
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    auxiliary = DATA / "reconstruction.parquet"
    if sha256_file(auxiliary) != manifest["aux_sha256"]:
        raise ValueError("auxiliary source changed")
    if PREFLIGHT.exists():
        prepared = json.loads(PREFLIGHT.read_text())
        require_fields(prepared, {"code_sha256": sources(), "data_manifest_sha256": sha256_file(DATA / "MANIFEST.json")}, "preflight")
        for info in prepared["files"].values():
            if sha256_file(HERE.parent / info["path"]) != info["sha256"]:
                raise ValueError("prepared data changed")
        return prepared
    table = pq.read_table(auxiliary)
    remainder = len(table) % 4
    excluded = table.slice(len(table) - remainder)["id"].to_pylist() if remainder else []
    table = table.slice(0, len(table) - remainder)
    if len(set(table["target_interest"].to_pylist())) != 112:
        raise ValueError("distributed alignment removed a class")
    novel = pq.read_table(TRAIN)
    files = {
        "auxiliary": save_table(DATA / "reconstruction_train.parquet", table),
        "novel": {"path": str(TRAIN.relative_to(HERE.parent)), "rows": len(novel), "sha256": TRAIN_SHA, "steps": 3500},
        "smoke_auxiliary": save_table(DATA / "smoke_auxiliary.parquet", table.take(smoke_indices(table))),
        "smoke_novel": save_table(DATA / "smoke_novel.parquet", novel.take(smoke_indices(novel))),
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    maxima = {}
    token_counts = {}
    for stage in ("auxiliary", "novel"):
        info = files[stage]
        dataset = MechanismDataset(HERE.parent / info["path"], tokenizer, "behavior_interest", "interest_id",
            mapping, 8192, None, None, None, info["rows"], stage=stage, relation=True)
        maximum = total = 0
        for index in range(len(dataset)):
            encoded = dataset[index]
            length = len(encoded["input_ids"])
            maximum = max(maximum, length)
            total += length
            if (index + 1) % 10000 == 0:
                print(f"mechanism preflight {stage} {index + 1}/{len(dataset)} max_tokens={maximum}", flush=True)
        maxima[stage] = maximum
        token_counts[stage] = total
    result = {"passed": True, "schema_version": 1, "model_repo": MODEL, "model_revision": REVISION,
        "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
        "token_registration": registration, "data_manifest_sha256": sha256_file(DATA / "MANIFEST.json"),
        "code_sha256": sources(), "files": files, "max_train_tokens": maxima,
        "train_tokens_per_epoch": token_counts, "ddp_alignment_excluded_aux_ids": excluded,
        "distributed_padding_duplicates": 0, "auxiliary_class_counts": dict(sorted(Counter(table["target_interest"].to_pylist()).items())),
        "evaluation": {"rows": 2000, "classes": 112, "beams": 20, "primary": "macro_ndcg_at_20",
                       "recall_gate": "macro_recall_at_20 >= frozen N-only", "max_seq_length": 8192}}
    immutable_json(PREFLIGHT, result)
    return result


def training_command(model_path: str, arm: str, stage: str, directory: Path,
                     smoke: bool, prepared: dict, init_adapter: Path | None = None) -> list[str]:
    info = prepared["files"][("smoke_" if smoke else "") + stage]
    command = teacher_command(model_path, HERE.parent / info["path"], directory, info["rows"])
    command[command.index('crossinterest.training.train')] = 'crossinterest.methods.mechanism.train'
    command.extend(["--stage", stage, "--arm", arm, "--mechanism-preflight", str(PREFLIGHT)])
    if init_adapter is not None:
        command.extend(["--init-adapter", str(init_adapter)])
    if smoke:
        command.append("--smoke")
    return command


def audit_training(directory: Path, arm: str, stage: str, smoke: bool, prepared: dict, model_path: str) -> dict:
    import torch
    from safetensors.torch import load_file

    info = prepared["files"][("smoke_" if smoke else "") + stage]
    metrics = json.loads((directory / "train_metrics.json").read_text())
    contract = json.loads((directory / "train_contract.json").read_text())
    for record in (metrics, contract):
        require_fields(record, {"model": model_path, "train_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
            "view": "behavior_interest", "label_mode": "interest_id", "epochs": 1.0, "learning_rate": 1e-4,
            "selection_file": None, "sample_limit": None, "effective_batch_size": 32,
            "loss_mode": "causal_lm", "semantic_label_token_weight": 1.0}, "mechanism stage")
        require_fields(record["experiment_metadata"], {"arm": arm, "stage": stage, "smoke": smoke,
            "preflight_sha256": sha256_file(PREFLIGHT), "relation_head": arm != "reconstruction"}, "stage identity")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"], "epoch": 1.0,
                             "sample_count": info["rows"]}, "complete stage")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("nonfinite or nonpositive training loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    require_fields(runtime, {"world_size": 4, "per_device_batch_size": 1, "gradient_accumulation_steps": 8,
                            "gradient_checkpointing": True, "ddp_find_unused_parameters": False}, "runtime")
    require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen kernels")
    mechanism_runtime = json.loads((directory / "MECHANISM_RUNTIME.json").read_text())
    require_fields(mechanism_runtime, {"stage": stage}, "mechanism runtime")
    if stage == "auxiliary":
        require_fields(mechanism_runtime, {"model_accepts_loss_kwargs": False,
            "auxiliary_mean_loss_scaled_by_trainer": True}, "auxiliary gradient accumulation")
    adapter = directory / "final_adapter"
    config = json.loads((adapter / "adapter_config.json").read_text())
    require_fields(config, {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid adapter weights")
    if not any(value.count_nonzero() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("LoRA did not train")
    result = {"passed": True, "stage": stage, "rows": info["rows"], "steps": info["steps"],
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
              "train_runtime_seconds": metrics["train_runtime"], "relation_sha256": None}
    result["mechanism_runtime"] = mechanism_runtime
    if arm != "reconstruction":
        head = load_file(str(adapter / "interest_relation.safetensors"))
        if not all(torch.isfinite(value).all() for value in head.values()) or head["beta"].abs().item() == 0:
            raise ValueError("relation head is nonfinite or never activated")
        initial = load_file(str(directory / "initial_relation/interest_relation.safetensors"))
        if torch.equal(head["readout.weight"], initial["readout.weight"]):
            raise ValueError("relation readout did not update")
        result.update(relation_sha256=sha256_file(adapter / "interest_relation.safetensors"), beta=head["beta"].item())
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def audit_evaluation(directory: Path, arm: str, smoke: bool, training: dict) -> dict:
    rows = pq.read_table(DEV).to_pylist()
    rows = rows[:4] if smoke else rows
    summaries = {}
    for evaluation, beams in (("classification", 10), ("ranking", 20)):
        root = directory / evaluation
        contract = json.loads((root / "run_contract.json").read_text())
        require_fields(contract, {"dataset_sha256": DEV_SHA, "label_map_sha256": MAPPING_SHA,
            "view": "behavior_interest", "label_mode": "interest_id", "num_beams": beams,
            "sample_limit": 4 if smoke else None, "do_sample": False}, "original evaluation")
        if evaluation == "ranking":
            require_fields(contract, {"decoder_label_constraint": False, "candidates_in_prompt": False,
                                     "decoding": "candidate_free_direct_beam_search"}, "original20beam")
        sidecar = json.loads((root / "MECHANISM_CONTRACT.json").read_text())
        require_fields(sidecar, {"relation": arm != "reconstruction", "adapter_weight_sha256": training["adapter_sha256"],
                                "head_weight_sha256": training["relation_sha256"]}, "evaluated checkpoint")
        details = latest_details(root / "details.jsonl", contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows}:
            raise ValueError("raw evaluation coverage mismatch")
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            require_fields(detail, {"source_sample_id": row["source_sample_id"], "target_interest": row["target_interest"]}, "raw label")
            if evaluation == "ranking":
                detail.update(evaluate_ranking.score_ranking(detail["ranking"], row["target_interest"]))
            else:
                detail["correct"] = detail["prediction"] == row["target_interest"]
            ordered.append(detail)
        calculated = (evaluate_ranking.metrics_from_rankings(ordered, rows, beams) if evaluation == "ranking" else
                      metrics_from_details(ordered, rows))
        saved = json.loads((root / "metrics.json").read_text())
        require_fields(saved, calculated, "recomputed metrics")
        require_fields(saved, {"complete": True, "missing_sample_count": 0, "recorded_sample_count": len(rows)}, "coverage")
        summaries[evaluation] = {key: value for key, value in calculated.items() if not key.startswith("per_class")}
    result = {"passed": True, "arm": arm, "smoke": smoke, "training": training, "results": summaries,
              "raw_ranking_sha256": sha256_file(directory / "ranking/details.jsonl")}
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def execute(arm: str, output: Path) -> None:
    from huggingface_hub import snapshot_download

    prepared = preflight()
    model_path = snapshot_download(MODEL, revision=REVISION, cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {**prepared, "arm": arm, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        previous = None
        stage_audits = {}
        for stage in (("novel",) if arm == "structure" else ("auxiliary", "novel")):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, arm, stage, stage_dir, smoke, prepared, previous),
                               cwd=HERE.parent, env=environment, check=True)
            stage_audits[stage] = audit_training(stage_dir, arm, stage, smoke, prepared, model_path)
            previous = stage_dir / "final_adapter"
        for evaluation in ("classification", "ranking"):
            command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate',
                "--evaluation", evaluation, "--model-key", "qwen35_9b", "--model", model_path,
                "--adapter", str(previous), "--view", "behavior_interest", "--label-mode", "interest_id",
                "--label-map", str(MAPPING), "--data-dir", str(DEV.parent), "--expected-label-count", "112",
                "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if arm != "reconstruction":
                command.append("--relation")
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                                "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        result = audit_evaluation(directory, arm, smoke, stage_audits["novel"])
        result["stage_audits"] = stage_audits
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps(result), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "arm": arm,
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "execute"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
        print(json.dumps({key: result[key] for key in ("passed", "max_train_tokens", "train_tokens_per_epoch", "files")}))
    elif args.arm is None or args.output_dir is None:
        parser.error("execute requires --arm and --output-dir")
    else:
        execute(args.arm, args.output_dir.resolve())


if __name__ == "__main__":
    main()