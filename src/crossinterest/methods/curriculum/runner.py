"""Preflight and execute the two frozen500-per-class curriculum conditions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import metrics_from_details, write_json_atomic
from crossinterest.training import evaluate_ranking
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.training.train import tokenize_assistant_only
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.backbones.runner import DEV, MAPPING, MAPPING_SHA256, TRAIN, TRAIN_SHA256, code_hashes, latest_details
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.crossfit.train import teacher_command, validate_teacher
from crossinterest.methods.curriculum.model import ROOT, balanced_pools, render_messages


DATA = ROOT / "seen500-novel500"


def source_hashes() -> dict:
    return {**code_hashes(), **{name: sha256_file(HERE / name) for name in (
        "curriculum.py", "curriculum_train.py", "curriculum_run.py", "crossfit.py", "audit_results.py")}}


def immutable_json(path: Path, value: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen contract changed: {path}")
    write_json_atomic(path, value)


def preflight(data: Path = DATA) -> dict:
    from transformers import AutoTokenizer

    manifest = json.loads((data / "MANIFEST.json").read_text())
    require_fields(manifest, {"arms": ["seen_then_novel", "shuffle"], "formal_pool_frozen": True}, "course")
    if manifest.get("full_novel"):
        require_fields(manifest, {"schema_version": 2, "label_count": 112,
                                 "seen_quota_per_class": 500, "novel_quota_per_class": 1000}, "full novel")
    else:
        require_fields(manifest, {"label_count": 93, "quota_per_class_per_task": 500}, "legacy course")
    if sha256_file(TRAIN) != TRAIN_SHA256 or sha256_file(MAPPING) != MAPPING_SHA256:
        raise ValueError("frozen source changed")
    if sha256_file(DEV) != manifest["evaluation"]["sha256"]:
        raise ValueError("original novel Dev changed")
    for name, info in manifest["files"].items():
        if sha256_file(data / f"{name}.parquet") != info["sha256"]:
            raise ValueError("course source changed")
    backbone = json.loads((HERE / "backbone/data/qwen35_9b/MANIFEST.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(backbone["model_repo"], revision=backbone["model_revision"],
                                            trust_remote_code=True)
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    maxima = {}
    pools = {}
    for task in ("seen", "novel"):
        rows = pq.read_table(data / f"{task}.parquet").to_pylist()
        maximum = 0
        for row in rows:
            encoded = tokenize_assistant_only(tokenizer, render_messages(row, mapping))
            if len(encoded["input_ids"]) > 8192:
                raise ValueError("course sequence exceeds8192; truncation forbidden")
            target = registration["token_ids"][mapping["category_to_id"][row["target_interest"]]]
            if encoded["labels"].count(target) != 1:
                raise ValueError("target must be supervised exactly once")
            maximum = max(maximum, len(encoded["input_ids"]))
        maxima[task] = maximum
        pools[task] = rows
        print(f"course task={task} checked={len(rows)} max_tokens={maximum}", flush=True)
    smoke = balanced_pools(pools["seen"], pools["novel"], 2)
    smoke_files = {}
    schema = pq.read_schema(data / "seen.parquet")
    for arm, rows in smoke["orders"].items():
        path = data / f"smoke_{arm}.parquet"
        table = pa.Table.from_pylist(rows, schema=schema)
        if path.exists() and not pq.read_table(path).equals(table):
            raise ValueError("smoke course changed")
        if not path.exists():
            pq.write_table(table, path, compression="zstd")
        smoke_files[arm] = {"rows": len(rows), "sha256": sha256_file(path)}
    result = {"passed": True, "model_repo": backbone["model_repo"],
              "model_revision": backbone["model_revision"], "token_registration": registration,
              "course_manifest_sha256": sha256_file(data / "MANIFEST.json"),
              "source_sha256": TRAIN_SHA256, "mapping_sha256": MAPPING_SHA256,
              "dev_sha256": sha256_file(DEV), "max_train_tokens": maxima,
              "smoke_files": smoke_files, "code_sha256": source_hashes()}
    immutable_json(data / "PREFLIGHT.json", result)
    return result


def train_command(model_path: str, train_path: Path, output: Path, count: int,
                  arm: str, smoke: bool, data: Path = DATA) -> list[str]:
    command = teacher_command(model_path, train_path, output, count)
    module_index = command.index('crossinterest.training.train')
    command[module_index] = 'crossinterest.methods.curriculum.train'
    command.extend(["--arm", arm, "--course-manifest", str(data / "MANIFEST.json")])
    if smoke:
        command.append("--smoke")
    return command


def audit_run(directory: Path, train_path: Path, model_path: str, arm: str, smoke: bool,
              data: Path = DATA) -> dict:
    manifest = json.loads((data / "MANIFEST.json").read_text())
    label_count = len(manifest.get("seen_labels", manifest["labels"])) if smoke else manifest["label_count"]
    count = label_count * 4 if smoke else manifest["files"][arm]["rows"]
    adapter_sha = validate_teacher(directory / "train", model_path, train_path, count)
    contract = json.loads((directory / "train/train_contract.json").read_text())
    require_fields(contract, {"label_count": label_count, "task": "amazon_seen_novel_curriculum_sft"}, "course")
    require_fields(contract["experiment_metadata"], {"arm": arm, "smoke": smoke,
        "training_label_count": label_count, "semantic_label_count": 112,
        "sampler": "sequential_rows_sharded_by_accelerate_without_padding",
        "optimizer_reset_between_tasks": False, "scheduler_reset_between_tasks": False}, "order")
    rows = pq.read_table(DEV).to_pylist()
    rows = rows[:4] if smoke else rows
    summaries = {}
    for evaluation, beams, decoding in (("classification", 10, "candidate_constrained_beam_search"),
                                       ("ranking", 20, "candidate_free_direct_beam_search")):
        root = directory / evaluation
        eval_contract = json.loads((root / "run_contract.json").read_text())
        saved = json.loads((root / "metrics.json").read_text())
        require_fields(eval_contract, {"model_key": "qwen35_9b", "model": model_path,
            "view": "behavior_interest", "label_mode": "interest_id",
            "label_map_sha256": MAPPING_SHA256, "dataset_sha256": sha256_file(DEV),
            "sample_limit": 4 if smoke else None, "num_beams": beams, "decoding": decoding,
            "do_sample": False, "model_device_map": "balanced",
            "semantic_token_registration": contract["semantic_token_registration"]}, "unchanged novel evaluation")
        if evaluation == "ranking":
            require_fields(eval_contract, {"decoder_label_constraint": False, "candidates_in_prompt": False,
                                            "label_file_sha256": TRAIN_SHA256}, "ranking")
        require_fields(saved, {"complete": True, "missing_sample_count": 0,
            "recorded_sample_count": len(rows), "requested_sample_count": len(rows),
            "class_count": len({row["target_interest"] for row in rows})}, "evaluation coverage")
        details = latest_details(root / "details.jsonl", eval_contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows}:
            raise ValueError("evaluation raw IDs differ")
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            require_fields(detail, {"source_sample_id": row["source_sample_id"],
                                    "target_interest": row["target_interest"]}, "raw target")
            if evaluation == "ranking":
                detail.update(evaluate_ranking.score_ranking(detail["ranking"], row["target_interest"]))
            else:
                detail["correct"] = detail["prediction"] == row["target_interest"]
            ordered.append(detail)
        metrics = (evaluate_ranking.metrics_from_rankings(ordered, rows, beams)
                   if evaluation == "ranking" else metrics_from_details(ordered, rows))
        require_fields(saved, metrics, "raw metric recomputation")
        if evaluation == "classification":
            require_fields(saved, {"successful_sample_count": len(rows), "request_error_count": 0}, "health")
        summaries[evaluation] = {key: value for key, value in metrics.items() if not key.startswith("per_class")}
    result = {"passed": True, "arm": arm, "smoke": smoke, "train_rows": count,
              "global_step": (count + 31) // 32, "adapter_sha256": adapter_sha, "results": summaries}
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def execute(arm: str, output: Path, data: Path = DATA) -> None:
    from huggingface_hub import snapshot_download

    prepared = json.loads((data / "PREFLIGHT.json").read_text())
    require_fields(prepared, {"passed": True, "code_sha256": source_hashes(),
        "course_manifest_sha256": sha256_file(data / "MANIFEST.json"),
        "source_sha256": sha256_file(TRAIN), "mapping_sha256": sha256_file(MAPPING),
        "dev_sha256": sha256_file(DEV)}, "preflight")
    manifest = json.loads((data / "MANIFEST.json").read_text())
    if arm not in manifest["arms"]:
        raise ValueError("unsupported arm")
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"],
                                   cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {**prepared, "arm": arm, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update({"TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        path = data / f"{'smoke_' if smoke else ''}{arm}.parquet"
        expected = prepared["smoke_files"][arm] if smoke else manifest["files"][arm]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError("course training rows changed")
        if not (directory / "train/train_metrics.json").exists():
            subprocess.run(train_command(model_path, path, directory / "train", expected["rows"], arm, smoke, data),
                           cwd=HERE.parent, env=environment, check=True)
        validate_teacher(directory / "train", model_path, path, expected["rows"])
        for evaluation in ("classification", "ranking"):
            module = 'crossinterest.training.evaluate' + ("_ranking" if evaluation == "ranking" else "")
            command = [sys.executable, "-m", module, "--model-key", "qwen35_9b", "--model", model_path,
                "--adapter", str(directory / "train/final_adapter"), "--view", "behavior_interest",
                "--label-mode", "interest_id", "--label-map", str(MAPPING), "--data-dir", str(DEV.parent),
                "--expected-label-count", "112", "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                                "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        print(json.dumps(audit_run(directory, path, model_path, arm, smoke, data)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "execute"))
    parser.add_argument("--arm", choices=("seen_then_novel", "shuffle"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args.data_dir.resolve())
    elif args.arm is None or args.output_dir is None:
        parser.error("execute requires --arm and --output-dir")
    else:
        execute(args.arm, args.output_dir.resolve(), args.data_dir.resolve())


if __name__ == "__main__":
    main()