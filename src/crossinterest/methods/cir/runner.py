"""Freeze and execute external-balanced or original CIR on either backbone."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.crossfit.train import teacher_command
from crossinterest.methods.mechanism.data import TRAIN_SHA, DEV_SHA, MAPPING_SHA
from crossinterest.methods.mechanism.runner import audit_evaluation, immutable_json, save_table, smoke_indices
from crossinterest.methods.mechanism.runner import sources as mechanism_sources
from crossinterest.methods.mechanism.train import MechanismDataset


ROOT = HERE / "cir"
TRAIN = BASE / "train.parquet"
DEV = BASE / "eval/dev_2k_behavior_interest.parquet"
MAPPING = BASE / "interest_id_semantic_mapping.json"
MODELS = {
    "qwen35_9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "llama31_8b": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
}
VARIANTS = ("external_balanced", "original", "bidirectional_forward", "bidirectional_ordinary",
            "bidirectional_paired", "bidirectional_shuffled")
ORIGINAL_SHA = "aa41f76a330cf1a44f0fac8c0420bd566a2b11861e7367d194f6e19d4c663fc5"


def sources() -> dict:
    return {**mechanism_sources(), **{f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name)
                                     for name in ("cir_train.py", "cir_run.py", "bidirectional_run.py",
                                                  "bidirectional_data.py")}}


def preflight_path(model_key: str, variant: str, auxiliary_epochs: int = 1) -> Path:
    suffix = "" if auxiliary_epochs == 1 else f"_ep{auxiliary_epochs}"
    return ROOT / "data" / f"{model_key}_{variant}{suffix}" / "PREFLIGHT.json"


def preflight(model_key: str, variant: str, auxiliary_epochs: int = 1) -> dict:
    from transformers import AutoTokenizer

    if variant.startswith("bidirectional_"):
        if auxiliary_epochs != 1:
            raise ValueError("directional controls must use one auxiliary epoch")
        from crossinterest.methods.directionality.runner import preflight as directional_preflight
        return directional_preflight(model_key, variant)
    if auxiliary_epochs not in (1, 3) or (auxiliary_epochs != 1 and variant != "external_balanced"):
        raise ValueError("only external-balanced CIR supports the authorized three-epoch condition")
    path = preflight_path(model_key, variant, auxiliary_epochs)
    model, revision = MODELS[model_key]
    for source, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        if sha256_file(source) != expected:
            raise ValueError(f"frozen source changed: {source}")
    if variant == "external_balanced":
        auxiliary = ROOT / "data/external_balanced/train.parquet"
        manifest_path = auxiliary.with_name("MANIFEST.json")
        manifest = json.loads(manifest_path.read_text())
        auxiliary_sha = manifest["train_sha256"]
        require_fields(manifest, {"passed": True, "classes": 112} if "classes" in manifest else {
            "passed": True, "original_n_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA,
            "mapping_sha256": MAPPING_SHA, "max_rows_per_user": 1, "cross_task_input_overlap": 0}, "external data")
        manifest_sha = sha256_file(manifest_path)
    else:
        auxiliary = HERE / "mechanism/data/reconstruction_train.parquet"
        auxiliary_sha, manifest_sha = ORIGINAL_SHA, None
    if sha256_file(auxiliary) != auxiliary_sha:
        raise ValueError("CIR auxiliary data changed")
    identity = {"model_key": model_key, "model_repo": model, "model_revision": revision,
                "variant": variant, "code_sha256": sources(), "data_manifest_sha256": manifest_sha}
    if auxiliary_epochs != 1:
        identity["auxiliary_epochs"] = auxiliary_epochs
    if path.exists():
        prepared = json.loads(path.read_text())
        require_fields(prepared, identity, "CIR preflight")
        for info in prepared["files"].values():
            if sha256_file(HERE.parent / info["path"]) != info["sha256"]:
                raise ValueError("CIR prepared data changed")
        return prepared
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    files, maxima, totals, token_hashes = {}, {}, {}, {}
    import hashlib

    for stage, source in (("auxiliary", auxiliary), ("novel", TRAIN)):
        table = pq.read_table(source)
        counts = Counter(table["target_interest"].to_pylist())
        if len(table) % 4 or set(counts) != set(mapping["category_to_id"]):
            raise ValueError("CIR class coverage or distributed alignment mismatch")
        if stage == "auxiliary" and variant == "external_balanced":
            if len(set(counts.values())) != 1 or not 2 <= max(counts.values()) <= 1000:
                raise ValueError("external CIR must be strictly balanced within the agreed cap")
            if len(set(table["id"].to_pylist())) != len(table) or len(set(table["user_id"].to_pylist())) != len(table):
                raise ValueError("external CIR repeats inputs or users")
        files[stage] = {"path": str(source.relative_to(HERE.parent)), "rows": len(table),
                        "sha256": sha256_file(source), "steps": math.ceil(len(table) / 32)}
        if stage == "auxiliary" and auxiliary_epochs != 1:
            files[stage].update(epochs=auxiliary_epochs, steps=files[stage]["steps"] * auxiliary_epochs)
        dataset = MechanismDataset(source, tokenizer, "behavior_interest", "interest_id", mapping,
                                   8192, None, None, None, len(table), stage=stage, relation=False)
        lengths, digest = [], hashlib.sha256()
        for index in range(len(dataset)):
            encoded = dataset[index]
            lengths.append(len(encoded["input_ids"]))
            digest.update(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
            if (index + 1) % 10000 == 0:
                print(f"CIR preflight {model_key}/{variant}/{stage} {index + 1}/{len(dataset)}", flush=True)
        longest_first = table.take(sorted(range(len(table)), key=lambda index: (-lengths[index], index)))
        files[f"smoke_{stage}"] = save_table(path.parent / f"smoke_{stage}.parquet",
                                            longest_first.take(smoke_indices(longest_first)))
        if stage == "auxiliary" and auxiliary_epochs != 1:
            smoke_info = files[f"smoke_{stage}"]
            smoke_info.update(epochs=auxiliary_epochs, steps=smoke_info["steps"] * auxiliary_epochs)
        maxima[stage], totals[stage], token_hashes[stage] = max(lengths), sum(lengths), digest.hexdigest()
    prepared = {"passed": True, "schema_version": 1, **identity, "files": files,
                "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
                "token_registration": registration, "max_train_tokens": maxima,
                "train_tokens_per_epoch": totals, "tokenized_sha256": token_hashes,
                "distributed_padding_duplicates": 0, "smoke_selection": "longest_two_per_class_per_stage",
                "evaluation": {"rows": 2000, "beams": 20, "candidate_free": True, "do_sample": False}}
    immutable_json(path, prepared)
    return prepared


def training_command(model_path: str, stage: str, directory: Path, smoke: bool,
                     prepared: dict, init_adapter: Path | None = None) -> list[str]:
    info = prepared["files"][("smoke_" if smoke else "") + stage]
    command = teacher_command(model_path, HERE.parent / info["path"], directory, info["rows"])
    command[command.index("--epochs") + 1] = str(info.get("epochs", 1))
    command[command.index('crossinterest.training.train')] = 'crossinterest.methods.cir.train'
    if prepared["model_key"] == "llama31_8b":
        command.remove("--require-qwen-fast-path")
    command.extend(["--stage", stage, "--cir-preflight", str(preflight_path(
        prepared["model_key"], prepared["variant"], prepared.get("auxiliary_epochs", 1)))])
    if init_adapter is not None:
        command.extend(["--init-adapter", str(init_adapter)])
    if smoke:
        command.append("--smoke")
    return command


def audit_training(directory: Path, stage: str, smoke: bool, prepared: dict, model_path: str) -> dict:
    import torch
    from safetensors.torch import load_file

    info = prepared["files"][("smoke_" if smoke else "") + stage]
    epochs = info.get("epochs", 1)
    metrics = json.loads((directory / "train_metrics.json").read_text())
    expected = {"model": model_path, "train_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
                "view": "behavior_interest", "label_mode": "interest_id", "epochs": float(epochs), "learning_rate": 1e-4,
                "effective_batch_size": 32, "loss_mode": "causal_lm", "selection_file": None,
                "sample_limit": None, "semantic_label_token_weight": 1.0}
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, expected, "CIR stage")
        require_fields(record["experiment_metadata"], {"stage": stage, "smoke": smoke, "relation_head": False,
            "variant": prepared["variant"], "model_key": prepared["model_key"],
            "preflight_sha256": sha256_file(preflight_path(prepared["model_key"], prepared["variant"],
                                                        prepared.get("auxiliary_epochs", 1)))}, "CIR identity")
        previous = directory.parent / "auxiliary/final_adapter/adapter_model.safetensors"
        require_fields(record, {"init_adapter_model_sha256": sha256_file(previous) if stage == "novel" else None}, "CIR weights handoff")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"], "epoch": float(epochs),
                             "sample_count": info["rows"]}, "CIR completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid CIR training loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    require_fields(runtime, {"world_size": 4, "per_device_batch_size": 1, "gradient_accumulation_steps": 8,
                            "gradient_checkpointing": True, "ddp_find_unused_parameters": False}, "CIR runtime")
    if prepared["model_key"] == "qwen35_9b":
        require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen kernels")
    mechanism = json.loads((directory / "MECHANISM_RUNTIME.json").read_text())
    require_fields(mechanism, {"stage": stage, "relation_trainable_parameters": 0}, "CIR runtime")
    if stage == "auxiliary":
        require_fields(mechanism, {"model_accepts_loss_kwargs": False,
                                  "auxiliary_mean_loss_scaled_by_trainer": True}, "CIR loss scaling")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "CIR LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid CIR adapter")
    if not any(value.count_nonzero().item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("CIR adapter did not update")
    result = {"passed": True, "stage": stage, "rows": info["rows"], "steps": info["steps"],
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"), "relation_sha256": None,
              "train_runtime_seconds": metrics["train_runtime"], "mechanism_runtime": mechanism}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def execute(model_key: str, variant: str, output: Path, auxiliary_epochs: int = 1) -> None:
    from huggingface_hub import snapshot_download

    prepared = preflight(model_key, variant, auxiliary_epochs)
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {**prepared, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        previous, stages = None, {}
        for stage in ("auxiliary", "novel"):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, stage, stage_dir, smoke, prepared, previous),
                               cwd=HERE.parent, env=environment, check=True)
            stages[stage] = audit_training(stage_dir, stage, smoke, prepared, model_path)
            previous = stage_dir / "final_adapter"
            if prepared.get("directional") and stage == "auxiliary" and not smoke:
                subprocess.run([sys.executable, "-m", 'crossinterest.methods.directionality.runner',
                                "--model", model_path, "--adapter", str(previous),
                                "--output", str(directory / "INVERSE_PROBE.json")],
                               cwd=HERE.parent, env=environment, check=True)
        for evaluation in ("classification", "ranking"):
            command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate',
                "--evaluation", evaluation, "--model-key", model_key, "--model", model_path,
                "--adapter", str(previous), "--view", "behavior_interest", "--label-mode", "interest_id",
                "--label-map", str(MAPPING), "--data-dir", str(DEV.parent), "--expected-label-count", "112",
                "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                                "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        result = audit_evaluation(directory, "reconstruction", smoke, stages["novel"])
        result.update(model_key=model_key, variant=variant, stage_audits=stages)
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps(result), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "model_key": model_key, "variant": variant,
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "execute"))
    parser.add_argument("--model-key", required=True, choices=MODELS)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--auxiliary-epochs", type=int, choices=(1, 3), default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        prepared = preflight(args.model_key, args.variant, args.auxiliary_epochs)
        print(json.dumps({key: prepared[key] for key in ("passed", "model_key", "variant", "files", "max_train_tokens")}))
    elif args.output_dir is None:
        parser.error("execute requires --output-dir")
    else:
        execute(args.model_key, args.variant, args.output_dir.resolve(), args.auxiliary_epochs)


if __name__ == "__main__":
    main()