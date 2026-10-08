"""Run the approved matched-background frequency comparison with N-only SFT."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.training.train import AmazonSftDataset
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file
from crossinterest.methods.crossfit.train import teacher_command, validate_teacher
from crossinterest.methods.frequency.data import FROZEN_FILES
from crossinterest.methods.mechanism.runner import audit_evaluation, immutable_json, save_table, smoke_indices
from crossinterest.methods.mechanism.runner import sources as shared_sources


ROOT = HERE / "frequency"
DATA = ROOT / "paired"
PREFLIGHT = ROOT / "PREFLIGHT_RECOVERY_V3.json"
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ARMS = ("once", "at_least_five")
MAPPING = BASE / "interest_id_semantic_mapping.json"
DEV = BASE / "eval/dev_2k_behavior_interest.parquet"


def sources():
    return {**shared_sources(), **{f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name)
                                  for name in ("frequency_data.py", "frequency_run.py")}}


def preflight():
    from transformers import AutoTokenizer

    for relative, expected in FROZEN_FILES.items():
        if sha256_file(BASE / relative) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    audit = json.loads((DATA / "INDEPENDENT_AUDIT.json").read_text())
    require_fields(audit, {"passed": True, "manifest_sha256": sha256_file(DATA / "MANIFEST.json"),
                          "rows_per_arm": 112000, "classes": 112, "quota": 1000,
                          "common_identical_original_rows": 82758, "replacement_rows_per_arm": 29242,
                          "within_arm_duplicate_ids": 0, "dev_user_overlap": 0}, "paired data")
    identity = {"model_repo": MODEL, "model_revision": REVISION, "code_sha256": sources(),
                "data_manifest_sha256": sha256_file(DATA / "MANIFEST.json"),
                "data_audit_sha256": sha256_file(DATA / "INDEPENDENT_AUDIT.json"),
                "protocol": "home112-paired-frequency-N-only-v1", "init_adapter": None,
                "training_stages": ["novel"], "frozen_inputs": FROZEN_FILES}
    if PREFLIGHT.exists():
        prepared = json.loads(PREFLIGHT.read_text())
        require_fields(prepared, identity, "frequency preflight")
        for info in prepared["files"].values():
            if sha256_file(HERE.parent / info["path"]) != info["sha256"]:
                raise ValueError("prepared data changed")
        return prepared
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    files, token_stats = {}, {}
    for arm in ARMS:
        source = DATA / arm / "train.parquet"
        table = pq.read_table(source)
        if sha256_file(source) != manifest["arms"][arm]["sha256"]:
            raise ValueError("paired training file hash mismatch")
        if Counter(table["target_interest"].to_pylist()) != Counter({category: 1000 for category in mapping["category_to_id"]}):
            raise ValueError("paired training balance changed")
        if any(name.startswith("future_") for name in table.column_names):
            raise ValueError("future selection fields in training schema")
        files[arm] = {"path": str(source.relative_to(HERE.parent)), "rows": len(table),
                      "sha256": sha256_file(source), "steps": 3500}
        dataset = AmazonSftDataset(source, tokenizer, "behavior_interest", "interest_id", mapping,
                                   8192, None, None, None, len(table))
        lengths, digest = [], hashlib.sha256()
        for index in range(len(dataset)):
            encoded = dataset[index]
            lengths.append(len(encoded["input_ids"]))
            digest.update(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            if (index + 1) % 20000 == 0:
                print(f"frequency preflight {arm} {index + 1}/{len(dataset)}", flush=True)
        longest_first = table.take(sorted(range(len(table)), key=lambda index: (-lengths[index], index)))
        files[f"smoke_{arm}"] = save_table(DATA / arm / "smoke.parquet", longest_first.take(smoke_indices(longest_first)))
        token_stats[arm] = {"maximum": max(lengths), "total": sum(lengths), "sha256": digest.hexdigest()}
    result = {"passed": True, **identity, "files": files, "token_stats": token_stats,
              "token_registration": registration, "smoke": "longest two per class, 224 rows, 7 steps; separate fresh initialization",
              "evaluation": {"rows": 2000, "beams": 20, "candidate_free": True, "do_sample": False}}
    immutable_json(PREFLIGHT, result)
    return result


def training_command(model_path, arm, output, smoke, prepared):
    if arm not in ARMS:
        raise ValueError("unknown frequency condition")
    info = prepared["files"][("smoke_" if smoke else "") + arm]
    return teacher_command(model_path, HERE.parent / info["path"], output, info["rows"])


def audit_training(output, arm, smoke, prepared, model_path):
    info = prepared["files"][("smoke_" if smoke else "") + arm]
    weight_sha = validate_teacher(output, model_path, HERE.parent / info["path"], info["rows"])
    runtime = json.loads((output / "runtime_config.json").read_text())
    require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen kernels")
    contract = json.loads((output / "train_contract.json").read_text())
    require_fields(contract, {"init_adapter_model_sha256": None, "semantic_label_token_weight": 1.0,
                              "loss_mode": "causal_lm"}, "N-only contract")
    result = {"passed": True, "condition": arm, "smoke": smoke, "stage": "novel",
              "rows": info["rows"], "steps": info["steps"], "adapter_sha256": weight_sha,
              "relation_sha256": None, "init_adapter": None}
    write_json_atomic(output / "STAGE_AUDIT.json", result)
    return result


def execution_contract(output, contract):
    path = output / "EXECUTION_CONTRACT.json"
    if not path.exists():
        immutable_json(path, contract)
        return
    previous = json.loads(path.read_text())
    if previous == contract:
        return
    if {key: value for key, value in previous.items() if key != "code_sha256"} != {
        key: value for key, value in contract.items() if key != "code_sha256"
    }:
        raise ValueError("recovery would change the training or data contract")
    before, after = previous["code_sha256"], contract["code_sha256"]
    allowed = {
        "eval_interest_id_minionerec/crossfit.py": "059ff2ea7be0907f5faf98f72c0f1e231801cd9dfb844da58e91787af2a448cc",
        "eval_interest_id_minionerec/frequency_run.py": "a95519f389a0f2a1108accb2db8ff535bd431ccafd0cb11aa80f0565f4456d4e",
        'crossinterest.training/evaluate.py': "2bf39a81d8e882ba90bd135648ce4725d5b985293195bb95d4ac2b75f56a0d6f",
        'crossinterest.training/evaluate_ranking.py': "dfcd1033de7f268df1fa0b4a9497f5f3f4962355b98795c97fc24a0995f761ff",
    }
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if changed != set(allowed) or any(before.get(name) != digest for name, digest in allowed.items()):
        raise ValueError("recovery contains changes beyond the audited completion compatibility fix")
    immutable_json(output / "AUDIT_RECOVERY_CONTRACT.json", {
        "reason": "verify missing metrics steps from matching final checkpoint; restore argv/device-map evaluation interfaces; no retraining",
        "previous_execution_contract_sha256": sha256_file(path),
        "changed_sources": sorted(changed), "execution_contract": contract,
    })


def execute(arm, output):
    from huggingface_hub import snapshot_download

    prepared = preflight()
    model_path = snapshot_download(MODEL, revision=REVISION, cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    execution_contract(output, {**prepared, "condition": arm, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        stage = directory / "novel"
        if not (stage / "train_metrics.json").exists():
            subprocess.run(training_command(model_path, arm, stage, smoke, prepared),
                           cwd=HERE.parent, env=environment, check=True)
        training = audit_training(stage, arm, smoke, prepared, model_path)
        for evaluation in ("classification", "ranking"):
            command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate',
                       "--evaluation", evaluation, "--model-key", "qwen35_9b", "--model", model_path,
                       "--adapter", str(stage / "final_adapter"), "--view", "behavior_interest",
                       "--label-mode", "interest_id", "--label-map", str(MAPPING), "--data-dir", str(DEV.parent),
                       "--expected-label-count", "112", "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(BASE / "train.parquet"), "--num-beams", "20",
                                "--eval-batch-size", "1", "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        result = audit_evaluation(directory, "reconstruction", smoke, training)
        result.update(arm=arm, protocol=prepared["protocol"], training_stages=["novel"])
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps(result), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "condition": arm,
                      "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
                      "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "execute"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "preflight":
        prepared = preflight()
        print(json.dumps({key: prepared[key] for key in ("passed", "files", "token_stats")}, indent=2))
    elif arguments.arm is None or arguments.output_dir is None:
        parser.error("execute requires --arm and --output-dir")
    else:
        execute(arguments.arm, arguments.output_dir.resolve())