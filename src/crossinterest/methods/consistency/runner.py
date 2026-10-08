"""Prepare and run the matched CE/dual-consistency pilot on existing source pairs."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import register_semantic_tokens
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE as FROZEN_HERE, sha256_file
from crossinterest.methods.consistency.loss import ARMS, VIEW_ORDER
from crossinterest.methods.consistency.train import DualDataset
from crossinterest.methods.cir.runner import TRAIN, DEV, MAPPING, TRAIN_SHA, DEV_SHA, MAPPING_SHA, MODELS
from crossinterest.methods.crossfit.train import teacher_command
from crossinterest.methods.mechanism.runner import audit_evaluation, immutable_json


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "dual"
PREFLIGHT = ROOT / "data/PREFLIGHT.json"


def source_file(info):
    return ROOT / "data" / info["path"] if info.get("source") == "dual" else FROZEN_HERE.parent / info["path"]


def code_identity():
    return {name: sha256_file(HERE / name) for name in
            ("cir_dual_loss.py", "cir_dual_train.py", "cir_dual_run.py")}


def paired_rows(source: Path):
    manifest = json.loads((source / "MANIFEST.json").read_text())
    require_fields(json.loads((source / "INDEPENDENT_AUDIT.json").read_text()), {
        "passed": True, "manifest_sha256": sha256_file(source / "MANIFEST.json"),
        "n_dev_user_overlap": 0, "n_dev_input_overlap": 0}, "dual source audit")
    if (sha256_file(source / "PAIRS.parquet") != manifest["pairs_sha256"]
            or sha256_file(source / "ordinary.parquet") != manifest["files"]["ordinary"]["sha256"]):
        raise ValueError("paired source changed")
    pairs = pq.read_table(source / "PAIRS.parquet").to_pylist()
    original = pq.read_table(source / "ordinary.parquet").to_pylist()
    if len(pairs) != 13440 or len(original) != 2 * len(pairs):
        raise ValueError("expected original 13440 paired sources")
    records = []
    for index, pair in enumerate(pairs):
        forward, inverse = original[2 * index:2 * index + 2]
        if (forward["source_id"] != pair["source_id"] or inverse["source_id"] != pair["source_id"]
                or forward["user_id"] != pair["user_id"] or inverse["user_id"] != pair["user_id"]
                or forward["target_interest"] != pair["B"] or inverse["target_interest"] != pair["A"]
                or forward["direction"] != "forward" or inverse["direction"] != "inverse"
                or pair["A"] == pair["B"]):
            raise ValueError("paired identity or direction mismatch")
        keep = [position for position, category in enumerate(forward["historical_interests"])
                if category != pair["A"]]
        for field in ("historical_titles", "historical_interests", "history_timestamps"):
            if inverse[field] != [forward[field][position] for position in keep]:
                raise ValueError("R is not the source history with all A events removed")
        if ({pair["A"], pair["B"]} & set(inverse["historical_interests"])
                or pair["B"] in forward["historical_interests"]
                or not inverse["historical_titles"]):
            raise ValueError("target leakage in paired history")
        records.append({"id": "dual:" + pair["source_id"], "source_id": pair["source_id"],
                        "user_id": pair["user_id"], "target_interest": pair["A"],
                        "future_interest": pair["B"], "historical_titles": inverse["historical_titles"],
                        "historical_interests": inverse["historical_interests"],
                        "full_seen_interests": inverse["full_seen_interests"]})
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("duplicate paired source")
    for field in ("target_interest", "future_interest"):
        counts = Counter(row[field] for row in records)
        if len(counts) != 112 or set(counts.values()) != {120}:
            raise ValueError("paired A/B labels must each be balanced at 120")
    return records


def smoke_pairs(rows, lengths):
    labels = sorted({row["target_interest"] for row in rows})
    label_index = {label: index for index, label in enumerate(labels)}
    longest = max(range(len(rows)), key=lambda index: (lengths[index], -index))
    grouped = defaultdict(list)
    for index in sorted(range(len(rows)), key=lambda index: (-lengths[index], index)):
        if index != longest:
            row = rows[index]
            grouped[(label_index[row["target_interest"]], label_index[row["future_interest"]])].append(index)
    count = len(labels)
    sink = 2 * count + 1
    starts, ends, capacities = [], [], []
    for index, label in enumerate(labels):
        starts.extend((0, count + index + 1))
        ends.extend((index + 1, sink))
        capacities.extend((2 - int(rows[longest]["target_interest"] == label),
                           2 - int(rows[longest]["future_interest"] == label)))
    for (left, right), indices in grouped.items():
        starts.append(left + 1)
        ends.append(count + right + 1)
        capacities.append(len(indices))
    graph = coo_matrix((np.asarray(capacities, dtype=np.int32), (starts, ends)),
                       shape=(sink + 1, sink + 1)).tocsr()
    result = maximum_flow(graph, 0, sink)
    if result.flow_value != 2 * count - 1:
        raise ValueError("cannot select balanced smoke with the longest pair")
    selected = [longest]
    for (left, right), indices in grouped.items():
        selected.extend(indices[:result.flow[left + 1, count + right + 1]])
    if len(selected) != 224 or len(set(selected)) != 224:
        raise ValueError("smoke pair count mismatch")
    for field in ("target_interest", "future_interest"):
        if set(Counter(rows[index][field] for index in selected).values()) != {2}:
            raise ValueError("unbalanced smoke pairs")
    return sorted(selected)


def prepare():
    from transformers import AutoTokenizer

    identity = code_identity()
    if PREFLIGHT.exists():
        prepared = json.loads(PREFLIGHT.read_text())
        require_fields(prepared, {"code_sha256": identity, "passed": True}, "dual preflight")
        for info in prepared["files"].values():
            if sha256_file(source_file(info)) != info["sha256"]:
                raise ValueError("prepared dual source changed")
        return prepared
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError("original data changed")
    source = FROZEN_HERE / "bidirectional/data"
    rows = paired_rows(source)
    ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)
    destination = ROOT / "data/train.parquet"
    table = pa.Table.from_pylist(rows)
    if destination.exists():
        if not pq.read_table(destination).equals(table):
            raise ValueError("refusing to overwrite different dual data")
    else:
        pq.write_table(table, destination, compression="zstd")
    model, revision = MODELS["qwen35_9b"]
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    dataset = DualDataset(destination, tokenizer, "behavior_interest", "interest_id", mapping,
                          8192, None, None, None, len(rows))
    digest = hashlib.sha256()
    lengths, token_totals = [], Counter()
    for index in range(len(dataset)):
        item = dataset[index]
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        lengths.append(max(len(view["input_ids"]) for view in item["views"]))
        for name, view in zip(VIEW_ORDER, item["views"]):
            token_totals[name] += len(view["input_ids"])
        if (index + 1) % 2000 == 0:
            print(f"dual tokenizer preflight {index+1}/{len(dataset)}", flush=True)
    selected = smoke_pairs(rows, lengths)
    smoke_path = ROOT / "data/smoke_auxiliary.parquet"
    smoke_table = table.take(selected)
    if smoke_path.exists():
        if not pq.read_table(smoke_path).equals(smoke_table):
            raise ValueError("refusing to overwrite different dual smoke")
    else:
        pq.write_table(smoke_table, smoke_path, compression="zstd")
    reference_path = FROZEN_HERE / "cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json"
    reference = json.loads(reference_path.read_text())
    require_fields(reference, {"passed": True, "train_sha256": TRAIN_SHA,
                               "mapping_sha256": MAPPING_SHA, "dev_sha256": DEV_SHA}, "N preflight")
    files = {key: reference["files"][key] for key in ("novel", "smoke_novel")}
    for name, path, count in (("auxiliary", destination, len(rows)), ("smoke_auxiliary", smoke_path, len(selected))):
        files[name] = {"source": "dual", "path": path.name, "rows": count, "views": 4 * count,
                       "steps": count // 8, "sha256": sha256_file(path), "epochs": 1}
    for info in files.values():
        if sha256_file(source_file(info)) != info["sha256"]:
            raise ValueError("reused file mismatch")
    prepared = {"passed": True, "schema_version": 1, "model_key": "qwen35_9b",
                "model_repo": model, "model_revision": revision, "code_sha256": identity,
                "baseline_sha256": sha256_file(Path(baseline.__file__)),
                "source_manifest_sha256": sha256_file(source / "MANIFEST.json"),
                "source_audit_sha256": sha256_file(source / "INDEPENDENT_AUDIT.json"),
                "n_preflight_sha256": sha256_file(reference_path), "files": files,
                "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
                "token_registration": registration, "tokenized_auxiliary_sha256": digest.hexdigest(),
                "auxiliary_tokens_by_view": dict(token_totals), "max_auxiliary_tokens": max(lengths),
                "max_novel_tokens": reference["max_train_tokens"]["novel"],
                "arms": ARMS, "view_order": VIEW_ORDER, "auxiliary_epochs": 1,
                "pairs": len(rows), "views": len(rows) * 4, "views_per_class": 480,
                "pair_batch_size": 8, "effective_view_batch_size": 32, "world_size": 8,
                "auxiliary_gradient_checkpointing": True, "novel_gradient_checkpointing": False,
                "distributed_padding_duplicates": 0, "smoke_contains_longest_pair": True,
                "fresh_formal_after_smoke": True, "world_knowledge_mechanism_proven": False,
                "evaluation": {"rows": 2000, "beams": 20, "candidate_free": True, "do_sample": False}}
    immutable_json(PREFLIGHT, prepared)
    return prepared


def training_command(model_path, stage, directory, smoke, prepared, arm, previous):
    info = prepared["files"][("smoke_" if smoke else "") + stage]
    command = teacher_command(model_path, source_file(info), directory, info["rows"])
    command[command.index("--nproc_per_node=4")] = "--nproc_per_node=8"
    command[command.index('crossinterest.training.train')] = 'crossinterest.methods.consistency.train'
    command[command.index("--expected-effective-batch-size") + 1] = "8" if stage == "auxiliary" else "32"
    command[command.index("--gradient-accumulation-steps") + 1] = "1" if stage == "auxiliary" else "4"
    command[command.index("--save-steps") + 1] = "100"
    if stage == "novel":
        command[command.index("--gradient-checkpointing")] = "--no-gradient-checkpointing"
    command.extend(["--stage", stage, "--dual-arm", arm, "--dual-preflight", str(PREFLIGHT), "--resume"])
    if smoke:
        command.append("--smoke")
    if previous is not None:
        command.extend(["--init-adapter", str(previous)])
    return command


def audit_stage(directory, stage, smoke, prepared, arm, model_path):
    import torch
    from safetensors.torch import load_file

    info = prepared["files"][("smoke_" if smoke else "") + stage]
    expected = {"model": model_path, "train_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
                "view": "behavior_interest", "label_mode": "interest_id", "epochs": 1.0,
                "learning_rate": 1e-4, "effective_batch_size": 8 if stage == "auxiliary" else 32,
                "loss_mode": "causal_lm", "selection_file": None, "sample_limit": None,
                "semantic_label_token_weight": 1.0}
    metrics = json.loads((directory / "train_metrics.json").read_text())
    previous = directory.parent / "auxiliary/final_adapter/adapter_model.safetensors"
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, expected, "dual training contract")
        require_fields(record["experiment_metadata"], {"dual_arm": arm, "dual_coefficient": ARMS[arm],
            "stage": stage, "smoke": smoke, "preflight_sha256": sha256_file(PREFLIGHT),
            "effective_view_batch_size": 32, "relation_head": False}, "dual stage identity")
        require_fields(record, {"init_adapter_model_sha256": sha256_file(previous) if stage == "novel" else None},
                       "dual stage weights")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
                             "sample_count": info["rows"], "epoch": 1.0}, "dual completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid dual training loss")
    require_fields(json.loads((directory / "runtime_config.json").read_text()), {
        "world_size": 8, "per_device_batch_size": 1, "gradient_accumulation_steps": 1 if stage == "auxiliary" else 4,
        "gradient_checkpointing": stage == "auxiliary", "ddp_find_unused_parameters": False}, "dual layout")
    mechanism = json.loads((directory / "MECHANISM_RUNTIME.json").read_text())
    if stage == "auxiliary":
        require_fields(mechanism, {"model_accepts_loss_kwargs": False,
                                  "auxiliary_mean_loss_scaled_by_trainer": True}, "dual accumulation")
        require_fields(json.loads((directory / "final_adapter/DUAL_RUNTIME.json").read_text()),
                       {"arm": arm, "coefficient": ARMS[arm], "view_order": list(VIEW_ORDER)}, "dual loss")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "dual LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid dual adapter")
    if not any(value.count_nonzero().item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("dual adapter did not update")
    result = {"passed": True, "stage": stage, "rows": info["rows"], "steps": info["steps"],
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"), "relation_sha256": None,
              "train_runtime_seconds": metrics["train_runtime"], "mechanism_runtime": mechanism}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def execute(arm, output):
    from huggingface_hub import snapshot_download

    prepared = prepare()
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"],
                                   cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {**prepared, "dual_arm": arm, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       OMP_NUM_THREADS="1")
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        previous, stages = None, {}
        for stage in ("auxiliary", "novel"):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, stage, stage_dir, smoke, prepared, arm, previous),
                               cwd=FROZEN_HERE.parent, env=environment, check=True)
            stages[stage] = audit_stage(stage_dir, stage, smoke, prepared, arm, model_path)
            previous = stage_dir / "final_adapter"
        processes = []
        try:
            for evaluation, devices in (("classification", "0,1"), ("ranking", "2,3,4,5,6,7")):
                command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate',
                    "--evaluation", evaluation, "--model-key", "qwen35_9b", "--model", model_path,
                    "--adapter", str(previous), "--view", "behavior_interest", "--label-mode", "interest_id",
                    "--label-map", str(MAPPING), "--data-dir", str(DEV.parent), "--expected-label-count", "112",
                    "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
                if smoke:
                    command.extend(["--limit", "4"])
                if evaluation == "ranking":
                    command.extend(["--label-file", str(TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                                    "--decoding", "direct_beam_search"])
                processes.append(subprocess.Popen(command, cwd=FROZEN_HERE.parent,
                                 env={**environment, "CUDA_VISIBLE_DEVICES": devices}))
            for process in processes:
                if process.wait() != 0:
                    raise RuntimeError("dual evaluation process failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        result = audit_evaluation(directory, "reconstruction", smoke, stages["novel"])
        result.update(dual_arm=arm, stage_audits=stages, preflight_sha256=sha256_file(PREFLIGHT))
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps(result), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "dual_arm": arm,
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.action == "prepare":
        result = prepare()
        print(json.dumps({key: result[key] for key in ("passed", "pairs", "views", "files", "max_auxiliary_tokens")}))
    elif options.arm is None or options.output is None:
        parser.error("execute requires --arm and --output")
    else:
        execute(options.arm, options.output.resolve())


if __name__ == "__main__":
    main()