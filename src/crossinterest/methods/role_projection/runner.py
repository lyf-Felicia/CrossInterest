"""Isolated, hash-pinned execution of the three role-projection conditions."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if not __package__:
    live = _source_workspace('eval_interest_id_minionerec')
    frozen = live / "cir_factorization/package"
    sys.path[:0] = [str(frozen), str(frozen / "prep")]
    import crossinterest.methods as eval_interest_id_minionerec
    eval_interest_id_minionerec.__path__.append(str(live))
    __package__ = "eval_interest_id_minionerec"

import pyarrow.parquet as pq

from crossinterest.evaluation.common import latest_details, write_json_atomic
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE as BASE_HERE, sha256_file
from crossinterest.methods.cir.runner import DEV, DEV_SHA, MAPPING, MAPPING_SHA, TRAIN, TRAIN_SHA
from crossinterest.methods.crossfit.train import teacher_command
from crossinterest.methods.mechanism.runner import immutable_json


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "role_projection"
CONTRACT = ROOT / "CONTRACT.json"
MODES = ("baseline", "unconstrained", "projected")
MODULES = ("role_projection.py", "role_train.py", "role_evaluate.py", "role_run.py",
           "test_role_projection.py", "test_interest_composition.py", "interest_composition.py",
           "role_diagnose.py", "role_collect.py")
METHOD = "role-projection-v2"


def freeze(destination, g0_reference=None, g0_environment=None):
    base = BASE_HERE.parent
    if base == HERE.parent or not (HERE / "cir_factorization/package").is_dir():
        raise ValueError("freeze must read the original frozen package from the live script")
    if destination.exists():
        raise ValueError("refusing to overwrite an existing experiment bundle")
    reference_path = BASE_HERE / "cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json"
    reference = json.loads(reference_path.read_text())
    alignment = None
    if g0_reference is not None:
        if g0_environment is None:
            raise ValueError("G0 alignment requires the original environment evidence")
        source = json.loads((g0_reference / "EXECUTION_CONTRACT.json").read_text())
        for relative, expected in source["code_sha256"].items():
            if sha256_file(base / relative) != expected:
                raise ValueError(f"original G0 source differs: {relative}")
        require_fields(source, {key: reference[key] for key in
            ("model_repo", "model_revision", "files", "train_sha256", "dev_sha256", "mapping_sha256")}, "original G0")
        names = ["EXECUTION_CONTRACT.json", "COMPLETE.json", "formal/ranking/run_contract.json",
                 "formal/ranking/metrics.json"]
        for stage in ("auxiliary", "novel"):
            names.extend(f"formal/{stage}/{name}.json" for name in ("train_contract", "runtime_config"))
        alignment = {"source_sha256": {name: sha256_file(g0_reference / name) for name in names},
                     "records": {name: json.loads((g0_reference / name).read_text()) for name in names},
                     "environment": json.loads(g0_environment.read_text()),
                     "environment_source_sha256": sha256_file(g0_environment),
                     "bitwise_training_trajectory_guaranteed": False}
        for stage in ("auxiliary", "novel"):
            require_fields(alignment["records"][f"formal/{stage}/runtime_config.json"],
                {"world_size": 4, "per_device_batch_size": 1, "gradient_accumulation_steps": 8,
                 "gradient_checkpointing": True, "effective_batch_size": 32}, "G0 layout")
    for relative, expected in reference["code_sha256"].items():
        if sha256_file(base / relative) != expected:
            raise ValueError(f"frozen reference code differs: {relative}")
    for info in reference["files"].values():
        if sha256_file(base / info["path"]) != info["sha256"]:
            raise ValueError("frozen reference data differs")
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source differs: {path.name}")
    originals = {str(path.relative_to(base)): sha256_file(path) for path in base.rglob("*")
                 if path.is_file() and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)}
    for relative in originals:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / relative, target)
    for name in MODULES:
        shutil.copyfile(HERE / name, destination / "eval_interest_id_minionerec" / name)
    new_root = destination / "eval_interest_id_minionerec/role_projection"
    rows = pq.read_table(DEV)
    shards = {}
    workers = 1 if alignment else 8
    for shard in range(workers):
        positions = list(range(shard, len(rows), workers))
        if alignment:
            path = destination / DEV.relative_to(base)
        else:
            path = new_root / f"data/shard-{shard}" / DEV.name
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(rows.take(positions), path, compression="zstd")
        shards[str(shard)] = {"path": str(path.relative_to(destination)), "rows": len(positions),
                              "sha256": sha256_file(path), "positions": positions}
    prepared = {**reference, "variant": METHOD, "method": METHOD, "modes": list(MODES),
        "taxonomy_scaling": "global_output_rms",
        "created_utc": datetime.now(timezone.utc).isoformat(), "reference_sha256": sha256_file(reference_path),
        "source_files": originals, "shards": shards,
        "method_files": {f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name) for name in MODULES},
        "training_layout": {"world_size": 4 if alignment else 8, "micro_batch": 1,
                    "accumulation": 8 if alignment else 4, "gradient_checkpointing": bool(alignment)},
        "evaluation": {"rows": 2000, "classes": 112, "workers": workers, "beams": 20, "batch_size": 1,
                   "do_sample": False, "max_new_tokens": 32, "max_prompt_length": 8192,
                       "smoke_limit": 4 if alignment else 2,
                   "device_map": "balanced", "all_training_devices": bool(alignment)},
        "gate": {"projected_ndcg20_minus_baseline_minimum": 0.005,
                 "projected_recall20_at_least_baseline": True,
                 "projected_ndcg20_above_unconstrained": True},
        "formal_requires_smoke_pass": True, "formal_results_available": False}
    if alignment:
        prepared["g0_alignment"] = alignment
        prepared["formal_smoke_scope"] = "each_requested_condition"
        constraints = new_root / "g0_constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in
                                       sorted(alignment["environment"]["versions"].items())))
        prepared["method_files"][str(constraints.relative_to(destination))] = sha256_file(constraints)
    immutable_json(new_root / "CONTRACT.json", prepared)
    for relative, expected in originals.items():
        if sha256_file(base / relative) != expected:
            raise ValueError("original frozen package changed during copying")
    print(json.dumps({"bundle": str(destination), "original_files_unchanged": len(originals),
                      "contract_sha256": sha256_file(new_root / "CONTRACT.json")}), flush=True)


def verify():
    prepared = json.loads(CONTRACT.read_text())
    require_fields(prepared, {"method": METHOD, "modes": list(MODES), "auxiliary_epochs": 3,
                             "taxonomy_scaling": "global_output_rms"}, "role contract")
    expected_files = {**prepared["source_files"], **prepared["method_files"]}
    for relative, expected in expected_files.items():
        if sha256_file(HERE.parent / relative) != expected:
            raise ValueError(f"execution file identity changed: {relative}")
    gathered = []
    original = pq.read_table(DEV).to_pylist()
    for info in prepared["shards"].values():
        path = HERE.parent / info["path"]
        if sha256_file(path) != info["sha256"]:
            raise ValueError("evaluation shard changed")
        shard = pq.read_table(path).to_pylist()
        if shard != [original[index] for index in info["positions"]]:
            raise ValueError("evaluation sharding changed a source row")
        gathered.extend(info["positions"])
    if sorted(gathered) != list(range(2000)):
        raise ValueError("evaluation shards do not exactly partition frozen Dev")
    return prepared


def training_command(model_path, output, mode, stage, smoke, prepared):
    info = prepared["files"][("smoke_" if smoke else "") + stage]
    layout = prepared["training_layout"]
    if layout["world_size"] * layout["micro_batch"] * layout["accumulation"] != 32:
        raise ValueError("role layout must preserve effective batch size 32")
    command = teacher_command(model_path, HERE.parent / info["path"], output, info["rows"])
    command[command.index("--nproc_per_node=4")] = f"--nproc_per_node={layout['world_size']}"
    command[command.index('crossinterest.training.train')] = 'crossinterest.methods.role_projection.train'
    command[command.index("--epochs") + 1] = str(info.get("epochs", 1))
    command[command.index("--gradient-accumulation-steps") + 1] = str(layout["accumulation"])
    command[command.index("--per-device-batch-size") + 1] = str(layout["micro_batch"])
    if not layout["gradient_checkpointing"]:
        command[command.index("--gradient-checkpointing")] = "--no-gradient-checkpointing"
    command.extend(["--stage", stage, "--role-mode", mode, "--role-contract", str(CONTRACT), "--resume"])
    if stage == "novel":
        command.extend(["--init-adapter", str(output.parent / "auxiliary/final_adapter")])
    if smoke:
        command.append("--smoke")
    return command


def audit_training(directory, mode, stage, smoke, prepared, model_path):
    import torch
    from safetensors.torch import load_file

    info = prepared["files"][("smoke_" if smoke else "") + stage]
    previous = directory.parent / "auxiliary/final_adapter" if stage == "novel" else None
    metrics = json.loads((directory / "train_metrics.json").read_text())
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, {"model": model_path, "train_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
            "view": "behavior_interest", "label_mode": "interest_id", "epochs": float(info.get("epochs", 1)),
            "learning_rate": 1e-4, "effective_batch_size": 32, "loss_mode": "causal_lm", "seed": 20260825,
            "selection_file": None, "sample_limit": None, "semantic_label_token_weight": 1.0,
            "init_adapter_model_sha256": sha256_file(previous / "adapter_model.safetensors") if previous else None},
            "role training budget")
        require_fields(record["experiment_metadata"], {"role_mode": mode, "stage": stage, "smoke": smoke,
            "preflight_sha256": sha256_file(CONTRACT), "relation_head": mode != "baseline"}, "role training identity")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
        "sample_count": info["rows"], "epoch": float(info.get("epochs", 1))}, "role training completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid stage loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    layout = prepared["training_layout"]
    require_fields(runtime, {"world_size": layout["world_size"], "per_device_batch_size": layout["micro_batch"],
        "gradient_accumulation_steps": layout["accumulation"], "gradient_checkpointing": layout["gradient_checkpointing"],
        "ddp_find_unused_parameters": False, "dataloader_drop_last": False}, "role runtime")
    require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen CUDA kernels")
    role_runtime = json.loads((directory / "ROLE_RUNTIME.json").read_text())
    require_fields(role_runtime, {"role_mode": mode, "stage": stage}, "role runtime identity")
    parameters = json.loads((directory / "role_trainable_parameters.json").read_text())
    require_fields(parameters, {"count": role_runtime["total_trainable_parameters"],
                               "tensor_count": len(parameters["names"])}, "role trainable parameter report")
    if stage == "auxiliary":
        require_fields(role_runtime, {"model_accepts_loss_kwargs": False}, "CIR accumulation")
    if "g0_alignment" in prepared:
        records = prepared["g0_alignment"]["records"]
        require_fields(runtime, records[f"formal/{stage}/runtime_config.json"], "exact G0 runtime")
        excluded = {"task", "label_map", "train_file", "init_adapter", "init_adapter_model_sha256",
                    "init_adapter_config_sha256", "experiment_metadata"}
        if smoke:
            excluded.update({"train_sha256", "selected_train_rows", "selected_label_count_min",
                             "selected_label_count_max", "selected_label_count_max_gap", "formal_train_rows"})
        expected = {key: value for key, value in records[f"formal/{stage}/train_contract.json"].items()
                    if key not in excluded}
        for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
            require_fields(record, expected, "exact G0 training settings")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "role LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid LoRA weights")
    if not any(value.count_nonzero().item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("LoRA did not update")
    result = {"passed": True, "role_mode": mode, "stage": stage, "smoke": smoke,
              "rows": info["rows"], "steps": info["steps"], "runtime_seconds": metrics["train_runtime"],
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"), "head_sha256": None,
              "role_runtime": role_runtime}
    performance = []
    for rank in range(layout["world_size"]):
        record = json.loads((directory / f"PERFORMANCE_RANK_{rank}.json").read_text())
        require_fields(record, {"rank": rank, "world_size": layout["world_size"]}, "per-rank performance")
        if not 0 < record["peak_allocated_bytes"] <= record["peak_reserved_bytes"] <= record["total_memory_bytes"]:
            raise ValueError("invalid GPU memory telemetry")
        performance.append(record)
    result["performance"] = performance
    if mode != "baseline":
        require_fields(role_runtime["head_metadata"], {"architecture": METHOD,
                       "taxonomy_scaling": "global_output_rms"}, "role numeric parameterization")
        head = load_file(str(adapter / "interest_relation.safetensors"))
        initial = load_file(str(directory / "initial_relation/interest_relation.safetensors"))
        if not all(torch.isfinite(value).all() for value in head.values()):
            raise ValueError("nonfinite role head")
        if not torch.equal(head["target_prototypes"], initial["target_prototypes"]):
            raise ValueError("fixed taxonomy prototypes changed")
        result["head_updates"] = {}
        for branch in ("compatible", "directional"):
            for projection in ("source", "target", "context"):
                name = f"{branch}.{projection}.weight"
                changed = not torch.equal(head[name], initial[name])
                difference = head[name].float() - initial[name].float()
                result["head_updates"][name] = {"changed_elements": int(difference.count_nonzero()),
                    "delta_norm": difference.norm().item(), "delta_max": difference.abs().max().item()}
                if changed != (branch == "compatible" or stage == "novel"):
                    raise ValueError(f"role branch update contract failed: {stage}/{name}")
        if previous is not None:
            inherited = load_file(str(previous / "interest_relation.safetensors"))
            if any(not torch.equal(value, inherited[name]) for name, value in initial.items()):
                raise ValueError("role head handoff did not preserve all auxiliary weights")
        result["head_sha256"] = sha256_file(adapter / "interest_relation.safetensors")
        result["head_norms"] = {name: value.float().norm().item() for name, value in head.items() if name.endswith("weight")}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def evaluation_command(model_path, adapter, directory, mode, shard, smoke, prepared):
    data = (HERE.parent / prepared["shards"][str(shard)]["path"]).parent
    command = [sys.executable, "-m", 'crossinterest.methods.role_projection.evaluate', "--evaluation", "ranking",
        "--role-mode", mode, "--model-key", "qwen35_9b", "--model", model_path, "--adapter", str(adapter),
        "--view", "behavior_interest", "--label-mode", "interest_id", "--label-map", str(MAPPING),
        "--label-file", str(TRAIN), "--data-dir", str(data), "--run-dir", str(directory),
        "--expected-label-count", "112", "--device-map", "balanced", "--num-beams", "20",
        "--eval-batch-size", "1", "--decoding", "direct_beam_search", "--max-new-tokens", "32",
        "--max-prompt-length", "8192"]
    if smoke:
        command.extend(["--limit", str(prepared["evaluation"].get("smoke_limit", 2))])
    return command


def audit_evaluation(directory, mode, smoke, prepared):
    from crossinterest.training import evaluate_ranking as ranking

    adapter = directory / "novel/final_adapter"
    identities = {"adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                  "head_sha256": sha256_file(adapter / "interest_relation.safetensors") if mode != "baseline" else None}
    gathered, expected, evidence = {}, [], {}
    for shard in range(prepared["evaluation"]["workers"]):
        info = prepared["shards"][str(shard)]
        rows = pq.read_table(HERE.parent / info["path"]).to_pylist()
        limit = prepared["evaluation"].get("smoke_limit", 2) if smoke else None
        rows = rows[:limit] if smoke else rows
        output = directory / "ranking" / f"shard-{shard}"
        contract = json.loads((output / "run_contract.json").read_text())
        require_fields(contract, {"dataset_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
            "label_file_sha256": TRAIN_SHA, "num_beams": 20, "num_return_sequences": 20,
            "sample_limit": limit, "do_sample": False, "candidates_in_prompt": False,
            "decoder_label_constraint": False, "decoding": "candidate_free_direct_beam_search",
            "max_new_tokens": 32, "eval_batch_size": 1, "label_mode": "interest_id", "view": "behavior_interest"},
            "role beam protocol")
        if "g0_alignment" in prepared:
            original = prepared["g0_alignment"]["records"]["formal/ranking/run_contract.json"]
            excluded = {"adapter", "adapter_config_sha256", "dataset", "label_map", "label_file", "fingerprint"}
            require_fields(contract, {**{key: value for key, value in original.items() if key not in excluded},
                                      "sample_limit": limit}, "exact G0 evaluation settings")
        require_fields(json.loads((output / "ROLE_CONTRACT.json").read_text()),
                       {**identities, "role_mode": mode}, "role evaluated weights")
        details = latest_details(output / "details.jsonl", contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows} or gathered.keys() & details.keys():
            raise ValueError("incomplete or duplicated role evaluation rows")
        for row in rows:
            detail = details[row["sample_id"]]
            require_fields(detail, {"source_sample_id": row["source_sample_id"],
                                   "target_interest": row["target_interest"]}, "role raw label")
            if len(detail["ranking"]) != 20 or any(not math.isfinite(item["sequence_score"]) for item in detail["ranking"]):
                raise ValueError("incomplete or nonfinite beam list")
            require_fields(detail, ranking.score_ranking(detail["ranking"], row["target_interest"]), "role raw scores")
        calculated = ranking.metrics_from_rankings(list(details.values()), rows, 20)
        require_fields(json.loads((output / "metrics.json").read_text()), calculated, "role recomputed shard")
        require_fields(calculated, {"complete": True, "missing_sample_count": 0,
                                   "recorded_sample_count": len(rows)}, "role shard coverage")
        gathered.update(details)
        expected.extend(rows)
        evidence[str(shard)] = {"raw_sha256": sha256_file(output / "details.jsonl"),
                                "contract_sha256": sha256_file(output / "run_contract.json")}
    metrics = ranking.metrics_from_rankings(list(gathered.values()), expected, 20)
    result = {"passed": True, "role_mode": mode, "smoke": smoke, "weights": identities,
              "metrics": metrics, "shards": evidence, "role_contract_sha256": sha256_file(CONTRACT)}
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def audit_environment(prepared, output):
    from importlib.metadata import version
    import torch

    expected = prepared["g0_alignment"]["environment"]["versions"]
    observed = {name: version(name) for name in expected}
    require_fields(observed, expected, "original G0 package versions")
    if sys.version_info[:2] != (3, 10) or torch.version.cuda != "12.6":
        raise ValueError("G0 requires Python 3.10 and CUDA 12.6")
    if torch.cuda.device_count() != prepared["training_layout"]["world_size"]:
        raise ValueError("visible device count differs from G0")
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        if "A100" not in device.name or device.total_memory < 79 * 1024 ** 3:
            raise ValueError("G0 requires A100 80GB devices")
        devices.append({"index": index, "name": device.name, "total_memory": device.total_memory})
    immutable_json(output / "ENVIRONMENT_AUDIT.json", {"passed": True, "versions": observed,
        "python": sys.version, "cuda": torch.version.cuda, "devices": devices,
        "source_sha256": prepared["g0_alignment"]["environment_source_sha256"]})


def execute(output, modes, smoke, smoke_evidence=None):
    from huggingface_hub import snapshot_download

    prepared = verify()
    if not smoke:
        if smoke_evidence is None:
            raise ValueError("formal execution requires the completed three-condition smoke evidence")
        evidence = json.loads(smoke_evidence.read_text())
        require_fields(evidence, {"passed": True, "smoke": True,
                                  "role_contract_sha256": sha256_file(CONTRACT)}, "smoke gate")
        expected_modes = modes if prepared.get("formal_smoke_scope") == "each_requested_condition" else list(MODES)
        if evidence["modes"] != expected_modes:
            raise ValueError("smoke condition coverage differs")
    output.mkdir(parents=True, exist_ok=True)
    if "g0_alignment" in prepared:
        audit_environment(prepared, output)
    cache = {"cache_dir": "/tmp/huggingface/hub"} if "g0_alignment" in prepared else {}
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], **cache)
    immutable_json(output / "EXECUTION_CONTRACT.json", {"role_contract_sha256": sha256_file(CONTRACT),
        "model_path": model_path, "modes": modes, "smoke": smoke,
        "smoke_evidence_sha256": sha256_file(smoke_evidence) if smoke_evidence else None})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    results = {}
    for mode in modes:
        directory = output / mode / ("smoke" if smoke else "formal")
        directory.mkdir(parents=True, exist_ok=True)
        stages = {}
        for stage in ("auxiliary", "novel"):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, stage_dir, mode, stage, smoke, prepared),
                               cwd=HERE.parent, env=environment, check=True)
            stages[stage] = audit_training(stage_dir, mode, stage, smoke, prepared, model_path)
            print(json.dumps({"event": "stage_passed", **stages[stage]}), flush=True)
        started = time.monotonic()
        processes = []
        try:
            for shard in range(prepared["evaluation"]["workers"]):
                shard_dir = directory / "ranking" / f"shard-{shard}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                stream = (shard_dir / "process.log").open("a")
                devices = (",".join(str(rank) for rank in range(prepared["training_layout"]["world_size"]))
                           if prepared["evaluation"].get("all_training_devices") else str(shard))
                process = subprocess.Popen(evaluation_command(model_path, directory / "novel/final_adapter",
                    shard_dir, mode, shard, smoke, prepared), cwd=HERE.parent,
                    env={**environment, "CUDA_VISIBLE_DEVICES": devices}, stdout=stream, stderr=subprocess.STDOUT)
                processes.append((process, stream))
            codes = [process.wait() for process, _ in processes]
            if any(codes):
                raise RuntimeError(f"role evaluation processes failed: {codes}")
        finally:
            for process, stream in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                stream.close()
        evaluation = audit_evaluation(directory, mode, smoke, prepared)
        results[mode] = {"stages": stages, "evaluation": evaluation,
                         "evaluation_wall_seconds": time.monotonic() - started}
        print(json.dumps({"event": "condition_passed", "mode": mode, "smoke": smoke}), flush=True)
    result = {"passed": True, "modes": modes, "smoke": smoke,
              "role_contract_sha256": sha256_file(CONTRACT), "results": results}
    if smoke:
        result["longest_smoke_scaled_training_seconds"] = {
            mode: sum(info["runtime_seconds"] / info["steps"] * prepared["files"][stage]["steps"]
                      for stage, info in record["stages"].items()) for mode, record in results.items()}
        result["timing_caveat"] = "Longest-per-class smoke includes startup; scaling is not a measured formal ETA."
    write_json_atomic(output / "COMPLETE.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    freezing = subparsers.add_parser("freeze")
    freezing.add_argument("--destination", type=Path, required=True)
    freezing.add_argument("--g0-reference", type=Path)
    freezing.add_argument("--g0-environment", type=Path)
    subparsers.add_parser("verify")
    running = subparsers.add_parser("execute")
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    running.add_argument("--phase", choices=("smoke", "formal", "smoke-then-formal"), required=True)
    running.add_argument("--smoke-evidence", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
         freeze(args.destination.resolve(), args.g0_reference.resolve() if args.g0_reference else None,
             args.g0_environment.resolve() if args.g0_environment else None)
    elif args.action == "verify":
        prepared = verify()
        print(json.dumps({"passed": True, "method": prepared["method"], "modes": prepared["modes"]}))
    else:
        if args.phase == "smoke-then-formal":
            if verify().get("formal_smoke_scope") != "each_requested_condition":
                raise ValueError("automatic formal execution requires a per-condition smoke contract")
            evidence_dir = args.output.resolve() / "preflight"
            execute(evidence_dir, args.modes, True)
            execute(args.output.resolve(), args.modes, False, evidence_dir / "COMPLETE.json")
        else:
            execute(args.output.resolve(), args.modes, args.phase == "smoke", args.smoke_evidence)


if __name__ == "__main__":
    main()