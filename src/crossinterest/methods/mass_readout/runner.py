"""Hash-pinned N-only experiment with completed CIR reuse and automatic evaluation."""

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

if not __package__:
    live = _source_workspace('eval_interest_id_minionerec')
    frozen = live / "cir_factorization/package"
    sys.path[:0] = [str(frozen), str(frozen / "prep")]
    import crossinterest.methods as eval_interest_id_minionerec
    eval_interest_id_minionerec.__path__.append(str(live))
    __package__ = "eval_interest_id_minionerec"

from crossinterest.evaluation.common import latest_details, write_json_atomic
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.runner import immutable_json
from crossinterest.methods.role_projection.audit import independent_metrics
import crossinterest.methods.role_projection.runner as role_run


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "mass_readout"
CONTRACT = ROOT / "CONTRACT.json"
MODULES = ("mass_readout.py", "mass_train.py", "mass_evaluate.py", "mass_run.py", "test_mass_readout.py")
METHOD = "mass-preserving-readout-v1"


def freeze(destination, reference, source, baseline):
    import pyarrow.parquet as pq

    if destination.exists():
        raise ValueError("refusing to replace an experiment bundle")
    old_contract = reference / "eval_interest_id_minionerec/role_projection/CONTRACT.json"
    prepared = json.loads(old_contract.read_text())
    for relative, expected in {**prepared["source_files"], **prepared["method_files"]}.items():
        if sha256_file(reference / relative) != expected:
            raise ValueError(f"reference source changed: {relative}")
    source_stage = json.loads((source / "STAGE_AUDIT.json").read_text())
    require_fields(source_stage, {"passed": True, "stage": "auxiliary", "smoke": False,
                                  "role_mode": "baseline", "steps": 7866}, "CIR source audit")
    adapter_hash = sha256_file(source / "final_adapter/adapter_model.safetensors")
    if adapter_hash != source_stage["adapter_sha256"]:
        raise ValueError("CIR adapter hash mismatch")
    baseline_train = json.loads((baseline / "novel/train_contract.json").read_text())
    if baseline_train["init_adapter_model_sha256"] != adapter_hash:
        raise ValueError("comparison baseline did not use the same CIR weights")
    baseline_contract = json.loads((baseline / "ranking/shard-0/run_contract.json").read_text())
    rows = pq.read_table(reference / prepared["shards"]["0"]["path"]).to_pylist()
    mapping_path = reference / role_run.MAPPING.relative_to(role_run.BASE_HERE.parent)
    mapping = json.loads(mapping_path.read_text())
    details = latest_details(baseline / "ranking/shard-0/details.jsonl", baseline_contract["fingerprint"])
    baseline_metrics = independent_metrics(list(details.values()), rows, mapping)
    reported = json.loads((baseline / "RESULT_AUDIT.json").read_text())["metrics"]
    for name, value in baseline_metrics.items():
        if not math.isclose(value, reported[name], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"baseline metric mismatch: {name}")
    for path in reference.rglob("*"):
        if path.is_file() and not {"__pycache__", ".pytest_cache"}.intersection(path.parts):
            target = destination / path.relative_to(reference)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    new_root = destination / "eval_interest_id_minionerec/mass_readout"
    for name in MODULES:
        shutil.copyfile(HERE / name, destination / "eval_interest_id_minionerec" / name)
    for path in source.rglob("*"):
        if path.is_file():
            target = new_root / "source/auxiliary" / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for relative in ("novel/train_contract.json", "novel/train_metrics.json", "RESULT_AUDIT.json",
                     "ranking/shard-0/run_contract.json", "ranking/shard-0/metrics.json",
                     "ranking/shard-0/details.jsonl"):
        target = new_root / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline / relative, target)
    contract = {"method": METHOD, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_contract": str(old_contract.relative_to(reference)),
        "source_contract_sha256": sha256_file(old_contract),
        "source_adapter": "eval_interest_id_minionerec/mass_readout/source/auxiliary/final_adapter",
        "source_adapter_files": {path.name: sha256_file(path) for path in (source / "final_adapter").iterdir() if path.is_file()},
        "source_train_metrics_sha256": sha256_file(source / "train_metrics.json"),
        "baseline_metrics": baseline_metrics,
        "gate": {"minimum_ndcg20_gain": 0.005, "recall20_not_lower": True},
        "formal_steps": 3500, "smoke_steps": 7, "no_cir_retraining": True,
        "files_sha256": {str(path.relative_to(destination)): sha256_file(path)
                         for path in destination.rglob("*") if path.is_file()}}
    immutable_json(new_root / "CONTRACT.json", contract)
    print(json.dumps({"bundle": str(destination), "files": len(contract["files_sha256"]),
                      "contract_sha256": sha256_file(new_root / "CONTRACT.json"),
                      "baseline_metrics": baseline_metrics}), flush=True)


def verify():
    contract = json.loads(CONTRACT.read_text())
    require_fields(contract, {"method": METHOD, "formal_steps": 3500, "smoke_steps": 7,
                              "no_cir_retraining": True}, "mass contract")
    for relative, expected in contract["files_sha256"].items():
        if sha256_file(HERE.parent / relative) != expected:
            raise ValueError(f"bundle identity changed: {relative}")
    prepared = role_run.verify()
    if sha256_file(role_run.CONTRACT) != contract["source_contract_sha256"]:
        raise ValueError("source contract mismatch")
    return contract, prepared


def training_command(model_path, output, smoke, contract, prepared):
    command = role_run.training_command(model_path, output, "baseline", "novel", smoke, prepared)
    command[command.index('crossinterest.methods.role_projection.train')] = 'crossinterest.methods.mass_readout.train'
    for option in ("--stage", "--role-mode", "--role-contract"):
        position = command.index(option)
        del command[position:position + 2]
    command[command.index("--init-adapter") + 1] = str(HERE.parent / contract["source_adapter"])
    command.extend(["--mass-contract", str(CONTRACT)])
    return command


def audit_training(directory, smoke, contract, prepared, model_path):
    import torch
    from safetensors.torch import load_file

    info = prepared["files"]["smoke_novel" if smoke else "novel"]
    metrics = json.loads((directory / "train_metrics.json").read_text())
    original = prepared["g0_alignment"]["records"]["formal/novel/train_contract.json"]
    excluded = {"task", "label_map", "train_file", "init_adapter", "init_adapter_model_sha256",
                "init_adapter_config_sha256", "experiment_metadata"}
    if smoke:
        excluded.update({"train_sha256", "selected_train_rows", "selected_label_count_min",
                         "selected_label_count_max", "selected_label_count_max_gap", "formal_train_rows"})
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, {key: value for key, value in original.items() if key not in excluded}, "unchanged G0 N settings")
        require_fields(record, {"model": model_path, "train_sha256": info["sha256"],
            "init_adapter_model_sha256": contract["source_adapter_files"]["adapter_model.safetensors"]}, "same CIR source")
        require_fields(record["experiment_metadata"], {"method": METHOD, "stage": "novel", "smoke": smoke,
            "preflight_sha256": sha256_file(CONTRACT)}, "new method identity")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
        "sample_count": info["rows"], "epoch": 1.0}, "N completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid N loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    require_fields(runtime, prepared["g0_alignment"]["records"]["formal/novel/runtime_config.json"], "G0 runtime")
    adapter = directory / "final_adapter"
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    source = load_file(str(HERE.parent / contract["source_adapter"] / "adapter_model.safetensors"))
    if weights.keys() != source.keys() or any(not value.isfinite().all() for value in weights.values()):
        raise ValueError("nonfinite or changed LoRA parameter set")
    if not any(not torch.equal(value, source[name]) for name, value in weights.items() if "lora_B" in name):
        raise ValueError("N LoRA did not update")
    head = load_file(str(adapter / "interest_readout.safetensors"))["weight"]
    initial = load_file(str(directory / "initial_readout/interest_readout.safetensors"))["weight"]
    if initial.count_nonzero() != 0 or not head.isfinite().all() or head.count_nonzero() == 0:
        raise ValueError("invalid readout initialization or update")
    head_runtime = json.loads((directory / "MASS_RUNTIME.json").read_text())
    require_fields(head_runtime["metadata"], {"architecture": METHOD, "class_count": 112}, "readout architecture")
    old_count = json.loads((directory / "trainable_parameters.json").read_text())["count"]
    if head_runtime["total_trainable_parameters"] != old_count + head.numel():
        raise ValueError("unexpected trainable parameter change")
    if not (directory / "GRADIENT_0000.json").exists():
        raise ValueError("gradient mechanism evidence missing")
    result = {"passed": True, "smoke": smoke, "steps": info["steps"], "rows": info["rows"],
        "loss": metrics["train_loss"], "train_runtime": metrics["train_runtime"],
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "readout_sha256": sha256_file(adapter / "interest_readout.safetensors"),
        "readout_norm": head.float().norm().item(), "head_parameters": head.numel(),
        "source_adapter_sha256": contract["source_adapter_files"]["adapter_model.safetensors"]}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def evaluation_command(model_path, directory, smoke, prepared):
    command = role_run.evaluation_command(model_path, directory / "novel/final_adapter",
                                          directory / "ranking", "baseline", 0, smoke, prepared)
    command[command.index('crossinterest.methods.role_projection.evaluate')] = 'crossinterest.methods.mass_readout.evaluate'
    for option in ("--evaluation", "--role-mode"):
        position = command.index(option)
        del command[position:position + 2]
    return command


def audit_evaluation(directory, smoke, contract, prepared):
    import pyarrow.parquet as pq
    from crossinterest.training import evaluate_ranking as ranking

    output = directory / "ranking"
    observed = json.loads((output / "run_contract.json").read_text())
    original = prepared["g0_alignment"]["records"]["formal/ranking/run_contract.json"]
    excluded = {"adapter", "adapter_config_sha256", "dataset", "label_map", "label_file", "fingerprint"}
    limit = prepared["evaluation"]["smoke_limit"] if smoke else None
    require_fields(observed, {**{key: value for key, value in original.items() if key not in excluded},
                              "sample_limit": limit}, "original full vocabulary beam protocol")
    adapter = directory / "novel/final_adapter"
    require_fields(json.loads((output / "MASS_CONTRACT.json").read_text()), {
        "architecture": METHOD, "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "head_sha256": sha256_file(adapter / "interest_readout.safetensors")}, "evaluated weights")
    rows = pq.read_table(role_run.DEV).to_pylist()
    if smoke:
        rows = rows[:limit]
    details = list(latest_details(output / "details.jsonl", observed["fingerprint"]).values())
    mapping = json.loads(role_run.MAPPING.read_text())
    independent = independent_metrics(details, rows, mapping)
    recalculated = ranking.metrics_from_rankings(details, rows, 20)
    require_fields(json.loads((output / "metrics.json").read_text()), recalculated, "recomputed ranking metrics")
    for name, value in independent.items():
        if not math.isclose(value, recalculated[name], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"independent metric disagreement: {name}")
    require_fields(json.loads((output / "MASS_EVAL_AUDIT.json").read_text()),
                   {"passed": True, "head_applications": len(rows)}, "first-position head applications")
    health = (independent["strict_beam_format_rate"] == 1 and independent["valid_beam_rate"] == 1
              and independent["duplicate_valid_beam_rate"] == 0)
    result = {"passed": True, "smoke": smoke, "health_passed": health, "metrics": independent,
              "details_sha256": sha256_file(output / "details.jsonl"), "contract_sha256": sha256_file(CONTRACT)}
    if not smoke:
        names = ("macro_recall_at_10", "macro_recall_at_20", "macro_ndcg_at_10", "macro_ndcg_at_20")
        result["delta_vs_same_cir_baseline"] = {name: independent[name] - contract["baseline_metrics"][name] for name in names}
        delta = result["delta_vs_same_cir_baseline"]
        result["improvement_gate"] = {"ndcg20_gain_at_least_0_005": delta["macro_ndcg_at_20"] >= contract["gate"]["minimum_ndcg20_gain"],
                                      "recall20_not_lower": delta["macro_recall_at_20"] >= 0}
        result["effect_passed"] = health and all(result["improvement_gate"].values())
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def execute(output):
    from huggingface_hub import snapshot_download
    import torch
    from safetensors.torch import load_file

    contract, prepared = verify()
    output.mkdir(parents=True, exist_ok=True)
    role_run.audit_environment(prepared, output)
    source = load_file(str(HERE.parent / contract["source_adapter"] / "adapter_model.safetensors"))
    if not source or any(not value.isfinite().all() for value in source.values()):
        raise ValueError("nonfinite source CIR adapter")
    del source
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    immutable_json(output / "EXECUTION_CONTRACT.json", {"contract_sha256": sha256_file(CONTRACT),
        "method": METHOD, "model_path": model_path, "source_adapter_sha256": contract["source_adapter_files"]["adapter_model.safetensors"],
        "no_cir_retraining": True, "formal_does_not_inherit_smoke": True})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    results = {}
    for phase in ("smoke", "formal"):
        smoke = phase == "smoke"
        directory = output / phase
        directory.mkdir(parents=True, exist_ok=True)
        if not smoke:
            require_fields(json.loads((output / "smoke/COMPLETE.json").read_text()),
                {"passed": True, "contract_sha256": sha256_file(CONTRACT)}, "smoke execution gate")
        if not (directory / "novel/train_metrics.json").exists():
            subprocess.run(training_command(model_path, directory / "novel", smoke, contract, prepared),
                           cwd=HERE.parent, env=environment, check=True)
        stage = audit_training(directory / "novel", smoke, contract, prepared, model_path)
        print(json.dumps({"event": "stage_passed", "phase": phase, **stage}), flush=True)
        (directory / "ranking").mkdir(exist_ok=True)
        with (directory / "ranking/process.log").open("a") as stream:
            subprocess.run(evaluation_command(model_path, directory, smoke, prepared), cwd=HERE.parent,
                           env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = audit_evaluation(directory, smoke, contract, prepared)
        results[phase] = {"stage": stage, "evaluation": result}
        write_json_atomic(directory / "COMPLETE.json", {"passed": True, "smoke": smoke,
            "contract_sha256": sha256_file(CONTRACT), **results[phase]})
        print(json.dumps({"event": "phase_complete", "phase": phase, **result}), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "method": METHOD,
        "contract_sha256": sha256_file(CONTRACT), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    freezing = subparsers.add_parser("freeze")
    for name in ("destination", "reference", "source", "baseline"):
        freezing.add_argument("--" + name, type=Path, required=True)
    subparsers.add_parser("verify")
    running = subparsers.add_parser("execute")
    running.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.destination.resolve(), args.reference.resolve(), args.source.resolve(), args.baseline.resolve())
    elif args.action == "verify":
        contract, _ = verify()
        print(json.dumps({"passed": True, "method": contract["method"], "files": len(contract["files_sha256"])}))
    else:
        execute(args.output.resolve())


if __name__ == "__main__":
    main()