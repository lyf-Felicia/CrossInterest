"""Versioned one-arm CIR residual transfer, using the completed G0 warm start."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE as FROZEN_HERE, sha256_file
from crossinterest.methods.cir.runner import TRAIN, DEV, MAPPING, TRAIN_SHA, DEV_SHA, MAPPING_SHA
from crossinterest.methods.crossfit.train import teacher_command
from crossinterest.methods.mechanism.runner import audit_evaluation, immutable_json
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismTrainer
from crossinterest.methods.residual.model import freeze_cir_and_reset_n
from crossinterest.methods.consistency.checkpoint_io import install, install_resume


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "residual"
PREFLIGHT = ROOT / "PREFLIGHT.json"
SOURCE = ROOT / "source_adapter"
METHOD = "frozen-cir-fresh-n-lora-v1"
SOURCE_SHA = "a6a6edf4255ba823d66ce0d9af370baf2a4b7ce8c3be2440a4d56238149a8e6f"


def prepare():
    import torch
    from safetensors.torch import load_file

    if PREFLIGHT.exists():
        return verify()
    old_root = HERE / "bidirectional/outputs/results_envfix"
    old = old_root / "qwen-cir-3ep/formal"
    independent = json.loads((old_root / "RAW_RESULT_AUDIT.json").read_text())
    require_fields(independent, {"passed": True, "smoke_only": False}, "G0 independent audit")
    audited = independent["audits"]["qwen-cir-3ep"]["formal"]
    stage = json.loads((old / "auxiliary/STAGE_AUDIT.json").read_text())
    require_fields(stage, {"passed": True, "rows": 83888, "steps": 7866,
                          "adapter_sha256": SOURCE_SHA}, "completed G0 CIR")
    require_fields(audited["stage_audits"]["auxiliary"], stage, "G0 audit chain")
    metrics = json.loads((old / "auxiliary/train_metrics.json").read_text())
    require_fields(metrics, {"global_step": 7866, "sample_count": 83888, "epochs": 3.0,
                            "label_map_sha256": MAPPING_SHA}, "G0 source training")
    require_fields(json.loads((old / "novel/train_metrics.json").read_text()),
                   {"init_adapter_model_sha256": SOURCE_SHA}, "same G0 N start")
    reference_path = FROZEN_HERE / "cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json"
    reference = json.loads(reference_path.read_text())
    require_fields(reference, {"passed": True, "train_sha256": TRAIN_SHA,
        "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA}, "frozen N preflight")
    adapter = old / "auxiliary/final_adapter"
    if sha256_file(adapter / "adapter_model.safetensors") != SOURCE_SHA:
        raise ValueError("G0 CIR weights changed")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid CIR source weights")
    count = sum(value.numel() for value in weights.values())
    if count != stage["mechanism_runtime"]["total_trainable_parameters"]:
        raise ValueError("unexpected source parameter payload")
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        destination = SOURCE / name
        if destination.exists() and sha256_file(destination) != sha256_file(adapter / name):
            raise ValueError("different residual source already present")
        if not destination.exists():
            shutil.copyfile(adapter / name, destination)
    evidence = ROOT / "source_evidence"
    evidence.mkdir(exist_ok=True)
    sources = {"RAW_RESULT_AUDIT.json": old_root / "RAW_RESULT_AUDIT.json",
               "CIR_STAGE_AUDIT.json": old / "auxiliary/STAGE_AUDIT.json",
               "CIR_TRAIN_METRICS.json": old / "auxiliary/train_metrics.json",
               "G0_RESULT_AUDIT.json": old / "RESULT_AUDIT.json"}
    for name, path in sources.items():
        destination = evidence / name
        if destination.exists() and sha256_file(destination) != sha256_file(path):
            raise ValueError("different source evidence already present")
        if not destination.exists():
            shutil.copyfile(path, destination)
    result = {"passed": True, "schema_version": 1, "method": METHOD,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": reference["model_repo"], "model_revision": reference["model_revision"],
        "source_adapter_sha256": SOURCE_SHA, "source_config_sha256": sha256_file(SOURCE / "adapter_config.json"),
        "source_evidence_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "reference_metrics": {key: value for key, value in audited["results"]["ranking"].items() if key.startswith("macro_")},
        "reference_n_adapter_sha256": audited["training"]["adapter_sha256"],
        "reference_layout": "historical G0; new N uses 8 ranks without gradient checkpointing",
        "trainable_parameters": count, "frozen_cir_lora_parameters": sum(
            value.numel() for name, value in weights.items() if "lora_" in name),
        "code_sha256": {name: sha256_file(HERE / name) for name in (
            "cir_residual_model.py", "cir_residual_run.py", "dual_checkpoint_io.py")},
        "baseline_sha256": sha256_file(Path(baseline.__file__)),
        "files": {name: reference["files"][name] for name in ("novel", "smoke_novel")},
        "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
        "max_novel_tokens": reference["max_train_tokens"]["novel"],
        "initialization_probe_token_ids": list(reference["token_registration"]["token_ids"].values())[:2],
        "cir_retrained": False, "n_epochs": 1, "n_steps": 3500, "effective_batch_size": 32,
        "fresh_formal_after_smoke": True, "teacher_or_additional_data": False,
        "cir_branch_dropout": 0.0, "n_lora_dropout": 0.05,
        "total_rank_capacity_increased": True, "world_knowledge_mechanism_proven": False,
        "improvement_gate": "macro_ndcg_at_20 > G0 and macro_recall_at_20 >= G0",
        "evaluation": {"rows": 2000, "classes": 112, "beams": 20, "candidate_free": True, "do_sample": False}}
    immutable_json(PREFLIGHT, result)
    return verify()


def verify():
    prepared = json.loads(PREFLIGHT.read_text())
    require_fields(prepared, {"passed": True, "method": METHOD, "source_adapter_sha256": SOURCE_SHA}, "residual preflight")
    for relative, expected in prepared["code_sha256"].items():
        if sha256_file(HERE / relative) != expected:
            raise ValueError("residual code changed after preflight")
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA),
                           (SOURCE / "adapter_model.safetensors", SOURCE_SHA),
                           (SOURCE / "adapter_config.json", prepared["source_config_sha256"]),
                           (Path(baseline.__file__), prepared["baseline_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"residual dependency changed: {path.name}")
    for info in prepared["files"].values():
        if sha256_file(FROZEN_HERE.parent / info["path"]) != info["sha256"]:
            raise ValueError("residual N data changed")
    return prepared


def train(remaining):
    import torch

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args(remaining)
    args = baseline.parse_args(remaining)
    prepared = verify()
    info = prepared["files"]["smoke_novel" if options.smoke else "novel"]
    require_fields(vars(args), {"view": "behavior_interest", "label_mode": "interest_id",
        "loss_mode": "causal_lm", "label_score_head": "lm_head", "selection_file": None,
        "limit": None, "max_steps": -1, "epochs": 1.0, "learning_rate": 1e-4,
        "expected_effective_batch_size": 32, "expected_train_rows": info["rows"],
        "per_device_batch_size": 1, "gradient_accumulation_steps": 4,
        "gradient_checkpointing": False, "save_steps": 100,
        "ddp_find_unused_parameters": False, "require_qwen_fast_path": True}, "residual N budget")
    if (args.init_adapter.resolve() != SOURCE.resolve()
            or sha256_file(args.train_file) != info["sha256"]
            or sha256_file(args.label_map) != MAPPING_SHA
            or Path(args.model).name != prepared["model_revision"]
            or int(os.environ.get("WORLD_SIZE", "1")) != 8):
        raise ValueError("residual source/model/data/world size mismatch")
    prior_contract = (args.output_dir / "train_contract.json").exists()
    checkpoints = list(args.output_dir.glob("checkpoint-*"))
    if prior_contract and not checkpoints:
        raise ValueError("interrupted N has no complete checkpoint; explicit recovery decision required")
    install(args.output_dir)
    if checkpoints:
        install_resume(baseline)
    original = baseline.attach_or_load_lora

    def attach(model, config, adapter):
        model = original(model, config, adapter)
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        model.to(device)
        model.eval()
        token_ids = [1, *prepared["initialization_probe_token_ids"], 2]
        sample = torch.tensor([token_ids], device=model.get_input_embeddings().weight.device)
        with torch.no_grad():
            expected = model(input_ids=sample).logits.float()
        receipt = freeze_cir_and_reset_n(model)
        with torch.no_grad():
            actual = model(input_ids=sample).logits.float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        receipt.update(initial_logits_exact=True, source_adapter_sha256=SOURCE_SHA,
                       rank=int(os.environ.get("RANK", "0")))
        if receipt["trainable_parameters"] != prepared["trainable_parameters"]:
            raise ValueError("real model trainable parameter count differs")
        if receipt["rank"] == 0:
            write_json_atomic(args.output_dir / "RESIDUAL_INITIALIZATION.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "modules"}), flush=True)
        model.train()
        return model

    baseline.attach_or_load_lora = attach
    metadata = {"task": "amazon_cir_residual_n", "prompt_schema_version": "interest-mechanism-v1",
        "method": METHOD, "stage": "novel", "smoke": options.smoke,
        "source_adapter_sha256": SOURCE_SHA, "preflight_sha256": sha256_file(PREFLIGHT),
        "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True,
        "relation_head": False, "cir_branch_dropout": 0.0, "auxiliary_retrained": False}
    baseline.main(args=args, experiment_metadata=metadata,
        dataset_class=partial(MechanismDataset, stage="novel", relation=False),
        trainer_class=partial(MechanismTrainer, stage="novel", relation=False, init_adapter=SOURCE))


def evaluate(remaining):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    import crossinterest.methods.mechanism.evaluate as mechanism_evaluate

    verify()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluation", choices=("classification", "ranking"), required=True)
    options, rest = parser.parse_known_args(remaining)
    if options.evaluation == "ranking":
        from crossinterest.training import evaluate_ranking as evaluator
    else:
        from crossinterest.training import evaluate as evaluator
    args = evaluator.parse_args(rest)
    original = evaluator.load_model
    contract = {"method": METHOD, "source_adapter_sha256": SOURCE_SHA,
                "n_adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
                "preflight_sha256": sha256_file(PREFLIGHT), "evaluation": options.evaluation}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(args.run_dir / "RESIDUAL_CONTRACT.json", contract)

    def load(model_key, model_path, adapter, tokenizer, mapping, registration, **kwargs):
        model = original(model_key, model_path, SOURCE, tokenizer, mapping, registration, **kwargs)
        freeze_cir_and_reset_n(model)
        weights = load_file(str(adapter / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(model, weights)
        if result.unexpected_keys:
            raise ValueError("unexpected residual adapter weights")
        model.eval()
        return model

    evaluator.load_model = load
    sys.argv = [sys.argv[0], *remaining]
    mechanism_evaluate.main()


def training_command(model_path, directory, smoke, prepared):
    info = prepared["files"]["smoke_novel" if smoke else "novel"]
    command = teacher_command(model_path, FROZEN_HERE.parent / info["path"], directory, info["rows"])
    command[command.index("--nproc_per_node=4")] = "--nproc_per_node=8"
    position = command.index('crossinterest.training.train')
    command[position:position + 1] = ['crossinterest.methods.residual.runner', "train"]
    command[command.index("--gradient-accumulation-steps") + 1] = "4"
    command[command.index("--gradient-checkpointing")] = "--no-gradient-checkpointing"
    command[command.index("--save-steps") + 1] = "100"
    command.extend(["--init-adapter", str(SOURCE), "--resume"])
    if smoke:
        command.append("--smoke")
    return command


def audit_stage(directory, smoke, prepared, model_path):
    import torch
    from safetensors.torch import load_file

    info = prepared["files"]["smoke_novel" if smoke else "novel"]
    metrics = json.loads((directory / "train_metrics.json").read_text())
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, {"model": model_path, "train_sha256": info["sha256"],
            "label_map_sha256": MAPPING_SHA, "view": "behavior_interest", "label_mode": "interest_id",
            "epochs": 1.0, "learning_rate": 1e-4, "effective_batch_size": 32,
            "loss_mode": "causal_lm", "selection_file": None, "sample_limit": None,
            "semantic_label_token_weight": 1.0, "init_adapter_model_sha256": SOURCE_SHA}, "N contract")
        require_fields(record["experiment_metadata"], {"method": METHOD, "smoke": smoke,
            "preflight_sha256": sha256_file(PREFLIGHT), "source_adapter_sha256": SOURCE_SHA}, "method identity")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
                             "sample_count": info["rows"], "epoch": 1.0}, "N completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("nonfinite N loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    require_fields(runtime, {"world_size": 8, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4, "gradient_checkpointing": False,
        "ddp_find_unused_parameters": False}, "N layout")
    require_fields(runtime["qwen_fast_path"], {"available": True}, "fast path")
    require_fields(json.loads((directory / "RESIDUAL_INITIALIZATION.json").read_text()), {
        "initial_logits_exact": True, "source_adapter_sha256": SOURCE_SHA,
        "trainable_parameters": prepared["trainable_parameters"]}, "residual initialization")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "N LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite N weights")
    if not any(value.count_nonzero().item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("residual LoRA never updated")
    result = {"passed": True, "method": METHOD, "stage": "novel", "rows": info["rows"],
              "steps": info["steps"], "source_adapter_sha256": SOURCE_SHA,
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
              "relation_sha256": None, "train_runtime_seconds": metrics["train_runtime"]}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def execute(output):
    from huggingface_hub import snapshot_download

    prepared = verify()
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"],
                                   cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {**prepared, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        stage = directory / "novel"
        if not (stage / "train_metrics.json").exists():
            subprocess.run(training_command(model_path, stage, smoke, prepared),
                           cwd=FROZEN_HERE.parent, env=environment, check=True)
        training = audit_stage(stage, smoke, prepared, model_path)
        processes = []
        try:
            for evaluation, devices in (("classification", "0,1"), ("ranking", "2,3,4,5,6,7")):
                command = [sys.executable, "-m", 'crossinterest.methods.residual.runner', "evaluate",
                    "--evaluation", evaluation, "--model-key", "qwen35_9b", "--model", model_path,
                    "--adapter", str(stage / "final_adapter"), "--view", "behavior_interest", "--label-mode", "interest_id",
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
                    raise RuntimeError("residual evaluation failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        result = audit_evaluation(directory, "reconstruction", smoke, training)
        result.update(method=METHOD, preflight_sha256=sha256_file(PREFLIGHT))
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps({"event": "residual_evaluation_audited", "smoke": smoke, "passed": True}), flush=True)
    result_metrics = result["results"]["ranking"]
    reference = prepared["reference_metrics"]
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "method": METHOD,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "improvement_gate_passed": (result_metrics["macro_ndcg_at_20"] > reference["macro_ndcg_at_20"]
                                    and result_metrics["macro_recall_at_20"] >= reference["macro_recall_at_20"]),
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "train", "evaluate", "execute"))
    parser.add_argument("--output", type=Path)
    args, remaining = parser.parse_known_args()
    if args.action in ("prepare", "verify"):
        result = prepare() if args.action == "prepare" else verify()
        print(json.dumps({key: result[key] for key in (
            "passed", "method", "trainable_parameters", "reference_metrics", "files")}, indent=2))
    elif args.action == "train":
        train(remaining)
    elif args.action == "evaluate":
        evaluate(remaining)
    elif args.output is None:
        parser.error("execute requires --output")
    else:
        execute(args.output.resolve())


if __name__ == "__main__":
    main()