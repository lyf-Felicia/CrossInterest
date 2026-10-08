"""Freeze and audit the isolated CIR factorization execution package."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import HERE, sha256_file
import crossinterest.methods.cir.runner as cir
from crossinterest.methods.factorization.loss import ARMS
from crossinterest.methods.mechanism.runner import immutable_json
from crossinterest.methods.factorization.probe import audit_probe


ROOT = _source_workspace('eval_interest_id_minionerec') / "cir_factorization"


def prepare_package() -> dict:
    source = ROOT / "gate_package"
    destination = ROOT / "package"
    previous = json.loads((ROOT / "GATE_PACKAGE_RECEIPT.json").read_text())
    files = dict(previous["files"])
    receipt_path = "eval_interest_id_minionerec/cir_factorization/GATE_PACKAGE_RECEIPT.json"
    files[receipt_path] = sha256_file(source / receipt_path)
    for relative, expected in files.items():
        if sha256_file(source / relative) != expected:
            raise ValueError(f"immutable gate package changed: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"existing execution dependency changed: {relative}")
        if not target.exists():
            shutil.copyfile(source / relative, target)
    for name in ("cir_factor_train.py", "cir_factor_run.py", "cir_factor_probe.py", "cir_factor_data.py", "cir_data.py", "test_cir_factorization.py"):
        original = _source_workspace('eval_interest_id_minionerec') / name
        target = destination / "eval_interest_id_minionerec" / name
        shutil.copyfile(original, target)
        files[str(target.relative_to(destination))] = sha256_file(original)
    for name in ("GATE_INDEPENDENT_AUDIT.json", "GATE_TOKENIZER_PREFLIGHT.json"):
        target = destination / "eval_interest_id_minionerec/cir_factorization" / name
        shutil.copyfile(ROOT / name, target)
        files[str(target.relative_to(destination))] = sha256_file(target)
    original_n = _source_workspace('eval_interest_id_minionerec') / "bidirectional/outputs/results_envfix/qwen-cir-3ep/formal/novel"
    original_audit = json.loads((original_n / "STAGE_AUDIT.json").read_text())
    if not original_audit["passed"] or original_audit["steps"] != 3500 or original_audit["rows"] != 112000:
        raise ValueError("G0 original N stage is not complete")
    if sha256_file(original_n / "final_adapter/adapter_model.safetensors") != original_audit["adapter_sha256"]:
        raise ValueError("G0 original N weight hash differs")
    for source_adapter in (original_n / "final_adapter").iterdir():
        if source_adapter.is_file():
            target = destination / "eval_interest_id_minionerec/cir_factorization/reference_adapters/novel" / source_adapter.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_adapter, target)
            files[str(target.relative_to(destination))] = sha256_file(source_adapter)
    result = {"passed": True, "files": files, "gate_receipt_sha256": sha256_file(ROOT / "GATE_PACKAGE_RECEIPT.json"),
              "gate_independent_audit_sha256": sha256_file(ROOT / "GATE_INDEPENDENT_AUDIT.json")}
    for relative, expected in files.items():
        if sha256_file(destination / relative) != expected:
            raise ValueError(f"execution copy differs: {relative}")
    write_json_atomic(destination / "eval_interest_id_minionerec/cir_factorization/PACKAGE_RECEIPT.json", result)
    write_json_atomic(ROOT / "PACKAGE_RECEIPT.json", result)
    return {"passed": True, "files_verified": len(files), "path": str(destination)}


def sources() -> dict:
    return {**cir.sources(), **{f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name)
                               for name in ("cir_factor_loss.py", "cir_factor_train.py", "cir_factor_run.py", "cir_factor_probe.py")}}


def preflight() -> dict:
    from transformers import AutoTokenizer
    from crossinterest.training.label_space import load_label_mapping
    from crossinterest.training.semantic_tokens import register_semantic_tokens
    from crossinterest.methods.factorization.train import FactorDataset

    prepared = cir.preflight("qwen35_9b", "external_balanced", 3)
    gate = json.loads((ROOT / "GATE_INDEPENDENT_AUDIT.json").read_text())
    if not gate["passed"] or not all(value["passed"] for value in gate["branches"].values()):
        raise ValueError("both factor branches require a passed independent intervention gate")
    path = ROOT / "PREFLIGHT.json"
    identity = {"code_sha256": sources(), "original_preflight_sha256": sha256_file(cir.preflight_path(
        "qwen35_9b", "external_balanced", 3)), "factor_manifest_sha256": sha256_file(ROOT / "data/MANIFEST.json"),
        "gate_audit_sha256": sha256_file(ROOT / "GATE_INDEPENDENT_AUDIT.json")}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(key) != value for key, value in identity.items()):
            old_sources = dict(result["code_sha256"])
            new_sources = dict(identity["code_sha256"])
            for name in ("cir_factor_run.py", "cir_factor_probe.py"):
                old_sources.pop(f"eval_interest_id_minionerec/{name}", None)
                new_sources.pop(f"eval_interest_id_minionerec/{name}", None)
            if old_sources != new_sources or any(result.get(key) != value for key, value in identity.items() if key != "code_sha256"):
                raise ValueError("factor tokenization dependencies or data changed")
            immutable_json(ROOT / "TOKENIZER_EQUIVALENCE_ORIGINAL.json", result)
            result = {**result, **identity, "tokenizer_receipt_reused_after_orchestration_only_change": True}
            write_json_atomic(path, result)
        return result
    mapping = load_label_mapping(cir.MAPPING, expected_count=112)
    tokenizer = AutoTokenizer.from_pretrained(prepared["model_repo"], revision=prepared["model_revision"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    registration = register_semantic_tokens(tokenizer, mapping)
    if registration != prepared["token_registration"]:
        raise ValueError("factor tokenizer registration differs from G0")
    info = prepared["files"]["auxiliary"]
    dataset = FactorDataset(HERE.parent / info["path"], tokenizer, "behavior_interest", "interest_id", mapping,
        8192, None, None, None, info["rows"], stage="auxiliary", relation=False,
        factor_arm="baseline", factor_masks=ROOT / "data/masks.npz")
    digest = hashlib.sha256()
    lengths = []
    for index in range(len(dataset)):
        encoded = dataset[index]
        mask = encoded.pop("seen_mask")
        if any(positive and visible for positive, visible in zip(encoded["positive_mask"], mask)):
            raise ValueError("factor auxiliary target leaked into history")
        digest.update(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        lengths.append(len(encoded["input_ids"]))
        if (index + 1) % 10000 == 0:
            print(json.dumps({"event": "factor_tokenization", "rows": index + 1}), flush=True)
    if digest.hexdigest() != prepared["tokenized_sha256"]["auxiliary"]:
        raise ValueError("factor input token sequence differs from G0")
    if sum(lengths) != prepared["train_tokens_per_epoch"]["auxiliary"]:
        raise ValueError("factor token budget differs from G0")
    result = {"passed": True, **identity, "original": prepared, "arms": ARMS,
              "auxiliary_token_hash_equal_to_g0": True, "novel_class_and_loss_unchanged": True,
              "auxiliary_rows": len(dataset), "auxiliary_tokens_per_epoch": sum(lengths), "max_tokens": max(lengths),
              "g0_reuse_requires_cloud_trainer_equivalence": True,
              "formal_requires_smoke": True, "smoke_weights_reused": False}
    immutable_json(path, result)
    return result


def training_command(model_path: str, stage: str, directory: Path, smoke: bool,
                     prepared: dict, arm: str, init_adapter: Path | None = None) -> list[str]:
    command = cir.training_command(model_path, stage, directory, smoke, prepared, init_adapter)
    command[command.index('crossinterest.methods.cir.train')] = 'crossinterest.methods.factorization.train'
    command.extend(["--factor-arm", arm])
    return command


def verify_package() -> dict:
    receipt = json.loads((ROOT / "PACKAGE_RECEIPT.json").read_text())
    for relative, expected in receipt["files"].items():
        if sha256_file(HERE.parent / relative) != expected:
            raise ValueError(f"factor execution package changed: {relative}")
    return receipt


def trainer_equivalence(output: Path) -> dict:
    import transformers
    import torch

    identity = {"code_sha256": sources(), "transformers": transformers.__version__, "torch": torch.__version__,
                "test_sha256": sha256_file(HERE / "test_cir_factorization.py")}
    path = output / "TRAINER_EQUIVALENCE.json"
    if path.exists():
        result = json.loads(path.read_text())
        if not result["passed"] or any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("previous Trainer equivalence receipt differs")
        return result
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "test_cir_factorization.py")],
                   cwd=HERE.parent, check=True)
    result = {"passed": True, **identity, "scope": "actual_cloud_Trainer_GA8_bitwise_small_model_and_real_9B_logit_gate",
              "novel_implementation": "unchanged_original_MechanismDataset_and_MechanismTrainer",
              "full_9b_trajectory_bitwise_rerun": False}
    immutable_json(path, result)
    return result


def audit_stage(directory, stage, smoke, prepared, model_path, arm):
    result = cir.audit_training(directory, stage, smoke, prepared, model_path)
    metadata = json.loads((directory / "train_metrics.json").read_text())["experiment_metadata"]
    expected = {"factor_arm": arm, "factor_manifest_sha256": sha256_file(ROOT / "data/MANIFEST.json"),
                "factor_masks_sha256": sha256_file(ROOT / "data/masks.npz"),
                "factor_loss_sha256": sha256_file(HERE / "cir_factor_loss.py")}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("factor training metadata differs")
    if stage == "auxiliary":
        runtime = json.loads((directory / "final_adapter/FACTOR_RUNTIME.json").read_text())
        if runtime["arm"] != arm or runtime["model_accepts_loss_kwargs"]:
            raise ValueError("factor runtime arm or GA scaling invalid")
    result.update(factor_arm=arm, factor_metadata=expected)
    write_json_atomic(directory / "FACTOR_STAGE_AUDIT.json", result)
    return result


def audit_output(output: Path, smoke_only=False) -> dict:
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    prepared = preflight()
    if execution["factor_preflight_sha256"] != sha256_file(ROOT / "PREFLIGHT.json"):
        raise ValueError("output belongs to a different factor preflight")
    identity = execution["factor_arm"]
    phases = {}
    for smoke in ((True,) if smoke_only else (True, False)):
        directory = output / ("smoke" if smoke else "formal")
        stages = {stage: audit_stage(directory / stage, stage, smoke, prepared["original"],
                                   execution["model_path"], identity) for stage in ("auxiliary", "novel")}
        probes = {stage: audit_probe(directory / f"{stage}_PROBE.json", directory / stage / "final_adapter",
                                    4 if smoke else None) for stage in ("auxiliary", "novel")}
        result = cir.audit_evaluation(directory, "reconstruction", smoke, stages["novel"])
        result.update(factor_arm=identity, stage_audits=stages, probes=probes)
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        phases[directory.name] = result
    return {"passed": True, "arm": identity, "phases": phases}


def execute(arm: str, output: Path, smoke_only=False):
    from huggingface_hub import snapshot_download

    if arm == "baseline":
        raise ValueError("G0 formal rerun is not authorized")
    verify_package()
    factor = preflight()
    prepared = factor["original"]
    output.mkdir(parents=True, exist_ok=True)
    trainer_equivalence(output)
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    immutable_json(output / "EXECUTION_CONTRACT.json", {"factor_arm": arm, "model_path": model_path,
        "factor_preflight_sha256": sha256_file(ROOT / "PREFLIGHT.json"), "code_sha256": sources(),
        "data_contract": prepared, "fresh_formal_after_smoke": True})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    for smoke in ((True,) if smoke_only else (True, False)):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        previous, stages = None, {}
        for stage in ("auxiliary", "novel"):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, stage, stage_dir, smoke, prepared, arm, previous),
                               cwd=HERE.parent, env=environment, check=True)
            stages[stage] = audit_stage(stage_dir, stage, smoke, prepared, model_path, arm)
            previous = stage_dir / "final_adapter"
            subprocess.run([sys.executable, "-m", 'crossinterest.methods.factorization.probe',
                "--model", model_path, "--adapter", str(previous), "--output", str(directory / f"{stage}_PROBE.json")]
                + (["--limit", "4"] if smoke else []), cwd=HERE.parent, env=environment, check=True)
        for evaluation in ("classification", "ranking"):
            command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate',
                "--evaluation", evaluation, "--model-key", "qwen35_9b", "--model", model_path,
                "--adapter", str(previous), "--view", "behavior_interest", "--label-mode", "interest_id",
                "--label-map", str(cir.MAPPING), "--data-dir", str(cir.DEV.parent), "--expected-label-count", "112",
                "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(cir.TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                                "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        audit = audit_output(output, smoke_only=smoke)
        print(json.dumps({"event": "factor_phase_passed", "arm": arm, "phase": directory.name}), flush=True)
    complete = {"passed": True, "arm": arm, "smoke_only": smoke_only,
                "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")}
    if not smoke_only:
        complete["formal_audit_sha256"] = sha256_file(output / "formal/RESULT_AUDIT.json")
    write_json_atomic(output / ("SMOKE_COMPLETE.json" if smoke_only else "COMPLETE.json"), complete)
    return complete


def reference_probes(output: Path):
    from huggingface_hub import snapshot_download

    verify_package()
    prepared = preflight()["original"]
    output.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    for name in ("aux3", "novel"):
        subprocess.run([sys.executable, "-m", 'crossinterest.methods.factorization.probe',
            "--model", model_path, "--adapter", str(ROOT / "reference_adapters" / name),
            "--output", str(output / f"{name}_PROBE.json")], cwd=HERE.parent, check=True)
    result = {"passed": True, "training_performed": False,
              "audits": {name: audit_probe(output / f"{name}_PROBE.json", ROOT / "reference_adapters" / name)
                         for name in ("aux3", "novel")}}
    write_json_atomic(output / "REFERENCE_PROBE_AUDIT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-package", "preflight", "execute", "audit-output", "reference-probes"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "prepare-package":
        result = prepare_package()
    elif arguments.action == "preflight":
        result = preflight()
    elif arguments.output is None:
        parser.error("--output required")
    elif arguments.action == "audit-output":
        result = audit_output(arguments.output.resolve(), arguments.smoke_only)
    elif arguments.action == "reference-probes":
        result = reference_probes(arguments.output.resolve())
    elif arguments.arm is None:
        parser.error("--arm required")
    else:
        result = execute(arguments.arm, arguments.output.resolve(), arguments.smoke_only)
    print(json.dumps({key: value for key, value in result.items() if key not in ("original", "code_sha256")}))