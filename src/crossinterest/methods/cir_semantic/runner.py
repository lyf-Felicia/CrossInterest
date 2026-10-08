"""Execute matched true/shuffled semantic CIR with the frozen base pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.cir.runner import ROOT, TRAIN, DEV, MAPPING, preflight_path, audit_training
from crossinterest.methods.cir.runner import preflight as base_preflight
from crossinterest.methods.cir.runner import training_command as base_command
from crossinterest.methods.cir_semantic.data import ROOT as TARGET_ROOT, text_hash, residual_targets
from crossinterest.methods.mechanism.runner import immutable_json, audit_evaluation


VARIANTS = ("semantic_true", "semantic_shuffled")


def preflight(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unknown semantic CIR condition")
    baseline = base_preflight("qwen35_9b", "external_balanced")
    manifest_path = TARGET_ROOT / "MANIFEST.json"
    target_path = TARGET_ROOT / "targets.safetensors"
    manifest = json.loads(manifest_path.read_text())
    require_fields(manifest, {"passed": True, "future_titles_used": False, "dev_used": False,
                              "student_input_changed": False, "title_truncations": 0}, "semantic supervision")
    if sha256_file(target_path) != manifest["targets_sha256"]:
        raise ValueError("semantic targets changed")
    source = HERE.parent / baseline["files"]["auxiliary"]["path"]
    if manifest["identity"]["data_sha256"] != sha256_file(source):
        raise ValueError("semantic targets do not match CIR inputs")
    rows = pq.read_table(source, columns=["id", "target_interest"]).to_pylist()
    arrays = load_file(target_path)
    recovered, audit = residual_targets(arrays["history_embeddings"], [row["target_interest"] for row in rows],
                                        [row["id"] for row in rows])
    np.testing.assert_allclose(recovered["targets"], arrays["targets"], atol=1e-6)
    np.testing.assert_array_equal(recovered["permutation"], arrays["permutation"])
    require_fields(manifest["audit"], {"rows": 83888, "classes": 112, "dimension": 768,
        "permutation_fixed_points": 0, "permutation_cross_class": 0,
        "row_order_sha256": text_hash([row["id"] for row in rows])}, "semantic alignment")
    if audit["max_class_mean_abs"] > 1e-6:
        raise ValueError("semantic targets retain class means")
    sources = {**baseline["code_sha256"], **{f"eval_interest_id_minionerec/{name}": sha256_file(HERE / name)
        for name in ("cir_semantic_data.py", "cir_semantic_train.py", "cir_semantic_run.py")}}
    semantic = {"targets_sha256": manifest["targets_sha256"], "manifest_sha256": sha256_file(manifest_path),
                "dimension": 768, "loss_weight": 1.0, "shuffled": variant == "semantic_shuffled",
                "source_data_sha256": manifest["identity"]["data_sha256"],
                "encoder": manifest["identity"]["encoder"], "encoder_revision": manifest["identity"]["encoder_revision"],
                "projection": "nonaffine_layernorm_biasfree_linear_zero_init", "train_only": True}
    prepared = {**baseline, "variant": variant, "code_sha256": sources, "semantic": semantic,
                "baseline_preflight_sha256": sha256_file(preflight_path("qwen35_9b", "external_balanced")),
                "tokenizer_preflight_reused": "identical_input_files_and_frozen_CIR_renderer",
                "projection_dropped_for_novel_and_evaluation": True}
    path = preflight_path("qwen35_9b", variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(path, prepared)
    return prepared


def training_command(model_path: str, stage: str, directory: Path, smoke: bool,
                     prepared: dict, init_adapter: Path | None = None) -> list[str]:
    command = base_command(model_path, stage, directory, smoke, prepared, init_adapter)
    if stage == "auxiliary":
        command[command.index('crossinterest.methods.cir.train')] = 'crossinterest.methods.cir_semantic.train'
    return command


def audit_semantic_stage(directory: Path, stage: str, smoke: bool, prepared: dict, model_path: str) -> dict:
    result = audit_training(directory, stage, smoke, prepared, model_path)
    adapter = directory / "final_adapter"
    projection_path = adapter / "cir_semantic_projection.safetensors"
    if stage == "novel":
        if projection_path.exists() or (directory / "SEMANTIC_RUNTIME.json").exists():
            raise ValueError("semantic projection leaked into N stage")
        return result
    runtime = json.loads((directory / "SEMANTIC_RUNTIME.json").read_text())
    require_fields(runtime, {"identity": prepared["semantic"], "dimension": 768,
        "loss_weight": 1.0, "model_accepts_loss_kwargs": False, "target_detached": True,
        "targets_passed_to_backbone": False, "auxiliary_only": True}, "semantic runtime")
    weights = load_file(projection_path)
    if not weights or not all(np.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite semantic projection")
    if not any(np.count_nonzero(value) for value in weights.values()):
        raise ValueError("semantic projection did not update")
    result.update(semantic_projection_sha256=sha256_file(projection_path), semantic_runtime=runtime)
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    return result


def execute(variant: str, output: Path) -> None:
    from huggingface_hub import snapshot_download

    prepared = preflight(variant)
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
            stages[stage] = audit_semantic_stage(stage_dir, stage, smoke, prepared, model_path)
            previous = stage_dir / "final_adapter"
        for evaluation in ("classification", "ranking"):
            command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate', "--evaluation", evaluation,
                "--model-key", "qwen35_9b", "--model", model_path, "--adapter", str(previous), "--view", "behavior_interest",
                "--label-mode", "interest_id", "--label-map", str(MAPPING), "--data-dir", str(DEV.parent),
                "--expected-label-count", "112", "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
            if smoke:
                command.extend(["--limit", "4"])
            if evaluation == "ranking":
                command.extend(["--label-file", str(TRAIN), "--num-beams", "20", "--eval-batch-size", "1", "--decoding", "direct_beam_search"])
            subprocess.run(command, cwd=HERE.parent, env=environment, check=True)
        result = audit_evaluation(directory, "reconstruction", smoke, stages["novel"])
        result.update(model_key="qwen35_9b", variant=variant, stage_audits=stages)
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        print(json.dumps(result), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "model_key": "qwen35_9b", "variant": variant,
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "execute"))
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        prepared = preflight(args.variant)
        print(json.dumps({"passed": prepared["passed"], "variant": args.variant, "semantic": prepared["semantic"]}))
    elif args.output_dir is None:
        parser.error("execute requires --output-dir")
    else:
        execute(args.variant, args.output_dir.resolve())


if __name__ == "__main__":
    main()