"""Immutable two-worker evaluation of the approved fixed probability mixture."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.common import append_jsonl, latest_details, write_json_atomic
from crossinterest.training import evaluate_ranking as ranking
from crossinterest.methods.alignment.data import HERE as FROZEN_HERE, sha256_file
from crossinterest.methods.cir.runner import TRAIN, DEV, MAPPING, TRAIN_SHA, DEV_SHA, MAPPING_SHA
from crossinterest.methods.mechanism.runner import immutable_json
from crossinterest.methods.fusion.model import HistoryLinearModel, bind_ranking_fusion


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "fusion"
PREFLIGHT = ROOT / "PREFLIGHT.json"
METHOD = "g0-linear-history-half-probability-v1"
ADAPTER_SHA = "ae6659bf4a4d6a8bf2fd3f69dbeba986a0601e96950811de625306015af6af57"
SCREEN_ARRAYS_SHA = "19d64fa5d3d7374557302b86bf2c06171db34bf4331ec1df4d26e25c0e28c7ab"
MODULES = ("linear_fusion.py", "linear_fusion_run.py", "test_linear_fusion.py")


def prepare():
    from transformers import AutoTokenizer
    from crossinterest.training.semantic_tokens import register_semantic_tokens

    if PREFLIGHT.exists():
        return verify()
    reference_root = HERE / "bidirectional/outputs/results_envfix"
    reference = json.loads((reference_root / "RAW_RESULT_AUDIT.json").read_text())
    assert reference["passed"] and not reference["smoke_only"]
    audited = reference["audits"]["qwen-cir-3ep"]["formal"]
    assert audited["training"]["adapter_sha256"] == ADAPTER_SHA
    source = reference_root / "qwen-cir-3ep/formal/novel/final_adapter"
    assert sha256_file(source / "adapter_model.safetensors") == ADAPTER_SHA
    source_preflight = json.loads((FROZEN_HERE / "cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json").read_text())
    arrays_path = HERE / "cir_factorization/outputs/N_LINEAR_HISTORY_SCREEN.npz"
    assert sha256_file(arrays_path) == SCREEN_ARRAYS_SHA
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        assert sha256_file(path) == expected
    ROOT.mkdir(parents=True, exist_ok=True)
    adapter = ROOT / "adapter"
    adapter.mkdir(exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(source / name, adapter / name)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if (ROOT / "linear_model.npz").exists():
            with np.load(ROOT / "linear_model.npz", allow_pickle=False) as existing:
                assert all(np.array_equal(existing[key], arrays[key]) for key in ("coefficients", "intercept"))
        else:
            with (ROOT / "linear_model.npz").open("xb") as handle:
                np.savez_compressed(handle, coefficients=arrays["coefficients"], intercept=arrays["intercept"])
    mapping = json.loads(MAPPING.read_text())
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    model = HistoryLinearModel(ROOT / "linear_model.npz", categories)
    assert model.coefficients.size + model.intercept.size == 12768
    tokenizer = AutoTokenizer.from_pretrained(source_preflight["model_repo"],
        revision=source_preflight["model_revision"], trust_remote_code=True, local_files_only=True)
    register_semantic_tokens(tokenizer, mapping)
    table = pq.read_table(DEV)
    rows = table.to_pylist()
    assert len(rows) == 2000 and len({row["sample_id"] for row in rows}) == 2000
    lengths = [len(ranking.render_prompt_ids(tokenizer, row, "qwen35_9b", "behavior_interest", "interest_id")) for row in rows]
    assert max(lengths) <= 8192
    shards = {}
    for index in range(2):
        positions = sorted(range(index, len(rows), 2), key=lambda position: -lengths[position])
        destination = ROOT / f"data/shard-{index}/{DEV.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.take(pa.array(positions)), destination)
        shards[str(index)] = {"path": str(destination.relative_to(ROOT)), "sha256": sha256_file(destination),
                             "rows": len(positions), "source_positions": positions,
                             "max_prompt_tokens": max(lengths[position] for position in positions)}
    files = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        adapter / "adapter_config.json", adapter / "adapter_model.safetensors", ROOT / "linear_model.npz")}
    result = {"passed": True, "method": METHOD, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "2026-09-21 user approved fixed fusion, full vocabulary and original candidate-free 20-beam",
        "model_repo": source_preflight["model_repo"], "model_revision": source_preflight["model_revision"],
        "source_adapter_sha256": ADAPTER_SHA, "screen_arrays_sha256": SCREEN_ARRAYS_SHA,
        "reference_audit_sha256": sha256_file(reference_root / "RAW_RESULT_AUDIT.json"),
        "reference_metrics": {key: value for key, value in audited["results"]["ranking"].items() if key.startswith("macro_")},
        "train_sha256": TRAIN_SHA, "dev_sha256": DEV_SHA, "mapping_sha256": MAPPING_SHA,
        "code_sha256": {name: sha256_file(HERE / name) for name in MODULES},
        "files": files, "shards": shards, "linear_parameters": 12768,
        "llm_retrained": False, "linear_refitted": False, "fusion_weight": 0.5,
        "evaluation": {"rows": 2000, "classes": 112, "beams": 20, "batch_size": 1,
                       "workers": 2, "gpus_per_worker": 4, "candidate_free": True,
                       "max_new_tokens": 32, "max_prompt_length": 8192,
                       "do_sample": False, "classification_included": False},
        "improvement_gate": "macro_ndcg_at_20 > G0 and macro_recall_at_20 >= G0"}
    immutable_json(PREFLIGHT, result)
    return verify()


def verify():
    prepared = json.loads(PREFLIGHT.read_text())
    assert prepared["passed"] and prepared["method"] == METHOD
    assert prepared["source_adapter_sha256"] == ADAPTER_SHA and prepared["fusion_weight"] == 0.5
    for relative, expected in prepared["files"].items():
        assert sha256_file(ROOT / relative) == expected, relative
    for relative, expected in prepared["code_sha256"].items():
        assert sha256_file(HERE / relative) == expected, relative
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA)):
        assert sha256_file(path) == expected
    original = pq.read_table(DEV).to_pylist()
    positions = []
    for info in prepared["shards"].values():
        assert sha256_file(ROOT / info["path"]) == info["sha256"]
        rows = pq.read_table(ROOT / info["path"]).to_pylist()
        assert len(rows) == info["rows"] == 1000
        assert rows == [original[position] for position in info["source_positions"]]
        positions.extend(info["source_positions"])
    assert sorted(positions) == list(range(2000))
    return prepared


def evaluate(model_path: str, output: Path, shard: int):
    prepared = verify()
    info = prepared["shards"][str(shard)]
    output.mkdir(parents=True, exist_ok=True)
    original_resolve = ranking.resolve_contract

    def resolve(path, contract, adapter):
        contract.update(fusion_method=METHOD, fusion_preflight_sha256=sha256_file(PREFLIGHT),
                        fusion_linear_sha256=prepared["files"]["linear_model.npz"],
                        fusion_weight=0.5, source_adapter_sha256=ADAPTER_SHA, fusion_shard=shard)
        contract.pop("fingerprint", None)
        contract["fingerprint"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return original_resolve(path, contract, adapter)

    ranking.resolve_contract = resolve
    mapping = json.loads(MAPPING.read_text())
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    linear = HistoryLinearModel(ROOT / "linear_model.npz", categories)

    def record_check(record):
        append_jsonl(output / "fusion_checks.jsonl", {**record, "method": METHOD,
                     "preflight_sha256": sha256_file(PREFLIGHT)})

    bind_ranking_fusion(ranking, linear, mapping["special_tokens"], record_check)
    sys.argv = [sys.argv[0], "--model-key", "qwen35_9b", "--model", model_path,
                "--adapter", str(ROOT / "adapter"), "--view", "behavior_interest", "--label-mode", "interest_id",
                "--label-map", str(MAPPING), "--label-file", str(TRAIN),
                "--data-dir", str((ROOT / info["path"]).parent), "--run-dir", str(output),
                "--device-map", "balanced", "--num-beams", "20", "--eval-batch-size", "1",
                "--decoding", "direct_beam_search", "--max-new-tokens", "32", "--max-prompt-length", "8192"]
    ranking.main()


def audit(output: Path):
    prepared = verify()
    rows = pq.read_table(DEV).to_pylist()
    gathered = {}
    evidence = {}
    for shard in range(2):
        directory = output / f"shard-{shard}"
        contract = json.loads((directory / "run_contract.json").read_text())
        expected = {"fusion_method": METHOD, "fusion_preflight_sha256": sha256_file(PREFLIGHT),
                    "fusion_linear_sha256": prepared["files"]["linear_model.npz"], "fusion_weight": 0.5,
                    "source_adapter_sha256": ADAPTER_SHA, "fusion_shard": shard,
                    "dataset_sha256": prepared["shards"][str(shard)]["sha256"], "label_map_sha256": MAPPING_SHA,
                    "label_file_sha256": TRAIN_SHA, "sample_limit": None, "num_beams": 20,
                    "num_return_sequences": 20, "do_sample": False, "candidates_in_prompt": False,
                    "decoder_label_constraint": False, "decoding": "candidate_free_direct_beam_search",
                    "max_new_tokens": 32, "eval_batch_size": 1, "label_mode": "interest_id",
                    "view": "behavior_interest", "model_device_map": "balanced"}
        assert all(contract.get(key) == value for key, value in expected.items())
        saved = latest_details(directory / "details.jsonl", contract["fingerprint"])
        expected_rows = pq.read_table(ROOT / prepared["shards"][str(shard)]["path"]).to_pylist()
        assert set(saved) == {row["sample_id"] for row in expected_rows}
        assert not gathered.keys() & saved.keys()
        checks = {row["sample_id"]: row for row in map(json.loads, (directory / "fusion_checks.jsonl").read_text().splitlines())}
        for row in expected_rows:
            detail = saved[row["sample_id"]]
            assert detail["target_interest"] == row["target_interest"]
            assert detail["source_sample_id"] == row["source_sample_id"]
            assert len(detail["ranking"]) == 20
            assert [entry["rank"] for entry in detail["ranking"]] == list(range(1, 21))
            assert all(math.isfinite(entry["sequence_score"]) for entry in detail["ranking"])
            for key, value in ranking.score_ranking(detail["ranking"], row["target_interest"]).items():
                assert detail[key] == value
            check = checks[row["sample_id"]]
            assert check["applied_calls"] == 1 and check["num_beams"] == 20 and check["max_mass_error"] <= 1e-5
            assert check["method"] == METHOD and check["preflight_sha256"] == sha256_file(PREFLIGHT)
        gathered.update(saved)
        shard_metrics = ranking.metrics_from_rankings(list(saved.values()), expected_rows, 20)
        cloud_metrics = json.loads((directory / "metrics.json").read_text())
        assert all(cloud_metrics[key] == value for key, value in shard_metrics.items())
        evidence[str(shard)] = {name: sha256_file(directory / name) for name in (
            "run_contract.json", "details.jsonl", "fusion_checks.jsonl", "metrics.json", "predictions.jsonl")}
    ordered = [gathered[row["sample_id"]] for row in rows]
    metrics = ranking.metrics_from_rankings(ordered, rows, 20)
    health = (metrics["complete"] and metrics["recorded_sample_count"] == 2000
              and metrics["class_count"] == 112 and metrics["missing_sample_count"] == 0
              and metrics["valid_beam_rate"] == metrics["strict_beam_format_rate"] == 1
              and metrics["duplicate_valid_beam_rate"] == 0)
    reference = prepared["reference_metrics"]
    gate = metrics["macro_ndcg_at_20"] > reference["macro_ndcg_at_20"] and metrics["macro_recall_at_20"] >= reference["macro_recall_at_20"]
    return {"passed": True, "method": METHOD, "preflight_sha256": sha256_file(PREFLIGHT),
            "metrics": metrics, "reference_metrics": reference, "health_passed": health,
            "improvement_gate_passed": gate, "improvement_supported": gate and health,
            "macro_deltas": {key: metrics[key] - value for key, value in reference.items()}, "evidence": evidence}


def build_launch():
    verify()
    package = FROZEN_HERE.parent
    receipt_path = FROZEN_HERE / "cir_factorization/PACKAGE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    assert all(sha256_file(package / relative) == expected for relative, expected in receipt["files"].items())
    launch = ROOT / "launch"
    launch.mkdir(exist_ok=False)
    shutil.copyfile(HERE / "residual/launch/bootstrap.py", launch / "bootstrap.py")
    paths = [HERE / name for name in MODULES] + [PREFLIGHT]
    paths.extend(ROOT / relative for relative in json.loads(PREFLIGHT.read_text())["files"])
    paths.extend(ROOT / info["path"] for info in json.loads(PREFLIGHT.read_text())["shards"].values())
    overlay = {}
    for source in paths:
        relative = str(source.relative_to(HERE.parent))
        assert relative not in receipt["files"]
        target = launch / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        overlay[relative] = {"source": str(target.relative_to(launch)), "sha256": sha256_file(source)}
    immutable_json(launch / "LAUNCH_MANIFEST.json", {"frozen_receipt_sha256": sha256_file(receipt_path), "overlay": overlay})
    print(json.dumps({"passed": True, "launch": str(launch), "overlay_files": len(overlay), "frozen_files": len(receipt["files"])}))


def execute(output: Path):
    from huggingface_hub import snapshot_download

    prepared = verify()
    subprocess.run([sys.executable, "-m", "pytest", str(HERE / "test_linear_fusion.py"), "-q"], check=True)
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {"preflight": prepared, "model_path": model_path})
    environment = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    processes = []
    try:
        for shard, devices in enumerate(("0,1,2,3", "4,5,6,7")):
            command = [sys.executable, "-u", "-m", 'crossinterest.methods.fusion.runner', "evaluate",
                       "--model", model_path, "--output", str(output / f"shard-{shard}"), "--shard", str(shard)]
            processes.append(subprocess.Popen(command, env={**environment, "CUDA_VISIBLE_DEVICES": devices}, cwd=FROZEN_HERE.parent))
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("Fusion evaluation worker failed; preserve completed rows for recovery")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    result = audit(output)
    write_json_atomic(output / "RESULT_AUDIT.json", result)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "method": METHOD,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "result_audit_sha256": sha256_file(output / "RESULT_AUDIT.json"),
        "health_passed": result["health_passed"], "improvement_gate_passed": result["improvement_gate_passed"]})
    print(json.dumps({key: result[key] for key in ("passed", "health_passed", "improvement_gate_passed", "macro_deltas")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "build-launch", "evaluate", "execute", "audit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.action in ("prepare", "verify"):
        result = prepare() if args.action == "prepare" else verify()
        print(json.dumps({key: result[key] for key in ("passed", "method", "evaluation", "reference_metrics")}))
    elif args.action == "build-launch":
        build_launch()
    elif args.output is None:
        parser.error("This action requires --output")
    elif args.action == "execute":
        execute(args.output.resolve())
    elif args.action == "audit":
        print(json.dumps(audit(args.output.resolve()), indent=2))
    else:
        if args.model is None or args.shard is None:
            parser.error("evaluate requires --model and --shard")
        evaluate(args.model, args.output.resolve(), args.shard)


if __name__ == "__main__":
    main()