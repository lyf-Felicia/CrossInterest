#!/usr/bin/env python3
"""Audit stage provenance and recompute frozen Dev metrics from raw records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from crossinterest.evaluation.common import latest_details, metrics_from_details
from crossinterest.training.evaluate_ranking import metrics_from_rankings, score_ranking
from crossinterest.methods.alignment.data import BASE, VIEWS, experiment_dir, sha256_file


MODELS = {
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "llama31_8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


def require_fields(record: dict, expected: dict, context: str) -> None:
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{context}: {key}: expected {value!r}, got {record.get(key)!r}")


def audit(run_dir: Path, model_key: str, smoke: bool, view: str = "interest") -> dict:
    manifest_path = experiment_dir(view) / "data" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require_fields(manifest, {"view": view}, "manifest")
    contracts = {}
    stage_audits = {}
    for stage in ("warmup", "recommendation"):
        stage_dir = run_dir / stage
        contract = json.loads((stage_dir / "train_contract.json").read_text())
        metrics = json.loads((stage_dir / "train_metrics.json").read_text())
        stage_audit = json.loads((stage_dir / "stage_audit.json").read_text())
        expected = {
            "model": MODELS[model_key], "view": view, "label_mode": "interest_id",
            "label_map_sha256": manifest["mapping_sha256"],
            "train_sha256": manifest[stage]["sha256"], "epochs": manifest[stage]["epochs"],
            "effective_batch_size": 32, "learning_rate": 1e-4, "seed": 20260825,
        }
        for record in (contract, metrics):
            require_fields(record, expected, stage)
        metadata = contract["experiment_metadata"]
        require_fields(metadata, {
            "stage": stage, "smoke": smoke, "manifest_sha256": sha256_file(manifest_path),
            "task_counts": manifest[stage]["task_counts"], "max_steps": 2 if smoke else -1,
        }, stage)
        require_fields(metrics, {"experiment_metadata": metadata, "sample_count": manifest[stage]["rows"]}, stage)
        require_fields(stage_audit, {"passed": True, "stage": stage, "smoke": smoke}, stage)
        if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
            raise ValueError(f"{stage}: invalid loss")
        if not smoke and metrics.get("epoch", 0) < manifest[stage]["epochs"]:
            raise ValueError(f"{stage}: unfinished epoch budget")
        contracts[stage] = contract
        stage_audits[stage] = stage_audit
    require_fields(contracts["recommendation"], {
        "init_adapter_model_sha256": stage_audits["warmup"]["adapter_model_sha256"],
        "semantic_token_registration": contracts["warmup"]["semantic_token_registration"],
    }, "handoff")
    require_fields(contracts["recommendation"]["experiment_metadata"]["handoff"], {
        "adapter_model_sha256": stage_audits["warmup"]["adapter_model_sha256"],
        "train_contract_sha256": sha256_file(run_dir / "warmup" / "train_contract.json"),
        "optimizer_reset": True, "scheduler_reset": True,
    }, "handoff")
    if stage_audits["warmup"]["adapter_model_sha256"] == stage_audits["recommendation"]["adapter_model_sha256"]:
        raise ValueError("recommendation weights did not change")
    dataset = BASE / "eval" / f"dev_2k_{view}.parquet"
    rows = pq.read_table(dataset).to_pylist()
    if len(rows) != 2000:
        raise ValueError("expected frozen Dev-2K")
    rows = rows[:4] if smoke else rows
    class_count = len({row["target_interest"] for row in rows})
    results = {}
    for evaluation, beams, decoding in (
        ("classification", 10, "candidate_constrained_beam_search"),
        ("ranking", 20, "candidate_free_direct_beam_search"),
    ):
        directory = run_dir / evaluation
        contract = json.loads((directory / "run_contract.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        require_fields(contract, {
            "model_key": model_key, "model": MODELS[model_key], "view": view,
            "label_mode": "interest_id", "label_map_sha256": manifest["mapping_sha256"],
            "dataset_sha256": sha256_file(dataset), "sample_limit": 4 if smoke else None,
            "num_beams": beams, "decoding": decoding, "do_sample": False,
            "semantic_token_registration": contracts["recommendation"]["semantic_token_registration"],
        }, evaluation)
        if evaluation == "ranking":
            require_fields(contract, {"decoder_label_constraint": False, "candidates_in_prompt": False,
                                     "label_file_sha256": manifest["train_sha256"]}, evaluation)
        require_fields(metrics, {
            "complete": True, "requested_sample_count": len(rows), "recorded_sample_count": len(rows),
            "missing_sample_count": 0, "class_count": class_count, "model_key": model_key,
            "input_type": view, "label_mode": "interest_id", "decoding": decoding,
        }, evaluation)
        details = latest_details(directory / "details.jsonl", contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows}:
            raise ValueError(f"{evaluation}: raw sample IDs differ from frozen evaluation")
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            require_fields(detail, {"target_interest": row["target_interest"], "source_sample_id": row["source_sample_id"]}, evaluation)
            if evaluation == "ranking":
                detail.update(score_ranking(detail["ranking"], row["target_interest"]))
            else:
                detail["correct"] = detail["prediction"] == row["target_interest"]
            ordered.append(detail)
        recalculated = (metrics_from_details(ordered, rows) if evaluation == "classification"
                        else metrics_from_rankings(ordered, rows, beams))
        for key, value in recalculated.items():
            if metrics.get(key) != value:
                raise ValueError(f"{evaluation}: recomputed metric differs: {key}")
        if evaluation == "classification":
            require_fields(metrics, {"successful_sample_count": len(rows), "request_error_count": 0}, evaluation)
        results[evaluation] = {key: value for key, value in recalculated.items() if not key.startswith("per_class")}
    return {"model_key": model_key, "view": view, "smoke": smoke, "passed": True,
            "stage_audits": stage_audits, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--view", choices=VIEWS, default="interest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir, args.model_key, args.smoke, args.view)
    (args.run_dir / "RESULT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()