"""Independently verify the recovered N-only frequency comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.frequency.runner import ROOT, PREFLIGHT, preflight, execution_contract, audit_training
from crossinterest.methods.mechanism.audit import download_artifact, wanted
from crossinterest.methods.mechanism.runner import audit_evaluation


JOBS = {"qwen-paired-once": "once", "qwen-paired-five": "at_least_five"}


def select_inventory(inventory: list[dict], smoke_only: bool) -> list[dict]:
    phases = {"smoke": 7} if smoke_only else {"smoke": 7, "formal": 3500}
    required = {"EXECUTION_CONTRACT.json", "AUDIT_RECOVERY_CONTRACT.json"}
    for phase, steps in phases.items():
        required.update({f"{phase}/RESULT_AUDIT.json", f"{phase}/novel/STAGE_AUDIT.json",
                         f"{phase}/novel/checkpoint-{steps}/trainer_state.json",
                         f"{phase}/novel/checkpoint-{steps}/adapter_model.safetensors"})
    if not smoke_only:
        required.add("COMPLETE.json")
    entries = [entry for entry in inventory if (wanted(entry["name"]) or entry["name"] in required)
               and (not smoke_only or entry["name"] in required or entry["name"].startswith("smoke/"))]
    if not required <= {entry["name"] for entry in entries}:
        raise ValueError("frequency phase or final checkpoint evidence is incomplete")
    return entries


def verify_preserved_smoke(directory: Path, arm: str, preservation: dict) -> None:
    previous = preservation["conditions"][arm]
    expected = {"EXECUTION_CONTRACT.json": previous["execution_contract_sha256"],
                "smoke/novel/train_metrics.json": previous["original_metrics_sha256"],
                "smoke/novel/final_adapter/adapter_model.safetensors": previous["final_adapter_sha256"]}
    for relative, digest in expected.items():
        if sha256_file(directory / relative) != digest:
            raise ValueError(f"original smoke evidence changed: {relative}")


def audit_download(directory: Path, arm: str, smoke_only: bool, preservation: dict) -> dict:
    prepared = preflight()
    verify_preserved_smoke(directory, arm, preservation)
    execution = json.loads((directory / "EXECUTION_CONTRACT.json").read_text())
    with TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        for name in ("EXECUTION_CONTRACT.json", "AUDIT_RECOVERY_CONTRACT.json"):
            shutil.copyfile(directory / name, isolated / name)
        execution_contract(isolated, prepared | {"condition": arm, "model_path": execution["model_path"]})
    phases = {}
    for smoke in ((True,) if smoke_only else (True, False)):
        phase = "smoke" if smoke else "formal"
        source = directory / phase
        with TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            stage = isolated / "novel"
            stage.mkdir()
            for path in (source / "novel").iterdir():
                if path.name != "STAGE_AUDIT.json":
                    (stage / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
            training = audit_training(stage, arm, smoke, prepared, execution["model_path"])
            if training != json.loads((source / "novel/STAGE_AUDIT.json").read_text()):
                raise ValueError("cloud and independent frequency training audits disagree")
            for evaluation in ("classification", "ranking"):
                (isolated / evaluation).symlink_to((source / evaluation).resolve(), target_is_directory=True)
                contract = json.loads((source / evaluation / "run_contract.json").read_text())
                require_fields(contract, {"model_key": "qwen35_9b", "model": execution["model_path"],
                                          "semantic_token_registration": prepared["token_registration"]},
                               "evaluated frequency model")
            audit = audit_evaluation(isolated, "reconstruction", smoke, training)
            audit.update(arm=arm, protocol=prepared["protocol"], training_stages=["novel"])
            if audit != json.loads((source / "RESULT_AUDIT.json").read_text()):
                raise ValueError("cloud and independent frequency raw audits disagree")
            phases[phase] = audit
    if not smoke_only:
        completion = json.loads((directory / "COMPLETE.json").read_text())
        require_fields(completion, {"passed": True, "condition": arm}, "frequency completion")
        for phase in phases:
            require_fields(completion, {f"{phase}_audit_sha256": sha256_file(directory / phase / "RESULT_AUDIT.json")},
                           "frequency audit receipt")
    return phases


def collect(experiment: str, output: Path, amlt: str, smoke_only: bool, preservation_path: Path) -> dict:
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobClient

    preflight()
    preservation = json.loads(preservation_path.read_text())
    require_fields(preservation, {"passed": True}, "original smoke preservation audit")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    credential = AzureCliCredential(process_timeout=30)
    audits = {}
    for job, arm in JOBS.items():
        response = subprocess.run([amlt, "--json-tables", "results", "list", experiment, f":{job}", "-u"],
                                  env=dict(os.environ, AMLT_NO_BATCH_STATUS_API="1"),
                                  capture_output=True, text=True, check=True, timeout=180)
        inventory = json.JSONDecoder().raw_decode(response.stdout.lstrip())[0]
        directory = output / arm
        for entry in select_inventory(inventory, smoke_only):
            key = f"{arm}/{entry['name']}"
            blob = BlobClient.from_blob_url(entry["url"], credential=credential, connection_timeout=15,
                                            read_timeout=60, retry_total=3)
            downloads[key] = download_artifact(blob, directory / entry["name"], downloads.get(key, {}))
            write_json_atomic(receipt_path, downloads)
        audits[arm] = audit_download(directory, arm, smoke_only, preservation)
        print(f"VERIFIED {arm}: {'smoke only' if smoke_only else 'full N-only training and original Dev2000'}", flush=True)
    result = {"passed": True, "experiment": experiment, "smoke_only": smoke_only,
              "observed_at": datetime.now(timezone.utc).isoformat(), "preflight_sha256": sha256_file(PREFLIGHT),
              "preservation_audit_sha256": sha256_file(preservation_path), "original_smoke_preserved": True,
              "collector_sha256": sha256_file(Path(__file__)), "audits": audits}
    write_json_atomic(output / ("GPU_SMOKE_AUDIT.json" if smoke_only else "RAW_RESULT_AUDIT.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="home112-frequency-paired-20260917")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/results_v3")
    parser.add_argument("--amlt", default="amlt")
    parser.add_argument("--preservation-audit", type=Path, default=ROOT / "outputs/RECOVERY_SMOKE_AUDIT_20260917.json")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    collect(args.experiment, args.output_dir.resolve(), args.amlt, args.smoke_only, args.preservation_audit)


if __name__ == "__main__":
    main()