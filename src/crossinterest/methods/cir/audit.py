"""Collect CIR checkpoints and independently verify smoke or full results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.cir.runner import ROOT, preflight, audit_training
from crossinterest.methods.mechanism.audit import wanted
from crossinterest.methods.mechanism.runner import audit_evaluation


JOBS = {"qwen-balanced-cir": ("qwen35_9b", "external_balanced"),
        "llama-balanced-cir": ("llama31_8b", "external_balanced"),
    "llama-original-cir": ("llama31_8b", "original"),
    "qwen-semantic-true": ("qwen35_9b", "semantic_true"),
    "qwen-semantic-shuffled": ("qwen35_9b", "semantic_shuffled")}
JOBS.update({"qwen-cir-3ep": ("qwen35_9b", "external_balanced_ep3"),
                 **{f"qwen-bidir-{arm}": ("qwen35_9b", f"bidirectional_{arm}")
                     for arm in ("forward", "ordinary", "paired", "shuffled")}})


def select_inventory(inventory: list[dict], smoke_only: bool) -> list[dict]:
    projections = {f"{phase}/{stage}/final_adapter/cir_semantic_projection.safetensors"
                   for phase in ("smoke", "formal") for stage in ("auxiliary", "novel")}
    entries = [entry for entry in inventory if (wanted(entry["name"]) or entry["name"] in projections
               or entry["name"] == "formal/INVERSE_PROBE.json") and (
        not smoke_only or entry["name"] == "EXECUTION_CONTRACT.json" or entry["name"].startswith("smoke/"))]
    required = {"EXECUTION_CONTRACT.json", "smoke/RESULT_AUDIT.json"}
    if not smoke_only:
        required.update({"COMPLETE.json", "formal/RESULT_AUDIT.json"})
    if not required <= {entry["name"] for entry in entries}:
        raise ValueError("requested CIR phase has not completed")
    return entries


def audit_download(directory: Path, model_key: str, variant: str, smoke_only: bool) -> dict:
    stage_auditor = audit_training
    auxiliary_epochs = 3 if variant == "external_balanced_ep3" else 1
    if auxiliary_epochs == 3:
        variant = "external_balanced"
    if variant.startswith("semantic_"):
        from crossinterest.methods.cir_semantic.runner import preflight as semantic_preflight, audit_semantic_stage
        prepared = semantic_preflight(variant)
        stage_auditor = audit_semantic_stage
    else:
        prepared = preflight(model_key, variant, auxiliary_epochs)
    execution = json.loads((directory / "EXECUTION_CONTRACT.json").read_text())
    require_fields(execution, prepared, "CIR cloud execution")
    phases = {}
    for smoke in ((True,) if smoke_only else (True, False)):
        phase = "smoke" if smoke else "formal"
        source = directory / phase
        with TemporaryDirectory() as temporary:
            isolated = Path(temporary)
            stages = {}
            for stage in ("auxiliary", "novel"):
                stage_dir = isolated / stage
                stage_dir.mkdir()
                for path in (source / stage).iterdir():
                    if path.name != "STAGE_AUDIT.json":
                        (stage_dir / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
                stages[stage] = stage_auditor(stage_dir, stage, smoke, prepared, execution["model_path"])
                if stages[stage] != json.loads((source / stage / "STAGE_AUDIT.json").read_text()):
                    raise ValueError("cloud and independent CIR training audits disagree")
            for evaluation in ("classification", "ranking"):
                (isolated / evaluation).symlink_to((source / evaluation).resolve(), target_is_directory=True)
                contract = json.loads((source / evaluation / "run_contract.json").read_text())
                require_fields(contract, {"model_key": model_key, "model": execution["model_path"],
                                         "semantic_token_registration": prepared["token_registration"]}, "evaluated CIR model")
            audit = audit_evaluation(isolated, "reconstruction", smoke, stages["novel"])
            audit.update(model_key=model_key, variant=variant, stage_audits=stages)
            if audit != json.loads((source / "RESULT_AUDIT.json").read_text()):
                raise ValueError("cloud and independent CIR raw audits disagree")
            phases[phase] = audit
    if not smoke_only:
        completion = json.loads((directory / "COMPLETE.json").read_text())
        require_fields(completion, {"passed": True, "model_key": model_key, "variant": variant}, "CIR completion")
        for phase in ("smoke", "formal"):
            require_fields(completion, {f"{phase}_audit_sha256": sha256_file(directory / phase / "RESULT_AUDIT.json")}, "CIR audit receipt")
    return phases


def collect(experiment: str, output: Path, amlt: str, jobs: list[str], smoke_only: bool = False) -> dict:
    from azure.core import MatchConditions
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobClient

    output.mkdir(parents=True, exist_ok=True)
    credential = AzureCliCredential()
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    audits = {}
    for job in jobs:
        environment = dict(os.environ, AMLT_NO_BATCH_STATUS_API="1")
        response = subprocess.run([amlt, "--json-tables", "results", "list", experiment, f":{job}", "-u"],
                                  env=environment, capture_output=True, text=True, check=True, timeout=180)
        inventory = json.JSONDecoder().raw_decode(response.stdout.lstrip())[0]
        entries = select_inventory(inventory, smoke_only)
        directory = output / job
        for entry in entries:
            key = f"{job}/{entry['name']}"
            destination = directory / entry["name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob = BlobClient.from_blob_url(entry["url"], credential=credential, connection_timeout=15,
                                            read_timeout=60, retry_total=3)
            properties = blob.get_blob_properties()
            prior = downloads.get(key, {})
            if (destination.exists() and prior.get("etag") == properties.etag and
                    destination.stat().st_size == properties.size and sha256_file(destination) == prior.get("sha256")):
                continue
            temporary = destination.with_name(destination.name + ".download")
            with temporary.open("wb") as handle:
                blob.download_blob(etag=properties.etag, match_condition=MatchConditions.IfNotModified,
                                   max_concurrency=4).readinto(handle)
            if temporary.stat().st_size != properties.size:
                raise ValueError("CIR artifact download size mismatch")
            temporary.replace(destination)
            downloads[key] = {"etag": properties.etag, "bytes": properties.size, "sha256": sha256_file(destination)}
            write_json_atomic(receipt_path, downloads)
        model_key, variant = JOBS[job]
        audits[job] = audit_download(directory, model_key, variant, smoke_only)
        print(f"VERIFIED {job}: {'smoke only' if smoke_only else 'full training and original Dev2000'}", flush=True)
    result = {"passed": True, "experiment": experiment, "smoke_only": smoke_only,
              "observed_utc": datetime.now(timezone.utc).isoformat(), "audits": audits}
    write_json_atomic(output / ("GPU_SMOKE_AUDIT.json" if smoke_only else "RAW_RESULT_AUDIT.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="home112-cir-base-20260916")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/results_base")
    parser.add_argument("--amlt", default="amlt")
    parser.add_argument("--job", action="append", choices=JOBS)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    jobs = args.job or ["qwen-balanced-cir", "llama-balanced-cir", "llama-original-cir"]
    collect(args.experiment, args.output_dir.resolve(), args.amlt, jobs, args.smoke_only)


if __name__ == "__main__":
    main()