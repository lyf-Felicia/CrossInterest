"""Local-only monitoring and independent collection for the residual N experiment."""

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.residual.runner import ROOT, PREFLIGHT, METHOD, SOURCE_SHA, audit_stage, verify
from crossinterest.methods.consistency.monitor import checkpoint_timing
from crossinterest.methods.mechanism.audit import wanted, download_artifact
from crossinterest.methods.mechanism.runner import audit_evaluation
import crossinterest.methods.runtime.performance as watch


def output_health(audit):
    classification = audit["results"]["classification"]
    ranking = audit["results"]["ranking"]
    checks = {
        "classification_complete": classification["complete"] and classification["recorded_sample_count"] == 2000,
        "ranking_complete": ranking["complete"] and ranking["recorded_sample_count"] == 2000,
        "all_classes": classification["class_count"] == ranking["class_count"] == 112,
        "no_missing": classification["missing_sample_count"] == ranking["missing_sample_count"] == 0,
        "no_request_errors": classification["request_error_count"] == 0,
        "classification_format": classification["strict_format_rate"] == 1,
        "ranking_format": ranking["strict_beam_format_rate"] == ranking["valid_beam_rate"] == 1,
        "no_duplicates": ranking["duplicate_valid_beam_rate"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def collect(container, submitted):
    prepared = verify()
    if sha256_file(PREFLIGHT) != submitted["preflight_sha256"]:
        raise ValueError("submission preflight differs")
    output = ROOT / "outputs/results"
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    prefix = submitted["results_blob_prefix"] + "/"
    entries = {blob.name[len(prefix):]: blob for blob in container.list_blobs(name_starts_with=prefix)
               if blob.size > 0 and wanted(blob.name[len(prefix):])}
    if not {"COMPLETE.json", "EXECUTION_CONTRACT.json", "formal/RESULT_AUDIT.json", "smoke/RESULT_AUDIT.json"} <= entries.keys():
        raise ValueError("residual results are incomplete")
    for name, blob in entries.items():
        downloads[name] = download_artifact(container.get_blob_client(blob.name), output / name, downloads.get(name, {}))
        write_json_atomic(receipt_path, downloads)
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    if any(execution.get(key) != value for key, value in prepared.items()):
        raise ValueError("residual execution contract differs")
    audits = {}
    for phase in ("smoke", "formal"):
        source = output / phase
        with TemporaryDirectory(prefix="residual-audit-") as temporary:
            isolated = Path(temporary)
            stage = isolated / "novel"
            stage.mkdir()
            for path in (source / "novel").iterdir():
                if path.name != "STAGE_AUDIT.json":
                    (stage / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
            training = audit_stage(stage, phase == "smoke", prepared, execution["model_path"])
            if training != json.loads((source / "novel/STAGE_AUDIT.json").read_text()):
                raise ValueError("cloud and local N audits differ")
            for evaluation in ("classification", "ranking"):
                (isolated / evaluation).symlink_to((source / evaluation).resolve(), target_is_directory=True)
                sidecar = json.loads((source / evaluation / "RESIDUAL_CONTRACT.json").read_text())
                expected = {"method": METHOD, "source_adapter_sha256": SOURCE_SHA,
                    "n_adapter_sha256": training["adapter_sha256"], "preflight_sha256": sha256_file(PREFLIGHT),
                    "evaluation": evaluation}
                if sidecar != expected:
                    raise ValueError("evaluated residual checkpoint differs")
            audit = audit_evaluation(isolated, "reconstruction", phase == "smoke", training)
            audit.update(method=METHOD, preflight_sha256=sha256_file(PREFLIGHT))
            if audit != json.loads((source / "RESULT_AUDIT.json").read_text()):
                raise ValueError("independent raw metrics differ from cloud")
            audits[phase] = audit
    completion = json.loads((output / "COMPLETE.json").read_text())
    if not completion["passed"] or completion["method"] != METHOD:
        raise ValueError("completion identity differs")
    for phase in audits:
        if completion[f"{phase}_audit_sha256"] != sha256_file(output / phase / "RESULT_AUDIT.json"):
            raise ValueError("completion references different audit bytes")
    ranking = audits["formal"]["results"]["ranking"]
    reference = prepared["reference_metrics"]
    gate = (ranking["macro_ndcg_at_20"] > reference["macro_ndcg_at_20"]
            and ranking["macro_recall_at_20"] >= reference["macro_recall_at_20"])
    if gate != completion["improvement_gate_passed"]:
        raise ValueError("cloud and local improvement decisions differ")
    completed = datetime.now(timezone.utc)
    clock = json.loads((ROOT / "BUDGET_CLOCK.json").read_text())
    health = output_health(audits["formal"])
    result = {"passed": True, "method": METHOD, "job_id": submitted["job_id"], "audits": audits,
        "observed_utc": completed.isoformat(), "submission_receipt_sha256": sha256_file(ROOT / "SUBMISSION_RECEIPT.json"),
        "source_adapter_sha256": SOURCE_SHA, "preflight_sha256": sha256_file(PREFLIGHT),
        "reference_metrics": reference, "improvement_gate_passed": gate,
        "formal_output_health": health, "improvement_supported": gate and health["passed"],
        "macro_deltas": {key: ranking[key] - value for key, value in reference.items()},
        "eight_hour_target_met": completed <= datetime.fromisoformat(clock["target_deadline_utc"]),
        "end_to_end_seconds": (completed - datetime.fromisoformat(clock["submission_started_utc"])).total_seconds(),
        "downloads": downloads}
    write_json_atomic(output / "RAW_RESULT_AUDIT.json", result)
    print(json.dumps({"event": "residual_independent_final_audit_passed", "ranking": ranking,
                      "improvement_gate_passed": gate, "formal_output_health_passed": health["passed"],
                      "eight_hour_target_met": result["eight_hour_target_met"]}), flush=True)
    return result


def main():
    from azure.identity import AzureCliCredential
    from azure.storage.blob import ContainerClient
    from azure.core.exceptions import ResourceNotFoundError

    submitted = json.loads((ROOT / "SUBMISSION_RECEIPT.json").read_text())
    clock = json.loads((ROOT / "BUDGET_CLOCK.json").read_text())
    container = ContainerClient(submitted["account_url"], "amulet", credential=AzureCliCredential(process_timeout=30),
                                connection_timeout=15, read_timeout=60, retry_total=3)
    original = watch.observe
    checked = set()
    state_path = ROOT / "HEALTH.json"

    def observe(client, session, job_id, previous, now):
        if job_id not in checked:
            platform = client.jobs.get(job_id)._to_dict()
            if (platform["inputs"]["amlt_code"] != submitted["code"]
                    or platform["outputs"]["amlt_results"]["path"] != submitted["output_path"]
                    or platform["inputs"]["frozen"] != submitted["frozen"]):
                raise ValueError("monitored job identity differs")
            verify()
            checked.add(job_id)
        current = original(client, session, job_id, previous, now)
        if current["log_file"]:
            details = client.jobs._runs_operations.get_run_details(job_id)
            response = session.get(details.log_files[current["log_file"]], headers={"Range": "bytes=-262144"}, timeout=40)
            response.raise_for_status()
            old_timing = (previous or {}).get("checkpoint_timing") if (previous or {}).get("log_file") == current["log_file"] else None
            current["checkpoint_timing"] = checkpoint_timing(old_timing, response.text)
            if current["checkpoint_timing"]["repeated_slow_saves"]:
                write_json_atomic(state_path, {job_id: current})
                print(json.dumps({"event": "residual_repeated_slow_checkpoint_saves", "job_id": job_id}), flush=True)
                raise SystemExit(13)
        if not current["smoke_audited"]:
            try:
                smoke = json.loads(container.get_blob_client(submitted["results_blob_prefix"] + "/smoke/RESULT_AUDIT.json").download_blob().readall())
                current["smoke_audited"] = (smoke["passed"] and smoke["method"] == METHOD)
            except ResourceNotFoundError:
                pass
        if current["status"] == "Completed":
            result = collect(container, submitted)
            current["independent_final_audit_passed"] = result["passed"]
            current["formal_output_health_passed"] = result["formal_output_health"]["passed"]
            if not current["formal_output_health_passed"]:
                write_json_atomic(state_path, {job_id: current})
                print(json.dumps({"event": "residual_formal_output_health_failed", "job_id": job_id}), flush=True)
                raise SystemExit(13)
        if now > datetime.fromisoformat(clock["target_deadline_utc"]) and current["status"] != "Completed":
            write_json_atomic(state_path, {job_id: current})
            print(json.dumps({"event": "residual_eight_hour_target_exceeded", "job_id": job_id}), flush=True)
            raise SystemExit(13)
        return current

    with (ROOT / "WATCH.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch.observe = observe
        sys.argv = [sys.argv[0], "--subscription", submitted["subscription"], "--resource-group", submitted["resource_group"],
            "--workspace", submitted["workspace"], "--state-file", str(state_path), "--interval", "60",
            "--stale-minutes", "25", "--job", submitted["job_id"]]
        return watch.main()


if __name__ == "__main__":
    raise SystemExit(main())