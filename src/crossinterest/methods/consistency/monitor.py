"""Monitor the two authorized dual jobs and independently audit completed results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.consistency.runner import ROOT, PREFLIGHT, audit_stage, code_identity
from crossinterest.methods.mechanism.audit import download_artifact, wanted
from crossinterest.methods.mechanism.runner import audit_evaluation
import crossinterest.methods.runtime.performance as watch


def attempt_previous(previous, log_file, text):
    if not previous or previous.get("log_file") != log_file:
        return {}
    old_paths = {record.get("path") for record in previous.get("io", {}).get("restored_ranks", {}).values()}
    old_paths.discard(None)
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{"event":\s*"dual_complete_checkpoint_selected"', text):
        try:
            record, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if old_paths and record.get("path") and record["path"] not in old_paths:
            return {}
    return previous


def checkpoint_timing(previous, text):
    elapsed = dict((previous or {}).get("elapsed", {}))
    for match in re.finditer(r"(\d+)/3500\s*\[([\d:]+)<", text):
        seconds = 0
        for part in match[2].split(":"):
            seconds = 60 * seconds + int(part)
        elapsed[match[1]] = seconds
    latest = max(map(int, elapsed), default=0)
    elapsed = {step: seconds for step, seconds in elapsed.items() if int(step) >= latest - 400}
    boundaries = [{"step": int(step), "seconds": elapsed[str(int(step) + 1)] - seconds}
                  for step, seconds in elapsed.items()
                  if int(step) > 0 and int(step) % 100 == 0 and str(int(step) + 1) in elapsed]
    boundaries = sorted((item for item in boundaries if item["seconds"] >= 0), key=lambda item: item["step"])[-3:]
    window = sorted(int(step) for step in elapsed if latest - 200 <= int(step) <= latest and int(step) > 0)
    rate = None
    if window and latest - window[0] >= 25:
        rate = (elapsed[str(latest)] - elapsed[str(window[0])]) / (latest - window[0])
        if rate <= 0:
            rate = None
    return {"elapsed": elapsed, "latest_step": latest, "last_save_boundaries": boundaries,
            "repeated_slow_saves": sum(item["seconds"] >= 120 for item in boundaries) >= 2,
            "recent_seconds_per_step_including_saves": rate}


def io_observation(previous, text):
    result = dict(previous or {})
    enabled = set(result.get("enabled_ranks", []))
    restored = dict(result.get("restored_ranks", {}))
    saves = dict(result.get("saves", {}))
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{"event":\s*"(?:dual_local_checkpoint_serialization_enabled|dual_complete_checkpoint_selected|local_checkpoint_archive_saved)"', text):
        try:
            record, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        rank = int(record["rank"])
        if record["event"] == "dual_local_checkpoint_serialization_enabled":
            enabled.add(rank)
        elif record["event"] == "dual_complete_checkpoint_selected":
            if (not record["passed"] or record["world_size"] != 8
                    or record["optimizer_steps"] != [record["step"]]
                    or record["scheduler_step"] != record["step"]):
                raise ValueError("runtime resume state differs")
            restored[str(rank)] = record
        else:
            checkpoint = saves.setdefault(record["checkpoint"], {})
            checkpoint[f'{rank}/{record["file"]}'] = record
    progress = re.findall(r"(\d+)/3500\s*\[", text)
    result.update(enabled_ranks=sorted(enabled), restored_ranks=restored, saves=saves,
                  step=int(progress[-1]) if progress else result.get("step", 0))
    result["verified"] = False
    if enabled != set(range(8)) or set(restored) != {str(rank) for rank in range(8)}:
        return result
    steps = {record["step"] for record in restored.values()}
    sources = {record["source"] for record in restored.values()}
    if len(steps) != 1 or len(sources) != 1:
        raise ValueError("ranks restored different checkpoints")
    required = {"0/optimizer.pt", "0/scheduler.pt", *[f"{rank}/rng_state_{rank}.pth" for rank in range(8)]}
    for name, records in sorted(saves.items(), key=lambda item: int(item[0].split("-")[1])):
        step = int(name.split("-")[1])
        if required <= set(records) and step > next(iter(steps)) and result["step"] > step:
            optimizer = records["0/optimizer.pt"]
            seconds = optimizer["serialize_seconds"] + optimizer["copy_seconds"]
            result.update(verified=True, checkpoint=name, optimizer_save_seconds=seconds,
                          optimizer_save_under_120_seconds=seconds < 120)
            break
    return result


def collect_arm(container, row, recovery, recovery_manifest=None):
    prepared = json.loads(PREFLIGHT.read_text())
    if prepared["code_sha256"] != code_identity():
        raise ValueError("scientific source identity changed")
    arm = row["arm"]
    output = ROOT / "outputs/results" / arm
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    prefix = row["results_blob_prefix"] + "/"
    entries = {blob.name[len(prefix):]: blob for blob in container.list_blobs(name_starts_with=prefix)
               if wanted(blob.name[len(prefix):]) and blob.size > 0}
    if not {"COMPLETE.json", "formal/RESULT_AUDIT.json", "smoke/RESULT_AUDIT.json"} <= entries.keys():
        raise ValueError("full dual results are not complete")
    for name, blob in entries.items():
        downloads[name] = download_artifact(container.get_blob_client(blob.name), output / name,
                                             downloads.get(name, {}))
        write_json_atomic(receipt_path, downloads)
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    if execution.get("dual_arm") != arm or any(execution.get(key) != value for key, value in prepared.items()):
        raise ValueError("dual execution differs from frozen preflight")
    audits = {}
    for phase in ("smoke", "formal"):
        source = output / phase
        with TemporaryDirectory(prefix="dual-audit-") as temporary:
            isolated = Path(temporary)
            stages = {}
            for stage in ("auxiliary", "novel"):
                destination = isolated / stage
                destination.mkdir()
                for path in (source / stage).iterdir():
                    if path.name != "STAGE_AUDIT.json":
                        (destination / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
                stages[stage] = audit_stage(destination, stage, phase == "smoke", prepared, arm, execution["model_path"])
                if stages[stage] != json.loads((source / stage / "STAGE_AUDIT.json").read_text()):
                    raise ValueError("cloud and local dual training audits differ")
            for evaluation in ("classification", "ranking"):
                (isolated / evaluation).symlink_to((source / evaluation).resolve(), target_is_directory=True)
            audit = audit_evaluation(isolated, "reconstruction", phase == "smoke", stages["novel"])
            audit.update(dual_arm=arm, stage_audits=stages, preflight_sha256=sha256_file(PREFLIGHT))
            if audit != json.loads((source / "RESULT_AUDIT.json").read_text()):
                raise ValueError("cloud and independently recomputed raw results differ")
            audits[phase] = audit
    completion = json.loads((output / "COMPLETE.json").read_text())
    if not completion["passed"] or completion["dual_arm"] != arm:
        raise ValueError("invalid completion identity")
    for phase in audits:
        if completion[f"{phase}_audit_sha256"] != sha256_file(output / phase / "RESULT_AUDIT.json"):
            raise ValueError("completion references different audit bytes")
    preservation = None
    if arm == "consistent":
        backup = json.loads((ROOT / "outputs/io_recovery/PRESTOP_BACKUP_AUDIT.json").read_text())
        for name, record in backup["preserved"].items():
            if sha256_file(output / name) != record["sha256"]:
                raise ValueError("completed pre-recovery work changed")
        preservation = {"passed": True, "files": list(backup["preserved"])}
    result = {"passed": True, "arm": arm, "job_id": row["job_id"], "audits": audits,
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "original_created_utc": recovery["original_created_utc"], "deadline_utc": recovery["deadline_utc"],
              "io_recovery_manifest_sha256": sha256_file(ROOT / "IO_RECOVERY_MANIFEST.json"),
              "active_recovery_manifest_sha256": sha256_file(recovery_manifest or ROOT / "IO_RECOVERY_MANIFEST.json"),
              "preservation": preservation, "downloads": downloads}
    write_json_atomic(output / "RAW_RESULT_AUDIT.json", result)
    print(json.dumps({"event": "dual_independent_final_audit_passed", "arm": arm,
                      "ranking": audits["formal"]["results"]["ranking"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-on-save", action="store_true")
    parser.add_argument("--recovery-manifest", type=Path, default=ROOT / "IO_RECOVERY_MANIFEST.json")
    options = parser.parse_args()
    from azure.identity import AzureCliCredential
    from azure.storage.blob import ContainerClient

    recovery = json.loads(options.recovery_manifest.read_text())
    submitted = json.loads((ROOT / "SUBMISSION_RECEIPT.json").read_text())
    rows = {row["job_id"]: row for row in submitted["jobs"] if row["arm"] == "ce"}
    rows[recovery["job_id"]] = {**recovery, "output_path": recovery["outputs"]["amlt_results"]["path"]}
    container = ContainerClient(recovery["account_url"], "amulet", credential=AzureCliCredential(process_timeout=30),
                                connection_timeout=15, read_timeout=60, retry_total=3)
    original = watch.observe
    verified_jobs = set()
    state_path = ROOT / "IO_HEALTH.json"

    def observe(client, session, job_id, previous, now):
        row = rows[job_id]
        if job_id not in verified_jobs:
            platform = client.jobs.get(job_id)._to_dict()
            expected_code = row["code"]["path"] if isinstance(row["code"], dict) else row["code"]
            if (platform["inputs"]["amlt_code"]["path"] != expected_code
                    or platform["outputs"]["amlt_results"]["path"] != row["output_path"]):
                raise ValueError("monitored job code/output identity differs")
            verified_jobs.add(job_id)
        current = original(client, session, job_id, previous, now)
        if current["log_file"]:
            details = client.jobs._runs_operations.get_run_details(job_id)
            response = session.get(details.log_files[current["log_file"]], headers={"Range": "bytes=-262144"}, timeout=40)
            response.raise_for_status()
            runtime_previous = attempt_previous(previous, current["log_file"], response.text)
            current["checkpoint_timing"] = checkpoint_timing(runtime_previous.get("checkpoint_timing"), response.text)
            current["io_alert_save_step"] = runtime_previous.get("io_alert_save_step")
            timing = current["checkpoint_timing"]
            if timing["repeated_slow_saves"] and current["io_alert_save_step"] != timing["last_save_boundaries"][-1]["step"]:
                current["io_alert_save_step"] = timing["last_save_boundaries"][-1]["step"]
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                state[job_id] = current
                write_json_atomic(state_path, state)
                print(json.dumps({"event": "repeated_slow_checkpoint_saves", "job_id": job_id,
                                  "last_save_boundaries": timing["last_save_boundaries"]}), flush=True)
                raise SystemExit(13)
        if job_id == recovery["job_id"] and current["log_file"]:
            current["io"] = io_observation(runtime_previous.get("io"), response.text)
            if current["io"]["verified"]:
                proof = ROOT / "outputs/io_recovery/RUNTIME_IO_VERIFICATION.json"
                if not proof.exists():
                    write_json_atomic(proof, {"job_id": job_id, "observed_utc": now.isoformat(), **current["io"]})
                if options.exit_on_save:
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    state[job_id] = current
                    write_json_atomic(state_path, state)
                    print(json.dumps({"event": "dual_first_checkpoint_ready_for_inspection", **current["io"]}), flush=True)
                    raise SystemExit(0 if current["io"]["optimizer_save_under_120_seconds"] else 13)
        if current["status"] == "Completed":
            result = collect_arm(container, row, recovery, options.recovery_manifest)
            current["independent_final_audit_passed"] = result["passed"]
        if now > datetime.fromisoformat(recovery["deadline_utc"]) and current["status"] != "Completed":
            print(json.dumps({"event": "original_eight_hour_deadline_exceeded", "job_id": job_id}), flush=True)
            raise SystemExit(13)
        return current

    lock_path = ROOT.parent / "cir_factorization/outputs/MECHANISM_FOLLOWUP_WATCH.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch.observe = observe
        sys.argv = [sys.argv[0], "--subscription", recovery["subscription"], "--resource-group", recovery["resource_group"],
                    "--workspace", recovery["workspace"], "--state-file", str(state_path), "--interval", "60",
                    "--stale-minutes", "20", *[argument for job_id in rows for argument in ("--job", job_id)]]
        return watch.main()


if __name__ == "__main__":
    raise SystemExit(main())