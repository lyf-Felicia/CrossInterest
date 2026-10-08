"""Collect composition artifacts and independently recompute raw ranking metrics."""

from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

import pyarrow.parquet as pq

from crossinterest.evaluation.common import latest_details, write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.composition.runner import ROOT, PREFLIGHT, METHOD, ARMS, DEV, MAPPING, verify, audit_training, audit_evaluation
from crossinterest.methods.mechanism.audit import download_artifact
import crossinterest.methods.runtime.performance as watch


def classify_observation(current, previous, now, stale_seconds):
    event = watch_classify(current, previous, now, stale_seconds)
    if event == "no_progress" and current["status"] == "Queued" and not current.get("log_file"):
        return "queued_no_progress"
    return event


watch_classify = watch.classify_observation


def verify_compatible_protocol(prepared, previous):
    operational = {"implementation_revision", "previous_preflight_sha256", "revision_reason",
                   "code_sha256", "training_layout"}
    if {key: value for key, value in prepared.items() if key not in operational} != {
            key: value for key, value in previous.items() if key not in operational}:
        raise ValueError("Versioned arm scientific protocol differs")
    for contract in (prepared, previous):
        layout = contract["training_layout"]
        if (layout["world_size"] != 8 or not layout["gradient_checkpointing"]
                or layout["world_size"] * layout["micro_batch"] * layout["accumulation"] != 32):
            raise ValueError("Versioned arm effective batch or precision layout differs")


def independent_metrics(directory, prepared):
    mapping = json.loads(MAPPING.read_text())["id_to_category"]
    expected = {row["sample_id"]: row for row in pq.read_table(DEV).to_pylist()}
    by_class, identities = defaultdict(list), set()
    for shard in prepared["shards"]:
        output = directory / "ranking" / f"shard-{shard}"
        contract = json.loads((output / "run_contract.json").read_text())
        for detail in latest_details(output / "details.jsonl", contract["fingerprint"]).values():
            identity = detail["sample_id"]
            if identity in identities or identity not in expected:
                raise ValueError("Duplicate or unexpected sample")
            identities.add(identity)
            target = expected[identity]["target_interest"]
            rank, seen = None, set()
            for position, entry in enumerate(detail["ranking"], 1):
                predicted = mapping.get(entry["predicted_output"])
                if entry["prediction"] != predicted:
                    raise ValueError("Saved prediction disagrees with ID mapping")
                strict = predicted is not None and entry["raw_text"] == entry["predicted_output"]
                if entry["format_valid"] != strict:
                    raise ValueError("Strict format flag disagrees with raw text")
                if predicted is not None:
                    if predicted in seen:
                        raise ValueError("Duplicate valid beam")
                    seen.add(predicted)
                if strict and predicted == target and rank is None:
                    rank = position
            by_class[target].append({f"macro_{metric}_at_{cutoff}":
                (1.0 / math.log2(rank + 1) if metric == "ndcg" else 1.0)
                if rank is not None and rank <= cutoff else 0.0
                for metric in ("recall", "ndcg") for cutoff in (10, 20)})
    if identities != set(expected) or len(by_class) != 112:
        raise ValueError("Independent evaluation coverage is incomplete")
    return {key: sum(sum(row[key] for row in rows) / len(rows) for rows in by_class.values()) / len(by_class)
            for key in next(iter(by_class.values()))[0]}


def collect(container, submitted, arm):
    prepared = verify()
    if submitted["preflight_sha256"] != sha256_file(PREFLIGHT):
        raise ValueError("Submission preflight changed")
    job = submitted["jobs"][arm]
    if job.get("audit_directory"):
        directory = (ROOT / job["audit_directory"]).resolve()
        if directory.parent != ROOT.resolve():
            raise ValueError("Versioned audit must use a local validated deployment")
        previous_path = directory / "eval_interest_id_minionerec/composition/PREFLIGHT.json"
        if sha256_file(previous_path) != job["preflight_sha256"]:
            raise ValueError("Versioned arm contract changed")
        verify_compatible_protocol(prepared, json.loads(previous_path.read_text()))
        previous_receipt = {**submitted, "preflight_sha256": job["preflight_sha256"]}
        worker = (
            "import sys,json; from pathlib import Path; "
            "sys.path[:0]=[sys.argv[1],sys.argv[1]+'/prep']; "
            "from azure.identity import AzureCliCredential; "
            "from azure.storage.blob import ContainerClient; "
            "from eval_interest_id_minionerec import composition_monitor as monitor; "
            "receipt=json.loads(sys.argv[4]); monitor.ROOT=Path(sys.argv[2]); "
            "container=ContainerClient(receipt['account_url'],receipt['container'],"
            "credential=AzureCliCredential(process_timeout=30),connection_timeout=15,read_timeout=60,retry_total=3); "
            "monitor.collect(container,receipt,sys.argv[3])")
        subprocess.run([sys.executable, "-I", "-c", worker, str(directory), str(ROOT), arm,
                        json.dumps(previous_receipt)], check=True)
        result = json.loads((ROOT / "outputs/results" / arm / "RAW_RESULT_AUDIT.json").read_text())
        if result["preflight_sha256"] != job["preflight_sha256"] or not result["passed"]:
            raise ValueError("Versioned arm independent audit failed")
        return result
    output = ROOT / "outputs/results" / arm
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    names = ["COMPLETE.json", "EXECUTION_CONTRACT.json", "formal/RESULT_AUDIT.json", "smoke/RESULT_AUDIT.json"]
    stages = ("novel",) if arm == "N" else ("auxiliary", "novel")
    names += [f"formal/{stage}/{name}" for stage in stages for name in (
        "train_metrics.json", "train_contract.json", "runtime_config.json", "COMPOSITION_RUNTIME.json",
        "trainable_parameters.json", "STAGE_AUDIT.json", "final_adapter/adapter_config.json",
        "final_adapter/adapter_model.safetensors", "final_adapter/interest_bridge.json",
        "final_adapter/interest_bridge.safetensors")]
    names += [f"formal/{stage}/PERFORMANCE_RANK_{rank}.json" for stage in stages
              for rank in range(prepared["training_layout"]["world_size"])]
    names += [f"formal/ranking/shard-{shard}/{name}" for shard in prepared["shards"] for name in (
        "run_contract.json", "details.jsonl", "composition_checks.jsonl", "metrics.json", "predictions.jsonl")]
    for name in names:
        blob = container.get_blob_client(job["results_blob_prefix"] + "/" + name)
        downloads[name] = download_artifact(blob, output / name, downloads.get(name, {}))
        write_json_atomic(receipt_path, downloads)
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    if execution["preflight"] != prepared or execution["arm"] != arm:
        raise ValueError("Cloud execution differs from frozen settings")
    training = {}
    for stage in stages:
        directory = output / "formal" / stage
        cloud = json.loads((directory / "STAGE_AUDIT.json").read_text())
        training[stage] = audit_training(directory, arm, stage, False, prepared, execution["model_path"])
        if training[stage] != cloud:
            raise ValueError("Local training audit differs from cloud")
    cloud = json.loads((output / "formal/RESULT_AUDIT.json").read_text())
    result = audit_evaluation(output / "formal", arm, False, prepared)
    if result != cloud:
        raise ValueError("Local result audit differs from cloud")
    completion = json.loads((output / "COMPLETE.json").read_text())
    if (completion["arm"] != arm or not completion["passed"]
            or completion["formal_audit_sha256"] != sha256_file(output / "formal/RESULT_AUDIT.json")):
        raise ValueError("Cloud completion identity differs")
    independent = independent_metrics(output / "formal", prepared)
    if not all(math.isclose(value, result["metrics"][key], abs_tol=1e-12, rel_tol=0)
               for key, value in independent.items()):
        raise ValueError("Independent raw metric calculation differs")
    if any(sha256_file(output / name) != record["sha256"] for name, record in downloads.items()):
        raise ValueError("Downloaded artifact bytes changed")
    result.update(training=training, independent_raw_metrics=independent, downloads=downloads,
                  observed_utc=datetime.now(timezone.utc).isoformat(), job_id=job["job_id"])
    write_json_atomic(output / "RAW_RESULT_AUDIT.json", result)
    print(json.dumps({"event": "composition_independent_arm_audit", "arm": arm,
                      "health_passed": result["health_passed"], "metrics": independent}), flush=True)
    return result


def summarize():
    prepared = verify()
    submitted = json.loads((ROOT / "SUBMISSION_RECEIPT.json").read_text())
    audits = {arm: json.loads((ROOT / "outputs/results" / arm / "RAW_RESULT_AUDIT.json").read_text()) for arm in ARMS}
    if any(not audit["passed"] or audit["preflight_sha256"] != submitted["jobs"][arm].get(
            "preflight_sha256", sha256_file(PREFLIGHT)) for arm, audit in audits.items()):
        raise ValueError("Incomplete independent arm audits")
    metrics = {arm: audit["independent_raw_metrics"] for arm, audit in audits.items()}
    reference = prepared["reference_metrics"]
    health = all(audit["health_passed"] for audit in audits.values())
    improvement = (metrics["C"]["macro_ndcg_at_20"] > max(reference["macro_ndcg_at_20"], metrics["R"]["macro_ndcg_at_20"])
                   and metrics["C"]["macro_recall_at_20"] >= reference["macro_recall_at_20"])
    clock = json.loads((ROOT / "BUDGET_CLOCK.json").read_text())
    completed = max(datetime.fromisoformat(audit["observed_utc"]) for audit in audits.values())
    result = {"passed": True, "method": METHOD, "health_passed": health,
        "improvement_gate_passed": health and improvement, "metrics": {"G0": reference, **metrics},
        "independent_audit_utc": completed.isoformat(), "preflight_sha256": sha256_file(PREFLIGHT),
        "end_to_end_seconds": (completed - datetime.fromisoformat(clock["submission_started_utc"])).total_seconds(),
        "eight_hour_target_met": completed <= datetime.fromisoformat(clock["target_deadline_utc"]),
        "arm_audit_sha256": {arm: sha256_file(ROOT / "outputs/results" / arm / "RAW_RESULT_AUDIT.json") for arm in ARMS}}
    write_json_atomic(ROOT / "outputs/RAW_RESULT_AUDIT.json", result)
    print(json.dumps({"event": "composition_final_audit", **result}), flush=True)
    return result


def main():
    from azure.identity import AzureCliCredential
    from azure.storage.blob import ContainerClient

    submitted = json.loads((ROOT / "SUBMISSION_RECEIPT.json").read_text())
    container = ContainerClient(submitted["account_url"], submitted["container"],
        credential=AzureCliCredential(process_timeout=30), connection_timeout=15, read_timeout=60, retry_total=3)
    if "--collect-once" in sys.argv:
        for arm in ARMS:
            collect(container, submitted, arm)
        summarize()
        return 0
    original, checked = watch.observe, set()
    jobs = {record["job_id"]: arm for arm, record in submitted["jobs"].items()}

    def observe(client, session, job_id, previous, now):
        arm = jobs[job_id]
        if job_id not in checked:
            actual = client.jobs.get(job_id)._to_dict()
            expected = submitted["jobs"][arm]
            if actual["inputs"] != expected["inputs"] or actual["outputs"]["amlt_results"]["path"] != expected["output_path"]:
                raise ValueError("Monitored job identity differs")
            verify()
            checked.add(job_id)
        current = original(client, session, job_id, previous, now)
        if current["status"] == "Completed":
            result = collect(container, submitted, arm)
            current["independent_final_audit_passed"] = result["passed"]
            current["formal_output_health_passed"] = result["health_passed"]
        return current

    with (ROOT / "WATCH.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch.observe = observe
        watch.classify_observation = classify_observation
        sys.argv = [sys.argv[0], "--subscription", submitted["subscription"], "--resource-group", submitted["resource_group"],
                    "--workspace", submitted["workspace"], "--state-file", str(ROOT / "HEALTH.json"),
                    "--interval", "60", "--stale-minutes", "25"]
        for job_id in jobs:
            sys.argv.extend(["--job", job_id])
        status = watch.main()
        if all((ROOT / "outputs/results" / arm / "RAW_RESULT_AUDIT.json").exists() for arm in ARMS):
            summarize()
        return status


if __name__ == "__main__":
    raise SystemExit(main())