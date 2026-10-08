"""Independently audit and complete exclusive checkpoint handoffs locally."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.factorization.resume import validate_benchmark, validate_layout, validate_stress_gate
from crossinterest.methods.factorization.runner import preflight, sources, verify_package
from crossinterest.methods.runtime.performance import latest_stdout


STOPPED = {"Canceled", "Cancelled", "Failed"}
TERMINAL = STOPPED | {"Completed", "NotResponding"}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_ready(directory, row, container):
    prefix = row["results_blob_prefix"] + "/"
    names = ["READY_FOR_HANDOFF.json", "PERFORMANCE_CONTRACT.json", "RESUME_AUDIT.json",
             "PERFORMANCE_BENCHMARK.json", "EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json",
             "smoke/RESULT_AUDIT.json", "formal/auxiliary/runtime_config.json"]
    names += [f"ORIGINAL_RNG/rng_state_{rank}.pth" for rank in range(4)]
    names += [f"formal/auxiliary/STRESS_RANK_{rank}.json" for rank in range(8)]
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(container.download_blob(prefix + name).readall())
    load = lambda name: json.loads((directory / name).read_text())
    performance, migration, ready = (load(name) for name in
                                    ("PERFORMANCE_CONTRACT.json", "RESUME_AUDIT.json", "READY_FOR_HANDOFF.json"))
    require(all(performance.get(key) == value for key, value in {
        "arm": row["arm"], "world_size": 8, "effective_batch_size": 32,
        "gradient_checkpointing": False, "budget_unchanged": True,
        "bitwise_trajectory_equivalence_claimed": False}.items()), "performance identity differs")
    expected = {"execution_code_sha256": Path(__file__).with_name("cir_factor_resume.py"),
        "parallel_command_sha256": Path(__file__).with_name("cir_factor_speed.py"),
        "original_execution_sha256": directory / "EXECUTION_CONTRACT.json",
        "original_smoke_sha256": directory / "smoke/RESULT_AUDIT.json",
        "benchmark_sha256": directory / "PERFORMANCE_BENCHMARK.json",
        "resume_audit_sha256": directory / "RESUME_AUDIT.json"}
    require(all(performance.get(key) == sha256_file(path) for key, path in expected.items()), "execution hash differs")
    require(validate_benchmark(load("PERFORMANCE_BENCHMARK.json"), 8, False) == performance["benchmark_gate"],
            "benchmark gate differs")
    execution = load("EXECUTION_CONTRACT.json")
    require(execution["factor_arm"] == row["arm"] and execution["code_sha256"] == sources()
            and execution["data_contract"] == preflight()["original"]
            and execution["fresh_formal_after_smoke"] is True, "original execution differs")
    require(migration["passed"] is True and migration["world_size_before"] == 4
            and migration["world_size_after"] == 8, "migration is incomplete")
    step = migration["checkpoint_step"]
    require(0 < step < 7866 and ready["passed"] is True and ready["optimizer_updates"] == 0
            and ready["checkpoint_step"] == step
            and ready["performance_contract_sha256"] == sha256_file(directory / "PERFORMANCE_CONTRACT.json"),
            "readiness identity differs")
    validate_stress_gate(ready["stress"], step, 8)
    validate_layout(load("formal/auxiliary/runtime_config.json"), 8, False)
    for rank in range(8):
        require(load(f"formal/auxiliary/STRESS_RANK_{rank}.json") == ready["stress"][rank], "rank stress differs")
    for rank in range(4):
        require(sha256_file(directory / f"ORIGINAL_RNG/rng_state_{rank}.pth") ==
                migration["original_sha256"][f"rng_state_{rank}.pth"], "original RNG differs")
    source_prefix = row["source_blob_prefix"]
    require(performance["source"] == "/mnt/frozen/" + source_prefix, "source output differs")
    require(migration["source"] == f"/mnt/frozen/{source_prefix}/formal/auxiliary/checkpoint-{step}",
            "source checkpoint differs")
    for name in ("EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json", "smoke/RESULT_AUDIT.json"):
        data = container.download_blob(source_prefix + "/" + name).readall()
        require(hashlib.sha256(data).hexdigest() == sha256_file(directory / name), "source receipt differs")
    for name, digest in migration["staged_sha256"].items():
        require(Path(name).name == name, "invalid checkpoint path")
        stream = container.download_blob(f"{prefix}formal/auxiliary/checkpoint-{step}/{name}", max_concurrency=4)
        actual = hashlib.sha256()
        for chunk in stream.chunks():
            actual.update(chunk)
        require(actual.hexdigest() == digest, f"staged checkpoint hash differs: {name}")
        if name in migration["original_sha256"] and not name.startswith("rng_state_"):
            require(digest == migration["original_sha256"][name], "original checkpoint identity differs")
    result = {"passed": True, "step": step, "old_job_id": row["old_job_id"], "new_job_id": row["new_job_id"],
        "performance_contract_sha256": sha256_file(directory / "PERFORMANCE_CONTRACT.json"),
        "ready_sha256": sha256_file(directory / "READY_FOR_HANDOFF.json"), "observed_utc": now()}
    write_json_atomic(directory / "INDEPENDENT_HANDOFF_AUDIT.json", result)
    return result


def advance(row, directory, backend, state):
    new, old = backend.details(row["new_job_id"]), backend.details(row["old_job_id"])
    require(new.status not in TERMINAL, f"replacement is terminal: {new.status}")
    if state.get("authorized"):
        require(old.status in STOPPED, "original job became active after authorization")
        text = backend.stdout(new)
        steps = [int(value) for value in re.findall(r"(\d+)/7866", text) if int(value) > state["step"]]
        if steps:
            state.update(continued=True, first_observed_step=min(steps), latest_observed_step=max(steps))
            write_json_atomic(directory / "CONTINUATION_AUDIT.json", {"passed": True, **state, "observed_utc": now()})
        return
    if new.status != "Running" or not backend.exists(row, "READY_FOR_HANDOFF.json"):
        return
    backend.validate_job(row)
    if "step" not in state:
        state.update(backend.audit(directory, row))
    backend.revalidate(row, state)
    require(not backend.auxiliary_complete(row), "original auxiliary stage completed; do not replay completed work")
    if old.status == "Completed":
        raise ValueError("original job completed; do not replace completed work")
    if old.status not in STOPPED:
        if not state.get("cancel_requested"):
            require(old.status != "NotResponding", "original termination is uncertain")
            backend.cancel(row)
            state.update(cancel_requested=True, cancel_requested_utc=now())
            write_json_atomic(directory / "CANCEL_REQUEST.json", state)
        return
    require(old.end_time_utc is not None, "original terminal time is missing")
    backend.revalidate(row, state)
    last_steps = re.findall(r"(\d+)/7866", backend.stdout(old))
    last_step = int(last_steps[-1]) if last_steps else None
    authorization = {"original_job_stopped": True,
        "performance_contract_sha256": state["performance_contract_sha256"], "ready_sha256": state["ready_sha256"],
        "original_job_id": row["old_job_id"], "new_job_id": row["new_job_id"], "arm": row["arm"],
        "original_terminal_status": old.status, "original_end_time_utc": str(old.end_time_utc),
        "checkpoint_step": state["step"], "original_last_observed_step": last_step,
        "steps_to_replay": last_step - state["step"] if last_step is not None else None,
        "independent_audit_sha256": sha256_file(directory / "INDEPENDENT_HANDOFF_AUDIT.json"),
        "controller_sha256": sha256_file(Path(__file__)), "observed_utc": now()}
    backend.authorize(row, directory, authorization)
    state.update(authorized=True, authorized_utc=authorization["observed_utc"])


class AzureBackend:
    def __init__(self, manifest):
        import requests
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential
        from azure.storage.blob import ContainerClient

        credential = AzureCliCredential(process_timeout=30)
        self.client = MLClient(credential, manifest["subscription"], manifest["resource_group"], manifest["workspace"])
        self.container = ContainerClient(manifest["account_url"], "amulet", credential=credential)
        self.session = requests.Session()
        self.amlt = manifest["amlt"]
        self.original_experiment = manifest["original_experiment"]

    def details(self, job_id):
        return self.client.jobs._runs_operations.get_run_details(job_id)

    def stdout(self, details):
        logs = details.log_files or {}
        name = latest_stdout(list(logs))
        if name is None:
            return ""
        response = self.session.get(logs[name], headers={"Range": "bytes=-131072"}, timeout=40)
        response.raise_for_status()
        return response.text

    def exists(self, row, name):
        return self.container.get_blob_client(row["results_blob_prefix"] + "/" + name).exists()

    def auxiliary_complete(self, row):
        prefix = row["source_blob_prefix"] + "/formal/auxiliary/"
        return any(self.container.get_blob_client(prefix + name).exists() for name in
                   ("train_metrics.json", "final_adapter/adapter_model.safetensors",
                    "checkpoint-7866/trainer_state.json"))

    def validate_job(self, row):
        record = self.client.jobs.get(row["new_job_id"])._to_dict()
        require(record["inputs"]["amlt_code"]["path"] == row["code"], "platform code differs")
        require(record["inputs"]["frozen"]["mode"] == "ro_mount", "source mount is not read-only")
        environment = record["component"]["environment_variables"]
        require(environment["FACTOR_ARM"] == row["arm"] and
                environment["FACTOR_SOURCE_PREFIX"] == row["source_blob_prefix"].rsplit("/", 1)[-1],
                "platform source/arm differs")
        require(record["outputs"]["amlt_results"]["path"].rstrip("/").endswith("/" + row["results_blob_prefix"]),
                "platform output differs")

    def audit(self, directory, row):
        return audit_ready(directory, row, self.container)

    def revalidate(self, row, state):
        require(self.details(row["new_job_id"]).status == "Running", "replacement stopped during audit")
        for name, field in (("PERFORMANCE_CONTRACT.json", "performance_contract_sha256"),
                            ("READY_FOR_HANDOFF.json", "ready_sha256")):
            data = self.container.download_blob(row["results_blob_prefix"] + "/" + name).readall()
            require(hashlib.sha256(data).hexdigest() == state[field], "audited receipt changed")
        require('"waiting_for_exclusive_handoff"' in self.stdout(self.details(row["new_job_id"])),
                "replacement is not waiting for handoff")

    def cancel(self, row):
        subprocess.run([self.amlt, "cancel", self.original_experiment, ":" + row["old_job_id"], "-y"],
                       env=dict(os.environ, AMLT_NO_BATCH_STATUS_API="1"), check=True)

    def authorize(self, row, directory, authorization):
        from azure.storage.blob import ContentSettings

        blob = self.container.get_blob_client(row["results_blob_prefix"] + "/HANDOFF_AUTHORIZATION.json")
        if blob.exists():
            data = blob.download_blob().readall()
            stored = json.loads(data)
            for field in ("original_job_stopped", "performance_contract_sha256", "ready_sha256",
                          "original_job_id", "new_job_id", "checkpoint_step"):
                require(stored.get(field) == authorization[field], "existing authorization differs")
        else:
            data = (json.dumps(authorization, indent=2) + "\n").encode()
            blob.upload_blob(data, overwrite=False, content_settings=ContentSettings(content_type="application/json"))
            require(blob.download_blob().readall() == data, "authorization readback differs")
        (directory / "HANDOFF_AUTHORIZATION.json").write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_package()
    manifest = json.loads(args.manifest.read_text())
    require(manifest.get("automatic_exclusive_handoff") is True, "automatic handoff not enabled")
    require(bool(manifest["jobs"]), "empty handoff manifest")
    require(len({row["old_job_id"] for row in manifest["jobs"].values()}) == len(manifest["jobs"]), "duplicate originals")
    require(len({row["new_job_id"] for row in manifest["jobs"].values()}) == len(manifest["jobs"]), "duplicate replacements")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity_path = args.output / "MANIFEST_IDENTITY.json"
    identity = {"sha256": sha256_file(args.manifest)}
    if identity_path.exists():
        require(json.loads(identity_path.read_text()) == identity, "controller manifest changed")
    else:
        write_json_atomic(identity_path, identity)
    state_path = args.output / "STATE.json"
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    backend, waiter = AzureBackend(manifest), threading.Event()
    while True:
        for name, row in manifest["jobs"].items():
            state = states.setdefault(name, {})
            if state.get("continued") or state.get("blocked"):
                continue
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            try:
                advance(row, directory, backend, state)
                state["inspection_errors"] = 0
            except ValueError as error:
                state.update(blocked=True, error=str(error))
            except Exception as error:
                state["inspection_errors"] = state.get("inspection_errors", 0) + 1
                state["error_type"] = type(error).__name__
                if state["inspection_errors"] >= 3:
                    state["blocked"] = True
            state["observed_utc"] = now()
            write_json_atomic(state_path, states)
            print(json.dumps({"job": name, **state}), flush=True)
        if all(state.get("continued") or state.get("blocked") for state in states.values()):
            return 13 if any(state.get("blocked") for state in states.values()) else 0
        waiter.wait(15)


if __name__ == "__main__":
    raise SystemExit(main())