"""Alert on fresh scheduling interruptions or stale logs without modifying jobs."""

from __future__ import annotations

import argparse
import json
import re
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from crossinterest.evaluation.common import write_json_atomic


TERMINAL_STATES = {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}


def latest_stdout(names: list[str]) -> str | None:
    candidates = [name for name in names if name.startswith("user_logs/") and name.endswith("std_log.txt")]
    if not candidates:
        return None

    def retry_number(name):
        match = re.search(r"/retry_(\d+)/", name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=retry_number)


def progress_line(text: str) -> str | None:
    lines = text.replace("\r", "\n").splitlines()
    matches = [line for line in lines if re.search(
        r"tokenizer audit|completed=\d+/\d+|\d+/(?:840|2622|2848|3500|4954|7866|2000)|train_runtime|Loading.*100%", line)]
    return matches[-1][-600:] if matches else None


def has_smoke_audit(text: str) -> bool:
    for line in text.replace("\r", "\n").splitlines():
        if not any(marker in line for marker in ('"stage_audits"', '"training"', '"factor_phase_passed"')):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("event") == "factor_phase_passed" and value.get("phase") == "smoke":
            return True
        if (isinstance(value, dict) and value.get("passed") is True and value.get("smoke") is True
            and ("stage_audits" in value or value.get("training", {}).get("passed") is True)):
            return True
    return False


def has_handoff_ready(text: str) -> bool:
    for line in text.replace("\r", "\n").splitlines():
        if '"waiting_for_exclusive_handoff"' not in line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("event") == "waiting_for_exclusive_handoff":
            return True
    return False


def classify_observation(current: dict, previous: dict | None, now: datetime,
                         stale_seconds: float) -> str | None:
    if current["status"] in {"Failed", "NotResponding"}:
        return "job_failure"
    if current["status"] in TERMINAL_STATES:
        return "terminal"
    if previous is None:
        return None
    if current["interruption"] and current["interruption"] != previous.get("interruption"):
        return "scheduling_interruption"
    if (previous.get("log_file") and current.get("log_file")
            and current["log_file"] != previous["log_file"] and "/retry_" in current["log_file"]):
        return "attempt_restart"
    if current["status"] in {"Running", "Queued", "Preparing", "Starting", "NotStarted"}:
        last_active = datetime.fromisoformat(current["last_active_at"])
        if (now - last_active).total_seconds() >= stale_seconds:
            return "no_progress"
    return None


def observe(client, session, job_id: str, previous: dict | None, now: datetime) -> dict:
    details = client.jobs._runs_operations.get_run_details(job_id)
    logs = details.log_files or {}
    filename = latest_stdout(list(logs))
    progress = None
    modified = None
    smoke_audited = bool(previous and previous.get("smoke_audited"))
    handoff_ready = False
    if filename:
        response = session.get(logs[filename], headers={"Range": "bytes=-131072"}, timeout=40)
        response.raise_for_status()
        text = response.content.decode("utf-8", errors="replace")
        progress = progress_line(text)
        smoke_audited = smoke_audited or has_smoke_audit(text)
        handoff_ready = has_handoff_ready(text)
        if response.headers.get("Last-Modified"):
            modified = parsedate_to_datetime(response.headers["Last-Modified"]).isoformat()
    activity = "logs/azureml/virtualcluster_activity_log.txt"
    interruption = None
    if activity in logs:
        response = session.get(logs[activity], timeout=40)
        response.raise_for_status()
        events = [line for line in response.text.splitlines()
                  if re.search(r"\] (Suspended|Requeued|Restarting):", line)]
        interruption = events[-1] if events else None
    identity = (details.status, filename, modified, progress)
    prior_identity = tuple(previous.get(key) for key in ("status", "log_file", "log_modified", "progress")) if previous else None
    last_active = (previous["last_active_at"] if previous and identity == prior_identity
                   else now.isoformat())
    return {"job_id": job_id, "status": details.status, "log_file": filename,
            "log_modified": modified, "progress": progress, "interruption": interruption,
            "last_active_at": last_active, "observed_at": now.isoformat(), "smoke_audited": smoke_audited,
            "handoff_ready": handoff_ready}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--stale-minutes", type=float, default=30)
    parser.add_argument("--exit-after-smoke", action="store_true")
    parser.add_argument("--exit-on-handoff", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.stale_minutes <= 0 or len(set(args.job)) != len(args.job):
        parser.error("positive intervals and unique jobs are required")
    import requests
    from azure.ai.ml import MLClient
    from azure.identity import AzureCliCredential

    client = MLClient(AzureCliCredential(process_timeout=30), args.subscription, args.resource_group, args.workspace)
    session = requests.Session()
    state = json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
    waiter = threading.Event()
    while True:
        alerts = []
        handoffs = []
        for job_id in args.job:
            if state.get(job_id, {}).get("status") in TERMINAL_STATES:
                continue
            now = datetime.now(timezone.utc)
            previous = state.get(job_id)
            try:
                current = observe(client, session, job_id, previous, now)
                event = classify_observation(current, previous, now, args.stale_minutes * 60)
                current["inspection_errors"] = 0
                state[job_id] = current
                if current.get("handoff_ready") and not (previous or {}).get("handoff_ready"):
                    handoffs.append(job_id)
                print(json.dumps({"event": event or "progress", **current}), flush=True)
                if event in {"scheduling_interruption", "attempt_restart", "no_progress", "job_failure"}:
                    alerts.append(job_id)
            except Exception as error:
                current = dict(previous or {"job_id": job_id})
                current["inspection_errors"] = current.get("inspection_errors", 0) + 1
                state[job_id] = current
                print(json.dumps({"event": "inspection_error", "job_id": job_id,
                                  "error_type": type(error).__name__,
                                  "consecutive_errors": current["inspection_errors"]}), flush=True)
                if current["inspection_errors"] >= 3:
                    alerts.append(job_id)
        write_json_atomic(args.state_file, state)
        if alerts:
            print(json.dumps({"event": "inspection_required", "jobs": alerts}), flush=True)
            return 13
        if args.exit_on_handoff and handoffs:
            print(json.dumps({"event": "handoff_ready_for_inspection", "jobs": handoffs}), flush=True)
            return 0
        if args.exit_after_smoke and all(state.get(job_id, {}).get("smoke_audited") for job_id in args.job):
            print(json.dumps({"event": "all_smoke_audited", "jobs": args.job}), flush=True)
            return 0
        if all(state.get(job_id, {}).get("status") in TERMINAL_STATES for job_id in args.job):
            return 0
        waiter.wait(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())