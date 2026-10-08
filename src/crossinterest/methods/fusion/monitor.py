"""Local-only completion collection and independent raw audit for fixed fusion."""

from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from crossinterest.evaluation.common import latest_details, write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.fusion.runner import ROOT, PREFLIGHT, METHOD, DEV, MAPPING, audit, verify
from crossinterest.methods.mechanism.audit import download_artifact
import crossinterest.methods.runtime.performance as watch


def independent_metrics(output):
    mapping = json.loads(MAPPING.read_text())["id_to_category"]
    by_class = defaultdict(list)
    identities = set()
    expected = {row["sample_id"]: row for row in pq.read_table(DEV).to_pylist()}
    for shard in range(2):
        directory = output / f"shard-{shard}"
        contract = json.loads((directory / "run_contract.json").read_text())
        for detail in latest_details(directory / "details.jsonl", contract["fingerprint"]).values():
            identity = detail["sample_id"]
            if identity in identities or identity not in expected:
                raise ValueError("Duplicate or unexpected sample")
            identities.add(identity)
            target = expected[identity]["target_interest"]
            rank = None
            seen = set()
            for position, entry in enumerate(detail["ranking"], 1):
                predicted = mapping.get(entry["predicted_output"])
                if entry["prediction"] != predicted:
                    raise ValueError("Saved prediction disagrees with ID mapping")
                if entry["format_valid"] and (predicted is None or entry["raw_text"] != entry["predicted_output"]):
                    raise ValueError("Strict format flag disagrees with raw text")
                if predicted is not None:
                    if predicted in seen:
                        raise ValueError("Duplicate valid beam")
                    seen.add(predicted)
                if predicted == target and rank is None:
                    rank = position
            by_class[target].append({f"macro_{metric}_at_{cutoff}":
                (1.0 / math.log2(rank + 1) if metric == "ndcg" else 1.0)
                if rank is not None and rank <= cutoff else 0.0
                for metric in ("recall", "ndcg") for cutoff in (10, 20)})
    if identities != set(expected) or len(by_class) != 112:
        raise ValueError("Independent result coverage is incomplete")
    return {key: sum(sum(row[key] for row in values) / len(values)
                     for values in by_class.values()) / len(by_class)
            for key in next(iter(by_class.values()))[0]}


def collect(container, submitted):
    prepared = verify()
    if sha256_file(PREFLIGHT) != submitted["preflight_sha256"]:
        raise ValueError("Submission preflight changed")
    output = ROOT / "outputs/results"
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    names = ["COMPLETE.json", "EXECUTION_CONTRACT.json", "RESULT_AUDIT.json"]
    names += [f"shard-{shard}/{name}" for shard in range(2) for name in (
        "run_contract.json", "details.jsonl", "fusion_checks.jsonl", "metrics.json", "predictions.jsonl")]
    for name in names:
        blob = container.get_blob_client(submitted["results_blob_prefix"] + "/" + name)
        downloads[name] = download_artifact(blob, output / name, downloads.get(name, {}))
        write_json_atomic(receipt_path, downloads)
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    if execution["preflight"] != prepared:
        raise ValueError("Cloud execution differs from approved preflight")
    result = audit(output)
    if result != json.loads((output / "RESULT_AUDIT.json").read_text()):
        raise ValueError("Independent audit differs from cloud")
    completion = json.loads((output / "COMPLETE.json").read_text())
    if (completion["method"] != METHOD or not completion["passed"]
            or completion["result_audit_sha256"] != sha256_file(output / "RESULT_AUDIT.json")
            or completion["health_passed"] != result["health_passed"]
            or completion["improvement_gate_passed"] != result["improvement_gate_passed"]):
        raise ValueError("Completion receipt differs")
    independent = independent_metrics(output)
    if not all(math.isclose(value, result["metrics"][key], abs_tol=1e-12, rel_tol=0)
               for key, value in independent.items()):
        raise ValueError("Independent raw-text metric calculation differs")
    for name, entry in downloads.items():
        if sha256_file(output / name) != entry["sha256"]:
            raise ValueError("Downloaded bytes changed")
    completed = datetime.now(timezone.utc)
    clock = json.loads((ROOT / "BUDGET_CLOCK.json").read_text())
    result.update(observed_utc=completed.isoformat(), job_id=submitted["job_id"], downloads=downloads,
                  independent_raw_metrics=independent,
                  submission_receipt_sha256=sha256_file(ROOT / "SUBMISSION_RECEIPT.json"),
                  end_to_end_seconds=(completed - datetime.fromisoformat(clock["submission_started_utc"])).total_seconds(),
                  eight_hour_target_met=completed <= datetime.fromisoformat(clock["target_deadline_utc"]))
    write_json_atomic(output / "RAW_RESULT_AUDIT.json", result)
    print(json.dumps({"event": "fusion_independent_final_audit", "metrics": independent,
                      **{key: result[key] for key in ("passed", "health_passed", "improvement_gate_passed",
                                                     "end_to_end_seconds", "eight_hour_target_met")}}), flush=True)
    return result


def self_test():
    from crossinterest.methods.fusion.runner import ranking

    prepared = verify()
    mapping = json.loads(MAPPING.read_text())
    category_to_id = {category: token for token, category in mapping["id_to_category"].items()}
    categories = sorted(category_to_id)
    with TemporaryDirectory(prefix="fusion-audit-", dir=ROOT) as temporary:
        output = Path(temporary)
        for shard in range(2):
            directory = output / f"shard-{shard}"
            directory.mkdir()
            rows = pq.read_table(ROOT / prepared["shards"][str(shard)]["path"]).to_pylist()
            fingerprint = f"fixture-{shard}"
            contract = {"fingerprint": fingerprint, "fusion_method": METHOD,
                "fusion_preflight_sha256": sha256_file(PREFLIGHT), "fusion_linear_sha256": prepared["files"]["linear_model.npz"],
                "fusion_weight": 0.5, "source_adapter_sha256": prepared["source_adapter_sha256"], "fusion_shard": shard,
                "dataset_sha256": prepared["shards"][str(shard)]["sha256"], "label_map_sha256": prepared["mapping_sha256"],
                "label_file_sha256": prepared["train_sha256"], "sample_limit": None, "num_beams": 20,
                "num_return_sequences": 20, "do_sample": False, "candidates_in_prompt": False,
                "decoder_label_constraint": False, "decoding": "candidate_free_direct_beam_search",
                "max_new_tokens": 32, "eval_batch_size": 1, "label_mode": "interest_id",
                "view": "behavior_interest", "model_device_map": "balanced"}
            write_json_atomic(directory / "run_contract.json", contract)
            details = []
            checks = []
            for row in rows:
                target = row["target_interest"]
                predictions = [target, *[category for category in categories if category != target][:19]]
                entries = [{"rank": position, "prediction": category, "predicted_output": category_to_id[category],
                            "raw_text": category_to_id[category], "format_valid": True, "sequence_score": -float(position)}
                           for position, category in enumerate(predictions, 1)]
                details.append({"sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
                    "target_interest": target, "ranking": entries, "run_fingerprint": fingerprint,
                    **ranking.score_ranking(entries, target)})
                checks.append({"sample_id": row["sample_id"], "method": METHOD, "preflight_sha256": sha256_file(PREFLIGHT),
                               "applied_calls": 1, "num_beams": 20, "max_mass_error": 0})
            for name, values in (("details.jsonl", details), ("fusion_checks.jsonl", checks)):
                with (directory / name).open("w") as handle:
                    for value in values:
                        handle.write(json.dumps(value) + "\n")
            write_json_atomic(directory / "metrics.json", ranking.metrics_from_rankings(details, rows, 20))
            ranking.write_predictions(directory / "predictions.jsonl", details)
        result = audit(output)
        assert result["health_passed"] and result["improvement_gate_passed"]
        assert all(value == 1 for value in independent_metrics(output).values())
        path = output / "shard-0/details.jsonl"
        original = path.read_text()
        values = [json.loads(line) for line in original.splitlines()]
        values[0]["ranking"][0]["raw_text"] = "invalid"
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n")
        try:
            independent_metrics(output)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt strict-format output was accepted")
    print("PASS: complete 2000-row audit, independent raw metrics, strict-format corruption rejection")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    from azure.identity import AzureCliCredential
    from azure.storage.blob import ContainerClient

    submitted = json.loads((ROOT / "SUBMISSION_RECEIPT.json").read_text())
    clock = json.loads((ROOT / "BUDGET_CLOCK.json").read_text())
    state_path = ROOT / "HEALTH.json"
    container = ContainerClient(submitted["account_url"], "amulet", credential=AzureCliCredential(process_timeout=30),
                                connection_timeout=15, read_timeout=60, retry_total=3)
    original = watch.observe
    checked = set()

    def observe(client, session, job_id, previous, now):
        if job_id not in checked:
            job = client.jobs.get(job_id)._to_dict()
            if (job["inputs"]["amlt_code"] != submitted["code"] or job["inputs"]["frozen"] != submitted["frozen"]
                    or job["outputs"]["amlt_results"]["path"] != submitted["output_path"]):
                raise ValueError("Monitored job identity differs")
            verify()
            checked.add(job_id)
        current = original(client, session, job_id, previous, now)
        if current["status"] == "Completed":
            result = collect(container, submitted)
            current["independent_final_audit_passed"] = result["passed"]
            current["formal_output_health_passed"] = result["health_passed"]
        if now > datetime.fromisoformat(clock["target_deadline_utc"]) and current["status"] != "Completed":
            write_json_atomic(state_path, {job_id: current})
            print(json.dumps({"event": "fusion_eight_hour_target_exceeded", "job_id": job_id}), flush=True)
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