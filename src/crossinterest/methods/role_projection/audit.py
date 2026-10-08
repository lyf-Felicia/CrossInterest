"""Download immutable formal artifacts and independently verify role results."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def save_new(path, record):
    payload = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite evidence: {path}")
    else:
        path.write_bytes(payload)


def download(receipts, output):
    from azure.identity import AzureCliCredential
    from azure.storage.blob import ContainerClient

    credential = AzureCliCredential()
    pending, manifests, plans = [], {}, {}
    for mode, receipt in receipts.items():
        container = ContainerClient(receipt["account_url"], receipt["container"], credential=credential,
                                    max_single_get_size=256 * 1024, max_chunk_get_size=256 * 1024,
                                    connection_timeout=15, read_timeout=60, retry_total=2)
        prefix = receipt["results_blob_prefix"] + "/"
        directory = output / mode
        manifest_path = directory / "DOWNLOAD_MANIFEST.json"
        plans[mode] = {item["path"]: item for item in json.loads(
            (output.parent / f"{mode}_DOWNLOAD_PLAN.json").read_text())}
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        elif (directory / "DOWNLOAD_PROGRESS.jsonl").exists():
            for line in (directory / "DOWNLOAD_PROGRESS.jsonl").read_text().splitlines():
                record = json.loads(line)
                manifest[record.pop("path")] = record
        for relative, item in manifest.items():
            planned = plans[mode][relative]
            if (digest(directory / relative) != item["sha256"] or item["size"] != planned["size"]
                    or item["etag"] != planned["etag"]):
                raise ValueError(f"download changed: {mode}/{relative}")
        manifests[mode] = manifest
        for relative, item in plans[mode].items():
            if relative in manifest:
                continue
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.with_name(path.name + ".partial").exists():
                raise ValueError(f"partial unverified download exists: {path}")
            pending.append((mode, relative, item, container, prefix, path))

    def transfer(task):
        mode, relative, item, container, prefix, path = task
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("xb") as stream:
            container.download_blob(prefix + relative, if_match=item["etag"], max_concurrency=8).readinto(stream)
        if temporary.stat().st_size != item["size"]:
            raise ValueError(f"download size mismatch: {relative}")
        result = {"sha256": digest(temporary), "size": item["size"], "etag": item["etag"]}
        temporary.rename(path)
        return mode, relative, result

    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(transfer, item): item for item in sorted(pending, key=lambda item: item[2]["size"])}
        for future in as_completed(futures):
            try:
                mode, relative, item = future.result()
            except Exception as error:
                mode, relative = futures[future][:2]
                failures.append({"mode": mode, "path": relative, "error_type": type(error).__name__})
                continue
            manifests[mode][relative] = item
            with (output / mode / "DOWNLOAD_PROGRESS.jsonl").open("a") as stream:
                stream.write(json.dumps({"path": relative, **item}) + "\n")
            count = sum(len(manifest) for manifest in manifests.values())
            if count % 20 == 0 or relative.endswith(".safetensors"):
                print(json.dumps({"downloaded_files": count, "last": f"{mode}/{relative}",
                                  "bytes": sum(record["size"] for manifest in manifests.values()
                                               for record in manifest.values())}), flush=True)
    if failures:
        raise RuntimeError(f"incomplete download; successful files retained in progress journals: {failures}")
    for mode, manifest in manifests.items():
        if set(manifest) != set(plans[mode]):
            raise ValueError(f"incomplete download: {mode}")
        save_new(output / mode / "DOWNLOAD_MANIFEST.json", manifest)


def independent_metrics(details, expected, mapping):
    gold = {row["sample_id"]: row for row in expected}
    if len(gold) != len(expected) or {row["sample_id"] for row in details} != set(gold) or len(details) != len(gold):
        raise ValueError("raw evaluation coverage mismatch")
    by_class = defaultdict(list)
    valid = strict = duplicate = total = 0
    for detail in details:
        row = gold[detail["sample_id"]]
        if detail["target_interest"] != row["target_interest"] or detail["source_sample_id"] != row["source_sample_id"]:
            raise ValueError("raw evaluation label mismatch")
        beams = detail["ranking"]
        if len(beams) != 20 or [beam["rank"] for beam in beams] != list(range(1, 21)):
            raise ValueError("raw beam order/count mismatch")
        seen = set()
        target_rank = None
        for beam in beams:
            if not math.isfinite(beam["sequence_score"]):
                raise ValueError("nonfinite beam score")
            prediction = beam["prediction"]
            if prediction is not None:
                if mapping["id_to_category"].get(beam["predicted_output"]) != prediction:
                    raise ValueError("raw ID mapping mismatch")
                valid += 1
                duplicate += prediction in seen
                seen.add(prediction)
            if beam["format_valid"]:
                if prediction is None or beam["raw_text"] != beam["predicted_output"]:
                    raise ValueError("strict beam text mismatch")
                strict += 1
            if target_rank is None and prediction == row["target_interest"]:
                target_rank = beam["rank"]
            total += 1
        values = {}
        for cutoff in (10, 20):
            hit = target_rank is not None and target_rank <= cutoff
            values[f"macro_recall_at_{cutoff}"] = float(hit)
            values[f"macro_ndcg_at_{cutoff}"] = 1 / math.log2(target_rank + 1) if hit else 0.0
        by_class[row["target_interest"]].append(values)
    names = ("macro_recall_at_10", "macro_recall_at_20", "macro_ndcg_at_10", "macro_ndcg_at_20")
    result = {name: sum(sum(row[name] for row in rows) / len(rows) for rows in by_class.values()) / len(by_class)
              for name in names}
    result.update(recorded_sample_count=len(details), class_count=len(by_class),
                  valid_beam_rate=valid / total, strict_beam_format_rate=strict / total,
                  duplicate_valid_beam_rate=duplicate / total)
    return result


def improvement_gate(metrics, criteria):
    baseline, ordinary, projected = (metrics[mode] for mode in ("baseline", "unconstrained", "projected"))
    delta = projected["macro_ndcg_at_20"] - baseline["macro_ndcg_at_20"]
    checks = {"ndcg20_gain_at_least_minimum": delta >= criteria["projected_ndcg20_minus_baseline_minimum"],
              "recall20_not_lower": projected["macro_recall_at_20"] >= baseline["macro_recall_at_20"],
              "ndcg20_above_unconstrained": projected["macro_ndcg_at_20"] > ordinary["macro_ndcg_at_20"]}
    return {"passed": all(checks.values()), "checks": checks, "ndcg20_delta_vs_baseline": delta,
            "ndcg20_delta_vs_unconstrained": projected["macro_ndcg_at_20"] - ordinary["macro_ndcg_at_20"],
            "recall20_delta_vs_baseline": projected["macro_recall_at_20"] - baseline["macro_recall_at_20"],
            "minimum_ndcg20_gain": criteria["projected_ndcg20_minus_baseline_minimum"]}


def audit(bundle, receipts, output):
    sys.path[:0] = [str(bundle), str(bundle / "prep")]
    import pyarrow.parquet as pq
    import crossinterest.methods.role_projection.runner as role_run

    if Path(role_run.__file__).resolve() != bundle / "eval_interest_id_minionerec/role_run.py":
        raise ValueError("audit must use the submitted frozen runner")
    prepared = role_run.verify()
    contract_sha = digest(role_run.CONTRACT)
    mapping = json.loads(role_run.MAPPING.read_text())
    original_write = role_run.write_json_atomic
    role_run.write_json_atomic = lambda path, value: save_new(path.with_name("LOCAL_" + path.name), value)
    metrics, evidence = {}, {}
    try:
        for mode in role_run.MODES:
            directory = output / mode
            manifest = json.loads((directory / "DOWNLOAD_MANIFEST.json").read_text())
            for relative, item in manifest.items():
                if digest(directory / relative) != item["sha256"]:
                    raise ValueError(f"download SHA mismatch: {mode}/{relative}")
            complete = json.loads((directory / "COMPLETE.json").read_text())
            execution = json.loads((directory / "EXECUTION_CONTRACT.json").read_text())
            if (not complete["passed"] or complete["smoke"] or complete["modes"] != [mode]
                    or complete["role_contract_sha256"] != contract_sha
                    or execution["role_contract_sha256"] != contract_sha
                    or receipts[mode]["contract_sha256"] != contract_sha):
                raise ValueError("formal execution identity mismatch")
            formal = directory / mode / "formal"
            for stage in ("auxiliary", "novel"):
                actual = role_run.audit_training(formal / stage, mode, stage, False, prepared, execution["model_path"])
                expected_audit = json.loads((formal / stage / "STAGE_AUDIT.json").read_text())
                if actual != expected_audit or actual != complete["results"][mode]["stages"][stage]:
                    raise ValueError(f"stage audit differs: {mode}/{stage}")
            result = role_run.audit_evaluation(formal, mode, False, prepared)
            if result != json.loads((formal / "RESULT_AUDIT.json").read_text()) or result != complete["results"][mode]["evaluation"]:
                raise ValueError("result audit differs")
            details, expected_rows = [], []
            for shard, info in prepared["shards"].items():
                path = formal / "ranking" / f"shard-{shard}"
                contract = json.loads((path / "run_contract.json").read_text())
                details.extend(role_run.latest_details(path / "details.jsonl", contract["fingerprint"]).values())
                shard_rows = pq.read_table(bundle / info["path"]).to_pylist()
                expected_rows.extend(shard_rows)
                head_audit = json.loads((path / "ROLE_EVAL_AUDIT.json").read_text())
                if head_audit["head_applications"] != (0 if mode == "baseline" else len(shard_rows)):
                    raise ValueError("role head application count mismatch")
            calculated = independent_metrics(details, expected_rows, mapping)
            for name, value in calculated.items():
                if not math.isclose(value, result["metrics"][name], rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"independent metric differs: {mode}/{name}")
            metrics[mode] = calculated
            evidence[mode] = {"job": receipts[mode]["name"], "verified_files": len(manifest),
                              "download_manifest_sha256": digest(directory / "DOWNLOAD_MANIFEST.json"),
                              "cloud_complete_sha256": digest(directory / "COMPLETE.json")}
            print(json.dumps({"audited": mode, "metrics": calculated}), flush=True)
    finally:
        role_run.write_json_atomic = original_write
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "audit_passed": True,
              "role_contract_sha256": contract_sha, "metrics": metrics, "evidence": evidence,
              "health_passed": all(row["valid_beam_rate"] == 1 and row["strict_beam_format_rate"] == 1
                                   and row["duplicate_valid_beam_rate"] == 0 for row in metrics.values()),
              "improvement_gate": improvement_gate(metrics, prepared["gate"])}
    save_new(output / "FINAL_AUDIT.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "audit"))
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    receipts = json.loads(args.receipts.read_text())
    if args.action == "download":
        download(receipts, args.output.resolve())
    else:
        if args.bundle is None:
            parser.error("audit requires --bundle")
        audit(args.bundle.resolve(), receipts, args.output.resolve())


if __name__ == "__main__":
    main()