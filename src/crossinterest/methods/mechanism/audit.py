"""Download final mechanism artifacts and independently recompute all run audits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import subprocess
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.runner import ROOT, PREFLIGHT, ARMS, sources, audit_training, audit_evaluation


def ranking_diagnostics(ordered: list[dict], baseline: list[dict], rows: list[dict], seen_labels: set[str]) -> dict:
    from crossinterest.methods.analysis.followup import METRICS, metric_values, rank_band, rank_of

    if len(ordered) != len(rows) or len(baseline) != len(rows):
        raise ValueError("paired prediction coverage differs")
    for current, original, row in zip(ordered, baseline, rows):
        if any(detail[field] != row[field] for detail in (current, original)
               for field in ("sample_id", "source_sample_id", "target_interest")):
            raise ValueError("paired prediction identity differs")
    labels = {row["target_interest"] for row in rows}
    counts = Counter(row["target_interest"] for row in rows)
    metrics = metric_values(ordered, rows)
    groups = {}
    for name, selected in (("o_supported_93", seen_labels), ("n_only_supported_19", labels - seen_labels)):
        indices = [index for index, row in enumerate(rows) if row["target_interest"] in selected]
        groups[name] = {"samples": len(indices), "classes": len(selected),
                       **metric_values([ordered[index] for index in indices], [rows[index] for index in indices])}
    for key in METRICS:
        if not math.isclose(sum(group["classes"] * group[key] for group in groups.values()) / len(labels),
                            metrics[key], abs_tol=1e-12):
            raise ValueError("group macro decomposition differs")
    transitions = [[0] * 3 for _ in range(3)]
    mass = [[0.0] * 3 for _ in range(3)]
    components = Counter()
    for original, updated, row in zip(baseline, ordered, rows):
        before, after = rank_band(rank_of(original)), rank_band(rank_of(updated))
        weight = 1 / (len(labels) * counts[row["target_interest"]])
        transitions[before][after] += 1
        mass[before][after] += weight
        component = ("both_hit" if before < 2 and after < 2 else
                     "gained_hit" if after < 2 else "lost_hit" if before < 2 else "both_miss")
        components[component] += weight * (updated["ndcg_at_20"] - original["ndcg_at_20"])
    original_metrics = metric_values(baseline, rows)
    if not math.isclose(sum(components.values()), metrics["macro_ndcg_at_20"] - original_metrics["macro_ndcg_at_20"], abs_tol=1e-12):
        raise ValueError("paired nDCG decomposition differs")
    for cutoff, band_limit in ((10, 1), (20, 2)):
        gained = sum(mass[before][after] for before in range(band_limit, 3) for after in range(band_limit))
        lost = sum(mass[before][after] for before in range(band_limit) for after in range(band_limit, 3))
        if not math.isclose(gained - lost, metrics[f"macro_recall_at_{cutoff}"] - original_metrics[f"macro_recall_at_{cutoff}"], abs_tol=1e-12):
            raise ValueError("paired recall decomposition differs")
    return {"metrics": metrics, "groups": groups, "macro_ndcg_at_20_delta_components": dict(components),
            "rank_band_transition_vs_n_only": {"rows": "n_only", "bands": ["1-10", "11-20", "outside_top20_censored"],
                                                "sample_counts": transitions, "macro_mass": mass}}


def compare_results(output: Path, audited: dict) -> dict:
    import pyarrow.parquet as pq
    from crossinterest.evaluation.common import latest_details
    from crossinterest.training.evaluate_ranking import score_ranking
    from crossinterest.methods.analysis.followup import DATA, RUNS, METRICS
    from crossinterest.methods.mechanism.runner import DEV, DEV_SHA, MAPPING_SHA

    historical_path = ROOT.parent / "curriculum/outputs/FULLNOVEL_DIAGNOSTICS_20260915.json"
    historical = json.loads(historical_path.read_text())
    manifest_path = DATA / "MANIFEST.json"
    if not audited["passed"] or not historical["passed"] or sha256_file(DEV) != DEV_SHA:
        raise ValueError("comparison requires audited results and original Dev")
    relative_manifest = str(manifest_path.relative_to(ROOT.parent.parent))
    if sha256_file(manifest_path) != historical["source_sha256"][relative_manifest]:
        raise ValueError("historical support groups changed")
    seen_labels = set(json.loads(manifest_path.read_text())["seen_labels"])
    rows = pq.read_table(DEV).to_pylist()
    if len(rows) != 2000 or len({row["target_interest"] for row in rows}) != 112 or len(seen_labels) != 93:
        raise ValueError("comparison support differs from frozen 112/93/19 groups")
    raw = {}
    conditions = {}
    provenance = {str(historical_path): sha256_file(historical_path), str(manifest_path): sha256_file(manifest_path)}
    for name, directory in {**RUNS, **{arm: output / arm / "formal/ranking" for arm in ARMS}}.items():
        contract_path, details_path = directory / "run_contract.json", directory / "details.jsonl"
        if name in RUNS:
            for path in (contract_path, details_path):
                if sha256_file(path) != historical["source_sha256"][str(path.relative_to(ROOT.parent.parent))]:
                    raise ValueError("historical baseline source changed")
            expected = historical["conditions"][name]["metrics"]
        else:
            formal = audited["audits"][name]["formal"]
            if sha256_file(details_path) != formal["raw_ranking_sha256"]:
                raise ValueError("new raw ranking differs from independent audit")
            expected = formal["results"]["ranking"]
        contract = json.loads(contract_path.read_text())
        required = {"dataset_sha256": DEV_SHA, "label_map_sha256": MAPPING_SHA, "num_beams": 20,
                    "do_sample": False, "view": "behavior_interest", "label_mode": "interest_id",
                    "decoder_label_constraint": False, "candidates_in_prompt": False,
                    "decoding": "candidate_free_direct_beam_search"}
        if any(contract.get(key) != value for key, value in required.items()):
            raise ValueError("comparison evaluation contract differs")
        details = latest_details(details_path, contract["fingerprint"])
        if set(details) != {row["sample_id"] for row in rows}:
            raise ValueError("comparison raw coverage differs")
        ordered = [dict(details[row["sample_id"]]) for row in rows]
        for detail, row in zip(ordered, rows):
            detail.update(score_ranking(detail["ranking"], row["target_interest"]))
        raw[name] = ordered
        conditions[name] = ranking_diagnostics(ordered, raw["n_only"], rows, seen_labels)
        if conditions[name]["metrics"] != {key: expected[key] for key in METRICS}:
            raise ValueError("comparison metrics differ from audited values")
        provenance.update({str(path): sha256_file(path) for path in (contract_path, details_path)})
    baseline = conditions["n_only"]["metrics"]
    gates = {arm: {"ndcg20_improved": conditions[arm]["metrics"]["macro_ndcg_at_20"] > baseline["macro_ndcg_at_20"],
                   "recall20_preserved": conditions[arm]["metrics"]["macro_recall_at_20"] >= baseline["macro_recall_at_20"]}
             for arm in ARMS}
    result = {"passed": True, "experiment": audited["experiment"], "conditions": conditions,
              "prespecified_gates_vs_n_only": gates, "source_sha256": provenance,
              "analyzer_sha256": sha256_file(Path(__file__)),
              "limitations": ["Single seed and reused Dev; no significance or held-out generalization claim.",
                              "Auxiliary arms have extra training compute; structure adds parameters.",
                              "Historical 93/19 groups are fixed; outside-top20 ranks remain censored."]}
    write_json_atomic(output / "COMPARISON.json", result)
    return result


def wanted(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe result path")
    if name in {"COMPLETE.json", "EXECUTION_CONTRACT.json"}:
        return True
    if not path.parts or path.parts[0] not in {"smoke", "formal"}:
        return False
    if any(part.startswith("checkpoint-") for part in path.parts):
        return False
    if path.suffix in {".json", ".jsonl"}:
        return path.name not in {"tokenizer.json", "tokenizer_config.json"}
    return path.name in {"adapter_model.safetensors", "interest_relation.safetensors"}


def download_artifact(blob, destination: Path, prior: dict) -> dict:
    from azure.core import MatchConditions

    properties = blob.get_blob_properties()
    if (destination.is_file() and prior.get("etag") == properties.etag and
            destination.stat().st_size == properties.size and sha256_file(destination) == prior.get("sha256")):
        return prior
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    with temporary.open("wb") as handle:
        blob.download_blob(etag=properties.etag, match_condition=MatchConditions.IfNotModified,
                           max_concurrency=4).readinto(handle)
    if temporary.stat().st_size != properties.size:
        raise ValueError("download size mismatch")
    temporary.replace(destination)
    return {"bytes": properties.size, "etag": properties.etag, "sha256": sha256_file(destination)}


def collect(experiment: str, output: Path, amlt: str) -> dict:
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobClient

    prepared = json.loads(PREFLIGHT.read_text())
    if prepared["code_sha256"] != sources():
        raise ValueError("current audit code no longer matches the frozen experiment")
    output.mkdir(parents=True, exist_ok=True)
    credential = AzureCliCredential()
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    progress_path = output / "DOWNLOAD_PROGRESS.json"
    downloads = (json.loads(receipt_path.read_text()) if receipt_path.exists() else
                 json.loads(progress_path.read_text())["files"] if progress_path.exists() else {})
    audits = {}
    for arm in ARMS:
        response = subprocess.run([amlt, "--json-tables", "results", "list", experiment, f":{arm}", "-u"],
                                  env=dict(os.environ, AMLT_NO_BATCH_STATUS_API="1"),
                                  capture_output=True, text=True, check=True, timeout=180)
        inventory = json.JSONDecoder().raw_decode(response.stdout.lstrip())[0]
        entries = [entry for entry in inventory if wanted(entry["name"])]
        if not {"COMPLETE.json", "formal/RESULT_AUDIT.json", "smoke/RESULT_AUDIT.json"} <= {entry["name"] for entry in entries}:
            raise ValueError(f"{arm}: full training/evaluation artifacts are not complete")
        directory = output / arm
        for entry in entries:
            destination = directory / entry["name"]
            blob = BlobClient.from_blob_url(entry["url"], credential=credential, connection_timeout=15,
                                            read_timeout=60, retry_total=3)
            key = f"{arm}/{entry['name']}"
            downloads[key] = download_artifact(blob, destination, downloads.get(key, {}))
            write_json_atomic(receipt_path, downloads)
        execution = json.loads((directory / "EXECUTION_CONTRACT.json").read_text())
        if any(execution.get(key) != value for key, value in prepared.items()) or execution["arm"] != arm:
            raise ValueError("cloud execution differs from the frozen preflight")
        arm_audits = {}
        for smoke in (True, False):
            source = directory / ("smoke" if smoke else "formal")
            with TemporaryDirectory() as temporary:
                isolated = Path(temporary)
                stages = {}
                for stage in (("novel",) if arm == "structure" else ("auxiliary", "novel")):
                    stage_dir = isolated / stage
                    stage_dir.mkdir()
                    for path in (source / stage).iterdir():
                        if path.name != "STAGE_AUDIT.json":
                            (stage_dir / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
                    stages[stage] = audit_training(stage_dir, arm, stage, smoke, prepared, execution["model_path"])
                    if stages[stage] != json.loads((source / stage / "STAGE_AUDIT.json").read_text()):
                        raise ValueError("cloud and local training audits disagree")
                for evaluation in ("classification", "ranking"):
                    (isolated / evaluation).symlink_to((source / evaluation).resolve(), target_is_directory=True)
                audit = audit_evaluation(isolated, arm, smoke, stages["novel"])
                audit["stage_audits"] = stages
                if audit != json.loads((source / "RESULT_AUDIT.json").read_text()):
                    raise ValueError("cloud and local raw result audits disagree")
                arm_audits["smoke" if smoke else "formal"] = audit
        completion = json.loads((directory / "COMPLETE.json").read_text())
        for phase in ("smoke", "formal"):
            if completion[f"{phase}_audit_sha256"] != sha256_file(directory / phase / "RESULT_AUDIT.json"):
                raise ValueError("cloud completion receipt does not identify downloaded audits")
        audits[arm] = arm_audits
        print(f"VERIFIED {arm}: complete training, smoke and original Dev2000 raw metrics", flush=True)
        write_json_atomic(output / "DOWNLOAD_PROGRESS.json", {"files": downloads, "verified_arms": list(audits)})
    result = {"passed": True, "experiment": experiment, "observed_utc": datetime.now(timezone.utc).isoformat(),
              "preflight_sha256": sha256_file(PREFLIGHT), "audits": audits, "downloads": downloads}
    write_json_atomic(output / "RAW_RESULT_AUDIT.json", result)
    compare_results(output, result)
    print(json.dumps({arm: audits[arm]["formal"]["results"]["ranking"] for arm in ARMS}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="home112-bi-mechanism-sft-20260915")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/results")
    parser.add_argument("--amlt", default="amlt")
    args = parser.parse_args()
    collect(args.experiment, args.output_dir.resolve(), args.amlt)


if __name__ == "__main__":
    main()