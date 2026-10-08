"""Download factorization artifacts and audit without overwriting cloud receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import HERE, BASE, sha256_file
import crossinterest.methods.cir.runner as cir
from crossinterest.methods.factorization.runner import ROOT, audit_stage, preflight, sources, verify_package
from crossinterest.methods.factorization.probe import macro_statistics, probe_rows
from crossinterest.methods.mechanism.audit import download_artifact, wanted


JOBS = {"qwen-factor-g1": "relation_true", "qwen-factor-g2": "relation_pseudo",
        "qwen-factor-g3": "exclusion_true", "qwen-factor-g4": "exclusion_pseudo"}
REFERENCE = "qwen-factor-g0-probes"
PROBE_STATISTIC_ATOL = 1e-12


def audit_replaced_job_termination(row, manifest_observed_utc, lookup):
    from azure.core.exceptions import ResourceNotFoundError

    identifier = row["previous_job_id"]
    status = row["previous_job_status"]
    ended = datetime.fromisoformat(row["previous_job_end_utc"])
    observed = datetime.fromisoformat(manifest_observed_utc)
    if (identifier in {row["old_job_id"], row["new_job_id"]}
            or status not in {"Canceled", "Cancelled", "Failed"}
            or ended.tzinfo is None or observed.tzinfo is None or ended >= observed):
        raise ValueError("invalid sealed replacement termination")
    try:
        details = lookup(identifier)
    except ResourceNotFoundError:
        source = "sealed_pre_retry_termination_after_run_removal"
    else:
        if details.status != status or details.end_time_utc != ended:
            raise ValueError("replacement termination differs from sealed manifest")
        source = "live_run_history"
    return {"passed": True, "job_id": identifier, "status": status,
            "end_utc": ended.isoformat(), "evidence_source": source}


def audit_metadata_continuation(text, runtime, checkpoint_step):
    from crossinterest.methods.factorization.resume import validate_layout

    steps = [int(value) for value in re.findall(r"(\d+)/7866", text)
             if int(value) > checkpoint_step]
    if not steps:
        return None
    decoder, ranks = json.JSONDecoder(), set()
    for match in re.finditer(r'\{"event"\s*:\s*"unique_metadata_temporary_files_enabled"', text):
        event = decoder.raw_decode(text, match.start())[0]
        ranks.add(event["rank"])
    if ranks != set(range(8)):
        raise ValueError("metadata startup rank coverage is incomplete")
    validate_layout(runtime, 8, False)
    return {"passed": True, "scope": "resumption_only_not_final_results",
            "checkpoint_step": checkpoint_step, "first_observed_positive_step": min(steps),
            "latest_step": max(steps), "writer_ranks": sorted(ranks)}


def performance_contract(directory, arm):
    path = directory / "PERFORMANCE_CONTRACT.json"
    if not path.exists():
        return None
    from crossinterest.methods.factorization.resume import validate_benchmark, validate_layout, validate_stress_gate

    result = json.loads(path.read_text())
    expected = {"arm": arm, "world_size": 8, "effective_batch_size": 32, "budget_unchanged": True,
                "bitwise_trajectory_equivalence_claimed": False,
                "execution_code_sha256": sha256_file(Path(__file__).with_name("cir_factor_resume.py")),
                "parallel_command_sha256": sha256_file(Path(__file__).with_name("cir_factor_speed.py")),
                "original_execution_sha256": sha256_file(directory / "EXECUTION_CONTRACT.json"),
                "original_smoke_sha256": sha256_file(directory / "smoke/RESULT_AUDIT.json"),
                "benchmark_sha256": sha256_file(directory / "PERFORMANCE_BENCHMARK.json"),
                "resume_audit_sha256": sha256_file(directory / "RESUME_AUDIT.json")}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("accelerated execution identity differs")
    benchmark = json.loads((directory / "PERFORMANCE_BENCHMARK.json").read_text())
    if validate_benchmark(benchmark, result["world_size"], result["gradient_checkpointing"]) != result["benchmark_gate"]:
        raise ValueError("accelerated throughput audit differs")
    authorization = json.loads((directory / "HANDOFF_AUTHORIZATION.json").read_text())
    if (authorization.get("original_job_stopped") is not True
            or authorization.get("performance_contract_sha256") != sha256_file(path)):
        raise ValueError("exclusive old-job handoff was not authorized")
    migration = json.loads((directory / "RESUME_AUDIT.json").read_text())
    if not migration["passed"] or migration["world_size_before"] != 4 or migration["world_size_after"] != 8:
        raise ValueError("invalid checkpoint migration")
    readiness = json.loads((directory / "READY_FOR_HANDOFF.json").read_text())
    if (readiness.get("passed") is not True or readiness.get("optimizer_updates") != 0
            or readiness.get("checkpoint_step") != migration["checkpoint_step"]
            or readiness.get("performance_contract_sha256") != sha256_file(path)):
        raise ValueError("pre-handoff GPU validation is incomplete")
    validate_stress_gate(readiness["stress"], migration["checkpoint_step"], result["world_size"])
    for rank in range(4):
        if sha256_file(directory / "ORIGINAL_RNG" / f"rng_state_{rank}.pth") != migration["original_sha256"][f"rng_state_{rank}.pth"]:
            raise ValueError("original checkpoint random state changed")
    for stage in ("auxiliary", "novel"):
        validate_layout(json.loads((directory / "formal" / stage / "runtime_config.json").read_text()),
                        result["world_size"], result["gradient_checkpointing"])
        for rank in range(8):
            stress = json.loads((directory / "formal" / stage / f"STRESS_RANK_{rank}.json").read_text())
            if (stress.get("passed") is not True or stress.get("tokens") != 3252
                    or stress.get("optimizer_updates") != 0
                    or stress.get("tested_objectives") != ["stage_loss", "original_full_vocabulary_causal_ce"]):
                raise ValueError("worst-length GPU stress gate is incomplete")
    return result


def require_probe_statistics(actual, expected, field="root"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"probe fields differ: {field}")
        for name, value in expected.items():
            require_probe_statistics(actual[name], value, name)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"probe record count differs: {field}")
        for observed, value in zip(actual, expected):
            require_probe_statistics(observed, value, field)
    elif field in {"conditional_nll", "history_advantage", "seen_mass", "target_nll_112"}:
        if (type(actual) is not float or not math.isfinite(actual) or not math.isfinite(expected)
                or not math.isclose(actual, expected, rel_tol=0, abs_tol=PROBE_STATISTIC_ATOL)):
            raise ValueError(f"probe statistic differs: {field}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"probe identity differs: {field}")


def audit_probe(output: Path, adapter: Path, limit=None):
    import numpy as np
    from scipy.special import logsumexp, softmax
    from crossinterest.training.label_space import load_label_mapping

    result = json.loads(output.read_text())
    identity = {"adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "probe_sha256": sha256_file(ROOT / "data/probe.parquet"), "limit": limit}
    if result.get("passed") is not True or any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("probe output identity changed")
    raw = output.with_suffix(".npz")
    if sha256_file(raw) != result["logits_sha256"]:
        raise ValueError("probe raw logits changed")
    mapping = load_label_mapping(BASE / "interest_id_semantic_mapping.json", expected_count=112)
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    indices = {category: index for index, category in enumerate(categories)}
    all_rows, rows = probe_rows(limit)
    with np.load(raw, allow_pickle=False) as archive:
        if set(archive.files) != {"correct", "donor", "novel"}:
            raise ValueError("probe raw array fields differ")
        arrays = {name: archive[name].astype(np.float64) for name in archive.files}
    if any(value.shape != (len(rows), 112) for value in arrays.values()):
        raise ValueError("probe raw array shape differs")
    records, novel_records = [], []
    for index, row in enumerate(rows):
        donor = all_rows[row["donor_index"]]
        seen = np.array([category in row["historical_interests"] for category in categories])
        donor_seen = np.array([category in donor["historical_interests"] for category in categories])
        common, target = ~(seen | donor_seen), indices[row["target_interest"]]
        scores, other = arrays["correct"][index], arrays["donor"][index]
        if seen[target] or donor_seen[target] or not np.isfinite(scores).all() or not np.isfinite(other).all():
            raise ValueError("invalid raw probe target or logits")
        records.append({"id": row["id"], "target_interest": row["target_interest"], "donor_id": donor["id"],
            "conditional_nll": float(logsumexp(scores[~seen]) - scores[target]),
            "history_advantage": float((scores[target] - logsumexp(scores[common]))
                                       - (other[target] - logsumexp(other[common]))),
            "seen_mass": float(softmax(scores)[seen].sum()), "common_unseen_count": int(common.sum())})
        source = row["probe_source"]
        scores = arrays["novel"][index]
        if source["target_interest"] in indices:
            target = indices[source["target_interest"]]
            seen = np.array([category in source["historical_interests"] for category in categories])
            if seen[target] or not np.isfinite(scores).all():
                raise ValueError("invalid original novel-interest probe")
            novel_records.append({"id": source["id"], "target_interest": source["target_interest"],
                "target_nll_112": float(logsumexp(scores) - scores[target]),
                "conditional_nll": float(logsumexp(scores[~seen]) - scores[target]),
                "seen_mass": float(softmax(scores)[seen].sum())})
        elif not np.isnan(scores).all():
            raise ValueError("unsupported novel probe must retain its NaN placeholder")
    for field, expected in (("records", records), ("macro", macro_statistics(records)), ("novel_records", novel_records)):
        require_probe_statistics(result[field], expected, field)
    return {"passed": True, "rows": len(rows), "classes": len({row["target_interest"] for row in rows}),
            "macro": result["macro"], "novel_rows": len(novel_records), "sha256": sha256_file(output), **identity}


def audit_output(output: Path, smoke_only=False):
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    prepared = preflight()
    if execution["factor_preflight_sha256"] != sha256_file(ROOT / "PREFLIGHT.json"):
        raise ValueError("output belongs to a different factor preflight")
    arm, phases = execution["factor_arm"], {}
    performance = json.loads((output / "PERFORMANCE_CONTRACT.json").read_text()) if (output / "PERFORMANCE_CONTRACT.json").exists() else None
    for smoke in ((True,) if smoke_only else (True, False)):
        directory = output / ("smoke" if smoke else "formal")
        if performance and not smoke:
            from crossinterest.methods.factorization.resume import audit_stage as audit_accelerated_stage
            stages = {stage: audit_accelerated_stage(directory / stage, stage, prepared["original"],
                execution["model_path"], arm, performance) for stage in ("auxiliary", "novel")}
        else:
            stages = {stage: audit_stage(directory / stage, stage, smoke, prepared["original"], execution["model_path"], arm)
                      for stage in ("auxiliary", "novel")}
        probes = {stage: audit_probe(directory / f"{stage}_PROBE.json", directory / stage / "final_adapter",
                                    4 if smoke else None) for stage in ("auxiliary", "novel")}
        result = cir.audit_evaluation(directory, "reconstruction", smoke, stages["novel"])
        result.update(factor_arm=arm, stage_audits=stages, probes=probes)
        if performance and not smoke:
            result["performance_contract_sha256"] = sha256_file(output / "PERFORMANCE_CONTRACT.json")
        write_json_atomic(directory / "RESULT_AUDIT.json", result)
        phases[directory.name] = result
    return {"passed": True, "arm": arm, "phases": phases}


def select_inventory(inventory, smoke_only=False, reference=False):
    phases = ("smoke",) if smoke_only else ("smoke", "formal")
    if reference:
        required = {"REFERENCE_PROBE_AUDIT.json"}
        required.update(f"{stage}_PROBE{suffix}" for stage in ("aux3", "novel") for suffix in (".json", ".npz"))
    else:
        required = {"EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json"}
        required.update(f"{phase}/RESULT_AUDIT.json" for phase in phases)
        required.update(f"{phase}/{stage}_PROBE{suffix}" for phase in phases
                        for stage in ("auxiliary", "novel") for suffix in (".json", ".npz"))
        if not smoke_only:
            required.add("COMPLETE.json")
    if not smoke_only and not reference and any(entry["name"] == "PERFORMANCE_CONTRACT.json" for entry in inventory):
        required.update({"PERFORMANCE_CONTRACT.json", "PERFORMANCE_BENCHMARK.json", "RESUME_AUDIT.json",
                 "HANDOFF_AUTHORIZATION.json", "READY_FOR_HANDOFF.json"})
        required.update(f"ORIGINAL_RNG/rng_state_{rank}.pth" for rank in range(4))
        required.update(f"formal/{stage}/STRESS_RANK_{rank}.json" for stage in ("auxiliary", "novel") for rank in range(8))
    entries = []
    for entry in inventory:
        name = entry["name"]
        original = wanted(name)
        if name in required or (not reference and original and name.split("/")[0] in phases):
            entries.append(entry)
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names) or not required <= set(names):
        raise ValueError("factor phase artifacts are incomplete or duplicated")
    return entries


def audit_download(directory: Path, arm: str, smoke_only=False):
    verify_package()
    prepared = preflight()
    execution = json.loads((directory / "EXECUTION_CONTRACT.json").read_text())
    expected = {"factor_arm": arm, "factor_preflight_sha256": sha256_file(ROOT / "PREFLIGHT.json"),
                "code_sha256": sources(), "data_contract": prepared["original"], "fresh_formal_after_smoke": True}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError("downloaded factor execution identity differs")
    performance = performance_contract(directory, arm) if not smoke_only else None
    trainer = json.loads((directory / "TRAINER_EQUIVALENCE.json").read_text())
    if (not trainer["passed"] or trainer["code_sha256"] != sources()
            or trainer["test_sha256"] != sha256_file(HERE / "test_cir_factorization.py")):
        raise ValueError("cloud Trainer equivalence identity differs")
    phases = ("smoke",) if smoke_only else ("smoke", "formal")
    cloud = {phase: json.loads((directory / phase / "RESULT_AUDIT.json").read_text()) for phase in phases}
    if not smoke_only:
        completion = json.loads((directory / "COMPLETE.json").read_text())
        if completion.get("passed") is not True or completion.get("arm") != arm or completion.get("smoke_only") is not False:
            raise ValueError("factor completion identity differs")
        for phase in phases:
            if completion[f"{phase}_audit_sha256"] != sha256_file(directory / phase / "RESULT_AUDIT.json"):
                raise ValueError("cloud factor completion audit hash differs")
        if performance and completion.get("performance_contract_sha256") != sha256_file(directory / "PERFORMANCE_CONTRACT.json"):
            raise ValueError("accelerated completion contract differs")
    with TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if path.is_dir() or relative.parts[0] not in (*phases, "EXECUTION_CONTRACT.json", "PERFORMANCE_CONTRACT.json"):
                continue
            target = isolated / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.name not in {"STAGE_AUDIT.json", "FACTOR_STAGE_AUDIT.json", "RESULT_AUDIT.json"}:
                target.symlink_to(path.resolve())
        result = audit_output(isolated, smoke_only)
        if result["phases"] != cloud:
            raise ValueError("cloud and independent factor raw audits disagree")
        for phase in phases:
            for stage in ("auxiliary", "novel"):
                for name in ("STAGE_AUDIT.json", "FACTOR_STAGE_AUDIT.json"):
                    relative = Path(phase) / stage / name
                    if json.loads((directory / relative).read_text()) != json.loads((isolated / relative).read_text()):
                        raise ValueError(f"cloud and independent stage audit disagree: {relative}")
    return {**result, "trainer_equivalence": trainer}


def audit_reference(directory: Path):
    verify_package()
    cloud = json.loads((directory / "REFERENCE_PROBE_AUDIT.json").read_text())
    result = {"passed": True, "training_performed": False,
              "audits": {stage: audit_probe(directory / f"{stage}_PROBE.json", ROOT / "reference_adapters" / stage)
                         for stage in ("aux3", "novel")}}
    if result != cloud:
        raise ValueError("G0 cloud and independent reference probes disagree")
    return result


def collect(experiment: str, output: Path, amlt: str, jobs: list[str], smoke_only=False):
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobClient

    if not jobs or len(set(jobs)) != len(jobs) or any(job not in (*JOBS, REFERENCE) for job in jobs):
        raise ValueError("unique known factor jobs required")
    credential = AzureCliCredential(process_timeout=30)
    receipt_path = output / "DOWNLOAD_RECEIPT.json"
    downloads = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    audits = {}
    for job in jobs:
        response = subprocess.run([amlt, "--json-tables", "results", "list", experiment, f":{job}", "-u"],
            env=dict(os.environ, AMLT_NO_BATCH_STATUS_API="1"), capture_output=True, text=True, check=True, timeout=180)
        inventory = json.JSONDecoder().raw_decode(response.stdout.lstrip())[0]
        entries = select_inventory(inventory, smoke_only, job == REFERENCE)
        directory = output / job
        for entry in entries:
            key = f"{job}/{entry['name']}"
            blob = BlobClient.from_blob_url(entry["url"], credential=credential,
                                           connection_timeout=15, read_timeout=60, retry_total=3)
            downloads[key] = download_artifact(blob, directory / entry["name"], downloads.get(key, {}))
            write_json_atomic(receipt_path, downloads)
        audits[job] = audit_reference(directory) if job == REFERENCE else audit_download(directory, JOBS[job], smoke_only)
        print(json.dumps({"event": "independent_factor_audit_passed", "job": job, "smoke_only": smoke_only}), flush=True)
    result = {"passed": True, "experiment": experiment, "smoke_only": smoke_only, "audits": audits,
              "observed_utc": datetime.now(timezone.utc).isoformat(), "collector_sha256": sha256_file(Path(__file__)),
              "probe_validation": {"implementation": "independent_scipy_float64", "statistic_rtol": 0,
                                   "statistic_atol": PROBE_STATISTIC_ATOL, "identity_and_sha_exact": True}}
    name = "GPU_SMOKE_AUDIT.json" if smoke_only else "RAW_RESULT_AUDIT.json"
    write_json_atomic(output / name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amlt", default="amlt")
    parser.add_argument("--job", action="append", choices=(*JOBS, REFERENCE), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    collect(arguments.experiment, arguments.output_dir.resolve(), arguments.amlt, arguments.job, arguments.smoke_only)