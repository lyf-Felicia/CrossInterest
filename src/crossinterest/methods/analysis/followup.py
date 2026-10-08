"""Diagnose frozen full-novel results without training or changing rankings."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import get_cosine_schedule_with_warmup

from crossinterest.evaluation.common import latest_details, write_json_atomic
from crossinterest.training.evaluate_ranking import metrics_from_rankings, score_ranking
from crossinterest.methods.alignment.data import BASE, HERE, sha256_file


METRICS = tuple(f"macro_{metric}_at_{cutoff}" for metric in ("recall", "ndcg")
                for cutoff in (10, 20))
DATA = HERE / "curriculum/seen500-novel112k"
FULL_RESULTS = HERE / (
    "curriculum/outputs/fullnovel_results/"
    "home112-bi-curriculum-fullnovel-sft-20260914"
)
RUNS = {
    "n_only": HERE / (
        "backbone/outputs/progress/home112-bi-backbone-sft-20260911/"
        "qwen35-9b/formal/ranking"
    ),
    "seen_then_novel": FULL_RESULTS / "seen-then-novel-full112k/formal/ranking",
    "shuffle": FULL_RESULTS / "shuffle-full112k/formal/ranking",
    "crossfit": HERE / (
        "crossfit/student/results/home112-bi-crossfit-weighted-student-20260912/"
        "qwen35-weighted/formal/ranking"
    ),
}


def metric_values(details: list[dict], rows: list[dict]) -> dict:
    result = metrics_from_rankings(details, rows, 20)
    return {key: result[key] for key in METRICS}


def rank_of(detail: dict) -> int:
    return next((entry["rank"] for entry in detail["ranking"]
                 if entry["prediction"] == detail["target_interest"]), 21)


def rank_band(rank: int) -> int:
    return 0 if rank <= 10 else 1 if rank <= 20 else 2


def paired_ranking_diagnostic(before: list[dict], after: list[dict],
                              rows: list[dict]) -> dict:
    if not rows or len(before) != len(rows) or len(after) != len(rows):
        raise ValueError("Paired diagnostics require complete aligned rows")
    class_counts = Counter(row["target_interest"] for row in rows)
    delta = Counter({key: 0.0 for key in METRICS})
    components = {name: {"samples": 0, "macro_mass": 0.0, "ndcg20_delta": 0.0}
                  for name in ("both_hit", "gained_hit", "lost_hit", "both_miss")}
    history_results = {}
    for history in ("historical_interests", "full_seen_interests"):
        groups = {name: {"samples": 0, "macro_mass": 0.0,
                         "delta": dict.fromkeys(METRICS, 0.0),
                         "gained_hits": 0, "lost_hits": 0}
                  for name in ("neither_has_seen", "before_only_has_seen",
                               "after_only_has_seen", "both_have_seen")}
        history_results[history] = {
            "groups": groups, "before_slot_fraction20": 0.0,
            "after_slot_fraction20": 0.0,
            "both_hit_rank_change": {"macro_mass": 0.0, "rank_delta": 0.0,
                                     "seen_before_target_delta": 0.0,
                                     "unseen_before_target_delta": 0.0},
        }
    per_class = {label: dict.fromkeys(METRICS, 0.0) for label in class_counts}
    for original, updated, row in zip(before, after, rows):
        for detail in (original, updated):
            if any(detail[field] != row[field]
                   for field in ("sample_id", "source_sample_id", "target_interest")):
                raise ValueError("Paired sample identity mismatch")
            entries = detail["ranking"]
            if (len(entries) != 20 or [entry["rank"] for entry in entries] != list(range(1, 21))
                    or len({entry["prediction"] for entry in entries}) != 20):
                raise ValueError("Expected 20 unique ordered predictions")
        weight = 1 / (len(class_counts) * class_counts[row["target_interest"]])
        original_scores = score_ranking(original["ranking"], row["target_interest"])
        updated_scores = score_ranking(updated["ranking"], row["target_interest"])
        differences = {key: updated_scores[key.removeprefix("macro_")] -
                       original_scores[key.removeprefix("macro_")] for key in METRICS}
        for key, difference in differences.items():
            delta[key] += weight * difference
            per_class[row["target_interest"]][key] += difference / class_counts[row["target_interest"]]
        before_rank, after_rank = rank_of(original), rank_of(updated)
        component = ("both_hit" if before_rank <= 20 and after_rank <= 20 else
                     "gained_hit" if after_rank <= 20 else
                     "lost_hit" if before_rank <= 20 else "both_miss")
        components[component]["samples"] += 1
        components[component]["macro_mass"] += weight
        components[component]["ndcg20_delta"] += weight * differences["macro_ndcg_at_20"]
        for history, result in history_results.items():
            seen = set(row[history])
            if row["target_interest"] in seen:
                raise ValueError("Novel target occurs in history")
            before_seen = sum(entry["prediction"] in seen for entry in original["ranking"])
            after_seen = sum(entry["prediction"] in seen for entry in updated["ranking"])
            group = ("both_have_seen" if before_seen and after_seen else
                     "before_only_has_seen" if before_seen else
                     "after_only_has_seen" if after_seen else "neither_has_seen")
            bucket = result["groups"][group]
            bucket["samples"] += 1
            bucket["macro_mass"] += weight
            bucket["gained_hits"] += int(component == "gained_hit")
            bucket["lost_hits"] += int(component == "lost_hit")
            for key, difference in differences.items():
                bucket["delta"][key] += weight * difference
            result["before_slot_fraction20"] += weight * before_seen / 20
            result["after_slot_fraction20"] += weight * after_seen / 20
            if component == "both_hit":
                movement = result["both_hit_rank_change"]
                seen_change = (sum(entry["prediction"] in seen for entry in updated["ranking"][:after_rank - 1]) -
                               sum(entry["prediction"] in seen for entry in original["ranking"][:before_rank - 1]))
                movement["macro_mass"] += weight
                movement["rank_delta"] += weight * (after_rank - before_rank)
                movement["seen_before_target_delta"] += weight * seen_change
                movement["unseen_before_target_delta"] += weight * (after_rank - before_rank - seen_change)
    assert math.isclose(sum(item["ndcg20_delta"] for item in components.values()),
                        delta["macro_ndcg_at_20"], abs_tol=1e-12)
    for key in METRICS:
        assert math.isclose(sum(item[key] for item in per_class.values()) / len(class_counts),
                            delta[key], abs_tol=1e-12)
        for result in history_results.values():
            assert math.isclose(sum(group["delta"][key] for group in result["groups"].values()),
                                delta[key], abs_tol=1e-12)
    return {"samples": len(rows), "classes": len(class_counts), "delta": dict(delta),
            "hit_components": components, "history": history_results,
            "per_class_delta": per_class,
            "interpretation": "Descriptive paired decomposition, not causal mediation; full Macro denominator preserved; outside-top20 ranks censored; original rankings unchanged."}


def normalized_lr(step: int, total: int) -> float:
    warmup = math.ceil(0.03 * total)
    if step < warmup:
        return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def schedule_summary(tasks: list[str]) -> dict:
    total = math.ceil(len(tasks) / 32)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(0.03 * total), total)
    assert all(math.isclose(normalized_lr(step, total), scheduler.lr_lambdas[0](step),
                            abs_tol=1e-14) for step in range(total))
    counts = Counter(tasks)
    exposure = Counter()
    for index, task in enumerate(tasks):
        exposure[task] += normalized_lr(index // 32, total)
    first_novel = tasks.index("novel")
    return {
        "steps": total,
        "warmup_steps": math.ceil(0.03 * total),
        "task_counts": dict(counts),
        "task_mean_lr_over_peak": {task: exposure[task] / count
                                   for task, count in counts.items()},
        "first_novel_update_1based": first_novel // 32 + 1,
        "first_novel_lr_over_peak": normalized_lr(first_novel // 32, total),
    }


def main() -> None:
    dev_path = BASE / "eval/dev_2k_behavior_interest.parquet"
    manifest_path = DATA / "MANIFEST.json"
    audit_path = HERE / "curriculum/outputs/FULLNOVEL_RAW_AUDIT_20260915.json"
    historic_path = HERE / "outputs/RAW_AUDIT_20260912.json"
    weights_path = HERE / "crossfit/weights/WEIGHT_AUDIT.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    historic = json.loads(historic_path.read_text())
    weights = json.loads(weights_path.read_text())
    assert audit["passed"] and sha256_file(manifest_path) == audit["manifest_sha256"]
    assert weights["passed"] and weights["rows"] == 112000
    rows = pq.read_table(dev_path).to_pylist()
    labels = {row["target_interest"] for row in rows}
    seen_labels = set(manifest["seen_labels"])
    assert len(rows) == 2000 and len(labels) == 112 and len(seen_labels) == 93
    assert seen_labels <= labels and set(weights["class_statistics"]) == labels
    assert all(row["target_interest"] not in row["full_seen_interests"] for row in rows)
    assert all(set(row["historical_interests"]) <= set(row["full_seen_interests"])
               for row in rows)
    expected = {
        "n_only": audit["baseline"]["metrics"],
        "crossfit": historic["crossfit/qwen35-weighted"]["metrics"],
        **{arm: audit["jobs"][arm]["formal"]["results"]["ranking"]
           for arm in ("seen_then_novel", "shuffle")},
    }
    provenance = {str(path.relative_to(HERE.parent)): sha256_file(path)
                  for path in (dev_path, manifest_path, audit_path, historic_path, weights_path,
                               Path(metrics_from_rankings.__code__.co_filename),
                               Path(latest_details.__code__.co_filename))}
    expected_raw_sha = {
        "n_only": audit["baseline"]["raw_sha256"],
        "crossfit": historic["crossfit/qwen35-weighted"]["raw_sha256"],
        **{arm: audit["jobs"][arm]["raw_sha256"]["ranking"]
           for arm in ("seen_then_novel", "shuffle")},
    }
    results = {}
    raw = {}
    for name, directory in RUNS.items():
        contract_path = directory / "run_contract.json"
        details_path = directory / "details.jsonl"
        assert sha256_file(details_path) == expected_raw_sha[name], name
        if name in ("n_only", "crossfit"):
            receipt = audit["baseline"] if name == "n_only" else historic["crossfit/qwen35-weighted"]
            assert receipt["passed"] and sha256_file(contract_path) == receipt["contract_sha256"]
        contract = json.loads(contract_path.read_text())
        assert contract["dataset_sha256"] == sha256_file(dev_path)
        assert contract["decoding"] == "candidate_free_direct_beam_search"
        assert contract["num_beams"] == 20 and contract["do_sample"] is False
        assert contract["decoder_label_constraint"] is False
        assert contract["candidates_in_prompt"] is False
        details = latest_details(details_path, contract["fingerprint"])
        assert set(details) == {row["sample_id"] for row in rows}
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            for field in ("source_sample_id", "target_interest"):
                assert detail[field] == row[field]
            entries = detail["ranking"]
            assert len(entries) == 20
            assert [entry["rank"] for entry in entries] == list(range(1, 21))
            assert len({entry["prediction"] for entry in entries}) == 20
            assert all(entry["prediction"] in labels and entry["format_valid"]
                       for entry in entries)
            detail.update(score_ranking(entries, row["target_interest"]))
            ordered.append(detail)
        raw[name] = ordered
        metrics = metric_values(ordered, rows)
        assert metrics == {key: expected[name][key] for key in METRICS}, name
        groups = {}
        for group, selected_labels in (("o_supported_93", seen_labels),
                                       ("n_only_supported_19", labels - seen_labels)):
            indices = [index for index, row in enumerate(rows)
                       if row["target_interest"] in selected_labels]
            groups[group] = {
                "samples": len(indices), "classes": len(selected_labels),
                **metric_values([ordered[index] for index in indices],
                                [rows[index] for index in indices]),
            }
        for key in METRICS:
            reconstructed = sum(group["classes"] * group[key]
                                for group in groups.values()) / 112
            assert math.isclose(reconstructed, metrics[key], abs_tol=1e-12)
        occupancy = {}
        for cutoff in (10, 20):
            for history in ("historical_interests", "full_seen_interests"):
                occupied = sum(entry["prediction"] in row[history]
                               for detail, row in zip(ordered, rows)
                               for entry in detail["ranking"][:cutoff])
                occupancy[f"{history}_slot_fraction_at_{cutoff}"] = occupied / (len(rows) * cutoff)
        results[name] = {"metrics": metrics, "groups": groups,
                         "seen_slot_occupancy_micro": occupancy}
        for path in (contract_path, details_path):
            provenance[str(path.relative_to(HERE.parent))] = sha256_file(path)
    class_counts = Counter(row["target_interest"] for row in rows)
    for name, ordered in raw.items():
        if name == "n_only":
            continue
        counts = [[0] * 3 for _ in range(3)]
        mass = [[0.0] * 3 for _ in range(3)]
        ndcg_contributions = Counter()
        for original, updated, row in zip(raw["n_only"], ordered, rows):
            before, after = rank_band(rank_of(original)), rank_band(rank_of(updated))
            counts[before][after] += 1
            sample_mass = 1 / (112 * class_counts[row["target_interest"]])
            mass[before][after] += sample_mass
            component = ("both_hit" if before < 2 and after < 2 else
                         "gained_hit" if after < 2 else "lost_hit" if before < 2 else "both_miss")
            ndcg_contributions[component] += sample_mass * (updated["ndcg_at_20"] - original["ndcg_at_20"])
        ndcg_delta = results[name]["metrics"]["macro_ndcg_at_20"] - results["n_only"]["metrics"]["macro_ndcg_at_20"]
        assert math.isclose(sum(ndcg_contributions.values()), ndcg_delta, abs_tol=1e-12)
        results[name]["macro_ndcg_at_20_delta_components"] = dict(ndcg_contributions)
        for cutoff, band_limit in ((10, 1), (20, 2)):
            gained = sum(mass[before][after] for before in range(band_limit, 3)
                         for after in range(band_limit))
            lost = sum(mass[before][after] for before in range(band_limit)
                       for after in range(band_limit, 3))
            delta = results[name]["metrics"][f"macro_recall_at_{cutoff}"] - results["n_only"]["metrics"][f"macro_recall_at_{cutoff}"]
            assert math.isclose(gained - lost, delta, abs_tol=1e-12)
        results[name]["rank_band_transition_vs_n_only"] = {
            "rows": "n_only", "columns": name,
            "bands": ["1-10", "11-20", "outside_top20_censored"],
            "sample_counts": counts, "macro_mass": mass,
        }
    schedules = {"n_only": schedule_summary(["novel"] * 112000)}
    for arm in ("seen_then_novel", "shuffle"):
        path = DATA / f"{arm}.parquet"
        assert sha256_file(path) == manifest["files"][arm]["sha256"]
        tasks = pq.read_table(path, columns=["task_type"])["task_type"].to_pylist()
        assert Counter(tasks) == {"seen": 46500, "novel": 112000}
        schedules[arm] = schedule_summary(tasks)
        provenance[str(path.relative_to(HERE.parent))] = sha256_file(path)
    weight_groups = {}
    for group, selected_labels in (("o_supported_93", seen_labels),
                                   ("n_only_supported_19", labels - seen_labels)):
        statistics = [weights["class_statistics"][label] for label in sorted(selected_labels)]
        assert all(stat["rows"] == 1000 and math.isclose(stat["mean"], 1.0)
                   for stat in statistics)
        weight_groups[group] = {
            "classes": len(statistics), "rows_per_class": 1000,
            "mean_class_effective_sample_size": sum(stat["effective_sample_size"]
                                                    for stat in statistics) / len(statistics),
        }
    output = {
        "passed": True, "scope": "post_hoc_diagnostic_not_new_training_or_evaluation",
        "source_sha256": provenance, "analyzer_sha256": sha256_file(Path(__file__)),
        "conditions": results, "schedules": schedules, "crossfit_weight_groups": weight_groups,
        "schedule_caveat": "Analytic row-weighted scheduled LR, not gradient influence; step 0 is the first optimizer update LR; effective batch 32 and no padding.",
        "seen_target_absent_from_visible_history": manifest["seen_target_absent_from_visible_history"],
        "limitations": ["Single seed and reused Dev; no causal or significance claim.",
                        "Seen-slot fractions are micro diagnostics, not primary Macro metrics.",
                        "Outside-top20 ranks are censored; rankings are never altered."],
    }
    path = HERE / "curriculum/outputs/FULLNOVEL_DIAGNOSTICS_20260915.json"
    write_json_atomic(path, output)
    print(f"PASS: {len(results)} conditions, 2000 samples each; metrics, coverage, groups and transitions verified.")
    print(json.dumps({"conditions": results, "schedules": schedules,
                      "crossfit_weight_groups": weight_groups}, indent=2))


def mechanism_main() -> dict:
    import numpy as np
    from scipy.special import logsumexp

    root = _source_workspace('eval_interest_id_minionerec')
    dev_path = root.parent / 'crossinterest.training/data/eval/dev_2k_behavior_interest.parquet'
    mapping_path = root.parent / 'crossinterest.training/data/interest_id_semantic_mapping.json'
    rows = pq.read_table(dev_path).to_pylist()
    labels = {row["target_interest"] for row in rows}
    assert len(rows) == 2000 and len(labels) == 112
    provenance = {}

    def read_source(path):
        provenance[str(path.relative_to(root.parent))] = sha256_file(path)
        return json.loads(path.read_text())

    historical = read_source(root / "curriculum/outputs/FULLNOVEL_DIAGNOSTICS_20260915.json")
    assert historical["passed"]
    support_path = root / "curriculum/seen500-novel112k/MANIFEST.json"
    assert sha256_file(support_path) == historical["source_sha256"][str(support_path.relative_to(root.parent))]
    supported = set(read_source(support_path)["seen_labels"])
    assert len(supported) == 93 and supported < labels
    specifications = {}
    for name, directory in RUNS.items():
        actual = root / directory.relative_to(HERE)
        specifications[name] = (actual, historical["conditions"][name]["metrics"],
            historical["source_sha256"][str((actual / "details.jsonl").relative_to(root.parent))])
    receipts = (
        ("mechanism/outputs/results_v2", ("structure", "reconstruction", "combined")),
        ("cir/outputs/results_base", ("qwen-balanced-cir",)),
        ("cir/outputs/results_semantic", ("qwen-semantic-true", "qwen-semantic-shuffled")),
        ("frequency/outputs/results_v3", ("once", "at_least_five")),
        ("bidirectional/outputs/results_envfix", ("qwen-cir-3ep", "qwen-bidir-forward",
            "qwen-bidir-ordinary", "qwen-bidir-paired", "qwen-bidir-shuffled")),
        ("cir_factorization/outputs/accel_g3_results", ("qwen-factor-g3",)),
    )
    inverse_probes = {}
    for relative, names in receipts:
        directory = root / relative
        receipt = read_source(directory / "RAW_RESULT_AUDIT.json")
        assert receipt["passed"] and not receipt.get("smoke_only", False)
        for name in names:
            record = receipt["audits"][name]
            formal = record.get("phases", record)["formal"]
            assert formal["passed"] and not formal["smoke"]
            specifications[name] = (directory / name / "formal/ranking",
                                   formal["results"]["ranking"], formal["raw_ranking_sha256"])
            if "inverse_probe" in formal:
                inverse_probes[name] = formal["inverse_probe"]
    required = {"dataset_sha256": sha256_file(dev_path), "label_map_sha256": sha256_file(mapping_path),
                "num_beams": 20, "do_sample": False, "view": "behavior_interest",
                "label_mode": "interest_id", "decoder_label_constraint": False,
                "candidates_in_prompt": False, "decoding": "candidate_free_direct_beam_search"}
    provenance.update({str(path.relative_to(root.parent)): sha256_file(path)
                       for path in (dev_path, mapping_path, Path(__file__),
                                    Path(metrics_from_rankings.__code__.co_filename),
                                    Path(latest_details.__code__.co_filename))})
    raw, conditions = {}, {}
    for name, (directory, expected, raw_sha) in specifications.items():
        details_path = directory / "details.jsonl"
        assert sha256_file(details_path) == raw_sha, name
        provenance[str(details_path.relative_to(root.parent))] = raw_sha
        contract = read_source(directory / "run_contract.json")
        assert all(contract.get(key) == value for key, value in required.items()), name
        details = latest_details(details_path, contract["fingerprint"])
        assert set(details) == {row["sample_id"] for row in rows}, name
        ordered = []
        for row in rows:
            detail = dict(details[row["sample_id"]])
            assert all(detail[field] == row[field] for field in
                       ("sample_id", "source_sample_id", "target_interest")), name
            entries = detail["ranking"]
            assert len(entries) == 20 and [entry["rank"] for entry in entries] == list(range(1, 21)), name
            assert len({entry["prediction"] for entry in entries}) == 20, name
            assert all(entry["prediction"] in labels and entry["format_valid"] for entry in entries), name
            detail.update(score_ranking(entries, row["target_interest"]))
            ordered.append(detail)
        metrics = metric_values(ordered, rows)
        assert metrics == {key: expected[key] for key in METRICS}, name
        raw[name] = ordered
        groups = {}
        for group, selected in (("o_supported_93", supported), ("n_only_supported_19", labels - supported)):
            indices = [index for index, row in enumerate(rows) if row["target_interest"] in selected]
            groups[group] = {"samples": len(indices), "classes": len(selected),
                **metric_values([ordered[index] for index in indices], [rows[index] for index in indices])}
        for key in METRICS:
            assert math.isclose(sum(group["classes"] * group[key] for group in groups.values()) / 112,
                                metrics[key], abs_tol=1e-12)
        conditions[name] = {"metrics": metrics, "groups": groups}
    pairs = [("n_only", name) for name in raw if name != "n_only"] + [
        ("qwen-balanced-cir", "qwen-cir-3ep"), ("reconstruction", "combined"),
        ("qwen-balanced-cir", "qwen-semantic-true"),
        ("qwen-semantic-shuffled", "qwen-semantic-true"),
        ("once", "at_least_five"), ("qwen-bidir-forward", "qwen-bidir-ordinary"),
        ("qwen-bidir-shuffled", "qwen-bidir-paired"), ("qwen-cir-3ep", "qwen-factor-g3")]
    comparisons = {}
    for before, after in pairs:
        result = paired_ranking_diagnostic(raw[before], raw[after], rows)
        for key in METRICS:
            assert math.isclose(result["delta"][key], conditions[after]["metrics"][key] -
                                conditions[before]["metrics"][key], abs_tol=1e-12)
        comparisons[f"{after}_vs_{before}"] = result
    probe_path = root / "cir_factorization/data/probe.parquet"
    probe_rows = pq.read_table(probe_path).to_pylist()
    probe_sha = sha256_file(probe_path)
    provenance[str(probe_path.relative_to(root.parent))] = probe_sha
    assert len(probe_rows) == 2220
    mapping = read_source(mapping_path)
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    category_indices = {category: index for index, category in enumerate(categories)}
    reference_dir = root / "cir_factorization/outputs/reference"
    reference = read_source(reference_dir / "RAW_RESULT_AUDIT.json")
    assert reference["passed"] and not reference["smoke_only"]
    reference_audits = reference["audits"]["qwen-factor-g0-probes"]["audits"]
    g3_dir = root / "cir_factorization/outputs/accel_g3_results"
    g3_audits = read_source(g3_dir / "RAW_RESULT_AUDIT.json")["audits"]["qwen-factor-g3"]["phases"]["formal"]["probes"]
    probe_specs = {
        "G0_aux": (reference_dir / "qwen-factor-g0-probes/aux3_PROBE.json", reference_audits["aux3"]),
        "G0_N": (reference_dir / "qwen-factor-g0-probes/novel_PROBE.json", reference_audits["novel"]),
        **{f"G3_{'aux' if stage == 'auxiliary' else 'N'}":
           (g3_dir / f"qwen-factor-g3/formal/{stage}_PROBE.json", g3_audits[stage])
           for stage in ("auxiliary", "novel")},
    }

    def summarize_probe(records):
        counts = Counter(record["target_interest"] for record in records)
        fields = [key for key in records[0] if key != "target_interest"]
        means = {key: sum(record[key] / counts[record["target_interest"]] for record in records) /
                 len(counts) for key in fields}
        groups = {}
        for name, selected in (("o_supported_93", supported), ("n_only_supported_19", labels - supported)):
            present = set(counts) & selected
            groups[name] = {"classes": len(present), "samples": sum(counts[label] for label in present),
                **{key: sum(record[key] / counts[record["target_interest"]] for record in records
                             if record["target_interest"] in present) / len(present) for key in fields}}
        return {"samples": len(records), "classes": len(counts), "macro": means, "groups": groups}

    probes = {}
    for name, (path, audit) in probe_specs.items():
        assert audit["passed"] and audit["limit"] is None and audit["probe_sha256"] == probe_sha
        assert sha256_file(path) == audit["sha256"]
        document = read_source(path)
        assert document["probe_sha256"] == probe_sha and document["adapter_sha256"] == audit["adapter_sha256"]
        raw_path = path.with_suffix(".npz")
        assert sha256_file(raw_path) == document["logits_sha256"]
        provenance[str(raw_path.relative_to(root.parent))] = document["logits_sha256"]
        cir_records, novel_records = [], []
        with np.load(raw_path, allow_pickle=False) as arrays:
            assert all(arrays[key].shape == (2220, 112) for key in ("correct", "donor", "novel"))
            for index, row in enumerate(probe_rows):
                donor = probe_rows[row["donor_index"]]
                seen = np.asarray([category in row["historical_interests"] for category in categories])
                donor_seen = np.asarray([category in donor["historical_interests"] for category in categories])
                common = ~(seen | donor_seen)
                target = category_indices[row["target_interest"]]
                scores, other = arrays["correct"][index].astype(float), arrays["donor"][index].astype(float)
                assert np.isfinite(scores).all() and np.isfinite(other).all() and common[target]
                full_nll = float(logsumexp(scores) - scores[target])
                conditional_nll = float(logsumexp(scores[~seen]) - scores[target])
                common_true = float(logsumexp(scores[common]) - scores[target])
                common_donor = float(logsumexp(other[common]) - other[target])
                record = {"target_interest": row["target_interest"], "conditional_nll": conditional_nll,
                    "full_nll": full_nll, "exclusion_nll": full_nll - conditional_nll,
                    "common_true_nll": common_true, "common_donor_nll": common_donor,
                    "history_advantage": common_donor - common_true,
                    "seen_mass": float(np.exp(scores - logsumexp(scores))[seen].sum())}
                saved = document["records"][index]
                assert saved["id"] == row["id"] and saved["donor_id"] == donor["id"]
                assert saved["target_interest"] == row["target_interest"]
                for key in ("conditional_nll", "history_advantage", "seen_mass"):
                    assert math.isclose(record[key], saved[key], rel_tol=0, abs_tol=1e-12)
                cir_records.append(record)
                source = row["probe_source"]
                if source["target_interest"] not in category_indices:
                    assert np.isnan(arrays["novel"][index]).all()
                    continue
                target = category_indices[source["target_interest"]]
                seen = np.asarray([category in source["historical_interests"] for category in categories])
                scores = arrays["novel"][index].astype(float)
                assert np.isfinite(scores).all() and not seen[target]
                full_nll = float(logsumexp(scores) - scores[target])
                conditional_nll = float(logsumexp(scores[~seen]) - scores[target])
                record = {"target_interest": source["target_interest"], "target_nll_112": full_nll,
                    "conditional_nll": conditional_nll, "exclusion_nll": full_nll - conditional_nll,
                    "seen_mass": float(np.exp(scores - logsumexp(scores))[seen].sum())}
                saved = document["novel_records"][len(novel_records)]
                assert saved["id"] == source["id"] and saved["target_interest"] == source["target_interest"]
                for key in ("target_nll_112", "conditional_nll", "seen_mass"):
                    assert math.isclose(record[key], saved[key], rel_tol=0, abs_tol=1e-12)
                novel_records.append(record)
        summary = summarize_probe(cir_records)
        for key, value in audit["macro"].items():
            assert math.isclose(summary["macro"][key], value, rel_tol=0, abs_tol=1e-12)
        assert len(novel_records) == audit["novel_rows"] == 2217
        probes[name] = {"cir": summary, "source_novel": summarize_probe(novel_records)}
    output = {"passed": True, "observed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "post_hoc_offline_decomposition_no_training_no_reranking",
              "source_sha256": provenance, "conditions": conditions, "comparisons": comparisons,
              "inverse_probes": inverse_probes, "probes": probes,
              "limitations": ["Single seed and reused Dev; no significance claim.",
                  "Post-treatment seen-slot partitions are descriptive, not causal mediation.",
                  "No-slot gains exclude direct occupied-slot correction, not indirect exclusion-training effects.",
                  "G0 original 4-rank GC and G3 resumed 8-rank no-GC do not share identical optimizer trajectories."]}
    output_path = root / "cir_factorization/outputs/MECHANISM_EVIDENCE_20260920.json"
    write_json_atomic(output_path, output)
    print(json.dumps({"passed": True, "conditions": len(conditions), "pairs": len(comparisons),
                      "output": str(output_path)}, ensure_ascii=False))
    return output


def probe_probability_metrics(probability, targets) -> dict:
    import numpy as np

    probability = np.asarray(probability, dtype=float)
    targets = np.asarray(targets, dtype=int)
    if (probability.ndim != 2 or targets.shape != (len(probability),)
            or not len(targets) or not np.isfinite(probability).all()
            or (probability <= 0).any() or (targets < 0).any()
            or (targets >= probability.shape[1]).any()
            or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Expected positive normalized probabilities and aligned targets")
    counts = Counter(targets.tolist())
    weights = np.asarray([1 / (len(counts) * counts[target]) for target in targets])
    order = np.argsort(-probability, axis=1, kind="stable")
    ranks = np.argmax(order == targets[:, None], axis=1) + 1
    result = {"macro_nll": float(weights @ -np.log(probability[np.arange(len(targets)), targets]))}
    for cutoff in (10, 20):
        result[f"macro_recall_at_{cutoff}"] = float(weights @ (ranks <= cutoff))
        result[f"macro_ndcg_at_{cutoff}"] = float(weights @ ((ranks <= cutoff) / np.log2(1 + ranks)))
    return result


def test_probe_probability_metrics() -> None:
    import numpy as np

    probability = np.full((3, 21), 1 / 21)
    targets = np.asarray([0, 0, 20])
    result = probe_probability_metrics(probability, targets)
    assert math.isclose(result["macro_nll"], math.log(21), abs_tol=1e-12)
    for cutoff in (10, 20):
        assert result[f"macro_recall_at_{cutoff}"] == 0.5
        assert result[f"macro_ndcg_at_{cutoff}"] == 0.5
    peaked = np.full((3, 21), 0.01)
    peaked[np.arange(3), targets] = 0.8
    result = probe_probability_metrics(peaked, targets)
    assert result["macro_recall_at_20"] == result["macro_ndcg_at_20"] == 1
    for invalid in (probability * 2, np.zeros_like(probability), probability * np.nan):
        try:
            probe_probability_metrics(invalid, targets)
        except ValueError:
            continue
        raise AssertionError("Invalid probability matrix was accepted")
    print("PASS: class-balanced metrics, stable ties, target alignment and invalid probabilities")


def paired_probe_summary(before, after, targets) -> dict:
    import numpy as np

    baseline = probe_probability_metrics(before, targets)
    updated = probe_probability_metrics(after, targets)
    names = ("macro_nll", *METRICS)
    per_condition = []
    for probability in (before, after):
        ranks = np.argmax(np.argsort(-probability, axis=1, kind="stable") == targets[:, None], axis=1) + 1
        values = {"macro_nll": -np.log(probability[np.arange(len(targets)), targets])}
        for cutoff in (10, 20):
            values[f"macro_recall_at_{cutoff}"] = (ranks <= cutoff).astype(float)
            values[f"macro_ndcg_at_{cutoff}"] = (ranks <= cutoff) / np.log2(1 + ranks)
        per_condition.append(np.column_stack([values[name] for name in names]))
    differences = per_condition[1] - per_condition[0]
    classes = sorted(set(targets.tolist()))
    rng = np.random.default_rng(42)
    bootstrap = np.zeros((2000, len(names)))
    per_class = {}
    for target in classes:
        selected = differences[targets == target]
        sampled = rng.integers(len(selected), size=(2000, len(selected)))
        bootstrap += selected[sampled].mean(axis=1) / len(classes)
        per_class[str(target)] = {"rows": len(selected), **dict(zip(names, selected.mean(axis=0).tolist()))}
    summary = {}
    for index, name in enumerate(names):
        class_deltas = np.asarray([values[name] for values in per_class.values()])
        assert math.isclose(float(class_deltas.mean()), updated[name] - baseline[name], abs_tol=1e-12)
        summary[name] = {
            "delta": updated[name] - baseline[name],
            "fixed_class_bootstrap_95_percent": np.quantile(bootstrap[:, index], [0.025, 0.975]).tolist(),
            "classes_positive_delta": int((class_deltas > 0).sum()),
            "classes_negative_delta": int((class_deltas < 0).sum()),
            "classes_zero_delta": int((class_deltas == 0).sum()),
        }
    return {"summary": summary, "per_class": per_class,
            "scope": "Exploratory paired resampling within each of 92 fixed classes; not adjusted for method search.",
            "bootstrap_seed": 42, "bootstrap_repeats": 2000}


def transition_screen(*, linear: bool = False) -> dict:
    import numpy as np
    from scipy.special import logsumexp, softmax

    root = _source_workspace('eval_interest_id_minionerec')
    factor = root / "cir_factorization"
    destination = factor / "outputs" / ("N_LINEAR_HISTORY_SCREEN.json" if linear else "N_TRANSITION_SCREEN.json")
    if destination.exists():
        raise FileExistsError(destination)
    paths = {
        "train": factor / 'package/crossinterest.training/data/train.parquet',
        "probe": factor / "data/probe.parquet",
        "mapping": factor / 'package/crossinterest.training/data/interest_id_semantic_mapping.json',
        "audit": factor / "outputs/reference/RAW_RESULT_AUDIT.json",
        "document": factor / "outputs/reference/qwen-factor-g0-probes/novel_PROBE.json",
        "logits": factor / "outputs/reference/qwen-factor-g0-probes/novel_PROBE.npz",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    expected = {
        "train": "43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd",
        "probe": "78d697bb1822d56ff7c37379026c916619761cd00240dfa2ea4f63615cce072a",
        "mapping": "5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da",
    }
    assert all(hashes[key] == value for key, value in expected.items())
    mapping = json.loads(paths["mapping"].read_text())
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    indices = {category: index for index, category in enumerate(categories)}
    assert len(indices) == 112
    audit_root = json.loads(paths["audit"].read_text())
    assert audit_root["passed"] and not audit_root["smoke_only"]
    audit = audit_root["audits"]["qwen-factor-g0-probes"]["audits"]["novel"]
    document = json.loads(paths["document"].read_text())
    assert audit["passed"] and audit["limit"] is None
    assert audit["sha256"] == hashes["document"]
    assert audit["probe_sha256"] == document["probe_sha256"] == hashes["probe"]
    assert document["logits_sha256"] == hashes["logits"]
    assert document["adapter_sha256"] == audit["adapter_sha256"]
    train = pq.read_table(paths["train"], columns=[
        "user_id", "historical_interests", "history_timestamps", "target_interest"
    ]).to_pylist()
    probe = pq.read_table(paths["probe"]).to_pylist()
    assert len(train) == 112000 and len(probe) == 2220
    sources = [row["probe_source"] for row in probe]
    assert not ({row["user_id"] for row in train} & {row["user_id"] for row in sources})
    assert len({row["user_id"] for row in sources}) == len(sources)
    assert all(saved["id"] == row["id"] for saved, row in zip(document["records"], probe))
    assert len(document["records"]) == len(probe)
    for row in train + sources:
        assert row["history_timestamps"] == sorted(row["history_timestamps"])
        assert len(row["historical_interests"]) == len(row["history_timestamps"]) > 0
    valid = np.asarray([row["target_interest"] in indices for row in sources])
    selected = [row for row, present in zip(sources, valid) if present]
    targets = np.asarray([indices[row["target_interest"]] for row in selected])
    assert len(selected) == audit["novel_rows"] == 2217 and len(set(targets)) == 92
    with np.load(paths["logits"], allow_pickle=False) as arrays:
        assert arrays["novel"].shape == (2220, 112)
        assert np.isnan(arrays["novel"][~valid]).all()
        logits = arrays["novel"][valid].astype(float)
    assert np.isfinite(logits).all()
    nll = logsumexp(logits, axis=1) - logits[np.arange(len(targets)), targets]
    assert len(document["novel_records"]) == len(selected)
    for row, saved, value in zip(selected, document["novel_records"], nll):
        assert saved["id"] == row["id"] and saved["target_interest"] == row["target_interest"]
        assert math.isclose(value, saved["target_nll_112"], rel_tol=0, abs_tol=1e-12)
    counts = Counter(row["target_interest"] for row in train)
    assert len(counts) == 112 and set(counts.values()) == {1000}
    transitions = np.full((113, 112), 0.5)
    for row in train:
        transitions[indices.get(row["historical_interests"][-1], 112), indices[row["target_interest"]]] += 1
    transitions /= transitions.sum(axis=1, keepdims=True)
    statistical = np.asarray([transitions[indices.get(row["historical_interests"][-1], 112)] for row in selected])
    statistic_name = "last_interest_transition"
    protocol = {"smoothing": 0.5}
    learned = {}
    if linear:
        import time
        import warnings
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        def features(rows):
            matrix = np.zeros((len(rows), 113), dtype=float)
            for row_index, row in enumerate(rows):
                matrix[row_index, [indices.get(category, 112) for category in row["historical_interests"]]] = 1
            return matrix

        started = time.monotonic()
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, tol=1e-4)
        train_targets = np.asarray([indices[row["target_interest"]] for row in train])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            classifier.fit(features(train), train_targets)
        assert not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
        assert np.array_equal(classifier.classes_, np.arange(112))
        statistical = classifier.predict_proba(features(selected))
        learned = {"coefficients": classifier.coef_, "intercept": classifier.intercept_}
        protocol = {"features": "binary visible-history category incidence, 112 known plus one unknown",
                    "C": 1.0, "solver": "lbfgs", "max_iter": 300, "tol": 1e-4,
                    "iterations": classifier.n_iter_.tolist(), "sklearn_version": sklearn.__version__,
                    "fit_seconds": time.monotonic() - started}
        statistic_name = "linear_history"
    original = softmax(logits, axis=1)
    conditions = {
        "g0": probe_probability_metrics(original, targets),
        statistic_name: probe_probability_metrics(statistical, targets),
        "equal_probability_mixture": probe_probability_metrics((original + statistical) / 2, targets),
        "unconditional_prior_control": probe_probability_metrics((original + 1 / 112) / 2, targets),
    }
    rng = np.random.default_rng(42)
    shuffled = [probe_probability_metrics((original + statistical[rng.permutation(len(targets))]) / 2, targets)
                for _ in range(200)]
    permutation = {}
    for key, observed in conditions["equal_probability_mixture"].items():
        values = np.asarray([result[key] for result in shuffled])
        permutation[key] = {
            "mean": float(values.mean()),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
            "exploratory_one_sided_p": float((1 + ((values <= observed) if key == "macro_nll"
                                                     else (values >= observed)).sum()) / 201),
        }
    baseline, mixture = conditions["g0"], conditions["equal_probability_mixture"]
    output = {
        "passed": True, "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory_train_only_statistical_screen_not_formal_beam",
        "script_sha256": sha256_file(Path(__file__)),
        "sources": {key: {"path": str(path.relative_to(root)), "sha256": hashes[key]} for key, path in paths.items()},
        "protocol": {**protocol, "g0_weight": 0.5, "statistic_weight": 0.5,
                     "shuffle_seed": 42, "shuffle_repeats": 200, "tuning": False},
        "train_rows": len(train), "probe_rows": len(selected), "probe_classes": len(set(targets)),
        "unknown_last_interest_train": sum(row["historical_interests"][-1] not in indices for row in train),
        "conditions": conditions, "shuffled_user_control": permutation,
        "mixture_delta": {key: mixture[key] - baseline[key] for key in baseline},
        "screen_gate_passed": (mixture["macro_ndcg_at_20"] > baseline["macro_ndcg_at_20"]
                               and mixture["macro_recall_at_20"] >= baseline["macro_recall_at_20"]),
        "limitations": ["92 classes only, not the formal 112-class Dev evaluation.",
                        "Diagnostic probability mixture is not an authorized inference method.",
                        "Permutation control was added after observing the mixture; p is exploratory, not uplift significance.",
                        "User-disjoint from original N training; no claim of complete historical-pipeline independence."],
    }
    if linear:
        output["paired"] = paired_probe_summary(original, (original + statistical) / 2, targets)
        output["limitations"].append("Linear history was investigated after last-interest and model-ensemble screens; this is adaptive exploration.")
        arrays_path = destination.with_suffix(".npz")
        with arrays_path.open("xb") as handle:
            np.savez_compressed(handle, original=original, statistical=statistical, targets=targets,
                                probe_indices=np.flatnonzero(valid), **learned)
        output["arrays"] = {"path": str(arrays_path.relative_to(root)), "sha256": sha256_file(arrays_path)}
    write_json_atomic(destination, output)
    print(json.dumps({"passed": True, "screen_gate_passed": output["screen_gate_passed"],
                      "conditions": conditions, "output": str(destination)}, indent=2))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mechanism", action="store_true")
    modes.add_argument("--transition-screen", action="store_true")
    modes.add_argument("--linear-history-screen", action="store_true")
    modes.add_argument("--test-probe-metrics", action="store_true")
    arguments = parser.parse_args()
    if arguments.test_probe_metrics:
        test_probe_probability_metrics()
    elif arguments.transition_screen:
        transition_screen()
    elif arguments.linear_history_screen:
        transition_screen(linear=True)
    elif arguments.mechanism:
        mechanism_main()
    else:
        main()