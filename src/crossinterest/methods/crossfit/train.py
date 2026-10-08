"""User/content-isolated Amazon folds for Qwen LoRA out-of-fold scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from crossinterest.evaluation.build_views import render_history
from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.backbones.runner import TRAIN, TRAIN_SHA256, MAPPING, MAPPING_SHA256, DEV, ROOT as BACKBONE_ROOT, code_hashes
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.alignment.audit import require_fields


ROOT = HERE / "crossfit"
SEED = 20260825


def input_signature(row: dict) -> str:
    text = unicodedata.normalize("NFKC", render_history(row, "behavior_interest"))
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def assign_folds(rows: list[dict], n_splits: int = 5) -> tuple[list[dict], dict]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.model_selection import StratifiedGroupKFold

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source IDs")
    signatures = [input_signature(row) for row in rows]
    users = {user: index for index, user in enumerate(sorted({row["user_id"] for row in rows}))}
    contents = {signature: len(users) + index for index, signature in enumerate(sorted(set(signatures)))}
    user_nodes = np.array([users[row["user_id"]] for row in rows])
    content_nodes = np.array([contents[signature] for signature in signatures])
    node_count = len(users) + len(contents)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (user_nodes, content_nodes)),
                       shape=(node_count, node_count)).tocsr()
    group_count, components = connected_components(graph, directed=False)
    groups = components[user_nodes]
    targets = np.array([row["target_interest"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    assigned = np.zeros(len(rows), dtype=np.int8)
    counts = []
    for fold, (train_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros(len(rows)), targets, groups), 1
    ):
        if set(groups[train_indices]) & set(groups[heldout_indices]):
            raise ValueError("group leakage across teacher train/heldout")
        if set(targets[train_indices]) != set(targets):
            raise ValueError("incomplete teacher training class support")
        if np.any(assigned[heldout_indices]):
            raise ValueError("heldout folds overlap")
        assigned[heldout_indices] = fold
        train_count = len(train_indices)
        counts.append({"fold": fold, "train_rows": train_count, "heldout_rows": len(heldout_indices),
                       "train_steps": math.ceil(math.ceil(train_count / 4) / 8),
                       "ddp_padding_rows": math.ceil(train_count / 4) * 4 - train_count,
                       "train_class_counts": dict(Counter(targets[train_indices])),
                       "heldout_class_counts": dict(Counter(targets[heldout_indices]))})
    if np.any(assigned == 0):
        raise ValueError("OOF folds do not cover every source row")
    records = [{"id": row["id"], "user_id": row["user_id"], "target_interest": row["target_interest"],
                "group_id": int(groups[index]), "fold": int(assigned[index]),
                "input_sha256": signatures[index]} for index, row in enumerate(rows)]
    return records, {"rows": len(rows), "users": len(users), "groups": int(group_count),
                     "duplicate_input_excess_rows": len(rows) - len(contents),
                     "largest_group_rows": max(Counter(groups).values()), "folds": counts,
                     "user_and_input_leakage": 0, "coverage_count_per_row": 1}


def target_nll(logits, encoded: dict, target_id: int) -> float:
    import torch

    positions = [index for index, label in enumerate(encoded["labels"]) if label == target_id]
    if len(positions) != 1 or positions[0] == 0:
        raise ValueError("expected exactly one supervised target ID with a causal prefix")
    position = positions[0]
    scores = logits[0, position - 1].float()
    loss = torch.logsumexp(scores, dim=-1) - scores[target_id]
    value = float(loss.item())
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid full-vocabulary target NLL")
    return value


def prepare() -> dict:
    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise ValueError("frozen Train changed")
    rows = pq.read_table(TRAIN, columns=["id", "user_id", "target_interest", "historical_titles",
                                        "historical_interests", "full_seen_interests"]).to_pylist()
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 112000 or len(Counter(row["target_interest"] for row in rows)) != 112:
        raise ValueError("expected full Amazon112K")
    if set(Counter(row["target_interest"] for row in rows).values()) != {1000}:
        raise ValueError("source labels are not balanced")
    if any(row["target_interest"] in row["full_seen_interests"] for row in rows):
        raise ValueError("source target is not novel")
    records, audit = assign_folds(rows)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "folds.parquet"
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")
    backbone = json.loads((BACKBONE_ROOT / "data/qwen35_9b/MANIFEST.json").read_text())
    manifest = {"schema_version": 1, "model_repo": "Qwen/Qwen3.5-9B",
                "model_revision": backbone["model_revision"], "seed": SEED,
                "train_sha256": TRAIN_SHA256, "mapping_sha256": MAPPING_SHA256,
                "dev_sha256": sha256_file(DEV), "folds_sha256": sha256_file(path),
                "input_grouping": "NFKC whitespace-normalized casefolded ordered B+I history",
                "scoring": "single supervised target ID full-vocabulary NLL, excluding EOS",
                "selection_ratio": None, "audit": audit, "passed": True}
    write_json_atomic(ROOT / "MANIFEST.json", manifest)
    print(json.dumps({**{key: value for key, value in audit.items() if key != "folds"},
                      "folds": [{key: value for key, value in fold.items() if not key.endswith("class_counts")}
                                for fold in audit["folds"]]}, indent=2), flush=True)
    return manifest


def teacher_command(model_path: str, train_path: Path, output: Path, count: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            "--module", 'crossinterest.training.train', "--model", model_path,
            "--train-file", str(train_path), "--output-dir", str(output),
            "--view", "behavior_interest", "--label-mode", "interest_id", "--label-map", str(MAPPING),
            "--expected-train-rows", str(count), "--expected-label-count", "112",
            "--epochs", "1", "--learning-rate", "1e-4", "--max-seq-length", "8192",
            "--per-device-batch-size", "1", "--gradient-accumulation-steps", "8",
            "--expected-effective-batch-size", "32", "--save-steps", "500",
            "--dataloader-num-workers", "4", "--gradient-checkpointing",
            "--no-ddp-find-unused-parameters", "--exclude-vision-lora", "--require-qwen-fast-path"]


def completion_metrics(directory: Path, metrics: dict, steps: int) -> dict:
    missing = {key for key in ("global_step", "max_steps") if key not in metrics}
    if not missing:
        return dict(metrics)
    checkpoint = directory / f"checkpoint-{steps}"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    require_fields(state, {"global_step": steps, "max_steps": steps, "epoch": 1.0}, "final trainer state")
    final_weight = directory / "final_adapter/adapter_model.safetensors"
    checkpoint_weight = checkpoint / "adapter_model.safetensors"
    if sha256_file(final_weight) != sha256_file(checkpoint_weight):
        raise ValueError("final adapter differs from completed checkpoint")
    return {**metrics, **{key: state[key] for key in missing}}


def validate_teacher(directory: Path, model_path: str, train_path: Path, count: int) -> str:
    import torch
    from safetensors.torch import load_file

    expected = {"model": model_path, "train_sha256": sha256_file(train_path),
                "label_map_sha256": MAPPING_SHA256, "view": "behavior_interest",
                "label_mode": "interest_id", "init_adapter": None, "selection_file": None,
                "epochs": 1.0, "learning_rate": 1e-4, "effective_batch_size": 32,
                "loss_mode": "causal_lm", "seed": SEED, "sample_limit": None,
                "formal_train_rows": count, "assistant_only": True}
    for name in ("train_contract.json", "train_metrics.json"):
        require_fields(json.loads((directory / name).read_text()), expected, "teacher")
    metrics = json.loads((directory / "train_metrics.json").read_text())
    steps = math.ceil(math.ceil(count / 4) / 8)
    metrics = completion_metrics(directory, metrics, steps)
    require_fields(metrics, {"global_step": steps, "max_steps": steps, "epoch": 1.0,
                             "sample_count": count}, "teacher completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid teacher loss")
    require_fields(json.loads((directory / "runtime_config.json").read_text()), {
        "world_size": 4, "per_device_batch_size": 1, "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True, "ddp_find_unused_parameters": False,
    }, "teacher runtime")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()), {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    }, "teacher adapter")
    weight_path = adapter / "adapter_model.safetensors"
    weights = load_file(str(weight_path))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("nonfinite teacher adapter")
    if not any(torch.count_nonzero(value).item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("teacher adapter did not update")
    return sha256_file(weight_path)


def score_fold(model_path: str, train_dir: Path, rows: list[dict], memberships: dict,
               output: Path, identity: dict) -> None:
    import torch
    from transformers import AutoTokenizer
    from crossinterest.training.evaluate import load_model
    from crossinterest.training.label_space import load_label_mapping
    from crossinterest.training.prompts import render_training_messages
    from crossinterest.training.semantic_tokens import register_semantic_tokens
    from crossinterest.training.train import tokenize_assistant_only

    output.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mapping = load_label_mapping(MAPPING, expected_count=112)
    registration = register_semantic_tokens(tokenizer, mapping)
    model = None
    coverage = set()
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        path = output / f"part-{start // 128:05d}.parquet"
        if path.exists():
            records = pq.read_table(path).to_pylist()
        else:
            if model is None:
                model = load_model("qwen35_9b", model_path, train_dir / "final_adapter",
                                   tokenizer, mapping, registration, device_map="balanced")
            records = []
            for row in chunk:
                encoded = tokenize_assistant_only(tokenizer, render_training_messages(
                    row, "behavior_interest", "interest_id", mapping))
                if len(encoded["input_ids"]) > 8192:
                    raise ValueError("teacher scoring truncation is forbidden")
                inputs = torch.tensor([encoded["input_ids"]], device=model.device)
                target_id = registration["token_ids"][mapping["category_to_id"][row["target_interest"]]]
                with torch.inference_mode():
                    logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
                    nll = target_nll(logits, encoded, target_id)
                del logits, inputs
                records.append({**memberships[row["id"]], "target_token_id": target_id,
                                "target_id": mapping["category_to_id"][row["target_interest"]],
                                "nll": nll, "score_count": 1, "fingerprint": fingerprint,
                                "teacher_adapter_sha256": identity["teacher_adapter_sha256"]})
            temporary = path.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
            temporary.replace(path)
        if [record["id"] for record in records] != [row["id"] for row in chunk]:
            raise ValueError("OOF score chunk has different IDs")
        for record in records:
            require_fields(record, {**memberships[record["id"]], "fingerprint": fingerprint,
                                    "score_count": 1, "teacher_adapter_sha256": identity["teacher_adapter_sha256"]},
                           "OOF score")
            if record["id"] in coverage or not math.isfinite(record["nll"]) or record["nll"] < 0:
                raise ValueError("duplicate or invalid OOF score")
            coverage.add(record["id"])
        print(f"fold={identity['fold']} scored={len(coverage)}/{len(rows)}", flush=True)
    if coverage != {row["id"] for row in rows}:
        raise ValueError("OOF coverage mismatch")
    write_json_atomic(output.parent / "RESULT_AUDIT.json", {
        **identity, "passed": True, "scored_rows": len(coverage), "score_count_per_row": 1,
        "fingerprint": fingerprint, "train_heldout_user_overlap": 0, "train_heldout_input_overlap": 0,
    })


def execute(fold: int, output: Path) -> None:
    from huggingface_hub import snapshot_download

    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    if sha256_file(TRAIN) != TRAIN_SHA256 or sha256_file(MAPPING) != MAPPING_SHA256:
        raise ValueError("frozen source changed")
    if sha256_file(ROOT / "folds.parquet") != manifest["folds_sha256"]:
        raise ValueError("fold membership changed")
    memberships = {row["id"]: row for row in pq.read_table(ROOT / "folds.parquet").to_pylist()}
    table = pq.read_table(TRAIN)
    rows = table.to_pylist()
    if len(memberships) != len(rows) or set(memberships) != {row["id"] for row in rows}:
        raise ValueError("fold IDs do not cover frozen source")
    train_indices = [index for index, row in enumerate(rows) if memberships[row["id"]]["fold"] != fold]
    heldout = [row for row in rows if memberships[row["id"]]["fold"] == fold]
    for field in ("user_id", "group_id", "input_sha256"):
        train_values = {memberships[rows[index]["id"]][field] for index in train_indices}
        if train_values & {memberships[row["id"]][field] for row in heldout}:
            raise ValueError(f"teacher isolation violation: {field}")
    fold_info = manifest["audit"]["folds"][fold - 1]
    require_fields(fold_info, {"fold": fold, "train_rows": len(train_indices),
                              "heldout_rows": len(heldout)}, "fold sizes")
    model_path = snapshot_download(manifest["model_repo"], revision=manifest["model_revision"],
                                   cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    execution = {"fold": fold, "model_repo": manifest["model_repo"], "model_path": model_path,
                 "model_revision": manifest["model_revision"], "folds_sha256": manifest["folds_sha256"],
                 "source_sha256": TRAIN_SHA256, "mapping_sha256": MAPPING_SHA256,
                 "code_sha256": {**code_hashes(), "crossfit.py": sha256_file(Path(__file__))},
                 "view": "behavior_interest", "selection_ratio": None}
    contract_path = output / "EXECUTION_CONTRACT.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != execution:
        raise ValueError("teacher execution identity changed")
    write_json_atomic(contract_path, execution)
    smoke_indices = []
    class_counts = Counter()
    for index in train_indices:
        label = rows[index]["target_interest"]
        if class_counts[label] < 2:
            smoke_indices.append(index)
            class_counts[label] += 1
    if len(smoke_indices) != 224 or set(class_counts.values()) != {2}:
        raise ValueError("cannot build balanced distinct teacher smoke")
    environment = os.environ | {"TOKENIZERS_PARALLELISM": "false",
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        directory.mkdir(exist_ok=True)
        indices = smoke_indices if smoke else train_indices
        train_path = directory / "teacher_train.parquet"
        expected_table = table.take(indices)
        if train_path.exists():
            if not pq.read_table(train_path).equals(expected_table):
                raise ValueError("existing teacher training data changed")
        else:
            pq.write_table(expected_table, train_path, compression="zstd")
        train_dir = directory / "train"
        if not (train_dir / "train_metrics.json").exists():
            subprocess.run(teacher_command(model_path, train_path, train_dir, len(indices)),
                           cwd=HERE.parent, env=environment, check=True)
        adapter_sha = validate_teacher(train_dir, model_path, train_path, len(indices))
        identity = {**execution, "smoke": smoke, "train_rows": len(indices),
                    "train_sha256": sha256_file(train_path), "teacher_adapter_sha256": adapter_sha}
        subprocess.run([sys.executable, "-m", 'crossinterest.methods.crossfit.train', "score",
                        "--fold", str(fold), "--output-dir", str(directory),
                        "--model-path", model_path, *( ["--smoke"] if smoke else [])],
                       cwd=HERE.parent, env=environment, check=True)
        audit = json.loads((directory / "RESULT_AUDIT.json").read_text())
        require_fields(audit, {**identity, "passed": True, "scored_rows": 4 if smoke else len(heldout)},
                       "teacher score audit")


def score(fold: int, output: Path, model_path: str, smoke: bool) -> None:
    execution = json.loads((output.parent / "EXECUTION_CONTRACT.json").read_text())
    require_fields(execution, {"fold": fold, "model_path": model_path}, "score identity")
    memberships = {row["id"]: row for row in pq.read_table(ROOT / "folds.parquet").to_pylist()}
    rows = [row for row in pq.read_table(TRAIN).to_pylist() if memberships[row["id"]]["fold"] == fold]
    rows = rows[:4] if smoke else rows
    train_path = output / "teacher_train.parquet"
    count = pq.ParquetFile(train_path).metadata.num_rows
    adapter_sha = validate_teacher(output / "train", model_path, train_path, count)
    score_fold(model_path, output / "train", rows, memberships, output / "scores", {
        **execution, "smoke": smoke, "train_rows": count,
        "train_sha256": sha256_file(train_path), "teacher_adapter_sha256": adapter_sha})


def class_normalized_weights(records: list[dict]) -> tuple[list[dict], dict]:
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("empty scores or duplicate OOF IDs")
    grouped = {}
    for record in records:
        if record.get("score_count") != 1 or not math.isfinite(record["nll"]) or record["nll"] < 0:
            raise ValueError("OOF scores must be finite nonnegative and unique")
        grouped.setdefault(record["target_interest"], []).append(record)
    weighted = []
    statistics = {}
    for label, rows in sorted(grouped.items()):
        scores = np.array([row["nll"] for row in rows], dtype=np.float64)
        probabilities = np.exp(-(scores - scores.min()))
        weights = len(rows) * probabilities / probabilities.sum()
        if not np.isfinite(weights).all() or np.any(weights < np.finfo(np.float32).tiny):
            raise ValueError("weight underflow requires explicit policy review; no silent clipping")
        if not np.isclose(weights.mean(), 1.0, atol=1e-12):
            raise ValueError("class weights must have mean one")
        statistics[label] = {"rows": len(rows), "mean": float(weights.mean()),
                             "min": float(weights.min()), "max": float(weights.max()),
                             "effective_sample_size": float(weights.sum() ** 2 / (weights ** 2).sum())}
        weighted.extend({**row, "sample_weight": float(weight)}
                        for row, weight in zip(rows, weights, strict=True))
    weighted.sort(key=lambda row: row["id"])
    return [{**row, "nested_rank": index} for index, row in enumerate(weighted, 1)], statistics


def build_weights(results: Path, output: Path) -> dict:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    if sha256_file(TRAIN) != TRAIN_SHA256 or sha256_file(MAPPING) != MAPPING_SHA256:
        raise ValueError("frozen source changed")
    if sha256_file(ROOT / "folds.parquet") != manifest["folds_sha256"]:
        raise ValueError("frozen folds changed")
    members = {row["id"]: row for row in pq.read_table(ROOT / "folds.parquet").to_pylist()}
    source = pq.read_table(TRAIN, columns=["id", "target_interest"]).to_pylist()
    if set(members) != {row["id"] for row in source} or len(source) != 112000:
        raise ValueError("expected complete frozen112K")
    records = []
    provenance = {}
    for fold_info in manifest["audit"]["folds"]:
        fold = fold_info["fold"]
        directory = results / f"teacher-f{fold}" / "formal"
        audit_path = directory / "RESULT_AUDIT.json"
        if not audit_path.exists():
            raise ValueError(f"fold {fold} formal OOF not complete")
        audit = json.loads(audit_path.read_text())
        require_fields(audit, {"passed": True, "smoke": False, "fold": fold,
            "source_sha256": TRAIN_SHA256, "mapping_sha256": MAPPING_SHA256,
            "folds_sha256": manifest["folds_sha256"], "model_repo": manifest["model_repo"],
            "model_revision": manifest["model_revision"], "view": "behavior_interest",
            "train_rows": fold_info["train_rows"], "scored_rows": fold_info["heldout_rows"],
            "score_count_per_row": 1, "train_heldout_user_overlap": 0,
            "train_heldout_input_overlap": 0}, "formal teacher scores")
        chunks = sorted((directory / "scores").glob("part-*.parquet"))
        heldout = [row for path in chunks for row in pq.read_table(path).to_pylist()]
        expected_ids = {identity for identity, member in members.items() if member["fold"] == fold}
        if len(heldout) != len(expected_ids) or {row["id"] for row in heldout} != expected_ids:
            raise ValueError(f"fold {fold} coverage mismatch")
        for row in heldout:
            require_fields(row, {**members[row["id"]], "score_count": 1,
                "fingerprint": audit["fingerprint"],
                "teacher_adapter_sha256": audit["teacher_adapter_sha256"]}, "score provenance")
        records.extend(heldout)
        provenance[str(fold)] = {"audit_sha256": sha256_file(audit_path),
            "teacher_adapter_sha256": audit["teacher_adapter_sha256"],
            "train_sha256": audit["train_sha256"],
            "score_chunks_sha256": {path.name: sha256_file(path) for path in chunks}}
    expected_labels = {row["id"]: row["target_interest"] for row in source}
    if any(expected_labels[row["id"]] != row["target_interest"] for row in records):
        raise ValueError("OOF target labels differ from frozen source")
    weighted, statistics = class_normalized_weights(records)
    if len(weighted) != 112000 or len(statistics) != 112 or any(
        stats["rows"] != 1000 for stats in statistics.values()
    ):
        raise ValueError("expected 112 classes with 1000 retained rows each")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "class_weights.parquet"
    table = pa.Table.from_pylist(weighted)
    if path.exists() and not pq.read_table(path).equals(table):
        raise ValueError("existing OOF weights changed")
    if not path.exists():
        pq.write_table(table, path, compression="zstd")
    audit = {"passed": True, "rows": len(weighted), "retained_fraction": 1.0,
             "formula": "n_class * exp(-nll) / sum_class(exp(-nll))", "temperature": 1.0,
             "normalization": "mean one within each target class across all five folds",
             "loss": "mean(sample_weight * per_sample_assistant_causal_lm_loss)",
             "source_sha256": TRAIN_SHA256, "folds_sha256": manifest["folds_sha256"],
             "weights_sha256": sha256_file(path), "class_statistics": statistics,
             "teacher_provenance": provenance}
    write_json_atomic(output / "WEIGHT_AUDIT.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "score", "weights"))
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--teacher-results", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "weights":
        if args.teacher_results is None or args.output_dir is None:
            parser.error("weights requires --teacher-results and --output-dir")
        build_weights(args.teacher_results.resolve(), args.output_dir.resolve())
    elif args.fold is None or args.output_dir is None:
        parser.error("execution requires --fold and --output-dir")
    elif args.action == "execute":
        execute(args.fold, args.output_dir.resolve())
    elif args.model_path is None:
        parser.error("score requires --model-path")
    else:
        score(args.fold, args.output_dir.resolve(), args.model_path, args.smoke)


if __name__ == "__main__":
    main()