"""Audited independent-interest composition, without modifying frozen experiments."""

from __future__ import annotations

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
from functools import partial
from tempfile import mkdtemp

import pyarrow as pa
import pyarrow.parquet as pq

if not __package__:
    live = _source_workspace('eval_interest_id_minionerec')
    package = live / "cir_factorization/package"
    if not package.is_dir():
        package = live.parent
    sys.path[:0] = [str(package), str(package / "prep")]
    import crossinterest.methods as eval_interest_id_minionerec
    if str(live) not in eval_interest_id_minionerec.__path__:
        eval_interest_id_minionerec.__path__.append(str(live))
    __package__ = "eval_interest_id_minionerec"

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import HERE as FROZEN_HERE, sha256_file
from crossinterest.methods.cir.runner import TRAIN, DEV, MAPPING, TRAIN_SHA, DEV_SHA, MAPPING_SHA
from crossinterest.methods.composition.model import encode_example, group_texts, history_groups
from crossinterest.methods.mechanism.runner import immutable_json
from crossinterest.methods.alignment.audit import require_fields


HERE = _source_workspace('eval_interest_id_minionerec')
ROOT = HERE / "composition"
REFERENCE = FROZEN_HERE / "cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json"
DATA_AUDIT = ROOT / "DATA_AUDIT.json"
PREFLIGHT = ROOT / "PREFLIGHT.json"
METHOD = "independent-interest-composition-v1"
ARMS = {"C": "category", "R": "random", "N": "category"}
MODULES = ("interest_composition.py", "composition_train.py", "composition_run.py",
           "test_interest_composition.py", "dual_checkpoint_io.py", "composition_monitor.py")
WORKER = {}


def tokenization_identity():
    return hashlib.sha256("\n".join(inspect.getsource(function) for function in
                                   (history_groups, group_texts, encode_example)).encode()).hexdigest()


def initialize_worker():
    from transformers import AutoTokenizer
    from crossinterest.training.semantic_tokens import register_semantic_tokens

    reference = json.loads(REFERENCE.read_text())
    mapping = json.loads(MAPPING.read_text())
    tokenizer = AutoTokenizer.from_pretrained(reference["model_repo"], revision=reference["model_revision"],
                                            local_files_only=True)
    register_semantic_tokens(tokenizer, mapping)
    WORKER.update(tokenizer=tokenizer, mapping=mapping)


def encode_rows(work):
    stage, rows = work
    encoded = {grouping: [] for grouping in ("category", "random")}
    changes = 0
    for row in rows:
        original = history_groups(row, "category")
        shuffled = history_groups(row, "random")
        if sorted(map(len, original)) != sorted(map(len, shuffled)):
            raise ValueError("random control changed group sizes")
        changes += original != shuffled
        if row["target_interest"] in row["historical_interests"]:
            raise ValueError("target category is present in visible history")
        for grouping in encoded:
            feature = encode_example(row, WORKER["tokenizer"], WORKER["mapping"],
                                     "novel" if stage == "dev" else stage, grouping,
                                     training=stage != "dev")
            if stage == "auxiliary":
                feature["positive_mask"] = [WORKER["mapping"]["id_to_category"][token] in row["positive_interests"]
                                            for token in WORKER["mapping"]["special_tokens"]]
                if not any(feature["positive_mask"]):
                    raise ValueError("CIR sample has no positive target")
            encoded[grouping].append({"id": row.get("id", row.get("sample_id")), **feature})
    return encoded, changes


def file_record(path, rows):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": rows}


def verify_data():
    record = json.loads(DATA_AUDIT.read_text())
    if record["tokenization_identity"] != tokenization_identity():
        raise ValueError("tokenization changed after data preparation")
    for path, expected in ((TRAIN, TRAIN_SHA), (DEV, DEV_SHA), (MAPPING, MAPPING_SHA),
                           (REFERENCE, record["reference_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen data identity changed: {path.name}")
    for info in record["sources"].values():
        if sha256_file(FROZEN_HERE.parent / info["path"]) != info["sha256"]:
            raise ValueError("source parquet changed")
    for grouping in record["groups"].values():
        for info in grouping.values():
            for key in ("cache", "smoke_cache", "smoke_source"):
                if key in info and sha256_file(ROOT / info[key]["path"]) != info[key]["sha256"]:
                    raise ValueError("prepared composition data changed")
    return record


def prepare_data(workers=4):
    if DATA_AUDIT.exists():
        return verify_data()
    reference = json.loads(REFERENCE.read_text())
    sources = {stage: reference["files"][stage] for stage in ("auxiliary", "novel")}
    sources["dev"] = {"path": str(DEV.relative_to(FROZEN_HERE.parent)), "sha256": DEV_SHA, "rows": 2000}
    record = {"passed": True, "tokenization_identity": tokenization_identity(),
              "reference_sha256": sha256_file(REFERENCE), "sources": sources,
              "groups": {grouping: {} for grouping in ("category", "random")}, "random_changed_rows": {}}
    ROOT.mkdir(exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(workers, initializer=initialize_worker) as pool:
        for stage, source in sources.items():
            path = FROZEN_HERE.parent / source["path"]
            if sha256_file(path) != source["sha256"]:
                raise ValueError("source differs from frozen preflight")
            table = pq.read_table(path)
            if len(table) != source["rows"] or len(set(table["target_interest"].to_pylist())) != 112:
                raise ValueError("source rows or class support differ")
            counts = Counter(table["target_interest"].to_pylist())
            if stage != "dev" and set(counts.values()) != {749 if stage == "auxiliary" else 1000}:
                raise ValueError("formal class budgets differ")
            writers, lengths, totals, group_counts = {}, {}, {}, {}
            maximum_decoder, maximum_groups, changes, completed = {}, {}, 0, 0
            try:
                batches = ((stage, batch.to_pylist()) for batch in table.to_batches(max_chunksize=128))
                for features, changed in pool.imap(encode_rows, batches, chunksize=1):
                    changes += changed
                    for grouping, rows in features.items():
                        cache = ROOT / "data" / grouping / f"{stage}_tokens.parquet"
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        batch = pa.Table.from_pylist(rows)
                        if grouping not in writers:
                            writers[grouping] = pq.ParquetWriter(cache, batch.schema, compression="zstd")
                            lengths[grouping], totals[grouping], group_counts[grouping] = [], 0, 0
                            maximum_decoder[grouping], maximum_groups[grouping] = 0, 0
                        writers[grouping].write_table(batch)
                        for row in rows:
                            lengths[grouping].append(max(map(len, row["groups"])))
                            totals[grouping] += sum(map(len, row["groups"])) + len(row["input_ids"])
                            group_counts[grouping] += len(row["groups"])
                            maximum_groups[grouping] = max(maximum_groups[grouping], len(row["groups"]))
                            maximum_decoder[grouping] = max(maximum_decoder[grouping], len(row["input_ids"]))
                    completed += len(features["category"])
                    if completed % 8192 == 0 or completed == len(table):
                        print(json.dumps({"event": "composition_data", "stage": stage, "rows": completed}), flush=True)
            finally:
                for writer in writers.values():
                    writer.close()
            record["random_changed_rows"][stage] = changes
            targets = table["target_interest"].to_pylist()
            for grouping in record["groups"]:
                cache = ROOT / "data" / grouping / f"{stage}_tokens.parquet"
                info = {"cache": file_record(cache, len(table)), "max_group_tokens": max(lengths[grouping]),
                        "max_decoder_tokens": maximum_decoder[grouping], "total_tokens": totals[grouping],
                        "group_count": group_counts[grouping], "max_groups": maximum_groups[grouping]}
                if stage != "dev":
                    selected, selected_counts = [], Counter()
                    for index in sorted(range(len(table)), key=lambda index: (-lengths[grouping][index], index)):
                        if selected_counts[targets[index]] < 2:
                            selected.append(index)
                            selected_counts[targets[index]] += 1
                    selected.sort()
                    if len(selected) != 224 or set(selected_counts.values()) != {2}:
                        raise ValueError("longest-per-class smoke coverage differs")
                    smoke = cache.with_name(f"smoke_{stage}.parquet")
                    smoke_cache = cache.with_name(f"smoke_{stage}_tokens.parquet")
                    pq.write_table(table.take(selected), smoke, compression="zstd")
                    pq.write_table(pq.read_table(cache).take(selected), smoke_cache, compression="zstd")
                    info.update(smoke_source=file_record(smoke, 224), smoke_cache=file_record(smoke_cache, 224))
                record["groups"][grouping][stage] = info
    immutable_json(DATA_AUDIT, record)
    return verify_data()


def stage_info(prepared, arm, stage, smoke):
    if arm not in ARMS or stage not in ("auxiliary", "novel") or (arm == "N" and stage != "novel"):
        raise ValueError("invalid arm/stage")
    source = prepared["data"]["sources"][stage]
    grouped = prepared["data"]["groups"][ARMS[arm]][stage]
    rows = 224 if smoke else source["rows"]
    epochs = 3 if stage == "auxiliary" else 1
    return {"source": ROOT / grouped["smoke_source"]["path"] if smoke else FROZEN_HERE.parent / source["path"],
            "source_sha256": grouped["smoke_source"]["sha256"] if smoke else source["sha256"],
            "cache": ROOT / grouped["smoke_cache" if smoke else "cache"]["path"],
            "rows": rows, "epochs": epochs, "steps": math.ceil(rows / 32) * epochs}


def freeze():
    if PREFLIGHT.exists():
        return verify()
    data = verify_data()
    reference = json.loads(REFERENCE.read_text())
    old_path = HERE / "bidirectional/outputs/results_envfix/RAW_RESULT_AUDIT.json"
    old = json.loads(old_path.read_text())
    require_fields(old, {"passed": True, "smoke_only": False}, "historical reference")
    historical = old["audits"]["qwen-cir-3ep"]["formal"]
    evidence = ROOT / "REFERENCE_AUDIT.json"
    if evidence.exists() and sha256_file(evidence) != sha256_file(old_path):
        raise ValueError("different reference evidence already exists")
    if not evidence.exists():
        shutil.copyfile(old_path, evidence)
    table = pq.read_table(DEV)
    lengths = [max(map(len, row["groups"])) for row in pq.read_table(
        ROOT / data["groups"]["category"]["dev"]["cache"]["path"], columns=["groups"]).to_pylist()]
    shards = {}
    for shard in range(8):
        positions = sorted(range(shard, len(table), 8), key=lambda index: (-lengths[index], index))
        path = ROOT / "data" / f"shard-{shard}" / DEV.name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.take(positions), path, compression="zstd")
        shards[str(shard)] = {**file_record(path, len(positions)), "source_positions": positions}
    prepared = {"passed": True, "method": METHOD, "arms": ARMS,
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "data": data,
        "data_audit_sha256": sha256_file(DATA_AUDIT), "model_repo": reference["model_repo"],
        "model_revision": reference["model_revision"], "shards": shards,
        "code_sha256": {name: sha256_file(HERE / name) for name in MODULES},
        "reference_audit_sha256": sha256_file(evidence),
        "reference_metrics": {key: value for key, value in historical["results"]["ranking"].items()
                              if key.startswith("macro_")},
        "training_layout": {"world_size": 8, "micro_batch": 1, "accumulation": 4,
                            "gradient_checkpointing": False},
        "evaluation": {"rows": 2000, "classes": 112, "workers": 8, "gpus_per_worker": 1,
                       "beams": 20, "batch_size": 1, "candidate_free": True,
                       "do_sample": False, "max_new_tokens": 32, "max_prompt_length": 8192},
        "execution_clock": "first GPU submission through independent final audit; 28800 seconds",
        "budget": {"auxiliary_rows": 83888, "auxiliary_epochs": 3, "auxiliary_steps": 7866,
                   "novel_rows": 112000, "novel_epochs": 1, "novel_steps": 3500, "effective_batch_size": 32},
        "improvement_gate": "C macro_ndcg_at_20 > G0 and R; C macro_recall_at_20 >= G0"}
    immutable_json(PREFLIGHT, prepared)
    return verify()


def verify():
    prepared = json.loads(PREFLIGHT.read_text())
    require_fields(prepared, {"passed": True, "method": METHOD, "arms": ARMS}, "composition preflight")
    for name, expected in prepared["code_sha256"].items():
        if sha256_file(HERE / name) != expected:
            raise ValueError(f"composition source changed: {name}")
    if sha256_file(DATA_AUDIT) != prepared["data_audit_sha256"] or verify_data() != prepared["data"]:
        raise ValueError("composition data contract changed")
    for info in prepared["shards"].values():
        if sha256_file(ROOT / info["path"]) != info["sha256"]:
            raise ValueError("evaluation shard changed")
    return prepared


def install_complete_resume(baseline, output, grouping):
    import torch
    from safetensors.torch import load_file
    from crossinterest.methods.consistency.checkpoint_io import validate_checkpoint

    def latest(directory):
        directory = Path(directory)
        candidates = sorted(directory.glob("checkpoint-*"),
                            key=lambda path: int(path.name.split("-")[1]), reverse=True)
        if not candidates and not (directory / "train_metrics.json").exists():
            raise ValueError("interrupted stage has no complete checkpoint; refusing weight-only restart")
        distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world = torch.distributed.get_world_size() if distributed else int(os.environ["WORLD_SIZE"])
        selected = [None]
        if rank == 0:
            errors = []
            for path in candidates:
                local = Path(mkdtemp(prefix="composition-resume-")) / path.name
                try:
                    shutil.copytree(path, local, copy_function=shutil.copyfile)
                    audit = validate_checkpoint(local, world)
                    metadata = json.loads((local / "interest_bridge.json").read_text())
                    require_fields(metadata, {"architecture": METHOD, "grouping": grouping}, "resume bridge")
                    weights = load_file(str(local / "interest_bridge.safetensors"))
                    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
                        raise ValueError("invalid saved bridge")
                    selected[0] = {"path": str(local), "source": str(path), **audit}
                    break
                except Exception as error:
                    errors.append({"checkpoint": path.name, "error": str(error)})
                    shutil.rmtree(local.parent)
            if selected[0] is None:
                selected[0] = {"error": "no complete composition checkpoint", "candidates": errors}
            else:
                write_json_atomic(output / "IO_RESUME_AUDIT.json", selected[0])
        if distributed:
            torch.distributed.broadcast_object_list(selected, src=0)
        if "error" in selected[0]:
            raise ValueError(selected[0])
        return Path(selected[0]["path"])

    baseline.latest_checkpoint = latest


def train(remaining):
    from crossinterest.training import train as baseline
    from crossinterest.methods.composition.train import CompositionDataset, CompositionTrainer
    from crossinterest.methods.consistency.checkpoint_io import install

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--stage", required=True, choices=("auxiliary", "novel"))
    parser.add_argument("--smoke", action="store_true")
    options, rest = parser.parse_known_args(remaining)
    args = baseline.parse_args(rest)
    prepared = verify()
    info = stage_info(prepared, options.arm, options.stage, options.smoke)
    layout = prepared["training_layout"]
    require_fields(vars(args), {"view": "behavior_interest", "label_mode": "interest_id",
        "loss_mode": "causal_lm", "label_score_head": "lm_head", "freeze_backbone": False,
        "selection_file": None, "limit": None, "max_steps": -1, "epochs": float(info["epochs"]),
        "learning_rate": 1e-4, "expected_effective_batch_size": 32, "expected_train_rows": info["rows"],
        "per_device_batch_size": layout["micro_batch"], "gradient_accumulation_steps": layout["accumulation"],
        "gradient_checkpointing": layout["gradient_checkpointing"], "save_steps": 100,
        "ddp_find_unused_parameters": False, "require_qwen_fast_path": True,
        "exclude_vision_lora": True, "max_seq_length": 8192, "resume": True}, "composition training budget")
    previous = args.output_dir.parent / "auxiliary/final_adapter" if options.stage == "novel" and options.arm != "N" else None
    if (args.init_adapter != previous or sha256_file(args.train_file) != info["source_sha256"]
            or sha256_file(args.label_map) != MAPPING_SHA
            or Path(args.model).name != prepared["model_revision"]
            or int(os.environ.get("WORLD_SIZE", "1")) != layout["world_size"]):
        raise ValueError("composition source/model/handoff/layout mismatch")
    if (args.output_dir / "train_contract.json").exists():
        install_complete_resume(baseline, args.output_dir, ARMS[options.arm])
    install(args.output_dir)
    metadata = {"task": "amazon_interest_composition", "prompt_schema_version": METHOD,
                "method": METHOD, "arm": options.arm, "grouping": ARMS[options.arm],
                "stage": options.stage, "smoke": options.smoke, "preflight_sha256": sha256_file(PREFLIGHT),
                "cached_tokens_sha256": sha256_file(info["cache"]),
                "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True,
                "init_bridge_sha256": sha256_file(previous / "interest_bridge.safetensors") if previous else None}
    baseline.main(args=args, experiment_metadata=metadata,
                  dataset_class=partial(CompositionDataset, stage=options.stage, grouping=ARMS[options.arm],
                                        encoded_file=info["cache"]),
                  trainer_class=partial(CompositionTrainer, stage=options.stage, grouping=ARMS[options.arm],
                                        init_adapter=previous))
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    rank = int(os.environ["RANK"])
    write_json_atomic(args.output_dir / f"PERFORMANCE_RANK_{rank}.json", {
        "rank": rank, "world_size": layout["world_size"], "device": properties.name,
        "total_memory_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})


def training_command(model_path, directory, arm, stage, smoke, prepared):
    from crossinterest.methods.crossfit.train import teacher_command

    info = stage_info(prepared, arm, stage, smoke)
    layout = prepared["training_layout"]
    command = teacher_command(model_path, info["source"], directory, info["rows"])
    command[command.index("--nproc_per_node=4")] = f"--nproc_per_node={layout['world_size']}"
    position = command.index('crossinterest.training.train')
    command[position:position + 1] = ['crossinterest.methods.composition.runner', "train"]
    for option, value in (("--epochs", info["epochs"]), ("--per-device-batch-size", layout["micro_batch"]),
                          ("--gradient-accumulation-steps", layout["accumulation"]), ("--save-steps", 100)):
        command[command.index(option) + 1] = str(value)
    if not layout["gradient_checkpointing"]:
        command[command.index("--gradient-checkpointing")] = "--no-gradient-checkpointing"
    command.extend(["--arm", arm, "--stage", stage, "--resume"])
    if arm != "N" and stage == "novel":
        command.extend(["--init-adapter", str(directory.parent / "auxiliary/final_adapter")])
    if smoke:
        command.append("--smoke")
    return command


def audit_training(directory, arm, stage, smoke, prepared, model_path):
    import torch
    from safetensors.torch import load_file

    info = stage_info(prepared, arm, stage, smoke)
    metrics = json.loads((directory / "train_metrics.json").read_text())
    previous = directory.parent / "auxiliary/final_adapter" if arm != "N" and stage == "novel" else None
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, {"model": model_path, "train_sha256": info["source_sha256"],
            "label_map_sha256": MAPPING_SHA, "view": "behavior_interest", "label_mode": "interest_id",
            "epochs": float(info["epochs"]), "effective_batch_size": 32, "learning_rate": 1e-4,
            "selection_file": None, "sample_limit": None, "loss_mode": "causal_lm",
            "init_adapter_model_sha256": sha256_file(previous / "adapter_model.safetensors") if previous else None}, "stage budget")
        require_fields(record["experiment_metadata"], {"method": METHOD, "arm": arm, "stage": stage,
            "smoke": smoke, "preflight_sha256": sha256_file(PREFLIGHT),
            "init_bridge_sha256": sha256_file(previous / "interest_bridge.safetensors") if previous else None}, "stage identity")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
                            "sample_count": info["rows"], "epoch": float(info["epochs"])}, "stage completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid training loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    layout = prepared["training_layout"]
    require_fields(runtime, {"world_size": layout["world_size"], "per_device_batch_size": layout["micro_batch"],
        "gradient_accumulation_steps": layout["accumulation"], "gradient_checkpointing": layout["gradient_checkpointing"],
        "ddp_find_unused_parameters": False, "dataloader_drop_last": False}, "runtime")
    require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen CUDA kernels")
    composition = json.loads((directory / "COMPOSITION_RUNTIME.json").read_text())
    require_fields(composition, {"architecture": METHOD, "grouping": ARMS[arm], "stage": stage,
                                "raw_history_decoder_bypass": False}, "composition runtime")
    if stage == "auxiliary":
        require_fields(composition, {"model_accepts_loss_kwargs": False}, "auxiliary loss scaling")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    bridge = load_file(str(adapter / "interest_bridge.safetensors"))
    if not all(torch.isfinite(value).all() for value in [*weights.values(), *bridge.values()]):
        raise ValueError("nonfinite adapter or bridge")
    if not any(value.count_nonzero() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("adapter did not update")
    if torch.equal(bridge["projection.weight"], torch.eye(composition["hidden_size"])):
        raise ValueError("bridge did not update")
    count = sum(value.numel() for value in weights.values()) + bridge["projection.weight"].numel()
    if count != composition["total_trainable_parameters"]:
        raise ValueError("saved payload does not match trainable parameter count")
    performance = []
    for rank in range(layout["world_size"]):
        record = json.loads((directory / f"PERFORMANCE_RANK_{rank}.json").read_text())
        require_fields(record, {"rank": rank, "world_size": layout["world_size"]}, "performance rank")
        if not 0 < record["peak_allocated_bytes"] <= record["peak_reserved_bytes"] <= record["total_memory_bytes"]:
            raise ValueError("invalid GPU memory telemetry")
        performance.append(record)
    result = {"passed": True, "arm": arm, "stage": stage, "rows": info["rows"], "steps": info["steps"],
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
              "bridge_sha256": sha256_file(adapter / "interest_bridge.safetensors"),
              "runtime_seconds": metrics["train_runtime"], "composition_runtime": composition,
              "performance": performance, "steps_per_second": metrics["train_steps_per_second"]}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    print(json.dumps({"event": "composition_stage_audited", "smoke": smoke, **result}), flush=True)
    return result


def evaluate(model_path, adapter, output, arm, shard, smoke):
    from crossinterest.evaluation.common import append_jsonl
    from crossinterest.training import evaluate_ranking as ranking
    from crossinterest.methods.composition.model import CompositionController, bind_ranking_composition

    prepared = verify()
    output.mkdir(parents=True, exist_ok=True)
    original_load, original_resolve = ranking.load_model, ranking.resolve_contract
    identity = {"composition_method": METHOD, "composition_arm": arm, "composition_grouping": ARMS[arm],
                "composition_preflight_sha256": sha256_file(PREFLIGHT),
                "composition_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "composition_bridge_sha256": sha256_file(adapter / "interest_bridge.safetensors"),
                "composition_shard": shard}

    def resolve(path, contract, source):
        contract.update(identity)
        contract.pop("fingerprint", None)
        contract["fingerprint"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return original_resolve(path, contract, source)

    def load(model_key, model, source, tokenizer, mapping, registration, **kwargs):
        result = original_load(model_key, model, source, tokenizer, mapping, registration, **kwargs)
        controller = CompositionController(result, ARMS[arm], directory=source)
        result.eval()
        bind_ranking_composition(ranking, controller, mapping,
            lambda record: append_jsonl(output / "composition_checks.jsonl", {**record, **identity}))
        return result

    ranking.load_model, ranking.resolve_contract = load, resolve
    info = prepared["shards"][str(shard)]
    sys.argv = [sys.argv[0], "--model-key", "qwen35_9b", "--model", model_path,
                "--adapter", str(adapter), "--view", "behavior_interest", "--label-mode", "interest_id",
                "--label-map", str(MAPPING), "--label-file", str(TRAIN),
                "--data-dir", str((ROOT / info["path"]).parent), "--run-dir", str(output),
                "--device-map", "balanced", "--num-beams", "20", "--eval-batch-size", "1",
                "--decoding", "direct_beam_search", "--max-new-tokens", "32", "--max-prompt-length", "8192"]
    if smoke:
        sys.argv.extend(["--limit", "2"])
    ranking.main()


def audit_evaluation(directory, arm, smoke, prepared):
    from crossinterest.evaluation.common import latest_details
    from crossinterest.training import evaluate_ranking as ranking

    adapter = directory / "novel/final_adapter"
    gathered, evidence, expected_rows = {}, {}, []
    for shard in (range(1) if smoke else range(8)):
        info = prepared["shards"][str(shard)]
        rows = pq.read_table(ROOT / info["path"]).to_pylist()
        if smoke:
            rows = rows[:2]
        output = directory / "ranking" / f"shard-{shard}"
        contract = json.loads((output / "run_contract.json").read_text())
        require_fields(contract, {"composition_method": METHOD, "composition_arm": arm,
            "composition_grouping": ARMS[arm], "composition_preflight_sha256": sha256_file(PREFLIGHT),
            "composition_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "composition_bridge_sha256": sha256_file(adapter / "interest_bridge.safetensors"),
            "composition_shard": shard, "dataset_sha256": info["sha256"], "label_map_sha256": MAPPING_SHA,
            "label_file_sha256": TRAIN_SHA, "sample_limit": 2 if smoke else None,
            "num_beams": 20, "num_return_sequences": 20, "do_sample": False,
            "candidates_in_prompt": False, "decoder_label_constraint": False,
            "decoding": "candidate_free_direct_beam_search", "max_new_tokens": 32,
            "eval_batch_size": 1, "label_mode": "interest_id", "view": "behavior_interest"}, "evaluation protocol")
        details = latest_details(output / "details.jsonl", contract["fingerprint"])
        checks = {item["sample_id"]: item for item in map(json.loads, (output / "composition_checks.jsonl").read_text().splitlines())}
        if set(details) != {row["sample_id"] for row in rows} or gathered.keys() & details.keys():
            raise ValueError("incomplete or overlapping evaluation coverage")
        for row in rows:
            detail = details[row["sample_id"]]
            require_fields(detail, {"target_interest": row["target_interest"], "source_sample_id": row["source_sample_id"]}, "evaluation row")
            if len(detail["ranking"]) != 20 or [item["rank"] for item in detail["ranking"]] != list(range(1, 21)):
                raise ValueError("incomplete beam list")
            if not all(math.isfinite(item["sequence_score"]) for item in detail["ranking"]):
                raise ValueError("nonfinite beam score")
            require_fields(detail, ranking.score_ranking(detail["ranking"], row["target_interest"]), "recomputed row metrics")
            require_fields(checks[row["sample_id"]], {"generate_calls": 1, "num_beams": 20,
                "groups": len(history_groups(row, ARMS[arm])), "grouping": ARMS[arm],
                "raw_history_decoder_bypass": False, "composition_preflight_sha256": sha256_file(PREFLIGHT)}, "input audit")
            if checks[row["sample_id"]]["encoder_calls"] < 1:
                raise ValueError("missing history encoding")
        shard_metrics = ranking.metrics_from_rankings(list(details.values()), rows, 20)
        require_fields(json.loads((output / "metrics.json").read_text()), shard_metrics, "shard metric recomputation")
        gathered.update(details)
        expected_rows.extend(rows)
        evidence[str(shard)] = {name: sha256_file(output / name) for name in
                               ("run_contract.json", "details.jsonl", "composition_checks.jsonl", "metrics.json", "predictions.jsonl")}
    if not smoke:
        expected_rows = pq.read_table(DEV).to_pylist()
        if set(gathered) != {row["sample_id"] for row in expected_rows}:
            raise ValueError("formal evaluation does not cover the original Dev")
    metrics = ranking.metrics_from_rankings([gathered[row["sample_id"]] for row in expected_rows], expected_rows, 20)
    health = (metrics["complete"] and metrics["missing_sample_count"] == 0
              and metrics["valid_beam_rate"] == metrics["strict_beam_format_rate"] == 1
              and metrics["duplicate_valid_beam_rate"] == 0)
    if not smoke:
        health = health and metrics["recorded_sample_count"] == 2000 and metrics["class_count"] == 112
    result = {"passed": True, "arm": arm, "smoke": smoke, "metrics": metrics,
              "health_passed": health, "evidence": evidence, "preflight_sha256": sha256_file(PREFLIGHT)}
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    return result


def execute(output, arm):
    from huggingface_hub import snapshot_download

    prepared = verify()
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "EXECUTION_CONTRACT.json", {"arm": arm, "preflight": prepared, "model_path": model_path})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TRAIN_SELECTION_")}
    environment.update(TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       HF_HOME="/tmp/huggingface", HF_HUB_CACHE="/tmp/huggingface/hub",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([sys.executable, "-m", "pytest", str(HERE / "test_interest_composition.py"), "-q"],
                   cwd=FROZEN_HERE.parent, env=environment, check=True)
    for smoke in (True, False):
        directory = output / ("smoke" if smoke else "formal")
        for stage in (("novel",) if arm == "N" else ("auxiliary", "novel")):
            stage_dir = directory / stage
            if not (stage_dir / "train_metrics.json").exists():
                subprocess.run(training_command(model_path, stage_dir, arm, stage, smoke, prepared),
                               cwd=FROZEN_HERE.parent, env=environment, check=True)
            audit_training(stage_dir, arm, stage, smoke, prepared, model_path)
        processes = []
        try:
            for shard in (range(1) if smoke else range(8)):
                command = [sys.executable, "-m", 'crossinterest.methods.composition.runner', "evaluate",
                    "--model", model_path, "--adapter", str(directory / "novel/final_adapter"),
                    "--output", str(directory / "ranking" / f"shard-{shard}"), "--arm", arm, "--shard", str(shard)]
                if smoke:
                    command.append("--smoke")
                processes.append(subprocess.Popen(command, cwd=FROZEN_HERE.parent,
                    env={**environment, "CUDA_VISIBLE_DEVICES": str(shard)}))
            for process in processes:
                if process.wait() != 0:
                    raise RuntimeError("composition evaluation worker failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        result = audit_evaluation(directory, arm, smoke, prepared)
        print(json.dumps({"event": "composition_phase_audited", "smoke": smoke,
                  "passed": True, "training": {"passed": True},
                  "evaluation_health_passed": result["health_passed"]}), flush=True)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "arm": arm,
        "formal_audit_sha256": sha256_file(output / "formal/RESULT_AUDIT.json"),
        "completed_utc": datetime.now(timezone.utc).isoformat()})


def build_launch():
    prepared = verify()
    receipt_path = FROZEN_HERE / "cir_factorization/PACKAGE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if not all(sha256_file(FROZEN_HERE.parent / path) == digest for path, digest in receipt["files"].items()):
        raise ValueError("frozen package changed")
    launch = ROOT / "launch"
    launch.mkdir(exist_ok=False)
    shutil.copyfile(HERE / "residual/launch/bootstrap.py", launch / "bootstrap.py")
    paths = [HERE / name for name in MODULES] + [PREFLIGHT, DATA_AUDIT, ROOT / "REFERENCE_AUDIT.json"]
    paths.extend(ROOT / info["path"] for info in prepared["shards"].values())
    for grouping in prepared["data"]["groups"].values():
        for info in grouping.values():
            paths.extend(ROOT / info[key]["path"] for key in ("cache", "smoke_cache", "smoke_source") if key in info)
    overlay = {}
    for source in paths:
        relative = str(source.relative_to(HERE.parent))
        if relative in receipt["files"]:
            raise ValueError("overlay cannot replace a frozen dependency")
        destination = launch / "payload" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        overlay[relative] = {"source": str(destination.relative_to(launch)), "sha256": sha256_file(source)}
    immutable_json(launch / "LAUNCH_MANIFEST.json", {"frozen_receipt_sha256": sha256_file(receipt_path), "overlay": overlay})
    return {"passed": True, "overlay_files": len(overlay), "frozen_files": len(receipt["files"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-data", "verify-data", "freeze", "verify", "train", "evaluate", "execute", "build-launch"))
    parser.add_argument("--workers", type=int, default=4)
    options, remaining = parser.parse_known_args()
    if options.action == "train":
        return train(remaining)
    if options.action in ("evaluate", "execute"):
        worker = argparse.ArgumentParser()
        worker.add_argument("--arm", required=True, choices=ARMS)
        worker.add_argument("--output", required=True, type=Path)
        if options.action == "evaluate":
            worker.add_argument("--model", required=True)
            worker.add_argument("--adapter", required=True, type=Path)
            worker.add_argument("--shard", required=True, type=int, choices=range(8))
            worker.add_argument("--smoke", action="store_true")
            args = worker.parse_args(remaining)
            return evaluate(args.model, args.adapter, args.output, args.arm, args.shard, args.smoke)
        args = worker.parse_args(remaining)
        return execute(args.output, args.arm)
    if remaining:
        parser.error(f"unexpected arguments: {remaining}")
    actions = {"prepare-data": lambda: prepare_data(options.workers), "verify-data": verify_data,
               "freeze": freeze, "verify": verify, "build-launch": build_launch}
    result = actions[options.action]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()