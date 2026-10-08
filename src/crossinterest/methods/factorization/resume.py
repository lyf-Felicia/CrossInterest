"""Audited execution-only acceleration from immutable original CIR checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import threading
import time

from crossinterest.evaluation.common import write_json_atomic
import crossinterest.methods.factorization.runner as frozen
import crossinterest.methods.cir.runner as cir
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import HERE, sha256_file
from crossinterest.methods.factorization.speed import parallel_command
from crossinterest.methods.mechanism.runner import immutable_json


@contextmanager
def preserve_training_state(trainer, model):
    import numpy as np
    import torch

    random_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    names = ("factor_totals", "factor_rows", "factor_stat_names")
    saved = {name: deepcopy(getattr(trainer, name)) for name in names if hasattr(trainer, name)}
    try:
        yield
    finally:
        model.zero_grad(set_to_none=True)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for name in names:
            if name in saved:
                setattr(trainer, name, saved[name])
            elif hasattr(trainer, name):
                delattr(trainer, name)


def validate_layout(runtime, world_size, checkpointing):
    require_fields(runtime, {"world_size": world_size, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 32 // world_size, "effective_batch_size": 32,
        "gradient_checkpointing": checkpointing, "ddp_find_unused_parameters": False}, "accelerated runtime")
    require_fields(runtime["qwen_fast_path"], {"available": True}, "Qwen kernels")


def audit_stage(directory, stage, prepared, model_path, arm, performance):
    import torch
    from safetensors.torch import load_file

    info = prepared["files"][stage]
    metrics = json.loads((directory / "train_metrics.json").read_text())
    expected = {"model": model_path, "train_sha256": info["sha256"], "label_map_sha256": cir.MAPPING_SHA,
        "view": "behavior_interest", "label_mode": "interest_id", "epochs": float(info.get("epochs", 1)),
        "learning_rate": 1e-4, "effective_batch_size": 32, "loss_mode": "causal_lm",
        "selection_file": None, "sample_limit": None, "semantic_label_token_weight": 1.0}
    factor_identity = {"factor_arm": arm,
        "factor_manifest_sha256": sha256_file(frozen.ROOT / "data/MANIFEST.json"),
        "factor_masks_sha256": sha256_file(frozen.ROOT / "data/masks.npz"),
        "factor_loss_sha256": sha256_file(HERE / "cir_factor_loss.py")}
    for record in (metrics, json.loads((directory / "train_contract.json").read_text())):
        require_fields(record, expected, "accelerated CIR stage")
        require_fields(record["experiment_metadata"], {"stage": stage, "smoke": False, "relation_head": False,
            "variant": prepared["variant"], "model_key": prepared["model_key"],
            "preflight_sha256": sha256_file(cir.preflight_path("qwen35_9b", "external_balanced", 3)),
            "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True,
            **factor_identity}, "accelerated CIR identity")
        previous = directory.parent / "auxiliary/final_adapter/adapter_model.safetensors"
        require_fields(record, {"init_adapter_model_sha256": sha256_file(previous) if stage == "novel" else None}, "handoff")
    require_fields(metrics, {"global_step": info["steps"], "max_steps": info["steps"],
                            "epoch": float(info.get("epochs", 1)), "sample_count": info["rows"]}, "completion")
    if not math.isfinite(metrics["train_loss"]) or metrics["train_loss"] <= 0:
        raise ValueError("invalid resumed training loss")
    runtime = json.loads((directory / "runtime_config.json").read_text())
    validate_layout(runtime, performance["world_size"], performance["gradient_checkpointing"])
    mechanism = json.loads((directory / "MECHANISM_RUNTIME.json").read_text())
    require_fields(mechanism, {"stage": stage, "relation_trainable_parameters": 0}, "mechanism runtime")
    if stage == "auxiliary":
        require_fields(mechanism, {"model_accepts_loss_kwargs": False,
            "auxiliary_mean_loss_scaled_by_trainer": True}, "loss scaling")
        require_fields(json.loads((directory / "final_adapter/FACTOR_RUNTIME.json").read_text()),
                       {"arm": arm, "model_accepts_loss_kwargs": False}, "factor runtime")
    adapter = directory / "final_adapter"
    require_fields(json.loads((adapter / "adapter_config.json").read_text()),
                   {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}, "LoRA")
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("invalid accelerated adapter")
    if not any(value.count_nonzero().item() for name, value in weights.items() if "lora_B" in name):
        raise ValueError("accelerated adapter did not update")
    result = {"passed": True, "stage": stage, "rows": info["rows"], "steps": info["steps"],
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"), "relation_sha256": None,
        "train_runtime_seconds": metrics["train_runtime"], "mechanism_runtime": mechanism}
    write_json_atomic(directory / "STAGE_AUDIT.json", result)
    result.update(factor_arm=arm, factor_metadata=factor_identity,
        performance_contract_sha256=sha256_file(directory.parents[1] / "PERFORMANCE_CONTRACT.json"))
    write_json_atomic(directory / "FACTOR_STAGE_AUDIT.json", result)
    return result


def copy_mutable_tree(source, destination):
    shutil.copytree(source, destination, copy_function=shutil.copyfile)
    for directory in (destination, *[path for path in destination.rglob("*") if path.is_dir()]):
        directory.chmod(directory.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)


def copy_checkpoint(source, destination, arm, world_size):
    import torch
    from safetensors.torch import load_file

    candidates = sorted(source.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]), reverse=True)
    selected = None
    for checkpoint in candidates:
        required = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "optimizer.pt",
                    "scheduler.pt", "FACTOR_RUNTIME.json", *[f"rng_state_{rank}.pth" for rank in range(4)]]
        if all((checkpoint / name).is_file() for name in required):
            selected = checkpoint
            break
    if selected is None:
        raise ValueError("no complete original checkpoint")
    target = destination / selected.name
    if target.exists():
        raise ValueError("refusing to overwrite a staged checkpoint")
    copy_mutable_tree(selected, target)
    hashes = {name: sha256_file(target / name) for name in required}
    if any(sha256_file(selected / name) != expected for name, expected in hashes.items()):
        raise ValueError("source checkpoint changed during copy")
    state = json.loads((target / "trainer_state.json").read_text())
    step = int(selected.name.split("-")[-1])
    require_fields(state, {"global_step": step, "max_steps": 7866, "num_train_epochs": 3}, "saved Trainer state")
    require_fields(json.loads((target / "FACTOR_RUNTIME.json").read_text()),
                   {"arm": arm, "model_accepts_loss_kwargs": False}, "saved arm")
    optimizer = torch.load(target / "optimizer.pt", map_location="cpu", weights_only=True)
    for values in optimizer["state"].values():
        if int(values["step"]) != step or not all(torch.isfinite(value).all() for value in values.values() if torch.is_tensor(value)):
            raise ValueError("saved optimizer is incomplete")
    scheduler = torch.load(target / "scheduler.pt", map_location="cpu", weights_only=True)
    if scheduler["last_epoch"] != step:
        raise ValueError("saved scheduler step differs")
    weights = load_file(str(target / "adapter_model.safetensors"))
    if not weights or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("saved weights are invalid")
    original_rng = []
    for rank in range(4):
        value = torch.load(target / f"rng_state_{rank}.pth", map_location="cpu", weights_only=False)
        if not {"python", "numpy", "cpu", "cuda"} <= value.keys():
            raise ValueError("saved random state is incomplete")
        original_rng.append(value)
    archive = destination.parent.parent / "ORIGINAL_RNG"
    archive.mkdir(exist_ok=True)
    for rank in range(4):
        shutil.copyfile(target / f"rng_state_{rank}.pth", archive / f"rng_state_{rank}.pth")
    for rank in range(world_size):
        value = dict(original_rng[rank % 4])
        cuda_states = value["cuda"]
        if not isinstance(cuda_states, list) or len(cuda_states) != 4:
            raise ValueError("unexpected original CUDA RNG topology")
        value["cuda"] = [cuda_states[index % 4].clone() for index in range(world_size)]
        torch.save(value, target / f"rng_state_{rank}.pth")
    return {"passed": True, "source": str(selected), "checkpoint_step": step, "original_sha256": hashes,
        "world_size_before": 4, "world_size_after": world_size,
        "rng_policy": "preserve_original_states_map_rank_modulo_4_expand_cuda_state_list",
        "bitwise_trajectory_equivalence_claimed": False,
        "staged_sha256": {path.name: sha256_file(path) for path in target.iterdir() if path.is_file()}}


class StressCheckComplete(Exception):
    pass


def stress_callback(trainer, check_only=False):
    import torch
    from transformers import TrainerCallback

    class StressCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, model, **kwargs):
            with preserve_training_state(self.trainer, model):
                self.check(args, state, model)
            if check_only:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                raise StressCheckComplete()

        def check(self, args, state, model):
            sample = dict(self.trainer.train_dataset[0])
            target = next(index for index, label in enumerate(sample["labels"]) if label != -100)
            extra = 3252 - len(sample["input_ids"])
            if extra < 0:
                raise ValueError("stress length is shorter than a real training sample")
            sample["input_ids"] = sample["input_ids"][:target] + [sample["input_ids"][0]] * extra + sample["input_ids"][target:]
            sample["labels"] = [-100] * extra + sample["labels"]
            sample["attention_mask"] = [1] * len(sample["input_ids"])
            batch = self.trainer._prepare_inputs(self.trainer.data_collator([sample]))
            causal_inputs = {name: value for name, value in batch.items()
                             if name in {"input_ids", "attention_mask", "labels"}}
            torch.cuda.reset_peak_memory_stats()
            with self.trainer.compute_loss_context_manager():
                loss = self.trainer.compute_loss(model, dict(batch))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite worst-length stress loss")
            loss.backward()
            model.zero_grad(set_to_none=True)
            del loss
            with self.trainer.compute_loss_context_manager():
                causal_output = model(**causal_inputs)
            if not torch.isfinite(causal_output.loss):
                raise ValueError("nonfinite original causal-CE stress loss")
            causal_output.loss.backward()
            torch.cuda.synchronize()
            write_json_atomic(Path(args.output_dir) / f"STRESS_RANK_{args.process_index}.json", {
                "passed": True, "tokens": 3252, "optimizer_updates": 0,
                "restored_checkpoint_step": state.global_step,
                "restored_optimizer_steps": sorted({int(values["step"]) for values in self.trainer.optimizer.state.values()
                                                    if "step" in values}),
                "tested_objectives": ["stage_loss", "original_full_vocabulary_causal_ce"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            model.zero_grad(set_to_none=True)
            del causal_output, causal_inputs, batch

    return StressCallback(trainer)


def stress_training(check_only=False):
    import torch
    from crossinterest.training import train as baseline
    import crossinterest.methods.factorization.train as cir_factor_train

    original = baseline.main

    def main(*args, trainer_class, **kwargs):
        def factory(*trainer_args, **trainer_kwargs):
            trainer = trainer_class(*trainer_args, **trainer_kwargs)
            trainer.add_callback(stress_callback(trainer, check_only))
            return trainer
        return original(*args, trainer_class=factory, **kwargs)

    baseline.main = main
    try:
        cir_factor_train.main()
    except StressCheckComplete:
        print(json.dumps({"event": "checkpoint_stress_only_passed", "optimizer_updates": 0}), flush=True)
    finally:
        if check_only and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def validate_stress_gate(records, step, world_size=8):
    if len(records) != world_size:
        raise ValueError("all ranks must pass the pre-handoff stress gate")
    for row in records:
        require_fields(row, {"passed": True, "tokens": 3252, "optimizer_updates": 0,
            "restored_checkpoint_step": step, "restored_optimizer_steps": [step],
            "tested_objectives": ["stage_loss", "original_full_vocabulary_causal_ce"]}, "pre-handoff stress")
        if not 0 < row["peak_allocated_bytes"] < 75 * 1024**3:
            raise ValueError("insufficient worst-length memory headroom")


def accelerated_training_command(model, stage, stage_dir, prepared, performance, previous=None, check_only=False):
    command = frozen.training_command(model, stage, stage_dir, False, prepared, performance["arm"], previous)
    command = parallel_command(command, performance["world_size"], performance["gradient_checkpointing"])
    command[command.index("--save-steps") + 1] = "100"
    index = command.index('crossinterest.methods.factorization.train')
    command[index:index + 1] = ['crossinterest.methods.factorization.resume',
                              "stress-check" if check_only else "stress-training"]
    return command


def validate_benchmark(result, world_size, checkpointing):
    if world_size != 8 or result.get("scope") != "not_formal_results":
        raise ValueError("an isolated eight-GPU benchmark is required")
    name = "eight_gc" if checkpointing else "eight_no_gc"
    for key, count, checked_gc in (("four_gc", 4, True), (name, world_size, checkpointing)):
        profile = result["profiles"][key]
        if not profile["completed"] or profile["returncode"] != 0 or len(profile["ranks"]) != count:
            raise ValueError("completed real-model throughput gate is required")
        if {row["rank"] for row in profile["ranks"]} != set(range(count)):
            raise ValueError("benchmark rank coverage differs")
        for row in profile["ranks"]:
            require_fields(row, {"world_size": count, "gradient_accumulation_steps": 32 // count,
                "gradient_checkpointing": checked_gc, "effective_batch_size": 32}, "benchmark layout")
            require_fields(row["sampler_configuration"], {"use_seedable_sampler": True,
                "even_batches": True, "split_batches": False}, "benchmark sampler")
            if len(row["steps"]) < 24 or not math.isfinite(row["median_step_seconds"]) or row["median_step_seconds"] <= 0:
                raise ValueError("benchmark timing is incomplete")
            stress = row["stress"]
            require_fields(stress, {"passed": True, "tokens": 3252, "optimizer_updates": 0,
                "tested_objectives": ["stage_loss", "original_full_vocabulary_causal_ce"]}, "benchmark stress")
            if not 0 < stress["peak_allocated_bytes"] < 75 * 1024**3:
                raise ValueError("benchmark memory headroom is insufficient")
    original = max(row["median_step_seconds"] for row in result["profiles"]["four_gc"]["ranks"])
    accelerated = max(row["median_step_seconds"] for row in result["profiles"][name]["ranks"])
    ratio = original / accelerated
    if ratio < 1.5:
        raise ValueError("measured speedup does not justify migration")
    return {"profile": name, "original_step_seconds": original, "accelerated_step_seconds": accelerated,
            "measured_speedup": ratio, "minimum_speedup": 1.5}


def prepare(source, output, arm, world_size, checkpointing, benchmark):
    frozen.verify_package()
    if (output / "PERFORMANCE_CONTRACT.json").exists():
        contract = json.loads((output / "PERFORMANCE_CONTRACT.json").read_text())
        require_fields(contract, {"arm": arm, "world_size": world_size, "gradient_checkpointing": checkpointing,
            "source": str(source), "benchmark_sha256": sha256_file(benchmark),
            "execution_code_sha256": sha256_file(Path(__file__))}, "resume preparation")
        return contract
    result = json.loads(benchmark.read_text())
    benchmark_gate = validate_benchmark(result, world_size, checkpointing)
    execution = json.loads((source / "EXECUTION_CONTRACT.json").read_text())
    require_fields(execution, {"factor_arm": arm, "code_sha256": frozen.sources(),
        "factor_preflight_sha256": sha256_file(frozen.ROOT / "PREFLIGHT.json"),
        "fresh_formal_after_smoke": True}, "original identity")
    if output.exists() and any(path.name != "logs" for path in output.iterdir()):
        raise ValueError("new accelerated output must not exist")
    output.mkdir(parents=True, exist_ok=True)
    copy_mutable_tree(source / "smoke", output / "smoke")
    for name in ("EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(benchmark, output / "PERFORMANCE_BENCHMARK.json")
    frozen.audit_output(output, smoke_only=True)
    stage = output / "formal/auxiliary"
    stage.mkdir(parents=True)
    migration = copy_checkpoint(source / "formal/auxiliary", stage, arm, world_size)
    immutable_json(output / "RESUME_AUDIT.json", migration)
    contract = {"arm": arm, "world_size": world_size, "gradient_checkpointing": checkpointing,
        "effective_batch_size": 32, "budget_unchanged": True, "source": str(source),
        "original_execution_sha256": sha256_file(source / "EXECUTION_CONTRACT.json"),
        "original_smoke_sha256": sha256_file(source / "smoke/RESULT_AUDIT.json"),
        "benchmark_sha256": sha256_file(benchmark), "resume_audit_sha256": sha256_file(output / "RESUME_AUDIT.json"),
        "execution_code_sha256": sha256_file(Path(__file__)),
        "parallel_command_sha256": sha256_file(Path(__file__).with_name("cir_factor_speed.py")),
        "benchmark_gate": benchmark_gate,
        "bitwise_trajectory_equivalence_claimed": False}
    immutable_json(output / "PERFORMANCE_CONTRACT.json", contract)
    print(json.dumps({"event": "accelerated_checkpoint_ready", "arm": arm, "step": migration["checkpoint_step"]}), flush=True)
    return contract


def execute(output):
    from huggingface_hub import snapshot_download

    frozen.verify_package()
    performance = json.loads((output / "PERFORMANCE_CONTRACT.json").read_text())
    if performance["execution_code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("accelerated orchestration changed after preparation")
    prepared = frozen.preflight()["original"]
    model = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"], cache_dir="/tmp/huggingface/hub")
    execution = json.loads((output / "EXECUTION_CONTRACT.json").read_text())
    if execution["model_path"] != model:
        raise ValueError("pinned model location differs from original execution")
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if not (output / "HANDOFF_AUTHORIZATION.json").exists():
        stage_dir = output / "formal/auxiliary"
        command = accelerated_training_command(model, "auxiliary", stage_dir, prepared, performance, check_only=True)
        subprocess.run(command, env=environment, check=True)
        migration = json.loads((output / "RESUME_AUDIT.json").read_text())
        records = [json.loads((stage_dir / f"STRESS_RANK_{rank}.json").read_text()) for rank in range(performance["world_size"])]
        validate_stress_gate(records, migration["checkpoint_step"], performance["world_size"])
        immutable_json(output / "READY_FOR_HANDOFF.json", {"passed": True, "optimizer_updates": 0,
            "checkpoint_step": migration["checkpoint_step"], "stress": records,
            "performance_contract_sha256": sha256_file(output / "PERFORMANCE_CONTRACT.json")})
    deadline = time.monotonic() + 1800
    waiter = threading.Event()
    while not (output / "HANDOFF_AUTHORIZATION.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("old job was not stopped and authorized within 30 minutes; no training performed")
        print(json.dumps({"event": "waiting_for_exclusive_handoff", "arm": performance["arm"]}), flush=True)
        waiter.wait(15)
    authorization = json.loads((output / "HANDOFF_AUTHORIZATION.json").read_text())
    require_fields(authorization, {"original_job_stopped": True,
        "performance_contract_sha256": sha256_file(output / "PERFORMANCE_CONTRACT.json")}, "exclusive handoff")
    readiness = json.loads((output / "READY_FOR_HANDOFF.json").read_text())
    require_fields(readiness, {"passed": True, "optimizer_updates": 0,
        "performance_contract_sha256": sha256_file(output / "PERFORMANCE_CONTRACT.json")}, "handoff readiness")
    validate_stress_gate(readiness["stress"], readiness["checkpoint_step"], performance["world_size"])
    directory = output / "formal"
    stages, previous = {}, None
    for stage in ("auxiliary", "novel"):
        stage_dir = directory / stage
        if not (stage_dir / "train_metrics.json").exists():
            command = accelerated_training_command(model, stage, stage_dir, prepared, performance, previous)
            subprocess.run(command, env=environment, check=True)
        stages[stage] = audit_stage(stage_dir, stage, prepared, model, performance["arm"], performance)
        previous = stage_dir / "final_adapter"
        if stage == "auxiliary":
            subprocess.run([sys.executable, "-m", 'crossinterest.methods.factorization.probe', "--model", model,
                "--adapter", str(previous), "--output", str(directory / "auxiliary_PROBE.json")], env=environment, check=True)
    processes = []
    commands = [([sys.executable, "-m", 'crossinterest.methods.factorization.probe', "--model", model,
        "--adapter", str(previous), "--output", str(directory / "novel_PROBE.json")], "6,7")]
    for evaluation, devices in (("classification", "0,1"), ("ranking", "2,3,4,5")):
        command = [sys.executable, "-m", 'crossinterest.methods.mechanism.evaluate', "--evaluation", evaluation,
            "--model-key", "qwen35_9b", "--model", model, "--adapter", str(previous), "--view", "behavior_interest",
            "--label-mode", "interest_id", "--label-map", str(cir.MAPPING), "--data-dir", str(cir.DEV.parent),
            "--expected-label-count", "112", "--run-dir", str(directory / evaluation), "--device-map", "balanced"]
        if evaluation == "ranking":
            command.extend(["--label-file", str(cir.TRAIN), "--num-beams", "20", "--eval-batch-size", "1",
                            "--decoding", "direct_beam_search"])
        commands.append((command, devices))
    for command, devices in commands:
        processes.append(subprocess.Popen(command, env=dict(environment, CUDA_VISIBLE_DEVICES=devices)))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"parallel evaluation failed: {codes}")
    probes = {stage: frozen.audit_probe(directory / f"{stage}_PROBE.json", directory / stage / "final_adapter")
              for stage in ("auxiliary", "novel")}
    result = cir.audit_evaluation(directory, "reconstruction", False, stages["novel"])
    result.update(factor_arm=performance["arm"], stage_audits=stages, probes=probes,
                  performance_contract_sha256=sha256_file(output / "PERFORMANCE_CONTRACT.json"))
    write_json_atomic(directory / "RESULT_AUDIT.json", result)
    write_json_atomic(output / "COMPLETE.json", {"passed": True, "arm": performance["arm"],
        "smoke_only": False,
        "formal_audit_sha256": sha256_file(directory / "RESULT_AUDIT.json"),
        "smoke_audit_sha256": sha256_file(output / "smoke/RESULT_AUDIT.json"),
        "performance_contract_sha256": sha256_file(output / "PERFORMANCE_CONTRACT.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "stress-training", "stress-check"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--benchmark", type=Path)
    options, remaining = parser.parse_known_args()
    if options.action in ("stress-training", "stress-check"):
        sys.argv = [sys.argv[0], *remaining]
        stress_training(check_only=options.action == "stress-check")
    elif remaining or options.output is None:
        parser.error("--output is required and unknown arguments are forbidden")
    elif options.action == "prepare":
        if options.source is None or options.arm is None or options.benchmark is None:
            parser.error("prepare requires --source, --arm, --benchmark")
        prepare(options.source, options.output, options.arm, options.world_size, options.checkpointing, options.benchmark)
    else:
        execute(options.output)