"""Isolated throughput measurements using the frozen CIR training implementation."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


def replace_option(command, option, value):
    command = list(command)
    for index, item in enumerate(command):
        if item == option:
            command[index + 1] = str(value)
            return command
        if item.startswith(option + "="):
            command[index] = option + "=" + str(value)
            return command
    raise ValueError(f"missing frozen option: {option}")


def parallel_command(command, world_size, checkpointing=True):
    if world_size not in (4, 8, 16) or 32 % world_size:
        raise ValueError("only verified effective-batch-32 layouts are allowed")
    option = next((item.split("=")[0] for item in command
                   if item.split("=")[0] in ("--nproc_per_node", "--nproc-per-node")), None)
    if option is None:
        raise ValueError("frozen command has no distributed process count")
    result = replace_option(command, option, world_size)
    result = replace_option(result, "--gradient-accumulation-steps", 32 // world_size)
    if not checkpointing:
        index = result.index("--gradient-checkpointing")
        result[index] = "--no-gradient-checkpointing"
    return result


def benchmark_training(steps):
    import torch
    from transformers import TrainerCallback
    from crossinterest.training import train as baseline
    import crossinterest.methods.factorization.train as cir_factor_train
    from crossinterest.methods.factorization.resume import stress_callback
    from crossinterest.evaluation.common import write_json_atomic

    class TimingCallback(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
            self.records = []
            self.telemetry = []
            self.stop = threading.Event()
            self.worker = None

        def sample_gpus(self):
            while not self.stop.is_set():
                try:
                    output = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
                    self.telemetry.append({"time": time.time(), "gpus": list(csv.reader(StringIO(output.stdout)))})
                except (OSError, subprocess.SubprocessError) as error:
                    self.telemetry.append({"error": type(error).__name__})
                self.stop.wait(2)

        def on_train_begin(self, args, state, control, **kwargs):
            stress_callback(self.trainer).on_train_begin(args, state, control, **kwargs)
            torch.cuda.reset_peak_memory_stats()
            self.previous_end = time.perf_counter()
            if args.process_index == 0:
                self.worker = threading.Thread(target=self.sample_gpus, daemon=True)
                self.worker.start()

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            finished = time.perf_counter()
            self.records.append({"step": state.global_step, "seconds": finished - self.previous_end,
                                 "compute_seconds": finished - self.started,
                                 "between_steps_seconds": self.started - self.previous_end})
            self.previous_end = finished
            control.should_save = False
            if len(self.records) >= steps:
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.stop.set()
            if self.worker is not None:
                self.worker.join(timeout=12)
            measured = self.records[min(4, len(self.records) // 2):]
            result = {"scope": "throughput_only_not_formal_result", "steps": self.records,
                      "median_step_seconds": statistics.median(item["seconds"] for item in measured),
                      "mean_step_seconds": statistics.mean(item["seconds"] for item in measured),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "gpu": torch.cuda.get_device_name(), "rank": args.process_index,
                      "world_size": args.world_size, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                      "gradient_checkpointing": args.gradient_checkpointing,
                      "stress": json.loads((Path(args.output_dir) / f"STRESS_RANK_{args.process_index}.json").read_text()),
                      "sampler_configuration": args.accelerator_config.to_dict(),
                      "gpu_telemetry": self.telemetry,
                      "effective_batch_size": args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps}
            write_json_atomic(Path(args.output_dir) / f"THROUGHPUT_RANK_{args.process_index}.json", result)

    original_main = baseline.main

    def timed_main(*args, trainer_class, **kwargs):
        def trainer_factory(*trainer_args, **trainer_kwargs):
            trainer = trainer_class(*trainer_args, **trainer_kwargs)
            trainer.add_callback(TimingCallback(trainer))
            return trainer
        return original_main(*args, trainer_class=trainer_factory, **kwargs)

    baseline.main = timed_main
    cir_factor_train.main()


def benchmark(output, steps):
    from huggingface_hub import snapshot_download
    import crossinterest.methods.factorization.runner as frozen
    from crossinterest.evaluation.common import write_json_atomic

    frozen.verify_package()
    prepared = frozen.preflight()["original"]
    model = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"],
                              cache_dir="/tmp/huggingface/hub")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    profiles = (("four_gc", 4, True), ("eight_gc", 8, True), ("eight_no_gc", 8, False))
    results = {}
    for name, world_size, checkpointing in profiles:
        directory = output / name
        if directory.exists():
            raise ValueError(f"benchmark output must be fresh: {directory}")
        command = frozen.training_command(model, "auxiliary", directory, False, prepared, "relation_true")
        command = parallel_command(command, world_size, checkpointing)
        command[command.index('crossinterest.methods.factorization.train')] = 'crossinterest.methods.factorization.speed'
        module_index = command.index('crossinterest.methods.factorization.speed')
        command[module_index + 1:module_index + 1] = ["benchmark-training", "--benchmark-steps", str(steps)]
        print(json.dumps({"event": "throughput_profile_start", "profile": name, "command": command}), flush=True)
        process = subprocess.run(command, env=environment, check=False)
        records = [json.loads(path.read_text()) for path in sorted(directory.glob("THROUGHPUT_RANK_*.json"))]
        passed = process.returncode == 0 and len(records) == world_size
        results[name] = {"completed": passed, "returncode": process.returncode, "ranks": records}
        write_json_atomic(output / "THROUGHPUT_RESULTS.json", {"scope": "not_formal_results", "profiles": results})
        print(json.dumps({"event": "throughput_profile_complete", "profile": name, "completed": passed,
                          "slowest_rank_median_seconds": max((record["median_step_seconds"] for record in records), default=None)}), flush=True)
        if not passed:
            raise RuntimeError(f"benchmark profile failed: {name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("benchmark", "benchmark-training"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-steps", type=int, default=24)
    options, remaining = parser.parse_known_args()
    if options.benchmark_steps < 8:
        parser.error("at least eight steps are required")
    if options.action == "benchmark-training":
        sys.argv = [sys.argv[0], *remaining]
        benchmark_training(options.benchmark_steps)
    elif options.output is None or remaining:
        parser.error("benchmark requires --output without training arguments")
    else:
        benchmark(options.output, options.benchmark_steps)