"""N-only training from a verified, completed CIR checkpoint."""

from crossinterest.paths import source_workspace as _source_workspace

import argparse
from functools import partial
import json
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.cir.train import validate_stage
from crossinterest.methods.mass_readout.model import MassReadoutController
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismTrainer


class MassDataset(MechanismDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, stage="novel", relation=False, **kwargs)

    def __getitem__(self, index):
        encoded = super().__getitem__(index)
        position = next(index for index, label in enumerate(encoded["labels"]) if label != -100) - 1
        encoded["history_frame"] = {"prediction_position": position}
        return encoded


class GradientAudit(TrainerCallback):
    def __init__(self, controller):
        self.controller = controller

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or state.global_step not in (0, 1, 9, 99, 499, 999, 1999, 3499):
            return
        gradient = self.controller.head.weight.grad
        if gradient is None or not gradient.isfinite().all() or gradient.norm() == 0:
            raise ValueError("missing, zero or nonfinite readout gradient")
        gradient = gradient.detach().float().cpu()
        gradient = gradient - gradient.mean(0, keepdim=True)
        output = model.get_output_embeddings()
        left = output.lora_B["default"].weight.detach().float().cpu()[self.controller.label_ids]
        left = left - left.mean(0, keepdim=True)
        right = output.lora_A["default"].weight.detach().float().cpu()
        left_basis = torch.linalg.svd(left, full_matrices=False).U[:, :torch.linalg.matrix_rank(left)]
        right_basis = torch.linalg.svd(right, full_matrices=False).Vh[:torch.linalg.matrix_rank(right)].T
        normal = gradient - left_basis @ (left_basis.T @ gradient)
        normal = normal - (normal @ right_basis) @ right_basis.T
        spectrum = torch.linalg.svdvals(gradient)
        write_json_atomic(Path(args.output_dir) / f"GRADIENT_{state.global_step:04d}.json", {
            "global_step_before_update": state.global_step, "gradient_norm": gradient.norm().item(),
            "singular_values": spectrum.tolist(),
            "tail_energy_after_rank16": spectrum[16:].square().sum().item(),
            "normal_energy_fraction": (normal.square().sum() / gradient.square().sum()).item(),
            "scope": "accumulated_readout_gradient_vs_current_lm_head_tangent_not_population_signal"})


class MassTrainer(MechanismTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, stage="novel", relation=False, init_adapter=None, **kwargs)
        self.controller = MassReadoutController(self.model, self.processing_class, self.train_dataset.mapping)
        self.add_callback(GradientAudit(self.controller))
        if self.is_world_process_zero():
            directory = Path(self.args.output_dir)
            if not (directory / "initial_readout/interest_readout.safetensors").exists():
                self.controller.save(directory / "initial_readout")
            trainable = [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]
            write_json_atomic(directory / "MASS_RUNTIME.json", {
                "metadata": self.controller.metadata,
                "head_parameters": self.controller.head.weight.numel(),
                "total_trainable_parameters": sum(value.numel() for _, value in trainable),
                "names": [name for name, _ in trainable],
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs})


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mass-contract", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    args = baseline.parse_args(remaining)
    contract = json.loads(options.mass_contract.read_text())
    root = _source_workspace('eval_interest_id_minionerec').parent
    source_path = root / contract["source_contract"]
    if sha256_file(source_path) != contract["source_contract_sha256"]:
        raise ValueError("source CIR contract changed")
    prepared = json.loads(source_path.read_text())
    if options.smoke:
        prepared["files"]["novel"] = prepared["files"]["smoke_novel"]
    if args.init_adapter is None:
        raise ValueError("completed CIR adapter is required")
    for name, expected in contract["source_adapter_files"].items():
        if sha256_file(args.init_adapter / name) != expected:
            raise ValueError(f"CIR source identity mismatch: {name}")
    if sha256_file(args.init_adapter.parent / "train_metrics.json") != contract["source_train_metrics_sha256"]:
        raise ValueError("CIR source metrics mismatch")
    previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())
    require_fields(previous["experiment_metadata"], {"role_mode": "baseline", "stage": "auxiliary"}, "CIR source")
    metadata = validate_stage(args, prepared, "novel", False, source_path)
    metadata.update(task="amazon_mass_readout_sft", smoke=options.smoke,
                    method="mass-preserving-readout-v1", source_preflight_sha256=metadata["preflight_sha256"],
                    preflight_sha256=sha256_file(options.mass_contract), relation_head=False)
    baseline.main(args=args, experiment_metadata=metadata, dataset_class=MassDataset, trainer_class=MassTrainer)
    rank = int(os.environ.get("RANK", "0"))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    write_json_atomic(Path(args.output_dir) / f"PERFORMANCE_RANK_{rank}.json", {
        "rank": rank, "world_size": int(os.environ.get("WORLD_SIZE", "1")), "gpu_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


if __name__ == "__main__":
    main()