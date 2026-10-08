"""Role-head stages using the frozen CIR trainer, objectives and input renderer."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path

import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.cir.train import validate_stage
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismTrainer
from crossinterest.methods.role_projection.model import RoleProjectionController


class RoleDataset(MechanismDataset):
    def __init__(self, *args, stage, role_mode, **kwargs):
        super().__init__(*args, stage=stage, relation=False, **kwargs)
        self.role_mode = role_mode

    def __getitem__(self, index):
        encoded = super().__getitem__(index)
        if self.role_mode != "baseline":
            row = self.table.slice(index, 1).to_pylist()[0]
            position = next(position for position, label in enumerate(encoded["labels"]) if label != -100) - 1
            encoded["history_frame"] = {"categories": list(row["historical_interests"]),
                                        "prediction_position": position}
        return encoded


class RoleTrainer(MechanismTrainer):
    def __init__(self, *args, stage, role_mode, init_adapter=None, **kwargs):
        super().__init__(*args, stage=stage, relation=False, init_adapter=None, **kwargs)
        if role_mode != "baseline":
            self.controller = RoleProjectionController(self.model, self.processing_class,
                self.train_dataset.mapping, directory=init_adapter, mode=role_mode, stage=stage,
                allow_auxiliary_handoff=init_adapter is not None)
        if self.is_world_process_zero():
            directory = Path(self.args.output_dir)
            if self.controller is not None:
                initial = directory / "initial_relation"
                if not (initial / "interest_relation.safetensors").exists():
                    self.controller.save(initial)
            path = directory / "trainable_parameters.json"
            parameters = json.loads(path.read_text()) if path.exists() else {}
            trainable = [(name, value) for name, value in self.model.named_parameters() if value.requires_grad]
            parameters.update(count=sum(value.numel() for _, value in trainable),
                              tensor_count=len(trainable), names=[name for name, _ in trainable])
            write_json_atomic(directory / "role_trainable_parameters.json", parameters)
            write_json_atomic(directory / "ROLE_RUNTIME.json", {
                "role_mode": role_mode, "stage": stage, "rank": 64,
                "head_metadata": self.controller.metadata if self.controller else None,
                "head_parameters": sum(value.numel() for value in self.controller.head.parameters()) if self.controller else 0,
                "head_trainable_parameters": sum(value.numel() for value in self.controller.head.parameters()
                    if value.requires_grad) if self.controller else 0,
                "total_trainable_parameters": parameters["count"],
                "model_accepts_loss_kwargs": self.model_accepts_loss_kwargs})


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("auxiliary", "novel"), required=True)
    parser.add_argument("--role-mode", choices=("baseline", "unconstrained", "projected"), required=True)
    parser.add_argument("--role-contract", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    args = baseline.parse_args(remaining)
    prepared = json.loads(options.role_contract.read_text())
    metadata = validate_stage(args, prepared, options.stage, options.smoke, options.role_contract)
    metadata.update(task="amazon_role_projection_sft", role_mode=options.role_mode,
                    relation_head=options.role_mode != "baseline", rank=64,
                    history_taxonomy="all_visible_categories", category_centering="uniform_output_taxonomy",
                    taxonomy_scaling="global_output_rms")
    if args.init_adapter is not None:
        previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())
        require_fields(previous["experiment_metadata"], {"role_mode": options.role_mode}, "role handoff")
    baseline.main(args=args, experiment_metadata=metadata,
        dataset_class=partial(RoleDataset, stage=options.stage, role_mode=options.role_mode),
        trainer_class=partial(RoleTrainer, stage=options.stage, role_mode=options.role_mode,
                              init_adapter=args.init_adapter))
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    rank = int(os.environ.get("RANK", "0"))
    write_json_atomic(Path(args.output_dir) / f"PERFORMANCE_RANK_{rank}.json", {
        "rank": rank, "world_size": int(os.environ.get("WORLD_SIZE", "1")), "gpu_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})


if __name__ == "__main__":
    main()