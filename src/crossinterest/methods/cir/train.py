"""Train frozen CIR stages while leaving previous experiment entrypoints intact."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from crossinterest.training import train as baseline
from crossinterest.methods.alignment.audit import require_fields
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.train import MechanismDataset, MechanismTrainer


def validate_stage(args, prepared: dict, stage: str, smoke: bool, preflight: Path) -> dict:
    info = prepared["files"][("smoke_" if smoke else "") + stage]
    if (args.view, args.label_mode, args.loss_mode) != ("behavior_interest", "interest_id", "causal_lm"):
        raise ValueError("CIR requires the original B+I causal protocol")
    if args.selection_file is not None or args.limit is not None or args.max_steps != -1:
        raise ValueError("CIR stages must use their complete frozen files")
    expected_epochs = prepared.get("auxiliary_epochs", 1) if stage == "auxiliary" else 1
    if info.get("epochs", 1) != expected_epochs:
        raise ValueError("CIR prepared epoch budget mismatch")
    if (args.epochs, args.learning_rate, args.expected_effective_batch_size) != (expected_epochs, 1e-4, 32):
        raise ValueError("CIR stage budget mismatch")
    if args.expected_train_rows != info["rows"] or sha256_file(args.train_file) != info["sha256"]:
        raise ValueError("CIR stage data mismatch")
    if sha256_file(args.label_map) != prepared["mapping_sha256"]:
        raise ValueError("CIR label mapping mismatch")
    if Path(args.model).name != prepared["model_revision"]:
        raise ValueError("CIR backbone must be the pinned local snapshot")
    if (stage == "novel") != (args.init_adapter is not None):
        raise ValueError("auxiliary starts fresh; novel requires its completed auxiliary adapter")
    metadata = {"task": "amazon_cir_sft", "prompt_schema_version": "interest-mechanism-v1",
                "variant": prepared["variant"], "model_key": prepared["model_key"],
                "stage": stage, "smoke": smoke, "relation_head": False,
                "preflight_sha256": sha256_file(preflight),
                "auxiliary_objective": "uniform_positive_set_ce_112" if stage == "auxiliary" else None,
                "optimizer_reset_at_handoff": True, "scheduler_reset_at_handoff": True}
    if args.init_adapter is not None:
        previous = json.loads((args.init_adapter.parent / "train_metrics.json").read_text())
        auxiliary = prepared["files"][("smoke_" if smoke else "") + "auxiliary"]
        require_fields(previous, {"global_step": auxiliary["steps"], "sample_count": auxiliary["rows"],
                                  "train_sha256": auxiliary["sha256"],
                                  "epochs": float(auxiliary.get("epochs", 1))}, "complete CIR handoff")
        require_fields(previous["experiment_metadata"], {
            key: metadata[key] for key in ("variant", "model_key", "smoke", "preflight_sha256")}, "CIR handoff identity")
        require_fields(previous["experiment_metadata"], {"stage": "auxiliary"}, "CIR handoff stage")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=("auxiliary", "novel"))
    parser.add_argument("--cir-preflight", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    options, remaining = parser.parse_known_args()
    args = baseline.parse_args(remaining)
    prepared = json.loads(options.cir_preflight.read_text())
    metadata = validate_stage(args, prepared, options.stage, options.smoke, options.cir_preflight)
    dataset_class = MechanismDataset
    if prepared.get("directional"):
        from crossinterest.methods.directionality.runner import BidirectionalDataset
        dataset_class = BidirectionalDataset
        metadata["prompt_schema_version"] = "interest-bidirectional-v1"
    baseline.main(args=args, experiment_metadata=metadata,
        dataset_class=partial(dataset_class, stage=options.stage, relation=False),
        trainer_class=partial(MechanismTrainer, stage=options.stage, relation=False, init_adapter=args.init_adapter))


if __name__ == "__main__":
    main()