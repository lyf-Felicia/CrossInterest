#!/usr/bin/env python3
"""Apply an explicit training protocol to the pinned official MeRA source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossinterest.baselines.mera.core import write_json


PROTOCOLS = ("official", "author_clarified")


def replace_exact(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one occurrence, found {count}: {old!r}")
    return source.replace(old, new)


def apply_protocol(
    train_path: Path,
    protocol: str,
    disable_periodic_checkpoints: bool = False,
) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported training protocol: {protocol}")
    if disable_periodic_checkpoints and protocol != "author_clarified":
        raise ValueError(
            "periodic checkpoints may only be disabled for author_clarified training"
        )

    source = train_path.read_text(encoding="utf-8")
    if protocol == "author_clarified":
        source = replace_exact(
            source,
            "def generate_prompt_with_prompt(data_point):\n",
            '''def generate_prompt_without_prompt(data_point):
    data_seq = eval(data_point["input"][81:].strip("\\n"))
    if type(data_seq) == str:
        data_seq = [data_seq]
    else:
        data_seq = list(data_seq)
    data_str = "".join('"' + item + '"' for item in data_seq)
    return data_str + data_point["output"]


def generate_prompt_with_prompt(data_point):
''',
        )
        source = replace_exact(
            source,
            "        generate_prompt = generate_prompt_with_prompt\n",
            "        generate_prompt = generate_prompt_without_prompt\n",
        )
        source = replace_exact(
            source,
            "            load_best_model_at_end=True,\n",
            "            load_best_model_at_end=False,\n",
        )
        source = replace_exact(
            source,
            "        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],\n",
            "        callbacks = [],\n",
        )
        if disable_periodic_checkpoints:
            source = replace_exact(
                source,
                '            save_strategy="steps",\n',
                '            save_strategy="no",\n',
            )
        source = replace_exact(
            source,
            "    trainer.train(resume_from_checkpoint=resume_from_checkpoint)\n",
            '''    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    completion_contract_path = os.environ.get("MERA_TRAINING_COMPLETION_PATH")
    if completion_contract_path:
        with open(completion_contract_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "global_step": trainer.state.global_step,
                    "max_steps": trainer.state.max_steps,
                    "epoch": trainer.state.epoch,
                    "num_train_epochs": trainer.args.num_train_epochs,
                },
                handle,
                indent=2,
            )
            handle.write("\\n")
''',
        )
        compile(source, str(train_path), "exec")
        train_path.write_text(source, encoding="utf-8")

    contract = {
        "schema_version": 1,
        "protocol": protocol,
        "source": (
            "official public code"
            if protocol == "official"
            else "author clarification"
        ),
        "training_prompt": (
            "generate_prompt_with_prompt"
            if protocol == "official"
            else "generate_prompt_without_prompt"
        ),
        "training_sequence": (
            "instruction + input + output"
            if protocol == "official"
            else "concatenated quoted history categories + output category"
        ),
        "early_stopping": protocol == "official",
        "load_best_model_at_end": protocol == "official",
        "completion_contract": protocol == "author_clarified",
        "checkpoint_policy": (
            "final_adapter_only"
            if disable_periodic_checkpoints
            else "periodic_and_final_adapter"
        ),
        "periodic_checkpoints": not disable_periodic_checkpoints,
        "final_adapter_save": True,
        "source_changed": protocol == "author_clarified",
    }
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--disable-periodic-checkpoints", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = apply_protocol(
        args.train.resolve(),
        args.protocol,
        disable_periodic_checkpoints=args.disable_periodic_checkpoints,
    )
    write_json(args.manifest.resolve(), contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()