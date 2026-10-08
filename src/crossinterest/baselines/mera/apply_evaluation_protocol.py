#!/usr/bin/env python3
"""Apply and record an explicit MeRA evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLLEAGUE_FORWARD = (
    "            output = model(\n"
    "                input_ids=input_ids,\n"
    "                attention_mask=inputs.attention_mask,\n"
    "            ).logits\n"
)


def apply_protocol(inference_path: Path, protocol: str) -> dict:
    source = inference_path.read_text(encoding="utf-8")
    official_forward = "            output = model(input_ids).logits\n"
    if protocol == "colleague":
        if source.count(official_forward) != 1:
            raise ValueError("official inference forward no longer has the expected form")
        source = source.replace(official_forward, COLLEAGUE_FORWARD)
        inference_path.write_text(source, encoding="utf-8")
    elif protocol != "official":
        raise ValueError(f"unsupported evaluation protocol: {protocol}")

    compile(source, str(inference_path), "exec")
    uses_attention_mask = COLLEAGUE_FORWARD in source
    expected_mask = protocol == "colleague"
    if uses_attention_mask != expected_mask:
        raise RuntimeError("evaluation attention-mask contract failed")
    return {
        "schema_version": 1,
        "protocol": protocol,
        "base_loading": "8-bit" if protocol == "colleague" else "official-default-fp16",
        "attention_mask_forwarded": uses_attention_mask,
        "ranking": "last-position logits over the complete category-token space",
        "training_reused": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--protocol", choices=("official", "colleague"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = apply_protocol(args.inference.resolve(), args.protocol)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()