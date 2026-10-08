"""Attach the saved relation head to the unchanged classification/20-beam evaluators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.model import RelationController, history_frame


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluation", choices=("classification", "ranking"), required=True)
    parser.add_argument("--relation", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.evaluation == "ranking":
        from crossinterest.training import evaluate_ranking as evaluation
    else:
        from crossinterest.training import evaluate as evaluation
    sys.argv = [sys.argv[0], *remaining]
    args = evaluation.parse_args(remaining)
    adapter = args.adapter.resolve()
    exists = (adapter / "interest_relation.safetensors").exists()
    if exists != options.relation:
        raise ValueError("relation head availability differs from the requested evaluator")
    contract = {"schema_version": 1, "relation": options.relation,
                "adapter_weight_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "head_weight_sha256": sha256_file(adapter / "interest_relation.safetensors") if exists else None,
                "head_config_sha256": sha256_file(adapter / "interest_relation.json") if exists else None,
                "evaluation": options.evaluation, "model_code_sha256": sha256_file(Path(__file__).with_name("mechanism_model.py"))}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    sidecar = args.run_dir / "MECHANISM_CONTRACT.json"
    if sidecar.exists() and json.loads(sidecar.read_text()) != contract:
        raise ValueError("evaluation checkpoint identity changed; refusing stale result reuse")
    write_json_atomic(sidecar, contract)
    controllers = []
    original_load = evaluation.load_model
    original_render = evaluation.render_prompt_ids

    def load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs):
        model = original_load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs)
        if options.relation:
            controllers.append(RelationController(model, tokenizer, mapping, directory=adapter_path))
            controllers[0].head.eval()
        return model

    def render(tokenizer, row, *args, **kwargs):
        ids = original_render(tokenizer, row, *args, **kwargs)
        if options.relation:
            controllers[0].set_frames([history_frame(tokenizer, ids, row, len(ids) - 1)])
        return ids

    evaluation.load_model = load
    evaluation.render_prompt_ids = render
    evaluation.main()
    write_json_atomic(args.run_dir / "MECHANISM_EVAL_AUDIT.json", {
        "passed": True, "checkpoint_contract": contract,
        "relation_forward_applications": controllers[0].applied_calls if controllers else 0,
        "run_contract_sha256": sha256_file(args.run_dir / "run_contract.json"),
        "metrics_sha256": sha256_file(args.run_dir / "metrics.json"),
    })


if __name__ == "__main__":
    main()