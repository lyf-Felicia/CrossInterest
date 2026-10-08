"""Attach a role head without changing the frozen full-vocabulary evaluators."""

import argparse
import json
import sys
from pathlib import Path

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mechanism.runner import immutable_json
from crossinterest.methods.role_projection.model import RoleProjectionController


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluation", choices=("classification", "ranking"), required=True)
    parser.add_argument("--role-mode", choices=("baseline", "unconstrained", "projected"), required=True)
    options, remaining = parser.parse_known_args(argv)
    if options.evaluation == "ranking":
        from crossinterest.training import evaluate_ranking as evaluation
    else:
        from crossinterest.training import evaluate as evaluation
    args = evaluation.parse_args(remaining)
    adapter = args.adapter.resolve()
    has_head = (adapter / "interest_relation.safetensors").is_file()
    if has_head != (options.role_mode != "baseline"):
        raise ValueError("role head availability differs from the evaluator")
    contract = {"role_mode": options.role_mode, "evaluation": options.evaluation,
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "head_sha256": sha256_file(adapter / "interest_relation.safetensors") if has_head else None,
                "metadata_sha256": sha256_file(adapter / "interest_relation.json") if has_head else None,
                "code_sha256": sha256_file(Path(__file__).with_name("role_projection.py"))}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(args.run_dir / "ROLE_CONTRACT.json", contract)
    original_load, original_render = evaluation.load_model, evaluation.render_prompt_ids
    controllers = []

    def load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs):
        model = original_load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs)
        if has_head:
            controller = RoleProjectionController(model, tokenizer, mapping, adapter_path,
                                                  mode=options.role_mode, stage="novel")
            controller.head.eval()
            controllers.append(controller)
        return model

    def render(tokenizer, row, *args, **kwargs):
        ids = original_render(tokenizer, row, *args, **kwargs)
        if has_head:
            controllers[0].set_frames([{"categories": list(row["historical_interests"]),
                                        "prediction_position": len(ids) - 1}])
        return ids

    evaluation.load_model, evaluation.render_prompt_ids = load, render
    sys.argv = [sys.argv[0], *remaining]
    try:
        evaluation.main()
    finally:
        evaluation.load_model, evaluation.render_prompt_ids = original_load, original_render
        for controller in controllers:
            controller.close()
    write_json_atomic(args.run_dir / "ROLE_EVAL_AUDIT.json", {
        "passed": True, "contract": contract,
        "head_applications": sum(controller.applied_calls for controller in controllers),
        "run_contract_sha256": sha256_file(args.run_dir / "run_contract.json"),
        "metrics_sha256": sha256_file(args.run_dir / "metrics.json")})


if __name__ == "__main__":
    main()