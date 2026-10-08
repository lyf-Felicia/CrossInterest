"""Apply the trained readout inside the original full-vocabulary beam evaluator."""

import json
import sys
from pathlib import Path

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import evaluate_ranking as evaluation
from crossinterest.methods.alignment.data import sha256_file
from crossinterest.methods.mass_readout.model import MassReadoutController
from crossinterest.methods.mechanism.runner import immutable_json


def main(argv=None):
    remaining = sys.argv[1:] if argv is None else argv
    args = evaluation.parse_args(remaining)
    adapter = args.adapter.resolve()
    contract = {"architecture": "mass-preserving-readout-v1",
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "head_sha256": sha256_file(adapter / "interest_readout.safetensors"),
                "metadata_sha256": sha256_file(adapter / "interest_readout.json"),
                "code_sha256": sha256_file(Path(__file__).with_name("mass_readout.py"))}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(args.run_dir / "MASS_CONTRACT.json", contract)
    original_load, original_render = evaluation.load_model, evaluation.render_prompt_ids
    controllers = []

    def load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs):
        model = original_load(model_key, model_path, adapter_path, tokenizer, mapping, registration, **kwargs)
        controller = MassReadoutController(model, tokenizer, mapping, adapter_path)
        controller.head.eval()
        controllers.append(controller)
        return model

    def render(tokenizer, row, *args, **kwargs):
        ids = original_render(tokenizer, row, *args, **kwargs)
        controllers[0].set_frames([{"prediction_position": len(ids) - 1}])
        return ids

    evaluation.load_model, evaluation.render_prompt_ids = load, render
    original_argv = sys.argv
    sys.argv = [sys.argv[0], *remaining]
    try:
        evaluation.main()
    finally:
        sys.argv = original_argv
        evaluation.load_model, evaluation.render_prompt_ids = original_load, original_render
        for controller in controllers:
            controller.close()
    write_json_atomic(args.run_dir / "MASS_EVAL_AUDIT.json", {
        "passed": True, "contract": contract,
        "head_applications": sum(controller.applied_calls for controller in controllers),
        "run_contract_sha256": sha256_file(args.run_dir / "run_contract.json"),
        "metrics_sha256": sha256_file(args.run_dir / "metrics.json")})


if __name__ == "__main__":
    main()