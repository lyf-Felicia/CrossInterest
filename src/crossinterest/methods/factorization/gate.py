"""Read-only, preregistered first-ID gradient intervention gate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
import torch

from crossinterest.evaluation.common import write_json_atomic
from crossinterest.training import train as baseline
from crossinterest.training.evaluate import load_model
from crossinterest.training.label_space import load_label_mapping
from crossinterest.training.semantic_tokens import initialize_semantic_token_embeddings, register_semantic_tokens
from crossinterest.methods.alignment.data import HERE, BASE, sha256_file
from crossinterest.methods.factorization.loss import ARMS, factorized_loss
from crossinterest.methods.mechanism.train import auxiliary_messages, positive_set_loss


ROOT = HERE / "cir_factorization"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
GATE_RULE = {
    "version": 1, "sample_rows": 224, "per_class": 2,
    "direction_delta_threshold": 0.01, "minimum_active_fraction": 0.10,
    "zero_gradient_norm": 1e-12, "near_zero_component_relative_norm": 1e-6,
    "gradient_norm_rtol": 5e-5, "gradient_norm_atol": 2e-6,
    "passes": "true_and_pseudo_each_active_at_base_and_at_least_one_of_aux1_aux3",
    "precision": "BF16_model_FP32_logits_and_loss_FP64_recheck",
    "selection": "fixed_train_ids_seed20260919_no_dev_no_parameter_updates",
    "failure_action": "stop_corresponding_true_and_pseudo_branch_no_weight_search",
}


def prepare_package() -> dict:
    old_root = HERE / "bidirectional"
    old_receipt = json.loads((old_root / "PACKAGE_RECEIPT.json").read_text())
    destination = ROOT / "gate_package"
    files = {}

    def copy_verified(source: Path, relative: str, expected: str):
        if sha256_file(source) != expected:
            raise ValueError(f"source hash mismatch: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"existing gate package differs: {target}")
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256_file(target) != expected:
            raise ValueError(f"package copy mismatch: {target}")
        files[relative] = expected

    for relative, info in old_receipt["files"].items():
        copy_verified(old_root / "package" / relative, relative, info["sha256"])
    for name in ("cir_factor_loss.py", "cir_factor_gate.py"):
        source = HERE / name
        copy_verified(source, f"eval_interest_id_minionerec/{name}", sha256_file(source))
    for source in sorted((ROOT / "data").iterdir()):
        if source.is_file():
            copy_verified(source, f"eval_interest_id_minionerec/cir_factorization/data/{source.name}", sha256_file(source))
    references = {
        "aux1": HERE / "cir/outputs/results_base/qwen-balanced-cir/formal/auxiliary",
        "aux3": HERE / "bidirectional/outputs/results_envfix/qwen-cir-3ep/formal/auxiliary",
    }
    for name, directory in references.items():
        audit = json.loads((directory / "STAGE_AUDIT.json").read_text())
        expected_steps = 2622 if name == "aux1" else 7866
        if not audit["passed"] or audit["rows"] != 83888 or audit["steps"] != expected_steps:
            raise ValueError(f"invalid reference stage audit: {name}")
        adapter = directory / "final_adapter"
        if sha256_file(adapter / "adapter_model.safetensors") != audit["adapter_sha256"]:
            raise ValueError(f"invalid reference adapter: {name}")
        for source in sorted(adapter.iterdir()):
            if source.is_file():
                relative = f"eval_interest_id_minionerec/cir_factorization/reference_adapters/{name}/{source.name}"
                copy_verified(source, relative, sha256_file(source))
    result = {"passed": True, "gate_rule": GATE_RULE, "old_receipt_sha256": sha256_file(old_root / "PACKAGE_RECEIPT.json"),
              "old_files_verified": len(old_receipt["files"]), "files": files,
              "total_bytes": sum((destination / relative).stat().st_size for relative in files)}
    receipt = destination / "eval_interest_id_minionerec/cir_factorization/GATE_PACKAGE_RECEIPT.json"
    write_json_atomic(receipt, result)
    write_json_atomic(ROOT / "GATE_PACKAGE_RECEIPT.json", result)
    return {key: value for key, value in result.items() if key != "files"}


def summarize_logits(logits: torch.Tensor, positives: torch.Tensor, seen: torch.Tensor,
                     pseudo: torch.Tensor) -> dict:
    result = {}
    for arm, (_, _, use_pseudo) in ARMS.items():
        selected = pseudo if use_pseudo else seen
        values = logits.detach().float().requires_grad_(True)
        loss, diagnostics = factorized_loss(values, positives, selected, arm)
        gradient = torch.autograd.grad(loss, values)[0] * len(values)
        original_values = logits.detach().float().requires_grad_(True)
        original = positive_set_loss(original_values, positives)
        original_gradient = torch.autograd.grad(original, original_values)[0] * len(values)
        if not torch.allclose(gradient.norm(dim=-1), original_gradient.norm(dim=-1),
                              rtol=GATE_RULE["gradient_norm_rtol"], atol=GATE_RULE["gradient_norm_atol"]):
            raise ValueError(f"{arm}: logit gradient norm mismatch")
        if arm == "baseline" and (not torch.equal(loss, original) or not torch.equal(gradient, original_gradient)):
            raise ValueError("G0 loss or logit gradient is not bitwise equal to original CE")
        _, precise = factorized_loss(logits.double(), positives, selected, arm)
        if not torch.allclose(diagnostics["direction_delta"].double(), precise["direction_delta"],
                              rtol=1e-3, atol=2e-6):
            raise ValueError(f"{arm}: FP32/FP64 direction calculation differs")
        denominator = diagnostics["original_gradient_norm"].clamp_min(GATE_RULE["zero_gradient_norm"])
        row_diagnostics = {name: value.detach().cpu().tolist() for name, value in diagnostics.items()}
        active = diagnostics["direction_delta"] >= GATE_RULE["direction_delta_threshold"]
        result[arm] = {
            "rows": len(values), "active_rows": int(active.sum()), "active_fraction": float(active.float().mean()),
            "active": float(active.float().mean()) >= GATE_RULE["minimum_active_fraction"],
            "direction_delta_mean": float(diagnostics["direction_delta"].mean()),
            "direction_delta_median": float(diagnostics["direction_delta"].median()),
            "gradient_cosine_mean": float(diagnostics["gradient_cosine"].mean()),
            "seen_mass_mean": float(diagnostics["seen_mass"].mean()),
            "near_zero_relation_fraction": float((diagnostics["relation_gradient_norm"] / denominator <
                                                  GATE_RULE["near_zero_component_relative_norm"]).float().mean()),
            "near_zero_exclusion_fraction": float((diagnostics["exclusion_gradient_norm"] / denominator <
                                                   GATE_RULE["near_zero_component_relative_norm"]).float().mean()),
            "fallback_rows": int(diagnostics["fallback"].sum()),
            "diagnostics": row_diagnostics,
        }
    return result


def branch_decisions(checkpoints: dict) -> dict:
    result = {}
    for branch in ("relation", "exclusion"):
        arms = (f"{branch}_true", f"{branch}_pseudo")
        passed = all(checkpoints["base"][arm]["active"] and
                     any(checkpoints[name][arm]["active"] for name in ("aux1", "aux3")) for arm in arms)
        result[branch] = {"passed": passed, "arms": list(arms),
                          "action": "proceed_to_equivalence_and_real_smoke" if passed else "stop_no_formal_training"}
    return result


def run(output: Path) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    package = json.loads((ROOT / "GATE_PACKAGE_RECEIPT.json").read_text())
    for relative, expected in package["files"].items():
        if sha256_file(HERE.parent / relative) != expected:
            raise ValueError(f"gate package changed: {relative}")
    mapping = load_label_mapping(BASE / "interest_id_semantic_mapping.json", expected_count=112)
    arrays = np.load(ROOT / "data/masks.npz", allow_pickle=False)
    ids = json.loads((ROOT / "data/gate_ids.json").read_text())
    id_indices = {sample_id: index for index, sample_id in enumerate(arrays["ids"].tolist())}
    indices = [id_indices[sample_id] for sample_id in ids]
    table = pq.read_table(HERE / "cir/data/external_balanced/train.parquet")
    rows = table.take(indices).to_pylist()
    categories = [mapping["id_to_category"][token] for token in mapping["special_tokens"]]
    if len(rows) != GATE_RULE["sample_rows"] or categories != arrays["categories"].tolist():
        raise ValueError("gate sample or category alignment differs")
    positives = torch.nn.functional.one_hot(torch.tensor(arrays["targets"][indices]), len(categories)).bool()
    seen = torch.tensor(arrays["seen"][indices])
    pseudo = torch.tensor(arrays["pseudo"][indices])
    contract = {"gate_rule": GATE_RULE, "model_revision": REVISION, "sample_ids": ids,
                "package_receipt_sha256": sha256_file(ROOT / "GATE_PACKAGE_RECEIPT.json"),
                "model_updates": 0, "official_evaluation": False}
    contract_path = output / "GATE_CONTRACT.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("existing gate contract differs")
    write_json_atomic(contract_path, contract)
    model_path = snapshot_download("Qwen/Qwen3.5-9B", revision=REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    registration = register_semantic_tokens(tokenizer, mapping)
    label_ids = [registration["token_ids"][token] for token in mapping["special_tokens"]]
    encoded = [baseline.tokenize_assistant_only(tokenizer, auxiliary_messages(row, mapping)) for row in rows]
    prompts = []
    for row in encoded:
        first = next(index for index, label in enumerate(row["labels"]) if label != -100)
        if first <= 0 or len(row["input_ids"]) > 8192:
            raise ValueError("gate prompt target position or length invalid")
        prompts.append(row["input_ids"][:first])
    checkpoints = {}
    for name in ("base", "aux1", "aux3"):
        logits_path = output / f"{name}_logits.npy"
        if logits_path.exists():
            logits = torch.from_numpy(np.load(logits_path, allow_pickle=False))
        else:
            if name == "base":
                model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                            trust_remote_code=True, device_map="cuda")
                initialize_semantic_token_embeddings(model, tokenizer, mapping, registration)
                model.eval()
            else:
                model = load_model("qwen35_9b", model_path, ROOT / "reference_adapters" / name,
                                   tokenizer, mapping, registration, device_map="cuda")
            values = []
            with torch.inference_mode():
                for index, prompt in enumerate(prompts):
                    inputs = torch.tensor([prompt], device=model.device)
                    outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
                    values.append(outputs.logits[0, -1, label_ids].float().cpu())
                    del outputs
                    if (index + 1) % 32 == 0:
                        print(json.dumps({"event": "gate_forward", "checkpoint": name, "rows": index + 1}), flush=True)
            logits = torch.stack(values)
            np.save(logits_path, logits.numpy())
            del model, values
            gc.collect()
            torch.cuda.empty_cache()
        if logits.shape != (224, 112) or not torch.isfinite(logits).all():
            raise ValueError("invalid gate logits")
        checkpoints[name] = summarize_logits(logits, positives, seen, pseudo)
        write_json_atomic(output / f"{name}_diagnostics.json", checkpoints[name])
    result = {"measurement_passed": True, "gate_rule": GATE_RULE, "branches": branch_decisions(checkpoints),
              "contract_sha256": sha256_file(contract_path),
              "files": {path.name: sha256_file(path) for path in output.iterdir()
                        if path.name.endswith(("_logits.npy", "_diagnostics.json"))},
              "summary": {name: {arm: {key: value for key, value in stats.items() if key != "diagnostics"}
                                  for arm, stats in values.items()} for name, values in checkpoints.items()}}
    write_json_atomic(output / "GATE_RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-package", action="store_true")
    options = parser.parse_args()
    if not options.prepare_package and options.output is None:
        parser.error("--output is required for model measurement")
    print(json.dumps(prepare_package() if options.prepare_package else run(options.output)), flush=True)