"""Record exact generated tokens while reusing the frozen ranking evaluator."""

import argparse
import hashlib
import json
from pathlib import Path


def generation_record(output, prompt_length, tokenizer, mapping):
    token_to_output = {
        tokenizer.convert_tokens_to_ids(token): token for token in mapping["special_tokens"]
    }
    beams = []
    for sequence, score in zip(output.sequences, output.sequences_scores, strict=True):
        suffix = sequence.tolist()[prompt_length:]
        eos = tokenizer.eos_token_id
        content = suffix[:suffix.index(eos)] if eos in suffix else suffix
        raw = tokenizer.decode(content, skip_special_tokens=False)
        exact = len(content) == 1 and content[0] in token_to_output
        visible = raw.strip()
        if exact:
            reason = "exact_semantic_token"
        elif visible in mapping["id_to_category"]:
            reason = ("semantic_token_with_extra_tokens" if any(token in token_to_output for token in content)
                      else "native_spelling_without_semantic_token")
        else:
            reason = "other_invalid_output"
        beams.append({"generated_token_ids": suffix, "content_token_ids": content,
                      "tokens": tokenizer.convert_ids_to_tokens(content), "raw_text_unstripped": raw,
                      "eos_reached": eos in suffix, "strict_valid": exact, "reason": reason,
                      "sequence_score": float(score)})
    return {"prompt_length": prompt_length, "beams": beams}


def attach_trace(model, tokenizer, mapping, path):
    original = model.generate

    def generate(*args, **kwargs):
        output = original(*args, **kwargs)
        inputs = kwargs["input_ids"]
        record = generation_record(output, inputs.shape[-1], tokenizer, mapping)
        with path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output

    model.generate = generate
    return original


def main():
    from huggingface_hub import snapshot_download
    from crossinterest.training import evaluate_ranking as ranking
    import crossinterest.methods.role_projection.evaluate as role_evaluate
    import crossinterest.methods.role_projection.runner as role_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    prepared = role_run.verify()
    complete = json.loads((options.source / "COMPLETE.json").read_text())
    assert complete["passed"] and complete["smoke"]
    assert complete["role_contract_sha256"] == role_run.sha256_file(role_run.CONTRACT)
    model_path = snapshot_download(prepared["model_repo"], revision=prepared["model_revision"])
    options.output.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    original_load = ranking.load_model
    for mode in role_run.MODES:
        source = options.source / mode / "smoke"
        adapter = source / "novel/final_adapter"
        expected = complete["results"][mode]["evaluation"]["weights"]
        assert role_run.sha256_file(adapter / "adapter_model.safetensors") == expected["adapter_sha256"]
        if mode != "baseline":
            assert role_run.sha256_file(adapter / "interest_relation.safetensors") == expected["head_sha256"]
        directory = options.output / mode
        directory.mkdir()
        trace_path = directory / "generation_tokens.jsonl"

        def load(model_key, base_path, adapter_path, tokenizer, mapping, registration, **kwargs):
            model = original_load(model_key, base_path, adapter_path, tokenizer, mapping, registration, **kwargs)
            attach_trace(model, tokenizer, mapping, trace_path)
            return model

        ranking.load_model = load
        try:
            command = role_run.evaluation_command(model_path, adapter, directory, mode, 0, True, prepared)
            role_evaluate.main(command[3:])
        finally:
            ranking.load_model = original_load
        traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
        details = [json.loads(line) for line in (directory / "details.jsonl").read_text().splitlines()]
        assert len(traces) == len(details) == 2
        counts = {}
        for trace, detail in zip(traces, details, strict=True):
            assert len(trace["beams"]) == len(detail["ranking"]) == 20
            for recorded, ranked in zip(trace["beams"], detail["ranking"], strict=True):
                assert recorded["strict_valid"] == ranked["format_valid"]
                assert recorded["raw_text_unstripped"].strip() == ranked["raw_text"]
                counts[recorded["reason"]] = counts.get(recorded["reason"], 0) + 1
        outcomes[mode] = {"rows": len(details), "reasons": counts,
                          "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                          "weights": expected}
        print(json.dumps({"event": "token_diagnostic", "mode": mode, **outcomes[mode]}), flush=True)
    result = {"passed": True, "role_contract_sha256": complete["role_contract_sha256"],
              "training_performed": False, "decoding_changed": False, "results": outcomes}
    (options.output / "DIAGNOSTIC.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()