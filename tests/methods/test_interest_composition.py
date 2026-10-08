"""Isolation, supervision and generation checks for interest composition."""

from crossinterest.paths import source_workspace as _source_workspace

from copy import deepcopy
from pathlib import Path

import pytest


import crossinterest.methods.composition.model as composition


def example():
    return {"historical_titles": [f"product-{index}" for index in range(7)],
            "historical_interests": ["A", "B", "A", "C", "B", "D", "A"],
            "target_interest": "UNSEEN", "sample_id": "row-one"}


@pytest.mark.parametrize("grouping", ["category", "random"])
def test_groups_preserve_history_and_ignore_targets(grouping):
    row = example()
    groups = composition.history_groups(row, grouping)
    assert sorted(position for group in groups for position in group) == list(range(7))
    assert sorted(map(len, groups)) == [1, 1, 2, 3]
    assert [group[-1] for group in groups] == sorted(group[-1] for group in groups)
    changed = deepcopy(row)
    changed.update(target_interest="OTHER", sample_id="another", target_title="secret")
    assert composition.group_texts(row, grouping) == composition.group_texts(changed, grouping)
    texts = composition.group_texts(row, grouping)
    for position, title in enumerate(row["historical_titles"]):
        expected = f"{position + 1}. [{row['historical_interests'][position]}] {title}"
        assert sum(expected in text for text in texts) == 1
    assert all("UNSEEN" not in text for text in texts)


def test_category_groups_are_isolated_and_random_control_is_not_identical():
    row = example()
    groups = composition.history_groups(row)
    assert all(len({row["historical_interests"][position] for position in group}) == 1
               for group in groups)
    assert composition.history_groups(row, "random") != groups
    assert composition.history_groups(row, "random") == composition.history_groups(row, "random")


@pytest.mark.parametrize("row", [
    {"historical_titles": [], "historical_interests": []},
    {"historical_titles": ["product"], "historical_interests": []},
    {"historical_titles": ["product"], "historical_interests": [""]},
])
def test_reject_invalid_history(row):
    with pytest.raises(ValueError):
        composition.history_groups(row)


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    return LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0, pad_token_id=0, eos_token_id=2))


def tensor_example():
    import torch

    return {"input_ids": torch.tensor([[1, 0, 0, 9, 10, 2], [1, 0, 0, 9, 11, 2]]),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, -100, 10, 2],
                                    [-100, -100, -100, -100, 11, 2]]),
            "memory_mask": torch.tensor([[False, True, True, False, False, False]] * 2),
            "group_input_ids": torch.tensor([[12, 13, 14], [15, 16, 0], [17, 18, 19], [20, 21, 0]]),
            "group_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]])}


def test_encoder_isolation_chunking_and_gradient():
    import torch

    model = tiny_model()
    controller = composition.CompositionController(model, encoder_token_budget=3)
    batch = tensor_example()
    model.eval()
    original = controller.encode(batch["group_input_ids"], batch["group_attention_mask"])
    changed = batch["group_input_ids"].clone()
    changed[0, 0] = 30
    altered = controller.encode(changed, batch["group_attention_mask"])
    torch.testing.assert_close(original[1:], altered[1:], rtol=0, atol=0)
    assert not torch.equal(original[0], altered[0])
    controller.encoder_token_budget = 8192
    combined = controller.encode(batch["group_input_ids"], batch["group_attention_mask"])
    torch.testing.assert_close(original, combined, rtol=1e-5, atol=1e-7)
    controller.attach_training()
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.interest_bridge.projection.weight.grad.norm() > 0
    assert model.get_input_embeddings().weight.grad[12].norm() > 0
    assert model.get_input_embeddings().weight.grad[20].norm() > 0
    controller.detach_training()


def test_full_vocabulary_twenty_beam_and_save_reload(tmp_path):
    import torch

    model = tiny_model().eval()
    controller = composition.CompositionController(model)
    batch = tensor_example()
    arguments = {key: value for key, value in batch.items() if key != "labels"}
    arguments["input_ids"] = arguments["input_ids"][:, :4]
    arguments["memory_mask"] = arguments["memory_mask"][:, :4]
    arguments.pop("attention_mask")
    with torch.no_grad():
        embeddings = controller.inputs_embeds(**arguments)
        result = model.generate(input_ids=arguments["input_ids"], inputs_embeds=embeddings,
            attention_mask=torch.ones(2, 4, dtype=torch.long), num_beams=20, num_return_sequences=20,
            do_sample=False, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
    assert result.sequences.shape[0] == 40
    assert result.scores[0].shape == (40, 64)
    assert torch.isfinite(result.sequences_scores).all()
    controller.save(tmp_path)
    restored = tiny_model().eval()
    loaded = composition.CompositionController(restored, directory=tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(loaded.inputs_embeds(**arguments), embeddings, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration"):
        composition.CompositionController(tiny_model(), grouping="random", directory=tmp_path)


def test_bfloat16_bridge_and_loss_rejects_history_bypass():
    import torch

    model = tiny_model().to(torch.bfloat16)
    controller = composition.CompositionController(model)
    model.to(torch.bfloat16)
    output = controller.forward(**tensor_example())
    assert torch.isfinite(output.loss)
    with pytest.raises(ValueError, match="external history"):
        controller.forward(**tensor_example(), past_key_values="forbidden")
    broken = tensor_example()
    broken["labels"][0, 1] = 7
    with pytest.raises(ValueError, match="supervised labels"):
        controller.forward(**broken)


def tiny_qwen_peft(lora_dropout=0):
    import torch
    from unittest.mock import patch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5

    torch.manual_seed(42)
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=4, layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, pad_token_id=0, eos_token_id=2)
    with patch.multiple(modeling_qwen3_5, FusedRMSNormGated=None, causal_conv1d_fn=None,
                        causal_conv1d_update=None, chunk_gated_delta_rule=None,
                        fused_recurrent_gated_delta_rule=None, is_fast_path_available=False):
        base = Qwen3_5ForCausalLM(config)
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, lora_dropout=lora_dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "in_proj_qkv", "lm_head"],
        trainable_token_indices=[10, 11]))
    return model


@pytest.mark.parametrize("bf16", [False, True])
def test_encoder_checkpointing_preserves_gradients_and_reduces_saved_activations(bf16):
    import torch

    results = []
    for checkpointing in (False, True):
        model = tiny_qwen_peft(lora_dropout=0.05).train()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.01)
        controller = composition.CompositionController(model, encoder_token_budget=3)
        controller.attach_training()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        saved_bytes = []

        def pack(tensor):
            saved_bytes.append(tensor.numel() * tensor.element_size())
            return tensor

        torch.manual_seed(91)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                loss = model(**tensor_example()).loss
        loss.backward()
        gradients = {name: parameter.grad.detach().clone()
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert controller.encoder_calls == 4
        assert gradients["interest_bridge.projection.weight"].norm() > 0
        results.append((loss.detach(), gradients, sum(saved_bytes)))
    reference, checkpointed = results
    torch.testing.assert_close(checkpointed[0], reference[0], rtol=0, atol=0)
    for name, gradient in reference[1].items():
        torch.testing.assert_close(checkpointed[1][name], gradient, rtol=0, atol=0, msg=name)
    assert checkpointed[2] < reference[2] * 0.5


def test_qwen_hybrid_peft_gradient_and_beam():
    import torch

    model = tiny_qwen_peft()
    controller = composition.CompositionController(model)
    batch = tensor_example()
    controller.attach_training()
    loss = model(**batch).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert controller.bridge.projection.weight.grad.norm() > 0
    assert any(parameter.grad is not None and parameter.grad.norm() > 0
               for name, parameter in model.named_parameters() if "in_proj_qkv" in name and "lora_B" in name)
    controller.detach_training()
    model.eval()
    with torch.no_grad():
        memories = controller.encode(batch["group_input_ids"], batch["group_attention_mask"])
        altered = batch["group_input_ids"].clone()
        altered[0, 0] = 31
        changed = controller.encode(altered, batch["group_attention_mask"])
        torch.testing.assert_close(memories[1:], changed[1:], rtol=0, atol=0)
        embeddings = controller.inputs_embeds(batch["input_ids"][:, :4], batch["memory_mask"][:, :4],
                                             batch["group_input_ids"], batch["group_attention_mask"])
        generated = model.generate(input_ids=batch["input_ids"][:, :4], inputs_embeds=embeddings,
            attention_mask=batch["attention_mask"][:, :4], num_beams=20, num_return_sequences=20,
            do_sample=False, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
    assert generated.sequences.shape[0] == 40
    assert generated.scores[0].shape == (40, 64)
    assert torch.isfinite(generated.sequences_scores).all()


def test_accelerate_bfloat16_preserves_controller_binding():
    import torch
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState

    model = tiny_qwen_peft()
    controller = composition.CompositionController(model)
    controller.attach_training()
    batch = tensor_example()
    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            expected = controller.forward(**batch)
        prepared = accelerator.prepare_model(model)
        output = prepared(**batch)
        torch.testing.assert_close(output.logits, expected.logits.float())
        torch.testing.assert_close(output.loss, expected.loss.float())
        accelerator.backward(output.loss)
        assert controller.bridge.projection.weight.grad.norm() > 0
        assert any(parameter.grad is not None and parameter.grad.norm() > 0
                   for name, parameter in model.named_parameters()
                   if "in_proj_qkv" in name and "lora_B" in name)
        accelerator.unwrap_model(prepared, keep_fp32_wrapper=False)
        controller.detach_training()
        assert model.forward == controller.original_forward
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def frozen_imports():
    import sys

    live = _source_workspace('eval_interest_id_minionerec')
    package = live / "cir_factorization/package"
    if not package.exists():
        package = live.parent
    sys.path[:0] = [str(package), str(package / "prep")]
    import crossinterest.methods as eval_interest_id_minionerec
    if str(live) not in eval_interest_id_minionerec.__path__:
        eval_interest_id_minionerec.__path__.append(str(live))
    return package


def test_real_tokenizer_supervision_and_target_blind_inference():
    import json
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    package = frozen_imports()
    from crossinterest.methods.cir.runner import MAPPING
    from crossinterest.methods.mechanism.train import auxiliary_messages
    from crossinterest.training import train as baseline
    from crossinterest.training.prompts import render_training_messages
    from crossinterest.training.semantic_tokens import register_semantic_tokens

    reference = json.loads((package / "eval_interest_id_minionerec/cir/data/qwen35_9b_external_balanced_ep3/PREFLIGHT.json").read_text())
    mapping = json.loads(MAPPING.read_text())
    tokenizer = AutoTokenizer.from_pretrained(reference["model_repo"], revision=reference["model_revision"], local_files_only=True)
    register_semantic_tokens(tokenizer, mapping)
    for stage in ("auxiliary", "novel"):
        row = next(pq.ParquetFile(package / reference["files"][stage]["path"]).iter_batches(batch_size=1)).to_pylist()[0]
        for grouping in ("category", "random"):
            feature = composition.encode_example(row, tokenizer, mapping, stage, grouping)
            messages = (auxiliary_messages(row, mapping) if stage == "auxiliary" else
                        render_training_messages(row, "behavior_interest", "interest_id", mapping))
            original = baseline.tokenize_assistant_only(tokenizer, messages)
            assert [value for value in feature["labels"] if value != -100] == [
                value for value in original["labels"] if value != -100]
            if stage == "novel":
                observed = {key: row[key] for key in ("historical_titles", "historical_interests")}
                inference = composition.encode_example(observed, tokenizer, mapping, stage, grouping, training=False)
                assert feature["input_ids"][:len(inference["input_ids"])] == inference["input_ids"]
                assert feature["groups"] == inference["groups"]


def test_queue_timeout_keeps_monitoring_without_hiding_other_alerts():
    from datetime import datetime, timedelta, timezone

    frozen_imports()
    from crossinterest.methods.composition.monitor import classify_observation

    now = datetime.now(timezone.utc)
    queued = {"status": "Queued", "log_file": None, "interruption": None,
              "last_active_at": (now - timedelta(hours=2)).isoformat()}
    original = deepcopy(queued)
    assert classify_observation(queued, queued, now, 1500) == "queued_no_progress"
    for updates, expected in (({"status": "Running"}, "no_progress"),
                              ({"status": "Failed"}, "job_failure"),
                              ({"status": "Completed"}, "terminal"),
                              ({"log_file": "user_logs/std_log.txt"}, "no_progress"),
                              ({"interruption": "new suspension"}, "scheduling_interruption")):
        assert classify_observation({**queued, **updates}, queued, now, 1500) == expected
    assert queued == original


def toy_features():
    batch = tensor_example()
    return [{**{key: batch[key][index].tolist() for key in ("input_ids", "attention_mask", "labels", "memory_mask")},
             "groups": [tokens[mask.bool()].tolist() for tokens, mask in zip(
                 batch["group_input_ids"][2 * index:2 * index + 2],
                 batch["group_attention_mask"][2 * index:2 * index + 2], strict=True)]}
            for index in range(2)]


def test_parameter_update_never_reuses_renamed_blob_tempfile(tmp_path, monkeypatch):
    import json

    frozen_imports()
    from crossinterest.methods.composition.train import write_parameters_atomic

    destination = tmp_path / "trainable_parameters.json"
    destination.write_text('{"count": 1}\n')
    original_replace = Path.replace
    used = {destination.with_suffix(".json.tmp")}

    def replace(source, target):
        assert source not in used
        used.add(source)
        assert target == destination
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    for count in (2, 3):
        write_parameters_atomic(destination, {"count": count, "names": ["interest_bridge.projection.weight"]})
        assert json.loads(destination.read_text())["count"] == count
        assert list(tmp_path.iterdir()) == [destination]


def toy_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"token{index}": index for index in range(64)}, unk_token="token3")),
        pad_token="token0", bos_token="token1", eos_token="token2", unk_token="token3")


@pytest.mark.parametrize("stage", ["novel", "auxiliary"])
@pytest.mark.parametrize("bf16", [False, True])
def test_trainer_complete_checkpoint_resume(tmp_path, stage, bf16):
    import json
    import torch
    from transformers import TrainerCallback, TrainingArguments

    frozen_imports()
    from crossinterest.methods.composition.train import CompositionTrainer

    features = toy_features() * 4
    if stage == "auxiliary":
        features = [{**feature, "positive_mask": [True, index % 2 == 0]}
                    for index, feature in enumerate(features)]

    def trainer(directory):
        return CompositionTrainer(model=tiny_qwen_peft(), processing_class=toy_tokenizer(),
            train_dataset=features, stage=stage, grouping="category", sparse_embedding_save=True,
            label_token_ids=[10, 11], args=TrainingArguments(output_dir=str(directory),
                use_cpu=True, bf16=bf16, max_steps=4, per_device_train_batch_size=1,
                gradient_accumulation_steps=2, learning_rate=1e-3, save_steps=2,
                logging_steps=1, report_to="none", disable_tqdm=True, remove_unused_columns=False,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}))

    class StopAtTwo(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True

    uninterrupted = trainer(tmp_path / "full")
    uninterrupted.train()
    interrupted = trainer(tmp_path / "resumed")
    interrupted.add_callback(StopAtTwo())
    interrupted.train()
    checkpoint = tmp_path / "resumed/checkpoint-2"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                 "adapter_model.safetensors", "interest_bridge.safetensors", "interest_bridge.json"):
        assert (checkpoint / name).is_file()
    assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 2
    continued = trainer(tmp_path / "resumed")
    continued.train(resume_from_checkpoint=str(checkpoint))
    assert continued.state.global_step == uninterrupted.state.global_step == 4
    expected = dict(uninterrupted.model.named_parameters())
    for name, value in continued.model.named_parameters():
        if value.requires_grad:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=1e-7, msg=name)
    assert continued.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    assert {int(value["step"]) for value in continued.optimizer.state.values()} == {4}
    assert any(parameter is continued.controller.bridge.projection.weight
               for group in continued.optimizer.param_groups for parameter in group["params"])


def test_original_ranker_composition_prefill_once(monkeypatch):
    from types import SimpleNamespace

    frozen_imports()
    from crossinterest.training import evaluate_ranking
    import crossinterest.methods.composition.model as module

    model = tiny_qwen_peft().eval()
    controller = module.CompositionController(model)
    evaluator = SimpleNamespace(direct_beam_rank=evaluate_ranking.direct_beam_rank)
    feature = toy_features()[0]
    feature = {key: value[:4] if key != "groups" else value for key, value in feature.items()}
    monkeypatch.setattr(module, "encode_example", lambda *args, **kwargs: feature)
    checks = []
    module.bind_ranking_composition(evaluator, controller, {}, checks.append)
    tokenizer = toy_tokenizer()
    prompt = evaluator.render_prompt_ids(tokenizer, {"sample_id": "one"}, "qwen35_9b", "behavior_interest", "interest_id")
    original_generate = model.generate
    result = evaluator.direct_beam_rank(model, tokenizer, prompt, {"token10": "A", "token11": "B"},
                                       20, 3, {(10,): "token10", (11,): "token11"})
    assert len(result) == 20
    assert checks[0]["generate_calls"] == checks[0]["encoder_calls"] == 1
    assert model.generate == original_generate
    with pytest.raises(ValueError, match="not aligned"):
        evaluator.direct_beam_rank(model, tokenizer, prompt, {}, 20, 3, {})


@pytest.mark.parametrize("micro_batch,accumulation,checkpointing", [(1, 4, False), (1, 4, True), (4, 1, True)])
def test_frozen_cache_and_full_budget_commands(tmp_path, micro_batch, accumulation, checkpointing):
    import json
    from transformers import AutoTokenizer

    frozen_imports()
    from crossinterest.training.semantic_tokens import register_semantic_tokens
    from crossinterest.training import train as baseline
    import crossinterest.methods.composition.runner as runner
    from crossinterest.methods.composition.train import CompositionDataset

    data = runner.verify_data()
    reference = json.loads(runner.REFERENCE.read_text())
    mapping = json.loads(runner.MAPPING.read_text())
    tokenizer = AutoTokenizer.from_pretrained(reference["model_repo"], revision=reference["model_revision"], local_files_only=True)
    register_semantic_tokens(tokenizer, mapping)
    prepared = {"data": data, "training_layout": {"world_size": 8, "micro_batch": micro_batch,
                "accumulation": accumulation, "gradient_checkpointing": checkpointing}}
    for arm in runner.ARMS:
        for stage in (("novel",) if arm == "N" else ("auxiliary", "novel")):
            info = runner.stage_info(prepared, arm, stage, False)
            assert info["steps"] == (7866 if stage == "auxiliary" else 3500)
            command = runner.training_command("base", tmp_path / stage, arm, stage, False, prepared)
            position = command.index("--model")
            options, arguments = runner.argparse.ArgumentParser(add_help=False), command[position:]
            options.add_argument("--arm")
            options.add_argument("--stage")
            _, arguments = options.parse_known_args(arguments)
            parsed = baseline.parse_args(arguments)
            assert parsed.epochs == info["epochs"] and parsed.expected_train_rows == info["rows"]
            assert parsed.per_device_batch_size * parsed.gradient_accumulation_steps * 8 == 32
            assert parsed.gradient_checkpointing == checkpointing
            smoke = runner.stage_info(prepared, arm, stage, True)
            args = (smoke["source"], tokenizer, "behavior_interest", "interest_id", mapping, 8192,
                    None, None, None, 224)
            cached = CompositionDataset(*args, stage=stage, grouping=runner.ARMS[arm], encoded_file=smoke["cache"])
            original = CompositionDataset(*args, stage=stage, grouping=runner.ARMS[arm])
            for index in (0, 111, 223):
                assert cached[index] == original[index]


def test_versioned_audit_preserves_science_and_effective_batch():
    frozen_imports()
    from crossinterest.methods.composition.monitor import verify_compatible_protocol

    previous = {"budget": {"novel_rows": 112000}, "training_layout": {
        "world_size": 8, "micro_batch": 1, "accumulation": 4, "gradient_checkpointing": True}}
    current = deepcopy(previous)
    current["training_layout"].update(micro_batch=4, accumulation=1)
    verify_compatible_protocol(current, previous)
    wrong = deepcopy(current)
    wrong["budget"]["novel_rows"] = 1000
    with pytest.raises(ValueError, match="scientific protocol"):
        verify_compatible_protocol(wrong, previous)
    wrong = deepcopy(current)
    wrong["training_layout"]["accumulation"] = 2
    with pytest.raises(ValueError, match="effective batch"):
        verify_compatible_protocol(wrong, previous)


def test_complete_resume_selector_rejects_partial_state(tmp_path, monkeypatch):
    import json
    import random
    import tempfile
    from types import SimpleNamespace
    import numpy as np
    import torch

    frozen_imports()
    import crossinterest.methods.composition.runner as runner
    from crossinterest.methods.composition.model import CompositionController

    model = tiny_qwen_peft()
    controller = CompositionController(model)
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint)
    controller.save(checkpoint)
    torch.save({"state": {0: {"step": torch.tensor(2)}}}, checkpoint / "optimizer.pt")
    torch.save({"last_epoch": 2}, checkpoint / "scheduler.pt")
    torch.save({}, checkpoint / "training_args.bin")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    for rank in range(2):
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state(), "cuda": []}, checkpoint / f"rng_state_{rank}.pth")
    (output / "checkpoint-4").mkdir()
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(runner, "mkdtemp", lambda **kwargs: tempfile.mkdtemp(dir=tmp_path, **kwargs))
    baseline = SimpleNamespace()
    runner.install_complete_resume(baseline, output, "category")
    restored = baseline.latest_checkpoint(output)
    assert restored.name == "checkpoint-2" and restored != checkpoint
    assert json.loads((output / "IO_RESUME_AUDIT.json").read_text())["step"] == 2
    (checkpoint / "interest_bridge.json").write_text(json.dumps({"architecture": runner.METHOD, "grouping": "random"}))
    with pytest.raises(ValueError, match="no complete composition checkpoint"):
        baseline.latest_checkpoint(output)
    controller.save(checkpoint)
    (checkpoint / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="no complete composition checkpoint"):
        baseline.latest_checkpoint(output)
    empty = tmp_path / "interrupted"
    empty.mkdir()
    with pytest.raises(ValueError, match="refusing weight-only restart"):
        baseline.latest_checkpoint(empty)


def test_full_evaluation_audit_and_raw_corruption(tmp_path, monkeypatch):
    import json
    import pyarrow.parquet as pq

    frozen_imports()
    from crossinterest.evaluation.common import write_json_atomic
    from crossinterest.training import evaluate_ranking as ranking
    import crossinterest.methods.composition.runner as runner
    import crossinterest.methods.composition.monitor as monitor

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PREFLIGHT", tmp_path / "PREFLIGHT.json")
    write_json_atomic(runner.PREFLIGHT, {"fixture": True})
    directory = tmp_path / "formal"
    adapter = directory / "novel/final_adapter"
    adapter.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "interest_bridge.safetensors"):
        (adapter / name).write_bytes(b"identity-only-fixture")
    mapping = json.loads(runner.MAPPING.read_text())["id_to_category"]
    category_to_id = {category: token for token, category in mapping.items()}
    categories = sorted(category_to_id)
    table = pq.read_table(runner.DEV)
    prepared = {"shards": {}}
    for shard in range(8):
        source = tmp_path / f"shard-{shard}.parquet"
        subset = table.take(list(range(shard, len(table), 8)))
        pq.write_table(subset, source)
        rows = subset.to_pylist()
        prepared["shards"][str(shard)] = runner.file_record(source, len(rows))
        output = directory / "ranking" / f"shard-{shard}"
        output.mkdir(parents=True)
        fingerprint = f"fixture-{shard}"
        contract = {"fingerprint": fingerprint, "composition_method": runner.METHOD,
            "composition_arm": "C", "composition_grouping": "category",
            "composition_preflight_sha256": runner.sha256_file(runner.PREFLIGHT),
            "composition_adapter_sha256": runner.sha256_file(adapter / "adapter_model.safetensors"),
            "composition_bridge_sha256": runner.sha256_file(adapter / "interest_bridge.safetensors"),
            "composition_shard": shard, "dataset_sha256": runner.sha256_file(source),
            "label_map_sha256": runner.MAPPING_SHA, "label_file_sha256": runner.TRAIN_SHA,
            "sample_limit": None, "num_beams": 20, "num_return_sequences": 20,
            "do_sample": False, "candidates_in_prompt": False, "decoder_label_constraint": False,
            "decoding": "candidate_free_direct_beam_search", "max_new_tokens": 32,
            "eval_batch_size": 1, "label_mode": "interest_id", "view": "behavior_interest"}
        write_json_atomic(output / "run_contract.json", contract)
        details, checks = [], []
        for row in rows:
            target = row["target_interest"]
            predictions = [target, *[category for category in categories if category != target][:19]]
            entries = [{"rank": position, "prediction": category, "predicted_output": category_to_id[category],
                        "raw_text": category_to_id[category], "format_valid": True, "sequence_score": -float(position)}
                       for position, category in enumerate(predictions, 1)]
            details.append({"sample_id": row["sample_id"], "source_sample_id": row["source_sample_id"],
                "target_interest": target, "ranking": entries, "run_fingerprint": fingerprint,
                **ranking.score_ranking(entries, target)})
            checks.append({"sample_id": row["sample_id"], "generate_calls": 1, "encoder_calls": 1,
                "num_beams": 20, "groups": len(runner.history_groups(row)), "grouping": "category",
                "raw_history_decoder_bypass": False, "composition_preflight_sha256": runner.sha256_file(runner.PREFLIGHT)})
        for name, values in (("details.jsonl", details), ("composition_checks.jsonl", checks)):
            (output / name).write_text("".join(json.dumps(value) + "\n" for value in values))
        write_json_atomic(output / "metrics.json", ranking.metrics_from_rankings(details, rows, 20))
        ranking.write_predictions(output / "predictions.jsonl", details)
    result = runner.audit_evaluation(directory, "C", False, prepared)
    assert result["health_passed"]
    assert set(monitor.independent_metrics(directory, prepared).values()) == {1.0}
    path = directory / "ranking/shard-0/details.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["ranking"][0]["raw_text"] = "invalid"
    path.write_text("".join(json.dumps(value) + "\n" for value in records))
    with pytest.raises(ValueError, match="Strict format"):
        monitor.independent_metrics(directory, prepared)
    (adapter / "interest_bridge.safetensors").write_bytes(b"different-bridge")
    with pytest.raises(ValueError):
        runner.audit_evaluation(directory, "C", False, prepared)


def ddp_smoke(output):
    import torch
    from transformers import TrainingArguments

    frozen_imports()
    from crossinterest.methods.composition.train import CompositionTrainer

    for stage in ("auxiliary", "novel"):
        features = toy_features() * 4
        if stage == "auxiliary":
            features = [{**feature, "positive_mask": [True, False]} for feature in features]
        trainer = CompositionTrainer(model=tiny_qwen_peft(), processing_class=toy_tokenizer(),
            train_dataset=features, stage=stage, grouping="category", sparse_embedding_save=True,
            label_token_ids=[10, 11], args=TrainingArguments(output_dir=str(output / stage),
                use_cpu=True, ddp_backend="gloo", ddp_find_unused_parameters=False,
                max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                learning_rate=1e-3, save_steps=2, logging_steps=1, report_to="none",
                disable_tqdm=True, remove_unused_columns=False, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}))
        trainer.train()
        assert trainer.state.global_step == 2 and torch.distributed.get_world_size() == 2
        weight = trainer.controller.bridge.projection.weight.detach().contiguous()
        gathered = [torch.empty_like(weight) for _ in range(2)]
        torch.distributed.all_gather(gathered, weight)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        if torch.distributed.get_rank() == 0:
            print(f"PASS: {stage} two-rank DDP with shared decoder, checkpointing and synchronized bridge", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import sys
    ddp_smoke(Path(sys.argv[1]))