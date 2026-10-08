"""Exact contractions, stage isolation and causal-model integration checks."""

import itertools
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from crossinterest.methods.role_projection.audit import independent_metrics, improvement_gate
from crossinterest.methods.role_projection.diagnose import attach_trace
from crossinterest.methods.role_projection.model import RoleProjectionController, RoleProjectionHead
from tests.methods.test_interest_composition import tiny_model, tiny_qwen_peft


def test_generation_token_trace_preserves_output(tmp_path):
    class TraceTokenizer:
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return 40

        def convert_ids_to_tokens(self, tokens):
            return [str(token) for token in tokens]

        def decode(self, tokens, skip_special_tokens=False):
            return {(40,): "<interest_001>", (4, 5): "<interest_001>",
                    (3, 40): " <interest_001>", (9,): "invalid"}[tuple(tokens)]

    generated = SimpleNamespace(
        sequences=torch.tensor([[1, 40, 2, 0], [1, 4, 5, 2], [1, 3, 40, 2], [1, 9, 2, 0]]),
        sequences_scores=torch.tensor([-0.1, -0.2, -0.3, -0.4]))

    class TraceModel:
        def generate(self, **kwargs):
            return generated

    model = TraceModel()
    trace = tmp_path / "generation_tokens.jsonl"
    attach_trace(model, TraceTokenizer(), {
        "special_tokens": ["<interest_001>"],
        "id_to_category": {"<interest_001>": "category"}}, trace)
    assert model.generate(input_ids=torch.tensor([[1]])) is generated
    record = json.loads(trace.read_text())
    assert [beam["reason"] for beam in record["beams"]] == [
        "exact_semantic_token", "native_spelling_without_semantic_token",
        "semantic_token_with_extra_tokens", "other_invalid_output"]
    assert [beam["strict_valid"] for beam in record["beams"]] == [True, False, False, False]
    assert record["beams"][0]["content_token_ids"] == [40]
    assert record["beams"][2]["raw_text_unstripped"] == " <interest_001>"


@pytest.mark.parametrize("world,accumulation,checkpointing", [(4, 8, True), (8, 4, False)])
@pytest.mark.parametrize("stage", ["auxiliary", "novel"])
def test_training_command_preserves_frozen_layout(tmp_path, world, accumulation, checkpointing, stage):
    from crossinterest.methods.role_projection.runner import training_command

    epochs = 3 if stage == "auxiliary" else 1
    prepared = {"training_layout": {"world_size": world, "micro_batch": 1,
                                   "accumulation": accumulation, "gradient_checkpointing": checkpointing},
                "files": {stage: {"path": "train.parquet", "rows": 83888 if stage == "auxiliary" else 112000,
                                  "epochs": epochs}}}
    command = training_command("model", tmp_path / stage, "baseline", stage, False, prepared)
    assert f"--nproc_per_node={world}" in command
    assert command[command.index("--gradient-accumulation-steps") + 1] == str(accumulation)
    assert ("--gradient-checkpointing" in command) == checkpointing
    assert ("--no-gradient-checkpointing" in command) != checkpointing
    assert command[command.index("--epochs") + 1] == str(epochs)
    assert ("--init-adapter" in command) == (stage == "novel")
    prepared["training_layout"]["accumulation"] = 1
    with pytest.raises(ValueError, match="effective batch"):
        training_command("model", tmp_path / stage, "baseline", stage, False, prepared)


def test_independent_formal_metrics_and_effect_gate():
    categories = {f"id{index}": f"class{index}" for index in range(21)}
    expected = [{"sample_id": str(index), "source_sample_id": str(index), "target_interest": target}
                for index, target in enumerate(("class0", "class0", "class1"))]
    details = []
    for row, position in zip(expected, (1, None, 11), strict=True):
        predictions = [category for category in categories.values() if category != row["target_interest"]][:20]
        if position is not None:
            predictions[position - 1] = row["target_interest"]
        beams = [{"rank": index + 1, "prediction": category,
                  "predicted_output": "id" + category.removeprefix("class"),
                  "raw_text": "id" + category.removeprefix("class"),
                  "format_valid": True, "sequence_score": -float(index)}
                 for index, category in enumerate(predictions)]
        details.append({**row, "ranking": beams})
    mapping = {"id_to_category": categories}
    metrics = independent_metrics(details, expected, mapping)
    assert metrics["macro_recall_at_10"] == pytest.approx(0.25)
    assert metrics["macro_recall_at_20"] == pytest.approx(0.75)
    assert metrics["macro_ndcg_at_20"] == pytest.approx((0.5 + 1 / math.log2(12)) / 2)
    assert metrics["valid_beam_rate"] == metrics["strict_beam_format_rate"] == 1
    assert metrics["duplicate_valid_beam_rate"] == 0
    with pytest.raises(ValueError, match="coverage"):
        independent_metrics(details[:-1], expected, mapping)
    details[0]["ranking"][-1] = {**details[0]["ranking"][0], "rank": 20}
    details[1]["ranking"][-1].update(prediction=None, predicted_output="invalid",
                                   raw_text="invalid", format_valid=False)
    unhealthy = independent_metrics(details, expected, mapping)
    assert unhealthy["duplicate_valid_beam_rate"] == pytest.approx(1 / 60)
    assert unhealthy["valid_beam_rate"] == unhealthy["strict_beam_format_rate"] == pytest.approx(59 / 60)
    comparison = {
        "baseline": {"macro_ndcg_at_20": 0.160, "macro_recall_at_20": 0.39},
        "unconstrained": {"macro_ndcg_at_20": 0.162, "macro_recall_at_20": 0.39},
        "projected": {"macro_ndcg_at_20": 0.164, "macro_recall_at_20": 0.40}}
    criteria = {"projected_ndcg20_minus_baseline_minimum": 0.005}
    assert not improvement_gate(comparison, criteria)["passed"]
    comparison["projected"]["macro_ndcg_at_20"] = 0.166
    assert improvement_gate(comparison, criteria)["passed"]
    comparison["projected"]["macro_recall_at_20"] = 0.38
    assert not improvement_gate(comparison, criteria)["passed"]


def make_head(mode="projected", stage="auxiliary"):
    with torch.random.fork_rng():
        torch.manual_seed(20260825)
        return RoleProjectionHead(12, torch.randn(7, 12), 4, mode, stage)


@pytest.mark.parametrize("mode", ["projected", "unconstrained"])
@pytest.mark.parametrize("stage", ["auxiliary", "novel"])
def test_exact_pair_contraction(mode, stage):
    head = make_head(mode, stage)
    with torch.no_grad():
        head.compatible.context.weight.normal_()
        head.directional.context.weight.normal_()
    hidden = torch.randn(5, 12)
    membership = torch.arange(7)[None, :] < torch.tensor([0, 1, 2, 5, 7])[:, None]
    expected = torch.zeros(5, 7)
    branches = [(head.compatible, "compatible")]
    if stage == "novel":
        branches.append((head.directional, "directional"))
    for branch, projection in branches:
        prototypes = head.target_prototypes / head.target_prototypes.square().mean().sqrt()
        source = branch.source(prototypes)
        target = branch.target(prototypes)
        source = source - source.mean(dim=0)
        target = target - target.mean(dim=0)
        context = branch.context(F.normalize(hidden, dim=-1))
        for batch_index, selected in enumerate(membership):
            pairs = list(itertools.combinations(selected.nonzero().flatten().tolist(), 2))
            for first, second in pairs:
                raw = (source[first] * source[second] * target * context[batch_index]).sum(-1)
                exchanged = ((source[second] * target[first] + source[first] * target[second])
                             * source * context[batch_index]).sum(-1)
                if mode == "unconstrained":
                    scores = raw
                elif projection == "compatible":
                    scores = (raw + exchanged) / 3
                else:
                    scores = (2 * raw - exchanged) / 3
                expected[batch_index] += scores / len(pairs)
    actual = head(hidden, membership)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(actual.mean(-1), torch.zeros(5), atol=2e-7, rtol=0)
    torch.testing.assert_close(actual[:2], torch.zeros(2, 7), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["projected", "unconstrained"])
def test_zero_start_gradients_and_stage_handoff(mode):
    head = make_head(mode)
    hidden = torch.randn(2, 12, requires_grad=True)
    visible = torch.tensor([[True, True, False, True, False, False, False]] * 2)
    initial = head(hidden, visible)
    torch.testing.assert_close(initial, torch.zeros_like(initial), atol=0, rtol=0)
    initial[:, 4].sum().backward()
    assert head.compatible.context.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in head.directional.parameters())
    assert all(not parameter.requires_grad for parameter in head.directional.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    head(hidden, visible)[:, 4].sum().backward()
    assert head.compatible.source.weight.grad.norm() > 0
    assert head.compatible.target.weight.grad.norm() > 0
    assert hidden.grad.norm() > 0
    auxiliary = head(hidden, visible).detach()
    head.set_stage("novel")
    torch.testing.assert_close(head(hidden, visible), auxiliary, atol=0, rtol=0)
    head(hidden, visible)[:, 4].sum().backward()
    assert head.directional.context.weight.grad.norm() > 0


@pytest.mark.parametrize("mode", ["projected", "unconstrained"])
def test_small_scale_taxonomy_updates_both_branches(mode):
    head = make_head(mode, "novel")
    head.target_prototypes.mul_(1e-3)
    initial = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    hidden = torch.linspace(-1, 1, 24).reshape(2, 12)
    visible = torch.tensor([[True, True, False, True, False, False, False]] * 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0)
    for _ in range(7):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(head(hidden, visible), torch.tensor([4, 5])).backward()
        optimizer.step()
    for name, parameter in head.named_parameters():
        assert not torch.equal(parameter, initial[name]), name


def test_matched_parameters_initialization_and_bfloat16():
    projected, ordinary = make_head(), make_head("unconstrained")
    assert sum(parameter.numel() for parameter in projected.parameters()) == 6 * 12 * 4
    for name, value in projected.state_dict().items():
        torch.testing.assert_close(value, ordinary.state_dict()[name], atol=0, rtol=0)
    projected.bfloat16()
    with torch.no_grad():
        projected.compatible.context.weight.fill_(0.1)
    output = projected(torch.randn(2, 12).bfloat16(), torch.ones(2, 7, dtype=torch.bool))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in projected.compatible.parameters())


class TinyTokenizer:
    def convert_tokens_to_ids(self, token):
        return int(token.removeprefix("id-")) + 40

    def encode(self, text, add_special_tokens=False):
        return [int(text.removeprefix("category-")) + 12, 25]

    def save_pretrained(self, directory):
        return ()


def mapping():
    tokens = [f"id-{index}" for index in range(7)]
    return {"special_tokens": tokens, "mapping_fingerprint": "test-mapping",
            "id_to_category": {token: f"category-{index}" for index, token in enumerate(tokens)}}


def frame(position=3):
    return {"prediction_position": position, "categories": ["category-0", "category-2", "category-0"]}


@pytest.mark.parametrize("mode", ["projected", "unconstrained"])
def test_first_id_injection_handoff_save_and_full_beam(tmp_path, mode):
    model = tiny_model().eval()
    controller = RoleProjectionController(model, TinyTokenizer(), mapping(), dimension=4, mode=mode)
    inputs = torch.tensor([[1, 4, 5, 6, 40, 2]])
    controller.set_frames([frame()])
    with torch.no_grad():
        initial = model(inputs).logits
        controller.head.compatible.context.weight.normal_()
        changed = model(inputs).logits
    difference = changed - initial
    assert difference[:, 3, controller.label_ids].abs().sum() > 0
    difference[:, 3, controller.label_ids] = 0
    torch.testing.assert_close(difference, torch.zeros_like(difference), atol=0, rtol=0)
    controller.save(tmp_path)
    restored = tiny_model().eval()
    loaded = RoleProjectionController(restored, TinyTokenizer(), mapping(), tmp_path,
        dimension=4, mode=mode, stage="novel", allow_auxiliary_handoff=True)
    loaded.set_frames([frame()])
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).logits, changed, atol=0, rtol=0)
        before = loaded.applied_calls
        generated = restored.generate(inputs[:, :4], attention_mask=torch.ones(1, 4, dtype=torch.long),
            num_beams=20, num_return_sequences=20, do_sample=False, max_new_tokens=3,
            return_dict_in_generate=True, output_scores=True)
    assert generated.sequences.shape[0] == 20
    assert generated.scores[0].shape == (20, 64)
    assert loaded.applied_calls == before + 1
    loaded.save(tmp_path / "novel")
    loaded.load(tmp_path / "novel")
    with pytest.raises(ValueError, match="metadata"):
        loaded.load(tmp_path)
    with pytest.raises(ValueError, match="metadata"):
        RoleProjectionController(tiny_model(), TinyTokenizer(), mapping(), tmp_path, dimension=4,
                                 mode="unconstrained" if mode == "projected" else "projected")
    controller.close()
    loaded.close()


def test_visible_set_only_and_checkpointed_backward():
    model = tiny_model().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    controller = RoleProjectionController(model, TinyTokenizer(), mapping(), dimension=4, stage="novel")
    controller.set_frames([frame()])
    inputs = torch.tensor([[1, 4, 5, 6, 40, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, 40, 2]])
    result = model(inputs, labels=labels, use_cache=False)
    result.loss.backward()
    assert controller.head.compatible.context.weight.grad.norm() > 0
    assert controller.head.directional.context.weight.grad.norm() > 0
    controller.set_frames([{**frame(), "categories": ["category-2", "category-0"]}])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs, use_cache=False).logits, result.logits, atol=0, rtol=0)
    controller.set_frames([{**frame(), "categories": ["category-0", "category-8"]}])
    assert torch.isfinite(model(inputs, use_cache=False).logits).all()
    assert "category-8" in controller.prototype_cache
    controller.close()


def test_outside_output_taxonomy_exact_contraction():
    head = make_head(stage="novel")
    hidden = torch.randn(1, 12)
    history = torch.randn(2, 12)
    expected = torch.zeros(1, 7)
    for branch, projection in ((head.compatible, "compatible"), (head.directional, "directional")):
        with torch.no_grad():
            branch.context.weight.normal_()
        taxonomy_scale = head.target_prototypes.square().mean().sqrt()
        source = branch.source(head.target_prototypes / taxonomy_scale)
        target = branch.target(head.target_prototypes / taxonomy_scale)
        history_source = branch.source(history / taxonomy_scale) - source.mean(0)
        history_target = branch.target(history / taxonomy_scale) - target.mean(0)
        source, target = source - source.mean(0), target - target.mean(0)
        context = branch.context(F.normalize(hidden, dim=-1))
        raw = (history_source[0] * history_source[1] * target * context).sum(-1)
        exchanged = ((history_source[1] * history_target[0] + history_source[0] * history_target[1])
                     * source * context).sum(-1)
        expected += (raw + exchanged if projection == "compatible" else 2 * raw - exchanged) / 3
    actual = head(hidden, torch.ones(1, 2, dtype=torch.bool), history)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.mean(-1), torch.zeros(1), atol=2e-7, rtol=0)


def test_controller_preserves_backbone_rng():
    model = tiny_model()
    before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    controller = RoleProjectionController(model, TinyTokenizer(), mapping(), dimension=4)
    assert torch.equal(before, torch.random.get_rng_state())
    for original, after in zip(cuda_before, torch.cuda.get_rng_state_all() if cuda_before else []):
        assert torch.equal(original, after)
    controller.close()


@pytest.mark.parametrize("mode", ["projected", "unconstrained"])
@pytest.mark.parametrize("stage", ["auxiliary", "novel"])
def test_qwen_checkpointing_preserves_role_gradients(mode, stage):
    outputs = []
    for checkpointing in (False, True):
        model = tiny_qwen_peft().train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        controller = RoleProjectionController(model, TinyTokenizer(), mapping(), dimension=4,
                                              mode=mode, stage=stage)
        with torch.no_grad():
            controller.head.compatible.context.weight.fill_(0.02)
            controller.head.directional.context.weight.fill_(0.03)
        controller.set_frames([frame()])
        inputs = torch.tensor([[1, 4, 5, 6, 40, 2]])
        labels = torch.tensor([[-100, -100, -100, -100, 40, 2]])
        torch.manual_seed(20260825)
        result = model(inputs, labels=labels, use_cache=False)
        loss = result.loss if stage == "novel" else -result.logits[:, 3, controller.label_ids].log_softmax(-1)[:, [0, 2]].mean()
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None}
        assert gradients["interest_relation.compatible.source.weight"].norm() > 0
        outputs.append((loss.detach(), gradients))
        controller.close()
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0, rtol=0)
    assert outputs[0][1].keys() == outputs[1][1].keys()
    for name, gradient in outputs[0][1].items():
        torch.testing.assert_close(gradient, outputs[1][1][name], atol=2e-7, rtol=2e-5, msg=name)


def test_qwen_hybrid_lora_gradient_and_beam():
    model = tiny_qwen_peft().train()
    controller = RoleProjectionController(model, TinyTokenizer(), mapping(), dimension=4, stage="novel")
    inputs = torch.tensor([[1, 4, 5, 6, 40, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, 40, 2]])
    controller.set_frames([frame()])
    loss = model(inputs, labels=labels, use_cache=False).loss
    loss.backward()
    assert controller.head.compatible.context.weight.grad.norm() > 0
    assert controller.head.directional.context.weight.grad.norm() > 0
    assert any(parameter.grad is not None and parameter.grad.norm() > 0
               for name, parameter in model.named_parameters() if "in_proj_qkv" in name and "lora_B" in name)
    model.eval()
    with torch.no_grad():
        generated = model.generate(inputs[:, :4], attention_mask=torch.ones(1, 4, dtype=torch.long),
            num_beams=20, num_return_sequences=20, do_sample=False, max_new_tokens=3,
            return_dict_in_generate=True, output_scores=True)
    assert generated.sequences.shape[0] == 20
    assert generated.scores[0].shape == (20, 64)
    controller.close()


@pytest.mark.parametrize("mode", ["baseline", "projected", "unconstrained"])
@pytest.mark.parametrize("stage", ["auxiliary", "novel"])
def test_trainer_preserves_objective_and_updates_head(tmp_path, mode, stage):
    import json

    from transformers import TrainingArguments
    from crossinterest.methods.mechanism.train import MechanismTrainer
    from crossinterest.methods.role_projection.train import RoleTrainer

    class Dataset:
        mapping = globals()["mapping"]()

        def __len__(self):
            return 1

    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = 0
    arguments = dict(args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True),
        train_dataset=Dataset(), processing_class=tokenizer, stage=stage,
        sparse_embedding_save=False, label_token_ids=list(range(40, 47)))
    original_parameters = tmp_path / "trainable_parameters.json"
    original_parameters.write_text(json.dumps({"owner": "baseline", "count": 0, "names": []}))
    original_bytes = original_parameters.read_bytes()
    original = MechanismTrainer(model=tiny_model(), relation=False, init_adapter=None, **arguments)
    trainer = RoleTrainer(model=tiny_model(), role_mode=mode, **arguments)
    assert original_parameters.read_bytes() == original_bytes
    role_parameters = json.loads((tmp_path / "role_trainable_parameters.json").read_text())
    assert role_parameters["count"] == sum(parameter.numel() for parameter in trainer.model.parameters()
                                          if parameter.requires_grad)
    inputs = {"input_ids": torch.tensor([[1, 4, 5, 6, 40, 2]]),
              "labels": torch.tensor([[-100, -100, -100, -100, 40, 2]])}
    if stage == "auxiliary":
        inputs["positive_mask"] = torch.tensor([[True, False, True, False, False, False, False]])
    expected = original.compute_loss(original.model, dict(inputs))
    if mode != "baseline":
        inputs["history_frames"] = [frame()]
    actual = trainer.compute_loss(trainer.model, dict(inputs))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    if mode != "baseline":
        assert trainer.controller.head.compatible.context.weight.grad.norm() > 0
        assert (trainer.controller.head.directional.context.weight.grad is not None) == (stage == "novel")
        trainer.controller.close()


@pytest.mark.parametrize("stage", ["auxiliary", "novel"])
def test_g0_baseline_accumulated_update_equivalence(tmp_path, stage):
    from transformers import TrainingArguments
    from crossinterest.methods.mechanism.train import MechanismTrainer
    from crossinterest.methods.role_projection.train import RoleTrainer

    class Dataset:
        mapping = globals()["mapping"]()

        def __len__(self):
            return 8

        def __getitem__(self, index):
            record = {"input_ids": [1, 4 + index % 2, 5, 6, 40, 2], "attention_mask": [1] * 6,
                      "labels": [-100, -100, -100, -100, 40, 2]}
            if stage == "auxiliary":
                record["positive_mask"] = [True, False, True, False, False, False, False]
            return record

    states = []
    for trainer_class in (MechanismTrainer, RoleTrainer):
        tokenizer = TinyTokenizer()
        tokenizer.pad_token_id = 0
        extra = {"relation": False, "init_adapter": None} if trainer_class is MechanismTrainer else {"role_mode": "baseline"}
        trainer = trainer_class(model=tiny_qwen_peft(), stage=stage, **extra,
            args=TrainingArguments(output_dir=str(tmp_path / trainer_class.__name__), report_to="none", use_cpu=True,
                max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                save_strategy="no", learning_rate=1e-4, seed=20260825,
                remove_unused_columns=False, dataloader_pin_memory=False),
            train_dataset=Dataset(), processing_class=tokenizer, sparse_embedding_save=True,
            label_token_ids=list(range(40, 47)))
        trainer.train()
        assert trainer.state.global_step == 1
        states.append({name: value.detach().clone() for name, value in trainer.model.named_parameters()
                       if value.requires_grad})
    assert states[0].keys() == states[1].keys()
    for name, value in states[0].items():
        torch.testing.assert_close(value, states[1][name], atol=0, rtol=0, msg=name)


def test_optimizer_checkpoint_resume_restores_role_head(tmp_path):
    from transformers import TrainingArguments
    from crossinterest.methods.role_projection.train import RoleTrainer

    class Dataset:
        mapping = globals()["mapping"]()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"input_ids": [1, 4, 5, 6, 40, 2], "attention_mask": [1] * 6,
                    "labels": [-100, -100, -100, -100, 40, 2], "history_frame": frame()}

    def make_trainer():
        tokenizer = TinyTokenizer()
        tokenizer.pad_token_id = 0
        return RoleTrainer(model=tiny_qwen_peft(), role_mode="projected", stage="novel",
            args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True,
                max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=2,
                save_steps=1, save_total_limit=2, learning_rate=1e-4, seed=20260825,
                remove_unused_columns=False, dataloader_pin_memory=False),
            train_dataset=Dataset(), processing_class=tokenizer, sparse_embedding_save=True,
            label_token_ids=list(range(40, 47)))

    trainer = make_trainer()
    trainer.train()
    expected = {name: value.detach().clone() for name, value in trainer.model.named_parameters() if value.requires_grad}
    trainer.controller.close()
    restored = make_trainer()
    restored.train(resume_from_checkpoint=str(tmp_path / "checkpoint-1"))
    for name, value in restored.model.named_parameters():
        if name in expected:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
    assert restored.state.global_step == 2
    restored.controller.close()