"""Readout algebra and causal generation integration regressions."""

import pytest
import torch

from crossinterest.methods.mass_readout.model import MassReadout, MassReadoutController, preserve_mass
from tests.methods.test_interest_composition import tiny_model, tiny_qwen_peft
from tests.methods.test_role_projection import TinyTokenizer, mapping


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_mass_and_gradient(dtype):
    torch.manual_seed(20260825)
    hidden = torch.randn(4, 12).to(dtype)
    logits = torch.randn(4, 112).to(dtype)
    random_state = torch.random.get_rng_state().clone()
    head = MassReadout(12, 112)
    assert torch.equal(random_state, torch.random.get_rng_state())
    updated = preserve_mass(logits, head(hidden))
    assert torch.equal(updated, logits.float())
    loss = torch.nn.functional.cross_entropy(updated, torch.arange(4))
    loss.backward()
    assert head.weight.grad.isfinite().all() and head.weight.grad.norm() > 0
    with torch.no_grad():
        head.weight.normal_(std=0.1)
    updated = preserve_mass(logits, head(hidden))
    torch.testing.assert_close(updated.logsumexp(-1), logits.float().logsumexp(-1))
    outside = torch.randn(4, 31)
    before = torch.cat((logits.float(), outside), dim=-1).softmax(-1)
    after = torch.cat((updated, outside), dim=-1).softmax(-1)
    torch.testing.assert_close(before[:, 112:], after[:, 112:])
    torch.testing.assert_close(before[:, :112].sum(-1), after[:, :112].sum(-1))


@pytest.mark.parametrize("factory", [tiny_model, tiny_qwen_peft])
def test_first_position_save_and_twenty_beam(tmp_path, factory):
    model = factory().eval()
    inputs = torch.tensor([[1, 4, 5, 6, 40, 2]])
    with torch.no_grad():
        original = model(inputs).logits
    controller = MassReadoutController(model, TinyTokenizer(), mapping())
    controller.set_frames([{"prediction_position": 3}])
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).logits, original, atol=0, rtol=0)
        controller.head.weight.normal_(std=0.05)
        changed = model(inputs).logits
    difference = changed - original
    assert difference[:, 3, controller.label_ids].norm() > 0
    difference[:, 3, controller.label_ids] = 0
    assert difference.count_nonzero() == 0
    controller.save(tmp_path)
    restored = factory().eval()
    loaded = MassReadoutController(restored, TinyTokenizer(), mapping(), tmp_path)
    loaded.set_frames([{"prediction_position": 3}])
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs).logits, changed, atol=0, rtol=0)
        count = loaded.applied_calls
        generated = restored.generate(inputs[:, :4], attention_mask=torch.ones(1, 4, dtype=torch.long),
            num_beams=20, num_return_sequences=20, do_sample=False, max_new_tokens=3,
            return_dict_in_generate=True, output_scores=True)
    assert generated.scores[0].shape == (20, 64)
    assert generated.sequences.shape[0] == 20
    assert loaded.applied_calls == count + 1
    controller.close()
    loaded.close()


def test_qwen_checkpointing_and_accumulation(tmp_path):
    from transformers import TrainingArguments
    from crossinterest.methods.mass_readout.train import MassTrainer

    class Dataset:
        mapping = globals()["mapping"]()

        def __len__(self):
            return 8

    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = 0
    results = []
    for checkpointing in (False, True):
        model = tiny_qwen_peft().train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainer = MassTrainer(model=model,
            args=TrainingArguments(output_dir=str(tmp_path / str(checkpointing)), report_to="none", use_cpu=True,
                                   gradient_accumulation_steps=8),
            train_dataset=Dataset(), processing_class=tokenizer, label_token_ids=list(range(40, 47)))
        with torch.no_grad():
            trainer.controller.head.weight.fill_(0.02)
            trainer.controller.head.weight[0].add_(0.01)
        torch.manual_seed(20260825)
        for step in range(8):
            loss = trainer.compute_loss(model, {
                "input_ids": torch.tensor([[1, 4, 5, 6, 40, 2]]),
                "labels": torch.tensor([[-100, -100, -100, -100, 40, 2]]),
                "history_frames": [{"prediction_position": 3}]})
            (loss / 8).backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None}
        assert gradients["interest_readout.weight"].norm() > 0
        assert any(value.norm() > 0 for name, value in gradients.items() if "lora_B" in name)
        results.append(gradients)
        trainer.controller.close()
    assert results[0].keys() == results[1].keys()
    for name, gradient in results[0].items():
        torch.testing.assert_close(gradient, results[1][name], atol=2e-7, rtol=2e-5, msg=name)


def test_n_only_commands(tmp_path):
    from crossinterest.methods.mass_readout.runner import training_command, evaluation_command

    prepared = {"files": {"novel": {"path": "train.parquet", "rows": 112000, "epochs": 1},
                          "smoke_novel": {"path": "smoke.parquet", "rows": 224, "epochs": 1}},
                "training_layout": {"world_size": 4, "micro_batch": 1, "accumulation": 8, "gradient_checkpointing": True},
                "shards": {"0": {"path": "data/dev.parquet"}}, "evaluation": {"smoke_limit": 4}}
    contract = {"source_adapter": "completed_cir/final_adapter"}
    for smoke in (True, False):
        command = training_command("model", tmp_path / "novel", smoke, contract, prepared)
        assert "--nproc_per_node=4" in command and "--gradient-checkpointing" in command
        assert command[command.index("--gradient-accumulation-steps") + 1] == "8"
        assert command[command.index("--init-adapter") + 1].endswith(contract["source_adapter"])
        assert "--role-mode" not in command and "--stage" not in command
        assert ("--smoke" in command) == smoke
        evaluation = evaluation_command("model", tmp_path, smoke, prepared)
        assert 'crossinterest.methods.mass_readout.evaluate' in evaluation
        assert evaluation[evaluation.index("--num-beams") + 1] == "20"
        assert evaluation[evaluation.index("--max-new-tokens") + 1] == "32"
        assert ("--limit" in evaluation) == smoke


def test_real_trainer_gradient_audit_and_save(tmp_path):
    import json
    from transformers import TrainingArguments
    from crossinterest.methods.mass_readout.train import MassTrainer

    class Dataset:
        mapping = globals()["mapping"]()

        def __len__(self):
            return 16

        def __getitem__(self, index):
            return {"input_ids": [1, 4, 5, 6, 40 + index % 7, 2], "attention_mask": [1] * 6,
                    "labels": [-100, -100, -100, -100, 40 + index % 7, 2],
                    "history_frame": {"prediction_position": 3}}

    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = 0
    trainer = MassTrainer(model=tiny_qwen_peft(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none", use_cpu=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=8, max_steps=2,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-4, remove_unused_columns=False, save_strategy="no"),
        train_dataset=Dataset(), processing_class=tokenizer, label_token_ids=list(range(40, 47)))
    result = trainer.train()
    assert result.global_step == 2
    record = json.loads((tmp_path / "GRADIENT_0000.json").read_text())
    assert 0 <= record["normal_energy_fraction"] <= 1.000001
    assert record["gradient_norm"] > 0
    trainer.save_model(str(tmp_path / "final_adapter"))
    assert (tmp_path / "final_adapter/interest_readout.safetensors").is_file()
    assert (tmp_path / "final_adapter/adapter_model.safetensors").is_file()
    before = trainer.controller.head.weight.detach().clone()
    with torch.no_grad():
        trainer.controller.head.weight.zero_()
    trainer.controller.load(tmp_path / "final_adapter")
    torch.testing.assert_close(before, trainer.controller.head.weight, atol=0, rtol=0)
    trainer.controller.close()