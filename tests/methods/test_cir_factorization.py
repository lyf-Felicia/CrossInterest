"""Focused regression tests for the CIR factorization experiment."""

from crossinterest.paths import source_workspace as _source_workspace

import pytest
import torch
from torch.nn import functional as F

from crossinterest.methods.factorization.loss import ARMS, factorized_loss


def example(dtype=torch.float64):
    generator = torch.Generator().manual_seed(20260919)
    logits = torch.randn(8, 112, generator=generator, dtype=dtype)
    targets = F.one_hot(torch.arange(8), 112).bool()
    seen = torch.rand(8, 112, generator=generator) < 0.2
    seen &= ~targets
    seen[0] = False
    seen[1] = ~targets[1]
    return logits, targets, seen


def test_original_ce_exact_and_accumulated_update():
    logits, targets, seen = example(torch.float32)
    original = torch.nn.Parameter(logits.clone())
    factored = torch.nn.Parameter(logits.clone())
    original_optimizer = torch.optim.AdamW([original], lr=1e-4)
    factored_optimizer = torch.optim.AdamW([factored], lr=1e-4)
    for index in range(8):
        reference = -(targets[index:index + 1].float() * F.log_softmax(original[index:index + 1].float(), -1)).sum(-1).mean()
        actual, _ = factorized_loss(factored[index:index + 1], targets[index:index + 1],
                                    seen[index:index + 1], "baseline")
        assert torch.equal(actual, reference)
        (reference / 8).backward()
        (actual / 8).backward()
    assert torch.equal(original.grad, factored.grad)
    original_optimizer.step()
    factored_optimizer.step()
    assert torch.equal(original, factored)


@pytest.mark.parametrize("arm", ARMS)
def test_decomposition_and_gradient_norm(arm):
    initial, targets, seen = example()
    logits = initial.clone().requires_grad_()
    loss, diagnostics = factorized_loss(logits, targets, seen, arm)
    assert torch.allclose(diagnostics["relation_loss"] + diagnostics["exclusion_loss"],
                          diagnostics["original_loss"], atol=1e-12)
    gradient = torch.autograd.grad(loss, logits)[0] * len(logits)
    reference = logits.detach().softmax(-1) - targets.double()
    assert torch.allclose(gradient.norm(dim=-1), reference.norm(dim=-1), atol=1e-12)
    assert torch.allclose(gradient[:2], reference[:2], atol=1e-12)
    if arm != "baseline":
        assert not torch.allclose(gradient[2:], reference[2:])


def test_invalid_targets_and_nonfinite():
    logits, targets, seen = example()
    with pytest.raises(ValueError, match="unique and unseen"):
        factorized_loss(logits, targets, seen | targets, "relation_true")
    with pytest.raises(ValueError, match="unique and unseen"):
        factorized_loss(logits, torch.zeros_like(targets), seen, "relation_true")
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        factorized_loss(logits, targets, seen, "baseline")


def test_saturated_fallback_finite():
    logits = torch.full((1, 112), -1000.0, requires_grad=True)
    logits = logits + F.one_hot(torch.tensor([0]), 112) * 2000
    targets = F.one_hot(torch.tensor([0]), 112).bool()
    seen = ~targets
    for arm in ARMS:
        loss, diagnostics = factorized_loss(logits, targets, seen, arm)
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert diagnostics["fallback"].all()


def test_randomized_masks_preserve_exact_margins():
    import numpy as np
    from crossinterest.methods.factorization.data import randomized_masks, match_donors

    _, positives, seen = example()
    masks = seen.numpy()
    targets = positives.long().argmax(-1).numpy()
    first, audit = randomized_masks(masks, targets, rounds=5)
    second, _ = randomized_masks(masks, targets, rounds=5)
    assert np.array_equal(first, second)
    assert np.array_equal(first.sum(0), masks.sum(0))
    assert np.array_equal(first.sum(1), masks.sum(1))
    assert not first[np.arange(len(first)), targets].any()
    assert audit["accepted"] > 0
    rows = [{"user_id": str(index), "target_interest": f"class{index}",
             "historical_interests": ["outside"] * 5} for index in range(4)]
    donors, _ = match_donors(rows, [f"class{index}" for index in range(4)])
    assert sorted(donors) == list(range(4))
    assert all(index != donor for index, donor in enumerate(donors))


def test_readonly_gate_decisions_and_fp32_audit():
    from crossinterest.methods.factorization.gate import summarize_logits, branch_decisions

    logits, positives, seen = example(dtype=torch.float32)
    result = summarize_logits(logits, positives, seen, seen)
    decisions = branch_decisions({name: result for name in ("base", "aux1", "aux3")})
    assert all(value["passed"] for value in decisions.values())
    saturated = {arm: {**stats, "active": False} for arm, stats in result.items()}
    decisions = branch_decisions({"base": result, "aux1": saturated, "aux3": saturated})
    assert not any(value["passed"] for value in decisions.values())


def test_real_trainer_ga8_baseline_update(tmp_path):
    import copy
    from unittest.mock import Mock
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
    from crossinterest.methods.mechanism.train import MechanismTrainer, MechanismCollator
    from crossinterest.methods.factorization.train import FactorTrainer, FactorCollator

    torch.manual_seed(42)
    original = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.0))
    models = (original, copy.deepcopy(original))
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    rows = [{"input_ids": [1, 4, 5, 16 + index, 2], "labels": [-100] * 3 + [16 + index, 2],
             "attention_mask": [1] * 5, "positive_mask": [label == index for label in range(112)],
             "seen_mask": [label >= 80 for label in range(112)]} for index in range(8)]
    trainers = []
    for name, trainer_type, model in zip(("original", "factor"), (MechanismTrainer, FactorTrainer), models):
        options = {"factor_arm": "baseline"} if name == "factor" else {}
        trainer = trainer_type(model=model, processing_class=tokenizer,
            args=TrainingArguments(output_dir=str(tmp_path / name), use_cpu=True, report_to="none",
                gradient_accumulation_steps=8, per_device_train_batch_size=1, max_steps=1,
                learning_rate=1e-4, save_strategy="no", disable_tqdm=True, remove_unused_columns=False),
            train_dataset=rows, stage="auxiliary", relation=False, init_adapter=None,
            label_token_ids=list(range(16, 128)), **options)
        assert trainer.model_accepts_loss_kwargs is False
        trainers.append(trainer)
    raw = trainers[0].compute_loss(models[0], MechanismCollator(0)(rows)).detach()
    factor = trainers[1].compute_loss(models[1], FactorCollator(0)(rows)).detach()
    assert torch.equal(raw, factor)
    for trainer in trainers:
        trainer.train()
    for first, second in zip(models[0].parameters(), models[1].parameters()):
        assert torch.equal(first, second)
    assert trainers[0].state.global_step == trainers[1].state.global_step == 1


def test_factor_commands_only_change_auxiliary_entrypoint(tmp_path):
    import json
    import crossinterest.methods.cir.runner as cir_run
    from crossinterest.methods.factorization.runner import training_command

    path = cir_run.preflight_path("qwen35_9b", "external_balanced", 3)
    prepared = json.loads(path.read_text())
    for smoke in (True, False):
        for stage in ("auxiliary", "novel"):
            adapter = tmp_path / "auxiliary/final_adapter" if stage == "novel" else None
            original = cir_run.training_command("model", stage, tmp_path / stage, smoke, prepared, adapter)
            factor = training_command("model", stage, tmp_path / stage, smoke, prepared, "relation_true", adapter)
            restored = factor[:-2]
            restored[restored.index('crossinterest.methods.factorization.train')] = 'crossinterest.methods.cir.train'
            assert original == restored
            assert factor[-2:] == ["--factor-arm", "relation_true"]
            assert ("--init-adapter" in factor) == (stage == "novel")


def test_probe_common_normalization():
    from crossinterest.methods.factorization.probe import diagnostic_statistics, macro_statistics

    logits = torch.tensor([2.0, 9.0, 1.0, 0.0])
    other = torch.tensor([1.0, -9.0, 9.0, 0.0])
    seen = torch.tensor([False, True, False, False])
    donor_seen = torch.tensor([False, False, True, False])
    values = diagnostic_statistics(logits, other, 0, seen, donor_seen)
    expected = torch.log_softmax(logits.double()[[0, 3]], 0)[0] - torch.log_softmax(other.double()[[0, 3]], 0)[0]
    assert values["history_advantage"] == pytest.approx(expected.item(), abs=1e-14, rel=0)
    assert values["common_unseen_count"] == 2
    summary = macro_statistics([{"target_interest": "A", **values}])
    assert summary["history_advantage"] == values["history_advantage"]


def test_execution_blocks_formal_until_smoke_audited(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import crossinterest.methods.factorization.runner as runner
    import huggingface_hub

    prepared = {"model_repo": "fake", "model_revision": "revision"}
    monkeypatch.setattr(runner, "verify_package", lambda: {})
    monkeypatch.setattr(runner, "preflight", lambda: {"original": prepared})
    monkeypatch.setattr(runner, "trainer_equivalence", lambda output: {"passed": True})
    monkeypatch.setattr(runner, "sha256_file", lambda path: "hash")
    monkeypatch.setattr(runner, "sources", lambda: {})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: "model")
    commands = []

    def training_command(model, stage, directory, smoke, contract, arm, previous):
        commands.append((stage, smoke, previous))
        return ["training"]

    monkeypatch.setattr(runner, "training_command", training_command)
    monkeypatch.setattr(runner.subprocess, "run", Mock())
    monkeypatch.setattr(runner, "audit_stage", lambda *args: {})

    def fail_smoke(output, smoke_only):
        assert smoke_only
        raise ValueError("smoke audit failed")

    monkeypatch.setattr(runner, "audit_output", fail_smoke)
    with pytest.raises(ValueError, match="smoke audit failed"):
        runner.execute("relation_true", tmp_path)
    assert [smoke for _, smoke, _ in commands] == [True, True]
    assert commands[0] == ("auxiliary", True, None)
    assert commands[1][2] == tmp_path / "smoke/auxiliary/final_adapter"
    commands.clear()
    monkeypatch.setattr(runner, "audit_output", lambda *args, **kwargs: {"passed": True})
    runner.execute("relation_true", tmp_path)
    assert [smoke for _, smoke, _ in commands] == [True, True, False, False]
    assert commands[2] == ("auxiliary", False, None)
    assert commands[3][2] == tmp_path / "formal/auxiliary/final_adapter"
    with pytest.raises(ValueError, match="not authorized"):
        runner.execute("baseline", tmp_path)


def test_collector_requires_raw_probe_and_safe_paths():
    from crossinterest.methods.factorization.audit import select_inventory

    names = ["EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json", "smoke/RESULT_AUDIT.json"]
    names += [f"smoke/{stage}_PROBE{suffix}" for stage in ("auxiliary", "novel") for suffix in (".json", ".npz")]
    inventory = [{"name": name} for name in names]
    assert select_inventory(inventory, smoke_only=True) == inventory
    with pytest.raises(ValueError, match="incomplete"):
        select_inventory(inventory[:-1], smoke_only=True)
    with pytest.raises(ValueError, match="incomplete"):
        select_inventory(inventory)
    with pytest.raises(ValueError, match="unsafe"):
        select_inventory(inventory + [{"name": "../outside"}], smoke_only=True)
    with pytest.raises(ValueError, match="duplicated"):
        select_inventory(inventory + inventory[:1], smoke_only=True)
    reference = [{"name": "REFERENCE_PROBE_AUDIT.json"}]
    reference += [{"name": f"{stage}_PROBE{suffix}"} for stage in ("aux3", "novel") for suffix in (".json", ".npz")]
    assert select_inventory(reference, reference=True) == reference


def test_collector_preserves_cloud_receipts(tmp_path, monkeypatch):
    import json
    import crossinterest.methods.factorization.audit as collector

    monkeypatch.setattr(collector, "verify_package", lambda: {})
    monkeypatch.setattr(collector, "preflight", lambda: {"original": {}})
    monkeypatch.setattr(collector, "sources", lambda: {})
    monkeypatch.setattr(collector, "sha256_file", lambda path: "hash")
    execution = {"factor_arm": "relation_true", "factor_preflight_sha256": "hash",
                 "code_sha256": {}, "data_contract": {}, "fresh_formal_after_smoke": True}
    (tmp_path / "EXECUTION_CONTRACT.json").write_text(json.dumps(execution))
    (tmp_path / "TRAINER_EQUIVALENCE.json").write_text(json.dumps({"passed": True, "code_sha256": {}, "test_sha256": "hash"}))
    for stage in ("auxiliary", "novel"):
        directory = tmp_path / "smoke" / stage
        directory.mkdir(parents=True)
        for name in ("STAGE_AUDIT.json", "FACTOR_STAGE_AUDIT.json"):
            (directory / name).write_text('{"passed": true}')
    (tmp_path / "smoke/RESULT_AUDIT.json").write_text('{"passed": true}')
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def recompute(directory, smoke_only):
        assert smoke_only
        assert (directory / "EXECUTION_CONTRACT.json").is_symlink()
        for relative in before:
            if relative.endswith("AUDIT.json"):
                path = directory / relative
                assert not path.exists()
                path.write_text('{"passed": true}')
        return {"passed": True, "phases": {"smoke": {"passed": True}}}

    monkeypatch.setattr(collector, "audit_output", recompute)
    assert collector.audit_download(tmp_path, "relation_true", smoke_only=True)["passed"]
    assert all((tmp_path / relative).read_bytes() == value for relative, value in before.items())


def test_collector_float_tolerance_does_not_relax_identity():
    from crossinterest.methods.factorization.audit import require_probe_statistics

    expected = {"records": [{"id": "sample", "common_unseen_count": 100, "conditional_nll": 4.0}]}
    require_probe_statistics({"records": [{"id": "sample", "common_unseen_count": 100,
                                          "conditional_nll": 4.0 + 4e-15}]}, expected)
    for field, value in (("id", "different"), ("common_unseen_count", 101), ("common_unseen_count", 100.0),
                         ("conditional_nll", 4.0 + 1e-8), ("conditional_nll", float("nan")),
                         ("conditional_nll", float("inf")), ("conditional_nll", "4.0")):
        actual = {"records": [{**expected["records"][0], field: value}]}
        with pytest.raises(ValueError, match="probe"):
            require_probe_statistics(actual, expected)
    with pytest.raises(ValueError, match="count"):
        require_probe_statistics({"records": []}, expected)


def test_watcher_detects_platform_restart_and_retry_log(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import crossinterest.methods.runtime.performance as watcher

    now = datetime.now(timezone.utc)
    previous = {"status": "Running", "log_file": "user_logs/retry_001/std_log.txt",
                "interruption": None, "last_active_at": now.isoformat()}
    event = "[09/19/2026 11:59:28 +00:00] Restarting: Reason: can't find any pods."
    response = SimpleNamespace(text=event, raise_for_status=lambda: None)
    session = SimpleNamespace(get=lambda *args, **kwargs: response)
    details = SimpleNamespace(status="Running", log_files={
        "logs/azureml/virtualcluster_activity_log.txt": "activity"})
    client = SimpleNamespace(jobs=SimpleNamespace(_runs_operations=SimpleNamespace(
        get_run_details=lambda job_id: details)))
    current = watcher.observe(client, session, "job", previous, now)
    assert current["interruption"] == event
    assert watcher.classify_observation(current, previous, now, 1800) == "scheduling_interruption"
    assert watcher.classify_observation(current, current, now, 1800) is None
    retry = {**previous, "log_file": "user_logs/retry_002/std_log.txt"}
    assert watcher.classify_observation(retry, previous, now, 1800) == "attempt_restart"
    assert watcher.classify_observation(retry, retry, now, 1800) is None
    assert watcher.classify_observation(retry, None, now, 1800) is None
    assert watcher.classify_observation({**retry, "status": "Completed"}, previous, now, 1800) == "terminal"
    assert watcher.classify_observation({**retry, "status": "Failed"}, previous, now, 1800) == "job_failure"


def test_speed_layout_preserves_frozen_budget(tmp_path):
    import json
    import crossinterest.methods.cir.runner as cir_run
    from crossinterest.methods.factorization.runner import training_command
    from crossinterest.methods.factorization.speed import parallel_command

    prepared = json.loads(cir_run.preflight_path("qwen35_9b", "external_balanced", 3).read_text())
    original = training_command("model", "auxiliary", tmp_path, False, prepared, "relation_true")
    for world_size in (4, 8, 16):
        command = parallel_command(original, world_size)
        restored = parallel_command(command, 4)
        assert restored == original
        assert int(command[command.index("--gradient-accumulation-steps") + 1]) * world_size == 32
        assert command[command.index("--per-device-batch-size") + 1] == "1"
    no_gc = parallel_command(original, 8, False)
    assert "--no-gradient-checkpointing" in no_gc and "--gradient-checkpointing" not in no_gc
    with pytest.raises(ValueError, match="effective-batch"):
        parallel_command(original, 3)


def test_checkpointing_toggle_preserves_dropout_update(tmp_path):
    import copy
    from unittest.mock import Mock
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments
    from crossinterest.methods.factorization.train import FactorTrainer

    torch.manual_seed(42)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.0))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"], task_type="CAUSAL_LM"))
    models = (model, copy.deepcopy(model))
    rows = [{"input_ids": [1, 4, 5, 16 + index, 2], "labels": [-100] * 3 + [16 + index, 2],
             "attention_mask": [1] * 5, "positive_mask": [label == index for label in range(112)],
             "seen_mask": [label >= 80 for label in range(112)]} for index in range(8)]
    for checkpointing, current in zip((True, False), models):
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        current.config.use_cache = False
        current.enable_input_require_grads()
        trainer = FactorTrainer(model=current, processing_class=tokenizer,
            args=TrainingArguments(output_dir=str(tmp_path / str(checkpointing)), use_cpu=True,
                report_to="none", gradient_accumulation_steps=8, per_device_train_batch_size=1,
                max_steps=1, learning_rate=1e-4, save_strategy="no", disable_tqdm=True,
                remove_unused_columns=False, gradient_checkpointing=checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False}),
            train_dataset=rows, stage="auxiliary", relation=False, init_adapter=None,
            factor_arm="relation_true", label_token_ids=list(range(16, 128)))
        trainer.train()
    for original, optimized in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(original, optimized, rtol=0, atol=0)


def test_accelerated_layout_rejects_changed_batch():
    from crossinterest.methods.factorization.resume import validate_layout

    runtime = {"world_size": 8, "per_device_batch_size": 1, "gradient_accumulation_steps": 4,
               "effective_batch_size": 32, "gradient_checkpointing": False,
               "ddp_find_unused_parameters": False, "qwen_fast_path": {"available": True}}
    validate_layout(runtime, 8, False)
    for field, value in (("world_size", 4), ("gradient_accumulation_steps", 8),
                         ("effective_batch_size", 64), ("gradient_checkpointing", True)):
        with pytest.raises(ValueError):
            validate_layout({**runtime, field: value}, 8, False)


def test_eight_gpu_layout_preserves_global_batches():
    from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
    from torch.utils.data import BatchSampler

    def batches(world_size, rows, epoch):
        ranks = []
        for rank in range(world_size):
            sampler = SeedableRandomSampler(range(rows), data_seed=42)
            sampler.set_epoch(epoch)
            ranks.append(list(BatchSamplerShard(BatchSampler(sampler, batch_size=1, drop_last=False),
                num_processes=world_size, process_index=rank, split_batches=False, even_batches=True)))
        accumulation = 32 // world_size
        global_batches = [sorted(item for rank in ranks for batch in rank[start:start + accumulation] for item in batch)
                          for start in range(0, len(ranks[0]), accumulation)]
        for step in (2500, 3000, 4500):
            offset = step % len(global_batches)
            remainder = [rank[offset * accumulation:] for rank in ranks]
            resumed = [sorted(item for rank in remainder for batch in rank[start:start + accumulation] for item in batch)
                       for start in range(0, len(remainder[0]), accumulation)]
            assert resumed == global_batches[offset:]
        return global_batches

    for rows, epochs in ((83888, 3), (112000, 1)):
        for epoch in range(epochs):
            original, accelerated = batches(4, rows, epoch), batches(8, rows, epoch)
            assert original == accelerated
            assert len(original) == (rows + 31) // 32 and len(original[-1]) == (rows % 32 or 32)
            assert sorted(item for batch in accelerated for item in batch) == list(range(rows))


def test_acceleration_checkpoint_copy_preserves_source(tmp_path):
    import json
    import random
    import numpy as np
    from safetensors.torch import save_file
    from crossinterest.methods.factorization.resume import copy_checkpoint
    from crossinterest.methods.alignment.data import sha256_file

    source = tmp_path / "source/checkpoint-2500"
    source.mkdir(parents=True)
    state = {"global_step": 2500, "max_steps": 7866, "num_train_epochs": 3}
    (source / "trainer_state.json").write_text(json.dumps(state))
    (source / "adapter_config.json").write_text("{}")
    (source / "FACTOR_RUNTIME.json").write_text(json.dumps({"arm": "relation_true", "model_accepts_loss_kwargs": False}))
    save_file({"layer.lora_B.weight": torch.ones(2, 2)}, str(source / "adapter_model.safetensors"))
    torch.save({"state": {0: {"step": torch.tensor(2500.), "exp_avg": torch.ones(2)}}}, source / "optimizer.pt")
    torch.save({"last_epoch": 2500}, source / "scheduler.pt")
    for rank in range(4):
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
                    "cuda": [torch.tensor([index + rank], dtype=torch.uint8) for index in range(4)]},
                   source / f"rng_state_{rank}.pth")
    before = {path.name: sha256_file(path) for path in source.iterdir()}
    for path in source.iterdir():
        path.chmod(0o444)
    source.chmod(0o555)
    destination = tmp_path / "resumed/formal/auxiliary"
    destination.mkdir(parents=True)
    try:
        result = copy_checkpoint(source.parent, destination, "relation_true", 8)
        assert source.stat().st_mode & 0o777 == 0o555
        assert all(path.stat().st_mode & 0o777 == 0o444 for path in source.iterdir())
    finally:
        source.chmod(0o755)
    assert result["passed"] and result["checkpoint_step"] == 2500
    assert result["original_sha256"] == before
    assert {path.name: sha256_file(path) for path in source.iterdir()} == before
    for rank in range(8):
        rng = torch.load(destination / "checkpoint-2500" / f"rng_state_{rank}.pth", weights_only=False)
        assert len(rng["cuda"]) == 8
        assert rng["cuda"][rank].item() == 2 * (rank % 4)
    assert result["bitwise_trajectory_equivalence_claimed"] is False


def test_accelerated_inventory_requires_handoff_and_stress():
    from crossinterest.methods.factorization.audit import select_inventory

    names = {"EXECUTION_CONTRACT.json", "TRAINER_EQUIVALENCE.json", "COMPLETE.json"}
    names.update(f"{phase}/RESULT_AUDIT.json" for phase in ("smoke", "formal"))
    names.update(f"{phase}/{stage}_PROBE{suffix}" for phase in ("smoke", "formal")
                 for stage in ("auxiliary", "novel") for suffix in (".json", ".npz"))
    names.update({"PERFORMANCE_CONTRACT.json", "PERFORMANCE_BENCHMARK.json", "RESUME_AUDIT.json",
                  "HANDOFF_AUTHORIZATION.json", "READY_FOR_HANDOFF.json"})
    names.update(f"ORIGINAL_RNG/rng_state_{rank}.pth" for rank in range(4))
    names.update(f"formal/{stage}/STRESS_RANK_{rank}.json" for stage in ("auxiliary", "novel") for rank in range(8))
    inventory = [{"name": name} for name in sorted(names)]
    assert {entry["name"] for entry in select_inventory(inventory)} == names
    for missing in ("HANDOFF_AUTHORIZATION.json", "READY_FOR_HANDOFF.json", "ORIGINAL_RNG/rng_state_3.pth", "formal/novel/STRESS_RANK_7.json"):
        with pytest.raises(ValueError, match="incomplete"):
            select_inventory([entry for entry in inventory if entry["name"] != missing])


def test_stress_restores_rng_counters_and_weights():
    import random
    from types import SimpleNamespace
    import numpy as np
    from crossinterest.methods.factorization.resume import preserve_training_state

    model = torch.nn.Linear(3, 2)
    trainer = SimpleNamespace(factor_totals=torch.tensor([1.0, 2.0]), factor_rows=7)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for fail in (False, True):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        try:
            with preserve_training_state(trainer, model):
                random.random()
                np.random.random()
                model(torch.rand(3)).sum().backward()
                trainer.factor_rows += 1
                trainer.factor_totals.add_(5)
                trainer.factor_stat_names = ("probe",)
                if fail:
                    raise RuntimeError("stress failure")
        except RuntimeError as error:
            assert fail and str(error) == "stress failure"
        assert random.random() == expected[0]
        assert np.random.random() == expected[1]
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
        assert trainer.factor_rows == 7 and not hasattr(trainer, "factor_stat_names")
        torch.testing.assert_close(trainer.factor_totals, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
        for name, value in model.named_parameters():
            assert value.grad is None
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_pre_handoff_stress_requires_real_resume_and_memory():
    from crossinterest.methods.factorization.resume import validate_stress_gate

    row = {"passed": True, "tokens": 3252, "optimizer_updates": 0,
           "restored_checkpoint_step": 3000, "restored_optimizer_steps": [3000],
           "tested_objectives": ["stage_loss", "original_full_vocabulary_causal_ce"],
           "peak_allocated_bytes": 60 * 1024**3}
    validate_stress_gate([row] * 8, 3000)
    for field, value in (("restored_checkpoint_step", 0), ("restored_optimizer_steps", []),
                         ("optimizer_updates", 1), ("peak_allocated_bytes", 79 * 1024**3)):
        with pytest.raises(ValueError):
            validate_stress_gate([row] * 7 + [{**row, field: value}], 3000)
    with pytest.raises(ValueError, match="all ranks"):
        validate_stress_gate([row] * 7, 3000)


def test_bootstrap_accepts_readonly_asset_without_modifying_it(tmp_path):
    import hashlib
    import importlib.util
    import json
    import stat
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("speed_bootstrap", _source_workspace('eval_interest_id_minionerec') / "cir_factorization/speed_launch/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    source, destination, overlay = (tmp_path / name for name in ("source", "destination", "overlay"))
    receipt_dir = source / "eval_interest_id_minionerec/cir_factorization"
    receipt_dir.mkdir(parents=True)
    relative = "eval_interest_id_minionerec/frozen.py"
    (source / relative).write_text("immutable = True\n")
    (receipt_dir / "PACKAGE_RECEIPT.json").write_text(json.dumps({"files": {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()}}))
    overlay.mkdir()
    (overlay / "cir_factor_speed.py").write_text("benchmark = True\n")
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    before = {str(path.relative_to(source)): stat.S_IMODE(path.stat().st_mode) for path in source.rglob("*")}
    try:
        bootstrap.stage_package(source, destination, overlay)
        assert (destination / "eval_interest_id_minionerec/cir_factor_speed.py").read_text() == "benchmark = True\n"
        assert (source / relative).read_bytes() == (destination / relative).read_bytes()
        assert {str(path.relative_to(source)): stat.S_IMODE(path.stat().st_mode) for path in source.rglob("*")} == before
    finally:
        source.chmod(0o755)
        for path in source.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)


def test_benchmark_gate_rejects_slow_incomplete_or_unsafe_profiles():
    from copy import deepcopy
    from crossinterest.methods.factorization.resume import validate_benchmark

    def profile(world_size, checkpointing, seconds):
        return {"completed": True, "returncode": 0, "ranks": [{"rank": rank, "world_size": world_size,
            "gradient_accumulation_steps": 32 // world_size, "gradient_checkpointing": checkpointing,
            "effective_batch_size": 32, "median_step_seconds": seconds, "steps": list(range(24)),
            "sampler_configuration": {"use_seedable_sampler": True, "even_batches": True, "split_batches": False},
            "stress": {"passed": True, "tokens": 3252, "optimizer_updates": 0,
                "tested_objectives": ["stage_loss", "original_full_vocabulary_causal_ce"],
                "peak_allocated_bytes": 60 * 1024**3}} for rank in range(world_size)]}

    result = {"scope": "not_formal_results", "profiles": {
        "four_gc": profile(4, True, 6.0), "eight_no_gc": profile(8, False, 2.5)}}
    assert validate_benchmark(result, 8, False)["measured_speedup"] == 2.4
    for field, value in (("median_step_seconds", 5.0), ("steps", []), ("effective_batch_size", 64)):
        changed = deepcopy(result)
        changed["profiles"]["eight_no_gc"]["ranks"][0][field] = value
        with pytest.raises(ValueError):
            validate_benchmark(changed, 8, False)
    changed = deepcopy(result)
    changed["profiles"]["eight_no_gc"]["ranks"][0]["stress"]["peak_allocated_bytes"] = 79 * 1024**3
    with pytest.raises(ValueError, match="memory"):
        validate_benchmark(changed, 8, False)


def test_watcher_handoff_requires_structured_ready_event():
    from crossinterest.methods.runtime.performance import has_handoff_ready

    assert has_handoff_ready('{"event": "waiting_for_exclusive_handoff", "arm": "relation_true"}')
    assert not has_handoff_ready('{"event": "accelerated_checkpoint_ready"}')
    assert not has_handoff_ready('print("waiting_for_exclusive_handoff")')
    assert not has_handoff_ready('{"other": "waiting_for_exclusive_handoff"}')


def test_automatic_handoff_requires_audit_stop_and_continuation(tmp_path):
    import json
    from types import SimpleNamespace
    from crossinterest.methods.factorization.checkpoint import advance

    row = {"old_job_id": "old", "new_job_id": "new", "arm": "relation_true"}

    class Backend:
        old_status, new_status, ready, invalid = "Running", "Running", False, False
        finished_auxiliary = False
        actions = []
        text = ""

        def details(self, job_id):
            return SimpleNamespace(status=self.old_status if job_id == "old" else self.new_status,
                                   end_time_utc="2026-09-19T20:00:00+00:00" if self.old_status == "Canceled" else None)

        def exists(self, row, name):
            return self.ready

        def auxiliary_complete(self, row):
            return self.finished_auxiliary

        def validate_job(self, row):
            if self.invalid:
                raise ValueError("identity")

        def audit(self, directory, row):
            result = {"passed": True, "step": 5000, "performance_contract_sha256": "hash", "ready_sha256": "ready"}
            (directory / "INDEPENDENT_HANDOFF_AUDIT.json").write_text(json.dumps(result))
            self.actions.append("audit")
            return result

        def revalidate(self, row, state):
            if self.invalid:
                raise ValueError("identity")

        def cancel(self, row):
            self.actions.append("cancel")

        def authorize(self, row, directory, authorization):
            assert self.old_status == "Canceled" and authorization["original_job_stopped"]
            self.actions.append("authorize")

        def stdout(self, details):
            return self.text

    backend, state = Backend(), {}
    advance(row, tmp_path, backend, state)
    assert backend.actions == []
    backend.ready, backend.invalid = True, True
    with pytest.raises(ValueError):
        advance(row, tmp_path, backend, state)
    assert backend.actions == []
    backend.invalid = False
    advance(row, tmp_path, backend, state)
    assert backend.actions == ["audit", "cancel"]
    backend.old_status = "CancelRequested"
    advance(row, tmp_path, backend, state)
    assert backend.actions == ["audit", "cancel"] and not state.get("authorized")
    backend.finished_auxiliary = True
    with pytest.raises(ValueError, match="auxiliary stage completed"):
        advance(row, tmp_path, backend, state)
    assert backend.actions == ["audit", "cancel"] and not state.get("authorized")
    backend.finished_auxiliary = False
    backend.invalid = True
    backend.old_status = "Canceled"
    with pytest.raises(ValueError, match="identity"):
        advance(row, tmp_path, backend, state)
    assert backend.actions == ["audit", "cancel"] and not state.get("authorized")
    backend.invalid = False
    backend.old_status = "Canceled"
    advance(row, tmp_path, backend, state)
    assert backend.actions == ["audit", "cancel", "authorize"] and state["authorized"]
    backend.text = "5001/7866 5002/7866"
    advance(row, tmp_path, backend, state)
    assert state["continued"] and state["first_observed_step"] == 5001
    backend.new_status = "Failed"
    with pytest.raises(ValueError, match="terminal"):
        advance(row, tmp_path, backend, {})
    assert backend.actions == ["audit", "cancel", "authorize"]


def test_bootstrap_archives_only_interrupted_smoke(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("accel_bootstrap", _source_workspace('eval_interest_id_minionerec') /
                                                "cir_factorization/accel_launch/bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "smoke").mkdir(parents=True)
    (output / "smoke").mkdir(parents=True)
    (source / "EXECUTION_CONTRACT.json").write_text(json.dumps({
        "factor_arm": "relation_true", "fresh_formal_after_smoke": True}))
    (source / "smoke/weights").write_bytes(b"complete")
    (output / "smoke/weights").write_bytes(b"part")
    with pytest.raises(ValueError, match="identity"):
        bootstrap.archive_interrupted_smoke(output, source, "relation_pseudo")
    (output / "HANDOFF_AUTHORIZATION.json").write_text("{}")
    with pytest.raises(ValueError, match="non-smoke"):
        bootstrap.archive_interrupted_smoke(output, source, "relation_true")
    (output / "HANDOFF_AUTHORIZATION.json").unlink()
    (output / "smoke/unexpected").write_text("unknown")
    with pytest.raises(ValueError, match="unexpected"):
        bootstrap.archive_interrupted_smoke(output, source, "relation_true")
    (output / "smoke/unexpected").unlink()
    archive = bootstrap.archive_interrupted_smoke(output, source, "relation_true")
    assert (archive / "smoke/weights").read_bytes() == b"part"
    assert (source / "smoke/weights").read_bytes() == b"complete"
    assert not (output / "smoke").exists()
    assert bootstrap.archive_interrupted_smoke(output, source, "relation_true") is None
    (output / "PERFORMANCE_CONTRACT.json").write_text("{}")
    (output / "formal").mkdir()
    assert bootstrap.archive_interrupted_smoke(output, source, "relation_true") is None


def test_metadata_writer_avoids_reused_temporary_paths(tmp_path, monkeypatch):
    import importlib.util
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    monkeypatch.delenv("RANK", raising=False)
    spec = importlib.util.spec_from_file_location("metadata_startup", _source_workspace('eval_interest_id_minionerec') /
                                                "cir_factorization/accel_launch/sitecustomize.py")
    startup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(startup)
    target = tmp_path / "stage/semantic_token_setup.json"
    original = Path.write_text
    temporary_paths = []

    def record_path(path, *args, **kwargs):
        if path.name == "semantic_token_setup.json.tmp":
            raise FileNotFoundError("cached temporary path")
        temporary_paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_path)
    value = {"unchanged_metadata": [1, 2, 3]}
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda unused: startup.write_json(target, value), range(32)))
    assert len(set(temporary_paths)) == 32
    assert target.read_text() == json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    assert list(target.parent.iterdir()) == [target]


def test_metadata_continuation_requires_all_ranks_and_positive_step():
    import json
    from crossinterest.methods.factorization.audit import audit_metadata_continuation

    runtime = {"world_size": 8, "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4, "effective_batch_size": 32,
        "gradient_checkpointing": False, "ddp_find_unused_parameters": False,
        "qwen_fast_path": {"available": True}}
    events = [json.dumps({"event": "unique_metadata_temporary_files_enabled", "rank": rank})
              for rank in range(8)]
    assert audit_metadata_continuation("".join(events) + "7500/7866", runtime, 7500) is None
    text = "".join(events) + "\r7501/7866\r7509/7866"
    result = audit_metadata_continuation(text, runtime, 7500)
    assert result["passed"] and result["writer_ranks"] == list(range(8))
    assert result["first_observed_positive_step"] == 7501 and result["latest_step"] == 7509
    with pytest.raises(ValueError, match="rank coverage"):
        audit_metadata_continuation("".join(events[:-1]) + "7501/7866", runtime, 7500)
    with pytest.raises(ValueError):
        audit_metadata_continuation(text, dict(runtime, effective_batch_size=64), 7500)


def test_replaced_job_termination_uses_sealed_evidence_only_for_removed_run():
    from datetime import datetime
    from types import SimpleNamespace
    from azure.core.exceptions import ResourceNotFoundError
    from crossinterest.methods.factorization.audit import audit_replaced_job_termination

    row = {"old_job_id": "original", "new_job_id": "current", "previous_job_id": "replaced",
        "previous_job_status": "Canceled", "previous_job_end_utc": "2026-09-19T21:08:00+00:00"}
    observed = "2026-09-19T21:10:00+00:00"

    def removed(identifier):
        assert identifier == "replaced"
        raise ResourceNotFoundError("Run not found")

    result = audit_replaced_job_termination(row, observed, removed)
    assert result["passed"] and result["evidence_source"] == "sealed_pre_retry_termination_after_run_removal"
    details = SimpleNamespace(status="Canceled", end_time_utc=datetime.fromisoformat(row["previous_job_end_utc"]))
    assert audit_replaced_job_termination(row, observed, lambda identifier: details)["evidence_source"] == "live_run_history"
    for field, value in (("previous_job_status", "Running"), ("previous_job_id", "current"),
                         ("previous_job_end_utc", "2026-09-19T21:11:00+00:00")):
        with pytest.raises(ValueError, match="invalid sealed"):
            audit_replaced_job_termination(dict(row, **{field: value}), observed, removed)
    details.status = "Running"
    with pytest.raises(ValueError, match="differs"):
        audit_replaced_job_termination(row, observed, lambda identifier: details)

    def unavailable(identifier):
        raise ConnectionError("network unavailable")

    with pytest.raises(ConnectionError):
        audit_replaced_job_termination(row, observed, unavailable)