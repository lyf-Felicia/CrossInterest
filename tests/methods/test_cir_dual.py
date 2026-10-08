from types import SimpleNamespace

import pytest
import torch

from crossinterest.methods.consistency.loss import VIEW_ORDER, dual_loss
from crossinterest.methods.consistency.train import DualCollator, DualTrainer, dual_messages


def test_ce_ablation_and_pair_accumulation():
    torch.manual_seed(42)
    logits = torch.randn(8, 4, 112, requires_grad=True)
    targets = torch.tensor([[index, index + 12] for index in range(8)])
    loss, _ = dual_loss(logits, targets, "ce")
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 112), targets[:, (0, 1, 0, 1)].reshape(-1))
    torch.testing.assert_close(loss, expected)
    whole, _ = dual_loss(logits, targets, "consistent")
    whole_gradient = torch.autograd.grad(whole, logits, retain_graph=True)[0]
    accumulated = sum(dual_loss(logits[index:index + 1], targets[index:index + 1], "consistent")[0] / 8
                      for index in range(8))
    torch.testing.assert_close(whole, accumulated)
    torch.testing.assert_close(whole_gradient, torch.autograd.grad(accumulated, logits)[0])
    assert (whole_gradient.abs().sum(dim=(0, 2)) > 0).all()


def test_independent_solution_still_requires_supervision():
    loss, diagnostic = dual_loss(torch.zeros(2, 4, 112), torch.tensor([[0, 1], [2, 3]]), "consistent")
    assert diagnostic["consistency"].eq(0).all()
    torch.testing.assert_close(loss, torch.tensor(112.0).log())


@pytest.mark.parametrize("logits", [torch.zeros(2, 3, 112), torch.full((2, 4, 112), float("nan"))])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError):
        dual_loss(logits, torch.tensor([[0, 1], [2, 3]]), "consistent")


def test_category_evidence_and_blind_targets():
    mapping = {"category_to_id": {"AA": "<interest_000>", "BB": "<interest_001>"}}
    row = {"historical_titles": ["neutral title"], "historical_interests": ["CC"],
           "target_interest": "AA", "future_interest": "BB"}
    for view in VIEW_ORDER:
        messages = dual_messages(row, mapping, view)
        target = "AA" if view.startswith("a_") else "BB"
        assert target not in messages[0]["content"]
        assert "<interest_000>" not in messages[0]["content"]
        assert "<interest_001>" not in messages[0]["content"]
        assert messages[1]["content"] == mapping["category_to_id"][target]
    first, second = dict(row), dict(row, future_interest="DD")
    extended = {"category_to_id": {**mapping["category_to_id"], "DD": "<interest_003>"}}
    assert dual_messages(first, extended, "b_given_r")[0] == dual_messages(second, extended, "b_given_r")[0]
    assert dual_messages(first, extended, "b_given_ra")[0] == dual_messages(second, extended, "b_given_ra")[0]


def test_collation_and_trainer_loss():
    features = [{"views": [{"input_ids": [2] * size, "labels": [-100] * (size - 1) + [3],
                            "attention_mask": [1] * size} for size in (3, 4, 5, 6)],
                 "dual_targets": [0, 1]} for _ in range(2)]
    batch = DualCollator(0)(features)
    assert batch["input_ids"].shape == (8, 6)
    assert batch["dual_targets"].shape == (2, 2)
    trainer = object.__new__(DualTrainer)
    trainer.label_token_ids = [0, 1, 2]
    trainer.dual_arm = "consistent"
    trainer.dual_totals = None
    trainer.dual_pairs = 0
    logits = torch.randn(8, 6, 3, requires_grad=True)

    def forward(**inputs):
        assert "labels" not in inputs and "dual_targets" not in inputs
        return SimpleNamespace(logits=logits)

    loss = trainer.compute_loss(forward, batch)
    positions = (batch["labels"] != -100).long().argmax(dim=-1) - 1
    expected, _ = dual_loss(logits[torch.arange(8), positions].reshape(2, 4, 3),
                            batch["dual_targets"], "consistent")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and trainer.dual_pairs == 2


def test_io_monitor_requires_all_ranks_and_completed_save():
    import json
    from crossinterest.methods.consistency.monitor import io_observation

    events = []
    for rank in range(8):
        events.extend([{"event": "dual_local_checkpoint_serialization_enabled", "rank": str(rank)},
                       {"event": "dual_complete_checkpoint_selected", "rank": rank, "passed": True,
                        "world_size": 8, "optimizer_steps": [200], "scheduler_step": 200,
                        "step": 200, "source": "/mnt/output/formal/novel/checkpoint-200"}])
    previous = io_observation(None, "".join(json.dumps(event) for event in events))
    assert not previous["verified"]
    saved = [{"event": "local_checkpoint_archive_saved", "rank": str(rank), "checkpoint": "checkpoint-300",
              "file": name, "serialize_seconds": 1, "copy_seconds": 4}
             for rank, name in [(0, "optimizer.pt"), (0, "scheduler.pt"),
                                *[(rank, f"rng_state_{rank}.pth") for rank in range(8)]]]
    text = "".join(json.dumps(event) for event in saved)
    assert not io_observation(previous, text + "300/3500 [")['verified']
    result = io_observation(previous, text + "301/3500 [")
    assert result["verified"] and result["optimizer_save_seconds"] == 5
    assert result["optimizer_save_under_120_seconds"]
    assert not io_observation(previous, "".join(json.dumps(event) for event in saved[:-1]) + "301/3500 [")["verified"]
    saved[0]["copy_seconds"] = 400
    assert not io_observation(previous, "".join(json.dumps(event) for event in saved) + "301/3500 [")["optimizer_save_under_120_seconds"]


def test_checkpoint_timing_distinguishes_isolated_and_repeated_stalls():
    from crossinterest.methods.consistency.monitor import checkpoint_timing

    text = "1500/3500 [1:00:22<later]\r1501/3500 [1:06:53<later]\r1600/3500 [1:10:37<later]\r1601/3500 [1:11:08<later]"
    current = checkpoint_timing(None, text)
    assert current["last_save_boundaries"] == [{"step": 1500, "seconds": 391}, {"step": 1600, "seconds": 31}]
    assert not current["repeated_slow_saves"]
    updated = checkpoint_timing(current, "1700/3500 [1:15:00<later]\r1701/3500 [1:21:00<later]")
    assert updated["repeated_slow_saves"]
    assert updated["recent_seconds_per_step_including_saves"] > 0
    fast = checkpoint_timing(None, "201/3500 [02:11<later]\r300/3500 [06:09<later]\r301/3500 [06:31<later]")
    assert fast["last_save_boundaries"] == [{"step": 300, "seconds": 22}]
    assert fast["recent_seconds_per_step_including_saves"] == 2.6


def test_monitor_retry_does_not_reuse_previous_runtime_evidence():
    import json
    from crossinterest.methods.consistency.monitor import attempt_previous, checkpoint_timing, io_observation

    previous = {"log_file": "user_logs/std_log.txt",
                "checkpoint_timing": checkpoint_timing(None, "1200/3500 [44:56<later]1228/3500 [46:20<later]"),
                "io": {"verified": True, "restored_ranks": {
                    str(rank): {"path": "/tmp/old/checkpoint-200"} for rank in range(8)}}}
    assert attempt_previous(previous, previous["log_file"], "loading") is previous
    assert attempt_previous(previous, "user_logs/retry_001/std_log.txt", "loading") == {}
    event = {"event": "dual_complete_checkpoint_selected", "rank": 0, "passed": True,
             "world_size": 8, "optimizer_steps": [1200], "scheduler_step": 1200,
             "step": 1200, "source": "/mnt/output/formal/novel/checkpoint-1200",
             "path": "/tmp/new/checkpoint-1200"}
    text = json.dumps(event) + "1201/3500 [02:15<later]"
    fresh = attempt_previous(previous, previous["log_file"], text)
    assert fresh == {}
    timing = checkpoint_timing(fresh.get("checkpoint_timing"), text)
    assert timing["latest_step"] == 1201 and timing["last_save_boundaries"] == []
    runtime = io_observation(fresh.get("io"), text)
    assert not runtime["verified"] and set(runtime["restored_ranks"]) == {"0"}
    retried = {"log_file": previous["log_file"], "io": runtime}
    assert attempt_previous(retried, retried["log_file"], text) is retried