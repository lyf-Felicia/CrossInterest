import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train import CurriculumTrainer, find_complete_checkpoint, truncate_metrics


class CheckpointTest(unittest.TestCase):
    def test_only_complete_latest_checkpoint_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (25, 50):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text("{}")
                (checkpoint / "curriculum_state_rank0.json").write_text("{}")
            (root / "checkpoint-25" / "curriculum_state_rank1.json").write_text("{}")

            self.assertEqual(find_complete_checkpoint(root, 2), root / "checkpoint-25")

    def test_metrics_are_truncated_at_resume_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("".join(json.dumps({"global_step": step}) + "\n" for step in range(7)))
            truncate_metrics(path, 5)

            steps = [json.loads(line)["global_step"] for line in path.read_text().splitlines()]
            self.assertEqual(steps, [0, 1, 2, 3, 4])

    def test_curriculum_state_round_trip_and_contract(self):
        trainer = object.__new__(CurriculumTrainer)
        trainer.method = "superloss"
        trainer.view = "behavior"
        trainer.threshold_ema = 0.7
        trainer.weight_ema = 1.2
        trainer.pending_losses = [0.5]
        trainer.pending_weights = [0.9]
        trainer.observed_global_step = 24
        trainer.last_logged_step = 24
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}):
            state = trainer.curriculum_state()
            restored = object.__new__(CurriculumTrainer)
            restored.method = "superloss"
            restored.view = "behavior"
            restored.restore_curriculum_state(state)
            self.assertEqual(restored.curriculum_state(), state)

            wrong_world_size = {**state, "world_size": 4}
            with self.assertRaisesRegex(ValueError, "world_size"):
                restored.restore_curriculum_state(wrong_world_size)
            wrong_view = {**state, "view": "interest"}
            with self.assertRaisesRegex(ValueError, "view"):
                restored.restore_curriculum_state(wrong_view)


if __name__ == "__main__":
    unittest.main()