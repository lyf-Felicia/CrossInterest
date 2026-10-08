import math
import unittest

import torch

from train import AssistantOnlyCollator
from superloss import (
    answer_logits_to_keep,
    principal_lambert_w,
    sequence_answer_loss,
    superloss_confidence,
)


class SuperLossTest(unittest.TestCase):
    def test_collator_computes_answer_suffix_without_device_sync(self):
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [-100, -100, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            },
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [-100, -100, -100, 4],
                "attention_mask": [1, 1, 1, 1],
            },
        ]
        batch = AssistantOnlyCollator(pad_token_id=0)(features)
        self.assertEqual(batch["logits_to_keep"], 4)

    def test_answer_logits_to_keep_includes_preceding_prediction(self):
        labels = torch.tensor(
            [
                [-100, -100, -100, 4, 5, 6],
                [-100, -100, 7, 8, 9, -100],
            ]
        )
        self.assertEqual(answer_logits_to_keep(labels), 5)

    def test_sequence_loss_averages_only_answer_tokens(self):
        logits = torch.tensor(
            [
                [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0], [8.0, 0.0, 0.0]],
                [[0.0, 8.0, 0.0], [0.0, 0.0, 8.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]],
            ]
        )
        labels = torch.tensor([[-100, 1, 2, -100], [-100, 2, 0, 1]])
        losses = sequence_answer_loss(logits, labels)
        manual = torch.stack(
            [
                torch.nn.functional.cross_entropy(
                    logits[0, :-1], labels[0, 1:], ignore_index=-100
                ),
                torch.nn.functional.cross_entropy(
                    logits[1, :-1], labels[1, 1:], ignore_index=-100
                ),
            ]
        )
        torch.testing.assert_close(losses, manual)

    def test_suffix_logits_match_full_sequence_loss(self):
        torch.manual_seed(17)
        logits = torch.randn(2, 8, 19)
        labels = torch.tensor(
            [
                [-100, -100, -100, -100, 3, 4, 5, 6],
                [-100, -100, -100, 7, 8, 9, 10, 11],
            ]
        )
        keep = answer_logits_to_keep(labels)
        torch.testing.assert_close(
            sequence_answer_loss(logits[:, -keep:], labels),
            sequence_answer_loss(logits, labels),
        )

    def test_lambert_w_satisfies_definition(self):
        values = torch.tensor([-1 / math.e, -0.2, 0.0, 0.5, 10.0], dtype=torch.float64)
        result = principal_lambert_w(values)
        torch.testing.assert_close(result * result.exp(), values, atol=1e-12, rtol=1e-12)

    def test_confidence_is_monotone_in_loss(self):
        losses = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        weights = superloss_confidence(losses, threshold=1.0, regularization=1.0)
        self.assertTrue(torch.all(weights[:-1] >= weights[1:]))
        self.assertTrue(torch.all(weights >= 0.1))
        self.assertLessEqual(float(weights.max()), math.e + 1e-12)

    def test_confidence_is_detached(self):
        losses = torch.tensor([0.5, 2.0], requires_grad=True)
        weights = superloss_confidence(losses, threshold=1.0, regularization=1.0)
        self.assertFalse(weights.requires_grad)

    def test_zero_minimum_preserves_paper_weight(self):
        loss = torch.tensor([100.0], dtype=torch.float64)
        weight = superloss_confidence(loss, threshold=1.0, regularization=1.0, minimum=0.0)
        self.assertGreater(float(weight), 0.0)
        self.assertLess(float(weight), 0.1)


if __name__ == "__main__":
    unittest.main()