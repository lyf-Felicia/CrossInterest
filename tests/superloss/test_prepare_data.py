import unittest
from collections import Counter

from prepare_data import fixed_support_sample, label_of


def row(label: str, index: int) -> dict:
    return {
        "sample_id": f"{label}:{index}",
        "target_interest": {"subcategory": label},
    }


class BalancedSampleTest(unittest.TestCase):
    def test_drops_rare_labels_and_downsamples_retained_labels(self):
        rows = [row("rare", 0)] + [row("common", index) for index in range(3)]
        sampled, report = fixed_support_sample(rows, samples_per_label=2, seed=7)

        self.assertEqual(Counter(label_of(item) for item in sampled), {"common": 2})
        self.assertEqual(len({item["sample_id"] for item in sampled}), 2)
        self.assertEqual(report["dropped_labels"], {"rare": 1})
        self.assertEqual(report["downsampled_rows"], 1)
        self.assertEqual(report["repeated_rows"], 0)

    def test_requires_positive_minimum(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            fixed_support_sample([row("a", 0)], samples_per_label=0, seed=11)

    def test_is_deterministic(self):
        rows = [row("a", index) for index in range(4)] + [
            row("b", index) for index in range(3)
        ]
        first, first_report = fixed_support_sample(rows, samples_per_label=3, seed=19)
        second, second_report = fixed_support_sample(
            list(reversed(rows)), samples_per_label=3, seed=19
        )

        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)


if __name__ == "__main__":
    unittest.main()