from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crossinterest.baselines.mera_strict.evaluate_official import official_formula_metrics, parse_category_map, parse_official_rows, validate_rankings


class EvaluateOfficialTest(unittest.TestCase):
    def test_official_formula(self) -> None:
        metrics = official_formula_metrics(
            [["a", "b", "c"], ["b", "c", "a"]],
            [["a", "c"], ["a"]],
            cutoffs=(1, 2),
        )
        self.assertEqual(metrics["precision_at_1"], 0.5)
        self.assertEqual(metrics["precision_at_2"], 0.25)
        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_2"], 0.25)

    def test_official_data_and_category_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            category_map = root / "layer2_map.txt"
            category_map.write_text("Arts\t0\nCrafts\t1\n", encoding="utf-8")
            test_file = root / "test.json"
            test_file.write_text(
                json.dumps(
                    [
                        {
                            "instruction": "predict",
                            "input": (
                                "The user has given high ratings to the following "
                                "categories: ['Arts']"
                            ),
                            "output": '"Crafts"',
                        }
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(parse_category_map(category_map), ["Arts", "Crafts"])
            self.assertEqual(parse_official_rows(test_file)[0]["targets"], ["Crafts"])

    def test_full_ranking_validation(self) -> None:
        labels = ["Arts", "Crafts"]
        rankings = validate_rankings(
            [{"predict": ['"Crafts"', '"Arts"']}],
            labels,
        )
        self.assertEqual(rankings, [["Crafts", "Arts"]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rankings([{"predict": ["Arts", "Arts"]}], labels)


if __name__ == "__main__":
    unittest.main()