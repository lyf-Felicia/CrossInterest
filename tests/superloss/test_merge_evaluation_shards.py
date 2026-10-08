import unittest

from merge_evaluation_shards import (
    merge_shard_rows,
    validate_contracts,
)


class MergeEvaluationShardsTest(unittest.TestCase):
    def test_merge_restores_original_dev_order(self):
        expected_ids = [f"sample-{index}" for index in range(17)]
        shards = [
            [{"sample_id": sample_id} for sample_id in expected_ids[index::6]]
            for index in range(6)
        ]
        self.assertEqual(
            [row["sample_id"] for row in merge_shard_rows(shards, expected_ids)],
            expected_ids,
        )

    def test_merge_rejects_missing_or_misordered_rows(self):
        expected_ids = ["a", "b", "c", "d"]
        with self.assertRaises(ValueError):
            merge_shard_rows(
                [[{"sample_id": "c"}, {"sample_id": "a"}], [{"sample_id": "b"}]],
                expected_ids,
            )

    def test_contracts_must_differ_only_by_shard_index(self):
        contracts = [
            {"task": "ranking", "view": "behavior", "shard_count": 2, "shard_index": 0},
            {"task": "ranking", "view": "behavior", "shard_count": 2, "shard_index": 1},
        ]
        reference = validate_contracts(contracts, 2)
        self.assertEqual(reference["task"], "ranking")
        contracts[1]["view"] = "interest"
        with self.assertRaises(ValueError):
            validate_contracts(contracts, 2)


if __name__ == "__main__":
    unittest.main()