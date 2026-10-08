import math
import unittest
from argparse import Namespace

from evaluate import (
    candidate_sequences,
    macro_mean,
    ranking_metrics,
    render_prompt,
    shard_rows,
    summarize,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_shards_partition_rows_without_overlap(self):
        rows = [{"sample_id": str(index)} for index in range(17)]
        shards = [shard_rows(rows, 6, index) for index in range(6)]
        observed = [row["sample_id"] for shard in shards for row in shard]
        self.assertCountEqual(observed, [row["sample_id"] for row in rows])
        self.assertEqual(len(observed), len(set(observed)))
        recovered = sorted(
            (int(row["sample_id"]), row) for shard in shards for row in shard
        )
        self.assertEqual([row for _, row in recovered], rows)

    def test_shard_index_must_be_in_range(self):
        with self.assertRaises(ValueError):
            shard_rows([], 2, 2)

    def test_candidate_sequences_use_chat_template_through_eos(self):
        class MockTokenizer:
            eos_token_id = 99

            def apply_chat_template(
                self, messages, *, add_generation_prompt, **_kwargs
            ):
                prompt = [1, 2]
                if add_generation_prompt:
                    return {"input_ids": prompt}
                label_token = {"New topic: A": 11, "New topic: B": 12}[
                    messages[-1]["content"]
                ]
                return {"input_ids": prompt + [label_token, self.eos_token_id, 10]}

        tokenizer = MockTokenizer()
        messages = [{"role": "user", "content": "classify"}]
        prompt_ids = render_prompt(tokenizer, messages)
        sequences = candidate_sequences(tokenizer, messages, prompt_ids, ["A", "B"])
        self.assertEqual(sequences, {(11, 99): "A", (12, 99): "B"})

    def test_single_gold_recall_and_ndcg(self):
        ranked = [f"wrong-{index}" for index in range(11)]
        ranked.insert(9, "gold")
        metrics = ranking_metrics(ranked, "gold")
        self.assertEqual(metrics["recall@10"], 1.0)
        self.assertEqual(metrics["recall@20"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@10"], 1.0 / math.log2(11))
        self.assertAlmostEqual(metrics["ndcg@20"], 1.0 / math.log2(11))

    def test_macro_mean_weights_labels_equally(self):
        rows = [
            {"target_subcategory": "frequent", "correct": True},
            {"target_subcategory": "frequent", "correct": True},
            {"target_subcategory": "frequent", "correct": False},
            {"target_subcategory": "rare", "correct": False},
        ]
        macro, per_label = macro_mean(rows, "correct")
        self.assertAlmostEqual(macro, 1.0 / 3.0)
        self.assertEqual(per_label, {"frequent": 2.0 / 3.0, "rare": 0.0})

    def test_ranking_summary_contains_only_requested_cutoffs(self):
        rows = []
        for target, ranked in (("a", ["a", "b"]), ("a", ["b", "a"]), ("b", ["b", "a"])):
            row = {
                "target_subcategory": target,
                "predicted_label": ranked[0],
                "ranked_labels": ranked,
                "correct": ranked[0] == target,
            }
            row.update(ranking_metrics(ranked, target))
            rows.append(row)
        metrics = summarize(
            rows,
            ["a", "b"],
            Namespace(
                model="test",
                adapter=None,
                view="behavior",
                task="ranking",
                num_beams=2,
                return_count=2,
            ),
        )
        self.assertEqual(metrics["cutoffs"], [10, 20])
        self.assertAlmostEqual(metrics["macro_recall@10"], 1.0)
        self.assertNotIn("macro_recall@1", metrics)


if __name__ == "__main__":
    unittest.main()