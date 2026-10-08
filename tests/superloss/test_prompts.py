import unittest

from prompts import SYSTEM_PROMPT, render_messages


class PromptContractTest(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"title": "Title A", "subcategory": "Topic A"},
            {"title": "Title B", "subcategory": "Topic B"},
        ]
        self.candidates = ["Gold"] + [f"Negative {index}" for index in range(1, 10)]

    def test_behavior_and_interest_expose_only_their_fields(self):
        behavior = render_messages(self.history, "behavior", target="Gold")
        interest = render_messages(self.history, "interest", target="Gold")
        self.assertEqual(
            SYSTEM_PROMPT,
            "Predict the user's next novel news topic from their chronological reading "
            "history. A novel topic is a subcategory that does not appear in the "
            'displayed history. Reply with exactly "New topic: <subcategory>" and no '
            "other text.",
        )
        self.assertEqual(behavior[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(interest[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(
            behavior[1]["content"],
            "The titles of news articles the user recently read, in chronological "
            "order, are listed below:\n\n"
            "1. Title A\n2. Title B\n\n"
            "Predict the one topic subcategory the user is most likely to read next. "
            "It must represent a new topic not reflected in the article-title history "
            "above.\n\n"
            'Reply with exactly "New topic: <subcategory>" and no other text.',
        )
        self.assertEqual(
            interest[1]["content"],
            "The topic subcategories associated with news articles the user recently "
            "read, in chronological order, are listed below:\n\n"
            "1. Topic A\n2. Topic B\n\n"
            "Predict the one topic subcategory the user is most likely to read next. "
            "It must not appear in the history sequence above.\n\n"
            'Reply with exactly "New topic: <subcategory>" and no other text.',
        )
        self.assertIn("Title A", behavior[1]["content"])
        self.assertNotIn("Topic A", behavior[1]["content"])
        self.assertIn("Topic A", interest[1]["content"])
        self.assertNotIn("Title A", interest[1]["content"])
        self.assertEqual(behavior[-1]["content"], "New topic: Gold")
        self.assertEqual(interest[-1]["content"], "New topic: Gold")

    def test_candidates_are_visible_only_in_choice(self):
        for view in ("behavior", "interest"):
            training = render_messages(self.history, view, target="Gold")
            ranking = render_messages(self.history, view)
            choice = render_messages(self.history, view, candidates=self.candidates)
            self.assertNotIn("Candidate topic subcategories:", training[1]["content"])
            self.assertNotIn("Candidate topic subcategories:", ranking[1]["content"])
            self.assertIn("Candidate topic subcategories:", choice[1]["content"])
            for candidate in self.candidates:
                self.assertIn(candidate, choice[1]["content"])

    def test_choice_requires_exactly_ten_unique_candidates(self):
        with self.assertRaisesRegex(ValueError, "exactly 10 unique"):
            render_messages(self.history, "behavior", candidates=["a", "b"])
        with self.assertRaisesRegex(ValueError, "exactly 10 unique"):
            render_messages(self.history, "interest", candidates=["same"] * 10)


if __name__ == "__main__":
    unittest.main()
