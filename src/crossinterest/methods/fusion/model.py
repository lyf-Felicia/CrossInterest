"""Fixed first-token collaborative/LM probability mixture for full-vocabulary beams."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor


class HistoryLinearModel:
    def __init__(self, path: Path, categories: list[str]):
        with np.load(path, allow_pickle=False) as arrays:
            self.coefficients = arrays["coefficients"].copy()
            self.intercept = arrays["intercept"].copy()
        self.indices = {category: index for index, category in enumerate(categories)}
        count = len(categories)
        if (len(self.indices) != count or self.coefficients.shape != (count, count + 1)
                or self.intercept.shape != (count,)
                or not np.isfinite(self.coefficients).all()
                or not np.isfinite(self.intercept).all()):
            raise ValueError("Invalid fixed linear model or category order")

    def log_probabilities(self, histories: list[list[str]]) -> torch.Tensor:
        features = np.zeros((len(histories), len(self.indices) + 1), dtype=np.float64)
        for row_index, history in enumerate(histories):
            if not history:
                raise ValueError("History must be nonempty")
            features[row_index, [self.indices.get(category, len(self.indices)) for category in history]] = 1
        logits = features @ self.coefficients.T + self.intercept
        return torch.from_numpy(logits).log_softmax(dim=-1)


class FirstTokenProbabilityFusion(LogitsProcessor):
    def __init__(self, token_ids: list[int], log_probabilities: torch.Tensor,
                 prompt_length: int, num_beams: int):
        if (not token_ids or len(set(token_ids)) != len(token_ids) or min(token_ids) < 0
                or prompt_length < 1 or num_beams < 1
                or log_probabilities.ndim != 2 or not len(log_probabilities)
                or log_probabilities.shape[1] != len(token_ids)
                or not torch.isfinite(log_probabilities).all()
                or not torch.allclose(log_probabilities.double().logsumexp(-1),
                                      torch.zeros(len(log_probabilities), dtype=torch.float64,
                                                  device=log_probabilities.device), atol=1e-6, rtol=0)):
            raise ValueError("Invalid first-token probability fusion contract")
        self.token_ids = token_ids
        self.log_probabilities = log_probabilities.detach()
        self.prompt_length = prompt_length
        self.num_beams = num_beams
        self.applied_calls = 0
        self.later_calls = 0
        self.max_mass_error = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("Generation prefix is shorter than the recorded prompt")
        if input_ids.shape[1] != self.prompt_length:
            self.later_calls += 1
            return scores
        if (scores.ndim != 2 or scores.shape[0] != len(self.log_probabilities) * self.num_beams
                or input_ids.shape[0] != scores.shape[0] or max(self.token_ids) >= scores.shape[1]
                or scores.dtype not in (torch.float32, torch.float64) or self.applied_calls):
            raise ValueError("Unexpected first-token beam layout or score precision")
        token_ids = torch.tensor(self.token_ids, device=scores.device)
        original = scores.index_select(-1, token_ids)
        if not torch.isfinite(original).all():
            raise ValueError("ID scores must not have been masked before fusion")
        log_probabilities = self.log_probabilities.to(scores).repeat_interleave(self.num_beams, dim=0)
        log_mass = original.logsumexp(-1, keepdim=True)
        mixed = torch.logaddexp(original, log_mass + log_probabilities) - math.log(2)
        error = float((mixed.logsumexp(-1, keepdim=True) - log_mass).abs().max())
        if error > 1e-5:
            raise ValueError("ID probability mass changed unexpectedly")
        output = scores.clone()
        output[:, token_ids] = mixed
        self.applied_calls += 1
        self.max_mass_error = error
        return output


def bind_ranking_fusion(evaluator, linear_model, special_tokens, record_check):
    original_render = evaluator.render_prompt_ids
    original_rank = evaluator.direct_beam_rank
    current = {}

    def render(tokenizer, row, *args, **kwargs):
        prompt_ids = original_render(tokenizer, row, *args, **kwargs)
        current.clear()
        current.update(sample_id=row["sample_id"], prompt_ids=prompt_ids,
                       log_probabilities=linear_model.log_probabilities([row["historical_interests"]]))
        return prompt_ids

    def rank(model, tokenizer, prompt_ids, output_to_category, num_beams, max_new_tokens,
             semantic_output_sequences=None):
        from transformers import LogitsProcessorList

        if current.get("prompt_ids") != prompt_ids:
            raise ValueError("History and generation prompt are not aligned")
        processor = FirstTokenProbabilityFusion(
            tokenizer.convert_tokens_to_ids(special_tokens), current["log_probabilities"],
            len(prompt_ids), num_beams)
        original_generate = model.generate

        def generate(*args, **kwargs):
            if kwargs.get("logits_processor") or kwargs.get("prefix_allowed_tokens_fn"):
                raise ValueError("Unexpected candidate constraint or additional score processor")
            kwargs["logits_processor"] = LogitsProcessorList([processor])
            return original_generate(*args, **kwargs)

        model.generate = generate
        try:
            result = original_rank(model, tokenizer, prompt_ids, output_to_category,
                                   num_beams, max_new_tokens, semantic_output_sequences)
        finally:
            model.generate = original_generate
        if processor.applied_calls != 1:
            raise ValueError("First-token fusion was not executed exactly once")
        record_check({"sample_id": current["sample_id"], "applied_calls": processor.applied_calls,
                      "later_calls": processor.later_calls, "max_mass_error": processor.max_mass_error,
                      "prompt_tokens": len(prompt_ids), "num_beams": num_beams})
        current.clear()
        return result

    evaluator.render_prompt_ids = render
    evaluator.direct_beam_rank = rank