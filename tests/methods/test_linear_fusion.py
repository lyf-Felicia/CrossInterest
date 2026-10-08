"""Focused probability and actual generation checks for the fixed fusion method."""

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList

from crossinterest.methods.fusion.model import FirstTokenProbabilityFusion, HistoryLinearModel, bind_ranking_fusion


def test_full_vocabulary_mass_and_later_steps():
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(6, 19, generator=generator, dtype=torch.float64).log_softmax(-1)
    conditional = torch.randn(2, 3, generator=generator, dtype=torch.float64).log_softmax(-1)
    processor = FirstTokenProbabilityFusion([3, 7, 11], conditional, 4, 3)
    mixed = processor(torch.ones(6, 4, dtype=torch.long), scores)
    original_probability, mixed_probability = scores.exp(), mixed.exp()
    native = [index for index in range(19) if index not in processor.token_ids]
    torch.testing.assert_close(mixed[:, native], scores[:, native], rtol=0, atol=0)
    mass = original_probability[:, processor.token_ids].sum(-1, keepdim=True)
    expected = (original_probability[:, processor.token_ids] + mass * conditional.exp().repeat_interleave(3, 0)) / 2
    torch.testing.assert_close(mixed_probability[:, processor.token_ids], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mixed_probability.sum(-1), torch.ones(6, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    assert processor(torch.ones(6, 5, dtype=torch.long), scores) is scores
    assert processor.applied_calls == processor.later_calls == 1
    with pytest.raises(ValueError):
        processor(torch.ones(6, 4, dtype=torch.long), scores)


def test_history_features_and_saved_coefficients(tmp_path):
    path = tmp_path / "weights.npz"
    coefficients = np.asarray([[1, 0, -1], [-1, 2, 1]], dtype=float)
    intercept = np.asarray([0.25, -0.25])
    np.savez(path, coefficients=coefficients, intercept=intercept)
    model = HistoryLinearModel(path, ["first", "second"])
    actual = model.log_probabilities([["first", "first", "other"], ["second"]])
    expected = torch.tensor(np.asarray([[1, 0, 1], [0, 1, 0]]) @ coefficients.T + intercept).log_softmax(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.log_probabilities([[]])


def test_real_transformers_beam_expansion_and_first_position():
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_layer=1, n_head=2, n_embd=16,
                                     bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    conditional = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]).log()
    processor = FirstTokenProbabilityFusion([5, 13, 24], conditional, 3, 20)
    inputs = torch.tensor([[1, 4, 8], [1, 9, 10]])
    with torch.no_grad():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                num_beams=20, num_return_sequences=20, do_sample=False,
                                max_new_tokens=3, early_stopping=True, output_scores=True,
                                return_dict_in_generate=True,
                                logits_processor=LogitsProcessorList([processor]))
    assert output.sequences.shape[0] == 40
    assert processor.applied_calls == 1 and processor.later_calls >= 1
    assert torch.isfinite(output.sequences_scores).all()
    torch.testing.assert_close(output.scores[0].exp().sum(-1), torch.ones(40), atol=1e-6, rtol=0)


def test_evaluator_binding_uses_only_current_history(tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "model.npz"
    np.savez(path, coefficients=np.asarray([[1, -1, 0], [-1, 1, 0]]), intercept=np.zeros(2))
    linear = HistoryLinearModel(path, ["first", "second"])
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_layer=1, n_head=2, n_embd=16,
                                     bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
    original_generate = model.generate
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda tokens: [5, 13])

    def original_rank(model, tokenizer, prompt, outputs, beams, maximum, sequences):
        inputs = torch.tensor([prompt])
        with torch.no_grad():
            return model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  num_beams=beams, num_return_sequences=beams, do_sample=False,
                                  max_new_tokens=maximum, output_scores=True, return_dict_in_generate=True)

    evaluator = SimpleNamespace(render_prompt_ids=lambda tokenizer, row: [1, 4, 8], direct_beam_rank=original_rank)
    checks = []
    bind_ranking_fusion(evaluator, linear, ["id_first", "id_second"], checks.append)
    for category in ("first", "second"):
        prompt = evaluator.render_prompt_ids(tokenizer, {"sample_id": category, "historical_interests": [category]})
        output = evaluator.direct_beam_rank(model, tokenizer, prompt, {}, 20, 2)
        assert output.sequences.shape[0] == 20 and model.generate == original_generate
    assert [row["sample_id"] for row in checks] == ["first", "second"]
    assert all(row["applied_calls"] == 1 for row in checks)
    with pytest.raises(ValueError):
        evaluator.direct_beam_rank(model, tokenizer, prompt, {}, 20, 2)