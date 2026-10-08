# Amazon 112-Label No-Training Evaluation

This directory provides the strict base-model control for the category-name classification results in `crossinterest.training`.

## Scope

- Models: the same eight models/configurations as the earlier no-training table: GPT-5.6 Sol (medium/none), GPT-5 (medium/none), Gemini 3.6 Flash (medium/minimal), Qwen3.5-9B, and Llama-3.1-8B-Instruct.
- Data: the exact three frozen Dev-2K parquet files used by the 112k SFT experiment.
- Prompt: category-name schema v3 with 10 natural-language candidates.
- Decoding: deterministic candidate-constrained 10-beam search for local HF models; the frozen direct API protocol from the earlier no-training table for API models.
- Reported metric: Macro Accuracy only.
- Excluded: Interest-ID output and candidate-free ranking.

The runner verifies all three dataset SHA-256 values before loading a model. The evaluator directly reuses `eval_amazon_sft_112.evaluate.render_prompt_ids` and `predict`, including the same `AutoModelForCausalLM` base loader, prompt assertion, beam trie, and scoring utilities. This fixes samples, candidate sets, candidate order, gold positions, prompt text, model loading, decoding, and scoring across the base and category-SFT evaluations. The only intended experimental difference is whether the SFT adapter is loaded.

## Data Contract

| View | Rows | Labels | SHA-256 |
|---|---:|---:|---|
| Behavior | 2,000 | 112 | `890f3ef105e2b85da6adf4e3c283a34f60b4636b8fe29ca5e491e29e5ac7e502` |
| Interest | 2,000 | 112 | `766eeeb971f3eb82b940084581a9f07a6cc814645a2d734d5b074e01c994e5cb` |
| Behavior + Interest | 2,000 | 112 | `aebd86d67243022c2dd76a77dccf30146124d54b437baf72788ac1ca98192cb3` |

The files remain owned by `crossinterest.training/data/eval`; they are referenced in place rather than duplicated.

## Run

The AMLT config contains six independent `80G1-A100`, Standard-SLA jobs:

```bash
amlt run crossinterest.experiments.no_training/amlt/amlt_eval.yaml <experiment-name> \
  --description "Strict base-model Macro Accuracy on the frozen 112-label SFT Dev-2K"
```

Each job writes resumable details and its final metric under `classification/`. A result may enter `RESULTS.md` only when it has 2,000 recorded samples, zero missing samples, 112 classes, and `complete=true`.

Run each API model locally with its frozen registry settings:

```bash
MODEL_KEY=gpt56_sol_medium bash crossinterest.experiments.no_training/run_api_eval.sh
MODEL_KEY=gpt56_sol_none bash crossinterest.experiments.no_training/run_api_eval.sh
MODEL_KEY=gpt5_medium bash crossinterest.experiments.no_training/run_api_eval.sh
MODEL_KEY=gpt5_none bash crossinterest.experiments.no_training/run_api_eval.sh
MODEL_KEY=gemini36_flash bash crossinterest.experiments.no_training/run_api_eval.sh
MODEL_KEY=gemini36_flash_minimal bash crossinterest.experiments.no_training/run_api_eval.sh
```

The GPT deployments use TRAPI `redmond/interactive`. Both Gemini configurations always use the same Copilot reverse proxy as the earlier formal evaluation; Gemini is not routed through TRAPI. Start the proxy with:

```bash
npx --yes copilot-api@latest start --port 4141 --rate-limit 1 --wait
```

The proxy must expose `gemini-3.6-flash` at `127.0.0.1:4141` before either Gemini run starts.

## Interpretation

This control supports a direct base-versus-SFT comparison on the 112-label task. It removes the sample and candidate mismatch that affects comparisons with the older 99-label no-training experiment. It does not by itself separate the effects of adding 13 labels from reducing the training quota from 4,000 to 1,000 examples per label.