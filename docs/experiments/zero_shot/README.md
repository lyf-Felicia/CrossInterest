# Amazon No-Training Evaluation

This directory owns the zero-shot Amazon 10-way evaluation. It is intentionally separate from `amazon/prep/`, which only owns dataset preprocessing.

Only the current validated prompt schema is retained:

- `data/`: paired 2,000-sample Behavior, Interest, and Behavior + Interest views using prompt schema v3.
- `runs/formal/`: resumable per-model details, predictions, contracts, and metrics.
- `logs/formal/`: logs for the formal runs.
- `RESULTS.md`: completed Macro Accuracy results. Empty cells are still running.

Run API models from the repository root:

```bash
python crossinterest_formal/amazon/crossinterest.evaluation/evaluate_api.py \
  --model-key gpt5_none \
  --input-type all \
  --run-name formal
```

Run local Hugging Face models with vLLM:

```bash
python crossinterest_formal/amazon/crossinterest.evaluation/evaluate_vllm.py \
  --model-key qwen35_9b \
  --input-type all \
  --run-name formal
```

Results are resumable by `sample_id` only when the existing `run_contract.json` exactly matches the current data and evaluation settings.