# Amazon 99-Label SFT Evaluation

This directory owns the quota-4,000 Amazon SFT protocol with 99 target labels.
It contains two output-label modes over the same three natural-language input views.

## Protocol

- Models: `Qwen/Qwen3.5-9B` and `NousResearch/Meta-Llama-3.1-8B-Instruct`.
- Views: `behavior`, `interest`, and `behavior_interest`.
- Training data: all 396,000 rows from `../train.parquet` for every run.
- Category mode: the assistant outputs the exact natural-language target category.
- Interest-ID mode: the assistant outputs the frozen opaque ID `<interest_NNN>`.
- Inputs never replace item titles or historical category names with interest IDs.
- Training is candidate-free and assistant-only in both output modes.
- Classification uses the frozen paired Dev-2K candidates and 10-way constrained beam.
- Ranking uses candidate-free, unconstrained direct 20-beam generation.
- Metrics are Macro Accuracy and Macro Recall/nDCG@10/20.

The ID mapping, data hashes, and six representative training messages are frozen under
`data/`. IDs are ordinary tokenizer sequences rather than newly registered special tokens.

## Status

The category-output experiment is complete for both models and all three views. The
interest-ID experiment is fully specified in `amlt/amlt_interest_id_formal.yaml` but has
not been submitted. All formal results, including both output modes, belong in the single
table in [RESULTS.md](results.md).

See [EXPERIMENT_LOG.md](log.md) for the completed category run history and
known infrastructure issues.