# Amazon 112-Label SFT Evaluation

This directory owns the quota-1,000 Amazon SFT protocol with 112 target labels. It is built from the same cleaned, user-disjoint Amazon mother pool as the 99-label protocol but has its own balanced Train, natural closed-set Dev, frozen Dev-2K views, label mapping, AMLT configuration, and result table.

## Data

- Train: 112,000 rows, exactly 1,000 rows for each of 112 target categories.
- Dev: 596,449 natural closed-set rows, not resampled.
- Evaluation: 2,000 paired and stratified samples for each of `behavior`, `interest`, and `behavior_interest`.
- Build rule: retain every mother-pool Train target with support at least 1,000, then select exactly 1,000 records per target by the existing stable SHA-256 order.
- Data files and immutable hashes are recorded in `data/DATASET_MANIFEST.json` and `data/MANIFEST.json`.

Rebuild from the existing mother pool without rescanning raw Amazon metadata or ratings:

```bash
cd crossinterest_formal/amazon
python -m crossinterest.training.build_dataset
python -m crossinterest.training.prepare_experiment_data \
  --train crossinterest.training/data/train.parquet \
  --dev crossinterest.training/data/dev.parquet \
  --eval-dir crossinterest.training/data/eval \
  --expected-label-count 112 \
  --expected-quota 1000 \
  --expected-train-rows 112000
```

## Experiment Matrix

Both Qwen3.5-9B and Llama-3.1-8B-Instruct use all three views under three output modes:

- `category`: output the exact natural-language category name.
- `interest_id`: output one frozen semantic token `<interest_000>` through `<interest_111>`.
- `<archived-job-id>`: output the literal composite label `[<interest_NNN>] Parent Category > Child Category`.

The input history is identical between output modes. Product titles and historical interests remain natural-language text. Training is candidate-free and assistant-only. Classification uses natural-language candidates with 10-way output-label-constrained beam search; ranking uses candidate-free, unconstrained direct 20-beam generation. ID predictions are mapped back to categories before metrics are computed.

For `<archived-job-id>`, the same 112 semantic Interest-ID tokens and child-label mean embedding initialization are used as in `interest_id`. The category suffix is generated from the base tokenizer. Classification constructs the 10 complete `[<interest_NNN>] Category` candidate strings and constrains decoding to those strings. Ranking remains decoder-unconstrained; scoring accepts only an exact complete sequence whose Interest ID maps to the accompanying category. The prompt schema is `8-interest-id-category`.

The exact output instructions are:

```text
Training/ranking:
Respond with only the Interest ID followed by the exact category name in the format [<interest_NNN>] Parent Category > Child Category. Include the square brackets and one space after the closing bracket. Do not output any explanation.

10-way classification:
Respond with only the Interest ID assigned to the selected category followed by that exact category name, in the format [<interest_NNN>] Parent Category > Child Category. Include the square brackets and one space after the closing bracket. Do not output the candidate number or any explanation.
```

Real rendered training examples for every view and output mode are frozen in `data/TRAINING_SAMPLE_PREVIEW.json`.

`amlt/amlt_formal.yaml` declares the original 12 category/Interest-ID jobs. `amlt/amlt_interest_id_category_recall.yaml` adds the six composite-output jobs on Recall. Each job trains one adapter and then writes both classification and ranking metrics.

The interest-ID jobs were submitted first as `amazon-sft-112-interest-id-v2-20260827-133837`; a Qwen `interest` job passed the training health gate before the remaining five jobs were appended. The category-name jobs were then submitted as `amazon-sft-112-category-20260827-134729`. The six completed composite-output jobs come from `amazon-sft-112-id-category-recall-20260831-105059`. Earlier ID launch attempts are preflight failures only and are documented in `EXPERIMENT_LOG.md`; they must not be used as result sources.

All results belong in the one unified table in [RESULTS.md](results.md). The 99-label and 112-label metrics use different label spaces and Dev supports, so they must not be described as a paired pure-quota ablation.
