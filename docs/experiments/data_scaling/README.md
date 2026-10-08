# Amazon 112-Label Training-Data Saturation

This experiment measures the Behavior + Interest (`behavior_interest`) learning curve for category-name SFT on the frozen 112-label Amazon task.

## Controlled Variables

- Models: Qwen3.5-9B and Llama-3.1-8B-Instruct.
- Train sizes: 5K, 10K, 20K, 40K, 60K, 80K, 100K, and all 112K rows.
- Output: exact natural-language category name.
- Evaluation: the unchanged frozen 2,000-row B+I Dev set from `crossinterest.training/data/eval`.
- Optimization: one epoch, effective batch size 32, and the same LoRA/prompt/evaluation code as `crossinterest.training`.

The label space is fixed at all 112 labels for every Train size; smaller sizes never remove labels. The eight Train sets are strict nested prefixes of `data/train_selection.parquet`. For an exact size $N$, let $q = \lfloor N / 112 \rfloor$ and $r = N \bmod 112$: exactly $r$ labels contain $q + 1$ examples and the remaining labels contain $q$ examples. This is the closest possible class balance while preserving exactly $N$ unique rows. Every complete block of 112 rows contains one example from every label, and label order within each block is deterministically shuffled so the remainder does not systematically favor the same labels. The 112K endpoint contains every row from the frozen 112K Train set exactly once.

Validation cross-checks selection labels against the frozen source Train file and requires all 112 labels, the exact requested row count, and the exact quotient/remainder allocation above. Training repeats this check before model loading. All configured sizes are divisible by four, so the four-rank distributed sampler neither duplicates nor drops rows; `dataloader_drop_last` is explicitly disabled.

## Build And Validate

```bash
python -m crossinterest.experiments.data_scaling.build_dataset
python -m crossinterest.experiments.data_scaling.validate_dataset
```

The jobs are defined in `amlt/amlt_formal.yaml` and run on the Feeds target. The 112K endpoint is not retrained: it reuses the already complete B+I/category result from `amazon-sft-112-category-20260827-134729`, whose Train, prompt, optimization, and evaluation contracts are identical. The config therefore contains 14 new jobs for the seven smaller sizes.