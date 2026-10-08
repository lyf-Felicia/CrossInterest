# Amazon 99-Label Training-Data Saturation

This experiment measures the Behavior + Interest (`behavior_interest`) category-name SFT learning curve on the frozen 99-label Amazon task. It is independent of the 112-label data-size experiment.

## Controlled Variables

- Models: Qwen3.5-9B and Llama-3.1-8B-Instruct.
- Train sizes: 10K, 20K, 40K, 80K, 120K, 160K, 200K, 240K, 280K, 320K, 360K, and all 396K rows.
- Output: exact natural-language category name.
- Evaluation: the unchanged frozen 2,000-row/99-label B+I Dev set from `crossinterest.evaluation/data`.
- Optimization: one epoch, learning rate 1e-4, effective batch size 32, and the same LoRA, prompt, and evaluation implementation as `crossinterest.legacy.training_99`.

The first four points approximately double from 10K through 80K. The high-data region then uses a regular 40K interval through 360K, followed by the natural 396K full-data endpoint. This provides readable, conventional sample-size labels while resolving the saturation region more densely.

Every subset is a strict prefix of `data/train_selection.parquet`. For an exact size $N$, let $q = \lfloor N/99 \rfloor$ and $r = N \bmod 99$. Exactly $r$ labels contain $q+1$ examples and the remaining labels contain $q$ examples, so every displayed size is exact and the class-count gap is at most one. Each complete block of 99 rows contains one example per label. Label order within each block is deterministically shuffled, while row order within each label follows the frozen 396K source Train set. All sizes are divisible by four, so the four-rank distributed sampler does not pad duplicate examples.

## Parallel Execution

The 396K endpoint reuses the complete B+I/category results in `crossinterest.legacy.training_99/RESULTS.md`. The 22 new jobs are split across Feeds and Recall according to the available Standard A100 capacity at submission time:

- Feeds: 10K, 40K, 80K, 160K, 200K, 280K, and 360K for both models.
- Recall: 20K, 120K, 240K, and 320K for both models.

Each job uses 4xA100 with per-device batch 1 and gradient accumulation 8. Classification on GPU 0 and candidate-free ranking on GPU 1 run concurrently after training, reducing the evaluation tail without changing either metric implementation.

## Build And Validate

```bash
python -m crossinterest.legacy.data_scaling_99.build_dataset
python -m crossinterest.legacy.data_scaling_99.validate_dataset
amlt run crossinterest.legacy.data_scaling_99/amlt/amlt_formal.yaml --dump
amlt run crossinterest.legacy.data_scaling_99/amlt/amlt_recall.yaml --dump
```
