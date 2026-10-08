# Local Data

Data is excluded from Git. Prepare or obtain the frozen Home & Kitchen split using the [dataset documentation](../docs/experiments/dataset/README.md).

```text
data/
  amazon/
    raw/                    Original reviews and metadata, if rebuilding
    home112/
      train.parquet
      interest_id_semantic_mapping.json
      eval/
        dev_2k_behavior_interest.parquet
  news/                     Historical MIND data, optional
```

Do not regenerate a split silently when reproducing reported results. Check source hashes, user separation, class support, and complete-past novelty against the [data contract](../docs/experiments/dataset/README.md).

The Home112 evaluator derives its 112-category label space from the training Parquet file's `target_interest` column. It does not require a separate labels JSON file.

The local migration retains existing datasets through ignored symbolic links; those links are not part of the public repository. New clones must prepare their own data. `CROSSINTEREST_DATA_ROOT` controls research-module data resolution; configuration file paths are resolved relative to the project root. Set explicit absolute paths in a configuration when using external storage.