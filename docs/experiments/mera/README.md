# MeRA Baseline on Amazon

This directory contains three deliberately separated MeRA experiment lines: fixed-public-code strict execution on the official Arts and Grocery releases, audited official-code cross-model execution, and an official-code extension to the frozen Home & Kitchen benchmark. The earlier mechanism adaptation is retained only as a legacy comparison.

The fixed-public-code strict line uses `Qwen/Qwen2-1.5B`. The official-code cross-model line uses `Qwen/Qwen3.5-9B` and `NousResearch/Meta-Llama-3.1-8B-Instruct`; it preserves the official method while applying only audited model-architecture and current-library compatibility changes. Home is an official-code extension rather than a paper-dataset strict reproduction because the paper did not release a Home experiment.

## Experiment Matrix

| Dataset | Models | Paper metrics | Unified project metrics |
|---|---|---|---|
| Arts strict | Qwen2-1.5B, pinned official source | R/P@1/2/3/4 | Query-mean Accuracy, Recall@10/20, nDCG@10/20 |
| Grocery strict | Qwen2-1.5B, pinned official source | R/P@1/2/3/4 | Query-mean Accuracy, Recall@10/20, nDCG@10/20 |
| Arts/Grocery official-code cross-model | Qwen3.5-9B, Llama-3.1-8B | R/P@1/2/3/4 | Query-mean Accuracy, Recall@10/20, nDCG@10/20 |
| Home official-code 2x2 | Qwen3.5-9B, Llama-3.1-8B | Not applicable | Query-mean metrics plus single-gold category macro |
| Legacy mechanism adaptation | Qwen3.5-9B, Llama-3.1-8B | Diagnostic only | Historical comparison only |

Arts and Grocery keep their multi-gold paper evaluation. Unified Recall uses set recall, and unified nDCG uses binary-relevance DCG normalized by the ideal ordering of the complete gold set. Metrics are averaged equally over source queries, not expanded gold rows; no gold is randomly selected. Ten-way accuracy first averages the per-gold decisions within each source query. Its negatives exclude the complete history and all other gold labels from the same source row. Home & Kitchen reuses the existing frozen candidates byte-for-byte. Both metric families are derived from one forward pass and one full category-token ranking per source row.

## Legacy Mechanism-Adaptation Decisions

This table describes the earlier local reimplementation, not the current official-code jobs.

| Component | Implementation |
|---|---|
| Category representation | One atomic token per context child category, initialized from the mean base-model rows of the child-category name |
| Input history | The same atomic tokens used by the targets |
| Language objective | Full-vocabulary next-token cross entropy at the assistant position |
| Relation objective | Structure and collaborative InfoNCE on both input embeddings and effective LM-head rows |
| Efficient tuning | Sparse new-token input rows plus all-linear LoRA, including the LM head |
| Arts/Grocery budget | Up to 10 epochs, validation patience 2, effective batch 36, LoRA alpha 16 |
| Home budget | One epoch, effective batch 32, LoRA alpha 32; no tuning on formal Dev-2K |
| Qwen-specific path | Thinking disabled; standard Transformers forward for training and evaluation |
| Inference | Direct logits over frozen category token IDs; no text generation or parser |

The official Arts/Grocery Train splits are naturally long-tailed and the released trainer contains no class sampler or class-weighted loss. The main reproduction therefore does not rebalance them. `target_distribution.json` records complete split-level counts, including zero-target labels. The historical balanced diagnostic keeps the Arts budget at 20,186 rows and deterministically resamples the 115 observed Train target classes to 175 or 176 rows each; it does not synthesize examples for the 28 zero-target classes and never changes Validation/Test. Its Natural and Balanced results are not strict reproduction results.

The strict Qwen2 line runs `GabyUSTC/MeRA@c07bdbe57bf82e5c938b478035baee78d630b17c` directly. It preserves the released 8-bit/fp16 parameterization, full embedding and LM-head `modules_to_save`, shifted CE through EOS, raw-dot relation sampling, effective batch 252, AdamW Torch, linear scheduler, and dataset-specific relation weights. The only source correction is `output.indice` to `output.indices` in inference. `Qwen/Qwen2-1.5B` is pinned to revision `8a16abf2848eda07cc5253dec660bf1ce007ad7a`; all source, data, model-asset, environment, and training contracts are written to the job output. This is a strict execution of the fixed public code and public assets, not a claim of bitwise replay of the authors' unpublished local model directory, container digest, or trainer state.

The cross-model runner checks out the same commit for every job and permits changes only in official `train.py` and `inference.py`. The audited patch replaces obsolete Transformers/PEFT APIs, uses public embedding accessors, covers every decoder linear projection with model-specific LoRA target names, handles model-parallel devices, and repairs the official inference typo. It does not replace the official prompts, semantic category-token initialization, shifted CE, raw-dot structure/collaborative losses, relation sampling, trainable embedding/head policy, optimizer, scheduler, or token-only full-ranking inference.

Home converts the frozen 112K Train, a user-disjoint 2K validation sample, and the frozen Dev-2K test into the official JSON/tree/map/cooccurrence contract. It has 136 valid context categories but only 112 frozen prediction labels. All 136 receive official category tokens and relation supervision; official inference produces a complete 136-category ranking, which is deterministically filtered to the 112 prediction labels before scoring. The four conditions differ only in the two relation weights: SFT `(0, 0)`, structure-only `(0.05, 0)`, collaborative-only `(0, 0.1)`, and both `(0.05, 0.1)`.

## Taxonomy and Interest-ID Contract

The paper counts levels below each Amazon domain root: its first-level categories are parent nodes, and its predicted second-level categories are their children. The Home metadata path is `Home & Kitchen > path[1] > path[2]`; `target_level_2` stores `path[1]`, while `target_interest` stores the structural label `path[1] > path[2]`. Thus the taxonomy edge and prediction granularity match the paper, although the local `L2/L3` field names are offset by the included domain root.

The paper does not define a separate SID algorithm; it registers each quoted child-category name as a special token. The official-code lines retain that representation and initialize Home tokens from the child names through the released code. The legacy reimplementation instead uses explicit `<category_NNN>` identifiers, which are mechanism-equivalent but not byte-identical official tokens. Neither representation is the project's Interest-ID method: that mapping assigns `<interest_NNN>` by the lexical order of the 112 child names, keeps concrete Interest IDs out of the history input, and trains ordinary SFT. See `EXPERIMENT_REPORT.md` for the full audit.

## Layout

```text
crossinterest.baselines.mera/
  core.py                 # label spaces, relation graphs, InfoNCE, and metrics
  data.py                 # official/Home adapters and unified candidate construction
  modeling.py             # category-token registration and effective LoRA head rows
  prepare_data.py         # immutable label and Train-only relation assets
  fetch_official_data.py  # pinned official commit download with SHA-256 gates
  prepare_official_strict.py # official checkout verifier and one-line inference fix
  adapt_official_model.py # audited current-model/API patch for official train/inference
  prepare_official_home.py # frozen Home to official JSON/tree/map/cooccurrence contract
  run_official_strict.sh  # pinned environment, GPU preflight, train, inference, evaluation
  <archived-job-id>.sh # official-code Qwen3.5/Llama runner
  evaluate_official_strict.py # full-ranking gates and dual paper-metric implementation
  evaluate_official_home.py # complete 136-rank gate and frozen 112-label scoring
  gate_and_submit_official_home.sh # dual-model training gate and formal submission
  train.py                # two-model MeRA training entry point
  evaluate.py             # paper and unified metrics from one forward pass
  validate_artifact.py    # adapter and training-contract gate
  run_train_eval.sh       # AMLT train, validate, and evaluate runner
  requirements_amlt.txt   # pinned worker dependencies
  amlt/
    amlt_official_strict.yaml # Qwen2 Arts/Grocery official-code strict jobs
    <archived-job-id>.yaml # Qwen3.5/Llama Arts/Grocery official-code jobs
    amlt_official_home_ablations.yaml # Home official-code 2x2 relation-loss jobs
    amlt_qwen2_arts.yaml # historical non-strict natural/balance diagnostic
    amlt_smoke.yaml       # two-step smoke for both main models
    amlt_formal.yaml      # 3 datasets x 2 main models
    <archived-job-id>.yaml # three remaining Home ablations x 2 models
  data/{arts,grocery,home}/ # frozen label and relation assets
  tests/test_core.py      # real-data and synthetic contract tests
  EXPERIMENT_REPORT.md    # detailed Chinese reproduction report
  EXPERIMENT_LOG.md       # chronological Chinese experiment record
  RESULTS.md              # formal result tables
```

## Setup

Run commands from `crossinterest_formal/amazon`:

```bash
python -m pip install -r crossinterest.baselines.mera/requirements_amlt.txt
```

The official data is not duplicated in this directory. Fetch the immutable release when a local copy is needed:

```bash
python -m crossinterest.baselines.mera.fetch_official_data \
  --output-dir /tmp/mera-official
```

This checks out commit `c07bdbe57bf82e5c938b478035baee78d630b17c` and verifies all twelve required data files by SHA-256.

## Quick Start

### 0. Run contract tests

```bash
conda run -n exp python -m pytest -q crossinterest.baselines.mera/tests
```

### 1. Rebuild relation assets

The checked-in assets are already frozen. These commands independently rebuild them:

```bash
python -m crossinterest.baselines.mera.prepare_data \
  --dataset arts \
  --official-root /tmp/mera-official \
  --output-dir /tmp/mera-assets

python -m crossinterest.baselines.mera.prepare_data \
  --dataset grocery \
  --official-root /tmp/mera-official \
  --output-dir /tmp/mera-assets

python -m crossinterest.baselines.mera.prepare_data \
  --dataset home \
  --output-dir /tmp/mera-assets
```

### 2. Preflight AMLT without submission

```bash
conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_smoke.yaml \
  mera-smoke-preflight --dump

conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_formal.yaml \
  mera-formal-preflight --dump

conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/<archived-job-id>.yaml \
  mera-ablation-preflight --dump

conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_official_home_ablations.yaml \
  mera-official-home-preflight --dump
```

### 3. Run the strict Qwen2 preflight and training

Run the real A100 preflight first. It installs and checks the pinned official environment, verifies source/data/model contracts, loads the frozen Qwen2 revision in 8-bit, and checks the complete category-token space:

```bash
conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_official_strict.yaml \
  :qwen2-official-strict-preflight \
  mera-official-strict-preflight-r7-20260901 \
  -d "Pinned official MeRA source, environment, tokenizer, and Qwen2 int8 GPU preflight" -y
```

Only after that job passes, submit the two formal jobs from the same config:

```bash
conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_official_strict.yaml \
  :qwen2-arts-official-strict :qwen2-grocery-official-strict \
  mera-official-strict-20260901 \
  -d "Official-code strict MeRA reproduction on Arts and Grocery" -y
```

Strict result cells remain blank until each job passes full-ranking and dual-metric validation.

### 4. Run the official-code Home factorial

Submit only the two one-step training gates first. The gate script validates both jobs and submits all eight formal jobs only after every source, environment, data, LoRA, loss-device, save, and reload contract passes:

```bash
conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_official_home_ablations.yaml \
  :qwen35-home-official-preflight :llama31-home-official-preflight \
  mera-official-home-preflight-r2-20260902 -y

bash crossinterest.baselines.mera/gate_and_submit_official_home.sh
```

### 5. Run the legacy adaptation smoke and matrix

```bash
conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_smoke.yaml \
  mera-two-model-smoke-20260831 \
  -d "MeRA two-model two-step smoke" -y

conda run -n exp amlt run \
  crossinterest.baselines.mera/amlt/amlt_formal.yaml \
  mera-two-model-formal-20260831 \
  -d "MeRA Arts Grocery Home formal matrix" -y
```

The `<archived-job-id>.yaml` jobs belong to the legacy mechanism reimplementation and must not populate official-code result cells.

## Outputs

Each job writes the following under its own `AMLT_OUTPUT_DIR`:

```text
train/
  train_contract.json
  train_metrics.json
  semantic_token_setup.json
  final_adapter/
evaluation/
  run_contract.json
  metrics.json
  source_rankings.jsonl
  unified_details.jsonl
nvidia-smi.txt
pip-freeze.txt
```

`evaluation/metrics.json` contains both `paper_metrics` and `unified_metrics` for Arts/Grocery. Home has `paper_metrics: null` and the same unified metric object.

The official-code strict jobs additionally write `official-assets.json`, `model-assets.json`, `project-source-contract.json`, `environment-contract.json`, `environment-lock.txt`, `preflight-contract.json`, `train_contract.json`, and `evaluation/<archived-job-id>.json`. Both metric implementations are stored in `evaluation/metrics.json` under `paper_metrics` and `official_formula_metrics`.

## Known Limits

- The official JSON files omit user IDs and contain observable exact duplicates across splits, so the released Arts/Grocery data cannot prove user-disjoint evaluation.
- The paper reports 2,524 Arts validation rows, while the released file contains 2,523.
- The official Grocery prediction file does not exactly reproduce the paper table; see `EXPERIMENT_REPORT.md` for the independent recomputation.
- The adaptation jobs resolve the current Hugging Face model commit once and pin that immutable SHA across training and evaluation. The strict Qwen2 jobs instead require one hard-coded model revision and exact package versions, but the authors' original container digest and local model checksum were not published; every formal artifact therefore records `pip-freeze.txt`, `nvidia-smi.txt`, model-file hashes, data hashes, and run contracts.

## Reference

Bi et al., *Large Language Models with Multi-faceted Relation Alignment for User Novel Interest Discovery*, PAKDD 2025, DOI `10.1007/978-981-96-8298-0_28`. Official code: <https://github.com/GabyUSTC/MeRA>.
