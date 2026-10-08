# MeRA Qwen2-1.5B Official-Code Strict Reproduction

This package reproduces the public Arts and Grocery experiments from *Large
Language Models with Multi-faceted Relation Alignment for User Novel Interest
Discovery* (Bi et al., PAKDD 2025). It executes the released MeRA training and
inference code at a fixed commit instead of using a local reimplementation.

## Reproduction Boundary

| Asset | Pinned value |
|---|---|
| Official repository | `https://github.com/GabyUSTC/MeRA.git` |
| Official commit | `c07bdbe57bf82e5c938b478035baee78d630b17c` |
| Base model | `Qwen/Qwen2-1.5B` |
| Model revision | `8a16abf2848eda07cc5253dec660bf1ce007ad7a` |
| Python / PyTorch / Transformers | `3.10` / `2.3.0` / `4.37.0` |
| Official-code changes | One typo repair: `output.indice` to `output.indices` in `inference.py` |

`prepare_official.py` verifies the commit and SHA-256 of every consumed official
source and data file before applying that one-line repair. `run.sh` also records
the complete public model snapshot manifest, environment lock, source manifest,
GPU information, tokenizer/category preflight, and training contract.

This is a strict execution reproduction of the fixed **public** code and assets.
It is not a claim of historical bitwise replay: the authors did not publish the
checksum of their local Qwen2 directory, their container digest, or trainer
state. No data balancing, local MeRA trainer, or cross-model compatibility patch
is included in this package.

## Exact Training Contract

| Setting | Arts | Grocery |
|---|---:|---:|
| Categories | 143 | 172 |
| Structure weight | 0.01 | 0.1 |
| Collaborative weight | 0.01 | 0.1 |
| Negatives per relation sample | 5 | 10 |
| Released natural long-tail split | Yes | Yes |

Both datasets use 8-bit Qwen2 weights with fp16 computation, 10 epochs maximum,
learning rate `1e-4`, cutoff length 512, LoRA rank/alpha/dropout `16/16/0.05`,
and all seven Qwen2 decoder projection types. The complete `embed_tokens` and
`lm_head` modules are trainable and saved. The released batch arguments are
`256/36`, which execute as 7 accumulation steps and an effective batch of 252.
The official trainer supplies AdamW Torch, a linear schedule, 20 warmup steps,
validation every 8 steps, checkpointing every 48 steps, and patience 2. The
shifted language-model CE includes EOS; both relation losses use the official
raw-dot scoring and sampling code.

## Layout

```text
crossinterest.baselines.mera_strict/
  run.sh                    # audit, preflight, official train/infer, and score
  prepare_official.py       # commit and per-file SHA gates; typo-only repair
  evaluate_official.py      # full-ranking validator and official P/R@1..4
  requirements.txt          # exact Python package versions
  amlt.yaml                 # one A100 preflight plus Arts/Grocery formal jobs
  SHA256SUMS                 # package-file transfer checksums
  tests/                    # CPU-only parsing and metric contract tests
```

Official source and data are cloned and verified at runtime; they are not
redistributed in the archive.

## Setup

Prerequisites are Linux, Git, Python 3.10, an NVIDIA GPU supported by
bitsandbytes 0.43.3, and access to Hugging Face. An A100 80 GB worker matches the
validated setup.

From the directory containing the extracted package:

```bash
sha256sum -c crossinterest.baselines.mera_strict/SHA256SUMS
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-build-isolation -r crossinterest.baselines.mera_strict/requirements.txt
python -m unittest discover -s crossinterest.baselines.mera_strict/tests -v
```

Do not install `apex`, `torchvision`, or `torchaudio` in this environment. The
runner rejects those packages because they caused incompatibilities in the
validated container.

## Quick Start

### 0. Inspect contracts without a GPU

```bash
CONTRACT_ONLY=1 DATASET=arts \
  OUTPUT_DIR="$PWD/contracts/arts" \
  bash crossinterest.baselines.mera_strict/run.sh

CONTRACT_ONLY=1 DATASET=grocery \
  OUTPUT_DIR="$PWD/contracts/grocery" \
  bash crossinterest.baselines.mera_strict/run.sh
```

Inspect `train_contract.json` in each output directory before using GPU time.

### 1. Run the real GPU preflight

```bash
PREFLIGHT_ONLY=1 DATASET=arts \
  OUTPUT_DIR="$PWD/outputs/preflight" \
  bash crossinterest.baselines.mera_strict/run.sh
```

This performs 48 optimizer steps with the original `train.py`, saves and reloads
the adapter, and checks finite logits. A contract-only run is not a substitute
for this GPU preflight.

### 2. Run Arts and Grocery

```bash
DATASET=arts OUTPUT_DIR="$PWD/outputs/arts" \
  bash crossinterest.baselines.mera_strict/run.sh

DATASET=grocery OUTPUT_DIR="$PWD/outputs/grocery" \
  bash crossinterest.baselines.mera_strict/run.sh
```

Use a distinct `OUTPUT_DIR` for each dataset. `WORK_DIR` may be set to relocate
the temporary official checkout and model cache. The runner produces the final
adapter under `model/`, its verified LoRA/embedding/head contract in
`adapter-contract.json`, full rankings under `evaluation/`, and official
Precision/Recall@1..4 in `evaluation/metrics.json`.

For evaluation-only recovery from a completed training output:

```bash
EVAL_ONLY=1 SOURCE_OUTPUT_DIR="$PWD/outputs/arts" \
  OUTPUT_DIR="$PWD/outputs/arts-recovery" \
  bash crossinterest.baselines.mera_strict/run.sh
```

The recovery path verifies that source, model, environment, preflight, and
training contracts match before loading the saved adapter.

## AMLT

`amlt.yaml` uses the exact validated base image and has no personal managed
identity embedded in it. Update only the `target` block if the collaborator uses
a different workspace. Then run the preflight before the formal jobs:

```bash
amlt run crossinterest.baselines.mera_strict/amlt.yaml \
  :qwen2-official-strict-preflight \
  mera-qwen2-strict-preflight --yes

amlt run crossinterest.baselines.mera_strict/amlt.yaml \
  :qwen2-arts-official-strict :qwen2-grocery-official-strict \
  mera-qwen2-strict-formal --yes
```

If the target has multiple user-assigned identities, add the appropriate
`_AZUREML_SINGULARITY_JOB_UAI` through the collaborator's submission settings;
do not commit a personal identity into this package.

## Validated Reference Run

The package was derived from the audited `r9` execution. Its completed public
code reproduction produced:

| Dataset | R@1 | R@2 | R@3 | R@4 | P@1 | P@2 | P@3 | P@4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arts | 0.0786 | 0.1485 | 0.2046 | 0.2520 | 0.2021 | 0.1882 | 0.1783 | 0.1649 |
| Grocery | 0.0503 | 0.0933 | 0.1406 | 0.1830 | 0.1180 | 0.1108 | 0.1126 | 0.1123 |

Numerical reruns can differ because the historical private trainer state is not
available. A valid run must first satisfy all source, environment, model,
tokenizer, adapter, and complete-ranking contracts; do not accept a metric file
from a run that bypassed those gates.