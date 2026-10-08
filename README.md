# CrossInterest: Generative New Interest Prediction

Next-new-interest prediction from ordered behavior and interest histories. The research task is to predict a category that has not appeared in a user's complete past, rather than the next product. The current benchmark uses Amazon Reviews 2023 Home & Kitchen, behavior-plus-interest input, and semantically initialized single-token Interest IDs.

This is a research codebase, not a claim of a completed paper reproduction. The MeRA baseline and MiniOneRec-inspired auxiliary tasks are documented separately. Negative results and changes in training budget are retained in the [experiment documentation](docs/experiments/overview.md).

The [consolidated experiment results](RESULTS.md) collect per-experiment tables, matched baselines, and protocol caveats.

The organization follows the [research presentation](https://docs.google.com/presentation/d/14JOumukKLuqFPzMQWnuxFqTMGTQAO6HjlgrYWsEgzOI/edit): task and dataset construction, evaluation, behavior-plus-interest representation with semantic Interest IDs, external baselines, and Cross-Interest Reconstruction (CIR). Later exploratory methods are kept distinct from the main method and its matched controls.

## Repository Structure

```text
CrossInterest/
  RESULTS.md             Consolidated formal experiment results
  configs/               Portable training and evaluation settings
  src/crossinterest/
    datasets/            Amazon data construction and contracts
    evaluation/          Parsing, ranking metrics, and zero-shot evaluation
    training/            B+I SFT with semantic Interest IDs
    baselines/           Google and MeRA adaptations
    methods/
      cir/               Cross-Interest Reconstruction data and training
      alignment/         ID alignment and warm-up controls
      ...                Other explicitly documented research variants
    experiments/         Data scaling, denoising, and summary studies
    losses/              Historical loss experiments
    legacy/              99-class protocol, not the Home112 main method
  tests/                 Unit tests and model integration regressions
  docs/                  Research overview and experiment-specific evidence
  tools/                 Publication and duplicate-content checks
  data/README.md         Data placement instructions; no datasets bundled
  pyproject.toml         Package, CLI, dependencies, and test configuration
  .github/workflows/     CPU continuous integration
```

This directory is the standalone source of truth: it contains no private archive, virtual environment, copied release tree, dataset symlink, or duplicate implementation snapshot. Place local datasets under `data/` and generated predictions and checkpoints under `outputs/`; both are excluded from Git. Alternatively, set `CROSSINTEREST_DATA_ROOT` and `CROSSINTEREST_OUTPUT_ROOT` to external directories.

These environment variables control module defaults. The provided YAML configurations specify paths explicitly; update their path arguments as well when using external data or output directories.

Private cloud receipts, original frozen submissions, environments, and large historical artifacts remain outside this repository. Public code does not require access to the original Azure workspace. Historical orchestration helpers may require archived contracts; they are not the default training interface.

## Research Map

| Presentation topic | Implementation | Experiment documentation |
|---|---|---|
| Task, novelty, and balanced data (slides 2-13) | [datasets](src/crossinterest/datasets/amazon) | [Dataset protocol](docs/experiments/dataset/README.md) |
| Prompts and macro ranking evaluation (slides 14-19) | [evaluation](src/crossinterest/evaluation), [ranking](src/crossinterest/training/evaluate_ranking.py) | [Supervised evaluation](docs/experiments/supervised_learning/README.md) |
| B+I and semantic Interest IDs (slide 20) | [training](src/crossinterest/training) | [Representation results](docs/experiments/supervised_learning/results.md) |
| Google and MeRA comparisons (slides 21-23) | [baselines](src/crossinterest/baselines) | [Google](docs/experiments/google_baseline/README.md), [MeRA](docs/experiments/mera/report.md) |
| Scaling, weighting, and warm-up controls (slides 24-29) | [data scaling](src/crossinterest/experiments/data_scaling), [crossfit](src/crossinterest/methods/crossfit), [alignment](src/crossinterest/methods/alignment) | [Scaling](docs/experiments/data_scaling/README.md), [crossfit](docs/experiments/crossfit/report.md), [backbones](docs/experiments/backbones/report.md) |
| CIR and its controls (slides 30-32) | [CIR](src/crossinterest/methods/cir) | [CIR results and budget caveats](docs/experiments/cir/report.md) |

The presentation is a research navigation source, not a new validation of its tables. Follow the experiment documents for run identity, training budget, raw-result verification, and negative results. Ranking saturation and candidate-based accuracy are separate questions; presentation tables from different SFT runs are not interchangeable baselines.

## Setup

Python 3.10 or newer is required. CPU checks do not download a model or dataset:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest tests/test_public_api.py tests/test_repository.py
crossinterest doctor
```

For training, install the matching PyTorch build for your CUDA environment, then install the research dependencies:

```bash
pip install -e '.[train,test]'
pip install -e '.[cuda]' --no-build-isolation
```

The reference runs used PyTorch 2.8.0, Transformers 5.14.1, PEFT 0.18.1, Accelerate 1.12.0, flash-linear-attention 0.5.2, and causal-conv1d 1.7.0. GPU kernels are optional for CPU unit tests, but the Qwen reference configuration requires the fast path. Check model access and the model's own license before downloading weights.

## Data

Place the frozen data described in [data/README.md](data/README.md) under `data/amazon/home112/`. Datasets, raw reviews, predictions, and checkpoints are intentionally not checked in. [Dataset construction](docs/experiments/dataset/README.md) explains the source and novelty checks. No small synthetic sample is presented as the official benchmark.

## Training and Evaluation

Inspect commands before starting an expensive job:

```bash
crossinterest train --config configs/training/home112.yaml --dry-run
crossinterest evaluate --config configs/evaluation/home112.yaml --dry-run
```

The training configuration uses four GPUs, micro-batch 1, accumulation 8, effective batch 32, and gradient checkpointing. Ranking runs in one process with the model distributed across GPUs:

```bash
crossinterest train --config configs/training/home112.yaml
crossinterest evaluate --config configs/evaluation/home112.yaml
```

These are reorganized public entry points, not byte-identical copies of historical cloud submission packages. No large-model run has been launched to validate the reorganization. Preserve the reference data, dependency versions, model revision, and each experiment's budget when comparing results.

## Evaluation Protocol

Primary metrics are class-macro Recall@10/20 and nDCG@10/20 over 112 target categories on Dev2000. Ranking uses candidate-free, deterministic, full-vocabulary 20-beam generation. Invalid IDs are not repaired and seen categories are not masked. Ten-candidate classification and 99-class historical experiments use different protocols and must not be pooled with the primary ranking results.

The development set has been reused during research. Point estimates are not evidence of statistical significance. See the [research overview](docs/experiments/overview.md), [method experiments](docs/experiments/mass_readout/report.md), and [repository conventions](docs/repository.md).

## Contributing and Licensing

Run the lightweight tests and repository checks before proposing changes; see [CONTRIBUTING.md](CONTRIBUTING.md). Project licensing and attribution must be confirmed by the repository owner before public redistribution; third-party models, datasets, and implementations retain their own terms. See [licensing and sources](docs/licensing.md).