# Google VIE Baseline on Amazon

This directory is a public-data mechanism reproduction of *LLMs for User Interest Exploration in Large-scale Recommendation Systems* (Wang et al., arXiv:2405.16363v2, 2024). It preserves the paper's high-level K=2 cluster-transition SFT mechanism and adds one deliberate extension: evaluation on the frozen 112-label Amazon benchmark with the same Macro Accuracy, Recall, and nDCG metrics used by the neighboring SFT experiments.

This is not an exact reproduction of Google's private industrial system. No official code, training data, interest clusters, or production recommender artifacts were found. The public substitute uses Amazon taxonomy labels and the existing user-disjoint Amazon Reviews'23 mother pool.

## Reproduction Scope

| Component | Paper | Public Amazon adaptation | Status |
|---|---|---|---|
| Interest unit | Designer-controlled language cluster with salient phrases/entities | One frozen `L2 > L3` Amazon taxonomy label rendered as `{L2, L3}` | Mechanism-preserving substitute |
| Context | Two most recent interacted unique clusters | Two most recent distinct supported taxonomy labels, oldest to newest | Preserved |
| Target | A new cluster absent from prior interests | `target_interest` absent from the complete pre-target history | Preserved |
| SFT curation | Top 10 transition pairs for each of 761 clusters, 7,610 examples | Top 10 ordered pairs for each of 112 labels, 1,120 examples | Preserved and rescaled |
| Reference model | Gemini-Pro | Qwen3.5-9B and Llama-3.1-8B-Instruct with LoRA | Public replacement |
| Reference optimization | Batch 16, approximately 3,000 steps | Batch 16; 442-step primary setting plus separate 3,000-step sensitivity | Pre-registered adaptation |
| Evaluation | Private industrial offline and live evaluation | Frozen Amazon 10-way Macro Accuracy plus candidate-free Macro Recall/nDCG@10/20 | Benchmark-comparable extension |
| Low-level item grounding | Sequential recommender restricted to generated clusters | Not reproduced | Out of scope |

The primary 442-step budget preserves the paper's approximate number of data passes:

$$
\operatorname{round}\left(3000 \times \frac{1120}{7610}\right)=442.
$$

The 3,000-step setting is intentionally isolated in a second AMLT config. It must not replace the primary result through post-hoc selection.

## Data Contract

The builder scans 5,372,964 rows from the immutable unbalanced Amazon Train mother pool. For each supported target label it counts ordered transitions `(C1, C2) -> target`, ranks pairs by descending frequency and then lexicographic pair order, and retains the top 10. The first matching row in immutable JSONL order is stored only as provenance.

- Train: 1,120 rows, exactly 10 ordered pairs for each of 112 targets.
- Pair Dev: 1,997 rows from the frozen stratified Dev-2K interest view; three rows lack two distinct supported history clusters and are excluded.
- Pair grid: all $112^2=12,544$ ordered pairs, including same-cluster pairs, for optional exploration analysis.
- Prompt schema: `google-interest-v2`, aligned to the paper's Figure 2 prompt shape and `New cluster: ` output prefix.
- Frozen hashes and source statistics: data/MANIFEST.json（私有原始证据，不随公开仓库分发）.
- Legacy v1 reconstruction audit: data/LEGACY_V1_AUDIT.md（私有原始证据，不随公开仓库分发）.

## Layout

```text
crossinterest.baselines.google/
  build_dataset.py          # streaming transition counts and deterministic freezes
  prompts.py                # paper-shaped K=2 generation/classification prompts
  train.py                  # assistant-only LoRA SFT
  evaluate.py               # constrained 10-way Macro Accuracy
  evaluate_ranking.py       # unconstrained 20-beam Recall/nDCG
  run_train_eval.sh         # one adapter followed by both evaluations
  requirements_amlt.txt     # pinned cluster dependencies
  amlt/
    amlt_formal.yaml        # primary 442-step Qwen/Llama jobs
    <archived-job-id>.yaml # strict single-A100 Llama ranking resume
    amlt_sensitivity_3000.yaml # separate 3,000-step sensitivity jobs
  data/                     # frozen Train, Dev, pair grid, and manifests
  tests/test_contract.py    # K=2, prompt, curation, and metric contracts
  EXPERIMENT_LOG.md         # chronological Chinese experiment record
  RESULTS.md                # unified result tables; unfinished cells stay blank
```

## Setup

Run commands from `crossinterest_formal/amazon`:

```bash
conda run -n exp python -m pytest -q crossinterest.baselines.google/tests
```

AMLT installs the pinned dependencies from `requirements_amlt.txt`. For a separate local environment:

```bash
python -m pip install -r crossinterest.baselines.google/requirements_amlt.txt
```

## Quick Start

### 0. Contract smoke test

```bash
cd crossinterest_formal/amazon
conda run -n exp python -m pytest -q crossinterest.baselines.google/tests
```

### 1. Rebuild the frozen v2 data

This streams the 18 GB mother-pool JSONL and overwrites only `crossinterest.baselines.google/data`.

```bash
cd crossinterest_formal/amazon
conda run -n exp python -m crossinterest.baselines.google.build_dataset \
  --prompt-schema google-interest-v2 \
  --output-dir crossinterest.baselines.google/data

conda run -n exp python -m crossinterest.baselines.google.build_dataset \
  --prompt-schema google-interest-v2 \
  --verify-frozen crossinterest.baselines.google/data
```

The second command rebuilds into a temporary directory and requires byte-identical SHA-256 values for all three parquet files plus equality of all stable manifest fields. The writer is pinned to PyArrow 24.0.0 because parquet bytes are part of this contract.

### 2. Preflight the AMLT configs without submission

```bash
cd crossinterest_formal/amazon/crossinterest.baselines.google/amlt
amlt run amlt_formal.yaml --dump
amlt run amlt_sensitivity_3000.yaml --dump
```

### 3. Run the primary setting

Submission is deliberately manual:

```bash
cd crossinterest_formal/amazon/crossinterest.baselines.google/amlt
amlt run amlt_formal.yaml google-vie-amazon-scaled-442-recall-20260828
```

The 3,000-step sensitivity setting is separate:

```bash
amlt run amlt_sensitivity_3000.yaml google-vie-amazon-paper-3000
```

Audited primary results are recorded in [RESULTS.md](results.md). Both models are complete for both protocols. Llama ranking was recovered from 1,745/1,997 rows on one Recall A100 and passed the full 1,997-row audit.

## Evaluation

Each job trains one adapter, then runs both protocols on the same 1,997 eligible Dev rows:

- `classification/metrics.json`: 10-way candidate-constrained decoding and Macro Accuracy. Candidates are appended only at evaluation time.
- `ranking/metrics.json`: candidate-free direct 20-beam generation and Macro Recall@10/20 plus Macro nDCG@10/20. No decoder label constraint is used.

Both protocols compress every user's input to exactly two distinct history clusters. This makes the baseline mechanism faithful but means it is not a three-view ablation and should not be labeled Behavior, Interest, or Behavior + Interest.

## Citation

```bibtex
@misc{wang2024llmsuserexplorationlargescale,
  title={LLMs for User Interest Exploration in Large-scale Recommendation Systems},
  author={Jianling Wang and Haokai Lu and Yifan Liu and He Ma and Yueqi Wang and Yang Gu and Shuzhou Zhang and Ningren Han and Shuchao Bi and Lexi Baugher and Ed Chi and Minmin Chen},
  year={2024},
  eprint={2405.16363},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```

## Known Limits

- The paper's private cluster vocabulary, prompt-generation pipeline, Gemini-Pro checkpoint, industrial data, and live serving system are unavailable.
- Amazon taxonomy names are metadata substitutes, not automatically extracted salient keyword sets.
- Same-cluster rows exist only in the exhaustive pair grid; SFT and Dev require two distinct recent clusters.
- The AMLT configs target Recall A100s and use `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04`. The image digest and Hugging Face model revisions are not pinned, so formal artifacts must record the resolved runtime versions.
- The primary 442-step experiment is `google-vie-amazon-scaled-442-recall-20260828`. All Qwen and Llama metrics have passed the 1,997-row audit; interrupted Llama ranking was completed by `google-vie-llama-ranking-recovery-r3-20260831`. The 3,000-step sensitivity jobs have not been submitted.
