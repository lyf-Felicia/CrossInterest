# Amazon 112-Label No-Training Results

> Metric: Macro Accuracy (%), higher is better. All cells use the exact frozen Dev-2K samples and paired candidate sets from `crossinterest.training`.
> Prompt schema: category-name v3 with 10 natural-language candidates. Local HF models use deterministic candidate-constrained 10-beam search; API models use their frozen reasoning-effort setting, no sampling temperature, and exact-candidate response parsing, matching the earlier no-training table.
> GPT models use TRAPI `redmond/interactive`. Both Gemini rows always use the `copilot-api` reverse proxy at `127.0.0.1:4141`; Gemini is not routed through TRAPI.
> No training or adapters are used. Interest-ID output and candidate-free ranking are outside this experiment.
> Empty cells mean that no complete 2,000-sample, 112-class `metrics.json` is available.
> Formal sources: `amazon-notrain-112k-v2-20260827-1940` for Qwen/Llama and `runs/formal/` for API models. The canceled v1 experiment is not a result source.
> Qwen audit (2026-08-28): the displayed 12.79% values are rounded from 12.7859477%, 12.7859477%, and 12.7917834%. The three jobs used distinct view contracts and dataset SHA-256 values; all 2,000 prompts differ across views, and pairwise predictions differ on 995/708/705 samples. An independent full retest is running as `amazon-notrain-112k-qwen-retest-20260828-083653`.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="3">Macro Accuracy (%) ↑</th>
    </tr>
    <tr>
      <th>Behavior</th>
      <th>Interest</th>
      <th>Behavior + Interest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GPT-5.6 Sol (Medium)</td>
      <td>13.54%</td><td>11.23%</td><td>16.17%</td>
    </tr>
    <tr>
      <td>GPT-5.6 Sol (None)</td>
      <td>12.56%</td><td>12.09%</td><td>16.60%</td>
    </tr>
    <tr>
      <td>GPT-5 (Medium)</td>
      <td>10.30%</td><td>10.48%</td><td>13.92%</td>
    </tr>
    <tr>
      <td>GPT-5 (None)</td>
      <td>6.98%</td><td>11.94%</td><td>14.09%</td>
    </tr>
    <tr>
      <td>Gemini 3.6 Flash (Medium)</td>
      <td>14.04%</td><td>10.25%</td><td>15.05%</td>
    </tr>
    <tr>
      <td>Gemini 3.6 Flash (Minimal)</td>
      <td>12.68%</td><td>12.29%</td><td>15.55%</td>
    </tr>
    <tr>
      <td>Qwen3.5-9B</td>
      <td>12.79%</td><td>12.79%</td><td>12.79%</td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>10.06%</td><td>11.43%</td><td>10.90%</td>
    </tr>
    <tr>
      <td>Random Baseline</td>
      <td>10.00%</td><td>10.00%</td><td>10.00%</td>
    </tr>
  </tbody>
</table>