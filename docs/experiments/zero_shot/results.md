# Amazon Zero-Shot 10-Way Evaluation Results

> Metric: Macro Accuracy (%), higher is better. Each model-input result uses the same 2,000 stratified Dev samples and paired candidate sets.
> Empty cells indicate runs that are still in progress. No confidence intervals or Wilson statistics are reported.
> Evaluation date: 2026-08-25. No training or adapters are used.
> Prompt schema: v3. The selection instruction appears before the 10 candidates, and models respond with only the exact candidate name. All earlier prompt versions are excluded.
> Progress at 2026-08-26 03:34 +0800: 24/24 views complete; 48,000/48,000 samples covered (100.00%).

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
      <td>15.21%</td>
      <td>11.91%</td>
      <td>15.28%</td>
    </tr>
    <tr>
      <td>GPT-5.6 Sol (None)</td>
      <td>11.96%</td>
      <td>12.23%</td>
      <td>16.06%</td>
    </tr>
    <tr>
      <td>GPT-5 (Medium)</td>
      <td>11.58%</td>
      <td>10.43%</td>
      <td>14.42%</td>
    </tr>
    <tr>
      <td>GPT-5 (None)</td>
      <td>7.59%</td>
      <td>11.80%</td>
      <td>13.63%</td>
    </tr>
    <tr>
      <td>Gemini 3.6 Flash (Medium)</td>
      <td>13.54%</td>
      <td>10.33%</td>
      <td>13.94%</td>
    </tr>
    <tr>
      <td>Gemini 3.6 Flash (Minimal)</td>
      <td>13.20%</td>
      <td>11.89%</td>
      <td>15.02%</td>
    </tr>
    <tr>
      <td>Qwen3.5-9B</td>
      <td>14.62%</td>
      <td>12.79%</td>
      <td>13.30%</td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>10.75%</td>
      <td>8.66%</td>
      <td>9.27%</td>
    </tr>
    <tr>
      <td>Random Baseline</td>
      <td>10.00%</td>
      <td>10.00%</td>
      <td>10.00%</td>
    </tr>
  </tbody>
</table>
