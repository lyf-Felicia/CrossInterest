# Amazon 112-Label Training-Data Saturation Results

> All runs use the B+I input, category-name output, one epoch, effective batch size 32, and the same frozen 2,000-row/112-label evaluation set. Empty cells mean no complete method-valid metrics are available.
> This is the independent 112-label/112K protocol. Its label space, Train set, and Dev-2K differ from the 99-label experiment in `crossinterest.legacy.data_scaling_99/RESULTS.md`; values must not be merged into one learning curve or compared as matched points.
> Status snapshot: 2026-09-01. All 14 newly submitted jobs passed. Classification is deterministic 10-beam constrained decoding; ranking is candidate-free direct 20-beam decoding. Accuracy is shown as a percentage; Recall and nDCG are 0-1 values.

<table>
  <thead>
    <tr><th rowspan="2">Model</th><th rowspan="2">Train rows</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
    <tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="8">Qwen3.5-9B</td><td>5K</td><td>7.38%</td><td>0.1511</td><td>0.2733</td><td>0.0750</td><td>0.1056</td></tr>
    <tr><td>10K</td><td>6.75%</td><td>0.1635</td><td>0.2977</td><td>0.0778</td><td>0.1114</td></tr>
    <tr><td>20K</td><td>9.71%</td><td>0.1876</td><td>0.3206</td><td>0.0891</td><td>0.1223</td></tr>
    <tr><td>40K</td><td>8.91%</td><td>0.1836</td><td>0.3330</td><td>0.0925</td><td>0.1300</td></tr>
    <tr><td>60K</td><td>14.09%</td><td>0.1998</td><td>0.3335</td><td>0.0980</td><td>0.1315</td></tr>
    <tr><td>80K</td><td>13.13%</td><td>0.2064</td><td>0.3578</td><td>0.1042</td><td>0.1420</td></tr>
    <tr><td>100K</td><td>17.36%</td><td>0.1974</td><td>0.3476</td><td>0.0985</td><td>0.1361</td></tr>
    <tr><td>112K (all)</td><td>18.68%</td><td>0.2059</td><td>0.3632</td><td>0.1005</td><td>0.1397</td></tr>
    <tr><td rowspan="8">Llama-3.1-8B-Instruct</td><td>5K</td><td>7.74%</td><td>0.1366</td><td>0.2639</td><td>0.0660</td><td>0.0979</td></tr>
    <tr><td>10K</td><td>8.39%</td><td>0.1479</td><td>0.2841</td><td>0.0721</td><td>0.1062</td></tr>
    <tr><td>20K</td><td>9.99%</td><td>0.1697</td><td>0.3212</td><td>0.0816</td><td>0.1193</td></tr>
    <tr><td>40K</td><td>12.87%</td><td>0.1884</td><td>0.3318</td><td>0.0929</td><td>0.1287</td></tr>
    <tr><td>60K</td><td>11.54%</td><td>0.1827</td><td>0.3250</td><td>0.0925</td><td>0.1281</td></tr>
    <tr><td>80K</td><td>11.97%</td><td>0.1950</td><td>0.3404</td><td>0.0967</td><td>0.1330</td></tr>
    <tr><td>100K</td><td>13.91%</td><td>0.1846</td><td>0.3428</td><td>0.0937</td><td>0.1333</td></tr>
    <tr><td>112K (all)</td><td>16.23%</td><td>0.1923</td><td>0.3512</td><td>0.0973</td><td>0.1375</td></tr>
  </tbody>
</table>

## Saturation diagnosis

!Amazon 112-label saturation curves（私有原始证据，不随公开仓库分发）

The figure preserves every measured point without smoothing. The empirical plateau criterion is the earliest sampled Train size from which every later point remains at least 95% of that metric's 112K endpoint; reaching the threshold only at 112K is treated as **not observed**, rather than as evidence of saturation.

| Model | Ranking plateau onset | Accuracy at 80K / 112K | Accuracy change, 80K→112K | Overall diagnosis |
|---|---:|---:|---:|---|
| Qwen3.5-9B | ≈80K across all four ranking metrics | 70.3% | +5.55 pp | Ranking is near a plateau; Accuracy is not saturated |
| Llama-3.1-8B-Instruct | ≈80K across all four ranking metrics | 73.8% | +4.26 pp | Ranking is near a plateau; Accuracy is not saturated |

For Qwen, the individual 95%-of-endpoint onsets are 60K for Recall@10 and nDCG@10, and 80K for Recall@20 and nDCG@20. For Llama, they are 40K for Recall@10 and nDCG@10, and 80K for Recall@20 and nDCG@20. Therefore, **80K is the shared conservative onset of the ranking plateau**, because all four ranking metrics for both models satisfy the criterion from that point onward.

This does **not** support the stronger conclusion that either model is fully saturated. Accuracy continues to rise materially after 80K for both models and reaches its maximum at 112K. Because each point is a single run, the marked region is an empirical plateau under the stated threshold, not a confidence-interval-supported statistical breakpoint.

Reproduce the PNG, vector PDF, and diagnostic CSV with:

```bash
python crossinterest.experiments.data_scaling/visualize_saturation.py
```

## Progress

| Scope | Complete results | Current state |
|---|---:|---|
| Newly submitted 5K-100K points | 14 / 14 | All passed |
| Reused 112K endpoints | 2 / 2 | Complete |
| Combined 112-label matrix | 16 / 16 | Complete |