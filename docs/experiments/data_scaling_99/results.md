# Amazon 99-Label Training-Data Saturation Results

> All runs use the B+I input, category-name output, one epoch, effective batch size 32, and the same frozen 2,000-row/99-label evaluation set. Empty cells mean no complete method-valid metrics are available. The 396K endpoint is reused from the complete full-data experiment in `crossinterest.legacy.training_99/RESULTS.md`.
> This is the independent 99-label/396K protocol. Its label space, Train set, and Dev-2K differ from the 112-label experiment in `crossinterest.experiments.data_scaling/RESULTS.md`; values must not be merged into one learning curve or compared as matched points.
> Status snapshot: 2026-09-01. Classification is deterministic 10-beam constrained decoding; ranking is candidate-free direct 20-beam decoding. Accuracy is shown as a percentage; Recall and nDCG are 0-1 values.

<table>
  <thead>
    <tr><th rowspan="2">Model</th><th rowspan="2">Train rows</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
    <tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="12">Qwen3.5-9B</td><td>10K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>20K</td><td>9.30%</td><td>0.1995</td><td>0.3514</td><td>0.0965</td><td>0.1344</td></tr>
    <tr><td>40K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>80K</td><td>11.76%</td><td>0.2114</td><td>0.3729</td><td>0.1070</td><td>0.1475</td></tr>
    <tr><td>120K</td><td>17.61%</td><td>0.2128</td><td>0.3785</td><td>0.1041</td><td>0.1456</td></tr>
    <tr><td>160K</td><td>17.86%</td><td>0.2158</td><td>0.3730</td><td>0.1096</td><td>0.1491</td></tr>
    <tr><td>200K</td><td>17.92%</td><td>0.2057</td><td>0.3720</td><td>0.1063</td><td>0.1478</td></tr>
    <tr><td>240K</td><td>18.26%</td><td>0.2196</td><td>0.3887</td><td>0.1144</td><td>0.1566</td></tr>
    <tr><td>280K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>320K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>360K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>396K (all)</td><td>17.13%</td><td>0.2176</td><td>0.3783</td><td>0.1158</td><td>0.1558</td></tr>
    <tr><td rowspan="12">Llama-3.1-8B-Instruct</td><td>10K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>20K</td><td>12.73%</td><td>0.1813</td><td>0.3332</td><td>0.0885</td><td>0.1265</td></tr>
    <tr><td>40K</td><td>10.50%</td><td>0.1860</td><td>0.3561</td><td>0.0934</td><td>0.1360</td></tr>
    <tr><td>80K</td><td>16.01%</td><td>0.1991</td><td>0.3602</td><td>0.1009</td><td>0.1410</td></tr>
    <tr><td>120K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>160K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>200K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>240K</td><td>16.98%</td><td>0.2182</td><td>0.3949</td><td>0.1159</td><td>0.1601</td></tr>
    <tr><td>280K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>320K</td><td>17.04%</td><td>0.2168</td><td>0.3927</td><td>0.1137</td><td>0.1578</td></tr>
    <tr><td>360K</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>396K (all)</td><td>17.46%</td><td>0.2153</td><td>0.3770</td><td>0.1135</td><td>0.1540</td></tr>
  </tbody>
</table>

## Progress

| Scope | Complete results | Current state |
|---|---:|---|
| Reused 396K endpoints | 2 / 2 | Complete |
| Newly submitted points | 11 / 22 | 11 pass; 7 running; 2 queued; 2 failed |
| Combined 99-label matrix | 13 / 24 | 11 cells remain blank pending complete method-valid metrics |

The repaired Qwen 20K/120K and Llama 240K reruns are now complete; Qwen 160K also passed. The Llama 120K repair rerun remains in progress, and the Feeds Llama 10K repair rerun is now running. Qwen 280K continues on Recall as job `<archived-job-id>`. Qwen 10K/40K are currently failed and remain blank pending artifact inspection or repair. Incomplete cells stay blank until both classification and ranking metrics are complete and contract-valid.
