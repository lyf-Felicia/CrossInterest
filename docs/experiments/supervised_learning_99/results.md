# Amazon 99-Label SFT Evaluation Results

> One table covers both output-label modes. All results evaluate post-SFT adapters trained for one epoch on all 396,000 candidate-free training rows (99 labels x 4,000 rows). Empty cells mean the full run has not produced a complete `metrics.json`; all non-empty cells are completed results.
> Classification metric: Macro Accuracy (%), higher is better. The prompt provides 10 paired candidates and decoding is candidate-constrained.
> Ranking metrics: Macro Recall@10/20 and Macro nDCG@10/20 on a 0–1 scale, higher is better. The prompt provides no candidates and decoding has no label constraint. Category outputs are matched against 99 category names; ID outputs are matched against the frozen `<interest_000>`–`<interest_098>` mapping and converted back to categories before scoring.
> All metrics are macro-averaged over the same 99 target classes in a frozen, stratified 2,000-sample Dev split per view. No confidence intervals or Wilson statistics are reported.
> Category output uses prompt schema v3 and is complete (6/6 classification, 6/6 ranking). Interest-ID output uses prompt schema v4-interest-id and has not been submitted.

## Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Output label</th>
      <th colspan="3">Macro Accuracy (%) ↑</th>
      <th colspan="3">Macro Recall@10 ↑</th>
      <th colspan="3">Macro Recall@20 ↑</th>
      <th colspan="3">Macro nDCG@10 ↑</th>
      <th colspan="3">Macro nDCG@20 ↑</th>
    </tr>
    <tr>
      <th>Behavior</th>
      <th>Interest</th>
      <th>B+I</th>
      <th>Behavior</th>
      <th>Interest</th>
      <th>B+I</th>
      <th>Behavior</th>
      <th>Interest</th>
      <th>B+I</th>
      <th>Behavior</th>
      <th>Interest</th>
      <th>B+I</th>
      <th>Behavior</th>
      <th>Interest</th>
      <th>B+I</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen3.5-9B</td>
      <td>Category name</td>
      <td>13.75%</td>
      <td>13.08%</td>
      <td>17.13%</td>
      <td>0.2133</td>
      <td>0.1835</td>
      <td>0.2176</td>
      <td>0.3671</td>
      <td>0.3428</td>
      <td>0.3783</td>
      <td>0.1091</td>
      <td>0.0926</td>
      <td>0.1158</td>
      <td>0.1476</td>
      <td>0.1322</td>
      <td>0.1558</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2">Llama-3.1-8B-Instruct</td>
      <td>Category name</td>
      <td>13.77%</td>
      <td>15.26%</td>
      <td>17.46%</td>
      <td>0.2043</td>
      <td>0.1844</td>
      <td>0.2153</td>
      <td>0.3619</td>
      <td>0.3472</td>
      <td>0.3770</td>
      <td>0.1072</td>
      <td>0.0933</td>
      <td>0.1135</td>
      <td>0.1468</td>
      <td>0.1340</td>
      <td>0.1540</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
