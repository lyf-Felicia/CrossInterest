# Amazon 112-Label SFT Evaluation Results

> One table covers category-name, single-token Interest-ID, and `[Interest ID] Category name` output modes. All adapters are trained for one epoch on all 112,000 candidate-free training rows (112 labels x 1,000 rows). Empty cells mean no method-valid complete `metrics.json` is available.
> Classification metric: Macro Accuracy (%), higher is better. The prompt provides 10 paired natural-language category candidates and decoding is constrained to their corresponding output labels.
> Ranking metrics: Macro Recall@10/20 and Macro nDCG@10/20 on a 0-1 scale, higher is better. The prompt provides no candidates and decoding has no label constraint. Interest IDs are converted back to categories before scoring.
> All metrics are macro-averaged over the same 112 target classes in a frozen, stratified 2,000-sample Dev split per view. Category output uses prompt schema v3; single-token Interest-ID output uses v7-single-interest-id; `[Interest ID] Category name` uses v8-interest-id-category. Both Interest-ID modes add exactly 112 atomic `<interest_NNN>` tokens, one per unique child label; the embeddings are initialized semantically from child-label text and do not encode the parent label. The valid category source is `amazon-sft-112-category-20260827-134729`; the semantically initialized Interest-ID source is `amazon-sft-112-interest-id-semantic-v1-20260828-104049`; the composite-output source is `amazon-sft-112-id-category-recall-20260831-105059`. Cells are populated only when both classification and ranking metrics pass the completeness contract.
> All six composite-output runs are complete. Classification strict-format rate is 1.0 throughout. Candidate-free ranking strict-beam-format rate is 0.7768/0.7750/0.7735 for Qwen Behavior/Interest/B+I and 0.7794/0.7742/0.7763 for Llama Behavior/Interest/B+I; duplicate-valid-beam rate is 0 throughout, and invalid free-generation beams remain invalid rather than being repaired.
> `amazon-sft-112-interest-id-v2-20260827-133837` is invalidated: it assigned opaque `<interest_NNN>` strings by hash order and did not add semantic tokens. Its metrics are excluded from this table regardless of run completeness.
> The 99-label and 112-label tables are not directly comparable: the latter changes both label support and per-label Train quota (4,000 to 1,000), and only 519/2,000 evaluation samples retain exactly the same source and candidate list. See `EXPERIMENT_LOG.md` for the paired audit.
> The strict no-training control for category-name Macro Accuracy is `../crossinterest.experiments.no_training/RESULTS.md`, using the exact same Dev-2K files and classification code path. All six Qwen/Llama control jobs in `amazon-notrain-112k-v2-20260827-1940` are complete.

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
      <td rowspan="3">Qwen3.5-9B</td>
      <td>Category name</td>
      <td>8.95%</td><td>12.59%</td><td>18.68%</td>
      <td>0.1841</td><td>0.1714</td><td>0.2059</td>
      <td>0.3280</td><td>0.3098</td><td>0.3632</td>
      <td>0.0888</td><td>0.0833</td><td>0.1005</td>
      <td>0.1250</td><td>0.1181</td><td>0.1397</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td>14.22%</td><td>12.16%</td><td>14.84%</td>
      <td>0.2130</td><td>0.1904</td><td>0.2287</td>
      <td>0.3438</td><td>0.3180</td><td>0.3725</td>
      <td>0.1068</td><td>0.0987</td><td>0.1174</td>
      <td>0.1396</td><td>0.1310</td><td>0.1536</td>
    </tr>
    <tr>
      <td>[Interest ID] Category name</td>
      <td>13.97%</td><td>12.23%</td><td>16.08%</td>
      <td>0.2003</td><td>0.1768</td><td>0.2025</td>
      <td>0.2756</td><td>0.2649</td><td>0.2878</td>
      <td>0.0977</td><td>0.0826</td><td>0.1000</td>
      <td>0.1175</td><td>0.1057</td><td>0.1222</td>
    </tr>
    <tr>
      <td rowspan="3">Llama-3.1-8B-Instruct</td>
      <td>Category name</td>
      <td>13.03%</td><td>15.24%</td><td>16.23%</td>
      <td>0.1874</td><td>0.1739</td><td>0.1923</td>
      <td>0.3210</td><td>0.3093</td><td>0.3512</td>
      <td>0.0948</td><td>0.0887</td><td>0.0973</td>
      <td>0.1283</td><td>0.1227</td><td>0.1375</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td>12.99%</td><td>12.33%</td><td>13.92%</td>
      <td>0.2066</td><td>0.1909</td><td>0.2217</td>
      <td>0.3428</td><td>0.3144</td><td>0.3666</td>
      <td>0.1035</td><td>0.0968</td><td>0.1147</td>
      <td>0.1378</td><td>0.1279</td><td>0.1511</td>
    </tr>
    <tr>
      <td>[Interest ID] Category name</td>
      <td>11.56%</td><td>13.28%</td><td>15.01%</td>
      <td>0.1961</td><td>0.1829</td><td>0.2081</td>
      <td>0.2706</td><td>0.2552</td><td>0.2943</td>
      <td>0.0934</td><td>0.0874</td><td>0.1028</td>
      <td>0.1128</td><td>0.1063</td><td>0.1254</td>
    </tr>
  </tbody>
</table>
