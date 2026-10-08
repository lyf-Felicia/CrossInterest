# Google Baseline 与 Interest ID 对比

> 数据集：Home & Kitchen，112 个预测类别。Interest ID 使用当前表现最佳的 Behavior + Interest 输入。
> 指标：类别宏平均 Accuracy、Recall@10/20 与 nDCG@10/20，均为越高越好。Accuracy 使用百分数，其余指标使用 0--1 小数。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>方法</th>
      <th>Macro Accuracy (%) ↑</th>
      <th>Macro Recall@10 ↑</th>
      <th>Macro Recall@20 ↑</th>
      <th>Macro nDCG@10 ↑</th>
      <th>Macro nDCG@20 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen3.5-9B</td>
      <td>Google</td>
      <td>11.248%</td>
      <td>0.1350</td>
      <td>0.2271</td>
      <td>0.0658</td>
      <td>0.0891</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td><strong>14.84%</strong></td>
      <td><strong>0.2287</strong></td>
      <td><strong>0.3725</strong></td>
      <td><strong>0.1174</strong></td>
      <td><strong>0.1536</strong></td>
    </tr>
    <tr>
      <td rowspan="2">Llama-3.1-8B-Instruct</td>
      <td>Google</td>
      <td>11.394%</td>
      <td>0.1373</td>
      <td>0.2213</td>
      <td>0.0669</td>
      <td>0.0881</td>
    </tr>
    <tr>
      <td>Interest ID</td>
      <td><strong>13.92%</strong></td>
      <td><strong>0.2217</strong></td>
      <td><strong>0.3666</strong></td>
      <td><strong>0.1147</strong></td>
      <td><strong>0.1511</strong></td>
    </tr>
  </tbody>
</table>

## 相对 Google 的提升

| 模型 | Accuracy（百分点） | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | +3.592 | +0.0937 | +0.1454 | +0.0516 | +0.0645 |
| Llama-3.1-8B-Instruct | +2.526 | +0.0844 | +0.1453 | +0.0478 | +0.0630 |
