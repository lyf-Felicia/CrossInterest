# MeRA 论文结果、复现结果与 Interest ID 对比

> 评测日期：2026-09-08。Recall、Precision 与 nDCG 均为 0--1 小数；所有箭头均表示越高越好。
> 空白表示复现结果尚未产出；“—”表示论文未报告。Home 表中的粗体表示同一模型、同一冻结评测协议下的最优方法。

## 发布代码 strict 历史诊断（非作者修正结果）

本节保留修正前按发布代码训练协议得到的历史 strict 结果，只用于定位公开代码与论文表值差异，不作为作者澄清协议下的最终复现结果。

<table>
  <thead>
    <tr>
      <th rowspan="2">数据集</th>
      <th rowspan="2">模型</th>
      <th rowspan="2">结果</th>
      <th colspan="3">Recall ↑</th>
      <th colspan="3">Precision ↑</th>
    </tr>
    <tr>
      <th>R@1</th><th>R@2</th><th>R@3</th>
      <th>P@1</th><th>P@2</th><th>P@3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Arts</td>
      <td rowspan="2">Qwen2-1.5B</td>
      <td>论文结果</td>
      <td>0.1204</td><td>0.2120</td><td>0.2786</td>
      <td>0.2896</td><td>0.2670</td><td>0.2413</td>
    </tr>
    <tr>
      <td>复现结果</td>
      <td>0.0786</td><td>0.1485</td><td>0.2046</td>
      <td>0.2021</td><td>0.1882</td><td>0.1783</td>
    </tr>
    <tr>
      <td rowspan="2">Grocery</td>
      <td rowspan="2">Qwen2-1.5B</td>
      <td>论文结果</td>
      <td>0.1179</td><td>0.1980</td><td>0.2662</td>
      <td>0.3206</td><td>0.2734</td><td>0.2515</td>
    </tr>
    <tr>
      <td>复现结果</td>
      <td>0.0503</td><td>0.0933</td><td>0.1406</td>
      <td>0.1180</td><td>0.1108</td><td>0.1126</td>
    </tr>
  </tbody>
</table>

## 作者澄清协议正式结果

训练固定 5 epochs，不使用 early stopping 或 best-checkpoint 回载；训练序列为连续双引号历史类别 token 加 gold 类别。Arts/Grocery 使用完整官方标签空间。Qwen2 Arts 的 r4 只读恢复评测已通过完整排名与双实现指标一致性验收；Grocery 的空白单元格表示指标尚未产出，不代表零分。

<table>
  <thead>
    <tr>
      <th rowspan="2">数据集</th>
      <th rowspan="2">模型</th>
      <th rowspan="2">结果</th>
      <th colspan="3">Recall ↑</th>
      <th colspan="3">Precision ↑</th>
    </tr>
    <tr>
      <th>R@1</th><th>R@2</th><th>R@3</th>
      <th>P@1</th><th>P@2</th><th>P@3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Arts</td>
      <td rowspan="2">Qwen2-1.5B</td>
      <td>论文结果</td>
      <td>0.1204</td><td>0.2120</td><td>0.2786</td>
      <td>0.2896</td><td>0.2670</td><td>0.2413</td>
    </tr>
    <tr>
      <td>修改后复现结果</td>
      <td>0.1180</td><td>0.2175</td><td>0.2796</td>
      <td>0.2876</td><td>0.2686</td><td>0.2376</td>
    </tr>
    <tr>
      <td>Qwen3.5-9B</td>
      <td>修改后复现结果</td>
      <td>0.1033</td><td>0.1753</td><td>0.2334</td>
      <td>0.2575</td><td>0.2302</td><td>0.2056</td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>修改后复现结果</td>
      <td>0.1209</td><td>0.2077</td><td>0.2752</td>
      <td>0.2908</td><td>0.2581</td><td>0.2322</td>
    </tr>
    <tr>
      <td rowspan="4">Grocery</td>
      <td rowspan="2">Qwen2-1.5B</td>
      <td>论文结果</td>
      <td>0.1179</td><td>0.1980</td><td>0.2662</td>
      <td>0.3206</td><td>0.2734</td><td>0.2515</td>
    </tr>
    <tr>
      <td>修改后复现结果</td>
      <td></td><td></td><td></td><td></td><td></td><td></td>
    </tr>
    <tr>
      <td>Qwen3.5-9B</td>
      <td>修改后复现结果</td>
      <td>0.0964</td><td>0.1720</td><td>0.2299</td>
      <td>0.2646</td><td>0.2433</td><td>0.2220</td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>修改后复现结果</td>
      <td>0.0982</td><td>0.1808</td><td>0.2503</td>
      <td>0.2778</td><td>0.2561</td><td>0.2387</td>
    </tr>
  </tbody>
</table>

只有 Qwen2-1.5B 与论文使用同一基础模型，构成直接复现对照；Qwen3.5/Llama 是作者协议下的基础模型替换扩展。作者回复另给出的 Arts Qwen2 参考值为 R@1/2/3 `0.1209/0.2149/0.2843`、P@1/2/3 `0.2987/0.2712/0.2415`。

> Qwen2 Arts 来源：`mera-author-qwen2-arts-eval-recovery-r4`，Backend ID `<archived-job-id>`，状态 `pass`；评测只读复用 400/400 step 的最终 adapter，覆盖 2,524 个 source 和完整 143 类排名，项目指标与官方循环公式逐项一致。Qwen3.5/Llama Arts/Grocery 与 Llama Home 来源：`mera-author-three-dataset-formal-r2-20260907`，六项均为 `pass`。Qwen3.5 Home 来源：先完成完整验收的 `mera-author-qwen35-home-fastlog-r1-20260907`；其计算仅比主任务少输出重复的 bitsandbytes warning。相同合同的主任务副本得到 13.76% Macro Accuracy，差异作为四卡 8-bit 训练非 bitwise deterministic 的稳定性限制保留。

## Home & Kitchen：MeRA vs Our Method

公平比较固定使用相同的 112K Train、冻结 Dev-2K Test 和 112 个预测类。MeRA 使用作者澄清协议的完整 `both` 条件，先生成 136 类完整排名，再确定性过滤到 112 类；Our Method 使用 **Behavior + Interest（B+I）历史输入**和语义初始化的单 token Interest ID 输出。B+I 是 Our Method 在 Behavior、Interest、B+I 三种输入视图中的正式最优版本：两个模型的四项排名指标均为三视图最高。所有指标均按 112 个目标类宏平均，粗体表示同一基础模型、同一指标下两种方法的较高值。

<table>
  <thead>
    <tr>
      <th rowspan="2">基础模型</th>
      <th rowspan="2">方法</th>
      <th rowspan="2">训练条件</th>
      <th colspan="2">Macro Recall ↑</th>
      <th colspan="2">Macro nDCG ↑</th>
    </tr>
    <tr>
      <th>@10</th><th>@20</th><th>@10</th><th>@20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen3.5-9B</td>
      <td>MeRA</td>
      <td>作者澄清协议 <code>both</code></td>
      <td>0.1944</td><td>0.3221</td><td>0.0988</td><td>0.1307</td>
    </tr>
    <tr>
      <td>Our Method</td>
      <td>B+I → Semantic Interest ID</td>
      <td><strong>0.2287</strong></td><td><strong>0.3725</strong></td><td><strong>0.1174</strong></td><td><strong>0.1536</strong></td>
    </tr>
    <tr>
      <td rowspan="2">Llama-3.1-8B-Instruct</td>
      <td>MeRA</td>
      <td>作者澄清协议 <code>both</code></td>
      <td>0.1889</td><td>0.3264</td><td>0.0964</td><td>0.1306</td>
    </tr>
    <tr>
      <td>Our Method</td>
      <td>B+I → Semantic Interest ID</td>
      <td><strong>0.2217</strong></td><td><strong>0.3666</strong></td><td><strong>0.1147</strong></td><td><strong>0.1511</strong></td>
    </tr>
  </tbody>
</table>

结论：使用正确的 B+I 最优版本后，Our Method 在 8 个“模型 × 指标”点估计中全部高于 MeRA。Qwen 的 Our - MeRA 差值依次为 `+0.0343/+0.0504/+0.0186/+0.0228`；Llama 为 `+0.0328/+0.0401/+0.0183/+0.0205`。该结果没有置信区间或显著性检验，因此只能说明当前固定运行的点估计优势，不能直接外推为总体显著优势。

> MeRA Qwen 来源：`mera-author-qwen35-home-fastlog-r1-20260907:qwen35-home-author-clarified`；MeRA Llama 来源：`mera-author-three-dataset-formal-r2-20260907:llama31-home-author-clarified`。Our Method 来源：`amazon-sft-112-interest-id-semantic-v1-20260828-104049` 的 Qwen/Llama `behavior-interest-interest-id-semantic-init` 作业，后端 ID 分别为 `<archived-job-id>` / `<archived-job-id>`，两项均为 `pass`。四份 ranking metrics 均验收为 2,000 条、112 类；Our Method 两份均为 `complete=true`、`view=behavior_interest`、`label_mode=interest_id`。两方法训练目标和类别表示不同，因此差值只比较最终方法表现，不解释为单一组件的因果增益。