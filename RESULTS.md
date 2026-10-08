# 正式实验结果汇总

> 整理与来源核对日期：2026-10-08；本次未重新运行训练或评测。各实验实际评测日期以链接的分支记录为准。
> 单位：Macro Accuracy为百分数（%）；Recall、Precision、nDCG为0–1小数。↑表示越高越好。空白表示尚未产出，不能按零分解释。
> 主协议：Amazon Reviews 2023 Home & Kitchen，112K Train、112类、冻结Dev2000，类别宏平均；排序为无候选、无标签约束的确定性20-beam。候选分类、99类历史协议和MeRA官方数据集分别列示。
> 粗体仅表示同一表内、同一模型、同轮可比条件下的最高点估计，不表示统计显著。不同轮次的同名SFT或G0不能互换。

本文件是统一结果入口，按[研究PPT](https://docs.google.com/presentation/d/14JOumukKLuqFPzMQWnuxFqTMGTQAO6HjlgrYWsEgzOI/edit)的实验问题组织；详细方法、合同、运行预算和审计仍由各分支报告承担。任务与研究叙述见[实验总览](docs/experiments/overview.md)。

导航：[零样本与SFT](#1-零样本与标准sft) · [外部比较](#2-外部方法比较) · [数据规模](#3-训练数据规模ppt第2435页) · [训练组织与骨干](#4-训练组织与骨干) · [CIR](#5-cir与语义方向控制ppt第3032页) · [COGS](#6-去噪与数据效率cogs) · [结构扩展](#7-机制与结构扩展包含负结果) · [历史附录](#8-历史协议与诊断附录) · [覆盖与状态](#10-覆盖与状态)

范围为当前可核对的完整正式结果；smoke、失败重试、吞吐测试、事后分组诊断和未执行提案不当作独立正式方法。复用baseline和跨表重复展示的参考行不重复计数。

**标签约定：** 表头沿用PPT的`Model / Method / Input / Output / Size`，Macro Recall与Macro nDCG分别合并为指标组。`Our Method`指Behavior + Interest输入、语义初始化的单token Interest ID输出；`Interest ID with warm-up`指名称↔ID预热后双输出多任务SFT，不等同于CIR预热。`N only`是仅新兴趣监督，`O → N`是旧兴趣后新兴趣，`O + N shuffle`是同一并集混合。表内`Llama-3.1-8B`均为`Llama-3.1-8B-Instruct`；Qwen2.5规模表均为Instruct模型。

每张表以**Finding**概括当前结果，训练预算、来源及解释边界留在表后。沿用PPT的表达方式，不复制其未经同口径核实的数值或更强结论。

## 1. 零样本与标准SFT

### 1.1 不训练时LLM能否预测新兴趣？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Reasoning</th><th colspan="3">Macro Accuracy (%) ↑</th></tr>
<tr><th>Behavior</th><th>Interest</th><th>Behavior + Interest</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">GPT-5.6 Sol</td><td>Medium</td><td>13.54</td><td>11.23</td><td>16.17</td></tr>
<tr><td>None</td><td>12.56</td><td>12.09</td><td><strong>16.60</strong></td></tr>
<tr><td rowspan="2">GPT-5</td><td>Medium</td><td>10.30</td><td>10.48</td><td>13.92</td></tr>
<tr><td>None</td><td>6.98</td><td>11.94</td><td>14.09</td></tr>
<tr><td rowspan="2">Gemini 3.6 Flash</td><td>Medium</td><td><strong>14.04</strong></td><td>10.25</td><td>15.05</td></tr>
<tr><td>Minimal</td><td>12.68</td><td>12.29</td><td>15.55</td></tr>
<tr><td>Qwen3.5-9B</td><td>—</td><td>12.79</td><td><strong>12.79</strong></td><td>12.79</td></tr>
<tr><td>Llama-3.1-8B</td><td>—</td><td>10.06</td><td>11.43</td><td>10.90</td></tr>
<tr><td>Random baseline</td><td>—</td><td>10.00</td><td>10.00</td><td>10.00</td></tr>
</tbody>
</table>

**Finding：** Home112的10候选任务中，多数模型的Behavior + Interest点估计高于单视图；这不是无候选Top-K排序结果。

来源：[Home112匹配零样本](docs/experiments/zero_shot_home112/results.md)。冻结样本、候选和分类代码与112类SFT匹配，不代表无候选生成能力。Qwen三格仅在两位小数下相同，来源记录已核对不同输入合同和不同预测；没有在本次汇总重跑其复测。

### 1.2 Interest ID与输入表示如何影响新兴趣预测？（PPT第20、36–39页）

18个配置均为112K、1 epoch；三种输出统一标记为`Category name`、`Interest ID`、`[Interest ID] Category name`，输入为Behavior、Interest、Behavior + Interest。表内粗体为每个模型跨本表配置的最高值。

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Output</th><th rowspan="2">Input</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="9">Qwen3.5-9B</td><td rowspan="3">Category name</td><td>Behavior</td><td>8.95</td><td>0.1841</td><td>0.3280</td><td>0.0888</td><td>0.1250</td></tr>
<tr><td>Interest</td><td>12.59</td><td>0.1714</td><td>0.3098</td><td>0.0833</td><td>0.1181</td></tr>
<tr><td>Behavior + Interest</td><td><strong>18.68</strong></td><td>0.2059</td><td>0.3632</td><td>0.1005</td><td>0.1397</td></tr>
<tr><td rowspan="3">Interest ID</td><td>Behavior</td><td>14.22</td><td>0.2130</td><td>0.3438</td><td>0.1068</td><td>0.1396</td></tr>
<tr><td>Interest</td><td>12.16</td><td>0.1904</td><td>0.3180</td><td>0.0987</td><td>0.1310</td></tr>
<tr><td>Behavior + Interest</td><td>14.84</td><td><strong>0.2287</strong></td><td><strong>0.3725</strong></td><td><strong>0.1174</strong></td><td><strong>0.1536</strong></td></tr>
<tr><td rowspan="3">[Interest ID] Category name</td><td>Behavior</td><td>13.97</td><td>0.2003</td><td>0.2756</td><td>0.0977</td><td>0.1175</td></tr>
<tr><td>Interest</td><td>12.23</td><td>0.1768</td><td>0.2649</td><td>0.0826</td><td>0.1057</td></tr>
<tr><td>Behavior + Interest</td><td>16.08</td><td>0.2025</td><td>0.2878</td><td>0.1000</td><td>0.1222</td></tr>
<tr><td rowspan="9">Llama-3.1-8B</td><td rowspan="3">Category name</td><td>Behavior</td><td>13.03</td><td>0.1874</td><td>0.3210</td><td>0.0948</td><td>0.1283</td></tr>
<tr><td>Interest</td><td>15.24</td><td>0.1739</td><td>0.3093</td><td>0.0887</td><td>0.1227</td></tr>
<tr><td>Behavior + Interest</td><td><strong>16.23</strong></td><td>0.1923</td><td>0.3512</td><td>0.0973</td><td>0.1375</td></tr>
<tr><td rowspan="3">Interest ID</td><td>Behavior</td><td>12.99</td><td>0.2066</td><td>0.3428</td><td>0.1035</td><td>0.1378</td></tr>
<tr><td>Interest</td><td>12.33</td><td>0.1909</td><td>0.3144</td><td>0.0968</td><td>0.1279</td></tr>
<tr><td>Behavior + Interest</td><td>13.92</td><td><strong>0.2217</strong></td><td><strong>0.3666</strong></td><td><strong>0.1147</strong></td><td><strong>0.1511</strong></td></tr>
<tr><td rowspan="3">[Interest ID] Category name</td><td>Behavior</td><td>11.56</td><td>0.1961</td><td>0.2706</td><td>0.0934</td><td>0.1128</td></tr>
<tr><td>Interest</td><td>13.28</td><td>0.1829</td><td>0.2552</td><td>0.0874</td><td>0.1063</td></tr>
<tr><td>Behavior + Interest</td><td>15.01</td><td>0.2081</td><td>0.2943</td><td>0.1028</td><td>0.1254</td></tr>
</tbody>
</table>

**Finding：** Behavior + Interest → Interest ID在两个模型的四项排序指标上均为本表最高；Accuracy最高的是Behavior + Interest → Category name。追加类别名输出未进一步改善排序。

来源：[112类SFT正式矩阵](docs/experiments/supervised_learning/results.md)。单token ID提升本表排序点估计，但不能概括为全面提升候选Accuracy。复合输出的自由排序strict-format约0.7735–0.7794，无效beam不修复；不把它与strict=1的实验当作健康度相同。未语义初始化的hash-order伪ID版本已作废，不纳入。

## 2. 外部方法比较

### 2.1 Our Method相比Google表现如何？（PPT第21页）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>Google</td><td>11.248</td><td>0.1350</td><td>0.2271</td><td>0.0658</td><td>0.0891</td></tr>
<tr><td>Our Method</td><td><strong>14.84</strong></td><td><strong>0.2287</strong></td><td><strong>0.3725</strong></td><td><strong>0.1174</strong></td><td><strong>0.1536</strong></td></tr>
<tr><td rowspan="2">Llama-3.1-8B</td><td>Google</td><td>11.394</td><td>0.1373</td><td>0.2213</td><td>0.0669</td><td>0.0881</td></tr>
<tr><td>Our Method</td><td><strong>13.92</strong></td><td><strong>0.2217</strong></td><td><strong>0.3666</strong></td><td><strong>0.1147</strong></td><td><strong>0.1511</strong></td></tr>
</tbody>
</table>

**Finding：** Our Method在两个模型的五项点估计上均高于Google。这里的Our Method统一指Behavior + Interest输入、语义初始化的单token Interest ID输出。

来源：[Google正式结果](docs/experiments/google_baseline/results.md)。Qwen Google的nDCG@10在PPT写0.0656，本地正式表写0.0658；本汇总采用来源表值，保留此差异，不声称此次已重算原始排名。两方法的输入和训练方法不同，不能将整套方法差值归因于ID单组件。

### 2.2 Our Method相比MeRA表现如何？（PPT第23页）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>MeRA</td><td>0.1944</td><td>0.3221</td><td>0.0988</td><td>0.1307</td></tr>
<tr><td>Our Method</td><td><strong>0.2287</strong></td><td><strong>0.3725</strong></td><td><strong>0.1174</strong></td><td><strong>0.1536</strong></td></tr>
<tr><td rowspan="2">Llama-3.1-8B</td><td>MeRA</td><td>0.1889</td><td>0.3264</td><td>0.0964</td><td>0.1306</td></tr>
<tr><td>Our Method</td><td><strong>0.2217</strong></td><td><strong>0.3666</strong></td><td><strong>0.1147</strong></td><td><strong>0.1511</strong></td></tr>
</tbody>
</table>

**Finding：** Our Method在两个模型的四项Macro排序点估计上均高于MeRA；尚无独立显著性检验，不写作总体显著优势。

来源：[MeRA正式比较](docs/experiments/mera/results.md)。MeRA使用作者澄清的`both`训练协议，136类完整排序后过滤到112类；B+I方法直接生成Interest ID。同数据和最终指标可比，但训练目标不同，不作单组件因果归因。

### 2.3 MeRA能否在官方数据集上复现？（PPT第22页）

本表是官方数据集的Recall/Precision，不是Home112宏平均排序。作者澄清协议为5 epochs、最终checkpoint，不使用early stopping。

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th rowspan="2">Model</th><th rowspan="2">Result</th><th colspan="3">Recall ↑</th><th colspan="3">Precision ↑</th></tr>
<tr><th>@1</th><th>@2</th><th>@3</th><th>@1</th><th>@2</th><th>@3</th></tr>
</thead>
<tbody>
<tr><td rowspan="4">Arts</td><td rowspan="2">Qwen2-1.5B</td><td>论文结果</td><td>0.1204</td><td>0.2120</td><td>0.2786</td><td>0.2896</td><td>0.2670</td><td>0.2413</td></tr>
<tr><td>同模型复现结果</td><td>0.1180</td><td>0.2175</td><td>0.2796</td><td>0.2876</td><td>0.2686</td><td>0.2376</td></tr>
<tr><td>Qwen3.5-9B</td><td>模型替换结果</td><td>0.1033</td><td>0.1753</td><td>0.2334</td><td>0.2575</td><td>0.2302</td><td>0.2056</td></tr>
<tr><td>Llama-3.1-8B</td><td>模型替换结果</td><td>0.1209</td><td>0.2077</td><td>0.2752</td><td>0.2908</td><td>0.2581</td><td>0.2322</td></tr>
<tr><td rowspan="4">Grocery</td><td rowspan="2">Qwen2-1.5B</td><td>论文结果</td><td>0.1179</td><td>0.1980</td><td>0.2662</td><td>0.3206</td><td>0.2734</td><td>0.2515</td></tr>
<tr><td>同模型复现结果</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>Qwen3.5-9B</td><td>模型替换结果</td><td>0.0964</td><td>0.1720</td><td>0.2299</td><td>0.2646</td><td>0.2433</td><td>0.2220</td></tr>
<tr><td>Llama-3.1-8B</td><td>模型替换结果</td><td>0.0982</td><td>0.1808</td><td>0.2503</td><td>0.2778</td><td>0.2561</td><td>0.2387</td></tr>
</tbody>
</table>

**Finding：** 作者澄清协议下的Qwen2 Arts复现与论文接近；Grocery同模型结果仍空缺。Qwen3.5和Llama属于模型替换，不能代替同模型复现结论。

来源：[MeRA作者协议正式结果](docs/experiments/mera/results.md)。PPT的Arts 0.1209/Grocery 0.0982属于Llama替换行；只有Qwen2行是同模型直接复现。Qwen2 Grocery尚无来源表中的完整正式指标，留空。修正前发布代码strict诊断另列历史附录，不与作者协议结果合并。论文值是参考文献结果，不计入本项目完成作业数；模型替换不属于同模型消融，因此本表不标跨模型最优。

## 3. 训练数据规模（PPT第24、35页）

本实验使用B+I → Category name，而非语义ID；1 epoch、有效batch32、Dev2000/112类固定。粗体仅表示同一模型各采样点的最高值。

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Train size</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="8">Qwen3.5-9B</td><td>5K</td><td>7.38</td><td>0.1511</td><td>0.2733</td><td>0.0750</td><td>0.1056</td></tr>
<tr><td>10K</td><td>6.75</td><td>0.1635</td><td>0.2977</td><td>0.0778</td><td>0.1114</td></tr>
<tr><td>20K</td><td>9.71</td><td>0.1876</td><td>0.3206</td><td>0.0891</td><td>0.1223</td></tr>
<tr><td>40K</td><td>8.91</td><td>0.1836</td><td>0.3330</td><td>0.0925</td><td>0.1300</td></tr>
<tr><td>60K</td><td>14.09</td><td>0.1998</td><td>0.3335</td><td>0.0980</td><td>0.1315</td></tr>
<tr><td>80K</td><td>13.13</td><td><strong>0.2064</strong></td><td>0.3578</td><td><strong>0.1042</strong></td><td><strong>0.1420</strong></td></tr>
<tr><td>100K</td><td>17.36</td><td>0.1974</td><td>0.3476</td><td>0.0985</td><td>0.1361</td></tr>
<tr><td>112K</td><td><strong>18.68</strong></td><td>0.2059</td><td><strong>0.3632</strong></td><td>0.1005</td><td>0.1397</td></tr>
<tr><td rowspan="8">Llama-3.1-8B</td><td>5K</td><td>7.74</td><td>0.1366</td><td>0.2639</td><td>0.0660</td><td>0.0979</td></tr>
<tr><td>10K</td><td>8.39</td><td>0.1479</td><td>0.2841</td><td>0.0721</td><td>0.1062</td></tr>
<tr><td>20K</td><td>9.99</td><td>0.1697</td><td>0.3212</td><td>0.0816</td><td>0.1193</td></tr>
<tr><td>40K</td><td>12.87</td><td>0.1884</td><td>0.3318</td><td>0.0929</td><td>0.1287</td></tr>
<tr><td>60K</td><td>11.54</td><td>0.1827</td><td>0.3250</td><td>0.0925</td><td>0.1281</td></tr>
<tr><td>80K</td><td>11.97</td><td><strong>0.1950</strong></td><td>0.3404</td><td>0.0967</td><td>0.1330</td></tr>
<tr><td>100K</td><td>13.91</td><td>0.1846</td><td>0.3428</td><td>0.0937</td><td>0.1333</td></tr>
<tr><td>112K</td><td><strong>16.23</strong></td><td>0.1923</td><td><strong>0.3512</strong></td><td><strong>0.0973</strong></td><td><strong>0.1375</strong></td></tr>
</tbody>
</table>

**Finding：** 排序指标约80K进入经验平台，但Macro Accuracy到112K仍有提升；“排序趋稳”不等于“全部性能饱和”。

来源：[112类数据规模正式结果](docs/experiments/data_scaling/results.md)。14个新作业和2个复用端点完整。按“所有后续点不低于112K端点95%”的经验标准，四项排序指标共同保守平台约80K；Accuracy在80K后仍增加，不能写作全部能力已饱和。

## 4. 训练组织与骨干

### 4.1 SID warm-up能否改善Our Method？（PPT第26页）

<table>
<thead>
<tr><th rowspan="2">Input</th><th rowspan="2">Model</th><th rowspan="2">Method</th><th rowspan="2">Steps</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="4">Interest</td><td rowspan="2">Qwen3.5-9B</td><td>Interest ID</td><td>3,500</td><td>0.1904</td><td>0.3180</td><td><strong>0.0987</strong></td><td>0.1310</td></tr>
<tr><td>Interest ID with warm-up</td><td>7,070</td><td><strong>0.1977</strong></td><td><strong>0.3325</strong></td><td>0.0983</td><td><strong>0.1321</strong></td></tr>
<tr><td rowspan="2">Llama-3.1-8B</td><td>Interest ID</td><td>3,500</td><td>0.1909</td><td>0.3144</td><td>0.0968</td><td>0.1279</td></tr>
<tr><td>Interest ID with warm-up</td><td>7,070</td><td><strong>0.1956</strong></td><td><strong>0.3233</strong></td><td><strong>0.0971</strong></td><td><strong>0.1293</strong></td></tr>
<tr><td rowspan="4">Behavior + Interest</td><td rowspan="2">Qwen3.5-9B</td><td>Interest ID</td><td>3,500</td><td><strong>0.2287</strong></td><td><strong>0.3725</strong></td><td><strong>0.1174</strong></td><td><strong>0.1536</strong></td></tr>
<tr><td>Interest ID with warm-up</td><td>7,070</td><td>0.2213</td><td>0.3685</td><td>0.1152</td><td>0.1523</td></tr>
<tr><td rowspan="2">Llama-3.1-8B</td><td>Interest ID</td><td>3,500</td><td>0.2217</td><td>0.3666</td><td>0.1147</td><td>0.1511</td></tr>
<tr><td>Interest ID with warm-up</td><td>7,070</td><td><strong>0.2230</strong></td><td><strong>0.3757</strong></td><td><strong>0.1170</strong></td><td><strong>0.1554</strong></td></tr>
</tbody>
</table>

**Finding：** Interest-only有小幅召回改善；在Our Method的Behavior + Interest设置下，Qwen下降、Llama上升，SID warm-up没有稳定跨模型收益。

来源：[Interest-only正式报告](docs/experiments/alignment/report.md)、[B+I正式报告](docs/experiments/alignment_behavior_interest/report.md)。224条字典任务预热70步，再224K双输出任务训练7,000步；相对单阶段同时改变预算与类别名称监督，不能归因于预热顺序。粗体按同输入、同模型配对；B+I不呈现稳定跨模型收益。

### 4.2 Cross-fitted能否改善Our Method？（PPT第25页）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>SFT</td><td>0.2242</td><td>0.3685</td><td>0.1135</td><td>0.1498</td></tr>
<tr><td>Cross-fitted</td><td><strong>0.2305</strong></td><td><strong>0.3689</strong></td><td><strong>0.1186</strong></td><td><strong>0.1535</strong></td></tr>
</tbody>
</table>

**Finding：** Cross-fitted四项点估计均上升，但Macro Recall@20近乎持平；收益有限且增加五教师成本。

来源：[Cross-fitted报告](docs/experiments/crossfit/report.md)。五教师各看四折、对未见折评分，学生保留全部112K，3,500步。额外教师成本不计入学生步数；Recall@20近乎持平，不据单seed称显著提高。

### 4.3 模型规模如何影响Our Method？（PPT第27页）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Size</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="6">Qwen2.5</td><td>0.5B</td><td>0.2010</td><td>0.3201</td><td>0.0987</td><td>0.1286</td></tr>
<tr><td>1.5B</td><td>0.2033</td><td>0.3326</td><td>0.1030</td><td>0.1355</td></tr>
<tr><td>3B</td><td>0.2115</td><td>0.3520</td><td>0.1060</td><td>0.1411</td></tr>
<tr><td>7B</td><td>0.2126</td><td>0.3554</td><td><strong>0.1077</strong></td><td>0.1437</td></tr>
<tr><td>14B</td><td><strong>0.2180</strong></td><td><strong>0.3629</strong></td><td>0.1074</td><td><strong>0.1440</strong></td></tr>
<tr><td>32B</td><td>0.1957</td><td>0.3416</td><td>0.0977</td><td>0.1342</td></tr>
</tbody>
</table>

**Finding：** Qwen2.5在本设置下以14B取得三项最高值，7B的Macro nDCG@10略高；32B未进一步改善，规模收益不单调。

来源：[骨干报告](docs/experiments/backbones/report.md)，已逐行核对冻结总设计§3.4的正式矩阵及PPT。六项均为B+I、Semantic ID、112K、3,500步。14B三项最高、7B nDCG@10最高，不能写成所有指标随规模单调提升。

### 4.4 模型版本及追加27B如何影响Our Method？（PPT第28页）

<table>
<thead>
<tr><th rowspan="2">Comparison</th><th rowspan="2">Model</th><th rowspan="2">Size</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Version</td><td>Qwen2.5</td><td>7B</td><td>0.2126</td><td>0.3554</td><td>0.1077</td><td>0.1437</td></tr>
<tr><td>Qwen3</td><td>8B</td><td>0.1945</td><td>0.3305</td><td>0.0986</td><td>0.1328</td></tr>
<tr><td>Qwen3.5</td><td>9B</td><td><strong>0.2242</strong></td><td><strong>0.3685</strong></td><td><strong>0.1135</strong></td><td><strong>0.1498</strong></td></tr>
<tr><td rowspan="2">Size</td><td rowspan="2">Qwen3.5</td><td>9B</td><td><strong>0.2242</strong></td><td>0.3685</td><td>0.1135</td><td>0.1498</td></tr>
<tr><td>27B</td><td>0.2219</td><td><strong>0.3734</strong></td><td><strong>0.1188</strong></td><td><strong>0.1572</strong></td></tr>
</tbody>
</table>

**Finding：** 相近规模组中Qwen3.5-9B四项最高；同版本27B相对9B三项提升、Macro Recall@10下降，不能概括为所有指标都更好。

来源：[骨干报告](docs/experiments/backbones/report.md)、[追加27B独立验收说明](docs/experiments/alignment_behavior_interest/report.md)。原八骨干加27B共九个唯一训练条件，复用行不重复计数。版本组非严格等参数，规模组非等FLOPs；同轮9B不是0.2287的历史SFT。

### 4.5 Old interest warm-up能否改善Our Method？（PPT第29页下表）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th rowspan="2">Training data / Steps</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Qwen3.5-9B</td><td>N only</td><td>N112K / 3,500</td><td>0.224236</td><td><strong>0.368464</strong></td><td>0.113470</td><td>0.149803</td></tr>
<tr><td>O → N</td><td>O46.5K+N112K / 4,954</td><td>0.229925</td><td>0.366363</td><td><strong>0.119087</strong></td><td><strong>0.153422</strong></td></tr>
<tr><td>O + N shuffle</td><td>O46.5K+N112K / 4,954</td><td><strong>0.230013</strong></td><td>0.367063</td><td>0.117649</td><td>0.152238</td></tr>
</tbody>
</table>

**Finding：** 完整新兴趣监督下，O → N和O + N shuffle提高前排排序，却降低Macro Recall@20；没有全面改善，也不足以证明顺序效应。

来源：[课程正式报告](docs/experiments/curriculum/report.md)。O→N与Shuffle同并集同预算，但与N-only不等预算；两课程nDCG升而Recall@20降，不能宣称全面改善。PPT第29页上表使用N93K历史版本，独立列于历史附录，不能合并。

## 5. CIR与语义、方向控制（PPT第30–32页）

### 5.1 CIR warm-up能否改善Our Method？

<table>
<thead>
<tr><th rowspan="2">Experiment</th><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="7">CIR warm-up</td><td rowspan="4">Qwen3.5-9B</td><td>N only</td><td>0.224236</td><td>0.368464</td><td>0.113470</td><td>0.149803</td></tr>
<tr><td>CIR → N SFT</td><td>0.240780</td><td>0.381215</td><td>0.124903</td><td>0.160060</td></tr>
<tr><td>Balanced CIR → N SFT</td><td>0.235994</td><td>0.377801</td><td>0.121613</td><td>0.157133</td></tr>
<tr><td>Balanced CIR → N SFT (3 epochs)</td><td>0.251255</td><td>0.401786</td><td>0.127867</td><td>0.165582</td></tr>
<tr><td rowspan="3">Llama-3.1-8B</td><td>N only</td><td>0.2217</td><td>0.3666</td><td>0.1147</td><td>0.1511</td></tr>
<tr><td>CIR → N SFT</td><td>0.240225</td><td>0.380369</td><td>0.119453</td><td>0.154541</td></tr>
<tr><td>Balanced CIR → N SFT</td><td>0.232901</td><td>0.380894</td><td>0.121194</td><td>0.158506</td></tr>
<tr><td rowspan="2">Semantic control</td><td rowspan="2">Qwen3.5-9B</td><td>CIR · True semantics</td><td>0.221843</td><td>0.368785</td><td>0.114764</td><td>0.151856</td></tr>
<tr><td>CIR · Shuffled semantics</td><td>0.238416</td><td>0.378356</td><td>0.121855</td><td>0.156795</td></tr>
</tbody>
</table>

**Finding：** CIR → N SFT相对对应N only参考有正向点估计；Balanced CIR三轮进一步改善Qwen，但预算不同。True semantics未优于Shuffled semantics，不能把收益单独归因于语义正确性。

来源：[CIR报告](docs/experiments/cir/report.md)、[重构正式审计](docs/experiments/mechanism/report.md)、[CIR三轮来源](docs/experiments/directionality/report.md)。original CIR为91,120条/2,848步预热；balanced为83,888条/2,622步，三轮7,866步；各后续N112K/3,500步。此表合列不同预算与语义版本作索引，**不标统一最优**；不把三轮相对一轮优势归因于单一语义因素。真实语义未优于shuffled，本组结果不支持将收益归因于语义正确性，但不能据此证明语义机制永远无效。

### 5.2 CIR收益是否来自信息方向？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="5">Qwen3.5-9B</td><td>forward</td><td>0.220676</td><td>0.363329</td><td>0.112743</td><td>0.148507</td></tr>
<tr><td>CIR → N SFT (3 epochs; reference)</td><td>0.251255</td><td>0.401786</td><td>0.127867</td><td>0.165582</td></tr>
<tr><td>ordinary</td><td>0.224411</td><td>0.366392</td><td>0.117317</td><td>0.152764</td></tr>
<tr><td>shuffled</td><td>0.222397</td><td>0.373979</td><td>0.116580</td><td>0.154533</td></tr>
<tr><td>paired</td><td>0.225373</td><td>0.375934</td><td>0.115966</td><td>0.153609</td></tr>
</tbody>
</table>

**Finding：** 方向控制并未超过CIR三轮参考，但额外预算仍是混淆因素，本组不是纯方向效应证明。

来源：[方向性正式报告](docs/experiments/directionality/report.md)。保留原条件标识；CIR三轮不与方向控制等预算，因此不标统一最优，也不当作纯方向效应。

### 5.3 新兴趣的未来复现次数是否影响预测？（PPT第32页）

沿用PPT的`behavior = 1 / behavior ≥ 5`标记，**指目标兴趣在训练用户未来正交互中的复现次数**，不是可见历史行为数。原作业标识分别为`once / at_least_five`。

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Future recurrence</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>behavior = 1</td><td>0.225053</td><td>0.361520</td><td><strong>0.115580</strong></td><td><strong>0.149790</strong></td></tr>
<tr><td>behavior ≥ 5</td><td><strong>0.225490</strong></td><td><strong>0.361928</strong></td><td>0.110885</td><td>0.145122</td></tr>
</tbody>
</table>

**Finding：** behavior ≥ 5的Recall略高，而nDCG更低；未来高复现兴趣样本并未带来全面提升。

来源：[频率配对正式报告](docs/experiments/frequency/report.md)，使用最终v3审计。条件根据训练用户的未来正交互复现次数构造，不是可见历史频率；高复现并未全面更好。PPT的random行是另一轮0.2242对照，未经同预算合同确认不并入此配对表。

## 6. 去噪与数据效率：COGS

正式预算79,968条（112类×714），Semantic ID、1 epoch、同视图OOF评分；表内仅含12个完整LLM作业，不含线性代理或10%合成噪声。粗体按同模型、同视图的Random配对。

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Input</th><th rowspan="2">Method</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="6">Qwen3.5-9B</td><td rowspan="2">Behavior</td><td>Random</td><td>12.19</td><td>0.2056</td><td>0.3288</td><td>0.1038</td><td>0.1348</td></tr>
<tr><td>COGS</td><td><strong>14.34</strong></td><td><strong>0.2114</strong></td><td><strong>0.3441</strong></td><td><strong>0.1043</strong></td><td><strong>0.1376</strong></td></tr>
<tr><td rowspan="2">Interest</td><td>Random</td><td>10.89</td><td>0.1863</td><td>0.3161</td><td>0.0894</td><td>0.1218</td></tr>
<tr><td>COGS</td><td><strong>15.03</strong></td><td><strong>0.2019</strong></td><td><strong>0.3231</strong></td><td><strong>0.0994</strong></td><td><strong>0.1297</strong></td></tr>
<tr><td rowspan="2">Behavior + Interest</td><td>Random</td><td>13.66</td><td>0.2198</td><td>0.3543</td><td>0.1118</td><td>0.1455</td></tr>
<tr><td>COGS</td><td><strong>15.26</strong></td><td><strong>0.2291</strong></td><td><strong>0.3665</strong></td><td><strong>0.1165</strong></td><td><strong>0.1510</strong></td></tr>
<tr><td rowspan="6">Llama-3.1-8B</td><td rowspan="2">Behavior</td><td>Random</td><td>11.27</td><td>0.2029</td><td>0.3375</td><td>0.1013</td><td>0.1349</td></tr>
<tr><td>COGS</td><td><strong>13.34</strong></td><td><strong>0.2097</strong></td><td><strong>0.3479</strong></td><td><strong>0.1043</strong></td><td><strong>0.1389</strong></td></tr>
<tr><td rowspan="2">Interest</td><td>Random</td><td>11.42</td><td>0.1838</td><td>0.3141</td><td>0.0929</td><td>0.1257</td></tr>
<tr><td>COGS</td><td><strong>14.37</strong></td><td><strong>0.2003</strong></td><td><strong>0.3378</strong></td><td><strong>0.0993</strong></td><td><strong>0.1337</strong></td></tr>
<tr><td rowspan="2">Behavior + Interest</td><td>Random</td><td>14.00</td><td>0.2182</td><td>0.3556</td><td>0.1115</td><td>0.1459</td></tr>
<tr><td>COGS</td><td><strong>16.47</strong></td><td><strong>0.2238</strong></td><td><strong>0.3671</strong></td><td><strong>0.1119</strong></td><td><strong>0.1482</strong></td></tr>
</tbody>
</table>

**Finding：** COGS相对同预算Random，在两个模型×三种输入的六配对中均提高五项点估计；不代表每格都达到统计显著。

来源：[去噪正式报告§6](docs/experiments/denoising/report.md)。六配对五项均正，但类别bootstrap并非每格都显著。相对112K全量训练的辅助比较改变样本量与步数，不能替代本表同预算Random；详细差值和置信区间见原报告。

## 7. 机制与结构扩展：包含负结果

以下均为Qwen3.5-9B、完整Dev2000/112类。每组只使用自身来源与baseline，不能横向挑选G0。

### 7.1 CIR与结构改动是否互补？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th rowspan="2">Steps</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="4">Qwen3.5-9B</td><td>N only</td><td>3,500</td><td>0.224236</td><td>0.368464</td><td>0.113470</td><td>0.149803</td></tr>
<tr><td>Structure + N SFT</td><td>3,500</td><td>0.221289</td><td>0.364904</td><td>0.112793</td><td>0.148969</td></tr>
<tr><td>CIR → N SFT</td><td>6,348</td><td><strong>0.240780</strong></td><td><strong>0.381215</strong></td><td><strong>0.124903</strong></td><td><strong>0.160060</strong></td></tr>
<tr><td>CIR + Structure → N SFT</td><td>6,348</td><td>0.239233</td><td>0.377480</td><td>0.123248</td><td>0.157685</td></tr>
</tbody>
</table>

**Finding：** CIR → N SFT四项高于CIR + Structure → N SFT，当前结果不支持两组件互补。

来源：[机制消融v2](docs/experiments/mechanism/report.md)。重构优于组合，不支持叠加结构进一步改善；与N-only比较包含额外预热预算。

### 7.2 CIR信息因子控制能否改善排序？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="5">Qwen3.5-9B</td><td>G0 · 历史参考复用</td><td><strong>0.251255</strong></td><td><strong>0.401786</strong></td><td><strong>0.127867</strong></td><td><strong>0.165582</strong></td></tr>
<tr><td>G1</td><td>0.246411</td><td>0.395863</td><td>0.125632</td><td>0.163053</td></tr>
<tr><td>G2</td><td>0.245769</td><td>0.397934</td><td>0.125915</td><td>0.164132</td></tr>
<tr><td>G3</td><td>0.242297</td><td>0.390202</td><td>0.124604</td><td>0.161978</td></tr>
<tr><td>G4</td><td>0.247228</td><td>0.395396</td><td>0.126208</td><td>0.163428</td></tr>
</tbody>
</table>

**Finding：** G1–G4均未超过G0参考，当前因子控制未带来整体排序收益。

来源：[因子分解报告](docs/experiments/factorization/report.md)。G1–G4均低于参考；条件含损失因子、mask和梯度控制，不单凭编号推断机制失效。

### 7.3 一致性约束能否改善排序？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>CE</td><td>0.223477</td><td><strong>0.371353</strong></td><td>0.115646</td><td>0.152846</td></tr>
<tr><td>CE + consistency</td><td><strong>0.223769</strong></td><td>0.367880</td><td><strong>0.117204</strong></td><td><strong>0.153364</strong></td></tr>
</tbody>
</table>

**Finding：** CE + consistency的nDCG略升、Macro Recall@20下降，没有全面改善。

来源：[一致性正式报告](docs/experiments/consistency/report.md)。四视图CE之外的一致性系数为1，Recall@20下降，不是全面收益。

### 7.4 冻结CIR或固定概率融合能否改善排序？

<table>
<thead>
<tr><th rowspan="2">Experiment</th><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">冻结CIR</td><td rowspan="2">Qwen3.5-9B</td><td>历史G0参考</td><td>0.251255</td><td>0.401786</td><td>0.127867</td><td>0.165582</td></tr>
<tr><td>冻结旧adapter、新建N LoRA</td><td>0.249767</td><td>0.394345</td><td>0.126828</td><td>0.163251</td></tr>
<tr><td rowspan="2">固定概率融合</td><td rowspan="2">Qwen3.5-9B</td><td>历史G0参考</td><td><strong>0.251255</strong></td><td><strong>0.401786</strong></td><td><strong>0.127867</strong></td><td><strong>0.165582</strong></td></tr>
<tr><td>首ID概率1:1融合</td><td>0.241801</td><td>0.383141</td><td>0.124131</td><td>0.159537</td></tr>
</tbody>
</table>

**Finding：** 冻结CIR的新N LoRA未超过历史参考；固定概率融合四项下降，两者不属于同一种参数融合方法。

来源：[冻结CIR](docs/experiments/residual/report.md)、[概率融合](docs/experiments/fusion/report.md)。冻结CIR的N布局与参考不同，不标严格配对最优；融合为分布混合，不是LoRA参数合并，四项下降。

### 7.5 独立兴趣编码能否改善排序？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="4">Qwen3.5-9B</td><td>G0 · 全文CIR→N参考</td><td>0.251255</td><td>0.401786</td><td>0.127867</td><td>0.165582</td></tr>
<tr><td>C · 真实类别独立编码+CIR</td><td>0.098010</td><td>0.193423</td><td>0.044046</td><td>0.067840</td></tr>
<tr><td>R · 随机分组独立编码+CIR</td><td>0.153887</td><td>0.285889</td><td>0.071860</td><td>0.104915</td></tr>
<tr><td>N · 真实分组、无CIR</td><td>0.125292</td><td>0.239992</td><td>0.057443</td><td>0.086272</td></tr>
</tbody>
</table>

**Finding：** 真实类别独立编码C低于随机分组R和全文G0，预设效果门槛失败；不能据此证明所有分组编码均无效。

来源：[独立编码最终验收§5.9](docs/experiments/composition/log.md)。C/R的CIR83,888条、3轮/7,866步，N阶段112K/3,500步；N没有预热，且物理分批不同，不标跨所有条件最优。C未优于R或G0，健康通过、预设效果门槛失败。

### 7.6 Role projection能否稳定改善排序？

<table>
<thead>
<tr><th rowspan="2">Run</th><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">首轮8卡</td><td rowspan="3">Qwen3.5-9B</td><td>Baseline</td><td>0.241597</td><td>0.389823</td><td>0.123334</td><td>0.160691</td></tr>
<tr><td>Unconstrained interaction</td><td>0.245915</td><td>0.390873</td><td>0.126345</td><td>0.162703</td></tr>
<tr><td>Role projection</td><td><strong>0.246674</strong></td><td><strong>0.396271</strong></td><td><strong>0.126753</strong></td><td><strong>0.164184</strong></td></tr>
<tr><td rowspan="3">G0对齐4卡</td><td rowspan="3">Qwen3.5-9B</td><td>Baseline</td><td><strong>0.245215</strong></td><td>0.389472</td><td><strong>0.125450</strong></td><td><strong>0.161815</strong></td></tr>
<tr><td>Unconstrained interaction</td><td>0.244748</td><td><strong>0.390319</strong></td><td>0.124223</td><td>0.161035</td></tr>
<tr><td>Role projection</td><td>0.242735</td><td>0.387780</td><td>0.123930</td><td>0.160588</td></tr>
</tbody>
</table>

**Finding：** 首轮小幅趋势未在G0对齐四卡轮复现，两轮均未通过效果门槛，不支持稳定改善。

来源：[角色投影正式日志实验6–7](docs/experiments/role_projection/log.md)、[短报告](docs/experiments/role_projection/report.md)。粗体仅在同轮比较。首轮nDCG@20 +0.003493低于+0.005门槛；四卡复验下降，两轮效果门槛均失败，不能合并为独立seed重复或稳定改善。

### 7.7 Mass-preserving readout能否改善排序？

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">Qwen3.5-9B</td><td>Baseline</td><td>0.245215</td><td><strong>0.389472</strong></td><td>0.125450</td><td><strong>0.161815</strong></td></tr>
<tr><td>Mass-preserving readout</td><td><strong>0.246148</strong></td><td>0.383228</td><td><strong>0.126392</strong></td><td>0.161046</td></tr>
</tbody>
</table>

**Finding：** Mass-preserving readout改善@10点估计，却降低@20点估计，两个效果门槛均失败；健康通过不等于方法有效。

来源：[完整读出正式报告](docs/experiments/mass_readout/report.md)。原CIR直接开始N112K/3,500步，新112×4096零初始化残差只作用首回答位置；完整词表损失与生成不变。正式strict=1、重复率0，但Recall@20下降且nDCG@20未达+0.005，两门槛失败。不能因梯度法向能量或理论可表达性高就宣称有效。

## 8. 历史协议与诊断附录

### 8.1 早期零样本：独立样本与候选

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Reasoning</th><th colspan="3">Macro Accuracy (%) ↑</th></tr>
<tr><th>Behavior</th><th>Interest</th><th>Behavior + Interest</th></tr>
</thead>
<tbody>
<tr><td rowspan="2">GPT-5.6 Sol</td><td>Medium</td><td><strong>15.21</strong></td><td>11.91</td><td>15.28</td></tr>
<tr><td>None</td><td>11.96</td><td>12.23</td><td><strong>16.06</strong></td></tr>
<tr><td rowspan="2">GPT-5</td><td>Medium</td><td>11.58</td><td>10.43</td><td>14.42</td></tr>
<tr><td>None</td><td>7.59</td><td>11.80</td><td>13.63</td></tr>
<tr><td rowspan="2">Gemini 3.6 Flash</td><td>Medium</td><td>13.54</td><td>10.33</td><td>13.94</td></tr>
<tr><td>Minimal</td><td>13.20</td><td>11.89</td><td>15.02</td></tr>
<tr><td>Qwen3.5-9B</td><td>—</td><td>14.62</td><td><strong>12.79</strong></td><td>13.30</td></tr>
<tr><td>Llama-3.1-8B</td><td>—</td><td>10.75</td><td>8.66</td><td>9.27</td></tr>
<tr><td>Random baseline</td><td>—</td><td>10.00</td><td>10.00</td><td>10.00</td></tr>
</tbody>
</table>

**Finding：** 早期零样本结果与Home112匹配零样本属于不同样本与候选版本，不能跨表计算标准SFT增益。

来源：[早期零样本正式表](docs/experiments/zero_shot/results.md)，2026-08-25评测、24/24视图完成。只保留v3，早期Dev与候选不是第1.1节的Home112匹配版本，不用于标准SFT增益对照。

### 8.2 99类、396K标准SFT

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Output</th><th rowspan="2">Input</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Qwen3.5-9B</td><td rowspan="3">Category name</td><td>Behavior</td><td>13.75</td><td>0.2133</td><td>0.3671</td><td>0.1091</td><td>0.1476</td></tr>
<tr><td>Interest</td><td>13.08</td><td>0.1835</td><td>0.3428</td><td>0.0926</td><td>0.1322</td></tr>
<tr><td>Behavior + Interest</td><td><strong>17.13</strong></td><td><strong>0.2176</strong></td><td><strong>0.3783</strong></td><td><strong>0.1158</strong></td><td><strong>0.1558</strong></td></tr>
<tr><td rowspan="3">Llama-3.1-8B</td><td rowspan="3">Category name</td><td>Behavior</td><td>13.77</td><td>0.2043</td><td>0.3619</td><td>0.1072</td><td>0.1468</td></tr>
<tr><td>Interest</td><td>15.26</td><td>0.1844</td><td>0.3472</td><td>0.0933</td><td>0.1340</td></tr>
<tr><td>Behavior + Interest</td><td><strong>17.46</strong></td><td><strong>0.2153</strong></td><td><strong>0.3770</strong></td><td><strong>0.1135</strong></td><td><strong>0.1540</strong></td></tr>
</tbody>
</table>

**Finding：** 99类Category name实验中，Behavior + Interest在两个模型的五项指标上均最高；没有该协议下已完成的Interest ID结果。

来源：[99类SFT正式表](docs/experiments/supervised_learning_99/results.md)。Train99类×4,000，Dev2000/99类；仅Category六配置完整。ID输出未提交，无正式值，不补为零或将112类ID结果移入。

### 8.3 99类数据规模：仅列已完成点

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Train size</th><th rowspan="2">Macro Accuracy (%) ↑</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="7">Qwen3.5-9B</td><td>20K</td><td>9.30</td><td>0.1995</td><td>0.3514</td><td>0.0965</td><td>0.1344</td></tr>
<tr><td>80K</td><td>11.76</td><td>0.2114</td><td>0.3729</td><td>0.1070</td><td>0.1475</td></tr>
<tr><td>120K</td><td>17.61</td><td>0.2128</td><td>0.3785</td><td>0.1041</td><td>0.1456</td></tr>
<tr><td>160K</td><td>17.86</td><td>0.2158</td><td>0.3730</td><td>0.1096</td><td>0.1491</td></tr>
<tr><td>200K</td><td>17.92</td><td>0.2057</td><td>0.3720</td><td>0.1063</td><td>0.1478</td></tr>
<tr><td>240K</td><td><strong>18.26</strong></td><td><strong>0.2196</strong></td><td><strong>0.3887</strong></td><td>0.1144</td><td><strong>0.1566</strong></td></tr>
<tr><td>396K</td><td>17.13</td><td>0.2176</td><td>0.3783</td><td><strong>0.1158</strong></td><td>0.1558</td></tr>
<tr><td rowspan="6">Llama-3.1-8B</td><td>20K</td><td>12.73</td><td>0.1813</td><td>0.3332</td><td>0.0885</td><td>0.1265</td></tr>
<tr><td>40K</td><td>10.50</td><td>0.1860</td><td>0.3561</td><td>0.0934</td><td>0.1360</td></tr>
<tr><td>80K</td><td>16.01</td><td>0.1991</td><td>0.3602</td><td>0.1009</td><td>0.1410</td></tr>
<tr><td>240K</td><td>16.98</td><td><strong>0.2182</strong></td><td><strong>0.3949</strong></td><td><strong>0.1159</strong></td><td><strong>0.1601</strong></td></tr>
<tr><td>320K</td><td>17.04</td><td>0.2168</td><td>0.3927</td><td>0.1137</td><td>0.1578</td></tr>
<tr><td>396K</td><td><strong>17.46</strong></td><td>0.2153</td><td>0.3770</td><td>0.1135</td><td>0.1540</td></tr>
</tbody>
</table>

**Finding：** 来源快照仅有13个完整点，其余11个点没有可信完整指标；不能据此给出完整曲线的饱和结论。

来源：[99类规模历史快照](docs/experiments/data_scaling_99/results.md)，记录日期2026-09-01，13/24完整（含两个复用端点）。其余11个点在来源中没有完整可信指标，不列入完成表，也不推断当前云端状态；原表空格仍代表未产出，不是零。不能与112类曲线合并或作配对差值。

### 8.4 历史93K课程（PPT第29页上表）

<table>
<thead>
<tr><th rowspan="2">Model</th><th rowspan="2">Method</th><th rowspan="2">Training data / Steps</th><th colspan="2">Macro Recall ↑</th><th colspan="2">Macro nDCG ↑</th></tr>
<tr><th>@10</th><th>@20</th><th>@10</th><th>@20</th></tr>
</thead>
<tbody>
<tr><td rowspan="3">Qwen3.5-9B</td><td>N only</td><td>112类 / N112K / 3,500</td><td>0.2242</td><td>0.3685</td><td>0.1135</td><td>0.1498</td></tr>
<tr><td>O → N</td><td>93类 / O46.5K+N46.5K / 2,907</td><td>0.2050</td><td>0.3324</td><td>0.0999</td><td>0.1319</td></tr>
<tr><td>O + N shuffle</td><td>93类 / 同一93K并集 / 2,907</td><td>0.2048</td><td>0.3283</td><td>0.1021</td><td>0.1331</td></tr>
</tbody>
</table>

**Finding：** 历史93K课程总分低于N only参考，但同时缺少19类目标监督；不能将差距单独解释为旧兴趣预热的负作用。

来源：[课程报告](docs/experiments/curriculum/report.md)，本次另核对冻结总设计§4.4。三行仍评测完整Dev2000/112类，但课程没有19类目标监督；N-only与课程不是同支持集对照，不标统一最优。该版本与第4.5节完整N112K课程不同，不能跨表替换。

### 8.5 MeRA发布代码strict历史诊断

<table>
<thead>
<tr><th rowspan="2">Dataset</th><th rowspan="2">Model</th><th rowspan="2">Result</th><th colspan="3">Recall ↑</th><th colspan="3">Precision ↑</th></tr>
<tr><th>@1</th><th>@2</th><th>@3</th><th>@1</th><th>@2</th><th>@3</th></tr>
</thead>
<tbody>
<tr><td>Arts</td><td>Qwen2-1.5B</td><td>修正前发布代码</td><td>0.0786</td><td>0.1485</td><td>0.2046</td><td>0.2021</td><td>0.1882</td><td>0.1783</td></tr>
<tr><td>Grocery</td><td>Qwen2-1.5B</td><td>修正前发布代码</td><td>0.0503</td><td>0.0933</td><td>0.1406</td><td>0.1180</td><td>0.1108</td><td>0.1126</td></tr>
</tbody>
</table>

**Finding：** 修正前发布代码结果用于历史诊断，不替代作者澄清协议下的同模型复现。

来源：[MeRA旧strict诊断](docs/experiments/mera/results.md)、[strict实现边界](docs/experiments/mera_strict/README.md)。这是有数值的历史诊断，但不是作者澄清协议的最终复现，不能用来替换第2.3节结果或证明修正后仍未复现。

## 9. 指标功能与健康检查

| 指标 | 衡量什么 | 判分方式 | 单位 / 方向 | 评测范围 |
|---|---|---|---|---|
| Macro Accuracy | 10候选中的正确类别选择 | 固定候选、严格解析、类别内平均后宏平均；模型族解码方式见来源 | % ↑ | Home112每视图Dev2000；随机参考10% |
| Macro Recall@10/20 | 新兴趣是否进入无候选生成Top-K | 冻结gold与解析后的预测，judge-free自动指标 | 0–1 ↑ | Home112 Dev2000、112类等权 |
| Macro nDCG@10/20 | 正确新兴趣在Top-K中的排序位置 | 同一批排名的折损增益，judge-free自动指标 | 0–1 ↑ | 同上，不与候选Accuracy互换 |
| MeRA官方Recall/Precision@1/2/3 | 官方数据集未来兴趣集合的覆盖与准确比例 | 官方循环公式与项目双实现核验；多目标口径 | 0–1 ↑ | Arts/Grocery官方协议，非Home112 Macro |
| valid / strict / duplicate | 生成是否属于合法ID、是否严格格式、合法beam是否重复 | 冻结解析器；无效输出不补修，历史版本见来源 | 合法/格式↑，重复↓ | 健康度，不是推荐收益 |

主排序不按gold选checkpoint，不新增已见类mask或候选修复。COGS之外的多数表为单seed点估计、无置信区间；COGS类别bootstrap也不代表多训练seed验证。开发集反复使用，不能把任一表的最高值外推为独立测试上的显著优势。

## 10. 覆盖与状态

此清单覆盖全部30个实验/数据分支。完成指来源存在完整正式指标或数据合同，不等于方法有效；“未纳入”指缺少本次可核对的完整正式结果，不等于断言云端从未运行。资源列只标任务类型或已确认布局，不估算总GPU成本。

| 分支 / 来源 | 本汇总位置 | 资源类型 | 可核对状态 |
|---|---|---|---|
| [数据构建](docs/experiments/dataset/README.md) | 顶部协议 | 数据处理 | 已完成冻结数据合同，不是模型结果 |
| [早期零样本](docs/experiments/zero_shot/results.md) | 8.1 | API/本地推理 | 24/24视图完成 |
| [Home112零样本](docs/experiments/zero_shot_home112/results.md) | 1.1 | API/本地推理 | 8模型条件×3视图完整，复测不替换来源 |
| [112类SFT](docs/experiments/supervised_learning/results.md) | 1.2 | LLM训练/评测 | 18配置完整；伪ID作废 |
| [99类SFT](docs/experiments/supervised_learning_99/results.md) | 8.2 | LLM训练/评测 | Category六配置完整；ID未提交 |
| [112类规模](docs/experiments/data_scaling/results.md) | 3 | LLM训练/评测 | 16/16完整，含2复用端点 |
| [99类规模](docs/experiments/data_scaling_99/results.md) | 8.3 | LLM训练/评测 | 来源快照13/24完整，其余无完整指标 |
| [Google](docs/experiments/google_baseline/results.md) | 2.1 | LLM训练/评测 | 两模型正式比较，PPT一格差异保留 |
| [MeRA](docs/experiments/mera/results.md) | 2.2–2.3 | LLM训练/评测 | Home及已完成官方结果；Qwen2 Grocery留空 |
| [MeRA旧strict](docs/experiments/mera_strict/README.md) | 8.5 | LLM训练/评测 | 历史诊断独立，不计作者修正结果 |
| [Interest-only对齐](docs/experiments/alignment/report.md) | 4.1 | 4×A100 80GB/模型 | 两模型正式完整 |
| [B+I对齐](docs/experiments/alignment_behavior_interest/report.md) | 4.1 | 4×A100 80GB/模型 | 两模型正式完整 |
| [骨干](docs/experiments/backbones/report.md) | 4.3–4.4 | LLM训练/评测 | 九个唯一骨干完整 |
| [Cross-fitted](docs/experiments/crossfit/report.md) | 4.2 | 五教师+学生 | OOF全112K评分及学生完整 |
| [课程](docs/experiments/curriculum/report.md) | 4.5、8.4 | LLM训练/评测 | 两个支持集各两臂完整；N→O未执行 |
| [CIR](docs/experiments/cir/report.md) | 5.1 | LLM训练/评测 | 基础、均衡、语义分支完整 |
| [方向性](docs/experiments/directionality/report.md) | 5.2 | LLM训练/评测 | 五条件完整，含历史CIR参考 |
| [频率](docs/experiments/frequency/report.md) | 5.3 | LLM训练/评测 | 两臂v3正式完整 |
| [去噪](docs/experiments/denoising/report.md) | 6 | 线性OOF教师+LLM学生 | 12个正式作业完整；代理不并表 |
| [结构与重构](docs/experiments/mechanism/report.md) | 7.1 | LLM训练/评测 | 三臂v2完整 |
| [因子分解](docs/experiments/factorization/report.md) | 7.2 | LLM训练/评测 | G1–G4完整，G0复用 |
| [一致性](docs/experiments/consistency/report.md) | 7.3 | LLM训练/评测 | CE/consistent两臂完整 |
| [冻结CIR](docs/experiments/residual/report.md) | 7.4 | LLM训练/评测 | 正式完整，未超过历史参考 |
| [概率融合](docs/experiments/fusion/report.md) | 7.4 | 线性模型+LLM推理 | 正式完整，四项下降 |
| [独立编码](docs/experiments/composition/report.md) | 7.5 | LLM训练/评测 | C/R/N完整，效果门槛失败 |
| [角色投影](docs/experiments/role_projection/report.md) | 7.6 | 首轮8卡/复验4卡 | 两轮三臂完整，两轮效果门槛失败 |
| [完整读出](docs/experiments/mass_readout/report.md) | 7.7 | 4卡N阶段 | 唯一正式作业完整，两效果门槛失败 |
| [摘要/检索/教师](docs/experiments/summaries/report.md) | 未纳入数值主表 | 历史生成探索 | 尚无本次全量验收的统一正式表 |
| [SuperLoss](docs/experiments/superloss/report.md) | 未纳入数值主表 | 设计/实现/smoke | 尚无可信完整正式结果表 |
| [MIND](docs/experiments/news_history/report.md) | 未纳入数值主表 | 数据处理 | 有数据，无对应完整正式模型报告 |

本次只整理本地已保存的结果和来源，没有查询新的云端终态、改写冻结设计、重新训练或生成预测。需要更新空格时，应先补齐对应完整指标及合同/原始预测核验，而不是依据PPT的措辞或预期收益填数。
