# Amazon 新兴趣预测去噪实验报告

> 数据：冻结 Amazon Train 112,000 条（112 类 x 1,000 条）与三视图 Dev-2K。方法：视图对齐的类别平衡跨用户 OOF 低损失选择。代理结论来自 22,400 条 Train 子池；正式结论来自 Qwen3.5-9B 与 Llama-3.1-8B-Instruct 的 12 个完整集群作业。主表指标均为 112 类 macro 平均；Accuracy 使用百分数，ranking 指标使用 0-1 小数。

## 1. 实验目标

本实验寻找一个尽量简单、可执行且能提高候选无关新兴趣预测能力的数据去噪方法。核心问题是：在无法从单次隐式点击辨认“真实兴趣、偶然点击、不可预测的新兴趣和错误标签”的条件下，能否估计每条训练样本对跨用户泛化的价值，并在固定训练预算内提高 Macro Recall 与 Macro nDCG。

具体回答三个问题：

1. 哪一种轻量训练价值信号能稳定优于同预算随机选样？
2. 该信号在三种输入视图、干净数据与合成噪声下是否保持方向一致？
3. 该方法迁移到真实 8B/9B candidate-free SFT 后是否仍有效？

## 2. 最终方法：COGS

最终方法命名为 **COGS（Class-balanced cross-user OOF Generalizability Selection）**。它不估计标签为真的概率，而是估计一条样本能否被其他用户的数据所支持。

将用户集合划分为五折。对属于用户折 $k(i)$ 的样本 $z_i=(x_i,y_i)$，仅使用其余四折用户训练轻量多分类教师 $f_{-k(i)}$，定义：

$$
q_i = p_{f_{-k(i)}}(y_i\mid x_i),
\qquad
s_i = -\log q_i.
$$

$s_i$ 是严格跨用户 OOF NLL。低分表示：不看该用户及该样本时，其他用户的历史模式仍能支持观测的新兴趣标签。高分可能来自标签错误、偶然或个体化点击、证据不足，或真实但难以迁移的长尾模式；方法不对这些原因作不可辨识的强区分。

训练集原本每类 1,000 条。对每个类别 $c$ 独立求解固定预算选择：

$$
S_c^* = \underset{S\subseteq D_c,\ |S|=714}{\arg\min}
\sum_{i\in S}s_i.
$$

因此正式子集为 112 类 x 714 条 = 79,968 条。实现按类别内 rank 轮转生成全局 `nested_rank`，任意适当前缀都保持类间样本数差不超过 1。正式 SFT 仍使用原始 hard label，不蒸馏教师输出，也不向模型提供候选类别。

### 2.1 为什么采用这一形式

- **跨用户 OOF**：避免模型通过见过同一用户或同一训练样本获得虚假的低损失。
- **低 OOF NLL**：直接衡量观测标签能否由 peer data 泛化预测，不引入未经验证的动态差分。
- **类别内选择**：训练数据和评测均强调 macro 指标，不能让易分类类别吞噬预算。
- **视图对齐评分**：Behavior、Interest、B+I 分别使用同一输入视图的教师打分，避免 train-time 证据与评分证据错位。
- **轻教师、重学生**：HashingVectorizer + 线性 log-loss 教师只负责排序数据，正式学生仍为 8B/9B LLM。

## 3. 数据与控制变量

| 项目 | 固定设置 |
|---|---|
| Train | 112,000 条，112 类 x 1,000 条 |
| Dev | 每个视图固定 2,000 条，112 类 macro 评测 |
| 用户边界 | Train/Dev 用户不相交；OOF fold 按用户划分 |
| 视图 | Behavior、Interest、Behavior + Interest |
| 正式预算 | 79,968 条，112 类 x 714 条 |
| 随机对照 | 类别内固定 SHA-256 哈希顺序，同预算同类分布 |
| 输出编码 | 112 个语义初始化的单 token Interest ID |
| 正式训练 | candidate-free、assistant-only、LoRA、1 epoch、BF16 |
| 优化口径 | learning rate `1e-4`，effective batch size 32 |
| 正式指标 | Macro Accuracy、Recall@10/20、nDCG@10/20 |

代理实验从每类稳定抽取 200 条，共 22,400 条；使用五折 `StratifiedGroupKFold`、$2^{18}$ 维 hashing bigram 特征与 `SGDClassifier(loss="log_loss")`。合成噪声按类别循环替换 10% 标签，保持每类计数完全不变。

## 4. 方法筛选与证伪

### 4.1 被否决的 RHO-like learning progress

最初比较低资源与高资源两个严格 OOF 教师：

$$
\Delta_i = \ell(f^{low}_{-k(i)};z_i)-\ell(f^{high}_{-k(i)};z_i).
$$

这一差分解决了训练内损失与 OOF 损失不可比的问题，但在足量 clean/noise 代理实验中基本不优于随机，并持续弱于直接的高资源 OOF 低损失。故 learning progress 仅保留为诊断，不控制正式 selection。

### 4.2 视图统一评分被否决

曾测试用一份 B+I OOF 排名统一选择三个视图的训练数据。clean 条件近似持平，但在 10% noise 下，Behavior 与 Interest 的 Recall@20 相对各自视图评分分别下降 0.68 和 0.38 个百分点。因此正式方法采用视图对齐评分，而非追求表面上的单一 selection 文件。

### 4.3 最终预算

已有 112-label 数据量曲线显示两个 LLM 的四项 ranking 指标从约 80K 起进入经验平台，而 Accuracy 尚未完全饱和。正式预算取最接近且完全类均衡、可被 4/8 卡整除的 79,968 条；相当于保留 71.4%，删除 28.6%。

## 5. 代理实验结果

下表报告 COGS 相对同预算随机类别平衡选择的绝对变化，单位为百分点。所有实验共享同一 Train 子池、Dev-2K、fold、seed、训练器和预算。

<table>
  <thead>
    <tr>
      <th rowspan="2">视图</th>
      <th rowspan="2">数据条件</th>
      <th colspan="5">COGS - Random（百分点）</th>
    </tr>
    <tr>
      <th>Accuracy</th><th>Recall@10</th><th>Recall@20</th><th>nDCG@10</th><th>nDCG@20</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Behavior</td><td>Clean</td><td>-0.76</td><td>+0.20</td><td><strong>+1.09</strong></td><td>-0.20</td><td>+0.05</td></tr>
    <tr><td>10% noise</td><td>-0.56</td><td>+1.59</td><td><strong>+1.95</strong></td><td>+0.21</td><td>+0.30</td></tr>
    <tr><td rowspan="2">Interest</td><td>Clean</td><td>-0.19</td><td>+0.97</td><td><strong>+1.46</strong></td><td>+0.20</td><td>+0.34</td></tr>
    <tr><td>10% noise</td><td>+0.05</td><td>+0.81</td><td><strong>+1.68</strong></td><td>+0.47</td><td>+0.70</td></tr>
    <tr><td rowspan="2">B+I</td><td>Clean</td><td>-0.91</td><td>+1.30</td><td><strong>+1.45</strong></td><td>+0.04</td><td>+0.07</td></tr>
    <tr><td>10% noise</td><td>-0.26</td><td>+0.89</td><td><strong>+1.41</strong></td><td>+0.03</td><td>+0.15</td></tr>
  </tbody>
</table>

三视图在 clean/noise 下的 Recall@20 与 nDCG@20 均为正，说明 71.4% 预算下的方向一致。Accuracy 在四个条件下降低，表明该方法当前更适合作为 **ranking recall-oriented data selection**，不能宣称统一提高所有指标。

合成噪声下，随机子集的污染率为 9.99%；Behavior、Interest、B+I 的 COGS 子集分别为 9.36%、9.02%、9.17%。OOF loss 对 clean/noise 的单样本 AUROC 仅约 0.54-0.56，进一步说明其价值是改善训练子集，而不是精确检测每个错误标签。

## 6. 正式 LLM 实验

正式矩阵比较两种方法：

- `COGS-79,968`：对应视图的跨用户 OOF 低损失子集。
- `Random-79,968`：同一 Train、同一类别预算的固定哈希随机子集。

两种模型为 `Qwen/Qwen3.5-9B` 与 `NousResearch/Meta-Llama-3.1-8B-Instruct`。每个模型覆盖三种视图，共 12 个任务；所有任务使用相同 semantic-token 输出、LoRA 超参数、训练 epoch、effective batch 和冻结 Dev-2K。

<table>
  <thead>
    <tr>
      <th>模型</th><th>视图</th><th>方法</th><th>Macro Accuracy (%) ↑</th><th>Recall@10 ↑</th><th>Recall@20 ↑</th><th>nDCG@10 ↑</th><th>nDCG@20 ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="6">Qwen3.5-9B</td><td rowspan="2">Behavior</td><td>Random</td><td>12.19</td><td>0.2056</td><td>0.3288</td><td>0.1038</td><td>0.1348</td></tr>
    <tr><td>COGS</td><td><strong>14.34</strong></td><td><strong>0.2114</strong></td><td><strong>0.3441</strong></td><td><strong>0.1043</strong></td><td><strong>0.1376</strong></td></tr>
    <tr><td rowspan="2">Interest</td><td>Random</td><td>10.89</td><td>0.1863</td><td>0.3161</td><td>0.0894</td><td>0.1218</td></tr>
    <tr><td>COGS</td><td><strong>15.03</strong></td><td><strong>0.2019</strong></td><td><strong>0.3231</strong></td><td><strong>0.0994</strong></td><td><strong>0.1297</strong></td></tr>
    <tr><td rowspan="2">B+I</td><td>Random</td><td>13.66</td><td>0.2198</td><td>0.3543</td><td>0.1118</td><td>0.1455</td></tr>
    <tr><td>COGS</td><td><strong>15.26</strong></td><td><strong>0.2291</strong></td><td><strong>0.3665</strong></td><td><strong>0.1165</strong></td><td><strong>0.1510</strong></td></tr>
    <tr><td rowspan="6">Llama-3.1-8B-Instruct</td><td rowspan="2">Behavior</td><td>Random</td><td>11.27</td><td>0.2029</td><td>0.3375</td><td>0.1013</td><td>0.1349</td></tr>
    <tr><td>COGS</td><td><strong>13.34</strong></td><td><strong>0.2097</strong></td><td><strong>0.3479</strong></td><td><strong>0.1043</strong></td><td><strong>0.1389</strong></td></tr>
    <tr><td rowspan="2">Interest</td><td>Random</td><td>11.42</td><td>0.1838</td><td>0.3141</td><td>0.0929</td><td>0.1257</td></tr>
    <tr><td>COGS</td><td><strong>14.37</strong></td><td><strong>0.2003</strong></td><td><strong>0.3378</strong></td><td><strong>0.0993</strong></td><td><strong>0.1337</strong></td></tr>
    <tr><td rowspan="2">B+I</td><td>Random</td><td>14.00</td><td>0.2182</td><td>0.3556</td><td>0.1115</td><td>0.1459</td></tr>
    <tr><td>COGS</td><td><strong>16.47</strong></td><td><strong>0.2238</strong></td><td><strong>0.3671</strong></td><td><strong>0.1119</strong></td><td><strong>0.1482</strong></td></tr>
  </tbody>
</table>

COGS 在 2 个模型 x 3 个视图的 **6/6 配对中同时提高全部五项指标**。六个设置的平均绝对增益为：Accuracy +2.56 个百分点、Recall@10 +0.99 个百分点、Recall@20 +1.33 个百分点、nDCG@10 +0.42 个百分点、nDCG@20 +0.51 个百分点。对应范围分别为 +1.59 至 +4.14、+0.56 至 +1.65、+0.69 至 +2.38、+0.04 至 +1.00、+0.23 至 +0.81 个百分点。

该结果也修正了代理实验的局部判断：线性代理的 Accuracy 曾在四个条件下降低，但正式 8B/9B SFT 中 Accuracy 为 6/6 正增益。因此代理只用于选方向，最终效果结论以本节正式指标为准。

### 6.1 类别级不确定性

对每个模型-视图配对，以 112 个目标类别为配对重采样单位进行 20,000 次 bootstrap。宏均值方向高度一致，但不能声称每个单元格都在 95% 区间下排除 0：

- Accuracy 的区间在 Qwen Behavior、Qwen Interest、Llama Interest、Llama B+I 四个设置中排除 0；其余两个设置仍为正点估计。
- Recall@20 仅 Llama Interest 的 95% 区间明确排除 0；Qwen Behavior 的单侧 bootstrap 非正概率为 0.0485，但双侧 95% 区间仍轻微跨 0。
- nDCG@10 仅 Qwen Interest 的区间排除 0；其他 ranking 单元格主要提供跨设置同向性证据，而不是逐单元格显著性证据。

因此最稳妥的陈述是：COGS 在两个模型和三种视图上呈现一致的正向宏平均效应，尤其稳定提升 Accuracy 与 Interest 视图；单个 ranking 设置的效应仍需多 seed 或更大评测集确认。

### 6.2 与全量 112K SFT 的辅助比较

使用同一 semantic Interest-ID、冻结 Dev-2K、1 epoch、learning rate 和 effective batch 的既有全量 SFT 作为辅助数据效率对照。该比较改变了训练样本数和优化步数，不替代同预算 Random 主对照。

| 模型 | 视图 | COGS - Full Accuracy (pp) | Recall@10 (pp) | Recall@20 (pp) | nDCG@10 (pp) | nDCG@20 (pp) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | Behavior | +0.12 | -0.16 | +0.03 | -0.25 | -0.21 |
| Qwen3.5-9B | Interest | +2.87 | +1.14 | +0.50 | +0.06 | -0.13 |
| Qwen3.5-9B | B+I | +0.42 | +0.04 | -0.60 | -0.09 | -0.26 |
| Llama-3.1-8B-Instruct | Behavior | +0.35 | +0.31 | +0.50 | +0.07 | +0.11 |
| Llama-3.1-8B-Instruct | Interest | +2.05 | +0.94 | +2.34 | +0.25 | +0.58 |
| Llama-3.1-8B-Instruct | B+I | +2.55 | +0.21 | +0.05 | -0.27 | -0.29 |

COGS 用少 28.6% 的训练样本获得 6/6 Accuracy、5/6 Recall@10 和 5/6 Recall@20 正增益；nDCG 相对全量训练接近持平且方向混合。这支持其数据效率价值，但不能将 nDCG 结果表述为全面优于全量数据。

### 6.3 运行与健康检查

- 有效 pilot：`amazon-oof-denoise-pilot-v2-20260901`，2 个 Qwen B+I matched jobs。
- 有效 full：`amazon-oof-denoise-full-v2-20260901`，其余 10 个 jobs。
- 提交目标：Feeds / `${AZURE_CONFIGURATION}`，`80G4-A100`，Standard SLA。
- 最终机器可读状态：pilot 2/2 `pass`，full 10/10 `pass`；watcher 未命中 traceback、OOM、NCCL、导入或运行时错误。
- 12 份训练合同均为 79,968 条、112 类 x 714 条、candidate-free、assistant-only、1 epoch、effective batch 32。六对 COGS/Random 合同除 selection 文件及其 SHA 外完全一致。
- 12 份 classification 与 12 份 ranking 均为 2,000/2,000、112 类、`missing_sample_count=0`、`complete=true`。Classification strict-format/resolved rate 均为 1.0；ranking valid/strict-beam rate 均为 1.0，duplicate-valid-beam rate 均为 0。
- 正式训练合同中的四个 selection SHA-256 与本地冻结 parquet 完全一致。
- 无效 pilot v1 因上传规则错误排除了 `crossinterest.evaluation.build_views`，在模型加载前失败，无 adapter 或指标；无效 full v1 已 10/10 `killed`。v2 已缩窄 ignore 规则并通过 import 与 AMLT dump 检查。

## 7. 结论与限制

**正式验证通过。** 在本项目已测试的简单去噪/数据选择信号中，COGS 是唯一同时通过代理筛选并在两个 8B/9B 模型、三种输入视图的正式 matched 实验中保持五项指标 6/6 正增益的方法。它可以作为论文核心训练数据选择方法，贡献应表述为“类别平衡、视图对齐的跨用户可泛化性选择”，而不是“识别真实兴趣”或“精确检测噪声”。

理论上，COGS 在“每类固定预算、最小化观测 OOF NLL 总和”的目标下给出精确最优子集；但该目标只是跨用户可预测性代理，不等价于真实心理兴趣或 clean-label probability。它可能过滤真实但稀有、难以由其他用户预测的模式，类别平衡也不能完全保证类内多样性。

当前正式矩阵每个模型-视图只有一个训练 seed。6/6 同向结果降低了偶然性的担忧，类别 bootstrap 也支持部分设置，但论文若要声称逐设置稳定或统计显著，仍需至少补充多 seed matched Random、强去噪基线的同预算正式比较，以及长尾/多样性分析。现有证据足以支持“在本协议与已测方法中一致有效”，不足以证明对所有推荐数据、所有噪声类型或所有模型全局最优。

## 8. 复现入口与产物

- 代理实验：`proxy_pilot.py`
- 正式 selection 构建：`build_selection.py`
- 三视图 OOF/random 排名与 manifest：`data/`
- Qwen B+I pilot 配置：`amlt/amlt_pilot.yaml`
- 完整矩阵配置：`amlt/amlt_full.yaml`
- 正式训练/评测入口：`../crossinterest.training/run_train_eval.sh`
- 正式 12-job 原始合同/指标：`formal_results_v2/`
- 严格汇总与类别 bootstrap：`aggregate_formal_results.py`
- 机器可读主汇总：`formal_results_v2/formal_summary.json`
- 可读主表与差值：`formal_results_v2/formal_summary.md`
- 全量 112K 辅助比较：`full_112k_baseline/cogs_vs_full_112k.json`