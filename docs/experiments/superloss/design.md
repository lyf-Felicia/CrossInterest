# MIND 双视图 SFT 与 SuperLoss 对比实验设计

## 1. 研究问题

在完全相同的基础模型、训练样本、训练步数、LoRA 配置和评测样本下，分别在
behavior 与 interest 两种输入视图中比较：

1. **Base-behavior / Base-interest（不训练）**：只做对应视图的 10-way choice；
2. **SFT-behavior / SFT-interest**：标准 assistant-only LoRA SFT；
3. **SuperLoss-behavior / SuperLoss-interest**：只将 SFT 的逐样本损失替换为动态
   SuperLoss 权重。

核心问题是 SuperLoss 是否比普通 SFT 更好地学习“历史中未出现的下一新兴趣”，而不是它是否优于未训练模型。`SFT − Base` 是训练有效性检查，`SuperLoss − SFT` 才是主要方法比较。

## 2. 唯一固定协议

本实验不搜索备选设置，所有运行固定为：

| 项目 | 固定设置 |
|---|---|
| 基础模型 | Qwen3.5-9B，thinking 关闭 |
| Behavior 视图 | 只显示历史新闻标题 |
| Interest 视图 | 只显示历史 subcategory |
| 长历史 | 时间上最近 50 条 |
| 训练规模 | 删除源支持度低于 200 的类别；保留类别各固定抽取 200 条 |
| Dev 规模 | 固定 5K，保持自然分布，不做 balance |
| 10-way 候选 | gold + 9 个由 SHA-256 固定的 train-taxonomy 负类 |
| SuperLoss 粒度 | 每序列 answer-token 平均 NLL |
| 阈值 $\tau$ | 过去 optimizer steps 的 loss EMA，$\beta=0.99$ |
| 权重 | 保留论文原式，可大于 1，权重下限为 0 |
| 课程阶段 | 5% 均匀 SFT + 80% SuperLoss + 15% 均匀 SFT |
| 随机种子 | 20260825 |

## 3. 固定数据视图

源数据：

- `mind/train.parquet`：587,187 条，124 个标签；
- `mind/dev.parquet`：37,817 条，111 个标签；
- dev 的 111 个标签全部出现在 train；
- train/dev 用户重叠为 0。

已由 `prepare_data.py` 生成到唯一正式数据目录 `mind/balanced/`，不在
`superloss_experiment/` 内保留数据副本：

- train：21,600 条，108 个标签，每类严格 200 条；
- dev：5,000 条，覆盖 104 个保留标签；
- seed：`20260825`；
- 先删除源支持度低于 200 的 16 个标签，共删除 749 条源记录；
- 对剩余 108 个标签分别按 `SHA-256(seed, sample_id)` 固定顺序抽取 200 条，
   不做上采样、不产生重复记录；
- 共从保留标签中下采样 564,838 条，最终 train 为 21,600 条；
- dev 先过滤到上述 108 标签 taxonomy，再按自然标签比例固定抽取 5,000 条；
   dev 本身不做类别均衡；
- history 保留最近 50 条，顺序仍为 oldest-to-newest；
- parquet 只保存结构化 `history`，不保存预渲染的 `messages`；四个训练条件共享同一个
   `train.parquet`，不会生成 per-view 数据副本；
- behavior 在运行时只渲染 `history.title`，interest 只渲染
   `history.subcategory`；
- target subcategory 不出现在保留后的 history；
- 每条 dev 固定 10 个唯一候选，gold 位置在每个标签内部均衡；
- 所有源文件、输出文件哈希、逐标签支持度和重复统计记录在
   `mind/balanced/MANIFEST.json`。

该策略对保留标签提供严格相同的训练行数，并完全排除低支持标签，不通过复制样本
增加独立信息量。所有训练方法必须使用完全相同的冻结训练行，保持方法比较的控制变量。

输出严格为：

```text
New topic: <subcategory>
```

108 个保留 subcategory 名称不存在跨 category 重名，因此输出 subcategory 足以唯一确定标签。
所有逐字 prompt 模板与候选隔离表见 `PROMPTS.md`。

## 4. 控制变量

| 项目 | 四个训练条件共同设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-9B` |
| 参数更新 | Base 不更新；SFT/SuperLoss 均为 LoRA |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear |
| 训练轮数 | 3 |
| 有效 batch | 32 |
| GPU micro-batch | 1 |
| 正式并行 | 8×A100，DDP |
| 梯度累积 | 4（$8\times1\times4=32$） |
| 学习率 | $10^{-4}$ |
| Scheduler | cosine |
| LR warmup | 3% |
| 优化器 | fused AdamW |
| 最大长度 | 8192；禁止 tokenizer 静默截断 |
| Loss mask | 只训练 assistant answer tokens |
| Seed | 20260825 |
| Checkpoint | 每 250 optimizer steps 保存完整状态，最多保留最近 3 份，自动从最新完整 checkpoint 恢复 |
| 训练后评测 | 2 张 GPU 分片运行 choice，6 张 GPU 分片运行 ranking；按原 dev 顺序严格校验并合并 |
| 训练预算 | SFT 与 SuperLoss 均看见相同样本、相同步数、相同 epoch |

普通 SFT 与 SuperLoss 的基础样本损失都定义为：

$$
\ell_i(\theta)
=
-\frac{1}{|A_i|}
\sum_{t\in A_i}
\log p_\theta(y_{it}\mid y_{i,<t},x_i),
$$

其中 $A_i$ 只包含 assistant answer token。由于 micro-batch 固定为 1，普通 SFT 与 Hugging Face 默认 token-mean loss 在单个 micro-step 上等价。

## 5. SuperLoss 的生成式适配

论文置信度闭式解为：

$$
\sigma_i^*
=
\exp\left[
-W\left(
\frac{1}{2}
\max\left(-\frac{2}{e},\frac{\ell_i-\tau}{\lambda}\right)
\right)
\right].
$$

实现采用纯 PyTorch 主 Lambert-$W$ 分支，不使用第三方仓库中的 SciPy/NumPy CPU 往返。训练对模型参数的有效梯度为：

$$
\nabla_\theta \mathcal L
=
\frac{1}{B}\sum_i
\operatorname{stopgrad}(\tilde\sigma_i)
\nabla_\theta\ell_i.
$$

### 5.1 与原论文一致的部分

- 使用闭式置信度；
- 对置信度停止梯度；
- 低损失样本权重可以大于 1，理论上界为 $e$；
- 主实验 `min_weight=0`，不人为截断到 0.1。

### 5.2 针对生成式 SFT 的必要适配

1. **序列级损失**：对 answer token 取均值后，每个序列得到一个难度值；不对 token 独立做课程学习。
2. **动态阈值**：分类论文常用 $\tau=\log C$，但生成式 token NLL 不具有同样的固定类别解释，因此使用既往 optimizer step 的 loss EMA：

   $$
   \tau_s=0.99\tau_{s-1}+0.01\bar\ell_{s-1}.
   $$

3. **跨 step 权重归一化**：micro-batch=1 时不能除以当前 batch 权重均值，否则 SuperLoss 恒等于 SFT。使用既往 optimizer step 的原始权重 EMA：

   $$
   \tilde\sigma_i=\frac{\sigma_i}{\operatorname{EMA}(\sigma)}.
   $$

4. **阶段化训练**：前 5% 使用普通 SFT 标定 loss，随后 80% 使用 SuperLoss，最后 15% 恢复普通 SFT。
5. **正则强度**：首轮固定 $\lambda=1.0$，不在同一 dev 上同时搜索并汇报最终效果。

这些适配必须在结果中明确标为 **SuperLoss-style sequence weighting**，不能声称是分类论文设置的逐字复刻。

## 6. 评测与统计

### 6.1 Prompt 与指标职责

| 方法 | 固定 10-way choice | 无候选 108 类 beam ranking |
|---|---|---|
| Base-behavior / Base-interest | Macro Accuracy@10 | 不运行 |
| SFT-behavior / SFT-interest | Macro Accuracy@10 | Macro Recall@10/@20、Macro nDCG@10/@20 |
| SuperLoss-behavior / SuperLoss-interest | Macro Accuracy@10 | Macro Recall@10/@20、Macro nDCG@10/@20 |

只有 choice prompt 显示 10 个候选，并使用 `num_beams=10` 返回 top-1。训练和 ranking
prompt 均不显示候选；ranking 使用 `num_beams=40` 在完整 108 标签输出 trie 上返回 20 个
唯一标签。Trie 只约束合法输出语法，不向 prompt 注入 taxonomy。

每条样本只有一个 gold subcategory。对 cutoff $K$：

$$
\operatorname{Recall@K}_i=\mathbb{1}[y_i\in R_i^{(K)}],
$$

$$
\operatorname{nDCG@K}_i=
\begin{cases}
1/\log_2(r_i+1), & r_i\le K,\\
0, & r_i>K,
\end{cases}
$$

其中 $K\in\{10,20\}$，$r_i$ 是 gold 的 1-based rank。所有 Macro 指标先在同一
gold subcategory 内平均，再对 dev 出现的 104 个 subcategory 等权平均。实验不计算或
汇报 ranking @1/@5。

统计比较使用同一 5K dev 上的 10,000 次**标签内分层成对 bootstrap**，每次重复保持各标签样本数不变：

- 每个视图独立进行统计，不跨 behavior/interest 混合样本；
- Choice：`SFT − Base` 与 `SuperLoss − SFT` 的 Macro Accuracy@10；
- Ranking：仅比较 `SuperLoss − SFT` 的 Macro Recall/nDCG@10/@20；
- 同时报 `right_only_correct` 与 `left_only_correct`。

成对 bootstrap 只反映固定训练运行下的 dev 样本不确定性，不将其解释为训练随机性。

## 7. 执行阶段

### 阶段 A：代码与 GPU smoke

四个训练条件各使用 2 条样本、2 个 optimizer steps；随后各评测 2 条 choice 和
ranking。Smoke 的有效 batch 临时设为 1，只验证执行路径，不用于结果比较：

```bash
amlt run amlt_smoke.yaml <smoke-experiment-name>
```

通过条件：LoRA-B 非零、loss 有限、SuperLoss 第二步进入课程阶段、产生非恒等权重、adapter 可加载。

### 阶段 B：108 类 exact-200 balanced train 正式比较

```bash
amlt run amlt_formal.yaml <formal-experiment-name>
python analyze_results.py \
   --behavior-base-choice runs/base_behavior/choice/predictions.jsonl \
   --behavior-sft-choice runs/sft_behavior/choice/predictions.jsonl \
   --behavior-superloss-choice runs/superloss_behavior/choice/predictions.jsonl \
   --behavior-sft-ranking runs/sft_behavior/ranking/predictions.jsonl \
   --behavior-superloss-ranking runs/superloss_behavior/ranking/predictions.jsonl \
   --interest-base-choice runs/base_interest/choice/predictions.jsonl \
   --interest-sft-choice runs/sft_interest/choice/predictions.jsonl \
   --interest-superloss-choice runs/superloss_interest/choice/predictions.jsonl \
   --interest-sft-ranking runs/sft_interest/ranking/predictions.jsonl \
   --interest-superloss-ranking runs/superloss_interest/ranking/predictions.jsonl \
   --output runs/comparison_balanced_200.json
```

## 8. 当前状态与阻塞

- 已完成低支持标签过滤与 retained-label exact-200 balance、结构化 parquet 重建、
   SHA-256 清单和独立不变量检查；
- 平衡 train 为 21,600 条、108 类、每类严格 200 条且无重复；dev 为过滤到同一
   taxonomy 后按自然分布固定抽取的 5K；
- 已完成共享双视图 prompt renderer、磁盘映射懒 tokenization、纯 PyTorch SuperLoss、
   sequence answer loss、四条件 trainer、约束评测器和按视图 bootstrap 分析器；
- prompt 候选隔离、平衡采样、评测指标和合成 bootstrap 测试已通过；
- 新 dev 与原冻结 dev 的 5,000 行内容和顺序逐条相同；由于重新序列化嵌套字段，
   parquet 字节哈希不同。Base 结果可按 `sample_id` 和内容对齐复用，但不声称文件哈希相同；
- 旧数据口径的四个正式作业均已终止；新方法作业将在课程学习方案调研完成后再提交。