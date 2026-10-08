# MeRA 复现实验报告

## 1. 实验目标

复现 Bi et al. 的 *Large Language Models with Multi-faceted Relation Alignment for User Novel Interest Discovery*（PAKDD 2025）及其官方实现 `GabyUSTC/MeRA`，验证扩展类别 token、结构关系正则和协同关系正则在 User Novel Interest Discovery（UNID）任务中的作用。

正式实验统一使用本项目的两个主模型：`Qwen/Qwen3.5-9B` 与 `NousResearch/Meta-Llama-3.1-8B-Instruct`。完整矩阵为 3 个数据集乘 2 个模型：

1. **论文数据适配主线**：保留官方 Amazon Arts 与 Grocery 数据、类别 token、结构/协同正则和论文集合 Recall/Precision，同时补充本项目统一五指标。
2. **Home & Kitchen 扩展主线**：在冻结 112K Train 与同一 Dev-2K 上运行相同方法，报告与 Arts/Grocery 同定义的 Query-Mean Accuracy、Recall@10/20 和 nDCG@10/20；另保留 Home 单 gold 类别宏指标，与既有 benchmark 对齐。

Qwen2-1.5B 不进入本项目双主模型公平比较矩阵。论文同模型主线直接执行固定官方 commit 的 `train.py` 与 `inference.py`，覆盖 Arts 和 Grocery；旧 Arts Natural/Balance 使用本项目 sparse-cosine 实现，只保留为非严格诊断。作者进一步确认发布代码的训练 prompt 与其真实实验不一致：真实训练直接拼接带双引号的历史类别与 gold 类别，固定训练 5 epochs，不使用 early stopping 或 best-checkpoint 回载。因此，新验收主线以该作者澄清协议为准；旧 public-code strict 结果继续保留为差异诊断，不再作为最终对齐结论。

Qwen3.5/Llama 的 Arts、Grocery 与特殊 Home 主线同样从固定官方源码生成，应用同一作者澄清协议，只允许基础模型身份、模型架构和当前库 API 所必需的兼容修改；官方数据、类别 token、初始化、CE、两项 raw-dot 关系损失及采样、完整 embedding/head 训练、batch、优化器、调度器和推理语义均保持不变。此前产生的 Qwen3.5/Llama Arts/Grocery 与第 9.2--9.4 节 Home 结果继续标为 **历史 MeRA 机制适配（非等价实现）**，不得填入新的作者协议跨模型单元格。第 9.5 节另列按发布代码训练协议运行的 Home 2×2 扩展实验，两者不混表。

## 2. 复现工作与原方法

当前已完成：固定官方 commit 与 12 个数据文件 SHA、全量数据与 balance 审计、三套标签/关系资产、Qwen2/Qwen3.5/Llama 训练与统一评测入口、27 项 CPU 合同测试。Qwen2 Arts/Grocery public-code strict 已完成训练、恢复评测和双公式验收；Qwen3.5/Llama Home public-code extension 的八项 2×2 训练与评测也已完成。作者澄清协议下，Qwen3.5/Llama 的 Arts、Grocery 与特殊 Home 六项正式任务均已通过完整五轮训练、最终 adapter 和全量评测门禁。Qwen2 Arts/Grocery 同样完成计划的 400/690 个 optimizer step 和最终 adapter 保存；Arts r4 只读恢复评测已通过，Grocery r4 正在运行。运行状态与失败记录集中在第 9.9 节，废弃任务不进入结果页。

对于用户历史类别序列 $S_u=[c_1,c_2,\ldots,c_t]$，UNID 预测未来的新类别 $c_f$，且 $c_f\notin S_u$。MeRA 为每个二级类别增加一个原子 token，并用类别名称在基础模型中的原始 token embedding 均值初始化该 token。训练目标为：

$$
\mathcal L
=
\mathcal L_{\mathrm{LLM}}
+\alpha\mathcal L_{\mathrm{structure}}
+\beta\mathcal L_{\mathrm{collab}}.
$$

其中：

- $\mathcal L_{\mathrm{LLM}}$：预测未来新类别 token 的自回归交叉熵；
- $\mathcal L_{\mathrm{structure}}$：同一父类下的 sibling 类别为正样本，其他父类类别为负样本，在扩展 embedding 与 LM head 上计算 InfoNCE；
- $\mathcal L_{\mathrm{collab}}$：根据训练用户类别共现矩阵采样正负类别，在扩展 embedding 与 LM head 上计算 InfoNCE。

推理时不做自然语言 beam search，而是读取最后位置在全部扩展类别 token 上的 logits，直接取 Top-K，因此不会生成标签空间外的类别。

## 3. 原论文数据集构建方法

论文使用 Amazon Reviews 2023 的 Arts, Crafts & Sewing 与 Grocery & Gourmet Food：

1. 将商品映射到二层类目树中的第二层类别，形成用户类别交互序列。
2. 对用户应用 10-core 过滤，删除交互数少于 10 的用户。
3. 按每个用户的时间顺序，将前 70% 交互作为历史兴趣，后 30% 作为未来兴趣。
4. 从未来部分中取未在历史部分出现的类别，形成 future novel interests。
5. 将样本按用户划分为 Train/Validation/Test；论文报告规模如下。

| 数据集 | 来源 | Train | Validation | Test | 二级类别数 |
|---|---|---:|---:|---:|---:|
| Arts | 论文 Table 2 | 20,186 | 2,524 | 2,524 | 143 |
| Arts | 官方发布 JSON 实测 | 20,186 | 2,523 | 2,524 | 143 |
| Grocery | 论文与官方发布 JSON | 34,981 | 4,373 | 4,373 | 172 |

### 3.1 类别层级与 SID 对齐审计

论文按数据集根类目以下的局部树计层：一级类别是父节点，二级类别是其直接子节点，也是 MeRA 实际预测和注册 token 的类别。官方资产给出直接证据：`category_tree*.json` 的 key 是一级父类，所有 value 的并集与 `layer2_map.txt` 的 143/172 个标签完全相等；Train/Test 历史和目标也都只使用这些 value 标签。

Home 数据来自 Amazon metadata 路径 `Home & Kitchen > path[1] > path[2] > ...`。两套命名的准确对应如下：

| 概念 | 论文/官方 MeRA 名称 | 本项目 Home 字段 | 实际取值 |
|---|---|---|---|
| 数据集根 | 数据集/domain root | `ROOT_CATEGORY` | `Home & Kitchen` |
| 父类 | first-level category | `target_level_2` | `path[1]`，如 `Bath` |
| 预测子类 | second-level category / `layer2` | `target_interest` 的子类部分 | `path[2]`，如 `Bath Rugs` |
| 保留结构路径 | category tree edge | `target_interest` | `path[1] > path[2]`，如 `Bath > Bath Rugs` |

因此，**父子关系和预测粒度一致，但层号字面不一致**：论文忽略 domain root 后称一级/二级；本项目字段名按包含 Amazon 根节点的 metadata path 称 L2/L3。后续文档统一使用“父类/预测子类”，不再只写有歧义的“一级/二级”。

严格说，论文没有定义或命名一套 SID 构建算法；它把每个预测子类名称整体注册为一个 special token。为避免把“输出都是一个新增 token”误写成“方法相同”，这里先给出明确判定：

- **本项目原 Interest ID 方法是 Natural-language-in / Interest-ID-out，不是 SID-in/SID-out。**历史始终是自然语言类别或商品标题，具体 `<interest_NNN>` 只出现在监督输出。
- **论文 MeRA 及本 MeRA 适配是 Category-token-in / Category-token-out。**若把每个类别的原子 token 宽泛地称为 SID，可以简称 SID-in/SID-out；但它不是生成式推荐中由多个 codeword 组成的层级/残差 SID，输入输出都只是“一类一 token”。

| 维度 | 论文/官方 MeRA | 本 MeRA Home 适配 | 本项目原 semantic Interest ID |
|---|---|---|---|
| 方法角色 | 待复现论文方法 | 保持 MeRA 机制的 Home 迁移 | 本项目待比较方法 |
| 原子标签空间 | Arts 143 类；Grocery 172 类 | 136 个合法上下文类，其中 112 类可预测 | 112 个可预测类 |
| token 表面形式 | 带引号的子类名称整体注册为一个 special token | 独立稳定映射 `<category_000>`--`<category_135>` | 独立稳定映射 `<interest_000>`--`<interest_111>` |
| 映射依据 | 官方类别名称/类别表 | 136 个完整类别身份排序 | 112 个子类短名词典序 |
| embedding 初始化 | 子类名称的基础模型 token embedding 均值 | 子类短名的基础模型 token embedding 均值 | 子类短名的基础模型 token embedding 均值 |
| 历史输入 | 类别名称在 prompt 中直接命中已注册的原子类别 token | 自然语言指令 + `<category_NNN>` 历史序列 | 自然语言指令 + 自然语言类别历史；不出现 `<interest_NNN>` |
| 训练输出 | 下一个原子类别 token | 下一个 `<category_NNN>` | assistant 文本中的一个 `<interest_NNN>` 及结束标记 |
| 主训练损失 | 全词表 next-token CE + structure InfoNCE + collaborative InfoNCE | 三项组成相同；关系采样与相似度实现不同 | 普通 assistant-only causal-LM SFT，无两项关系损失 |
| 排序推理 | 读取最后位置的全部类别-token logits，直接 Top-K | 读取最后位置的 112 个可预测 `<category_NNN>` logits，直接 Top-K | 无候选 Recall 评测使用 20-beam 文本生成，再将合法 `<interest_NNN>` 映射回类别 |
| 10-way 推理 | 从同一全类别 logits 排名派生 | 从同一 112 类 logits 排名派生，候选不写入 prompt | 候选类别名写入 prompt，并将 beam 约束到对应 10 个 Interest ID |
| 严格 SID-in/SID-out 判定 | 否：原子类别 token，不是多 codeword SID | 否；宽泛口径可称 Category-SID-in/out | 否：Natural-language-in / ID-out |

#### 同一真实 Home 样本

以下样本来自冻结 Train 行 `train:AGQAMTSBEHU5AWH3XKRVN6EC4SPA:1583894821818`。gold 类别均为 `Bath > Bath Rugs`，两种方法看到的是同一条按时间排列的 Interest 历史；差异来自输入编码、标签映射、目标函数和推理方式，而不是样本内容。

**本项目原 Interest ID 方法（Interest view）**

```text
User input:
The topic categories associated with the products I recently rated positively,
in chronological order, are listed below:

1. Vacuums & Floor Care > Vacuums
2. Kitchen & Dining > Kitchen Utensils & Gadgets
3. Furniture > Kids' Furniture
4. Kitchen & Dining > Storage & Organization
5. Kitchen & Dining > Storage & Organization

Predict the Interest ID for the one new topic category that I am most likely
to be interested in next. Respond with only <interest_NNN>.

Assistant target:
<interest_009>
```

这里输入是可读的自然语言类别名；`<interest_009>` 只作为输出标签，按冻结映射还原为 `Bath > Bath Rugs`。模型通过普通 assistant-only SFT 学习该输出。无候选 Recall 排序时，它实际生成 20 条 beam；只有能精确解析为合法 Interest ID 的输出才计入排名。

**本 MeRA Home 适配（同一行）**

```text
User input:
The user's positively rated interest categories, in chronological order, are:

1. <category_130>
2. <category_086>
3. <category_040>
4. <category_089>
5. <category_089>

Predict the one future novel interest category most likely to occur next.
It must not repeat a category from the history.

Next-token target:
<category_001>
```

其中 `<category_130/086/040/089>` 分别对应上面的 `Vacuums`、`Kitchen Utensils & Gadgets`、`Kids' Furniture` 和 `Storage & Organization`，`<category_001>` 对应 gold `Bath Rugs`。这些编号来自包含 24 个“只可作为历史”的类别在内的独立 136 类 MeRA mapping，因此与 `<interest_009>` 没有数值对应关系。训练在 prompt 最后位置直接对 `<category_001>` 做全词表 CE，并额外施加结构与协同关系损失；推理不生成 `<category_001>` 文本，而是一次 forward 后对 112 个可预测类别 token 的 logits 排序。

因此，二者唯一直接同构的部分是：**一个目标类别最终由一个语义初始化的新增 token 表示**。不能据此说两者都是同一种 SID 方法；关键区别是 **Natural-language-in vs. Category-token-in、生成/约束 beam vs. 类别 logits 直接排序、普通 SFT vs. 多关系对齐损失**。本复现保留独立 `<category_NNN>` mapping，正是为了不把论文 baseline 偷换成本项目的 `<interest_NNN>` 方法。

官方发布的 JSON、类目映射、类目树和共现矩阵固定到提交 `c07bdbe57bf82e5c938b478035baee78d630b17c`。全量审计结果如下：

- Train/Validation 每条恰有一个 future novel interest；Test 为多目标集合，Arts 平均 3.024 个、Grocery 平均 3.380 个；
- 全部历史和目标均落在冻结标签空间，且全部样本满足目标不在历史类别集合中；
- 类目树完整且无重复地覆盖全部 143/172 个二级类别；
- 官方 JSON 不含用户 ID，无法验证论文所称的按用户划分；更进一步，按发布字段比较发现 Arts 的 Train/Validation、Train/Test、Validation/Test 分别有 138/19/7 个完全相同样本，Grocery 分别有 177/16/10 个，因此发布资产不能作为 user-disjoint 的证据；
- Arts 论文 Validation 数为 2,524，发布文件实际为 2,523，当前公开资产无法消除该一条样本差异。

因此，论文数据主线命名为“官方发布资产复现”，不声称重新完成了原始评论级数据切分。Home & Kitchen 扩展则继续使用带用户 ID 且已经验证 Train/Dev 用户零交集的本项目冻结数据。

### 3.2 类别 Balance 审计

论文的 10-core 是用户/交互稠密度过滤，不是类别均衡。官方发布代码只对 Train 做普通 `shuffle(seed)`，Transformers Trainer 使用常规 sampler；代码中没有按类别过采样、欠采样、`WeightedRandomSampler`、class-weighted CE 或 focal loss。发布 Train 的目标频次也直接证明其保留自然长尾：

| 数据集 | Train 行/Gold | 标签空间 | Train 目标出现类 | 零目标类 | 非零最小/中位/最大 | 全标签 CV | 归一化熵 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arts | 20,186 | 143 | 115 | 28 | 1 / 69 / 1,403 | 1.673 | 0.803 |
| Grocery | 34,981 | 172 | 141 | 31 | 1 / 58 / 2,084 | 2.083 | 0.752 |
| Home | 112,000 | 112 个预测类 | 112 | 0 | 1,000 / 1,000 / 1,000 | 0.000 | 1.000 |

Arts Test 只有 `Shop Upholstery Fabrics` 未作为 Train 目标出现，Grocery Test 只有 `Breakfast Syrups & Toppings`；二者都在 Train 历史中出现，因此仍可通过输入 token 和关系约束学习。逐类完整计数保存在 `data/{dataset}/target_distribution.json`。

对齐规则如下：

1. Arts/Grocery 主复现必须保留官方自然频次，不做 balance；否则训练分布已被改变，不能与论文结果直接比较。
2. Home 的 112K 本来就是每类 1,000 条的冻结均衡 Train。跨数据集绝对分数同时受分布差异影响，不能只归因于方法。
3. 仅增加一组 balance 消融：Arts + Qwen2-1.5B。它保持总行数 20,186 不变，只对 115 个已出现目标类做确定性均匀重采样，最终每类 175 或 176 条；28 个零目标类不合成伪样本。Validation/Test、关系图、训练步数和全部超参数均不变。

## 4. Home & Kitchen 统一比较数据

Home & Kitchen 扩展直接复用本项目冻结数据，不重新切分或采样：

- Train：112,000 条，112 个预测子类，每类 1,000 条；
- Dev pool：596,449 条，Train/Dev 用户交集为 0；
- 正式评测：固定 2,000 条，112 类每类 17--18 条；
- 历史：按时间排列的正反馈交互，模型可见最近 50 条；
- 新兴趣：目标预测子类不在目标前完整历史中。

MeRA 的结构树使用现有 `父类(L2) > 预测子类(L3)` 边。Train 的预测目标支持集为 112 类，但历史上下文实际覆盖 136 类；因此 136 个上下文类别均注册原子 token 并进入结构/协同关系，推理 logits 严格限制在 112 个冻结目标类别。token 语义初始化只编码 L3 子类短名，父类只用于保存完整类别身份和构造 sibling 关系。协同矩阵只由 112K Train 的完整可见类别历史与目标类别构造，禁止使用 Dev 或评测标签。

为与现有 baseline 公平比较，Home & Kitchen 主结果固定使用 **Interest 视图**，即输入只包含按时间排列的历史二级类别；Behavior 与 Behavior + Interest 可作为额外视图，但不得替代 Interest 主结果。

## 5. 官方代码审计

审计锚点为官方提交 `c07bdbe57bf82e5c938b478035baee78d630b17c`。

下表的“机制适配处理”描述旧 Qwen3.5/Llama Arts/Grocery 与 Home 所用的 **历史非等价实现**。不等价不只来自基础模型版本，还来自关系损失公式与采样、可训练参数、CE 监督范围、batch、调度器和数值精度。Qwen2 复现与新的 Qwen3.5/Llama 三数据集主线均不采用这些替代实现，而是从固定官方源码执行；其精确合同见第 6.0--6.2 节。

| 项目 | 论文/意图 | 官方代码现状 | 机制适配处理（非等价） |
|---|---|---|---|
| 基础模型 | Qwen2-1.5B | 启动脚本为本地 `Qwen2-1.5B`，未公开 revision | 机制适配矩阵为 Qwen3.5/Llama；复现固定 `Qwen/Qwen2-1.5B@8a16abf...` |
| 类别 token | 每个预测子类一个 token | 带引号的子类名注册为 special token | 保留单 token 语义，使用 baseline 独立的稳定显式 token ID |
| token 初始化 | 类别名称原 token embedding 均值 | `train.py` 中实现均值初始化 | 保留，并增加逐类初始化审计 |
| 结构正则 | sibling 正样本、非 sibling 负样本 | 每步按父类采一对正样本，使用未归一化点积 | 每个有效类别作为 anchor，使用 cosine InfoNCE；关系语义相同，但采样粒度和相似度函数不等价 |
| 协同正则 | 共现关系正负采样 | 从未阈值化 `.npy` 计数矩阵按概率采样 | 使用论文 $T=0.05$ 条件概率阈值并将阈值外类别作为负例；数据来源相同，但采样分布不等价 |
| 作用参数 | embedding 与 LM head | `contrastive_module=both` | 保持 `both` |
| LoRA | 论文未完整列出 | r=16、alpha=16、dropout=0.05、all-linear；整个 embedding 与 LM head 作为 `modules_to_save` | Transformer 保留同一 LoRA；只训练新增 input rows，LM head 使用 LoRA；参数化不等价 |
| 训练轮数 | 最多 10，patience=2 | 10 epoch、early stopping | 保持；保存并恢复最佳 checkpoint |
| 学习率 | $10^{-4}$ | $10^{-4}$ | 保持 |
| 截断长度 | 论文未披露 | 512 | 保持并报告截断率 |
| 输出 token 监督 | 预测类别 token | 类别 token 与随后 EOS 均进入 causal-LM loss | 只在 prompt 末位监督类别 token；主分类目标一致，EOS 辅助监督不同 |
| 调度器 | 论文未披露 | 默认 linear，`warmup_steps=20` | cosine，`warmup_ratio=0.03` |
| 推理 | 扩展 token logits 直接 Top-K | 实现为附加 special tokens 上 Top-K | 每个源样本一次前向，在冻结预测类别 token ID 上精确排序 |
| Grocery 入口 | 应可训练 | `train_grocery.sh` 调用仓库中不存在的 `train_with_extended_tokenizer.py` | 使用经测试的统一训练入口，记录为发布缺陷修复 |
| 共现阈值 | 论文描述 $T=0.05$ | notebook 与训练代码均未应用阈值；发布矩阵可由 Train 集合共现计数逐元素精确重建 | 论文模式显式应用 $T$；另保留 `official_code` 无阈值复核分支 |
| 推理实现 | 类别 token Top-K | `inference.py` 使用不存在的 `output.indice` 属性 | 数学等价修复为 `torch.topk(...).indices`，并增加标签空间断言 |

官方仓库和论文还存在配置冲突：论文正文写 batch size 36、$\alpha=0.05$、$\beta=0.1$；发布脚本写 `batch_size=256`、`micro_batch_size=36`，Arts 使用 $\alpha=\beta=0.01$，Grocery 使用 $\alpha=\beta=0.1$。复现主线按“固定官方代码的实际执行语义”处理：有效 batch 为 $36\times7=252$，Arts 使用 0.01/0.01，Grocery 使用 0.1/0.1。以下规则只属于旧 Qwen3.5/Llama Arts/Grocery 与 Home 的历史 MeRA 机制适配：

1. Arts/Grocery 机制适配使用论文明确值：有效 batch 36、$\alpha=0.05$、$\beta=0.1$；负样本数分别为 5/10。
2. Home & Kitchen 不在正式 Dev-2K 上调参；使用有效 batch 32、1 epoch、$\alpha=0.05$、$\beta=0.1$ 和 10 个负样本，与本项目 112K 主训练预算一致。
3. Qwen3.5 显式关闭 thinking，Llama 使用其原生 chat template。两模型都使用标准 Transformers forward、bf16、gradient checkpointing 和 4×A100；Hub kernels 因与镜像 Torch 2.8 不兼容而禁用。
4. 为适配 8B/9B 模型，只训练新增类别 token 的输入 embedding 行；Transformer 与 LM head 使用 LoRA。关系损失读取包含 LoRA 增量的有效 LM-head 行，仍同时约束 `embed_tokens + lm_head`，但这只是模块覆盖一致，不代表与官方整块参数训练、采样方式和相似度函数数值等价。

## 6. 推荐复现实验设置

### 6.0 Qwen2-1.5B official-code strict 主线

| 配置 | Arts strict | Grocery strict |
|---|---|---|
| 数据 | 官方 Arts Train 20,186 / Valid 2,523 / Test 2,524 | 官方 Grocery Train 34,981 / Valid 4,373 / Test 4,373 |
| 官方代码 | `GabyUSTC/MeRA@c07bdbe...` 的 `train.py` | 同一 `train.py`；绕过发布 shell 中不存在的入口文件 |
| 基础模型 | `Qwen/Qwen2-1.5B@8a16abf...` | 同左 |
| 类别 token | 官方带双引号完整子类名；只对 input 行做语义初始化；input/output untied | 同左 |
| CE | 官方 shifted causal CE，监督类别 token 与 EOS | 同左 |
| 关系计算 | 官方 raw-dot、原始 `.npy` 与原采样函数 | 同左 |
| structure / collab | 0.01 / 0.01 | 0.1 / 0.1 |
| 负样本 | 5 | 10 |
| batch | micro 36、gradient accumulation 7、实际有效 batch 252 | 同左 |
| PEFT | r=16、alpha=16、dropout=0.05；七类 Transformer linear；完整 `embed_tokens` 与 `lm_head` 为 `modules_to_save` | 同左 |
| 优化 | AdamW Torch、linear、warmup 20、最多 10 epoch、patience 2 | 同左 |
| 精度与依赖 | 8-bit base + fp16；Torch 2.3.0、Transformers 4.37.0、PEFT 0.9.0、bitsandbytes 0.43.3 | 同左 |
| 推理 | 官方 `inference.py`，仅修复 `.indice` 为 `.indices`；完整 143 类排序 | 完整 172 类排序 |

strict 路径不复用本项目 `MeRATrainer`。`prepare_official_strict.py` 逐文件校验官方源码和数据 SHA，runner 只允许固定模型 revision，并保存模型文件 SHA、环境锁、训练合同与唯一源码 diff。评测同时运行官方循环公式和项目 `paper_metrics`，任一指标不一致即失败。

这里的 strict 精确定义为**固定公开代码与公开资产的严格执行复现**。作者本地 `Qwen2-1.5B` 目录 checksum、容器 digest和原 trainer state 未发布，因此无法证明历史运行的 bitwise replay；这些不可观测资产不再由本项目默认值冒充。旧 `qwen2-arts-natural/balanced` 仅为非严格诊断结果。

### 6.1 作者澄清协议主线

作者澄清协议只修改发布代码与真实实验不一致的训练控制：`generate_prompt_without_prompt` 将历史类别按原顺序拼为连续双引号类别 token，并在末尾追加 gold 类别；训练固定完整 5 epochs，`callbacks=[]`、`load_best_model_at_end=False`。其余数据、类别映射、关系资产、loss、采样、batch、优化器、scheduler、warmup、精度和官方 FP16/no-mask 推理保持发布实现。正式 runner 在 `trainer.train()` 返回后记录 `training-completion.json`，并强制验证 `epoch=5` 与 `global_step>0`。

Arts 的作者回复参考值为 R@1/2/3 `0.1209/0.2149/0.2843`、P@1/2/3 `0.2987/0.2712/0.2415`。这些数值仅用于 Qwen2 Arts 对齐，不外推到 Grocery、其他基础模型或 Home。作者未提供 Grocery 澄清协议结果，因此 Grocery 只报告本项目实测值。

### 6.2 作者澄清协议的跨模型实现

Qwen3.5-9B 与 Llama-3.1-8B-Instruct 的新主线由 `GabyUSTC/MeRA@c07bdbe...` 的同一 `train.py`/`inference.py` 生成，不复用本项目旧 `MeRATrainer`。除基础模型及其直接引起的 tokenizer、embedding API、8-bit/PEFT/Trainer API、设备放置和动态 logits 维度兼容外，训练和推理合同与第 6.1 节一致。LoRA 精确覆盖模型全部 decoder linear projections：Qwen3.5 为 248 个，Llama 为 224 个；两个模型都将完整 `embed_tokens` 与 `lm_head` 作为 `modules_to_save`。训练级 preflight 必须真实完成一次 optimizer step，且 structure/collaborative loss 均至少调用一次，随后保存 adapter、在新模型实例中重载并得到有限 logits；任一合同失败都不会提交正式任务。

该设置表示**在基础模型替换条件下对官方 MeRA 方法的等价实现**，不表示不同基础模型之间的数值结果应相等，也不声称与作者未公开环境进行 bitwise replay。正式结果将写入独立单元格，不覆盖下节旧机制适配结果。

### 6.3 Arts/Grocery 历史 MeRA 机制适配（非等价实现）

| 配置 | 设置 |
|---|---|
| 数据 | 官方 Arts、Grocery 冻结 JSON |
| 模型 | `Qwen/Qwen3.5-9B`、`NousResearch/Meta-Llama-3.1-8B-Instruct` |
| 输入 | 同一 category history 语义；按各模型 chat template 编码 |
| 输出 | 每个预测子类一个扩展 token |
| 语义初始化 | 子类别名称原 token embedding 均值 |
| LoRA | r=16，alpha=16，dropout=0.05，all-linear（含 LM head） |
| 可训练附加参数 | 新类别 input embedding 行；LM head 为 LoRA |
| 训练目标 | 全词表 next-token CE + structure + collaborative |
| 学习率 | $10^{-4}$ |
| Scheduler | cosine |
| Epoch | 最多 10，validation patience=2 |
| 有效 batch | 36 |
| 最大长度 | 512；报告截断率 |
| $\alpha,\beta$ | 0.05、0.1 |
| 负样本数 | Arts 5，Grocery 10 |
| 推理 | 类别 token logits 直接 Top-K，无 beam search |
| 指标 | 论文 R/P@1/2/3/4 + 统一 Query-Mean Accuracy/Recall/nDCG |
| 随机种子 | 2024 |

### 6.4 Home & Kitchen 统一比较主线

| 配置 | 设置 |
|---|---|
| 数据 | 冻结 112K Train + 同一 Dev-2K |
| 模型 | `Qwen/Qwen3.5-9B`、`NousResearch/Meta-Llama-3.1-8B-Instruct` |
| 主输入视图 | Interest |
| 标签 | 136 个上下文 token；其中 112 个可预测目标类 |
| 结构关系 | 136 类中同一父类下的预测子类 sibling |
| 协同关系 | 仅由 112K Train 的 136 类集合共现构造 |
| 训练参数 | 1 epoch、有效 batch 32、LoRA alpha=32；其余关系参数同上 |
| 10-way 分类 | 同一 2K 候选集合，在 10 个候选 token logits 中取 Top-1 |
| Candidate-free ranking | 在全部 112 个类别 token logits 上直接 Top-20 |
| 共同主指标 | Query-Mean Accuracy、Recall@10/20、nDCG@10/20 |
| Home 对齐指标 | 单 gold 类别 Macro Accuracy、Recall@10/20、nDCG@10/20 |
| 完整性门禁 | 2,000/2,000、112 类、missing=0、重复 ID=0 |

为隔离 MeRA 方法增益，Home & Kitchen 至少运行下列四个严格消融，所有条件共享同一数据、模型、训练预算和评测器：

| 方法 | 语义初始化 | 结构正则 | 协同正则 |
|---|---:|---:|---:|
| Category-token SFT | 是 | 否 | 否 |
| MeRA w/o structure | 是 | 否 | 是 |
| MeRA w/o collab | 是 | 是 | 否 |
| MeRA | 是 | 是 | 是 |

`w/o pooling` 使用高斯初始化，只作为论文消融复现，不作为主要 baseline。

## 7. 统一评测定义

三套数据统一从同一次全预测标签 logits 排名派生项目指标。Home & Kitchen 每个源样本只有一个 gold；Arts/Grocery Test 是多 gold。正式评测不随机抽取一个 gold，也不把 `(源样本, gold)` 当作彼此独立的最终统计样本，因为前者丢失相关标签并引入随机方差，后者会让 gold 较多的用户获得更高权重。

这里的 **Query** 指数据文件中的一条原始测试请求，即“一段用户历史及其完整未来新兴趣集合”，不是搜索关键词，也不是一个单独 gold。Arts/Grocery 一条 query 可含多个 gold；Home 一条 query 恰有一个 gold。`Query-*` 表示先在每条原始请求内部算分，再让每条 query 以相同权重进入全局平均：

$$
\operatorname{QueryMean}(m)=\frac{1}{|Q|}\sum_{q\in Q}m_q.
$$

| 报告名称 | 单条 query 的定义 | 聚合与取值 | 直观含义 |
|---|---|---|---|
| 论文 `R@K` | $\lvert\pi_{q,1:K}\cap G_q\rvert/\lvert G_q\rvert$ | query 等权均值，0--1 | 完整 gold 集中有多少比例进入前 $K$ |
| 论文 `P@K` | $\lvert\pi_{q,1:K}\cap G_q\rvert/K$ | query 等权均值，0--1 | 前 $K$ 个预测中有多少比例是 gold |
| `Query Recall@K` | 与论文 `R@K` 相同的 set recall | query 等权均值，0--1 | 统一表使用的集合召回；相同 $K$ 时必须等于论文 `R@K` |
| `Query nDCG@K` | 二值相关性的 $\mathrm{DCG@K}/\mathrm{IDCG@K}$ | query 等权均值，0--1 | 不仅看是否召回，还奖励 gold 排得更靠前 |
| `Query Accuracy` | 每个 gold 各建一个 10-way 候选集，判断该 gold 是否为候选内最高分；先在 query 内对多个 gold 平均 | 再对 query 等权平均，表中为百分数 | 候选条件下的 10 选 1 准确率；不是 143/172 类全局 Top-1 |
| `Macro Accuracy` | Home 单 gold 的 10-way 正确率 | 先按 gold 类别平均，再对 112 类等权平均，百分数 | 消除各类别样本数差异后的分类准确率 |
| `Macro Recall@K` | Home 单 gold 下退化为 $\mathbb 1[\operatorname{rank}(g)\le K]$ | 先类内、再 112 类等权平均，0--1 | 各类别等权的 Hit/Recall@K |
| `Macro nDCG@K` | Home 单 gold 命中时为 $1/\log_2(\operatorname{rank}(g)+1)$，否则为 0 | 先类内、再 112 类等权平均，0--1 | 各类别等权且考虑正确类别排名位置 |

因此，Qwen2 Arts Natural 的 `Query Accuracy=46.03%` 与 `P@1=0.1478` 并不矛盾：前者只要求 gold 在为它构造的 10 个候选中得分最高，后者要求全 143 类排序的第一名直接命中任一 gold。`source_sample_count`、`target_count_min/max/mean` 和 `gold_evaluation_count` 是完整性统计，不是效果指标；其中 `gold_evaluation_count` 只表示 10-way 内部一共评了多少个 `(query, gold)` 判断。

对源 query $q$，令去重 gold 集为 $G_q$，完整预测排序前 $K$ 项为 $\pi_{q,1:K}$。采用标准二值相关性集合指标：

$$
\operatorname{Recall@K}_q
=\frac{|\pi_{q,1:K}\cap G_q|}{|G_q|},
$$

$$
\operatorname{DCG@K}_q
=\sum_{r=1}^{K}\frac{\mathbb 1[\pi_{q,r}\in G_q]}{\log_2(r+1)},
\qquad
\operatorname{IDCG@K}_q
=\sum_{r=1}^{\min(K,|G_q|)}\frac{1}{\log_2(r+1)},
$$

$$
\operatorname{nDCG@K}_q
=\frac{\operatorname{DCG@K}_q}{\operatorname{IDCG@K}_q}.
$$

最终对源 query 等权平均。这样所有 gold 都排在最前时 nDCG 为 1，且在 Home 单 gold 情况下严格退化为原公式。10-way Accuracy 为每个 gold 构造一组候选，排除完整历史和同源其他 gold，并确定性采样最多 3 个同父类负例；随后先在同一 query 的 gold 内平均正确率，再对 query 等权平均。Home 逐条复用既有冻结候选，不重新采样。

Arts/Grocery 的 2,524/4,373 个源 Test query 分别含 7,633/14,782 个 gold 判断；后者只用于 10-way 内部明细和完整性检查，不决定 query 权重。Home 为 2,000 个源 query、2,000 个 gold。Home 另报告按 112 个 gold 类别等权的类别 Macro 指标，仅用于与冻结 benchmark 对齐，不作为三数据集共同聚合口径。

论文 Arts/Grocery 的一个用户样本可能包含多个 future novel interests，因此严格按官方脚本使用集合 Recall/Precision：

$$
R@K=\frac{|\hat C_{1:K}\cap C^+|}{|C^+|},
\qquad
P@K=\frac{|\hat C_{1:K}\cap C^+|}{K}.
$$

论文集合 Recall 与统一 Query-Mean Recall 使用同一集合定义，因此在相同 $K$ 上必须逐值一致；论文 Precision 继续单独报告。统一 nDCG、10-way Accuracy 和 Home 类别 Macro 是额外指标，但全部严格复用同一次源样本前向和同一全标签排序。

## 8. 验证与验收

正式训练前必须通过：

1. 官方 commit、训练/推理源码及 12 个 Arts/Grocery 数据文件逐文件 SHA；
2. Home & Kitchen 112K/2K 冻结文件 SHA 和用户隔离检查；
3. 每个类别 token 恰好编码为一个唯一 token ID；
4. 语义初始化逐类与原 token embedding 均值一致；
5. 结构正样本必为 sibling，负样本必为非 sibling；
6. 协同矩阵只由 Train 构造，矩阵顺序与 token 顺序一致；
7. `structure=0, collab=0` 时跳过两项关系计算并严格退化为 category-token SFT；
8. 10-way 与 candidate-free 评测共享样本顺序和 gold；
9. Top-K 只包含冻结标签空间且无重复；
10. 正式结果必须满足完整性门禁，smoke 结果不得入表。
11. strict 环境精确匹配 Python 3.10、Torch 2.3.0、Transformers 4.37.0、PEFT 0.9.0、Accelerate 0.29.3 与 bitsandbytes 0.43.3；
12. strict 模型从固定 Qwen2 revision 以 8-bit 成功加载，additional-special-token 集恰等于 143/172 个类别 token；
13. strict 输出每条是完整、无重复、无缺失的全类别排序，且官方与项目两套 R/P 实现逐值一致。

## 9. 结果表

> 空白表示正式结果尚未产出。
> 数据集、模型和方法三级合并单元格总表见 [RESULTS.md](results.md)；本节保留来源说明与诊断分析。

### 9.1 论文数据复现

| 数据集 | 方法 | R@1 ↑ | R@2 ↑ | R@3 ↑ | P@1 ↑ | P@2 ↑ | P@3 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Arts | 论文结果 | 0.1204 | 0.2120 | 0.2786 | 0.2896 | 0.2670 | 0.2413 |
| Arts | 复现结果 | 0.0786 | 0.1485 | 0.2046 | 0.2021 | 0.1882 | 0.1783 |
| Arts | Qwen3.5-9B 机制适配（非等价） |  |  |  |  |  |  |
| Arts | Llama-3.1-8B 机制适配（非等价） | 0.1284 | 0.2089 | 0.2781 | 0.3031 | 0.2617 | 0.2392 |
| Grocery | 论文结果 | 0.1179 | 0.1980 | 0.2662 | 0.3206 | 0.2734 | 0.2515 |
| Grocery | 复现结果 | 0.0503 | 0.0933 | 0.1406 | 0.1180 | 0.1108 | 0.1126 |
| Grocery | Qwen3.5-9B 机制适配（非等价） |  |  |  |  |  |  |
| Grocery | Llama-3.1-8B 机制适配（非等价） |  |  |  |  |  |  |

官方仓库还发布了各样本 Top-4 预测。使用官方 `analyze_output.py` 的集合定义独立复算后，Arts 得到 R@1/2/3/4 = 0.1204/0.2120/0.2786/0.3288、P@1/2/3/4 = 0.2896/0.2670/0.2413/0.2183，与论文一致；Grocery 得到 R@1/2/3/4 = 0.1175/0.1989/0.2635/0.3224、P@1/2/3/4 = 0.3174/0.2752/0.2491/0.2310，与论文表不一致。该差异来自同一初始提交中的发布产物，Git 历史没有可供追踪的中间版本，因此不能用发布预测替代 Grocery 论文参考值。

### 9.2 三数据集统一比较（源 Query 等权）

| 数据集 | 模型 | Query Accuracy (%) ↑ | Query Recall@10 ↑ | Query Recall@20 ↑ | Query nDCG@10 ↑ | Query nDCG@20 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Arts | Qwen2-1.5B official-code strict | 46.16 | 0.4641 | 0.6582 | 0.3162 | 0.3805 |
| Arts | Qwen3.5-9B |  |  |  |  |  |
| Arts | Llama-3.1-8B | 53.67 | 0.5637 | 0.7466 | 0.4118 | 0.4724 |
| Grocery | Qwen2-1.5B official-code strict | 36.63 | 0.3666 | 0.5965 | 0.2315 | 0.3091 |
| Grocery | Qwen3.5-9B |  |  |  |  |  |
| Grocery | Llama-3.1-8B |  |  |  |  |  |
| Home & Kitchen | Qwen3.5-9B | 13.50 | 0.1765 | 0.2980 | 0.0886 | 0.1192 |
| Home & Kitchen | Llama-3.1-8B | **14.30** | **0.1845** | **0.3160** | **0.0899** | **0.1230** |

### 9.3 Home & Kitchen 类别宏对齐

| 模型 | Macro Accuracy (%) ↑ | Macro Recall@10 ↑ | Macro Recall@20 ↑ | Macro nDCG@10 ↑ | Macro nDCG@20 ↑ |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 13.51 | 0.1766 | 0.2982 | 0.0888 | 0.1193 |
| Llama-3.1-8B | **14.30** | **0.1845** | **0.3160** | **0.0900** | **0.1230** |

### 9.4 Home & Kitchen 历史机制适配消融（类别宏，非等价实现）

| 模型 | 方法 | Macro Accuracy (%) ↑ | Macro Recall@10 ↑ | Macro Recall@20 ↑ | Macro nDCG@10 ↑ | Macro nDCG@20 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | Category-token SFT | 13.71 | 0.1754 | 0.3080 | 0.0867 | 0.1199 |
| Qwen3.5-9B | MeRA w/o structure | 14.10 | 0.1846 | 0.3064 | **0.0953** | 0.1257 |
| Qwen3.5-9B | MeRA w/o collab | 12.91 | 0.1720 | 0.2943 | 0.0858 | 0.1166 |
| Qwen3.5-9B | MeRA | 13.51 | 0.1766 | 0.2982 | 0.0888 | 0.1193 |
| Llama-3.1-8B | Category-token SFT | **15.10** | 0.1895 | 0.3201 | **0.0961** | 0.1287 |
| Llama-3.1-8B | MeRA w/o structure | **14.39** | **0.1898** | **0.3325** | 0.0935 | **0.1290** |
| Llama-3.1-8B | MeRA w/o collab | 13.95 | 0.1881 | 0.3181 | 0.0924 | 0.1248 |
| Llama-3.1-8B | MeRA | 14.30 | 0.1845 | 0.3160 | 0.0900 | 0.1230 |

### 9.5 Home & Kitchen official-code extension 2×2 消融

本节从固定官方 commit `c07bdbe57bf82e5c938b478035baee78d630b17c` 执行 Home 扩展实验。双模型均使用 112,000 条 Train、2,000 条 user-disjoint Validation 和冻结 Dev-2K Test；136 个合法历史类全部参与官方类别 token 与关系约束，112 个 Train target 类构成预测空间。除 structure/collaborative 权重外，官方 shifted CE、raw-dot 关系损失与采样、完整 `embed_tokens`/`lm_head`、LoRA、有效 batch 252、优化器、调度器、epoch、seed 和评测器均固定。

| 模型 | 条件（structure, collaborative） | Macro Accuracy (%) ↑ | Macro Recall@10 ↑ | Macro Recall@20 ↑ | Macro nDCG@10 ↑ | Macro nDCG@20 ↑ |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | SFT `(0, 0)` | 9.77 | **0.1073** | **0.1940** | **0.0462** | **0.0679** |
| Qwen3.5-9B | Structure `(0.05, 0)` | **10.63** | 0.0959 | 0.1890 | 0.0427 | 0.0661 |
| Qwen3.5-9B | Collaborative `(0, 0.1)` | 9.33 | 0.0938 | 0.1901 | 0.0414 | 0.0655 |
| Qwen3.5-9B | Both `(0.05, 0.1)` | 10.33 | 0.0959 | 0.1866 | 0.0431 | 0.0659 |
| Llama-3.1-8B | SFT `(0, 0)` | 10.32 | 0.0878 | 0.1841 | 0.0412 | 0.0654 |
| Llama-3.1-8B | Structure `(0.05, 0)` | **11.66** | **0.1231** | **0.2209** | **0.0570** | **0.0813** |
| Llama-3.1-8B | Collaborative `(0, 0.1)` | 10.90 | 0.0928 | 0.1935 | 0.0441 | 0.0691 |
| Llama-3.1-8B | Both `(0.05, 0.1)` | 10.87 | 0.0955 | 0.1911 | 0.0422 | 0.0660 |

正式来源为 `mera-official-home-eval-recovery-r3-20260902`。八个作业全部为 `pass`；每份 `metrics.json` 均含 2,000 个 source、136 个上下文类、112 个预测类和 112 个类别宏分组。恢复评测只读复用正式训练产物，八份 `recovery-contract.json` 均记录 `training_executed=false`，因此 API 兼容修复没有引入重训差异。

### 9.6 作者澄清协议正式结果

下表只接收固定 5 epochs、无 early stopping、无 best-checkpoint 回载且训练 prompt 为“历史类别 token 串 + gold 类别”的正式产物。空白表示对应正式结果尚未产出。

当前状态：Qwen3.5/Llama r2 主矩阵六项均为 `pass`，每项均已输出完整正式 `metrics.json`。Qwen2 Arts/Grocery 已分别达到计划 `400/400` 与 `690/690` optimizer step 并保存完整最终 adapter；旧完成合同中的浮点 epoch 为 4.991087/4.969136，属于 Transformers 4.37 在达到 `max_steps` 时的显示偏差。验收现改为 `global_step == max_steps`。r3 因 recovery heredoc 缩进错误失败后，r4 继续只读复用同一最终 adapter；Arts r4 已为 `pass`，Grocery r4 仍在运行，因此只有 Grocery 单元格暂时留空。

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

其中只有 Qwen2-1.5B 与论文使用同一基础模型，构成直接复现对照；Qwen3.5/Llama 行是按作者协议完成的基础模型替换扩展。作者另提供的 Arts Qwen2 参考值为 R@1/2/3 `0.1209/0.2149/0.2843`、P@1/2/3 `0.2987/0.2712/0.2415`，作为补充参考，不替换论文表值。

Qwen2 Arts r4 来源为 `mera-author-qwen2-arts-eval-recovery-r4`（Backend ID `<archived-job-id>`）。该任务只读复用 400/400 step 的最终 adapter，状态为 `pass`；评测覆盖 2,524 个 source 和完整 143 类排名，每行均通过长度、去重及标签全集门禁，且项目实现与官方循环公式逐项一致。相对论文表值，R@1/2/3 差值为 `-0.0024/+0.0055/+0.0010`，P@1/2/3 为 `-0.0020/+0.0016/-0.0037`，六项最大绝对差为 `0.0055`。

Home 继续使用冻结 Dev-2K、136 个上下文类到 112 个预测类的特殊评测合同。下表只比较完整 MeRA `both` 与同模型 Our Method 的正式最优 **B+I 输入 + Semantic Interest ID 输出**；该版本在两个模型上均为 Behavior、Interest、B+I 三视图的四项排名指标最优。所有 Recall/nDCG 均先在目标类内平均、再对 112 类等权平均。MeRA 的 Qwen3.5/Llama Macro Accuracy 分别为 16.00%/15.34%。

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
			<td><strong>0.1944</strong></td><td><strong>0.3221</strong></td><td><strong>0.0988</strong></td><td>0.1307</td>
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
			<td>0.1889</td><td><strong>0.3264</strong></td><td>0.0964</td><td><strong>0.1306</strong></td>
		</tr>
		<tr>
			<td>Our Method</td>
			<td>B+I → Semantic Interest ID</td>
			<td><strong>0.2217</strong></td><td><strong>0.3666</strong></td><td><strong>0.1147</strong></td><td><strong>0.1511</strong></td>
		</tr>
	</tbody>
</table>

Home 点估计中，使用正确的 B+I 最优版本后，Our Method 在 8 项“模型 × 指标”比较中全部高于 MeRA。Qwen 的 Our - MeRA 差值依次为 `+0.0343/+0.0504/+0.0186/+0.0228`；Llama 为 `+0.0328/+0.0401/+0.0183/+0.0205`。Our Method 来源作业为 `qwen35-behavior-interest-interest-id-semantic-init` 与 `llama31-behavior-interest-interest-id-semantic-init`，均记录 `view=behavior_interest`、`label_mode=interest_id` 且为 `pass`。由于没有置信区间或显著性检验，这些差值不解释为总体显著优劣。

Qwen3.5 Home 正式值来自 `mera-author-qwen35-home-fastlog-r1-20260907`，它只抑制 bitsandbytes 重复日志，未修改训练计算，并于 2026-09-07 18:30 CST 首先通过完整验收。保留的主矩阵独立运行随后得到 Macro Accuracy 13.76%、Recall@10/20 0.1855/0.3180、nDCG@10/20 0.0941/0.1273。两次训练合同、revision、seed 与超参数一致，但四卡 8-bit 训练不是 bitwise deterministic；依据提交前约定采用首个完整验证结果，同时保留重复运行差异作为稳定性限制。

### 9.7 阶段性结论

旧 Qwen2 Arts adaptation 的固定预算均衡消融相对 Natural 下降 23.83 个 Accuracy 百分点，Recall@10/20 分别下降 0.2256/0.2931；该结果只说明均衡重采样不适合作为旧 adaptation 的修复，不能替代 official-code strict。Qwen2 strict 已完整产出，但 Arts/Grocery 的 R/P@1--3 均低于论文表值；该差异是在官方源码、公开数据、固定模型 revision 和锁定软件栈下观察到的，不得用旧 adaptation 的诊断结果替代。

Home official-code 2×2 的受控结果显示，关系损失的收益依赖模型和指标。Qwen 的 structure-only 将 Macro Accuracy 从 SFT 的 9.77% 提高到 10.63%（+0.86 pp），但 SFT 在 Recall@10/20 与 nDCG@10/20 四项排名指标上均最优；完整 Both 也未超过 SFT 的四项排名指标。Llama 的 structure-only 则在五项指标上均为同模型最优：相对 SFT，Accuracy 提高 1.34 pp，Recall@10/20 提高 0.0353/0.0368，nDCG@10/20 提高 0.0158/0.0159。Collaborative-only 和 Both 在两个模型上均没有形成五指标一致优势。该结论只适用于当前 Home split、两个指定模型和官方权重，不外推为关系损失的一般性无效结论。

### 9.8 旧 Qwen2-1.5B Arts Natural 与官方结果差异定位

该早期本地改写实现的正式结果如下，仅用于排错，不是复现结果：

| 训练分布 | R@1 | R@2 | R@3 | P@1 | P@2 | P@3 |
|---|---:|---:|---:|---:|---:|---:|
| 自然分布 | 0.0534 | 0.0997 | 0.1479 | 0.1478 | 0.1359 | 0.1337 |
| 类别均衡 | 0.0079 | 0.0225 | 0.0372 | 0.0273 | 0.0323 | 0.0353 |

#### 已排除的口径问题

官方发布 `arts_resutls.json`、官方 Test 与本复现排名可按原顺序逐条对齐，共 2,524 个 source query。本复现每条均包含完整 143 类、无重复且不遗漏 gold；论文 `R/P@K` 也使用官方 `analyze_output.py` 的同一集合公式。因此，`R@1 0.1204 -> 0.0534`、`R@4 0.3288 -> 0.1919` 的下降不是由 Test 文件、标签 mapping、解析器或指标分母不同造成的。

正式训练也不是空跑：训练在 step 2,000、epoch 3.565 因连续两次无改进早停，最佳 Validation loss 为 4.1514（step 1,500），最终评测使用 `load_best_model_at_end` 恢复后的 adapter。正式输出与官方输出的 Top-1 完全一致率仅 15.17%，Top-4 槽位集合平均重合率为 37.75%，说明差异来自模型打分本身。

#### 可直接观测的主要失分模式

| 诊断 | 官方发布预测 | 本复现 | 本复现过滤历史后的反事实 |
|---|---:|---:|---:|
| Top-1 为历史已见类别 | 0.00% | 40.77% | 0.00% |
| Top-4 槽位为历史已见类别 | 0.44% | 28.63% | 0.00% |
| `R@1` | 0.1204 | 0.0534 | 0.0853 |
| `R@4` | 0.3288 | 0.1919 | 0.2444 |

gold 按定义不在历史中，因此本复现的 1,029 个历史复制 Top-1 全部必错；其中 318 个直接复制最后一个历史类别，503 个复制历史中的最高频类别。只在评测后处理中删除历史类别，就能恢复 `R@1 +0.0318` 和 `R@4 +0.0525`，分别解释官方与本复现 Recall 缺口的 47.5% 和 38.4%。但过滤后仍低于官方，且官方推理代码本身也没有 history mask，所以该反事实只用于定位症状，不能作为正式修复或替代结果。

#### 高风险实现差异

| 优先级 | 差异 | 定量证据与可能影响 | 当前结论强度 |
|---|---|---|---|
| 1 | 输入/输出类别表示的绑定方式 | Qwen2 原配置 `<archived-job-id>=true`。官方在 resize 后只替换 `embed_tokens`，实际将 input embedding 与 LM head 解耦；随后把两个完整矩阵都作为 `modules_to_save` 训练。按 151,789 x 1,536 计算，两矩阵约 4.663 亿参数。我们的正式合同记录 `<archived-job-id>=true`，将语义初始化同时带到输出侧，只训练 143 x 1,536 个新增 input rows，并以 LoRA 调整 LM head；完整 adapter 共 2,114 万参数。共享的语义几何可能使已见输入 token 在输出侧获得过高自相似分，与“复制最近/高频历史类别”症状高度一致。 | 最可能主因，但仍需单变量重训证明因果 |
| 2 | 关系损失并非数值等价 | 官方使用未归一化点积；结构项每步按 17 个有效父类各采一对，协同项使用未阈值化计数。我们使用 cosine InfoNCE；结构项覆盖 138 个有效类别 anchor；协同项按论文 $T=0.05$ 将 10,826 条非零有向共现边裁为 4,070 条，并将阈值外关系视为负例。 | 已确认实现不等价，影响大小待消融 |
| 3 | 论文正文与官方 Arts recipe 冲突 | 本复现采用论文正文的 structure/collab `0.05/0.1`；发布 `train_arts.sh` 实际是 `0.01/0.01`。在上述不同采样规模与相似度下，权重不能直接换算。 | 明确配置差异，方向无法仅靠静态分析判断 |
| 4 | batch、调度器与早停轨迹 | 官方 shell 的 `256 // 36=7`，实际有效 batch 约 252，每 epoch 约 81 次 optimizer update；本复现 batch 36，每 epoch 约 561 次 update。官方是 linear + 20 warmup steps，我们是 cosine + 3% warmup。 | 足以改变收敛点；官方未发布 trainer state，无法离线对齐最佳 step |
| 5 | 精度与软件栈 | 官方为 8-bit/fp16、Torch 2.3、Transformers 4.37、PEFT 0.9；本复现为 bf16、Torch 2.8、Transformers 5.14、PEFT 0.18。 | 次要但不可忽略，特别是旧版 QLoRA/`modules_to_save` 行为 |
| 6 | 输出序列监督 | 官方对类别 token 后的 EOS 也计算 CE；本复现只监督类别 token。 | 可能影响校准，预计小于前四项 |
| 7 | 基础模型快照 | 本复现冻结到 HF revision `8a16abf...`；官方只写本地 `Qwen2-1.5B`，没有公开 revision 或训练 checkpoint。 | 无法核实，构成严格重放上限 |

因此，该 Qwen2 Natural 结果已重新归类为**论文参数优先的非严格机制诊断**，不能用它单独断言官方 MeRA 不可复现，也不能填入 strict 单元格。项目不再通过逐项修改旧 adaptation 来逼近官方值，而是由第 6.0 节的新主线一次性复刻官方 untied full embedding/LM-head、raw-dot 关系采样、未阈值共现、Arts 权重、batch、scheduler、8-bit/fp16 与旧软件栈。新 Arts/Grocery adapter 和完整全类别排名通过双公式门禁后，才形成 strict 结论。

### 9.9 实验运行进度

| 区块 | 实验 / 任务 | 2026-09-07 状态 |
|---|---|---|
| Qwen2 复现训练级检查 | `mera-official-strict-preflight-r9-20260901` | `pass`；48 步原始官方训练、checkpoint-48 与 adapter 重载合同全部通过 |
| Qwen2 Arts / Grocery 复现 | `mera-official-strict-r9-20260901` + `mera-official-strict-eval-recovery-r1-20260901` | 两个 adapter 训练完成；恢复评测均 `pass`，完整指标已回填 |
| Qwen3.5/Llama 官方 MeRA 训练级检查 | `mera-official-crossmodel-preflight-r5-20260901` | 历史 public-code 门禁已完成；作者协议新主线使用独立训练级门禁 |
| Qwen3.5 Arts / Grocery 旧机制适配 | `mera-qwen35-ag-standard-r4-20260901` | 已取消，为官方源码跨模型实现释放 4×A100 |
| Llama Grocery 机制适配 | `mera-llama31-main-repair-20260831` | 运行中 |
| Home official-code 2×2 | `mera-official-home-factorial-r1-20260902` + `mera-official-home-eval-recovery-r3-20260902` | 双模型共八项正式训练产物有效；八项单卡恢复评测均 `pass`，正式指标已回填 |
| 作者协议训练级门禁 | `mera-author-arts-preflight-r2-20260906` + `mera-author-grocery-preflight-r1-20260906` + `mera-author-crossmodel-preflight-r2-20260906` | Qwen2 Arts/Grocery、Qwen3.5 Home、Llama Home 均 `pass`；真实训练、checkpoint 与 adapter reload 合同通过 |
| Qwen2 作者协议正式 r2 | `mera-author-arts-formal-r2-20260907` + `mera-author-grocery-formal-r2-20260907` | Arts `400/400`、Grocery `690/690`，最终 adapter 完整；浮点 epoch 精确比较造成验收假失败，eval-only recovery 已提交 |
| Qwen3.5/Llama 作者协议正式 r2 | `mera-author-three-dataset-formal-r2-20260907` | 六项均为 `pass`；Arts/Grocery 与 Llama Home 指标已回填 |
| Qwen3.5 Home fastlog 备份 | `mera-author-qwen35-home-fastlog-r1-20260907` | `pass`；先于主矩阵 Home 任务完成并通过同一合同，按预先规则采用其正式结果 |

作者协议正式 r1 不进入结果：Qwen2 Arts 在 step 336/400 保存周期 checkpoint 时因 BlobFuse 不支持可靠的临时目录原子 rename 而失败；Qwen2 Grocery 与六项跨模型任务为避免同类损耗被主动取消。r2 的 formal-only `save_strategy="no"` 仅移除训练中的中间快照，保留最终根 adapter 保存；优化器步数、数据、loss、scheduler、五轮训练以及 Home 136→112 特殊评测合同均不变。完成验收已从不可靠的浮点 `epoch == 5` 修正为 `global_step == max_steps`，新合同同时记录请求 epochs 与计划步数；旧 Qwen2 产物按已知计划步数 400/690 兼容验收。完整测试为 `27 passed`，两个正式 runner 的 shell 语法检查通过。

复现 preflight 的前序失败均未进入结果：Docker Hub 基底镜像解析返回 404；r4 暴露平台预装 Apex 与 Torch 2.3 冲突；r6 因无效 image-setup 模板覆盖被拒绝；r8 只验证了 8-bit 模型加载，未覆盖 PEFT LoRA 注入，导致首轮正式任务在 `get_peft_model` 处暴露 PEFT 0.9 与 bitsandbytes 0.45.5 的 API 不兼容。r9 固定到仍提供所需字段且包含 CUDA 12.x 二进制的 bitsandbytes 0.43.3，并将 preflight 升级为 1,728 个 Train 样本、48 个 optimizer step、官方 Validation/checkpoint、最终 adapter 保存与新进程重载。正式 Arts/Grocery 任务只能在该训练级门禁以及源码、数据、模型、环境和 GPU 合同全部通过后提交。

Qwen2 r9 已通过上述门禁：产物记录 `loaded_in_8bit=true`、bitsandbytes 0.43.3、48 个 optimizer step、`preflight-model/checkpoint-48`、`adapter_reload=passed`，重载后的 logits 形状为 `[1, 2, 151789]`；锁定环境的 expected/actual 逐项一致。正式 Arts `<archived-job-id>` 与 Grocery `<archived-job-id>` 均完成训练，但因推理入口缺少 Gradio 依赖在训练后退出。恢复实验只复用已持久化 adapter 做官方推理与双公式评测；Arts `<archived-job-id>`、Grocery `<archived-job-id>` 均为 `pass`。恢复合同验证 adapter SHA、完整 checkpoint state、训练合同、官方资产和模型资产均与原任务一致。

跨模型前序门禁均未提交正式任务：r2 暴露缺少 `fire`，r3/r4 继续补齐 `datasets`、`gradio` 和完整训练入口依赖；r4 的 Qwen3.5/Llama 最终都在构造 `TrainingArguments` 时发现 Transformers 5.14.1 已移除旧 `group_by_length` 参数，尚未执行 optimizer step。r5 将该官方布尔开关等价迁移为 `<archived-job-id>="group_by_length"`；5.14.1 Trainer 仍调用同一 `LengthGroupedSampler`，其 batch 计算与排序算法不变。运行合同新增实际采样策略断言，本地 21 项测试和 Bash 语法均通过。r5 Qwen3.5/Llama Backend ID 分别为 `<archived-job-id>`、`<archived-job-id>`，双任务已由平台接收。

## 10. 训练 JSON Sample

下例来自官方 Grocery Train 首条样本的解析视图；训练时历史和目标名称均替换为同一冻结映射中的原子类别 token。

```json
{
	"sample_id": "train:0",
	"history": [
		"Cooking & Baking",
		"Canned, Jarred & Packaged Foods",
		"Pastries & Bakery",
		"Chocolate Candy"
	],
	"targets": ["Chocolate"]
}
```

## 11. 差异 Sample 与原因

> 空白表示双模型正式评测尚未产出，不能用 smoke 或官方已发布预测代替。

| 数据集 | 样本 ID | 模型 | Gold | Top-1 | 差异原因 |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## 参考资料

- Bi et al., *Large Language Models with Multi-faceted Relation Alignment for User Novel Interest Discovery*, PAKDD 2025, DOI: `10.1007/978-981-96-8298-0_28`。
- Official repository: `https://github.com/GabyUSTC/MeRA`, audited commit `c07bdbe57bf82e5c938b478035baee78d630b17c`。
