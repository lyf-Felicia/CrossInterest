# Interest ID 两阶段多任务 SFT 实验报告

> 数据：Amazon Home 112 类，冻结 Train-112K / Dev-2K。  
> 日期：2026-09-10。  
> 模型：Qwen3.5-9B、Llama-3.1-8B-Instruct。  
> 结果范围：仅完整正式评测进入结果表，smoke 不进入结论。  
> 指标单位：Macro Recall、Macro nDCG 均为 0-1。

## 1. 实验目标

把 MiniOneRec 的文本与离散 ID 双向对齐任务迁移到类别级 Interest ID，改成先对齐、后推荐。推荐目标仍是完整历史中未出现的新兴趣类别，而不是下一商品。

研究问题：相对现有单阶段 Interest ID SFT，先做非推荐对齐、再做新兴趣多任务 SFT，能否改善同一 Dev-2K 上的类别排序表现？本轮评估整个方法，不把效果差异单独归因于任务顺序。

参考本地 MiniOneRec 的标准 SFT 实现，commit 为 `5f4f733cfda387f7f6f8f158db0d300eed9471c6`；读取时其核心 `sft.py`、`data.py` 无未提交修改。标准实现拼接 SID 推荐、SID/标题双向对齐、历史到未来商品标题三类数据。后者仍是推荐任务，必须放在第二阶段。这里不复现商品 SID 的三级 RQ-VAE，也不运行 RL。

## 2. 数据与控制变量

正式评测直接复用现有 Interest 视图的冻结 Dev-2K，112 个目标类别分层抽样，所有方法使用相同样本、历史、候选和解析器。Dev 不用于构造辅助监督或选 checkpoint。

| 项目 | 设置 |
|---|---|
| 原始训练样本 | 112,000 条；112 类，每类 1,000 条 |
| 输入 | 原有自然语言 Interest 历史序列；不改成历史 ID 序列 |
| 新兴趣约束 | `target_interest` 不属于 `<archived-job-id>`，构造时逐条断言 |
| 标签映射 | 复用既有 112 个单 token `<interest_NNN>`；不为历史中额外类别增加 ID |
| 预热数据来源 | 已冻结、与 Train 目标支持集完全一致的类别字典；无用户历史或未来点击字段 |
| 推荐数据来源 | 仅冻结 Train；每个样本生成两个不同输出形式的推荐任务 |
| 随机种子 | 20260825 |

训练源 SHA-256：`43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd`。  
mapping SHA-256：`5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da`。  
映射指纹：`d21096f188c9e2b757299469cd1a400332444ed3c87d6e17068b97bbbf00414c`。

完整数据合同、任务计数与输出文件哈希见 data/MANIFEST.json（私有原始证据，不随公开仓库分发）。

## 3. 两阶段训练方法

| 阶段 | 任务 | 唯一任务记录数 | 混合与训练预算 |
|---|---|---:|---|
| A：非推荐预热 | 类别名称到 Interest ID；Interest ID 到类别名称 | 112 + 112 = 224 | 1:1；10 epoch；2,240 次样本呈现，70 step |
| B：新兴趣推荐 | 历史到新兴趣 ID；历史到新兴趣类别名称 | 112,000 + 112,000 = 224,000 | 1:1；1 epoch；7,000 step |

每个阶段将任务拼接后交给 Trainer 随机打乱，因此 1:1 是整个 epoch 的样本比例，不保证每个 minibatch 恰好对半。损失沿用 assistant-only token CE；两种输出长度不同，样本比不意味着严格相等的 token 损失贡献。A 不含任何历史到未来目标的任务；B 不继续混入字典对齐任务。

类别 ID 对应唯一类别，而非唯一商品。为避免一个类别 ID 对应许多商品标题的歧义，A 使用精确类别名称，不使用商品标题伪装一对一映射。名称文本为完整 `Parent > Child`；ID 本身及初始化仍只基于 child label，不添加 parent token。

| 参数 | 设置 |
|---|---|
| 模型标识 | `Qwen/Qwen3.5-9B`；`NousResearch/Meta-Llama-3.1-8B-Instruct` |
| 初始化 | 基础模型冻结；新增 ID embedding 由 child label 的基础 token embedding 均值初始化 |
| 可训练参数 | decoder 线性层 LoRA、`lm_head` LoRA、112 个稀疏可训练 ID embedding；Qwen 排除视觉 LoRA |
| LoRA | rank 16、alpha 32、dropout 0.05 |
| 精度与损失 | BF16；assistant-only causal LM CE，不做标签空间额外损失或加权 |
| 优化器 | fused AdamW；学习率 1e-4；cosine；warmup ratio 0.03；max grad norm 1.0 |
| Batch | 有效 batch 32；每模型 4 张 A100 80GB，Standard SLA；Qwen micro-batch 4、累积 2，Llama micro-batch 2、累积 4 |
| 序列长度 | 最大 8192；超长直接报错，不截断 |
| Qwen thinking | 显式关闭；要求 fast path 可用 |
| 阶段衔接 | B 从 A 的 `final_adapter` 继续训练所有相同 LoRA/稀疏参数；重新创建优化器和调度器 |
| checkpoint | 每 500 step 保存；保留最后两个；使用最终 checkpoint，不按 Dev 选优 |

阶段 B 会重新加载同一基础模型并恢复同一语义初始化，再加载 A 的训练权重覆盖对应稀疏 ID embedding 和 LoRA。保存/重载连续性由小型真实 Llama + PEFT 测试核验，重载前后 logits 逐值一致；Qwen 实际大模型的两阶段训练、权重审计及评测链路也已通过 smoke。

已核验 Qwen smoke 的实际依赖快照：Torch 2.8.0+cu126、Transformers 5.14.1、PEFT 0.18.1、Accelerate 1.12.0、flash-linear-attention 0.5.2、causal-conv1d 1.7.0。正式配置使用同一镜像及依赖清单，每项作业另存环境快照。

旧单阶段基线是 112K、1 epoch、3,500 step。本方法总计 7,070 step，额外增加类别名推荐监督和对齐预热，因此不是等预算顺序消融。没有新增等预算控制组，不据此宣称预热本身带来因果增益。

## 4. 任务模板

A 的两个模板分别为：

```text
Which Interest ID represents the following category?

Category: {Parent > Child}

Respond with only the single-token Interest ID in the exact format <interest_NNN>. Do not output any explanation.
```

```text
What is the exact category name represented by the following Interest ID?

Interest ID: {interest_id}

Respond with only the exact category name. Do not output any explanation.
```

B 直接调用原训练 renderer：`history_to_id` 使用原 Interest/Interest-ID 模板，`history_to_name` 使用原 Interest/category-name 模板。两个任务使用同一源样本、历史和新兴趣标签，没有候选列表。测试断言生成的消息与原 renderer 完全一致。

四项任务的真实消息样例见 data/SAMPLE_PREVIEW.json（私有原始证据，不随公开仓库分发）。

## 5. 评测与健康门槛

评测一律要求单 Interest ID 输出，沿用原分类和排序实现。

| 指标 | 口径 |
|---|---|
| Macro Recall@10/20 | 无候选、无标签 decoder 约束的 direct 20-beam；ID 映射回类别；112 类宏平均 |
| Macro nDCG@10/20 | 相同无候选排序结果，按目标类别宏平均 |
| Macro Accuracy | 10 个固定自然语言候选；对应输出标签约束 beam；112 类宏平均 |

Macro Recall 和 Macro nDCG 均先在每个目标类别内对样本指标求平均，再对 112 个类别等权平均，而非将全部样本直接平均。基线与新方法均使用这一口径。正式结果表只展示排序指标；分类结果保留在原始评测与审计产物中。

10 候选分类的均匀随机基线为 10%，不能与全 112 类的 1/112 混用。排序中的无效 ID、格式错误和重复按原解析器处理，不补成合法标签，也不使用候选约束提高排序格式率。

正式验收要求：两个阶段达到冻结预算；有限正训练 loss；adapter 含有限非零的稀疏 embedding 和输出 LoRA；B 起始 adapter SHA 等于 A 成品；分类和排序各完成 2,000 条、112 类、无缺失和请求错误；模型、视图、mapping、数据与解码合同一致。另单列格式率，低格式率作为效果解释因素，不用修复输出掩盖。

smoke 每阶段仅 2 个 optimizer step，两个评测各 4 条，只检验运行链路，不代表模型效果；smoke 和正式的输出目录及合同标记隔离。

## 6. 正式结果

<table>
  <thead>
    <tr><th>模型</th><th>方法</th><th>Macro Recall@10 ↑</th><th>Macro Recall@20 ↑</th><th>Macro nDCG@10 ↑</th><th>Macro nDCG@20 ↑</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Qwen3.5-9B</td><td>单阶段 Interest ID SFT</td><td>0.1904</td><td>0.3180</td><td>0.0987</td><td>0.1310</td></tr>
    <tr><td>对齐预热 → 新兴趣多任务 SFT</td><td>0.1977</td><td>0.3325</td><td>0.0983</td><td>0.1321</td></tr>
    <tr><td rowspan="2">Llama-3.1-8B-Instruct</td><td>单阶段 Interest ID SFT</td><td>0.1909</td><td>0.3144</td><td>0.0968</td><td>0.1279</td></tr>
    <tr><td>对齐预热 → 新兴趣多任务 SFT</td><td>0.1956</td><td>0.3233</td><td>0.0971</td><td>0.1293</td></tr>
  </tbody>
</table>

基线转录自 [../crossinterest.training/RESULTS.md](../supervised_learning/results.md) 的 Interest 输入列；来源实验为 `amazon-sft-112-interest-id-semantic-v1-20260828-104049`。两模型正式结果已完成合同审计，并从原始预测重算一致。

两模型 Macro Recall@10/20 均高于单阶段基线，Macro nDCG@20 仅小幅提升，Qwen 的 Macro nDCG@10 略降。因此本轮支持“该组合方法略微改善排序召回”，不支持全面提升，也没有证据表明接近翻倍的训练预算获得了相称收益。

## 7. 执行状态与限制

两个模型均完成非推荐预热、新兴趣多任务 SFT、完整 Dev-2K 分类和排序。正式结果均通过权重交接、数据与模型合同和原始预测重算检查。

最终 checkpoint 均为预热 70 step / 10 epoch、推荐 7,000 step / 1 epoch。训练期间的集群中断已恢复；最终运行记录显示 Qwen 从推荐 checkpoint-2500、Llama 从 checkpoint-4500 续训。两模型分类与排序均完成 2,000/2,000 条、覆盖 112 类，无缺失；分类请求错误为 0，排序合法 beam 重复率为 0。本地最终验收及报告回填完成于 2026-09-11。

| 模型 | 预热 loss | 推荐 loss | 分类格式率 | 排序合法 beam 率 | 排序严格格式率 |
|---|---:|---:|---:|---:|---:|
| qwen35_9b | 0.6655 | 0.4414 | 1.0000 | 1.0000 | 1.0000 |
| llama31_8b | 0.4125 | 0.1677 | 1.0000 | 1.0000 | 1.0000 |

loss 为 Trainer 写出的 train_loss 字段。断点恢复后的累计口径可能不覆盖恢复前的全部损失，不把该字段作为跨模型的完整 epoch 平均损失比较。

正式实验：`home112-minionerec-two-stage-formal-20260910`。
机器可读结果见 outputs/formal/AUDITED_RESULTS.json（私有原始证据，不随公开仓库分发），逐模型原始预测和合同保存在同目录下的实验子目录。

本轮仅两个模型、一个 seed 和 Home 单域，不推断跨域或种子稳定性。相对旧基线同时增加预热和类别名预测监督，训练预算从 3,500 增至 7,070 step，不能把效果差异单独归因于任务顺序。无效自由生成保持无效，格式率差异必须纳入解释。

## 8. 实验入口

在 Amazon 项目目录运行：

```bash
python -m unittest eval_interest_id_minionerec.test_pipeline -v
python -m crossinterest.methods.alignment.data
```

| 内容 | 文件 |
|---|---|
| 数据构造 | [build_dataset.py](../../../src/crossinterest/methods/alignment/data.py) |
| 训练与阶段审计 | [train.py](../../../src/crossinterest/methods/alignment/train.py) |
| 连续训练评测入口 | run_train_eval.sh（私有原始证据，不随公开仓库分发） |
| smoke 配置 | amlt_smoke.yaml（私有原始证据，不随公开仓库分发） |
| 正式配置 | amlt_formal.yaml（私有原始证据，不随公开仓库分发） |
| 门禁、提交、下载与验收执行器 | [execute_experiment.py](../../../src/crossinterest/methods/alignment/runner.py) |
| 原始预测重算及合同验收 | [audit_results.py](../../../src/crossinterest/methods/alignment/audit.py) |
| 自动测试 | test_pipeline.py（私有原始证据，不随公开仓库分发） |