# Behavior+Interest 两阶段多任务 SFT 实验报告

> 数据：Amazon Home 112 类，冻结 Train-112K / Dev-2K。  
> 日期：2026-09-12；两模型正式训练、完整评测及原始排名复算完成。  
> 模型：Qwen3.5-9B、Llama-3.1-8B-Instruct。  
> 输入：Behavior+Interest（B+I），训练与评测一致。  
> 指标：Macro Recall@10/20、Macro nDCG@10/20，均为 0-1，越高越好；只接受完整正式结果。

## 1. 实验目标

在现有较优的 Behavior+Interest 输入上，评估“非推荐对齐预热 → 新兴趣多任务 SFT”。目标仍是完整历史中未出现的新兴趣类别，不是下一商品。主对照为同一 B+I 视图的单阶段语义 Interest ID SFT，不能用较弱的 Interest-only 基线替代。

此前完成的是 Interest-only 版本，其设置与结果保留在[原实验报告](../alignment/report.md)。本轮与该两阶段版本保持源样本、任务配比、模型、初始化、训练预算和解码方式一致，将推荐输入改为 B+I；Qwen 因显存不足调整 micro-batch 与梯度累积，有效 batch 不变。不从上一轮已经推荐微调过的 adapter 继续训练。

## 2. 数据与控制变量

| 项目 | 设置 |
|---|---|
| 源训练集 | 原冻结 Train-112K；112 类，每类 1,000 条 |
| B+I 历史 | 按时间顺序逐条输入 `[历史兴趣类别] 已交互商品标题`；标题与类别一一对应 |
| 目标 | `target_interest` 不在 `<archived-job-id>` 中 |
| 辅助任务 | 类别名称与单 token Interest ID 的双向映射，无用户历史 |
| 训练文件 | 复用原 warmup/recommendation parquet，不重建样本或标签 |
| 正式评测 | 冻结 `dev_2k_behavior_interest.parquet`，2,000 条、112 类；不用于监督构造或选 checkpoint |
| 随机种子 | 20260825 |

训练源 SHA-256：`43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd`。mapping SHA-256：`5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da`。与 Interest-only 版本相同，只有独立合同的 view 为 `behavior_interest`；训练、评测或验收视图不匹配均拒绝。

## 3. 训练方法

| 阶段 | 任务 | 任务记录数 | 预算 |
|---|---|---:|---|
| A：非推荐预热 | 类别名称到 ID、ID 到类别名称，1:1 | 224 | 10 epoch，70 step |
| B：新兴趣推荐 | B+I 历史到新兴趣 ID、B+I 历史到新兴趣类别名称，1:1 | 224,000 | 1 epoch，7,000 step |

阶段 A 从同一基础模型和语义初始化开始；B 加载 A 的 LoRA 和稀疏 ID embedding，重置优化器与调度器，不再混入对齐任务。1:1 为样本比例，不代表不同输出长度的 token loss 贡献相同。

| 参数 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-9B`；`NousResearch/Meta-Llama-3.1-8B-Instruct` |
| 可训练参数 | decoder 线性层及 lm_head LoRA；112 个稀疏 ID embedding；基础模型冻结，Qwen 排除视觉 LoRA |
| ID 初始化 | child label 基础 embedding 均值，不编码 parent；类别名称文本使用完整 Parent > Child |
| LoRA | r16、alpha32、dropout0.05 |
| 优化器及损失 | fused AdamW，lr1e-4，cosine，warmup0.03；BF16，assistant-only causal LM CE，不额外加权 |
| 设备及 batch | 每模型 4×A100 80GB，Standard；有效 batch32；Qwen micro1×累积8，Llama micro2×累积4 |
| 长度 | 最大8192，禁止截断；原提交关闭gradient checkpointing，Llama推荐恢复段因OOM开启 |
| 保存与恢复 | 每500step保存，保留最后两个；从完整 checkpoint 恢复，完成阶段不重训；最终 checkpoint 评测 |
| 环境 | 同上一轮成功镜像及固定依赖，Qwen thinking 关闭并要求 fast path |

单阶段 B+I 基线为3,500step，本方法为7,070step。因此相对单阶段的差异包含额外预热、类别名监督及训练预算，不能单独归因于预热顺序。

## 4. 输入与任务模板

预热不读取历史，模板与原两阶段实验相同。推荐阶段直接复用已有 B+I renderer，两个推荐任务的输入历史相同，输出分别为新兴趣 ID 和类别名称。输入结构如下：

```text
The titles and corresponding topic categories of the products I recently rated positively, in chronological order, are listed below. Each item is formatted as "[Topic category] Product title":

1. [历史兴趣类别] 已交互商品标题
2. [历史兴趣类别] 已交互商品标题
...
```

真实四任务样例见 data/SAMPLE_PREVIEW.json（私有原始证据，不随公开仓库分发）。训练不提供候选列表，也不把未来商品标题放入历史。

## 5. 评测口径

正式表使用无候选、无标签 decoder 约束的 direct20-beam Interest ID 排序，不使用112标签直接 logits 排序替代。ID 映射回类别，先对每类样本求平均，再对112类等权平均，得到 Macro Recall 和 Macro nDCG。无效或重复 beam 按原解析器处理，不补为合法结果。

另保留10候选分类作为完整流程验收，不在主表展示 Accuracy。正式分类和排序各须完成2,000条、112类、无缺失，分类请求错误为0；核验映射、数据、输入视图、权重交接、完整epoch及原始预测重算一致。Smoke仅每阶段2step、每种评测4条，不进入主表或效果结论。

## 6. 正式结果

<table>
  <thead>
    <tr><th>模型</th><th>方法</th><th>Macro Recall@10 ↑</th><th>Macro Recall@20 ↑</th><th>Macro nDCG@10 ↑</th><th>Macro nDCG@20 ↑</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Qwen3.5-9B</td><td>单阶段 B+I Interest ID SFT</td><td>0.2287</td><td>0.3725</td><td>0.1174</td><td>0.1536</td></tr>
    <tr><td>对齐预热 → B+I 新兴趣多任务 SFT</td><td>0.2213</td><td>0.3685</td><td>0.1152</td><td>0.1523</td></tr>
    <tr><td rowspan="2">Llama-3.1-8B-Instruct</td><td>单阶段 B+I Interest ID SFT</td><td>0.2217</td><td>0.3666</td><td>0.1147</td><td>0.1511</td></tr>
    <tr><td>对齐预热 → B+I 新兴趣多任务 SFT</td><td>0.2230</td><td>0.3757</td><td>0.1170</td><td>0.1554</td></tr>
  </tbody>
</table>

基线转录自[原112类结果表](../supervised_learning/results.md)中 single Interest ID 的 B+I 列，来源实验为 `amazon-sft-112-interest-id-semantic-v1-20260828-104049`。新方法两模型均完成70+7,000步及原Dev2K，云端完整审计通过，本轮原始排名复算（私有原始证据，不随公开仓库分发）与保存指标一致；均112类、无缺失，valid/strict beam率1、重复率0，分类请求错误0。

相对上述历史单阶段基线，Qwen四项略低，Llama四项略高；没有一致的跨模型提升。该比较含7,070对3,500步预算差异、额外类别名监督和物理分批/中途GC恢复差异，不能将变化单独归因于预热。后续骨干矩阵中新跑的Qwen3.5单阶段结果为0.2242/0.3685/0.1135/0.1498，与此处历史基线是不同运行，不能混为同一对照或替换旧数值。

## 7. 执行状态与限制

B+I 完整本地测试通过（16项测试、7项子测试），覆盖模板、Dataset 传递、训练合同、结果审计、执行器隔离及失败恢复。数据 manifest 除 `view` 外与 Interest-only 版本一致；两视图 Dev-2K 的源样本、目标、历史标题和历史类别逐条一致。

2026-09-11，实验 `home112-minionerec-bi-two-stage-smoke-20260911` 的 Llama smoke 通过。Qwen 原作业完成预热，但在推荐首个优化步前 CUDA OOM：GPU2 已使用约78GB，无法再分配2.87GiB。将 Qwen micro-batch 从4降至1、梯度累积从2增至8，有效 batch 仍为32；训练样本、70/7000step预算、最大8192和禁止截断策略均不变。此调整保持预算，不保证与原物理分批逐位数值一致。原失败记录保留，不计入正式结果。

| 模型 | Smoke 作业 ID | 状态 |
|---|---|---|
| Qwen3.5-9B，原micro4 | `<archived-job-id>` | failed：推荐首步OOM |
| Qwen3.5-9B，修复micro1 | `<archived-job-id>` | pass，已完成本地原始结果审计 |
| Llama-3.1-8B-Instruct | `<archived-job-id>` | pass，保留不重跑 |

三个smoke记录均在 Recall，每项4×A100 80GB、Standard。替代Qwen与原Llama于2026-09-11 02:45 UTC完成本地合同和原始结果审计，见 outputs/smoke/AUDITED_RESULTS.json（私有原始证据，不随公开仓库分发）。Smoke只证明短流程与审计可执行，不作为正式效果或自由生成格式已学会的证据。

随后按新鲜容量（Feeds余量21、Recall余量17，仅为Standard配额余量）提交正式实验 `home112-minionerec-bi-two-stage-formal-20260911`。后端终态于2026-09-12 06:56 UTC核验：

| 模型 | 正式作业 ID | 目标 | 后端状态 |
|---|---|---|---|
| Qwen3.5-9B | `<archived-job-id>` | Feeds | pass：完整训练与评测 |
| Llama-3.1-8B-Instruct，原作业 | `<archived-job-id>` | Recall | failed：推荐3,141步反向OOM |
| Llama-3.1-8B-Instruct，恢复 | `<archived-job-id>` | Feeds | pass：完整训练与评测，恢复段GC=true |

两模型均完成预热70step/10epoch和推荐7,000step/1epoch，分类与排名各2,000条。官方日志确认原Llama在3,141步真实OOM：GPU0剩余2.53GiB，反向传播需额外2.97GiB。此前两次SIGTERM由更高SLA任务驱逐并自动重排引起，与OOM分开记录。最终训练产物中Qwen记录从checkpoint-1000恢复，Llama从checkpoint-5500恢复；这表示后续自动恢复记录，不覆盖最初GC修复时的checkpoint-3000来源。

恢复仅开启推荐阶段gradient checkpointing，micro2、累积4、有效batch32、8192长度、数据、optimizer和7,000步目标不变；预热复用，不重新训练。已核验checkpoint-3000的adapter、optimizer、scheduler、trainer state及四卡RNG文件在远端非空列表中。通过`rerun --in-place --upload-code`保留原目录提交替代任务，上传SHA和新旧ID见恢复记录（私有原始证据，不随公开仓库分发）。恢复后完整训练和评测已通过。本轮后续执行器回归共32项测试、20项子测试通过。重计算可能产生数值细微差异，报告保留中途资源修复说明。

正式结果已填入第6节。历史执行记录见 outputs/EXECUTION_STATE.json（私有原始证据，不随公开仓库分发）、outputs/execution.log（私有原始证据，不随公开仓库分发）；最新状态以09-12快照（私有原始证据，不随公开仓库分发）、正式云端审计和本轮原始排名复算为准，旧监督状态不代表当前终态。

本轮只有一个 seed、Home 单域和两个模型，不推断跨域稳定性。B+I 比 Interest 输入更长，不改变最大长度、截断策略或损失。结果不支持多任务方法稳定超过B+I单阶段基线。

## 8. 实验入口

从 Amazon 项目目录运行：

```bash
python -m unittest eval_interest_id_minionerec.test_pipeline -v
python -m crossinterest.methods.alignment.data --view behavior_interest
python -m crossinterest.methods.alignment.runner --view behavior_interest
```

最后一条命令需已提交对应 smoke，随后等待 smoke 审计、提交固定正式实验、下载并验收。已有正式作业仅用 `--monitor-existing` 接管，不重复提交。

本轮已停止于 smoke 的恢复命令如下；仅在正式训练从未提交时允许使用，不查询实验注册表：

```bash
python -m crossinterest.methods.alignment.runner --view behavior_interest \
  --resume-smoke --smoke-job qwen35=qwen35-two-stage-smoke-micro1
```

| 内容 | 文件 |
|---|---|
| 冻结视图合同 | data/MANIFEST.json（私有原始证据，不随公开仓库分发） |
| Smoke / 正式配置 | amlt_smoke.yaml（私有原始证据，不随公开仓库分发）、amlt_formal.yaml（私有原始证据，不随公开仓库分发） |
| 共享训练与审计 | [训练入口](../../../src/crossinterest/methods/alignment/train.py)、连续运行脚本（私有原始证据，不随公开仓库分发）、[结果验收](../../../src/crossinterest/methods/alignment/audit.py) |
| 提交与收尾执行器 | [execute_experiment.py](../../../src/crossinterest/methods/alignment/runner.py) |

## 9. 后续实验核对

后续统一Amazon Home & Kitchen、B+I，见全部实验设置与分组结果汇总（私有原始证据，不随公开仓库分发），包含输入、单阶段/两阶段、骨干大小/版本、课程顺序及Cross-fitted的逐组对照。截至2026-09-13 13:59 UTC，本轮21项实验全部完成并验收：包含原八骨干、追加Qwen3.5-27B、旧→新和Shuffle两臂、五教师全112K OOF评分，以及全量类内加权学生的完整训练与原Dev2K评测。学生四项Macro指标为0.2305/0.3689/0.1186/0.1535，原始预测复算和最终adapter核验通过；与本轮无权重9B比较四项正向，Recall@20基本持平，不据单seed推断显著性。

追加Qwen3.5-27B的四项Macro指标为0.2219/0.3734/0.1188/0.1572，依次为Recall@10、Recall@20、nDCG@10、nDCG@20。相较本轮无权重9B，Recall@10略低，其余三项更高，不代表所有指标均随规模提升。完整112K/3,500步训练、最终adapter及分类/排序各2,000条均通过最终独立审计（私有原始证据，不随公开仓库分发）。三次抢占后同目录迁至Recall，最终权重和分类原文未变，304条有效排序逐字节保留并补齐1,696条；见恢复回执（私有原始证据，不随公开仓库分发）和迁移回执（私有原始证据，不随公开仓库分发）。自动下载的路径过滤错误已通过Blob SDK补救，两类监听均已结束，各实验仍独立记录，不与本轮两阶段结果混表。