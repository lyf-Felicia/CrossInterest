# Generative New Interest Prediction实验总览

> Amazon Home & Kitchen下一新兴趣预测。整理日期：2026-10-08。
> 按论文研究问题组织实验，而非云端提交时间。正式结果、旧协议、smoke与未完成探索分别解释。

集中查看分实验指标表：[正式实验结果汇总](../../RESULTS.md)。本总览负责研究导航，不重复承载完整数值矩阵。

研究导航对齐[54页演示文稿](https://docs.google.com/presentation/d/14JOumukKLuqFPzMQWnuxFqTMGTQAO6HjlgrYWsEgzOI/edit)：任务与数据（2-13页）、提示与评测（14-19页）、B+I与语义Interest ID（20页）、外部比较（21-23页）、规模和控制实验（24-29页）、CIR（30-32页）。后期结构探索作为补充研究保留，不替代PPT中的主方法。

## 1. 任务与论文主线

目标是从用户可见的有序行为历史预测完整过去中尚未出现的新兴趣类别，而非下一商品。核心问题是如何表达历史、如何输出类别，以及已有兴趣辅助监督是否能提高新兴趣预测。

论文主线为：严格定义下一新兴趣；以行为、兴趣及联合历史研究输入表示；使用类别名称语义初始化的单token Interest ID作为输出；与Google和MeRA比较；研究CIR历史兴趣重构预热及其控制。结构改进与负结果另行呈现。不能把运行成功、额外参数或固定表示条件下的理论优势写成推荐指标提升。

当前主协议为112K训练、Dev2000、112目标类，按类别宏平均Recall@10/20和nDCG@10/20。主排序使用完整词表、无候选、确定性20-beam；10候选分类、99类旧协议和MeRA论文原数据集不与之混表。开发集多次复用，单seed点估计不等于统计显著。

## 2. 数据与表示

| 研究问题 | 实验文档 | 在论文中的作用 |
|---|---|---|
| 新兴趣如何定义、如何防止泄漏？ | [数据构建](dataset/README.md) | 完整过去定义标签，可见历史定义输入 |
| 不训练时模型有何表现？ | [早期零样本](zero_shot/README.md)、[112K匹配零样本](zero_shot_home112/README.md) | 对照监督学习，保持解析与候选口径 |
| 行为、类别及联合历史怎样表示？ | [112类监督学习](supervised_learning/README.md)、[结果](supervised_learning/results.md) | B+I与语义Interest ID的主要表示实验 |
| 数据规模是否影响效果？ | [数据规模](data_scaling/README.md) | 与所属版本的全量对照比较 |
| 标签协议如何演变？ | [99类训练](supervised_learning_99/README.md)、[99类规模](data_scaling_99/README.md) | 历史探索，不替代112类结果 |

公开数据入口为[data/README.md](../../data/README.md)。模型源码位于[training](../../src/crossinterest/training)，配置见[训练](../../configs/training/home112.yaml)和[评测](../../configs/evaluation/home112.yaml)。

PPT第24页的“约80K趋于饱和”限定于排序指标；第35页同时指出候选Accuracy到112K仍有增长，不能将其改写为全部能力已饱和。第21/23页与第25/28-31页使用的9B SFT点估计不同，必须保留各自同轮对照，不跨表替换baseline。

## 3. 辅助监督与控制实验

最初的[Interest-only两阶段对齐](alignment/report.md)先做类别名称与ID字典对齐，再混合两种推荐输出。[B+I两阶段](alignment_behavior_interest/report.md)使用更强的联合历史对照。二者相对各自单阶段基线都改变了预算，不能把差值单独归因于预热顺序。

- 基础能力与训练组织：[骨干与规模](backbones/report.md)、[课程顺序](curriculum/report.md)、[Cross-fitted加权](crossfit/report.md)。
- 历史级辅助监督：[CIR识别](cir/report.md)、[方向性](directionality/report.md)、[未来复现频率子集](frequency/report.md)。频率子集使用训练用户的后续记录筛选，不属于后期禁止改数据的结构协议。
- 机制核对：[结构与重构](mechanism/report.md)、[信息因子分解](factorization/report.md)、[四视图一致性](consistency/report.md)。比较对象、额外loss和预热预算必须分别列明。

CIR三轮是后续历史G0的重要来源，但不同运行的同名baseline不可互换。真实语义、额外目标与结构组合没有稳定地得到全面收益，详见各子报告。

## 4. 参数与结构研究

先区分[冻结CIR并新建N LoRA](residual/report.md)与[线性模型概率融合](fusion/report.md)。前者改变可训练参数，后者在首ID位置固定1:1混合概率，并非LoRA参数合并。

后期候选包括[独立兴趣编码](composition/report.md)、[角色投影](role_projection/report.md)、[质量守恒完整类别读出](mass_readout/report.md)。三者均有独立对照和负结果。角色投影首轮小趋势未在四卡G0对齐中复现；完整读出相对同源baseline的Recall@20和nDCG@20下降，两个预设门槛失败。

完整读出正式训练3500步和Dev2000均已完成，严格格式100%，但执行健康不等于方法有效。条件性理论与梯度诊断见[文献与理论讨论](composition/literature.md)，不能覆盖效果门槛失败。

## 5. 外部方法与历史旁支

[MeRA报告](mera/report.md)和[比较结果](mera/results.md)区分作者澄清协议、公开代码旧strict诊断和Home方法比较。冻结Home比较中，B+I单token Interest ID在Qwen3.5-9B和Llama-3.1-8B-Instruct的四项排名点估计均高于MeRA，未进行独立显著性检验。[Google基线](google_baseline/README.md)是另一条外部比较。

[MeRA strict实现](mera_strict/README.md)、[MIND新闻数据](news_history/report.md)、[SuperLoss](superloss/report.md)、[LLM摘要](summaries/report.md)、[去噪](denoising/report.md)保留前期探索。未产出或未全量验收的结果不补造为正式指标，也不将数据加权、教师或额外loss带入原数据和原目标不变的新结构协议。

## 6. 如何使用这些材料

子报告描述科学问题、正式点估计和结论边界。原始排名、用户数据、权重、冻结合同和内部回执不公开分发；本地研究者可通过忽略的私有归档核查来源。归档合同SHA不能冒充重组后源码SHA。

历史报告保留原运行预算和恢复说明，其中的旧云端命令不是新仓库入口。公开安装和默认训练评测以[README](../../README.md)为准，目录职责及历史恢复边界见[仓库约定](../repository.md)。

后续优先补充独立测试、多seed及尚缺的机制控制，而不是重复提交已失败候选。新结果先检查覆盖、格式、权重身份与独立复算，再更新结论。