# 新兴趣预测：文献核对、误差依据与下一步实验

> 日期：2026-09-23。任务：从可见B+I历史生成历史未出现的新类别Interest ID。
> 本文是方法筛选与实验设计，不是新方法结果报告。无教师蒸馏、无外部知识伪标签，不修改原冻结数据、既有作业或用户设计文档。
> 文献收益不等于本任务收益；下面区分原文结论、已观测事实和待检验假设。先完成这一轮取舍，再启动新的昂贵训练。

> 研究要求更新：用户要求按论文主方法标准进行任务定制。第3至5节的BPR-max直接适配方案降为历史基线，不再作为当前主方法或启动授权；本轮重新核对最近邻文献、任务特有机制与可证伪设计，不接入训练或提交新作业。

> 最近邻核验后的修正：第7节的四分数归属目标与CPR (WWW 2022) 的二交互交叉排序在分数缩放后相同，撤回其论文主方法推荐，只保留为已有方法的任务适配基线。Train可行性和CPU性质检查仍有效，但不是创新证据。第3至7节均不能自动触发训练。

> 用户范围更新（2026-09-23）：停止数据构造、前缀增广、采样和重加权优化。第8节首次进入事件归一化方案已被用户否决，只保留历史记录，不再推进隔离、实现或实验。当前仅研究固定原数据上的模型结构、损失与机器学习机制；不得把增广或采样更名后重新作为主方法。

> 当前研究候选见第9节：按预测角色分解的兴趣扩展算子。仅为待证伪的结构设计，不是已证实创新或有效的主方法。固定原数据、损失和全文路径；本轮不改trainer、不提交训练。第9节的数学检查不能替代模型实验或完整新颖性检索。

## 1. 先纠正问题定位

当前G0不是N-only，而是普通全文历史路径的均衡CIR三轮后N一轮。CIR从历史中删除目标类别的全部记录，再预测该类别；目标不在可见输入中。因此不能把G0或本轮CIR描述成直接复制可见类别，也不能用C低于N就宣布CIR本身无效。

独立编码版本则把每个兴趣组压成一个向量，经过桥接后替换整个历史文本；第二次生成前向看不到原始历史。N也明显退步，说明目前没有证据支持继续把这种信息瓶颈当作默认结构。可能涉及压缩、隐藏状态到输入embedding的接口适配、优化或监督不足；这些原因尚未被单因素干预分开。

### 已有预测的离线核验

同一Dev2000，112类，合法20-beam原始结果。排名四指标已重新核对至绝对误差小于1e-12；本次没有新模型推理，也没有改变排名。完整来源SHA见离线误差证据（私有原始证据，不随公开仓库分发）。

| 条件 | Macro nDCG@20 | Top-1覆盖类别数 | Top-1中最常见5类占比 | Top-20可见已见类别占位率 |
|---|---:|---:|---:|---:|
| G0 | 0.165582 | 103 | 23.35% | 1.3275% |
| C：独立编码+CIR | 0.067840 | 11 | 87.80% | 8.5675% |
| N：独立编码、不做CIR | 0.086272 | 22 | 83.55% | 7.3100% |

占位率分母是2000乘20，共40,000个排名位置，不是“用户有一个已见推荐”的比例。Top-1覆盖也不是Top-20总类别覆盖。C/N表现出预测集中，但这不能直接证明隐藏表示坍塌。

G0的1,913个Top-1错误中，1,907个仍是可见历史未出现的类别，占99.69%。所以单纯更强地压低已见类别不是首选。此统计使用可见历史；完整seen列表只在离线诊断里单独计数，绝不传给新模型。相对G0，C在Top-20命中上新增203条、丢失620条，N新增187条、丢失512条。这是逐样本描述，不是显著性检验。

不能从按兴趣组数分层的统计宣称压缩的因果作用：只有3条单组、85条2-3组样本，1,912条均为4组以上，分组样本量和类别组成极不均衡。

最终补充：R随机分组条件在09-23 01:57:16完成本地独立审核，macro Recall@10/20为0.153887/0.285889，nDCG@10/20为0.071860/0.104915；Dev2000/112类完整且健康PASS。三组正式汇总（私有原始证据，不随公开仓库分发）效果门槛FAIL，C也低于R。这进一步不支持当前真实分组独立编码的优越性，不改变“原因尚未单因素识别”的边界。此前Top-1覆盖和seen占位分析未扩展到R，不补造这些数值。

## 2. 文献与适配取舍

本轮直接核对以下论文原始页面；DIEN、BERT4Rec、GRU4Rec+进一步阅读了方法正文。其他条目以论文摘要/原始页面为依据，不冒充完整复现或穷尽的新颖性检索。

| 文献 | 原方法的关键内容 | 对本任务的意义与取舍 |
|---|---|---|
| [DIN, Zhou等, KDD 2018](https://arxiv.org/abs/1706.06978) | 按候选广告动态激活相关历史，而非给所有候选同一个用户汇总 | 支持按预测问题选择历史证据；原文是候选条件CTR，不可直接称为无候选新兴趣生成。不能把真实目标类别作为查询送入编码器 |
| [DIEN, Zhou等, AAAI 2019](https://arxiv.org/html/1809.03672v5) | 用下一行为监督中间兴趣状态，再用AUGRU建模目标相关演化 | 支持辅助监督应贴近下游。普通下一商品不等于下一未见类别；原文还指出辅助与目标异质时收益可能减弱，不保证迁移 |
| [MIND, Li等, 2019](https://arxiv.org/abs/1904.08030) / [ComiRec, Cen等, KDD 2020](https://arxiv.org/abs/2005.09347) | 多兴趣向量、路由或注意力、兴趣检索与候选聚合 | 支持多兴趣表示有价值，不支持“每个观测类别压一个向量就应更好”。其检索/多样性目标与本任务不同，不直接复刻复杂路由或推理重排 |
| [BERT4Rec, Sun等, CIKM 2019](https://arxiv.org/html/1904.06690v2), §3.6 | Cloze预测遮蔽商品；正文承认遮蔽重建与未来预测的任务差异 | 当前整类留出CIR已使用类似思想，不包装成新方法。原文的商品级遮蔽、双向编码与采样负例评测也不等于当前协议 |
| [Mult-VAE, Liang等, WWW 2018](https://arxiv.org/abs/1802.05814) / [EASE, Steck, WWW 2019](https://arxiv.org/abs/1905.03375) | 隐式反馈上的多项式似然或线性协同重建 | 可提供简单协同基线，但历史线性方法与固定融合已经做过，不能仅靠重新命名线性关系再开一轮 |
| [GRU4Rec+, Hidasi与Karatzoglou, CIKM 2018](https://arxiv.org/html/1706.03847v3), §3.1-3.3、§4.2 | 排名损失BPR-max强调高分负例，并与稳定softmax CE比较 | 首选做小而明确的损失对照。原文大部分大幅提升同时来自负采样改变；不能把这些收益归因于loss，更不能直接套到仅112类的LLM |
| [Unlikelihood, Welleck等, 2019](https://arxiv.org/abs/1908.04319) / [RepeatNet, Ren等, 2018](https://arxiv.org/abs/1812.02646) | 压低不希望生成的token；区分重复与探索推荐 | 思想相关，但G0错误几乎都在未见类别之间，且已有排除因子实验。暂不优先追加seen惩罚或repeat分支，更不硬屏蔽seen |
| [Supervised Contrastive Learning, Khosla等, 2020](https://arxiv.org/abs/2004.11362) | 拉近同类样本、推远异类样本 | 同一个未来类别不等于相同历史偏好；当前数据又只有观测标签，强行按单标签拉近/排斥用户可能造成错误归纳。暂不新增SupCon |
| [LLaMA-Adapter, Zhang等, ICLR 2024](https://arxiv.org/abs/2303.16199) / [Poly-encoders, Humeau等, ICLR 2020](https://arxiv.org/abs/1905.01969) | 零门控逐步注入新分支；多表示与条件读取折中表达和效率 | 为保留全文路径、逐步加入兴趣读取提供结构依据；不是本任务有效性证明。保留为独立结构备选，不与首轮新loss叠加 |
| [Adapting Auxiliary Losses Using Gradient Similarity, Du等, 2018](https://arxiv.org/abs/1812.02224) / [Learning to Reweight Examples, Ren等, ICML 2018](https://arxiv.org/abs/1803.09050) | 用主任务梯度或干净验证集指导辅助权重 | 比按CIR训练loss选样更直接，但会提前使用N监督或新增目标集访问，改变原阶段合同；不是“免费预热”。暂不执行 |
| [PCGrad, Yu等, NeurIPS 2020](https://arxiv.org/abs/2001.06782) | 对联合多任务的冲突梯度做投影 | 当前是CIR后N的顺序训练，没有同时优化的两项loss；不能仅凭C低分就加入PCGrad |

## 3. 首选：在普通全文路径上检验排名辅助损失

### 3.1 研究问题，而非效果承诺

保留当前最强且已经验收的G0输入、骨干和CIR预热。只问一个问题：N阶段除原生成CE外，显式强调高分错误ID，是否能改善最终新兴趣排序？

这不是说CE没有排序能力。softmax CE本身具有listwise性质；在正确模型和总体分布下，交叉熵学到的条件概率排序本来就是合理基线。BPR-max原文主要解决平均成对loss被大量容易负例稀释的问题，而我们并不使用平均BPR。因此这里仅有有限预算下优化偏置可能有用的假设，不能声称修复了已证明的CE缺陷。

### 3.2 一个额外目标

设z为首个Interest ID预测位置的112个ID logits，y为训练目标。其他111类为当前单标签排名任务的竞争项，不等于已证明用户不喜欢它们。对j不等于y：

$$
w_j=\frac{\exp z_j}{\sum_{k\ne y}\exp z_k},\qquad
\mathcal L_{\mathrm{rank}}=-\log\sum_{j\ne y}w_j\,\sigma(z_y-z_j).
$$

这是GRU4Rec+式(12)的未加score正则版本。实现用log-softmax、log-sigmoid和log-sum-exp，FP32归约，不在BF16里直接取极小概率的log。

$$
\mathcal L=\mathcal L_{\mathrm{original\ full\ vocab\ assistant\ CE}}+
\lambda\mathcal L_{\mathrm{rank}}.
$$

首轮只比较lambda为0和1；1是预先固定的设计选择，不是论文针对本模型推荐的最优值。保持原CE全部监督token、分母、梯度累积缩放以及EOS训练不变。负类权重参与求导，不偷偷detach。112类很小，不新增负采样。

原论文最终式(16)还使用绝对负分数平方正则。这里暂不采用：LM logits整体平移不应改变ID内部排序，而把部分ID的原始logits拉向零还会改变它们相对全词表的行为。这个取舍意味着我们是借鉴式(12)，不是复现论文最优配置。

不得用Dev生成hard-negative列表，不对完整seen做mask，不输入候选表，不增加教师、不更改推理。推理仍是原full-vocabulary deterministic 20-beam与原解析器，不拿112类直接排序替代正式指标。

**风险。** 高分竞争项可能是真实但未记录的其他新兴趣，硬负例压力可能过强；首ID分数与含终止token的beam序列分数也不完全相同。不同loss会改变梯度规模，不能把所有收益都归因于难负例机制。若首轮无效，不继续扫描lambda来“调到赢”。

### 3.3 最小且可解释的实验

| 条件 | 起点 | N阶段 | 目的 |
|---|---|---|---|
| CE匹配对照 | 同一个已验收的普通CIR三轮成品 | 原112K一轮/3500步，原CE | 控制本次执行环境与布局 |
| CE+排名 | 完全相同的CIR成品与初始化 | 同112K、行序、3500步，仅lambda=1 | 唯一干预是新增排名目标 |

不使用C或N独立编码成品，不从已训练完N的G0成品再多训一轮。两组都重置N阶段optimizer/scheduler，使用相同seed、LoRA、BF16、LR调度、有效batch32和执行布局。复用CIR权重是在原阶段边界复用已完成的计算，不是砍掉CIR预算。旧G0保留为历史参考，不代替可能有执行轨迹差异的匹配对照。

本轮不同时改结构、加数据或改变CIR。正式启动前记录CIR成品SHA、N缓存SHA、运行布局与新合同；后端和包必须在独立目录，不能覆盖原117个冻结文件或旧回执。

## 4. 结构备选：保留全文的一次兴趣重读

只有需要结构实验时单独检验，不与上面的loss合并。由一次普通LLM前向得到原预测位置h与历史事件状态，按可见类别池化得到m_c；用h而非真实输出类别作query：

$$
\widetilde h=h+\tanh(g)W_o\sum_c\operatorname{softmax}_c
\left(\frac{(W_qh)^\top W_km_c}{\sqrt d}\right)W_vm_c,\quad g=0.
$$

随后复用原lm_head生成全词表。原历史始终可见，不增加第二遍LLM、独立类别编码器、外部记忆或候选查询。零门控只保证初始化时可还原原路径，不保证训练后不退化。投影不能与门控全部同时置零而导致无梯度。

相对旧RelationHead，这里是历史状态驱动的读取，而非对112个类别prototype逐对补分；但最后线性head下，隐藏残差仍可写成logit残差，不能把“融合发生在hidden”本身当作本质创新。应以“动态query是否必要”和“真实分组是否优于同大小随机组”判断归纳偏置是否有效。已有LLM自注意力也能聚合历史，这项额外结构可能完全多余，故排在简单loss检验之后。

## 5. 启动门禁与停止规则

1. 先过数学/代码门禁：目标升分时rank loss下降；所有logits整体平移后值不变；极端logits与BF16输入归约有限；目标不进入负类；lambda=0退化为原CE；单卡累积与整批归约一致。通过这些不等于9B训练有效。
2. 首轮真实GPU smoke验证原full-vocabulary CE未变、标签无输入泄漏、8卡梯度归约、完整checkpoint恢复和20-beam链路。使用已有最长样本smoke惯例，不把少步数smoke分数当作方法效果筛选。
3. 只有上述门禁通过才运行两组完整N预算；仅授权Feeds/Recall、Standard、A10080。提交前新查两目标，quota不是物理可调度能力证明。已有R健康作业不取消、不重复提交。
4. 用实际稳态吞吐、checkpoint写盘、评测及收集时间估计总执行预算，再安排并行；不能用前一次恢复进度条估计吞吐，不因目标8小时而减少数据/轮数/beam。原两次八小时失约继续保留，新研究执行记录单列，不抹掉历史耗时。
5. 最終原Dev2000、112类、四项Macro完整报告，格式、缺失、重复率和权重来源单独验收。初筛沿用已使用的实际意义门槛：相对匹配CE的nDCG@20至少增加0.005，Recall@20不下降，同时对照历史G0；这不是显著性检验。
6. 若只改善辅助loss、训练loss、类别覆盖或已见排除而没有N排序收益，停止。若匹配CE明显偏离历史G0，先核合同与执行，不能选择性只报较弱基线。
7. Dev已经被多轮研究复用，只能称探索性初筛。真正宣称改进需要配对seed与预先封存的独立用户测试；在确认其来源、隔离与授权前不编造独立测试结果，也不从原112K中悄悄删样本当验证集。

## 6. 本轮完成范围

已完成文献适配核对和G0/C/N已有预测的离线误差分析；首选一个loss对照，结构仅为备选。本轮调研不把任何候选称为已验证改进，也不作首创声明。正式训练尚未提交，原CIR/N合同不被这份设计自动替换。

代码依据：[现有压缩路径](../../../src/crossinterest/methods/composition/model.py)、[实际CIR来源与目标不可见断言](../../../src/crossinterest/methods/composition/runner.py)、[整类留出](../../../src/crossinterest/methods/mechanism/data.py)、[既有关系头](../../../src/crossinterest/methods/mechanism/model.py)、[既有因子loss](../../../src/crossinterest/methods/factorization/loss.py)、[已有冻结CIR残差迁移](../../../src/crossinterest/methods/residual/model.py)。历史方法边界只读核对了原设计与结果（私有原始证据，不随公开仓库分发），未修改该文件。

## 7. 已降级基线：共同未见类别的归属学习

本节保留候选的推导、Train检查和被否决的主方法依据，不再具有执行优先级。最近邻CPR已使用相同的二用户二目标交叉分数；仅加入共同未见资格和生成CE不满足当前用户要求的论文主方法标准。名称 jointly-unseen category assignment learning 仅描述适配，不申报新损失。

### 7.1 从文献得到的取舍

本轮补读了ICLRec、DuoRec、CauseRec、DCRec、DICE v1、CrossDistil的方法正文与损失；下表明确区分正文核验和摘要核验。不能将论文报告的收益照搬到9B生成模型，也不能根据关键词检索没有命中就声称首创。

| 文献与核验层级 | 直接相关机制 | 本任务的区别或不采用理由 |
|---|---|---|
| [CPR, Wan等, WWW 2022, §3.1式(8)、§3.2、§4.1](https://arxiv.org/html/2204.12176v1), 正文 | 两个真实交互与两个交叉未观测交互，softplus作用于四分数差的一半；抵消用户/商品项，并明确联合次序不保证个体次序 | **直接最近邻。** 本节loss为softplus(-Delta)，CPR为softplus(-Delta/2)，同一机制、不同分数尺度。共同未见和LLM首ID是任务适配，不能把它们包装成全新归属目标；CPR曝光分解的无偏结论也不能移植到本数据 |
| [ICLRec, WWW 2022, §4.1-4.2](https://arxiv.org/html/2202.02519v1), 正文 | K-means意图原型、序列到原型InfoNCE、同意图负例排除 | 这里不估计隐意图、不把用户表示拉到原型；直接在最终输出类别分数上监督两份真实标签的归属 |
| [DuoRec, WSDM 2022, §4.3](https://arxiv.org/html/2110.05730v1), 正文 | Dropout视图、同下一商品的不同序列作为正例、表示对比 | 不假设同一个未来类别意味着整个用户兴趣相同，不把不同未来类别的完整用户向量推远 |
| [CauseRec, SIGIR 2021, §3.3.2-3.3.5](https://arxiv.org/html/2109.05261v1), 正文 | 按目标相关性识别必要/非必要概念，替换历史，构造正负反事实表示与triplet损失 | 不合成历史，不假定删掉某行为后未来标签仍不变或一定改变；交换的是两份标签的归属假设，不是伪造用户行为 |
| [DCRec, WWW 2023, §2.3-2.5](https://arxiv.org/html/2303.11780v1), 正文 | 序列、转移图、共现图的对比和从众权重 | 不引入GNN或从众估计器；只对可证明抵消的共享加性类别项作陈述，不能称真实兴趣与从众已被识别 |
| [DICE, arXiv 2006.11011v1, §2](https://arxiv.org/html/2006.11011v1), 正文 | 将兴趣/流行度分成两组embedding，依赖点击与流行度关系作BPR监督 | 本轮读的是题为Interest and Popularity Bias的早期版本，不混称最终版本全文；N标签均衡且没有曝光数据，不引入该因果分解假设 |
| [CrossDistil, AAAI 2022, §3.1](https://arxiv.org/html/2202.09852v1), 正文 | 四类跨任务反馈的三组成对次序损失，再校准蒸馏 | 其quadruplet不是“两位用户、两个目标”的交换归属似然；这里没有跨任务软标签或教师，仍须承认四分数排序并非凭空出现的新范式 |
| [ICSRec, WSDM 2024](https://arxiv.org/abs/2310.14318), 摘要 | 同目标子序列的粗细粒度意图对比 | 同目标对齐并不是本候选的机制；目前不能声称核对过所有实现细节 |
| [IDCLRec, 2025](https://arxiv.org/abs/2501.07096), 摘要 | 稳定兴趣/动态意图分解、交叉注意力和item-aware对比 | 不用更复杂分解代替尚未确立的任务假设；不是已完整排除的所有相邻方法 |
| [CETRec, 2025](https://arxiv.org/abs/2507.03047), 摘要 | 为LLM引入时间反事实调优 | 时间顺序不足是另一假设，不能用现有低分直接断定；暂不同时加入时间损失 |
| [CaseRec, SIGIR 2025](https://arxiv.org/abs/2504.13482), 摘要 | 曝光序列、替换曝光、用户模拟器预测反馈与decision transformer | 当前没有曝光序列，也不接受不可靠教师/模拟器标签，因此不能直接借用 |
| [Intent Disentanglement and Feature Self-supervision for Novel Recommendation, TKDE 2022](https://arxiv.org/abs/2106.14388), 摘要 | 尾部及冷启动商品的novelty | 平台长尾/冷启动不等于用户完整历史从未出现过的类别，不能据标题认为任务相同 |

覆盖的路线包括多兴趣结构、时间演化、重建预热、协同建模、排序目标、表示对比和反事实/去偏；本节不是穷尽检索。条件logistic、配对选择与交换匹配本身是已有统计工具，不能把下面的softplus或四项相减宣称为新发明。

### 7.2 唯一核心问题

对于真实样本$(H_u,a)$和$(H_v,b)$，若$a,b$对两位用户都属于尚未进入的类别，模型能否根据两份历史区分“谁走向哪一个”，而不是仅提高一个对所有人都高分的类别？

这是由G0错误主要位于未见类别之间而提出的可证伪假设，不是已证明G0只学类别偏置。保留完整原B+I路径、普通CIR三轮预热和N一轮；CIR仍是辅助，不将回顾性补全改称真实首次兴趣监督。本候选只干预N阶段监督，不增加压缩桥、教师、查询模块或推理重排。

### 7.3 交换归属的条件似然

令$z_u(c)$为读取原历史$H_u$后、首个Interest ID预测位置的全词表logit在类别$c$上的值。令$S_u$为该训练样本时间点之前的完整已见类别集合。合法比较须满足：不同用户、$a\ne b$，且$\{a,b\}\cap(S_u\cup S_v)=\varnothing$。完整seen仅在训练外部构造比较资格，绝不进入prompt、hidden state或推理；它是额外训练元数据用途，正式实施前须显式登记，不能伪称与原CE的监督合同完全相同。

两种归属的分数差为

$$
\Delta_{uv}=z_u(a)+z_v(b)-z_u(b)-z_v(a).
$$

在独立用户样本的条件预测模型下，给定历史和无序目标集合$\{a,b\}$，观测归属相对于交换归属的概率为

$$
\frac{p(a\mid H_u)p(b\mid H_v)}{p(a\mid H_u)p(b\mid H_v)+p(b\mid H_u)p(a\mid H_v)}=\sigma(\Delta_{uv}),
\qquad \ell_{uv}=\operatorname{softplus}(-\Delta_{uv}).
$$

每位用户的全词表softmax分母在比值中抵消，因此不需要为四种组合做四遍LLM，也不需要改变推理词表。训练时两份原样本各前向一次，交叉读取各自已有的两个ID分数即可。这个概率是观测标签归属的模型概率，不是用户潜在偏好的因果概率。

同一批多个pair会共享样本，所以求和是复合条件目标，不能把所有pair当成独立新增观测来计算显著性。完整过去$S$超过模型可见窗口，资格选择可能改变观测样本分布；上式是模型内部归一化恒等式，不证明只见$H$的模型在筛选后仍然校准。

重要的选择条件：比较资格和权重必须在$a,b$互换后不变。均匀枚举批内合法pair满足该条件；只由两份历史决定的相似度权重也满足。按“本用户当前分错的目标”非对称挖hard pair或按单边目标选邻居会破坏这个推导，第一版不做。按类别作独立重采样的因子$q_aq_b$也会在两种归属之间抵消，但用户/历史相关的未知抽样概率不会自动消失。

### 7.4 精确性质和不能宣称的结论

若$z_u(c)=r_u(c)+b_c+d_u$，则$\Delta_{uv}$恰好消去共享类别项$b_c$和每个样本的常数项$d_u$。其梯度对每个共享类别偏置为零；这不要求模型显式有bias参数。若所有用户仅有相同类别偏好，$\Delta=0$，该辅助项始终为$\log2$，不能靠共同抬高类别分数降低。

若输出head线性，则$\Delta=(h_u-h_v)^\top(w_a-w_b)$；仅鼓励与目标差异相关的历史差异，不强迫两位用户的完整表示互相远离。这与旧设计§14.4的差分蒸馏有代数联系，区别是此处由两份真实N标签提供监督，不匹配冻结CIR教师、不约束其全部112类差分、不额外读取历史删减视图。抵消偏置的代数本身不能重复申报创新。

与两条BPR的区别：令$x=z_u(a)-z_u(b)$，$y=z_v(b)-z_v(a)$，本项是$\operatorname{softplus}(-(x+y))$，而BPR平均是$[\operatorname{softplus}(-x)+\operatorname{softplus}(-y)]/2$。前者联合判断标签归属，后者要求两条绝对排序分别成立。前者可能由一个用户的强证据补偿另一个用户的错误，所以不能替代CE，也不能把pair准确率等同N的Recall/nDCG。

限制必须保留：未知曝光和用户特定流行度不被消除；历史集合不一致时，背景类别共现项不被精确消除；未观测类别不是永久负兴趣；归属监督不会解决全部false negatives；首ID分数不是完整beam序列分数；在足够模型容量和总体风险最优时，CE本来可以学到条件分布，收益只能作为有限样本/有限更新下的归纳偏置假设检验。

### 7.5 Train支持检查与收敛后的版本

只读原Train（私有原始证据，不随公开仓库分发），不读Dev来构造pair。112,000行、112类各1,000；`target_selection=seeded_uniform_label_then_product`，`same_history_new_interest_count`全部为1。此字段统计不代替对上游首次出现判定的已有审计。

来源核验时间2026-09-23T01:29:17Z，SHA256为`43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd`，共102,633个用户。逐行核验`split=train`、目标不在完整seen、可见seen属于完整seen，违规0条。统计及CPU检查摘要保存于研究审计（私有原始证据，不随公开仓库分发）。

| 条件：目标不同、用户不同、双方完整过去均未见两个目标 | 可参与行 | 目标类别覆盖 | 每类可参与行数 | 无序pair数 |
|---|---:|---:|---:|---:|
| 可见历史类别集合精确一致 | 10,388 (9.275%) | 112 | 42-176 | 39,748 |
| 可见类别及每类事件频次精确一致 | 2,598 (2.320%) | 112 | 10-55 | 2,989 |
| 可见类别序列精确一致 | 318 (0.284%) | 109 | 1-10 (已覆盖类) | 418 |

集合精确一致只控制“有哪些类别”，不控制类别频次、顺序、标题或隐藏前缀。不能把这一比较解释成纯商品标题贡献。严格序列一致太稀疏，故不作为全体训练假设；也不删掉其余样本来迁就方法。

作为可行性而非实际Trainer行序证明，用NumPy `default_rng(20260825)`随机置换全部Train、每32行一批：112,000行均至少有一个合法同批对象；合法对象数分位数(min/p10/p50/p90/max)=4/21/27/30/31；合法pair平均可见集合Jaccard=0.119687，21.4304%完全不重叠。支持“共同未见”的批内比较，但不支持把随机批内用户称为兴趣背景相同。

因此第一版只主张共同未见归属学习：所有原样本继续完整CE，在每个实际全局微批内枚举合法pair并均匀平均，$\mathcal L=\mathcal L_{\mathrm{original\ assistant\ CE}}+\lambda\,\mathrm{mean}_{(u,v)\in E}\ell_{uv}$，无合法pair时辅助项为0。拟固定$\lambda=1$、不加温度或margin，不做Dev调参；这是待正式门禁锁定的设计选择，不是已验证最优值。精确同集合只作为机制诊断候选，不强行扩写成第二套主方法。不同批次pair数量会改变辅助样本权重，因此须审计每类参与率/度数，不能将pair级均匀误称样本级均匀。

### 7.6 成为论文主方法需要的证据

最近邻排查已否决本节作为当前论文主方法的推荐：CPR不是仅有类似动机，而是核心目标同式。共同新类别资格、真实N标签和无额外推理结构仍是合理适配，但目前不足以支撑所要求的贡献。下面三组矩阵保留为历史基线设计，暂停实施；超过CE或BPR也不能证明超越CPR，不能靠更换名称或省略CPR基线规避这一点。

第一轮最小辨别矩阵拟为三组，从同一个普通CIR三轮成品启动同预算N：原CE；原CE+同一合法pair上的两条BPR；原CE+交换归属。全部使用同一数据顺序、微批布局、seed和完整监督。三组先回答是否有效及是否超出普通成对排序，不同时训练更多结构。共享偏置不变性和同输入归属不可分是CPU性质检查，不需要占GPU开新组。

控制梯度尺度：正式BPR对照对每个pair使用两项之和$\ell_{\mathrm{BPR}}=\operatorname{softplus}(-x)+\operatorname{softplus}(-y)$，再与归属项一样对pair求平均、取同一个$\lambda=1$。这样在$x=y=0$时，两者对$x,y$的梯度均为$-1/2$；若使用两项平均则会差一倍。零点相同不等于训练全程梯度范数相同，仍应记录两类损失和梯度统计，不把这种对照写成完全等优化难度。

只有优于匹配CE及同pair BPR才继续检验共同未见资格的必要性：相同预算下取消交叉未见资格，并报告实际pair数与监督强度差异；若要归因纯资格机制，还须匹配pair数量/类别组成。不能把这一后续消融与首轮结果预先合并宣称完成。若要进一步主张商品细节决定分叉，还需类别背景控制、标题置换诊断及独立数据验证；当前证据不足，不将其写成主贡献。

效果初筛沿用探索性门槛：相对匹配CE的macro nDCG@20至少+0.005、Recall@20不下降，且超过同pair BPR；历史G0保留，不以较弱新对照替换它。健康门槛及原Dev2000/112类/无候选20-beam全部不变。pair loss降低、覆盖增加或bias不变性通过都不能算N任务成功。初筛失败即停，不继续扫描超参直至胜出。Dev多轮复用，不能称独立测试；论文结论仍需预先封存测试、多seed、适当配对不确定性分析和有针对性的基线。

### 7.7 计算与执行边界

不新增参数、不新增LLM前向、不改变推理。额外张量为全局微批大小乘112的FP32 ID logits及小型pair矩阵，而非全序列hidden states。跨卡收集必须可微，不能detach另一用户，也不能只在micro1本卡配对；标准梯度累积并不会自动构造跨微步pair。8卡micro4时一个同步微步才覆盖有效batch32；任何布局改变须三组同时采用并通过最长输入、完整预算和恢复门禁，不以理论小张量开销代替实测速率。

本轮只做文献、数据只读统计和数学性质验证，不接入正式trainer、不重打冻结包、不提交新训练。后续执行前须保存独立合同、输入与CIR权重SHA、实际全局微批pair覆盖、DDP梯度一致性、完整恢复、运行时间分解；新查Feeds/Recall Standard A10080容量。原R作业独立收尾，不取消健康作业，不重置既有两次8小时失约记录。

核验记录：CPU FP64验证条件似然等价、共享类别/样本平移不变性、bias梯度为零、相同预测归属不可分、交换反对称、与独立BPR不等价；BF16极端输入转FP32后的loss和梯度有限。文档本地链接与公式定界符检查通过。这些只保证上述性质，不替代真实9B、CUDA、DDP、长输入或beam验证。

## 8. 已停止方向：首次进入事件归一化学习

### 8.1 从输出排序转向任务的时间单位

当前任务是在类别首次进入之前预测该类别，不是在每次消费之后预测下一商品。一次新类别进入前可能夹着多次旧类别消费；这些消费应作为证据保留，但不应仅因次数多就让同一扩展事件占据更多辅助监督权重。核心假设是：对真实扩展前的不同前缀进行事件归一化监督，能比仅监督紧邻扩展的最后一个前缀学到更可迁移的扩展规律。同一类别内每个事件等权，112类之间按原macro任务均衡，避免额外引入长尾类别先验。

这不是已发现G0的根因，也不假设未来标签在任意删除/替换历史后保持不变。使用的都是同一真实时间线的过去前缀和后续首次进入事件，不合成行为、不使用教师、不要求各前缀预测分布相同。暂用英文描述 first-entry event-normalized learning，不为名称或普通CE本身申报创新。

### 8.2 严格可观测的进入区间

对完整的合法正反馈历史$H=(x_1,\ldots,x_T)$，用原构造器定义类别首次进入位置$\tau_1<\cdots<\tau_K$，约定$\tau_0=0$。重复事件仍留在$H$。对于目标类别$c_k$在$\tau_k$的首次进入，定义可预测前缀集合

$$
I_k=\{t:\max(5,\tau_{k-1})\le t<\tau_k\}.
$$

每个$t\in I_k$的下一个新类别均为$c_k$，但下一个商品可能属于旧类别。若区间为空则不产生辅助事件；不越过不在112类词表的真实新类别去选择更晚的支持类。首次进入按完整合法交互前缀定义，不等于现实生活中的第一次购买。

只从原Train可见历史内部提取事件，不使用原N目标商品/标题、不读Dev、不补入H50之外的文本。原构造器[build_user_records](../../../src/crossinterest/datasets/amazon/scripts/build.py#L267)在截取H50之前保存`full_history_length`和`<archived-job-id>`。因此第一版仅对`full_history_length == len(history)`且完整seen一致的源历史建立辅助事件；其余原N记录仍完整参加主CE，不从112K删样本。存在同时间戳的源历史暂不建立辅助事件，避免强行给同时发生的类别排序。

已只读核验：108,185/112,000条没有左截断，其中完整seen不一致0条、含同时间戳10条；原N目标覆盖112类，每类完整前缀支持880至999条。进一步对108,175条完整且无时间戳并列的历史扫描，81,743条至少含一个合法辅助事件，得到370,275次事件出现；去掉跨源重复29,118次后为341,157个唯一事件、546,420个可用前缀。112类均覆盖，每类67至17,373个事件。113,502个事件有多个可预测前缀，占33.27%，这部分也覆盖112类，每类25至5,521个事件；区间平均1.6017个前缀、最长30个。

这些是**输入签名隔离之前的候选池支持**，不是已经冻结的辅助训练集。逐前缀断言目标不在完整过去、下一真实新类别恰好是目标；跨源同事件的完整上下文一致性通过。摘要见只读研究审计（私有原始证据，不随公开仓库分发）的`<archived-job-id>`字段。

跨源记录用`(user_id, <archived-job-id>, category)`去重，不能让同一物理事件因为出现在多条N历史中被重复计为独立监督。同一事件可见的完整过去应一致；若不一致则停止回源检查，不能挑一条更有利的历史。

正式构造还要复用已有的规范化B+I历史签名，隔离原N/O/Dev输入及检查Train/Dev用户隔离。为使两种增广保持可比，只要某事件的任一辅助前缀触发受保护输入重复，就从两组共同事件池删除整个事件，不删主任务记录；随后重新核验逐类支持，再锁定事件抽样序列。Dev只参与隔离资格检查，不提供目标、选样难度或训练信号。本轮没有实施这一步，所以不提前声称最终池规模。

### 8.3 只增加一个事件归一化目标

令$\mathcal E_c$为类别$c$去重且合法的辅助进入事件。$p^{112}_\theta(c\mid H_{\le t})$来自**同一模型、原新兴趣prompt、首ID位置**的112个ID logits归一化，不对已见类作mask。辅助目标为

$$
\mathcal L_{\mathrm{entry}}=
\frac{1}{112}\sum_{c=1}^{112}\frac{1}{|\mathcal E_c|}\sum_{k\in\mathcal E_c}
\frac{1}{|I_k|}\sum_{t\in I_k}
-\log p^{112}_\theta(c_k\mid H_{\le t}),\qquad
\mathcal L=\mathcal L_{\mathrm{original\ assistant\ CE}}+\lambda\mathcal L_{\mathrm{entry}}.
$$

先均匀抽类别，再类内均匀抽去重事件，再在其$I_k$中均匀抽前缀，是这个有限池目标的无偏随机估计；这不等于对真实用户分布、曝光机制或宏指标无偏。无权重地枚举所有前缀则使事件权重正比于$|I_k|$。本设计只保证同类事件总权重不随区间长度线性增长，不保证新增重复行为后模型输出或损失不变，因为输入证据已经变化。类均衡和分层抽样本身是常规工具，不另列创新。

所有原N112K仍做一次完整生成CE，包括原监督token和EOS；普通CIR三轮成品只作共同初始化，不改变其辅助地位。暂定每条N搭配一个预先固定seed抽取的辅助前缀，112类各1,000次辅助抽样、类内事件允许有放回，$\lambda=1$不作Dev搜索。67个事件重复抽1,000次不等于1,000个独立事件，须报告逐类唯一事件数和重复率。这是新增监督及计算，需独立合同，不能声称数据预算或FLOPs与G0完全相同。

不在原长历史中直接取事件位置的hidden作为已有assistant输出状态：原任务指令位于历史之后，中间事件状态尚未读到它。第一版显式用原renderer渲染独立的前缀问答，额外做一次前向；不新增head或参数、不让该前缀注意其后的历史、不改推理。将多个前缀折叠成一次前向是另一个需要因果mask和混合注意力验证的工程问题，当前不承诺零额外前向。

### 8.4 最近邻与论文贡献门槛

| 最近邻 | 已有做法 | 本候选必须实证成立的区别 |
|---|---|---|
| DIEN及标准自回归序列学习 | 中间状态预测下一行为/商品 | 辅助预测跨过旧类别重复消费后的下一真实首次进入类别；不是将普通next-item CE改名 |
| BERT4Rec、CIR；[MHL, ACL 2026](https://arxiv.org/abs/2509.23649)，后者本轮仅核对摘要 | 历史遮蔽重建；MHL含熵引导mask和重建到预测课程 | 无遮蔽、无事后上下文补全，使用真实时间前缀；不把本方法说成首次使用历史辅助监督 |
| RepeatNet；[Novelty-Seeking HRL, 2023](https://arxiv.org/abs/2306.01476)，后者本轮仅核对摘要 | 区分重复/探索或用分层策略和奖励控制新颖性 | 已知下游专门预测新类别，不预测是否探索、不修改策略奖励；研究的是进入类别及其事件采样单位 |
| CPR | 跨用户交叉排序与偏置抵消 | 不跨用户造负例、不使用四分数交叉项；直接监督同一真实时间线上可验证的进入事件 |
| 本项目旧设计14.3 | CIR的F/M/P撤除课程，P仅窗口内首目标之前 | 本方法不迁移原CIR样本或逐轮撤除证据，严格完整前缀验新，在N阶段学习真实进入区间；不是给旧课程更名 |

单纯“next-new辅助CE”仍可能被视为任务适配或数据增强，不能仅凭这张表保证论文贡献。值得验证的主张应限定为：**以真实扩展事件而非消费步为监督单位，覆盖扩展前的观测前缀，是否在固定新增标签和计算对照下改善下一新兴趣预测。** 必须证明进入区间采样优于额外历史边界样本，并进一步证明事件归一化有价值，否则退回普通数据增强基线，不包装成主方法。更专门的首次进入/序列增广近邻仍需在论文定稿前系统补查。

### 8.5 最小可证伪矩阵与停止规则

首轮三个条件复用同一普通CIR成品，完整N112K/一轮、seed、微批和原20-beam协议相同：

| 条件 | 辅助事件与标签 | 辅助可见前缀 | 能回答什么 |
|---|---|---|---|
| 匹配CE | 无 | 无 | 原任务基线，保留历史G0参考 |
| 历史边界增广 | 固定112K次去重事件抽样 | 每次固定为$H_{<\tau_k}$ | 仅增加真实首次进入训练样本是否足够 |
| 事件归一化区间学习 | **完全相同的事件顺序和目标标签** | 每次在$I_k$内固定seed均匀抽取 | 更早的真实前缀覆盖是否超出边界增广 |

后两组相同模型、辅助次数和目标分布；前缀长度不同，所以不是天然等tokens/FLOPs，应报告实际tokens、GPU小时及预算差异。不存在重复间隔的事件两组相同，不强造差异；必须先量化有非紧邻前缀的事件支持。阶段一通过后，才比较前缀均匀与事件均匀，以及普通下一类别辅助任务；涉及类别分布变化须匹配或单独报告，不能把所有变化归因于归一化。

最主要的风险是监督难度和分布错位：当前正式N输入紧邻一次新类别进入，更早前缀的信息更少，未来首次进入也可能难以预测。区间训练可能分散有限更新而非帮助泛化。不得把这部分损失下降称为已找到了潜在兴趣产生时刻；只要同事件边界增广更好，就否决更早前缀的主张。其次，辅助历史类别并不服从原N用户抽样分布，类均衡不能消除所有选择效应。

沿用探索性门槛：区间学习相对匹配CE的macro nDCG@20至少+0.005、Recall@20不降，且超过同事件的边界增广。只提高辅助准确率或仅优于无额外监督的CE，不足以支持主机制。失败停止，不扫描时间窗口/权重直至赢。论文级结论还需独立封存测试、多seed、用户级配对不确定性，以及普通序列增广和CPR等针对性对照。

实施前先审计辅助112类支持、真实重复间隔覆盖、源记录重叠、Train/Dev用户隔离和前缀输入签名隔离；这些均未被“完整前缀占96.6%”替代。数据不足或所有有效区间几乎只有一个前缀将直接否决本候选，而不是放宽首次进入定义。GPU门禁还包括原CE不变、无未来token可见、最长输入、完整恢复、辅助loss归约及额外前向的实测总耗时。研究时间不计入新执行窗口；不保证未测的8小时，不减少数据/轮数/beam。当前仅设计和CPU/Train只读检查，未改trainer或提交新作业。

## 9. 固定数据上的结构候选：按预测角色分解的兴趣扩展算子

### 9.1 最近邻与不能再声称的创新

本节区分正文核对与摘要核对。不是穷尽检索，也不是把没有搜到相同名称当作新颖性证明。

| 文献与阅读范围 | 已有机制 | 对当前设计的约束 |
|---|---|---|
| [PIMI, IJCAI 2021](https://arxiv.org/html/2106.04415v1), 方法3.2-3.4 | 时间间隔、序列邻接与中心节点GNN交互，再抽取多兴趣；训练按目标选最大相关兴趣，测试聚合各兴趣检索 | 多兴趣交互并非空白，且其兴趣表示已经包含其他行为信息，不能简单说成独立单源模型 |
| [Modeling High-order Interactions across Multi-interests, 2021](https://arxiv.org/html/2104.00305v2), 方法正文 | like/follow两种行为层级之间co-attention，再做层内self-attention和融合 | 原文是AAAI Student Abstract，不是完整长文；其多兴趣主要指行为层级，不能直接等同两个已见类别支持一个未见类别 |
| [DSIE, ECAI 2023](https://arxiv.org/html/2310.10025v1), 方法3.2-3.5及评测4.4 | 全局偏好编码、局部兴趣、全局引导聚合；shuffle对比与prototype去相关；明确讨论跨出历史类别 | 与“联合已见兴趣推断潜在新兴趣”的动机非常接近。其新颖性评测过滤已见类别商品并补足推荐数量，不是本项目的无过滤全词表生成；不能照搬其增广或评测 |
| [Neural Collaborative Reasoning](https://arxiv.org/html/2005.08129v3), 方法4.1-4.5 | 将联合历史前提到目标写为逻辑蕴含；用MLP实现NOT/OR等算子，逻辑正则与BPR训练 | 本轮读到的v3正文称NLR框架；不能把“从匹配转为推理”或多个兴趣的AND本身当作贡献；也不能把消费事件当作严格逻辑真值 |
| [Higher-Order Factorization Machines, 2016](https://arxiv.org/html/1607.07195v2), 方法2-5 | 低秩多线性交互、不同特征组合、ANOVA核及高效求和 | 下文三线性张量和平方和恒等式都属已有数学工具，不能包装为首次高阶交互或新加速算法 |
| [Purifying Interaction Effects with the Functional ANOVA, AISTATS 2020](https://arxiv.org/html/1911.04974v2), 方法3-5 | 指定参考分布后，通过条件零均值约束把低阶效应与交互分开；相关变量下简单中心化一般不够 | 均匀类别轴中心化不等于真实用户分布上的纯协同，更不是因果识别；后验分解本身也不改变预测 |
| [ComplEx, ICML 2016](https://arxiv.org/abs/1606.06357), 摘要 | 复数嵌入可表达对称和反对称二元关系 | 对称/非对称关系分解不是新数学。本候选必须证明CIR到N的特定参数共享方式有价值 |
| [Statistical Ranking and Combinatorial Hodge Theory](https://arxiv.org/abs/0811.1067), 摘要 | 排序边流的梯度与循环分解 | 下文只是有限角色置换投影，不称为新Hodge分解，也不额外引入集合格能量模型 |

[Multi-Interest Recommendation: A Survey (2025)](https://arxiv.org/abs/2506.15284)本轮仅核对摘要与书目信息，不声称完成其参考文献回溯。[S2-CAR (2026)](https://arxiv.org/abs/2606.25415)本轮仅读摘要，其energy用于Soft-TPP意图分段，不能据此宣称所有能量式推荐都已排除。FPMC的偏好加转移思路已由PIMI和DSIE正文及参考文献定位；[Crossref登记](https://api.crossref.org/works/10.1145/1772690.1772773)确认Rendle等、WWW 2010、811-820页。其出版页访问受限、PDF未被网页工具解析，本轮不宣称独立核对FPMC全文。

本项目[已有关系头](../../../src/crossinterest/methods/mechanism/model.py)也不是线性转移表：它用全文上下文化行为状态做类别组内pooling，含源类别与目标类别的乘积、非线性打分及logsumexp。LLM本身已有跨兴趣交互能力。因此新设计只能主张一种可检验的归纳偏置，不能声称填补了已有模型完全缺失的能力。

### 9.2 任务矛盾与唯一核心假设

**相容不等于接下来进入。** CIR整类删除后的补全可以利用被删除事件之后的其他类别，N只能根据过去预测首次出现的类别。一个类别组合可以相容，但其中不同类别作为下一进入目标的可能性不同。普通CIR到N微调当然可能自行学会区分；目前没有证据证明G0失败是因为不能区分，更不能从压缩负结果推出CIR负迁移。

本候选只检验：**给新增的多兴趣关系算子设置明确共享边界，CIR预热角色无关的相容部分，N再学习目标角色相关的偏差，是否比同参数量的不受约束交互更有效。** 新兴趣预测仍是主任务；无需每个目标都由两种兴趣组合产生，完整LLM路径负责单源、标题、顺序及其他线索。

关键假设不是“CIR真实条件概率对称”。类别频率、删除规则和标题均可使条件概率不对称。本方法只是给一部分可迁移关系施加角色对称先验，必须由对照决定这一先验是否合理。它也不证明N中的角色偏差就是时间因果效应。

### 9.3 最小模型与精确定义

保留G0的原全文B+I输入、Qwen单次前向和原LoRA训练。设$H$为可见历史，$V(H)$为其中不同类别的集合，$h_H$为首ID预测前的隐藏状态。112个类别的固定taxonomy原生token embedding均值为$e_c$，沿用旧关系头的合法信息来源；不添加外部知识、用户ID或完整seen。所有类别都用同一种$e_c$，不能对历史角色用上下文化商品向量、对目标角色用taxonomy向量后仍声称交换对称。

定义一个源角色可交换的低秩算子：

$$
T_H(a,b;c)=\sum_{r=1}^{R}\lambda_r(h_H)\,u_{ar}u_{br}v_{cr},\qquad
u_c=W_u e_c,\quad v_c=W_v e_c.
$$

这里$a,b$是两个不同的可见类别，$c$依次遍历全部112个输出类别。$\lambda(h_H)=W_q\operatorname{normalize}(h_H)$是有正有负的上下文系数，不是新标签或采样权重。每次前向对$u,v$各列减去112类的均值，以排除固定$H$、均匀笛卡尔类别域下的低阶项；不将此称为真实数据分布上的fANOVA纯交互。这也使聚合后的112类残差均值为零，但不保证全词表softmax中类别总概率质量不变。

用相同的$H$及相同类别向量，计算三种目标角色的代数投影：

$$
(\Pi T)_H(a,b,c)=\frac{T_H(a,b;c)+T_H(b,c;a)+T_H(c,a;b)}{3}.
$$

使用两套独立的小型参数，定义$C_H=\Pi T_H^{\mathrm{comp}}$和$D_H=(I-\Pi)T_H^{\mathrm{role}}$。前者对三种目标角色相同；后者满足

$$
D_H(a,b;c)+D_H(b,c;a)+D_H(c,a;b)=0.
$$

例如三个角色的原始分数为$(6,0,0)$，相容部分为$(2,2,2)$，角色偏差为$(4,-2,-2)$。这只是代数例子，不是Amazon样本或效果。角色交换只在输出层公式内计算，绝不生成交换历史、伪造逆向正标签或把真实目标送回历史编码器。此性质对**固定$H$**成立，不保证换一条真实历史后的分数仍相同。

对所有可见不同类别对聚合，若不足两个类别则残差为零：

$$
r_{\mathrm{CIR}}(H,c)=\frac{1}{\binom{|V(H)|}{2}}\sum_{a<b\in V(H)}C_H(a,b,c),\qquad
r_N(H,c)=\frac{1}{\binom{|V(H)|}{2}}\sum_{a<b\in V(H)}[C_H(a,b,c)+D_H(a,b;c)].
$$

这是模型内部的pooling，不改变训练行权重、顺序、数量或标签。首ID位置的全部112个类别logits加入对应残差；其余词表和位置不变。已见类别也计算分数，不做seen-mask或候选过滤。推理继续原全词表20-beam，不转成112类打分排序器。

训练顺序仍为原CIR83,888条三轮后N112,000条一轮，原CIR set CE及N完整assistant-only CE保持，不添加BPR、对比或逻辑正则。CIR只启用$C$；N继承$C$并启用新的$D$，两者均可被N更新。每个分支的$W_q$在初次启用时置零，$W_u,W_v$非零初始化，使残差初始为零且$W_q$有梯度；不要再叠加另一个零门控造成全分支无梯度。共享LoRA仍按G0继承，不改成之前失败的冻结教师加新LoRA方案。

**保证的范围。** 角色子空间在完整三角色、均匀内积下正交，且投影幂等；这不等于两个训练任务的梯度正交。CIR不直接训练新增$D$参数，但仍会更新基础LLM，所以不能宣称整个模型完全消除了回顾性偏置。有限标签也不能保证张量因子、真实兴趣或某条规则被统计识别。$C,D$的训练支持、条件聚合与LLM残差相加后均不自动具有“纯因果协同”解释。

### 9.4 一次前向下的精确快速计算

无需枚举每个类别对或对112个类别分别调用LLM。对每套$u,v$，令

$$
A=\sum_{a\in V(H)}u_a,\quad B=\sum_{a\in V(H)}v_a,\quad
Q=\frac{A\odot A-\sum_a u_a\odot u_a}{2},\quad
M=A\odot B-\sum_a u_a\odot v_a.
$$

则投影后的未归一化聚合恰为

$$
\sum_{a<b} C_H(a,b,c)=\frac13\langle\lambda(h_H),v_c\odot Q+u_c\odot M\rangle,
$$

$$
\sum_{a<b} D_H(a,b;c)=\frac13\langle\lambda(h_H),2v_c\odot Q-u_c\odot M\rangle.
$$

两式各自使用所属分支参数。原始类别对求和的$O(|V|^2\cdot112\cdot R)$计算降为投影之后的$O((|V|+112)R)$；类别embedding投影另计$O(112dR)$，不能从总成本中隐藏。以$R=64,d=4096$为设计示例，六个无bias投影共1,572,864个参数；秩64沿用附近关系头宽度，不是实验选出的最优值。平方和技巧是已有FM工具，不作为创新。

与旧独立编码不同，这里不压缩并替换历史、不增加第二次9B前向、不调用教师，也不按类别对构造新训练样本。额外算子很小不等于整轮能在8小时完成；原LLM训练、beam评测、集群排队和产物收集仍可能主导总耗时。实际执行前仍需新合同、两target即时容量查询、最长输入与恢复检查、实测吞吐后安排多条件并行，不降低原预算来承诺8小时。

### 9.5 能否成为主方法：必须通过的反证

潜在贡献不是三线性乘法、投影、CIR或零初始化这些已有部件，而是一个可验证的学习原则：**回顾补全监督与未来进入监督，对多兴趣关系的预测角色采用不同的参数共享约束。** 这是当前优先候选，不是已经成立的文献空白或顶会贡献。若同参数量普通交互同样好，就没有证据支持该核心原则，不靠命名保留主方法地位。

| 条件 | 唯一主要改动 | 判别目的 |
|---|---|---|
| 匹配G0 | 无新增关系算子 | 是否超出已验收的全文CIR到N基线 |
| 普通双分支三线性交互 | 同样六个投影、同样初始化与阶段；以$T^{comp}$和$T^{role}$代替$\Pi T^{comp}$和$(I-\Pi)T^{role}$ | 区分角色约束与增加参数/普通高阶交互 |
| 角色分解候选 | 第9.3节原样 | 主条件 |
| 进入N时重置$C$ | 保留同一CIR基础LLM，只把$C$恢复到预热前初始化；$D$照常 | CIR学到的结构参数是否真的有迁移价值，不把CIR整体移除当作唯一证据 |
| 单源版本 | 把双源三角色投影换成单源二角色的对称/反对称投影，保留相同投影矩阵规模 | 是否确实需要双兴趣组合，而非普通源到目标关系 |
| 只保留相容分支 | N不启用$D$ | 额外角色自由度是否必要；本条件参数量不同，不能单独证明机制 |

各条件固定原数据、seed、epoch、effective batch、LoRA、可见信息和评测。数学投影的计算量和有效函数空间不同，需报告，不能把矩阵数相同说成完全同等容量。普通双分支使用同样中心化，避免把低阶项约束混成角色投影收益。已有关系头结果用于历史背景；需要严谨直接比较时必须对齐其训练预算，而非拿不匹配历史结果作显著性对照。

探索性通过门槛：相对匹配G0的macro nDCG@20至少+0.005、Recall@20不降，且优于同参数普通双分支；重置$C$、去除$D$与单源版本分别检验迁移、角色自由度和组合主张。某一消融没有支持，就撤回对应主张，不能只靠总分上涨串成完整故事。失败不扫描多个秩/权重直至赢。论文结论仍需独立封存测试、多seed及用户级配对不确定性；当前反复查看的Dev2000只能探索使用，不能冒称独立检验。

### 9.6 可执行的代数反证检查

下面只产生随机小张量用于公式单元检查，不读取或构造任何训练样本，不优化模型。检查角色对称、零和、投影幂等、均匀内积正交、固定上下文均匀类别轴零均值，以及快速公式和显式枚举在0/1/2/多类别下相等。它既不能证明真实分布上的纯交互，也不能证明泛化、训练梯度健康或模型收益。

本轮已在本地CPU执行下面的代码，结果PASS；文档编辑器检查无错误。未运行9B前向、训练或新评测，未改动原trainer、冻结包或用户禁止编辑的设计文档。

<!-- role-projection-check:start -->
```python
import itertools
import numpy as np

generator = np.random.default_rng(20260825)
category_count, rank = 9, 4
source = generator.normal(size=(category_count, rank))
target = generator.normal(size=(category_count, rank))
source -= source.mean(axis=0)
target -= target.mean(axis=0)
context = generator.normal(size=rank)
tensor = np.einsum("ar,br,cr,r->abc", source, source, target, context)

def project(values):
	return (values + values.transpose(1, 2, 0)
			+ values.transpose(2, 0, 1)) / 3

compatible = project(tensor)
directional = tensor - compatible
np.testing.assert_allclose(project(compatible), compatible, atol=1e-12)
np.testing.assert_allclose(project(directional), 0, atol=1e-12)
np.testing.assert_allclose(compatible + directional, tensor, atol=1e-12)
for permutation in itertools.permutations(range(3)):
	np.testing.assert_allclose(compatible.transpose(permutation), compatible,
							   atol=1e-12)
assert abs(np.sum(compatible * directional)) < 1e-10
for values in (tensor, compatible, directional):
	for axis in range(3):
		np.testing.assert_allclose(values.mean(axis=axis), 0, atol=1e-12)

for history_count in (0, 1, 2, 5, category_count):
	selected = generator.permutation(category_count)[:history_count]
	source_sum = source[selected].sum(axis=0)
	target_sum = target[selected].sum(axis=0)
	pair_sum = (source_sum**2 - (source[selected]**2).sum(axis=0)) / 2
	cross_sum = source_sum * target_sum - (
		source[selected] * target[selected]).sum(axis=0)
	fast_compatible = ((target * pair_sum + source * cross_sum) * context).sum(axis=1) / 3
	fast_directional = ((2 * target * pair_sum - source * cross_sum) * context).sum(axis=1) / 3
	slow_compatible = np.zeros(category_count)
	slow_directional = np.zeros(category_count)
	for first, second in itertools.combinations(selected, 2):
		slow_compatible += compatible[first, second]
		slow_directional += directional[first, second]
	np.testing.assert_allclose(fast_compatible, slow_compatible, atol=1e-12)
	np.testing.assert_allclose(fast_directional, slow_directional, atol=1e-12)

example = np.array([6.0, 0.0, 0.0])
np.testing.assert_allclose(example - example.mean(), [4.0, -2.0, -2.0])
assert np.linalg.norm(directional) > 1e-6
print("PASS: role projection and exact pooled contraction; no training/effect claim")
```
<!-- role-projection-check:end -->

## 10. 2026-09-27：重新设计的可证伪起点

两轮角色投影均未达到效果门槛。当前不重跑 G0，不再以增大 C/D 的尺度或秩作为主方案。新候选先验证一个局部机制：在原始完整词表生成器内部，解除 Interest ID 输出更新的 rank-16 限制，同时将类别间概率重分配与 ID 集合总概率质量分开。这里的分解是概率坐标分解，不是 CIR/N 知识正交分解。

原实现已有 `classifier` / `residual_classifier`，因此“加一个全秩分类头”不是新发现。下面仅是研究用代数探针，不是 Amazon 数据构造、训练代码或效果实验；通过后也不能声称真实 Recall/nDCG 提升。正式模型接入尚未实施。

### 10.1 选择与排除

| 候选 | 判断 | 原因 |
|---|---|---|
| 加强旧 C/D 投影、增加交互阶数 | 停止作为当前主方案 | 两轮未达门槛；无法证明真实 CIR 条件分布对称；LLM 本来具有交互能力 |
| 合并 CIR LoRA，再为 N 新建 LoRA | 不作为主方案 | 可释放活动秩，但 CIR 分量可能需要撤销；`M_C + rank-r` 与 `rank-r` 不是无条件包含关系 |
| 只迁移对 N 有益的梯度、调整 LoRA 初始化 | 不作为新贡献 | ATTITTUD、LoRA-GA、LoRA-Pro 已有相关机制；初始梯度近似不能保证真实排序收益 |
| 原有独立分类器或裸残差分类器 | 不重新包装为新方法 | 仓库已有；独立分类排名不能替代本项目完整词表生成评测 |
| **原生成器内部的质量守恒、完整类别更新** | **保留为一个可证伪主候选** | 对实际输出更新瓶颈有严格局部结论，不需要假设 CIR 与 N 对称，不改数据或评分协议 |

这里“完整类别更新”指开放 112 类之间的全部 111 个对比方向，不是全参数微调 9B。候选没有证明超越所有通用模型，也没有证明独立的方法新颖性。它是对 Interest ID + CIR 预热 + 下一新兴趣预测整体方案的一项结构改进，不将三者拆成多个创新点。

### 10.2 精确结构与训练位置

在首个回答词元的预测位置，令 $h\in\mathbb R^d$ 为原 decoder 最终表示，$z\in\mathbb R^V$ 为原完整词表 logits，$S$ 为固定的 112 个注册 Interest ID。新增一个仅在 N 阶段训练、零初始化的 $U\in\mathbb R^{112\times d}$，定义

$$
P=I-\frac{1}{112}\mathbf 1\mathbf 1^\top,\qquad
\delta=PUh,\qquad b=\operatorname{LSE}(z_S),\qquad
\widetilde z_S=z_S+\delta+b-\operatorname{LSE}(z_S+\delta).
$$

集合外保持 $\widetilde z_{\bar S}=z_{\bar S}$。`LSE` 是 `logsumexp`，不能改成逐类缩放或对完整词表重新设温度。中心化只消去 softmax 的公共平移自由度，不解释成统计 fANOVA 或因果分解。

执行约束如下：

1. CIR 阶段完全沿用已验证协议；N 继承其全部 LoRA、可训练 ID embedding 及模型状态，不重置有用领域适配，不加入 C/D 头。
2. N 保留原有可训练参数，再增加零初始化 $U$；原完整词表、ID+EOS 的 assistant-only causal CE 不变，原 AdamW、学习率、步数和数据顺序不变。不额外加类别 CE、重加权、负例或监督。
3. 仅首回答预测位置应用变换。训练位置由回答边界确定，不能使用 gold 类别来路由；推理时由生成位置确定。EOS 及其他位置的 forward 不应用该模块。
4. 训练和推理均在模型 forward 内使用同一变换，然后仍运行原完整词表 20-beam。不是事后 rerank、seen-mask、候选过滤或 112-logits 替代生成；所有非 ID 路径仍能竞争。
5. $U=0$ 时初始 logits 在实数算术中与继承模型一致；低精度实现须单独验证。`logsumexp` 与归一化应使用 FP32，不将代数恒等误称为 BF16 逐 bit 相等。

额外存储参数为 $112d$，有效自由度为 $111d$；不新增第二次 LLM 前向。额外计算仅为首回答位置的矩阵乘法和两次 112 维 `logsumexp`。实际速度、$d$ 和显存增量须由运行时报告确认，不把理论 FLOPs 当作测得的加速。

### 10.3 命题一：概率质量精确守恒

固定同一前缀、原参数和 $h$，记原模型输出 ID 集合的概率为 $\pi$。由

$$
\sum_{c\in S}e^{\widetilde z_c}=e^b=\sum_{c\in S}e^{z_c}
$$

可知完整词表 softmax 的分母不变。因此

$$
\widetilde p(S\mid h)=\pi,\quad
\widetilde p(v\mid h)=p(v\mid h)\ (v\notin S),\quad
\widetilde p(c\mid S,h)=q_c=\operatorname{softmax}(z_S+PUh)_c.
$$

这比“只修改 ID logits”更强：裸残差会改变分母，从而改变所有非 ID 词元概率，本结构不会。对真实 ID 标签 $y$，原首词元负对数似然严格分解为

$$
-\log\widetilde p(y\mid h)=-\log\pi-\log q_y,
\qquad \nabla_U\ell=(q-e_y)h^\top.
$$

这里 $P(q-e_y)=q-e_y$。原训练器对 ID、EOS 的 token 平均会乘上原有比例，不改变此分解，也不能据此把 ID 权重翻倍。

进一步，以 $a=\operatorname{logit}\pi$ 和类内 logits 为独立概率坐标，Fisher 的交叉块为零：ID 内的条件 score 均值为零，ID 外的类内 score 为零。因此“输出 ID 还是普通词”与“输出哪一个 ID”在这些坐标下正交。

**边界：**这不是共享 decoder/LoRA 参数的任务梯度正交。更新骨干仍会改变 $h,z,\pi$。命题仅保证新增 $U$ 本身在同一前缀下不扰动 ID 总质量；也不保证完整 beam 的严格格式率不变，因为不同 ID 后续生成 EOS 的概率可能不同。

### 10.4 命题二：相同有效更新范数下的严格一阶优势

只比较**固定表示的类别读出参数化**。将 LoRA 尺度吸收入矩阵，原类别修正为 $M=BA$，秩 $r=16$。在 $A,B$ 满秩处，其可实现的一阶变化属于

$$
\mathcal T=\{B\,dA+dB\,A\}.
$$

令 $G$ 为同一条件损失相对于完整类别矩阵的梯度，$P_B,P_A$ 分别投影到 $B$ 的列空间和 $A$ 的行空间。正交分解为

$$
G_\perp=(I-P_B)G(I-P_A),\qquad G_{\mathcal T}=G-G_\perp.
$$

证明：$G_\perp$ 与任意 $B\,dA+dB\,A$ 的 Frobenius 内积均为零；其余部分属于 $\mathcal T$。由 Cauchy-Schwarz，在同样的有效矩阵步长预算 $\|dM\|_F\le\rho$ 下，完整读出与最优 LoRA 切空间更新的最大一阶下降量分别为

$$
\rho\|G\|_F\quad\text{与}\quad\rho\|G_{\mathcal T}\|_F.
$$

只要 $G_\perp\ne0$，前者严格更大。这比较的是**相同有效矩阵更新大小**，不是通过放大学习率制造优势，甚至给予 LoRA 最优切空间方向而非仅原 AdamW。若损失梯度为 $L$-Lipschitz，Taylor 余项绝对值不超过 $L\rho^2/2$；当 $\rho<(\|G\|_F-\|G_{\mathcal T}\|_F)/L$ 时，完整梯度方向优于任何该预算内切空间步的有限步结论也成立。

**边界：**这是局部读出结论，不是联合 decoder 更新的下降率定理，也不是原固定学习率 AdamW 一定满足该步长条件的证明。最终矩阵秩至多 $r$，但 LoRA 一阶变化的秩可达 $2r$；不能混淆两者。将 CIR 与 N 两个最终 rank-$r$ 矩阵相减，也可达 $2r$，不能把阶段差值误称为 rank-$r$。

### 10.5 命题三：严格的总体风险差距，而不只是函数类包含

比较对象仍是固定表示上的读出。令 $W_0$ 为**语义 ID 初始化之后、LoRA 之前**的类别权重，不是 CIR 结束权重。假设：

1. 特征二阶矩 $\Sigma=\mathbb E[hh^\top]$ 正定；若不满秩，则限制在其有效支撑上。
2. N 的真实条件分布满足 $q^*(\cdot\mid h)=\operatorname{softmax}(W_0h+M_*h)$，其中 $\mathbf1^\top M_*=0$。
3. 对被比较矩阵与 $M_*$ 之间的线段，所有条件类别概率均至少为 $\varepsilon>0$。这是有界性假设，不是已经证实的 Amazon 性质。
4. 原读出仍限制 $\operatorname{rank}(M)\le r$；完整类别修正可以表示 $M_*$。若继承的 CIR 类别矩阵为 $M_C$，新残差取 $PU=M_*-PM_C$ 即可。

令 $\mathcal R$ 为总体条件交叉熵，$\sigma_j$ 为 $M_*\Sigma^{1/2}$ 的奇异值，则任意被比较的 rank-$r$ 读出满足

$$
\boxed{\mathcal R(M)-\mathcal R(M_*)
\ \ge\ \frac{\varepsilon}{2}\sum_{j>r}\sigma_j^2.}
$$

**证明：**条件风险差等于 $\mathbb E\,\mathrm{KL}(q^*\|q_M)$。`logsumexp` 的 Hessian 为 $\operatorname{diag}(q)-qq^\top$。对 $\mathbf1^\top v=0$，

$$
v^\top(\operatorname{diag}(q)-qq^\top)v
=\min_a\sum_c q_c(v_c-a)^2
\ge\varepsilon\min_a\sum_c(v_c-a)^2
=\varepsilon\|v\|_2^2.
$$

沿真实 logits 与预测 logits 之间积分，得到风险差至少为 $\frac\varepsilon2\mathbb E\|(M-M_*)h\|_2^2=\frac\varepsilon2\|(M-M_*)\Sigma^{1/2}\|_F^2$；最后应用 Eckart-Young 低秩逼近定理。公共行分量可先消去，不影响 softmax 或增大矩阵秩。只要 $r$ 之后存在非零奇异值，便有严格正差距，而完整读出在这些假设下可消除该逼近误差。

这给出了一个**可被数据否定的严格风险优势条件**。但 $M_*$ 未知，训练梯度的尾部谱不等于 $M_*$ 的尾部谱；不能拿一次 minibatch 的高秩梯度代替此假设的证据。有限样本的估计误差、实际优化误差和模型失配若超过该下界，实际优势仍可消失。尤其 $\varepsilon$ 可能很小，下界正数不等于达到 nDCG +0.005。

更重要的是，通用 LLM 可以通过更新 $h$ 改变所需读出秩。这个定理**不能证明完整可训练 G0 的全局风险下界**；相同 $h$ 是控制比较，而不是把 G0 偷换为冻结骨干后宣称击败了原模型。实际模型保留原骨干更新，收益取决于开放直接类别通路是否比通过骨干间接修正更容易学到、且不过拟合。

### 10.6 排名提升还需要什么

在一个标签对应一个正例的条件分布下，按真实 $q^*$ 降序排列同时最优于期望 Recall@K 与 nDCG@K。令 $\Delta_K=q^*_{(K)}-q^*_{(K+1)}>0$，若

$$
\|q-q^*\|_\infty<\Delta_K/2,
$$

则预测 top-K 集合与 Bayes 集合相同。要保证 nDCG 的顺序也相同，还需控制 top-K 内相邻概率间隔。若旧读出在正概率的一批样本上选错且新读出满足上述条件、其他样本不产生更大损失，才能推出严格的期望指标改善。

另一条一般上界是 $\mathrm{Regret}_{K}(q)\le\|q-q^*\|_1\le\sqrt{2\mathrm{KL}(q^*\|q)}$，其中位置折扣在 $[0,1]$ 内。证明是加入并减去 $q$ 对两种排序的期望得分，利用 $q$ 的排序最优性，再用 Pinsker 不等式。**上界减小不代表实际 regret 必然减小。**

探针给出反例：真实概率为 $(0.45,0.35,0.20)$；预测 $(0.60,0.21,0.19)$ 比 $(0.39,0.41,0.20)$ 的 KL 更大，却有更好的期望 Recall@1。这排除了“CE 更低所以 Recall 必然提高”的错误推导。

以上仍是条件类别排序结论。实际完整 beam 得分还包含 ID 后 EOS/其他 token 的概率和生成配置；即便首 ID 的 $q$ 更准确，也必须通过原始 20-beam 验证。训练分布与 112 类 macro 评测对应的目标分布若不一致，校准关系也不能直接搬用；本方案不通过重加权修补这个问题。

### 10.7 近邻与新颖性边界

| 来源 | 本轮实际核对范围 | 与当前候选的关系 |
|---|---|---|
| [LoRA](https://arxiv.org/abs/2106.09685) | 摘要及本地实际参数实现 | 低秩适配本身是既有技术 |
| [LoRA-GA](https://arxiv.org/pdf/2407.05000) | 原 PDF 方法 3.1-3.4、定理 3.1/3.2、算法 1 | 通过初始奇异子空间近似完整梯度；不是当前候选的完整类别自由度 |
| [LoRA-Pro](https://arxiv.org/pdf/2407.18242) | 原 PDF 2.2-2.3、定理 2.1-2.3 | 优化低秩因子的等效梯度；切空间论证不是本项目首创，不能重新命名为新理论 |
| [Decoupled Knowledge Distillation](https://arxiv.org/html/2203.08679v2) | 正文 3.1-3.3、附录 A.1 的概率/KL 分解 | 说明集合质量与条件分布的分解也不是新数学；这里只作近邻核对，不引入教师、蒸馏或重加权 |
| 仓库原 `<archived-job-id>` 与 `residual_classifier` | 实际训练与 loss 分支 | 已有全秩头；本候选的工程区别是将质量守恒读出纳入原完整词表 forward，而非独立类别评测 |

因此目前可辩护的是**具体任务结构适配及其可证伪机制**，不是“首次提出概率分解/全秩头/低秩梯度缺陷”。若目标还包含独立方法论文的新颖性，仅这些推导不够；不能为了满足“必须找到”而虚构已有工作空白。

### 10.8 最小验证路线与停止条件

不安排单独 G0 复现，不改任何旧冻结包，不覆盖失败结果。未来接入应使用新版本合同，并优先复用已有、经过身份校验的同一 CIR checkpoint，只训练新候选的 N 阶段，与现存同源 G0 的 N 结果比较。需要下载完整 CIR 权重并核验 SHA 后才能实际使用；目前仅有云端完整权重审计，不声称权重已在本地可用。

1. **实现门槛：**FP32/BF16 零初始化等价、ID 总质量/非 ID 概率、首回答位置、EOS 不应用、梯度检查点、梯度累积、20-beam 扩展与保存加载一致。不能只通过本文的 CPU 数学检查就提交正式作业。
2. **机制证据：**在原 N 更新过程中记录完整类别梯度的谱、相对 LoRA 切空间的 $\|G_\perp\|_F^2/\|G\|_F^2$ 及其跨原始批次稳定性。仅高秩不足以证明有用，还需观察原协议排名收益；不改变训练样本或额外制造前缀。
3. **效果门槛：**在看结果之前固定相对同源 G0 的 nDCG@20 至少 +0.005、Recall@20 不下降，并同时报告 @10、严格格式率、重复率。未达门槛则不给“显著有效”的结论，也不自动扩大秩/尺度搜索。
4. **仅在主候选通过后辨别机制：**同协议普通全秩残差检验质量守恒的作用；质量守恒但 rank-16 的残差检验完整类别更新的作用。这些是待授权/执行的控制，不是已经跑过的结果。
5. **统计边界：**现有 Dev 已反复使用，单 seed 达到门槛也只是探索性。独立显著性确认需要未被选择过程使用的评测与合理重复；不擅自构造、重划或重采样训练数据来获得它。

应当放弃或降级该候选的情形包括：真实所需读出修正近似 rank-16；有用信号不在 $h$ 中；新增方向主要拟合噪声；骨干原本就能有效补偿读出；格式/EOS 而非类别判别才是瓶颈；仅训练 CE 改善而原始排名门槛失败。原正式结果严格格式率已经为 1，故不能声称当前收益主要来自修复既有格式错误。

### 10.9 本轮已验证与未验证

已在 CPU Float64 执行下述探针：质量守恒、零初始化、公共平移不变、完整词表 CE 分解及 autograd 梯度、低秩逼近尾能量、总体风险下界的一个有限支撑例子、切空间正交、Fisher 交叉块、top-K 间隔充分条件，以及 CE/Recall 反例。有限支撑例子中尾能量为 1.26，所比较低秩矩阵的超额 CE 为 0.0885601922，下界为 0.0026096480；这些数字只验证构造，不是 Amazon 指标，也不是对所有低秩矩阵进行数值穷举。

当前结论：**找到一个具有严格、非平凡局部优势和条件性总体风险优势的具体结构候选；没有找到能无条件保证真实 Amazon 显著超越完整 G0 的证明。** 正式模型接入、真实机制测量和训练评测均未执行。本轮没有云端新作业；旧 G0 数值复现不再是当前前置任务。

### 10.10 可执行数学探针

<!-- conditional-readout-check:start -->
```python
import json
import torch

torch.set_num_threads(1)
torch.manual_seed(20260927)
dtype = torch.float64
batch_size, vocabulary_size, class_count, feature_count = 5, 19, 7, 9
label_ids = torch.tensor([1, 3, 5, 7, 9, 11, 13])
outside_ids = torch.tensor([token for token in range(vocabulary_size) if token not in label_ids])
base_logits = torch.randn(batch_size, vocabulary_size, dtype=dtype)
features = torch.randn(batch_size, feature_count, dtype=dtype)
weight = torch.randn(class_count, feature_count, dtype=dtype, requires_grad=True)
targets = torch.tensor([0, 2, 4, 6, 1])


def conditional_logits(original, residual):
	original_labels = original[:, label_ids]
	proposed_labels = original_labels + residual
	corrected = proposed_labels + torch.logsumexp(original_labels, dim=-1, keepdim=True)
	corrected = corrected - torch.logsumexp(proposed_labels, dim=-1, keepdim=True)
	result = original.clone()
	result[:, label_ids] = corrected
	return result


residual = features @ weight.T
updated = conditional_logits(base_logits, residual)
original_probability = base_logits.softmax(-1)
updated_probability = updated.softmax(-1)
torch.testing.assert_close(updated_probability[:, outside_ids], original_probability[:, outside_ids])
torch.testing.assert_close(updated_probability[:, label_ids].sum(-1), original_probability[:, label_ids].sum(-1))
torch.testing.assert_close(conditional_logits(base_logits, torch.zeros_like(residual)), base_logits)
torch.testing.assert_close(
	conditional_logits(base_logits, residual + 13.0), updated,
)
full_loss = torch.nn.functional.cross_entropy(updated, label_ids[targets])
conditional_loss = torch.nn.functional.cross_entropy(base_logits[:, label_ids] + residual, targets)
format_loss = -original_probability[:, label_ids].sum(-1).log().mean()
torch.testing.assert_close(full_loss, conditional_loss + format_loss)
actual_gradient = torch.autograd.grad(full_loss, weight)[0]
conditional_probability = (base_logits[:, label_ids] + residual).softmax(-1)
one_hot = torch.nn.functional.one_hot(targets, class_count).to(dtype)
expected_gradient = (conditional_probability - one_hot).T @ features / batch_size
torch.testing.assert_close(actual_gradient, expected_gradient)

center = torch.eye(class_count, dtype=dtype) - torch.ones(class_count, class_count, dtype=dtype) / class_count
orthonormal = torch.linalg.eigh(center).eigenvectors[:, 1:]
singular_values = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=dtype)
target_matrix = orthonormal @ torch.diag(singular_values)
comparison_rank = 2
low_rank_matrix = orthonormal[:, :comparison_rank] @ torch.diag(singular_values[:comparison_rank])
low_rank_matrix = torch.nn.functional.pad(low_rank_matrix, (0, class_count - 1 - comparison_rank))
population_features = torch.cat((torch.eye(class_count - 1, dtype=dtype), -torch.eye(class_count - 1, dtype=dtype)))
population_features = population_features * (class_count - 1) ** 0.5
torch.testing.assert_close(population_features.T @ population_features / len(population_features), torch.eye(class_count - 1, dtype=dtype))
target_scores = population_features @ target_matrix.T
low_rank_scores = population_features @ low_rank_matrix.T
truth = target_scores.softmax(-1)
excess_risk = (truth * (target_scores.log_softmax(-1) - low_rank_scores.log_softmax(-1))).sum(-1).mean()
logit_bound = torch.maximum(target_scores.abs().max(), low_rank_scores.abs().max())
probability_floor = torch.exp(-2 * logit_bound) / class_count
tail_energy = singular_values[comparison_rank:].square().sum()
risk_lower_bound = probability_floor * tail_energy / 2
assert excess_risk > risk_lower_bound > 0
torch.testing.assert_close((target_matrix - low_rank_matrix).square().sum(), tail_energy)

truth_example = torch.tensor([0.45, 0.35, 0.20], dtype=dtype)
better_ranking = torch.tensor([0.60, 0.21, 0.19], dtype=dtype)
better_logloss = torch.tensor([0.39, 0.41, 0.20], dtype=dtype)
first_kl = (truth_example * (truth_example.log() - better_ranking.log())).sum()
second_kl = (truth_example * (truth_example.log() - better_logloss.log())).sum()
assert second_kl < first_kl
assert truth_example[better_logloss.argmax()] < truth_example[better_ranking.argmax()]

conditional_gradient = center @ torch.randn(class_count, feature_count, dtype=dtype)
column_basis = orthonormal[:, :comparison_rank]
row_basis = torch.linalg.qr(torch.randn(feature_count, comparison_rank, dtype=dtype)).Q
column_projector = column_basis @ column_basis.T
row_projector = row_basis @ row_basis.T
normal_gradient = (torch.eye(class_count, dtype=dtype) - column_projector) @ conditional_gradient
normal_gradient = normal_gradient @ (torch.eye(feature_count, dtype=dtype) - row_projector)
tangent_gradient = conditional_gradient - normal_gradient
torch.testing.assert_close((normal_gradient * tangent_gradient).sum(), torch.zeros((), dtype=dtype), atol=1e-12, rtol=0)
assert normal_gradient.norm() > 0
assert conditional_gradient.norm() > tangent_gradient.norm()
column_change = torch.randn_like(column_basis)
row_change = torch.randn_like(row_basis.T)
tangent_change = column_change @ row_basis.T + column_basis @ row_change
torch.testing.assert_close((normal_gradient * tangent_change).sum(), torch.zeros((), dtype=dtype), atol=1e-12, rtol=0)

joint_coordinates = torch.randn(class_count + 1, dtype=dtype, requires_grad=True)


def <archived-job-id>(coordinates):
	return torch.cat((
		torch.nn.functional.logsigmoid(coordinates[:1]) + coordinates[1:].log_softmax(-1),
		torch.nn.functional.logsigmoid(-coordinates[:1]),
	))


score_jacobian = torch.autograd.functional.jacobian(<archived-job-id>, joint_coordinates)
joint_probability = <archived-job-id>(joint_coordinates).exp()
fisher = score_jacobian.T @ (joint_probability[:, None] * score_jacobian)
torch.testing.assert_close(fisher[0, 1:], torch.zeros(class_count, dtype=dtype), atol=1e-12, rtol=0)

ranking_truth = torch.softmax(torch.tensor([2.0, 1.5, 0.7, -0.1, -0.6, -1.0, -1.5], dtype=dtype), -1)
cutoff = 3
ranking_prediction = ranking_truth * 0.99 + torch.ones_like(ranking_truth) * (0.01 / class_count)
sorted_truth = ranking_truth.sort(descending=True).values
ranking_margin = sorted_truth[cutoff - 1] - sorted_truth[cutoff]
assert (ranking_truth - ranking_prediction).abs().max() < ranking_margin / 2
assert set(ranking_truth.topk(cutoff).indices.tolist()) == set(ranking_prediction.topk(cutoff).indices.tolist())

print(json.dumps({
	"probability_mass_and_gradient_checks": "passed",
	"rank_tail_squared": tail_energy.item(),
	"toy_excess_cross_entropy": excess_risk.item(),
	"toy_lower_bound": risk_lower_bound.item(),
	"missing_gradient_squared": normal_gradient.square().sum().item(),
	"fisher_cross_block_and_topk_margin_checks": "passed",
	"lower_cross_entropy_can_have_worse_recall": True,
	"<archived-job-id>": False,
}, indent=2))
```
<!-- conditional-readout-check:end -->