# 角色分解兴趣扩展方法：定义及其与CIR+N SFT的完整差异

> 版本：2026-09-27（北京时间），`role-projection-v2`。
> 本文方法仍为v2；前轮执行快照为`launch-v3`，本轮旧G0对齐快照为`g0-aligned-v1/launch`。v1烟测发现三线性梯度退化，v2加入固定全局RMS标准化；执行快照v3仅修复参数报告复写。G0对齐只恢复运行协议，不改变下面的方法定义。旧快照保留。任务是预测下一新兴趣，CIR只是辅助预热。
> 文中的数学性质、实现检查和推荐效果是三种不同证据；两轮验证均未达到效果门槛，本轮G0对齐后projected低于两项对照，不能把本方法写成已验证有效或已证明具有文献新颖性的方法。

## 1. 一句话定义

**保留原CIR+N SFT的完整语言模型、输入、监督和训练顺序，仅增加一个作用于首个Interest ID预测位置的低秩关系残差头：CIR只训练角色对称的相容分支C；N继承C，并启用独立的零和角色偏差分支D。**

核心假设是：回顾性补全与未来兴趣进入可以共享一部分相容关系，但不应强制共享所有“谁作为预测目标”的关系。这个假设通过结构约束实现，不通过新的loss、教师、数据增广或候选过滤实现。

这不是“LLM原来不会建模兴趣交互”的主张。原完整序列LLM已经可以利用标题、时间顺序和多兴趣交互；新头只增加一种显式、可检验的归纳偏置。

## 2. 比较对象：原CIR+N SFT是什么

本文基线指本项目的**全文CIR预热后进行N SFT**，不是N-only，不是独立兴趣压缩，也不是历史上带其他关系头的条件。

### 2.1 两个任务

| 项目 | CIR辅助任务 | N主任务 |
|---|---|---|
| 输入 | 从一段原历史中删除目标类别的全部行为后，保留的有序B+I历史 | 预测时刻之前的有序B+I历史 |
| 预测目标 | 被删除的历史类别；训练使用原文件的合法正类集合 | 下一次进入的新兴趣类别 |
| 时间语义 | 回顾性补全，剩余行为可能发生在被删除行为之后 | 面向未来，不能利用未来行为 |
| 与“未见”的关系 | 目标类别从可见输入中整类删除 | 新类别依据完整过去判定；模型仍只看到冻结的可见窗口 |
| 作用 | 辅助预热 | 最终推荐任务 |

B+I表示按时间排列的`[类别] 商品标题`。完整过去中的seen集合只用于原任务标注，不新增为模型输入。未来商品标题、真实目标、候选列表和用户ID均不作为新增输入。

原可见历史窗口、tokenizer、prompt、chat template和8192长度预算不变。不能把“相对完整过去是新类别”误写成只要不在可见窗口中就算新类别。

### 2.2 原模型和可训练参数

骨干为Qwen3.5-9B，固定revision为`c202236235762e1c871ad0ccb60c8ee5ba337b9a`。令其首ID位置的原始输出logit为$z_\theta(H,v)$，$v$遍历完整语言模型词表$\mathcal V$。

这里$\theta$表示原训练路径上的可训练部分，包括decoder/lm_head的LoRA和原语义Interest ID输入embedding行；不是对9B基础权重做全参数微调。原基础权重保持冻结。

112个输出类别各对应一个新增单token Interest ID。原有类别名到ID映射及语义初始化完整保留，新方法不重新设计ID编码。PEFT的可训练token行沿用原替换行语义，不另行解释为一个加性embedding修正。

### 2.3 原目标函数

令$\mathcal C$为112个输出类别，$\tau(c)$为类别$c$的Interest ID，$\mathcal P(H)\subseteq\mathcal C$为原CIR样本提供的非空正类集合。基线CIR在首ID位置使用：

$$
\mathcal L_{\mathrm{CIR}}
=-\frac{1}{|\mathcal P(H)|}\sum_{c\in\mathcal P(H)}
\log\frac{\exp z_\theta(H,\tau(c))}
{\sum_{j\in\mathcal C}\exp z_\theta(H,\tau(j))}.
$$

这是**112类范围内、正类均匀分配概率质量的交叉熵**。不是只对一个被删类别做CE，也不是$-\log\sum_{c\in\mathcal P}p(c)$形式的集合概率损失。CIR不使用后续EOS位置来定义这个辅助loss。

N使用原assistant-only causal LM交叉熵。设$\mathcal A$为原tokenizer标出的assistant监督位置，$y_t$为该位置真实token：

$$
\mathcal L_N
=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}
\log\frac{\exp z_{\theta,t}(y_t)}{\sum_{v\in\mathcal V}\exp z_{\theta,t}(v)}.
$$

该式表达原token级平均目标；跨microbatch的实际归一化与梯度累积沿用原Trainer。N保留Interest ID和EOS等原assistant监督位置，不改成只训练112类的分类器。

### 2.4 原两阶段过程

1. CIR：83,888条固定均衡数据，3轮，7,866个optimizer updates。
2. N：继承CIR最终LoRA与语义ID参数，112,000条固定数据，1轮，3,500个updates。
3. 阶段切换重建optimizer和scheduler；继承模型参数，不继承上一阶段优化器动量。
4. 推理保留原完整词表、确定性20-beam生成，不做seen-mask、候选过滤或rerank。

## 3. 新方法的变量与信息边界

| 符号 | 定义 | 信息来源 |
|---|---|---|
| $H$ | 当前任务可见的完整有序B+I输入 | 原CIR或N样本 |
| $S(H)$ | 可见历史中去重后的类别集合 | 仅`historical_interests` |
| $K$ | $|S(H)|$ | 可见类别数量，不是行为条数 |
| $\mathcal C$ | 固定112个输出类别 | 原映射 |
| $d$ | 语言模型隐藏状态及输入embedding维度 | 骨干结构 |
| $R$ | 低秩宽度，固定64 | 预先设定，不按结果调参 |
| $h_H$ | decoder最终norm之后、用于预测首ID的隐藏状态 | 原LM同一次前向 |
| $e_x$ | 类别$x$的taxonomy原生token均值向量，经共同固定RMS缩放 | 原类别文本与基础模型embedding |
| $\phi_C,\phi_D$ | 两套独立分支参数 | 新增小型关系头 |

**$h_H$包含当前任务prompt的影响。** CIR与N prompt不同，即使可见购买列表相同，两者的$h_H$也不必相同。“固定$H$下交换角色”的证明必须同时固定prompt和$h_H$，不能跨两种真实prompt直接套用对称性。

taxonomy向量定义为：

$$
e_x^{\mathrm{raw}}=\frac{1}{|\operatorname{tok}(x)|}
\sum_{j\in\operatorname{tok}(x)} E_j.
$$

类别文本使用原生token，不包含新增Interest ID token。$e_x^{\mathrm{raw}}$取值后detach，在关系头中不训练；输出类别原始向量作为buffer保存，可见历史类别原始向量按需缓存。新头不调用外部知识源或教师模型，但会复用骨干已有的词向量信息。

v2采用同一个固定标量规范全部角色的输入尺度：

$$
s=\max\left(\sqrt{\frac{1}{112d}\sum_{j\in\mathcal C}\|e_j^{\mathrm{raw}}\|_2^2},10^{-12}\right),
\qquad e_x=e_x^{\mathrm{raw}}/s.
$$

该标量仅取决于固定输出taxonomy；映射外历史也除以同一个$s$。它保留类别间相对向量大小，没有逐类别L2归一化，不训练额外温度，也不修改LM原语义ID初始化。v1没有此步骤，云端发现部分矩阵只出现约1e-12的微小变化，方向target完全不变；因此v1未通过更新审计，不能作为有效方法结果。后文公式均采用v2的$e_x$。

**重要实现修正：$S(H)$不一定是$\mathcal C$的子集。** 数据核对发现两个训练阶段的历史各覆盖136个类别，其中25个不属于112类输出映射。新方法保留这些类别，为它们计算同样的taxonomy向量，不丢弃对应行为，不扩充输出类别。

原LM仍读取所有重复行为、标题和顺序。只有新增头的类别对聚合按去重集合计算；这不等于对整个输入去重或用类别集合替代全文。

## 4. 精确数学定义

### 4.1 两套上下文条件三线性算子

对每个分支$k\in\{C,D\}$，设三个无bias矩阵：

$$
\phi_k=(W_u^k,W_v^k,W_q^k),\qquad
W_u^k,W_v^k,W_q^k\in\mathbb R^{R\times d}.
$$

每次前向，以112个输出类别为统一中心化基准：

$$
\mu_u^k=\frac{1}{112}\sum_{j\in\mathcal C}W_u^k e_j,
\qquad
\mu_v^k=\frac{1}{112}\sum_{j\in\mathcal C}W_v^k e_j,
$$

$$
u_x^k=W_u^k e_x-\mu_u^k,
\qquad
v_x^k=W_v^k e_x-\mu_v^k,
\qquad
\lambda_H^k=W_q^k\operatorname{normalize}(h_H).
$$

`normalize`为L2归一化；taxonomy向量只做第3节的共同RMS缩放，不额外做逐类别L2归一化。中心化均值会随矩阵训练变化，不是只在初始化时减一次均值。映射内外的$x$均使用同一均值。

原始算子为：

$$
T_H^k(a,b;c)=\sum_{r=1}^{R}\lambda_{H,r}^k\,u_{a,r}^k u_{b,r}^k v_{c,r}^k.
$$

$a,b$是两个不同可见源类别，$c$是待打分的输出类别。两源角色天然对称，即$T_H^k(a,b;c)=T_H^k(b,a;c)$。$\lambda_H^k$可正可负，不是attention概率、采样权重或额外监督。

同一$x$无论作为源还是目标，始终先由同一个$e_x$表示，再分别经过$W_u^k$或$W_v^k$。不能把源换成上下文化商品状态、目标保留taxonomy向量，然后仍沿用下面的交换性质。

### 4.2 相容分支C

定义三种目标角色的平均投影：

$$
(\Pi T)_H(a,b,c)
=\frac{T_H(a,b;c)+T_H(b,c;a)+T_H(c,a;b)}{3}.
$$

相容分支为：

$$
C_H(a,b,c)=(\Pi T^C)_H(a,b,c).
$$

在固定$h_H$下，$C_H$对$a,b,c$的全部排列对称。它鼓励新增的这一部分关系表达“这三个类别是否相容”，而非偏好其中一个类别作为目标。

这只是分支的归纳偏置。CIR真实条件概率不必对称，基础LM的预测也不受这种对称性限制。

### 4.3 角色偏差分支D

使用**另一套独立参数**定义：

$$
D_H(a,b;c)=T_H^D(a,b;c)-(\Pi T^D)_H(a,b,c).
$$

由此得到：

$$
D_H(a,b;c)+D_H(b,c;a)+D_H(c,a;b)=0.
$$

$D$刻画同一三类别关系中，指定某个类别作为目标时相对三角色平均的偏差。它可以为一个目标角色加分，同时为其他角色减分。

这是**三目标角色零和**，不是二元关系中的简单反对称条件；也不代表每个真实用户的预测概率和为零。角色偏差不是时间因果效应。

两分支不能混写成同一张量的机械拆分再加回。实际为：

$$
C+D=\Pi T^C+(I-\Pi)T^D,
$$

其中$T^C$和$T^D$参数独立、训练经历不同。若强制使用同一个$T$，则$\Pi T+(I-\Pi)T=T$，会丢失本方法所要检验的独立参数共享边界。

### 4.4 可见类别对聚合

设$Z_H=\binom{K}{2}$。当$K\geq2$时：

$$
r_C(H,c)=\frac{1}{Z_H}\sum_{\{a,b\}\subset S(H),\,a\ne b}C_H(a,b,c),
$$

$$
r_D(H,c)=\frac{1}{Z_H}\sum_{\{a,b\}\subset S(H),\,a\ne b}D_H(a,b;c).
$$

上式按无序二元子集计数，每对只算一次。$K<2$时两个残差均为零。

所有不同类别对等权平均，没有逐对attention、热门类别加权或可学习门控。对所有$c\in\mathcal C$均计算残差，包括已在历史中出现的类别；不要求$c\notin\{a,b\}$，不对已见输出做硬排除。

### 4.5 注入原完整词表logit

令$p$为首个assistant监督token的前一位置，即预测首个Interest ID的logit位置。新logit为：

$$
\widetilde z_{t,v}=z_{\theta,t,v}
+\mathbf 1[t=p]\sum_{c\in\mathcal C}\mathbf 1[v=\tau(c)]r_s(H,c),
$$

$$
r_s(H,c)=
\begin{cases}
r_C(H,c), & s=\mathrm{CIR},\\
r_C(H,c)+r_D(H,c), & s=N.
\end{cases}
$$

残差系数固定为1，无额外温度、混合系数或正则强度。后续EOS等位置不直接注入残差，其他词表位置不直接修改。训练更新仍会间接改变未来的基础LM输出，不能把“不直接加残差”理解为这些输出在训练后恒定不变。

N依然在完整词表上归一化。显式打分112个Interest ID并不意味着将生成约束为112类：其他token仍能竞争，非法输出仍按原评测规则保留和判分。

### 4.6 一个纯示意例子

令$a$为磨豆机、$b$为水壶、$c$为滤杯，固定一条可见历史及其$h_H$。

| 分支 | 三种目标角色的原始分数 | 投影后的三个分数 |
|---|---|---|
| C的独立原始算子 | $(1.2,0.6,1.2)$ | $(1.0,1.0,1.0)$ |
| D的独立原始算子 | $(1.4,0.5,0.2)$ | $(0.7,-0.2,-0.5)$ |

若可见历史只含$a,b$两种不同类别，则该对对滤杯的N残差为$1.0+0.7=1.7$；原logit若为2.0，注入后为3.7。真实有多对时再按$Z_H$平均。

这些数值不是训练样本、人工规则或实测输出。交换目标角色只是输出头内部计算，并没有把滤杯作为真实未来答案送入LM历史，也没有生成逆向正标签。其他两个角色分数不等于换一条真实历史后模型的最终预测分数。

## 5. 训练过程：与基线到底改了什么

### 5.1 参数启用与继承

| 参数或状态 | 基线CIR | 新方法CIR | 基线N | 新方法N |
|---|---|---|---|---|
| 原基础模型权重 | 冻结 | 冻结 | 冻结 | 冻结 |
| 原decoder/lm_head LoRA | 训练 | 训练 | 继承CIR后训练 | 继承CIR后训练 |
| 原语义ID可训练embedding行 | 训练 | 训练 | 继承后训练 | 继承后训练 |
| $\phi_C$ | 不存在 | 训练 | 不存在 | 完整继承并继续训练 |
| $\phi_D$ | 不存在 | 已初始化但冻结、不参与前向 | 不存在 | 载入其未训练状态并启用训练 |
| taxonomy向量$e_x$ | 无独立关系头副本 | 固定，不反传更新 | 无独立关系头副本 | 固定，不反传更新 |
| optimizer/scheduler | 新建 | 新建 | 阶段切换重建 | 阶段切换重建 |

代码中C、D两个分支在CIR开始时都已经分配参数。所谓“N阶段新增D”，准确含义是**启用原来未使用、未训练的D**，不是CIR结束后根据结果重新选择一个D初始化。

每个分支初始$W_q=0$，$W_u,W_v$为非零随机初始化，因此初始残差为零。没有再叠加一个零初始化门控，避免双零使整个新增分支无法启动。

在非退化样本上，首次反传可以更新$W_q$；随后$W_u,W_v$及经过$h_H$的梯度路径可以发挥作用。$K<2$等退化情况不保证新增头有非零梯度。初始化过程使用隔离的CPU RNG，不重置骨干CPU/CUDA随机状态。

### 5.2 Loss形式完全不变，输入的logit改变

两阶段使用第2.3节原loss，只将$z$替换为$\widetilde z$：

$$
\mathcal L^{\mathrm{new}}_{\mathrm{CIR}}=\mathcal L_{\mathrm{CIR}}(\widetilde z_C),
\qquad
\mathcal L^{\mathrm{new}}_N=\mathcal L_N(\widetilde z_{C+D}).
$$

没有BPR、对比学习、角色交换正则、知识蒸馏或额外辅助标签。CIR不反传到D参数，但会更新原LoRA。因此不能声称整个模型彻底隔离了CIR偏置，或两个任务的梯度正交。

新增残差会改变训练后的类别概率与梯度，并通过$\lambda_H$对原LoRA产生额外梯度路径。本方法是端到端联合训练的残差结构，不是训练后加一个固定reranker。

### 5.3 与基线相比的伪代码

```text
共同准备：原B+I renderer、原Interest ID映射、原LoRA与原语义ID初始化

基线：
  CIR: 原全文LM -> 原首ID类别logit -> 原CIR set CE
  继承LoRA与语义ID参数，重建optimizer/scheduler
  N:   原全文LM -> 原完整assistant logit -> 原causal LM CE

新方法：
  初始化独立C/D分支；各自context投影为0；冻结D
  CIR: 原全文LM得到首ID隐藏状态和logit
       仅根据可见类别集合计算r_C，加到首ID的112类logit
       原CIR set CE，更新原可训练参数和C
  继承LoRA、语义ID和全部C/D参数；启用D；重建optimizer/scheduler
  N:   原全文LM得到首ID隐藏状态和logit
       计算r_C+r_D，加到首ID的112类logit
       原完整assistant-only causal CE，更新原可训练参数和C/D
```

“一次前向”指每个原LM前向不新增第二次LM编码。训练仍有原反向传播，beam生成仍需逐token解码；不是整个20-beam过程只调用一次LM。

## 6. 全部差异与不变项对照

| 维度 | 原全文CIR+N SFT | 当前角色分解方法 | 性质 |
|---|---|---|---|
| 最终任务 | 下一新兴趣 | 相同 | 不变 |
| CIR定位 | 回顾补全辅助预热 | 相同 | 不变 |
| 数据与顺序 | 冻结CIR/N文件及原行顺序 | 相同，无新增/替换/重加权样本 | 不变 |
| 类别空间 | 112个Interest ID输出 | 相同 | 不变 |
| 原ID语义初始化 | 类别原生token均值初始化 | 相同 | 不变 |
| prompt与模板 | 原CIR及N各自模板 | 逐任务相同，不新增角色描述 | 不变 |
| LM可见输入 | 原有序完整B+I窗口 | 相同，不独立压缩兴趣 | 不变 |
| 结构化旁路输入 | 无新增头的类别集合接口 | 给头传可见类别集合与首ID位置，信息已在原输入中 | 新增接口，不新增观测信息 |
| 类别文本用途 | 原ID初始化和原B+I输入 | 额外用于关系头固定taxonomy向量 | 新增使用方式 |
| 关系头数值尺度 | 无独立关系头 | 所有角色共同除以固定输出taxonomy全局RMS | v2数值修复；两种交互对照一致 |
| 关系表示 | 由LM隐式建模 | 增加两个独立上下文条件三线性算子 | 新增结构 |
| 参数共享约束 | 不显式区分相容和目标角色 | C对称、D三角色零和；CIR只启用C | 核心方法差异 |
| 类别对聚合 | 无该显式聚合 | 所有可见不同类别对等权平均 | 新增结构 |
| 标题、次数、顺序 | 原LM直接利用 | 原LM照常利用；新增头由$h_H$条件化，类别对成员不按次数重复 | 主路径不变 |
| 新增logit位置 | 无 | 仅首ID的112类logit | 新增残差 |
| 已见类别 | 无硬过滤 | 仍无硬过滤，亦参与残差打分 | 不变 |
| 阶段loss | CIR set CE；N完整assistant CE | 公式不变，使用新logit | 监督不变，梯度改变 |
| 两阶段继承 | LoRA及语义ID | 再增加C/D头权重的继承 | 新增状态 |
| 新增参数量 | 0 | 总计$6dR$；CIR可训练$3dR$，N可训练$6dR$ | 增加参数 |
| 新增参数初始化 | 无 | 每分支context零、source/target非零 | 新增机制 |
| 新增头计算精度 | 无 | 显式关闭该头的autocast，以FP32计算投影和聚合 | 数值实现差异 |
| 词表logit缓存 | 原dtype及原实现 | 注入时对logit作FP32副本；有额外显存/带宽开销 | 工程开销，不是免费操作 |
| optimizer类型与超参 | 原配置 | 原配置，新增头也由原optimizer优化 | 参数集合改变 |
| scheduler和动量继承 | 阶段切换重建 | 相同 | 不变 |
| 推理 | 原全词表20-beam | 相同生成器，仅首ID位置注入；后续生成位置不加残差，保留原缓存路径 | 解码规则不变 |
| 权重保存 | 原PEFT adapter/tokenizer | 额外保存关系头权重和严格元数据 | 新增产物 |
| 参数报告 | 原trainable_parameters.json | 保留原报告，另存包含新头的role_trainable_parameters.json | 执行快照v3避免复写原报告 |
| 外部知识与教师 | 不要求 | 不新增教师、检索或蒸馏 | 不变 |

### 6.1 完全相同的训练与评测超参数

| 项目 | 本轮三条件共同值 |
|---|---|
| 模型、revision、seed | Qwen3.5-9B；第2.2节revision；20260825 |
| 正式CIR / N | 83,888条x3轮 / 112,000条x1轮 |
| LoRA | r16、alpha32、dropout0.05，沿用原目标模块，排除vision LoRA |
| 优化 | fused AdamW、LR1e-4、cosine、warmup0.03、maxgrad1 |
| 精度与有效batch | 原LM BF16、有效batch32 |
| 输入预算 | 原窗口，最大8192，不静默截断 |
| 主评测 | 原Dev2000、112类macro |
| 排名生成 | full-vocabulary direct20-beam、确定性、batch1、maxnew32 |
| 指标 | Macro Recall@10、Recall@20、nDCG@10、nDCG@20 |

### 6.2 历史G0与本轮匹配baseline不能混为一谈

历史G0使用4卡、micro1、accumulation8、开启gradient checkpointing。9月23日首轮`baseline`、`unconstrained`、`projected`统一为8卡、micro1、accumulation4、关闭gradient checkpointing，有效batch和数据轮数不变。

这是首轮三组共同的执行布局变化，不是角色分解独有的算法变化，也不保证与历史G0逐位相同。首轮效果判断以该轮匹配baseline为主要对照，历史G0仅提供背景；首轮nDCG@20增益0.003493未达到原0.005门槛，结论保留。

9月26日按用户要求重新提交三组[旧G0对齐验证](design.md)：恢复4卡/micro1/acc8、非重入GC开启，以及原Dev2000单进程、四GPU可见的balanced评测，原文件SHA与顺序不变。旧G0实际代码、合同与60包版本均作为新合同审计来源；方法、loss和学习预算不变。02:40:20三组云端环境及CIR烟测runtime均通过对齐检查，完整9B烟测、正式训练和评测仍未验收。旧镜像digest未记录，不能承诺逐bit一致或预先保证Recall@10回到0.25。

新增本地检查验证旧trainer与RoleTrainer baseline在GA8/GC开启下的一步参数更新相同，以及两种头、两阶段在GC开关时的loss/梯度一致；实际冻结包34项测试通过。新实验状态和完整结果表见[实验日志](log.md)。进一步增强须有可检验依据并独立记录，不能把布局恢复或盲目放大残差算作已验证的方法提升。

9月27日北京时间的进度证据（私有原始证据，不随公开仓库分发）已确认三组正式CIR 7,866步完成，正式N正在训练，两阶段实际合同/runtime及adapter继承身份与预期一致。烟测执行完成但严格ID健康仍未通过；正式Recall/nDCG尚未产生。本次只恢复本地监控和核验已有产物，没有修改方法或运行中的训练。

9月27日09:09北京时间最终独立验收（私有原始证据，不随公开仓库分发）已完成：三组完整训练、Dev2000及严格ID健康通过，projected的nDCG@20为0.160588，低于baseline的0.161815与unconstrained的0.161035；原效果门槛三项全部失败。新baseline的Recall@10为0.245215，仍比旧G0低0.006040，因此只能说可核验协议已对齐，不能说旧数值已完全复现；具体差异来源未确定。原始排名、合同及8份关系头经本地复核，完整LoRA张量仅有云端审计，边界明确保留。

完整训练占约91%墙钟，baseline纯训练比旧G0同协议少约4.3%，没有异常停滞证据。本轮未为加速修改明确要求的4卡、GC开启和单进程评测布局；本地必需产物下载已通过小块并发在35秒内完成。负结果不支持直接放大残差，不追加盲目调参或把计算布局变化解释成方法收益。

## 7. 精确快速实现、参数量与代价

每个分支使用自身的$u,v,\lambda$，令：

$$
A=\sum_{a\in S(H)}u_a,\quad B=\sum_{a\in S(H)}v_a,\quad
Q=\frac{A\odot A-\sum_a u_a\odot u_a}{2},\quad
M=A\odot B-\sum_a u_a\odot v_a.
$$

定义：

$$
\operatorname{raw}_c=\langle\lambda,v_c\odot Q\rangle,
\qquad
\operatorname{exchanged}_c=\langle\lambda,u_c\odot M\rangle.
$$

于是：

$$
r_C(H,c)=\frac{\operatorname{raw}^C_c+\operatorname{exchanged}^C_c}{3Z_H},
\qquad
r_D(H,c)=\frac{2\operatorname{raw}^D_c-\operatorname{exchanged}^D_c}{3Z_H}.
$$

这是显式类别对枚举的精确等价式，不是近似或随机采样。它避免物化$K\times K\times112$张量，使用已有FM式求和技巧；这种加速恒等式不作为新贡献。

- 参数：两套三个无bias矩阵，总计$6dR$。若$d=4096,R=64$，则为1,572,864个新增参数；单个启用分支为786,432。实际值以`ROLE_RUNTIME.json`为准。
- 固定输出taxonomy buffer有$112d$个数，不计入可训练参数；另有按需历史taxonomy缓存。
- 投影后的聚合复杂度为$O((K+112)R)$；当前实现还要计算输出及可见历史类别的投影，成本约$O((K+112)dR)$，不能从总开销中省略。
- 两分支均启用时重复各自投影，另有类别缓存、mask、FP32 logit复制等成本。首个beam扩展前向可能按beam批次计算头，不应将单样本公式的成本直接当成完整20-beam成本。
- 新方法不新增9B编码器前向，但总耗时仍由原训练、解码、排队和IO等共同决定，不能据“小型头”推断整轮必能在8小时内完成。

## 8. 数学保证与不能声称的结论

| 可以由定义或检查支持 | 不能由此推出 |
|---|---|
| 固定上下文下，C对三个类别角色对称 | CIR真实条件概率对称；不同真实历史的预测分数相同 |
| D在三种目标角色上零和 | D是因果方向、真实心理兴趣或可识别的用户状态 |
| $\Pi^2=\Pi$，C和D落在不同角色子空间 | 两个任务的参数梯度正交或完全没有负迁移 |
| 固定$H$、完整均匀$\mathcal C^3$域上，两角色子空间内积正交 | 在真实相关类别分布、实际历史子集、监督样本加权下仍正交 |
| 在112类基准轴上中心化；每条历史的112类残差均值为零 | 真实用户分布上的纯高阶协同；全词表中Interest ID总概率质量不变 |
| 头使用全部可见类别，不查询gold来选择某个源对 | 从有限训练样本唯一识别了每条类别关系 |
| K不足2时新增残差为零 | 全模型只能依靠双兴趣组合完成预测 |
| 单元测试验证了前向、梯度、生成和恢复 | 已获得9B推荐收益、跨域泛化或统计显著性 |

即使残差均值为零，softmax的非线性仍可能改变112类整体相对于其他词表token的概率质量。中心化是参数化约束，不是完整的概率校准保证。

“CIR删除监督可能无法唯一确定下一新兴趣”的反例可作为另一个分析问题，但不是本方法运行所需的假设或附加loss，也不能证明C/D分解最优。本文不把该反例算成已成立的独立理论创新。

## 9. 怎样判断新增结构有价值

| 条件 | 新增头 | 主要排除的解释 |
|---|---|---|
| `baseline` | 无 | 本轮匹配的原CIR+N SFT |
| `unconstrained` | CIR使用$T^C$，N使用$T^C+T^D$，均按源对平均 | 仅因为多了参数或普通高阶交互而改善 |
| `projected` | CIR使用$\Pi T^C$，N使用$\Pi T^C+(I-\Pi)T^D$ | 角色约束和阶段共享方式是否有效 |

普通交互对照保留相同六个矩阵、相同初始化、taxonomy信息、中心化和分支启用时机。它与主方法的自由参数数量相同，但计算量和有效函数空间不完全相同，不称严格同容量。

首轮探索性门槛：`projected`相对匹配`baseline`的macro nDCG@20至少增加0.005，Recall@20不降，并且nDCG@20高于`unconstrained`。

如果只超过baseline、未超过普通交互，只能支持新增交互结构可能有用，不能支持角色分解这项核心主张。如果通过，再用N开始重置C、单源版本、C-only分别检验迁移、组合和目标角色自由度；这些后续条件不算已经完成。

当前Dev经过多轮开发，单seed结果仅用于探索。独立封存测试、多seed和用户级配对不确定性仍是后续论文证据需求。

## 10. 相关文献：联系、差异和使用边界

下列文献用于定位已有技术与最近邻，不意味着它们已经提出本项目的完整CIR到N角色共享方案，也不意味着本方案的新颖性已经得到穷尽检索确认。

本轮重新核验了下面7篇论文的arXiv摘要页、题名、作者和版本信息。正文阅读范围沿用前期记录，不把本轮摘要核对写成全文复核。R7按2021年v5书目信息引用，不将早期版本的模型命名或细节无条件归于最终版本。

| 编号 | 文献与链接 | 最相关的已有内容 | 与本方法的区别及引用用途 | 已核对范围 |
|---|---|---|---|---|
| R1 | Mathieu Blondel等：[Higher-Order Factorization Machines](https://arxiv.org/abs/1607.07195)，2016，HOFM；[已读正文v2](https://arxiv.org/html/1607.07195v2) | 低秩高阶交互、ANOVA核和高效求和 | 本方法三线性乘法和快速类别对聚合的数学近邻；不能把这些工具本身称为新贡献 | 前期已读方法2-5；本轮书目核对 |
| R2 | Liangliang Chen等：[Dual-Scale Interest Extraction Framework with Self-Supervision for Sequential Recommendation](https://arxiv.org/abs/2310.10025)，ECAI 2023，DSIE；[已读正文v1](https://arxiv.org/html/2310.10025v1) | 全局与局部兴趣、全局引导、跨出已有类别的探索 | 新兴趣探索动机的直接近邻；其评测中已见类别过滤与本项目完整词表无过滤生成不同 | 前期已读方法3.2-3.5及评测4.4；本轮书目核对 |
| R3 | Gaode Chen等：[Exploring Periodicity and Interactivity in Multi-Interest Framework for Sequential Recommendation](https://arxiv.org/abs/2106.04415)，IJCAI 2021，PIMI；[已读正文v1](https://arxiv.org/html/2106.04415v1) | 行为图上的item交互、周期信息与多兴趣提取 | 说明多兴趣与交互联合建模已有研究；本方法要证明的是CIR/N角色共享约束，不是首次利用交互建模多兴趣 | 前期已读方法3.2-3.4；本轮书目核对 |
| R4 | Théo Trouillon等：[Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357)，ICML 2016，ComplEx | 用复数表示建模对称与非对称关系 | 对称/非对称关系表达已有先例；本方法不是ComplEx，也不是新的二元反对称分解 | 前期及本轮摘要、书目核对 |
| R5 | Benjamin Lengerich等：[Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models](https://arxiv.org/abs/1911.04974)，AISTATS 2020；[已读正文v2](https://arxiv.org/html/1911.04974v2) | 给定参考分布后分离低阶效应和交互效应 | 用于限定“中心化/纯交互”的解释：本方法输出类均匀中心化不等于真实用户分布上的fANOVA净化 | 前期已读方法3-5；本轮书目核对 |
| R6 | Jason D. Lee等：[Predicting What You Already Know Helps: Provable Self-Supervised Learning](https://arxiv.org/abs/2008.01064)，NeurIPS 2021 | 重建式自监督在明确统计假设下的迁移收益 | 可支持CIR作为辅助任务的背景讨论，不直接证明角色投影、任意CIR迁移或本项目效果 | 前期及本轮摘要、书目核对 |
| R7 | Hanxiong Chen等：[Neural Collaborative Reasoning](https://arxiv.org/abs/2005.08129v5)，WWW 2021，NCR；[出版DOI](https://doi.org/10.1145/3442381.3449973) | 可微AND/OR/NOT模块与从历史推导偏好的逻辑表达式 | “从匹配走向组合推理”不是新的立论；本方法不学习逻辑真值或逻辑运算，而以原CE训练CIR/N角色约束残差 | 本轮核对v3/v5摘要页及最终书目信息，不据此宣称已复核v5正文 |

**最应该优先对照的是R1、R2、R3；若使用“组合推理”叙述，还必须对照R7。** R4、R5用于约束数学与解释性措辞；R6是迁移背景，不是角色算子的直接来源。

本文不把CPR/BPR、教师蒸馏、数据构造、Hodge分解或新的因果识别理论混入当前方法定义。潜在贡献必须来自经过对照验证的“回顾补全与未来进入采用不同角色共享约束”，而不是给已有数学工具重新命名。

## 11. 文档与代码的对应及当前证据

| 说明内容 | 代码或记录 |
|---|---|
| 三线性、中心化、快速聚合 | [TrilinearBranch](../../../src/crossinterest/methods/role_projection/model.py#L16) |
| C/D阶段启用与FP32计算 | [RoleProjectionHead](../../../src/crossinterest/methods/role_projection/model.py#L58) |
| 首ID隐藏状态、可见类别、元数据加载 | [RoleProjectionController](../../../src/crossinterest/methods/role_projection/model.py#L98) |
| 复用的taxonomy取值、logit注入与头保存 | [原RelationController](../../../src/crossinterest/methods/mechanism/model.py#L80) |
| 原CIR输入与set CE、原两阶段loss路由 | 冻结mechanism_train（私有原始证据，不随公开仓库分发） |
| 新Dataset及Trainer | [role_train](../../../src/crossinterest/methods/role_projection/train.py) |
| 原评测器接入 | [role_evaluate](../../../src/crossinterest/methods/role_projection/evaluate.py) |
| 数据/代码合同、执行、阶段和原始排名审计 | [role_run](../../../src/crossinterest/methods/role_projection/runner.py) |
| 公式、梯度、RNG、LM/PEFT、beam及checkpoint测试 | <archived-job-id>（私有原始证据，不随公开仓库分发） |
| 固定实验条件与通过门槛 | [EXPERIMENT_DESIGN.md](design.md) |
| 运行状态与结果 | [EXPERIMENT_LOG.md](log.md) |
| 已提交快照合同 | CONTRACT.json（私有原始证据，不随公开仓库分发） |

当前v2代码完成了22项本地及提交快照检查，包括小尺度taxonomy更新、初始loss与原trainer逐值一致、完整20-beam、以及optimizer checkpoint恢复后参数逐值一致。这是实现正确性证据，不是推荐收益证据。

v1烟测在unconstrained的N参数更新审计处失败，baseline结构检查通过，projected未执行；v2在baseline初始化遇到参数报告I/O失败。执行快照v3保留RMS方法并独立保存参数报告，已于2026-09-23 08:31 UTC完成，三条件阶段更新及结构审计通过，但三组各320个beam的严格有效ID均为0。随后同权重token诊断确认，重放样本中看似合法的ID文本由普通子词拼写，原单token协议正确拒绝；原烟测验收（私有原始证据，不随公开仓库分发）及后续放行决策（私有原始证据，不随公开仓库分发）保留，不回写原零分或失败状态。

正式三条件使用原launch-v3和原完整预算，均于9月23日晚Completed；9月25日原始排名独立复算通过，三组Dev2000严格有效ID率均100%。baseline/unconstrained/projected的macro nDCG@20分别为0.160691/0.162703/0.164184。projected优于两组匹配对照，但相对baseline仅+0.003493，未达到预设+0.005门槛；单seed且重复使用Dev，不能据此宣称角色约束已验证有效或有统计显著性。关系头张量及继承已本地复验，完整LoRA张量仍因下载超时未完成本地复验，有限性及更新依据云端审计。详见独立结果验收（私有原始证据，不随公开仓库分发）及[实验日志](log.md)。本轮未自动追加消融、调参或实现文献讨论的新候选。

本文是方法说明，不修改已提交训练代码、冻结数据或原实验合同。早期文献笔记中仅围绕112类描述历史源的简化，以本文保留全部可见历史类别的实现定义为准。