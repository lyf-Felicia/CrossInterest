# MeRA Amazon 双主模型复现实验记录 (Experiment Log)

> 项目：`crossinterest.baselines.mera/`，在官方 Arts/Grocery 与冻结 Home & Kitchen 上适配复现 MeRA。
> 关键约束：Qwen3.5-9B 与 Llama-3.1-8B 构成适配矩阵；Qwen2-1.5B 在 Arts/Grocery 直接执行固定官方源码；旧 Arts Natural/Balance 仅作非严格诊断；未完成 GPU 结果保持空白。
最近更新：2026-09-07

---

## 1. 实验目标

1. 审计并固定 MeRA 官方数据、代码和发布预测，区分论文事实与发布资产事实。
2. 在两个项目主模型上保留类别 token、语义初始化、结构关系和协同关系的核心机制。
3. 在 Arts/Grocery 保留论文 R/P@K，同时为三套数据统一补充源 query 等权的 Accuracy、set Recall@10/20 和 binary nDCG@10/20；Home 类别宏另表保留。
4. Home & Kitchen 逐条复用冻结 Dev-2K 候选与用户隔离数据，不在正式评测集调参。
5. 先通过双模型 smoke，再运行 6-job 正式矩阵和 6-job Home 消融。
6. 用 Qwen2-1.5B 在 Arts/Grocery 直接执行固定官方代码和公开资产；旧 Natural/Balance 两个同预算 job 仅用于隔离类别均衡影响。

## 2. 模型与方法

| 设置 | 模型 | 数据 | 有效 batch | Epoch | LoRA alpha | 关系负样本 |
|---|---|---|---:|---:|---:|---:|
| 官方数据适配 | Qwen3.5-9B / Llama-3.1-8B | Arts | 36 | 最多 10，patience 2 | 16 | 5 |
| 官方数据适配 | Qwen3.5-9B / Llama-3.1-8B | Grocery | 36 | 最多 10，patience 2 | 16 | 10 |
| 统一扩展 | Qwen3.5-9B / Llama-3.1-8B | Home 112K | 32 | 1 | 32 | 10 |
| official-code strict | Qwen2-1.5B | Arts / Grocery | 252 | 最多 10，patience 2 | 16 | 5 / 10 |
| 作者澄清协议 | Qwen2-1.5B / Qwen3.5-9B / Llama-3.1-8B | Arts / Grocery / Home | 252 | 固定 5，无 early stopping | 16 | 5 / 10 |
| 历史非严格诊断 | Qwen2-1.5B | Arts Natural / Balanced | 36 | 最多 10，patience 2 | 16 | 5 |

适配矩阵设置：LR `1e-4`、cosine、warmup ratio `0.03`、bf16、最大长度 512、LoRA r=16/dropout=0.05/all-linear、结构权重 0.05、协同权重 0.1、seed 2024。新增类别 input embedding 仅训练新增行，LM head 使用 LoRA；InfoNCE 同时作用于 input embedding 与包含 LoRA 增量的有效输出行。

strict 设置：固定 `GabyUSTC/MeRA@c07bdbe...` 与 `Qwen/Qwen2-1.5B@8a16abf...`，8-bit base + fp16，完整 input/output category representation 与 `embed_tokens`/`lm_head`，官方 shifted CE、raw-dot 关系采样、AdamW Torch、linear scheduler、warmup 20；Arts 权重 0.01/0.01，Grocery 权重 0.1/0.1。官方源码只修复推理中的 `.indice` 为 `.indices`。

Qwen3.5 使用其 chat template并显式关闭 thinking，Llama 使用原生 chat template。当前 runner 对两者均使用标准 Transformers forward，不启用与 Torch 2.8 不兼容的 Hub kernels；两者均不做文本生成，直接对冻结类别 token logits 排序。

## 3. 仓库结构

```text
crossinterest.baselines.mera/
├── core.py / data.py / modeling.py  # 纯函数核心、数据适配、模型适配
├── prepare_data.py                  # Train-only 关系资产
├── fetch_official_data.py           # 固定 commit 与 SHA 下载门禁
├── prepare_official_strict.py       # 官方 checkout 与唯一源码修复门禁
├── run_official_strict.sh           # strict 环境、GPU preflight、训练/推理 runner
├── evaluate_official_strict.py      # 完整排名与双实现指标验收
├── train.py / evaluate.py           # 双模型训练与同次前向双口径评测
├── validate_artifact.py             # 训练产物合同检查
├── run_train_eval.sh                # AMLT runner
├── amlt/                            # Qwen2 对齐、smoke、正式矩阵、Home 消融
├── data/{arts,grocery,home}/        # 冻结小型关系资产
├── tests/test_core.py               # 真实数据与合成合同测试
├── README.md / RESULTS.md
└── EXPERIMENT_REPORT.md
```

## 4. 运行环境

- 本地：Linux；工作目录 `crossinterest_formal/amazon`。
- CPU 验证：当前基础环境未安装 pytest 命令，因此先用直接函数 runner 执行；最终使用 conda `exp` 的 pytest 复验。
- AMLT CLI：`11.15.0.post1`；项目 `mind-crossinterest-eval`。
- 目标：Feeds Singularity，workspace `${AZURE_CONFIGURATION}@${AZURE_CONFIGURATION}`，UAI `${AZURE_CONFIGURATION}`。
- 2026-08-31 提交前配额：Feeds Standard A100 58/97 已用、39 空闲；Recall 16/33 已用、17 空闲。Qwen2 两个单卡 job 选择 Feeds。
- GPU 计划：适配矩阵每个 job 为 4×80 GB A100；strict 每个 job 为 1×80 GB A100；均为 Standard SLA。
- 适配镜像为 `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04`。strict 使用同一平台可拉取基底，再以 `requirements_official_strict.txt` 覆盖并断言 Python 3.10、Torch 2.3.0、Transformers 4.37.0、PEFT 0.9.0、Accelerate 0.29.3、bitsandbytes 0.43.3、Datasets 2.19.0、NumPy 1.26.4、PyArrow 16.0.0。

## 5. 实验进度

### 实验 1 — 论文、官方代码与发布资产审计（✅ 已完成）

- **配置**：官方仓库 `GabyUSTC/MeRA`，commit `c07bdbe57bf82e5c938b478035baee78d630b17c`。
- **启动方式**：固定 commit 本地只读 checkout；全量读取 JSON、taxonomy、共现矩阵和发布预测。
- **踩坑**：Grocery shell 调用不存在的 `train_with_extended_tokenizer.py`；推理写成不存在的 `output.indice`；论文与脚本的 batch/关系权重冲突。
- **已确认正常**：Arts 20,186/2,523/2,524、143 类；Grocery 34,981/4,373/4,373、172 类。全部样本满足目标新颖性，taxonomy 完整覆盖标签。
- **结果**：Arts 发布预测复算与论文一致；Grocery 发布预测复算为 R@1/2/3/4 `0.1175/0.1989/0.2635/0.3224`、P@1/2/3/4 `0.3174/0.2752/0.2491/0.2310`，与论文表不完全一致。
- **产物**：`fetch_official_data.py`、`EXPERIMENT_REPORT.md`。
- **耗时 / EXIT**：CPU 审计命令 EXIT 0。

### 实验 2 — 三数据集关系资产与统一评测构造（✅ 已完成）

- **配置**：Train-only 集合共现，条件共现阈值 0.05；同父类 sibling 结构图；统一候选 seed 20260806。
- **启动方式**：`python -m crossinterest.baselines.mera.prepare_data --dataset <name> ...`。
- **踩坑**：Home 的 112 类只表示目标支持集，历史实际覆盖 136 类 -> 改为 136 个上下文 token、112 个预测标签的双空间合同，禁止丢弃合法历史。
- **已确认正常**：Arts/Grocery/Home 的上下文/预测类别分别为 143/143、172/172、136/112；源 query 为 2,524、4,373、2,000，分别包含 7,633、14,782、2,000 个 gold 判断。多 gold Recall/nDCG 使用完整 gold 集并按源 query 等权；10-way 先在 query 内按 gold 平均。Home 候选逐条复用冻结 `candidate_interests`。
- **Balance 审计**：Arts/Grocery Train 分别仅 115/143、141/172 个类作为目标出现，非零频次范围为 1--1,403 与 1--2,084；官方代码无重采样或 class-weighted loss。Home Train 原生为 112 类各 1,000 条。完整计数已写入 `target_distribution.json`。
- **结果**：关系资产构建完成；无模型指标。
- **产物**：`data/{arts,grocery,home}/label_space.json`、`*.npy`、`MANIFEST.json`。
- **耗时 / EXIT**：三套正式资产构建 EXIT 0。

### 实验 3 — 双模型实现、契约测试与 AMLT 预检（✅ 已完成）

- **配置**：Qwen3.5-9B 与 Llama-3.1-8B 共用训练/评测入口；稀疏类别行 + LM-head LoRA；每源样本一次前向。
- **启动方式**：

```bash
conda run -n exp python -m pytest -q crossinterest.baselines.mera/tests
conda run -n exp amlt run crossinterest.baselines.mera/amlt/amlt_formal.yaml \
  mera-formal-config-check --dump
```

- **踩坑**：本地 Torch 无 `torch.flatnonzero` -> 改为兼容的 `torch.nonzero(...).flatten()`；官方下载根目录与资产构建器目录接口不同 -> 同时兼容数据根和仓库根；四卡 Trainer 将模型包装为 DDP，关系表示入口必须先沿 `.module` 解包。
- **已确认正常**：Python/Bash 编译通过；17 项 pytest 通过（新增 DDP wrapper 回归测试）；各 AMLT 配置均由 `amlt run --dump` 成功解析。
- **结果**：仅验证链路与合同，不产生正式模型指标。
- **产物**：`train.py`、`evaluate.py`、`validate_artifact.py`、`run_train_eval.sh`、`amlt/*.yaml`。
- **耗时 / EXIT**：本地验证 EXIT 0；未提交 AMLT。

### 实验 4 — Arts Qwen2 Natural 与单组 Balance 历史诊断（✅ 已完成，非 strict）

- **配置**：`amlt/amlt_qwen2_arts.yaml`；`qwen2-arts-natural` 保留官方自然长尾，`qwen2-arts-balanced` 仅对 Train 的 115 个已观测目标类做固定预算均匀重采样。
- **控制变量**：两个 job 均为 20,186 Train 行、论文参数优先、单卡 Feeds Standard `80G1-A100`；只改变 `BALANCE_STRATEGY`。Validation/Test、关系资产、seed 与评测器完全共用。
- **实现合同**：Qwen2 官方带引号类别 token、官方 Alpaca 训练 prompt、官方 token-only 推理 prompt；启动时解析并冻结 Hugging Face commit SHA。
- **启动方式**：`amlt run crossinterest.baselines.mera/amlt/amlt_qwen2_arts.yaml mera-qwen2-arts-alignment-20260831 -d "Qwen2 Arts natural reference and one target-balance ablation" -y`。
- **已确认正常**：真实 Arts balance 为 54 类×175、61 类×176，共 20,186 行；Python/Bash 编译、16 项 pytest、四份 AMLT `--dump` 与 Markdown 表格检查均通过。
- **结果**：Natural 的论文口径 R@1/2/3/4 为 `0.0534/0.0997/0.1479/0.1919`，P@1/2/3/4 为 `0.1478/0.1359/0.1337/0.1297`；统一 Query Accuracy `46.03%`、Recall@10/20 `0.3785/0.6063`、nDCG@10/20 `0.2485/0.3239`。Balanced 的对应结果为 R `0.0079/0.0225/0.0372/0.0553`、P `0.0273/0.0323/0.0353/0.0381`、Accuracy `22.20%`、Recall `0.1529/0.3132`、nDCG `0.0857/0.1371`。固定预算均衡显著降低全部指标。
- **产物**：两个 job 均完整输出 frozen model revision、balance 前后频次合同、final adapter、8 MB 源排名、统一明细与 metrics。
- **AMLT experiment**：`mera-qwen2-arts-alignment-20260831`，Feeds，Standard `80G1-A100`。
- **Backend ID**：Natural `<archived-job-id>`；Balanced `<archived-job-id>`。
- **当前状态**：Natural 与 Balanced 均为 `pass`，产物合同完整；但参数化、CE、关系采样、batch、scheduler 与精度均不等价于官方实现，不得作为 strict 复现结果。
- **耗时 / EXIT**：Natural 39 分钟；Balanced 33 分钟；两项均 EXIT 0。

### 实验 5 — 双模型 GPU smoke（Llama ✅ / Qwen r3 🟡）

- **配置**：`amlt/amlt_smoke.yaml`；Home 每模型 2 updates、8 个训练/评测样本。
- **首轮失败**：`mera-all-smoke-20260831`。Qwen3.5 `<archived-job-id>` 因未安装 Transformers `kernels` 失败；Llama `<archived-job-id>` 因关系损失未解包 DDP 模型失败。两项结果均废弃。
- **r2 修复与结果**：固定 `kernels==0.15.2`；`category_representations()` 沿 `.module` 解包到 embedding API；本地新增回归测试后共 17 项通过。Llama r2 `<archived-job-id>` 完成 2-step 训练、产物校验和 8 样本评测并 `pass`。Qwen r2 `<archived-job-id>` 完成训练和 adapter 校验，但评测启用 Hub activation kernel 时失败：可用构建要求 Torch 2.10，镜像为 Torch 2.8；该结果废弃。
- **r3/r4 修复**：r3 先在外部评测关闭 Qwen kernels；正式 Arts/Grocery 随后证明 Trainer validation 也会触发同一不兼容 kernel。r4 runner 因此不再自动传 `--use-kernels`，训练与评测统一使用标准 Transformers forward。17 项 pytest、Bash 语法和 AMLT dump 均通过。
- **有效 AMLT experiment**：Llama `mera-smoke-repair-20260831`；Qwen `mera-qwen35-smoke-repair-r3-20260831`，均为 Feeds Standard `80G4-A100`。
- **有效 Backend ID**：Llama `<archived-job-id>`（pass）；Qwen3.5 r3 `<archived-job-id>`（queued）。
- **结果 / 产物**：Llama smoke 产物完整；Qwen r3 等待调度。smoke 数值不进入正式结果表。
- **耗时 / EXIT**：提交命令 EXIT 0。

### 实验 6 — 三数据集双模型正式矩阵（⚠️ 3 完成 / 3 待运行）

- **配置**：`amlt/amlt_formal.yaml`，6 个独立 4×A100 job。
- **废弃首轮**：`mera-main-matrix-20260831` 使用修复前代码；所有 Qwen/Llama job 已 killed 或自行失败，不作为结果来源。
- **废弃 Qwen r2/r3**：r2 的三项使用评测 kernels，已全部 killed；r3 Arts/Grocery 在 step 250 的 Trainer validation 调用 Hub activation kernel，因只有 Torch 2.10 build 而镜像为 Torch 2.8 失败。runner 已彻底关闭该执行优化，方法与超参数不变。
- **有效 AMLT experiment**：Qwen Home `mera-qwen35-main-repair-r3-20260831`；Qwen Arts/Grocery `mera-qwen35-ag-standard-r4-20260901`；Llama `mera-llama31-main-repair-20260831`。
- **有效 Backend ID（Qwen3.5）**：Arts r4 `<archived-job-id>`、Grocery r4 `<archived-job-id>`、Home r3 `<archived-job-id>`。
- **有效 Backend ID（Llama）**：Arts `<archived-job-id>`、Grocery `<archived-job-id>`、Home `<archived-job-id>`。
- **当前状态**：Llama Arts/Home 与 Qwen Home `pass`；Llama Grocery、Qwen Arts r4、Qwen Grocery r4 `queued`。
- **已确认正常**：AMLT `--dump` 与结构化提交验收通过；正式结果表已建立且保持空白。
- **结果**：Llama Arts R@1/2/3/4 `0.1284/0.2089/0.2781/0.3399`，P@1/2/3/4 `0.3031/0.2617/0.2392/0.2223`；统一 Accuracy `53.67%`、Recall@10/20 `0.5637/0.7466`、nDCG@10/20 `0.4118/0.4724`。Home Qwen/Llama 统一 Accuracy 为 `13.50%/14.30%`；完整指标见 `RESULTS.md`。
- **产物**：计划每个 job 输出训练合同、adapter、源排名、统一明细与 metrics。
- **耗时 / EXIT**：提交命令 EXIT 0。

### 实验 7 — Home 双模型消融（✅ 6/6 完成）

- **配置**：`amlt/<archived-job-id>.yaml`；Category-token SFT、w/o structure、w/o collab，各两个模型。完整 MeRA 复用实验 6 的 Home 两个 job。
- **废弃首轮**：`mera-home-ablations-20260831` 使用修复前代码；所有 job 已 killed 或自行失败，不作为结果来源。
- **废弃 Qwen r2**：`mera-qwen35-ablations-repair-20260831` 的三项使用评测 kernels，已全部 killed，不作为结果来源。
- **有效 AMLT experiment**：Qwen3.5 `mera-qwen35-ablations-repair-r3-20260831`；Llama `mera-llama31-ablations-repair-20260831`，均为 Feeds Standard `80G4-A100`。
- **有效 Backend ID（Qwen3.5 r3）**：Category-token SFT `<archived-job-id>`；w/o structure `<archived-job-id>`；w/o collab `<archived-job-id>`。
- **有效 Backend ID（Llama）**：Category-token SFT `<archived-job-id>`；w/o structure `<archived-job-id>`；w/o collab `<archived-job-id>`。
- **当前状态**：Qwen/Llama Category-token SFT、w/o structure、w/o collab 六个任务均为 `pass`；完整 MeRA 复用实验 6 的 Home 两个结果。
- **已确认正常**：零权重关系项会被真正跳过；AMLT `--dump` 与结构化提交验收通过。
- **结果**：Qwen w/o collab 的 Macro Accuracy/Recall@10/Recall@20/nDCG@10/nDCG@20 为 `12.91%/0.1720/0.2943/0.0858/0.1166`；Qwen/Llama w/o structure 均全面超过各自完整 MeRA。Llama Category-token SFT 为 `15.10%/0.1895/0.3201/0.0961/0.1287`，是当前 Home 消融的最高 Accuracy。完整矩阵见 `RESULTS.md`。
- **产物**：计划同实验 5。
- **耗时 / EXIT**：提交命令 EXIT 0。

### 实验 8 — Qwen2 Arts/Grocery official-code strict（✅ 训练与恢复评测完成）

- **配置**：`amlt/amlt_official_strict.yaml`；固定官方 commit、Qwen2 revision、12 份数据 SHA、模型资产 SHA、精确依赖版本和训练合同。Arts/Grocery 分别保留官方 5/10 个关系负样本和 0.01/0.01、0.1/0.1 关系权重。
- **启动方式**：有效当前版为 `amlt run crossinterest.baselines.mera/amlt/amlt_official_strict.yaml :qwen2-official-strict-preflight mera-official-strict-preflight-r9-20260901 --yes`；后台 `gate_and_submit_official_strict.sh` 仅在训练级合同通过后提交 `mera-official-strict-r9-20260901`。
- **踩坑**：首轮基底镜像 404；r4 暴露新版 Apex 与 Torch 2.3 冲突；r6 image-setup 模板无效。r8 的加载级 preflight 未执行 LoRA 注入，导致首轮正式任务在 `get_peft_model` 暴露 PEFT 0.9/bitsandbytes 0.45.5 API 不兼容，未进入训练。r9 固定 bitsandbytes 0.43.3，并将检查升级为 1,728 个 Train 样本、48 次真实 optimizer step、官方 checkpoint、adapter 保存与新进程重载。
- **已确认正常**：本地 21 项 pytest、两个 Bash 脚本语法、官方源码 diff、AMLT dump、三 job 合同和训练级 r9 断言均通过。GPU 产物记录 `loaded_in_8bit=true`、bitsandbytes 0.43.3、48 个 optimizer step、`preflight-model/checkpoint-48`、`adapter_reload=passed`，重载 logits 形状 `[1, 2, 151789]`；锁定环境 expected/actual 逐项一致。恢复评测另验证原任务训练合同、官方/模型资产合同、最终 adapter SHA 与全部 checkpoint state。
- **结果**：Arts R@1/2/3/4 `0.0786/0.1485/0.2046/0.2520`，P@1/2/3/4 `0.2021/0.1882/0.1783/0.1649`；统一 Accuracy/Recall@10/Recall@20/nDCG@10/nDCG@20 为 `46.16%/0.4641/0.6582/0.3162/0.3805`。Grocery R@1/2/3/4 `0.0503/0.0933/0.1406/0.1830`，P@1/2/3/4 `0.1180/0.1108/0.1126/0.1123`；统一五指标为 `36.63%/0.3666/0.5965/0.2315/0.3091`。官方循环公式与项目独立实现逐值一致。
- **产物**：包含 `official-assets.json`（含唯一官方源码 diff 与数据 SHA）、`model-assets.json`、`project-source-contract.json`（关键项目文件 SHA）、`environment-contract.json`、`environment-lock.txt`、`preflight-contract.json`（8-bit 与类别 token 空间）、`train_contract.json`、完整全类别排名，以及同一 `evaluation/metrics.json` 中的官方公式与项目公式双算结果。
- **AMLT experiment**：训练级检查 `mera-official-strict-preflight-r9-20260901`，Backend ID `<archived-job-id>`，状态 `pass`；正式实验 `mera-official-strict-r9-20260901`，Arts Backend ID `<archived-job-id>`，Grocery Backend ID `<archived-job-id>`。
- **恢复评测 experiment**：`mera-official-strict-eval-recovery-r1-20260901`；Arts Backend ID `<archived-job-id>`，Grocery Backend ID `<archived-job-id>`，均为 `pass`。
- **当前状态**：Arts/Grocery 正式训练均完成；原任务只因训练结束后的官方推理入口缺少 Gradio 而退出。恢复任务复用已持久化 adapter，仅执行官方推理和双公式评测，没有重训。
- **耗时 / EXIT**：恢复评测 Arts 9 分钟、Grocery 10 分钟，均 EXIT 0。

### 实验 9 — Qwen3.5/Llama Arts/Grocery 官方源码跨模型实现（⚠️ preflight r5 已接收）

- **配置**：`amlt/<archived-job-id>.yaml`；固定官方 commit、两模型 revision 和官方数据，差异只允许来自基础模型架构、tokenizer/embedding API 与当前 PEFT/Trainer API。Qwen3.5/Llama LoRA 分别精确覆盖 248/224 个 decoder linear projections，完整 input embedding 与 LM head 保持官方 `modules_to_save` 语义。
- **启动方式**：`amlt run crossinterest.baselines.mera/amlt/<archived-job-id>.yaml :qwen35-official-preflight :llama31-official-preflight mera-official-crossmodel-preflight-r5-20260901 --yes`；双门禁通过后由 `gate_and_submit_official_crossmodel.sh` 验收同源合同并提交四个 Arts/Grocery 正式任务。
- **踩坑**：r2 暴露环境漏装 `fire`；r3/r4 继续补齐 `datasets`、`gradio` 和完整训练/推理入口依赖。r4 两个模型均在构造 `TrainingArguments` 时发现 Transformers 5.14.1 删除旧 `group_by_length` 参数，未执行 optimizer step。r5 等价迁移到 `<archived-job-id>="group_by_length"`；5.14.1 Trainer 仍调用相同 `LengthGroupedSampler`，batch 计算和排序算法不变。
- **已确认正常**：21 项 pytest、两个 Bash 脚本语法、六任务 AMLT dump 和锁定依赖解析通过。训练级检查要求一次真实 optimizer step、两项关系损失均执行、运行合同记录 `group_by_length` 采样、adapter 保存并在新模型实例中重载。
- **AMLT experiment**：`mera-official-crossmodel-preflight-r5-20260901`；Qwen3.5 Backend ID `<archived-job-id>`，Llama Backend ID `<archived-job-id>`。
- **当前状态**：两个任务均已由平台接收；未产生正式结果。
- **耗时 / EXIT**：r5 提交 EXIT 0。

### 实验 10 — Home 官方代码两项关系损失 2×2 消融（✅ 已完成）

- **配置**：`amlt/amlt_official_home_ablations.yaml`；Qwen3.5-9B 与 Llama-3.1-8B 各运行 `SFT / structure-only / collaborative-only / both` 四组。除 `structure_weight` 和 `collaborative_weight` 外，模型 revision、官方源码、Home 数据、类别 token、LoRA、完整 embedding/head、batch、epoch、优化器、scheduler、seed 和评测器均相同。
- **Home 必要适配**：`prepare_official_home.py` 将冻结 112K Train 和 Dev-2K Interest 视图转换为官方 MeRA JSON；136 个合法历史类全部注册官方类别 token并参与关系约束，112 个 Train target 类构成预测空间。validation 从完整 Dev 中分层抽取 2,000 条，并排除 Dev-2K 的全部 1,963 个用户；validation/test 用户交集为 0。原始共现矩阵只由官方 JSON 中可见的 Train 历史和 target 重建。
- **训练合同**：固定官方 commit `c07bdbe...`，官方 shifted CE、raw-dot 关系采样、有效 batch 252、10 epoch、LR `1e-4`、LoRA r/alpha `16/16`、linear scheduler、warmup 20、patience 2；非零关系权重固定为 structure `0.05`、collaborative `0.1`。推理先生成完整 136 类排名，再确定性过滤到冻结 112 类，复用同一 Dev-2K 候选计算类别宏指标。
- **已确认正常**：Home 资产生成门禁通过，112,000/2,000/2,000 行、136/112 类、官方 `input[81:]` 解析回环、Train-only 共现和 validation/test 用户隔离均通过；Home evaluator 在真实 2,000 条 metadata 上通过合成 136 类完整排名的端到端测试；完整 pytest `22 passed`，Shell 语法、AMLT dump 和静态诊断通过。
- **启动方式**：`amlt run crossinterest.baselines.mera/amlt/amlt_official_home_ablations.yaml :qwen35-home-official-preflight :llama31-home-official-preflight mera-official-home-preflight-r2-20260902 --yes`。`gate_and_submit_official_home.sh` 已在双模型一步真实训练、两项关系 loss、跨卡 loss 设备、全 decoder LoRA、adapter 保存和 8-bit 重载合同全部通过后，自动提交八个正式任务到 `mera-official-home-factorial-r1-20260902`。
- **踩坑**：r1 的 Home `category_cooccurrence.npy` 沿用了通用构建器的 `int64` 计数 dtype；官方 trainer 初始化时执行 `self.cooccurence_matrix.var(0)`，当前 PyTorch 报 `RuntimeError: std and var only support floating point and complex dtypes`。修复只在 official 资产保存边界将同一整数计数矩阵无损转换为 `float32`，不改矩阵值、关系采样或损失；真实 136×136 资产已通过同款 `torch.from_numpy(matrix).var(0)` 检查，所有 variance 均为有限值。r1 双作业已取消。r2 完成后，gate 首次下载因 AzCopy 不支持带目录的 include pattern 失败；将唯一 Home manifest 的过滤器从 `home-assets/manifest.json` 改为 basename `manifest.json` 后，已验证只下载目标 manifest 并完成全部合同验收。正式训练保存完成后，官方 inference 在 Transformers 5.14.1 读取已删除的 `tokenizer.additional_special_tokens_ids` 动态属性，Qwen 与 Llama 分别报 `Qwen2Tokenizer` / `TokenizersBackend` `AttributeError`；训练 adapter 不受影响。兼容层现改用公开 `tokenizer.get_added_vocab()`，只取训练时追加在词表末尾的最后 136 个 token，并严格断言 ID 连续、数量为 136 且 token 均带官方双引号。正式产物已交叉验证 Qwen ID `248077..248212`、Llama ID `128256..128391`，两者均满足该合同；官方 logits 切片和 `topk` 排名算法不变。
- **AMLT experiment**：修复版 preflight `mera-official-home-preflight-r2-20260902` 双任务均为 `pass`：Qwen3.5 `<archived-job-id>`、Llama `<archived-job-id>`。两者 `global_step=1`，structure/collaborative loss 均调用 63 次，主/组合 loss 同在 `cuda:0`，LoRA 分别覆盖 248/224 个 decoder projection，完整 `embed_tokens`/`lm_head` 已保存，8-bit adapter 重载通过，Home 共现资产为 `float32`。正式训练产物 8/8 均已持久化；Qwen 四项均由官方 early stopping 停在 step 56，Llama SFT/both 停在 step 88，structure 停在 step 160。平台状态表对训练后推理失败存在滞后，训练完成以根 adapter、`training-runtime-contract.json` 与日志为准。
- **恢复评测**：r1 `mera-official-home-eval-recovery-r1-20260902` 在类别 token ID 修复后继续暴露 Transformers 5.14.1 已删除官方 inference 的旧 `load_in_8bit=` 参数；Qwen/Llama 均在 base model 构造时失败，未执行测试前向。兼容层已等价迁移为当前 `BitsAndBytesConfig(load_in_8bit=True)` API，默认正式 inference 的 `load_8bit=False` 仍保持非量化加载；完整测试 `22 passed`。四卡 r2 在排队/准备阶段取消，未作为结果来源。唯一有效恢复实验为单卡 r3 `mera-official-home-eval-recovery-r3-20260902`：恢复 runner 从各源任务 `train_contract.json` 自动恢复模型 revision 和损失条件，只读复用原 `model/`、`home-assets/` 与训练合同，不执行 `train.py`。八项均为 `pass`，并分别输出 136 类完整排名、112 类过滤排名、`metrics.json`、`preflight-contract.json` 和 `recovery-contract.json`；八份恢复合同均记录 `training_executed=false`。Backend ID：Qwen SFT/structure/collaborative/both 为 `<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`；Llama 对应为 `<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。
- **结果**：Qwen SFT/structure/collaborative/both 的 Macro Accuracy (%) 为 `9.77/10.63/9.33/10.33`，Macro Recall@10 为 `0.1073/0.0959/0.0938/0.0959`；Llama 对应 Accuracy 为 `10.32/11.66/10.90/10.87`，Recall@10 为 `0.0878/0.1231/0.0928/0.0955`。完整 Recall@20 与 nDCG@10/20 矩阵见 `RESULTS.md`。Qwen 的 structure-only 只在 Accuracy 上最优，SFT 在四项排名指标上最优；Llama 的 structure-only 在五项指标上均最优。
- **耗时 / EXIT**：r3 单项恢复评测墙钟 9--30 分钟；八项均 EXIT 0。最终合同验收确认每项 2,000 个 source、136 个上下文类、112 个预测类、112 个类别宏分组，双模型各四条件恰好齐全。

### 实验 11 — Qwen2 Arts/Grocery 作者澄清协议复现（⚠️ Arts 已完成，Grocery r4 运行中）

- **配置**：`amlt/amlt_author_clarified_arts.yaml` 与 `amlt/amlt_author_clarified_qwen2.yaml`；固定 `GabyUSTC/MeRA@c07bdbe57bf82e5c938b478035baee78d630b17c`、`Qwen/Qwen2-1.5B@8a16abf2848eda07cc5253dec660bf1ce007ad7a` 和官方 Arts/Grocery 数据。训练输入改为与推理一致的无 instruction 历史类别串，固定完整 5 epochs，不启用 early stopping，也不回载 best checkpoint；其余保持官方设置：8-bit base + FP16、有效 batch 252、LR `1e-4`、LoRA r/alpha `16/16`、contrastive module `both`。Arts 的 structure/collaborative/negative 为 `0.01/0.01/5`，Grocery 为 `0.1/0.1/10`；评测均使用公开脚本对应的默认 FP16/no-mask 口径。
- **实现**：`<archived-job-id>.py` 在固定官方源码上新增 `generate_prompt_without_prompt(data_point)`，训练文本严格为“连续双引号历史类别 + gold 类别”；`train_on_inputs=False` 因而只监督 gold 类别 token。旧 `official` 协议保持不变，作者协议在 Arts 与 Grocery 使用各自官方数据参数。
- **启动方式**：有效训练级门禁为 Arts `mera-author-arts-preflight-r2-20260906` 与 Grocery `mera-author-grocery-preflight-r1-20260906`；修复正式产物策略后提交 Arts `mera-author-arts-formal-r2-20260907` 与 Grocery `mera-author-grocery-formal-r2-20260907`，不重复运行已通过的 GPU 门禁。
- **踩坑**：首轮 Arts 预检 `mera-author-arts-preflight-r1-20260906` 在生成 `train_contract.json` 前失败；根因是 runner 的 Python heredoc 顶层 `if` 多缩进两个空格，`bash -n` 无法检查 heredoc 内 Python。正式 r1 Arts 训练随后在 step 336/400、epoch 4.19 保存周期 checkpoint 时触发 BlobFuse `os.rename(tmp-checkpoint-336, checkpoint-336)` 的 `FileNotFoundError`；这不是 OOM、loss、数据或模型错误。Grocery r1 与六项跨模型 r1 为防止同类失败被主动取消，八项 r1 均不得作为结果来源。
- **已确认正常**：Arts r2 预检 `mera-author-arts-preflight-r2-20260906` 状态 `pass`，Backend ID `<archived-job-id>`，墙钟 13 分钟；Grocery r1 预检 `mera-author-grocery-preflight-r1-20260906` 状态 `pass`，Backend ID `<archived-job-id>`，墙钟 25 分钟。两项机器验收均确认固定源码/模型 revision、8-bit 加载、真实训练 48 optimizer steps和 `adapter_reload=passed`；类别数分别为 143/172，训练合同均为 5 epochs、no-prompt、无 early stopping/最佳 checkpoint 回载。当前全量 `27 passed`，Shell 语法、AMLT dump 与真实 formatter 样本均通过。
- **作者参考值（非本次实测）**：R@1/2/3 `0.1209/0.2149/0.2843`，P@1/2/3 `0.2987/0.2712/0.2415`。
- **结果**：Arts/Grocery 训练已分别达到计划 `400/400` 与 `690/690` optimizer step，最终 1.9 GB adapter 及 embedding/head 产物完整。Arts r4 的 R@1/2/3 为 `0.1180/0.2175/0.2796`，P@1/2/3 为 `0.2876/0.2686/0.2376`；Grocery 正式指标等待 r4 完成后回填。
- **产物与修复**：旧 `training-completion.json` 记录 Arts/Grocery epoch 4.991087/4.969136；这是 Transformers 4.37 达到 `max_steps` 后的浮点显示，不是训练中断。完成合同改为记录 `max_steps`、请求 epochs，并以 `global_step == max_steps` 验收；旧产物按确定的计划步数 400/690 兼容。首轮只读恢复命令未传 `TRAIN_PROTOCOL=author_clarified`，runner 因默认 `official` 而重建出不同训练合同，并在映射合同比较处失败，未执行正式推理。恢复配置现显式传入作者协议；runner 还会在 eval-only 且未显式传参时从源 `train_contract.json` 自动恢复 `training_protocol`。
- **AMLT experiment**：训练来源为 Arts r2 `mera-author-arts-formal-r2-20260907`（Backend ID `<archived-job-id>`）与 Grocery r2 `mera-author-grocery-formal-r2-20260907`（Backend ID `<archived-job-id>`）。首轮恢复实验 `qwen2-official-strict-eval-recovery-mera-author-arts-formal-r2-20260907` 与 `qwen2-official-strict-eval-recovery-mera-author-grocery-formal-r2-20260907` 均因上述协议缺参失败。修复版 r3 为 `mera-author-qwen2-arts-eval-recovery-r3`（Backend ID `<archived-job-id>`）与 `mera-author-qwen2-grocery-eval-recovery-r3`（Backend ID `<archived-job-id>`），均只读映射原训练产物、不执行重训。
- **提交包**：recovery 配置排除与 MeRA 无关的 `eval_interest_id_minionerec/` 大型产物后，上传包为 95 个文件、1.9 MB（`amlt code not-ignored` 显示约 2.0 MB），避免再次复制 2.3 GB 无关数据。
- **r3 失败与 r4 修复**：r3 两项均通过固定源码/数据合同生成、模型下载、环境检查与 8-bit/category-token preflight，随后在 recovery heredoc 第 3 行因 `import math` 多两个前导空格触发 `IndentationError: unexpected indent`；未执行正式推理，也未改变映射训练产物。runner 已修正该缩进，并将测试从“包含 `import math` 字符串”升级为提取完整 recovery heredoc 后执行 `compile()`；聚焦测试、`bash -n`、heredoc 独立编译及完整测试均通过，完整结果为 `27 passed`。
- **r4 提交**：Arts `mera-author-qwen2-arts-eval-recovery-r4`（Backend ID `<archived-job-id>`）与 Grocery `mera-author-qwen2-grocery-eval-recovery-r4`（Backend ID `<archived-job-id>`）均已由 Feeds Standard `80G1-A100` 接收，初始状态为 `queued`。代码包为 95 个文件、1.9 MB；两项均通过 map 只读复用对应 r2 最终 adapter，不执行重训。
- **耗时 / EXIT**：训练均完成；原任务仅因完成合同假阴性退出。r3 Arts/Grocery 均 EXIT 非零，失败 Backend ID 分别为 `<archived-job-id>` / `<archived-job-id>`。Arts r4 于 2026-09-14 12:55 CST 通过，2,524 个 source、143 类完整排名与官方/项目双公式一致性门禁全部满足；Grocery r4 仍在运行。

### 实验 12 — Qwen3.5/Llama 三数据集作者澄清协议复现（✅ 六项正式评测完成）

- **配置**：`amlt/amlt_author_clarified_crossmodel.yaml`；Qwen3.5-9B 与 Llama-3.1-8B 在 Arts、Grocery、Home 均使用 no-prompt、完整 5 epochs、无 early stopping/最佳 checkpoint 回载。除模型架构直接关联的兼容适配外，保留官方类别 token、初始化、shifted CE、raw-dot 关系采样、完整 embedding/head、有效 batch 252、LR `1e-4`、linear scheduler 和 warmup 20。
- **Home 特殊合同**：只运行完整 MeRA `LOSS_CONDITION=both`，不重复四组消融；使用冻结 112K Train、与正式 Dev-2K 用户隔离的 2K validation、冻结 Dev-2K test、136 个合法历史上下文类及 112 个预测类。训练关系权重为 structure `0.05`、collaborative `0.1`、negative 10；先生成 136 类完整排名，再过滤至 112 类并复用冻结候选计算 Home 类别宏指标。
- **启动方式**：首轮 `mera-author-crossmodel-preflight-r1-20260906` 在发现完成合同插桩与兼容保存块组合缩进错误后、分配 GPU 前主动取消。修复经固定官方源码真实组合编译与 `27 passed` 验证后，`mera-author-crossmodel-preflight-r2-20260906` 双模型门禁通过。周期 checkpoint 的 BlobFuse 故障修复后，六项重新提交到 `mera-author-three-dataset-formal-r2-20260907`。
- **预检门禁**：Qwen3.5/Llama 各执行 Home 一步真实训练，要求两项关系 loss 均运行、8-bit base、全 decoder LoRA、完整 embedding/head 保存、adapter 重新加载及 136/112 数据合同全部通过。两项运行时合同均记录 `global_step=1`，structure/collaborative loss 各调用 63 次，主损失与组合损失均位于 `cuda:0`，采样策略为 `group_by_length`，并禁用外部 reporting。Qwen3.5/Llama 重载合同分别得到有限的 `[1, 2, 248213]` / `[1, 3, 128392]` logits，且 `loaded_in_8bit=true`。
- **训练完成证明**：作者协议在 `trainer.train()` 返回后写出 `training-completion.json`；完成验收使用计划 optimizer step，而不是可能略低于整数的浮点 epoch。该插桩只读 Trainer state，不参与优化。
- **结果**：Arts Qwen3.5/Llama 的 R@1/2/3 为 `0.1033/0.1753/0.2334` 与 `0.1209/0.2077/0.2752`，P@1/2/3 为 `0.2575/0.2302/0.2056` 与 `0.2908/0.2581/0.2322`。Grocery 对应 R 为 `0.0964/0.1720/0.2299` 与 `0.0982/0.1808/0.2503`，P 为 `0.2646/0.2433/0.2220` 与 `0.2778/0.2561/0.2387`。Home Qwen3.5/Llama 的类别宏 Accuracy 为 `16.00%/15.34%`，Recall@10/20 为 `0.1944/0.3221` 与 `0.1889/0.3264`，nDCG@10/20 为 `0.0988/0.1307` 与 `0.0964/0.1306`。
- **AMLT experiment**：正式主矩阵 `mera-author-three-dataset-formal-r2-20260907` 六项均为 `pass`。Qwen3.5 Home fastlog `mera-author-qwen35-home-fastlog-r1-20260907` 也为 `pass`，且先于主矩阵副本满足完整验收，故按提交前规则作为正式来源。
- **重复运行差异**：Qwen3.5 Home fastlog 与主矩阵合同、revision、seed 和超参数相同，但主矩阵副本的 Macro Accuracy/Recall@10/Recall@20/nDCG@10/nDCG@20 为 `13.76%/0.1855/0.3180/0.0941/0.1273`。四卡 8-bit 训练不是 bitwise deterministic；正式采用先完成的 fastlog，同时保留该差异作为稳定性限制。
- **踩坑**：首次自动下载合同使用 `-I home-assets/manifest.json`，AzCopy 不支持 include pattern 中带目录，因而在两个 GPU job 已通过后退出；改为 basename `-I manifest.json` 后重新下载并通过全部合同断言，再提交正式任务。该失败不涉及训练或模型产物。
- **资源风险**：Qwen3.5/Llama 的完整 `embed_tokens`/`lm_head` adapter 分别约 24.6/12.8 GB。正式 r2 不再向 BlobFuse 写周期 checkpoint，只保留最终 adapter；预检的 checkpoint/reload 门禁和正式任务的五轮完成证明均保留。
- **耗时 / EXIT**：r2 Llama/Qwen3.5 预检分别 19/21 分钟，均 EXIT 0；六项正式主矩阵和 Qwen3.5 Home fastlog 均通过评测并写出完整 `metrics.json`。

## 6. 已知限制 / 未来工作

### 2026-08-31 类别层级、SID 与多 gold 口径复核

- 官方 `category_tree` 的 key 为一级父类，value 并集与 `layer2_map` 完全相等，证明 MeRA 实际预测父类的直接子类。
- Home 路径为 `Home & Kitchen > path[1] > path[2]`：论文一级/二级对应本项目 `target_level_2=path[1]` 与 `target_interest` 的 `path[2]` 子类部分。父子边与预测粒度一致，但因论文不计 domain root，层号名称比本项目 metadata L2/L3 少一层。
- MeRA 与 semantic Interest ID 都是“每个预测子类一个原子 token + 子类名均值初始化”，但并非同一方法：MeRA 历史也使用类别 token，并增加结构/协同关系损失；Interest ID 历史仍为自然语言类别，使用独立的 112 类 `<interest_NNN>` mapping。Home 目标 token 后缀仅 1/112 偶然相同，mapping 不可互换。
- 修正 Home MeRA 初始化：结构身份继续使用 `父 > 子`，embedding 初始化改为只使用子类短名，与论文官方代码一致。重建前三数据集九个关系 `.npy` 哈希逐一不变。
- 多 gold 正式口径改为标准 set Recall、binary-relevance nDCG，并按源 query 等权；不随机抽 gold，也不按展开 gold 行作为最终权重。17 项 pytest 全部通过。

- 双主模型结果主要验收方法适配和跨 baseline 同口径表现；旧 Arts Qwen2 Natural/Balance 只作诊断。新 Qwen2 strict 支线执行固定公开代码与公开资产，但作者本地模型 checksum、原容器 digest 和 trainer state 未公开，因此不声称历史 bitwise replay。
- 官方 Arts/Grocery JSON 无用户 ID，且存在跨 split 完全相同的可观测样本，无法证明 user-disjoint。
- Arts 论文 Validation 为 2,524，发布文件实际为 2,523。
- Hugging Face model revision 会在每个 job 启动时解析为 commit SHA，并在训练/评测间冻结；AMLT 镜像 digest 尚未固定，正式产物必须保留运行环境与合同文件。
- 正式矩阵已按用户要求与 smoke 并行排队，但结果验收仍以 smoke 为门禁：若对应模型 smoke 失败，应停止受影响的正式/消融任务，修复后重跑；smoke 数值不得填入正式结果表。
