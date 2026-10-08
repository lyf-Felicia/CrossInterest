# Amazon 99 类训练数据量实验记录

最近更新：2026-09-01

## 目标与协议

在独立的 99 类 x 4,000 = 396,000 行完整 Train 数据集上，固定 B+I 输入、category-name 输出、冻结 99 类 Dev-2K、1 epoch、LR=1e-4、LoRA 与评测口径，只改变严格嵌套的训练样本数，测量 Qwen3.5-9B 与 Llama-3.1-8B-Instruct 的数据量饱和曲线。

本实验与 `crossinterest.experiments.data_scaling/` 的 112 类 / 112K 实验是两套独立协议：Train、标签空间和 Dev-2K 均不同。结果分别维护，不合并成同一条数据量曲线，也不把相同 Train rows 当作严格控制变量点直接比较。

规模设计为 10K、20K、40K、80K、120K、160K、200K、240K、280K、320K、360K、396K。前四点近似倍增，80K 后采用 40K 等距，396K 是完整数据集自然端点。对规模 $N$ 使用 quotient/remainder 分配：$q=\lfloor N/99\rfloor$、$r=N\bmod99$，恰有 $r$ 类取 $q+1$ 行，其余类取 $q$ 行，因此总行数精确等于表中整数、始终覆盖 99 类且类间差最多 1。所有规模均可被 4 ranks 整除，不发生 DistributedSampler 补样本；除 10K 外均可被 effective batch 32 整除，10K 仅有一个正常的 partial final optimizer step。

完整 Train SHA 为 `aca748bccd79623c02fbbbfda2a384cb72fa0e993f2863029ab96c5f88a49b29`，selection SHA 为 `b05f052597f572813ba04e5e4309ce6f1d5e1c4f50ee87d563584f58591c8b34`。独立 validator 已确认 selection 是 396K source ID 的完整排列，所有规模严格嵌套，selection 标签与 source Train 一致，并逐点满足 99 类 quotient/remainder 平衡。

396K 不重复训练，直接复用 `crossinterest.legacy.training_99/RESULTS.md` 已完成的 B+I/category 结果。需新训练 11 个规模 x 2 模型，共 22 个 job。

## 并发规划

2026-08-31 资源快照：Feeds Standard A100 使用 48/97，用户并发上限 32 GPU；Recall Standard A100 使用 12/33，约有 21 GPU 可用。所有新任务使用 4xA100、per-device batch 1、gradient accumulation 8、effective batch 32。历史 `amazon-sft-qwen-bi-fla-4gpu-b1a8-20260826-021109` 已稳定推进到至少 184/12,375 steps，约 3.2-3.6 秒/step，无 OOM、NCCL error 或 traceback，终止原因为外部 SIGTERM，因此该拓扑可用于 B+I 长视图。

按可用并发容量约 32:20 分配累计训练量：

- Feeds：两模型各跑 10K、40K、80K、160K、200K、280K、360K，共 14 jobs / 2,260,000 行。
- Recall：两模型各跑 20K、120K、240K、320K，共 8 jobs / 1,400,000 行。

训练量比例为 61.75%:38.25%，接近可并发 GPU 比例 61.54%:38.46%。训练完成后 classification 固定 GPU 0、ranking 固定 GPU 1，并行执行并等待两者完成；评测实现和口径不变。

## 纠正记录

曾误将用户提出的 396K 理解为 112 类曲线延伸，并提交 `amazon-datasize-extension-20260831-212502` 的 6 个 112 类 128K/140K/153,216 jobs。2026-08-31 收到澄清后立即取消，机器状态确认 6/6 均为 `killed`，没有继续占用资源；对应扩展 AMLT 配置、构建器、validator、数据目录和 112 类结果表行均已撤销。当前实验是独立的 99 类完整数据集协议。

## 提交状态

2026-09-01 两套配置均通过 `amlt run --dump`，并显式使用 workspace UAI：

`${AZURE_RESOURCE_ID}`

正式实验使用相同代码包 `2362794799b4c6f0990e416ac9a6ed90e8f7f57361dd97539e531e965229ea5d`：

- Feeds：`amazon-datasize99-feeds-20260901-002211`，14 jobs。
- Recall：`amazon-datasize99-recall-20260901-002211`，8 jobs。

提交后机器状态验证确认 22 个唯一 job 完整覆盖两模型 x 11 个新规模，全部使用 `80G4-A100`、Standard SLA、4 ranks、per-device batch 1、gradient accumulation 8，selection size 与 job 名一致，且 `status_message` 均为空。验证快照中 Feeds 为 14 `queued`；Recall 为 7 `queued`、1 `running`。

首个运行时门禁已在 Recall Qwen 240K job `<archived-job-id>` 上通过。定向下载并解析 `train/train_contract.json` 与 `train/runtime_config.json` 后确认：Train SHA 和 selection SHA 均精确匹配本实验冻结值；`selection_size=selected_train_rows=240000`；99 类计数范围为 2,424-2,425；B+I/category、candidate-free、assistant-only 与 effective batch 32 正确；`world_size=4`、`dataloader_drop_last=false`；Qwen fast path 可用且四个 FLA / causal-conv symbol 均为 `true`。stdout 已进入 7,500-step 训练循环，四个 NCCL rank 完成 ring 连接，检查时未发现 traceback、CUDA OOM 或 NCCL error。

Feeds job IDs：

- Qwen 10K `<archived-job-id>`；40K `<archived-job-id>`；80K `<archived-job-id>`；160K `<archived-job-id>`；200K `<archived-job-id>`；280K `<archived-job-id>`；360K `<archived-job-id>`。
- Llama 10K `<archived-job-id>`；40K `<archived-job-id>`；80K `<archived-job-id>`；160K `<archived-job-id>`；200K `<archived-job-id>`；280K `<archived-job-id>`；360K `<archived-job-id>`。

Recall job IDs：

- Qwen 20K `<archived-job-id>`；120K `<archived-job-id>`；240K `<archived-job-id>`；320K `<archived-job-id>`。
- Llama 20K `<archived-job-id>`；120K `<archived-job-id>`；240K `<archived-job-id>`；320K `<archived-job-id>`。

## 2026-09-01 进度与阶段结果

机器状态快照：Feeds 14 jobs 为 3 `pass`、6 `running`、3 `queued`、1 `paused`、1 `failed`；Recall 8 jobs 为 2 `pass`、2 `running`、4 `failed`。合计 5/22 个新任务完整通过，8 个运行中，3 个排队，1 个暂停，5 个失败。加上复用的两个 396K 端点，当前 99 类主表已有 7/24 行完整结果。

已通过任务的 classification/ranking `metrics.json` 均验证：2,000/2,000 样本、99 类、B+I/category、`complete=true`、无 missing sample，数值已写入 `RESULTS.md`：

| 模型 | Train | Macro Accuracy (%) ↑ | Recall@10 ↑ | Recall@20 ↑ | nDCG@10 ↑ | nDCG@20 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 200K | 17.92 | 0.2057 | 0.3720 | 0.1063 | 0.1478 |
| Qwen3.5-9B | 240K | 18.26 | 0.2196 | 0.3887 | 0.1144 | 0.1566 |
| Llama-3.1-8B-Instruct | 20K | 12.73 | 0.1813 | 0.3332 | 0.0885 | 0.1265 |
| Llama-3.1-8B-Instruct | 40K | 10.50 | 0.1860 | 0.3561 | 0.0934 | 0.1360 |
| Llama-3.1-8B-Instruct | 80K | 16.01 | 0.1991 | 0.3602 | 0.1009 | 0.1410 |

失败任务未将部分结果写入主表。审计五个失败 job 后，均没有同时完成 classification 与 ranking：Llama 240K 和 Qwen 120K 仅有 classification metrics，其余三项无 metrics。已确认两类恢复问题：Llama 10K、Qwen 20K、Qwen 120K 在重新进入评测时触发 `existing run contract differs`；Llama 120K 恢复点缺少 `checkpoint-1000/trainer_state.json`；Llama 240K 的 classification 已完成但 ranking 未完成。以上属于恢复/输出状态冲突，不是已完成指标的模型性能失败。

运行健康度抽查：Qwen 10K 与 160K 已进入 classification/ranking 双评测；Qwen 320K 已到 8,287/10,000 steps，Qwen 360K 已到 4,265/11,250 steps，未见 traceback、CUDA OOM 或 NCCL error。后续仅在 classification 与 ranking 都完整且合同匹配后回填剩余单元格。

## 2026-09-01 恢复修复与继续推进

状态刷新后新增两个完整结果，classification/ranking 均为 2,000/2,000、99 类、`complete=true` 且无 missing sample：Qwen 80K 为 11.76% / 0.2114 / 0.3729 / 0.1070 / 0.1475；Llama 320K 为 17.04% / 0.2168 / 0.3927 / 0.1137 / 0.1578。至此 7/22 个新任务完整通过，加上两个复用的 396K 端点，主表完成 9/24 行。

对五个失败任务做了根因修复：

- `train.py` 的 checkpoint 选择器不再只取编号最大的目录，而是要求 adapter config/weights、optimizer、scheduler 和 `trainer_state.json` 全部存在，避免恢复到尚未同步完整的 checkpoint。Llama 120K 因而会从完整的 `checkpoint-500` 恢复，而不是缺少 `trainer_state.json` 的 `checkpoint-1000`。
- classification/ranking 合同使用排序后的 adapter config 语义哈希，并新增 adapter 权重 SHA；仅 `target_modules` 序列化顺序变化不会再制造假失配，真实权重变化仍会触发合同变化。
- `run_train_eval.sh` 显式启用合同失配重置：旧合同、details、metrics 和 predictions 先归档到 `stale/<old-fingerprint>/`，再从头评测新 adapter，避免混用不同权重的部分明细。

行为测试验证了完整 checkpoint 回退、两类评测默认拒绝合同失配、显式重置归档、adapter 配置规范化和权重 SHA 绑定；Python 编译、shell 语法和编辑器诊断均通过。

AMLT 初次上传修复时因 `--code-dir .` 绕过原 YAML 的 `prep/` ignore，将 38GB 原始 artifacts 纳入快照并填满 `/tmp`，提交未发生。清理该失败临时目录后，构建了仅含正式 Train、99 类训练/评测代码、Dev 数据和 selection 的 448.5MB 快照；Train/selection SHA 再次验证正确。修复代码包为 `11dad16352e86f2d3cfc0525ec00d59191620bc607bb2eb72967121a31442b64`。

五个修复重跑均已被后端接受，旧训练产物保留用于断点恢复，新 job IDs 为：

- Feeds Llama 10K：`<archived-job-id>`。
- Recall Qwen 20K：`<archived-job-id>`；Qwen 120K：`<archived-job-id>`。
- Recall Llama 120K：`<archived-job-id>`；Llama 240K：`<archived-job-id>`。

提交后 22 个当前 job 仍保持两模型 x 11 规模无重无漏，五个重跑均使用 `80G4-A100`、Standard SLA、正确 UAI 和新代码包。原暂停的 Llama 200K 后端已恢复为 `running`，无需另行重跑。

中断后继续检查的最新状态为 7 `pass`、10 `running`、5 `queued`，22 个当前 job 均无 `status_message`，也没有 `failed`、`killed` 或 `expired` 终态。四个 Recall 修复重跑均已获得资源：

- Qwen 20K 从完整 `checkpoint-625` 恢复，Qwen 120K 从完整 `checkpoint-3750` 恢复，两者均已重新进入 classification/ranking 双评测。
- Llama 240K 从完整 `checkpoint-7500` 恢复，旧 classification/ranking 的合同与部分产物已按原 fingerprint 归档至各自 `stale/` 子目录，新双评测从零写入。
- Llama 120K 的 `checkpoint-1000` 只有 adapter 权重而没有 `trainer_state.json`，`checkpoint-500` 同时具备两者；完整 checkpoint 选择器因此跳过 1000 并从 500 恢复，已继续推进到至少 671/3,750 steps。
- Feeds Llama 10K 修复重跑仍为 `queued`，其余 5 个排队任务等待 Feeds 容量。

Qwen 20K、Qwen 120K 和 Llama 240K 的结果目录均已确认存在 `stale/<old-fingerprint>/` 归档及新的 `details.jsonl`，证明合同失配重置没有混用旧权重的评测明细。运行抽查时，Qwen 40K classification 已到 1,580/2,000，Qwen 160K 双评测约为 1,120/720，Qwen 320K 训练到 8,951/10,000，Qwen 360K 到 4,915/11,250，Llama 200K 到 582/6,250，Llama 360K 到 6,971/11,250；未见 traceback、CUDA OOM 或 NCCL error。

最新容量快照中 Feeds Standard A100 为 92/97、Recall 为 28/33，两边均只剩约一组四卡容量。为减少长任务尾部，将尚未启动的 Feeds Qwen 280K 原位重跑到 Recall；第一次因未显式指定 workspace 而非零退出，机器状态确认原 job 未变。补充 `--ws ${AZURE_CONFIGURATION}` 后提交成功，新 job ID 为 `<archived-job-id>`，目标 Recall、Standard `80G4-A100`、UAI 与修复代码包 `11dad16352e86f2d3cfc0525ec00d59191620bc607bb2eb72967121a31442b64` 均已验证。迁移后 22 个模型 x 规模点仍唯一完整，物理分布为 Feeds 13 jobs、Recall 9 jobs；当前 Qwen 280K 为 `queued`，没有重复训练实例。
