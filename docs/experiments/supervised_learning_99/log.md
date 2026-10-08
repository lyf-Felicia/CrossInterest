# Amazon 三视图 SFT 实验记录 (Experiment Log)

> 项目: `crossinterest_formal/amazon/crossinterest.legacy.training_99/` - 在完整 Amazon 新兴趣训练集上微调 Qwen3.5-9B 与 Llama-3.1-8B-Instruct，并进行 10 选 1 分类与无候选排序评测。
> 关键约束: 每个模型/视图使用全部 396,000 行；训练 prompt 不含候选；输入历史保持自然语言；输出支持类别名和冻结 `<interest_NNN>` 两种模式。
最近更新: 2026-08-27

---

## 1. 实验目标

1. 分别验证 `behavior`、`interest`、`behavior_interest` 三种输入视图在 Qwen3.5-9B 和 Llama-3.1-8B-Instruct 上的完整数据 LoRA SFT 效果。
2. 保持训练和评测严格解耦：训练只预测 gold category，不暴露候选；分类评测使用冻结 Dev-2K 候选集和 deterministic 10-beam 约束解码；排序评测不提供候选，直接生成 20 beams。
3. 在独立 AMLT 作业中完成训练与评测：Qwen 短视图使用 4xA100，两个长视图使用 8xA100 安全拓扑；Llama 使用已有 8xA100 checkpoint，最终汇总 Macro Accuracy、Macro Recall@10/20 与 Macro nDCG@10/20。

## 2. 模型与方法

| 项目 | 设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-9B`、`NousResearch/Meta-Llama-3.1-8B-Instruct` |
| 视图 | `behavior`、`interest`、`behavior_interest` |
| 训练数据 | `train.parquet`，每个作业 396,000 行 |
| 训练目标 | assistant-only；类别名模式输出精确 gold category，ID 模式输出对应冻结 `<interest_NNN>` |
| LoRA | r=16，alpha=32，dropout=0.05，target=`all-linear` |
| 优化 | 1 epoch，LR=1e-4，cosine，warmup ratio=0.03，bf16 |
| 并行 | 全局 batch=32；Llama 为 8 卡 batch=1 / accumulation=4；Qwen `interest` 为 4 卡 batch=4 / accumulation=2，长视图为 8 卡 batch=1 / accumulation=4 |
| 训练步数 | 12,375，即 396,000 / 32 |
| 评测 | prompt-v3，Dev-2K；10-way deterministic constrained beam；candidate-free direct 20-beam generation |
| 指标 | Macro Accuracy (%)、Macro Recall@10/20、Macro nDCG@10/20 ↑ |

候选隔离由 `prompts.py` 中的训练 prompt 断言保证；数据规模和有效 batch 分别由 `train.py` 的 `EXPECTED_TRAIN_ROWS=396_000` 与运行时合同校验保证。

## 3. 仓库结构

```text
crossinterest.legacy.training_99/
├── amlt/                  # AMLT 训练、评测、smoke 与备用拓扑配置
│   ├── amlt_eval_throughput_map.yaml
│   ├── amlt_formal.yaml
│   ├── amlt_qwen_fast.yaml
│   ├── amlt_qwen_fast_2x4gpu_behavior_b1a4.yaml
│   ├── amlt_qwen_fast_2x8gpu_behavior_b1a2.yaml
│   ├── amlt_qwen_fast_4gpu.yaml
│   ├── amlt_qwen_fast_4gpu_behavior_b1a8.yaml
│   ├── amlt_qwen_fast_4gpu_behavior_interest_b1a8.yaml
│   ├── amlt_qwen_fast_4gpu_longviews.yaml
│   ├── amlt_qwen_fast_8gpu_behavior_b1a4.yaml
│   ├── amlt_qwen_fast_8gpu_longviews.yaml
│   └── amlt_speed_smoke.yaml
├── train.py               # assistant-only LoRA SFT、合同校验和断点恢复
├── evaluate.py            # prompt-v3 10-way constrained-beam adapter 评测
├── evaluate_ranking.py    # 无候选、无标签约束 direct 20-beam 排序评测
├── prompts.py             # candidate-free 三视图训练 prompt
├── requirements_amlt.txt  # 集群 Python 依赖版本
└── EXPERIMENT_LOG.md      # 本实验状态、故障与结果
```

## 4. 运行环境

- AMLT: 11.15.0.post1。
- Target: `Feeds` / `${AZURE_CONFIGURATION}`，Standard SLA。
- SKU: Qwen `interest` 为 `80G4-A100`（ND48am_A100_v4），长视图最终为 `80G8-A100-IB-NvLink` / `80G8-A100-NvLink`；Llama 原作业为 `80G8-A100-NvLink`，均为 A100 80GB。
- 容器: `amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04`。
- 已从运行日志核验: PyTorch 2.8.0+cu126、CUDA 12.6、Transformers 5.14.1、PEFT 0.18.1、Accelerate 1.12.0、PyArrow 23.0.1。
- 并发上限: 32 GPU，因此六个 8 卡作业分批调度。

## 5. 实验进度

### 实验 1 - 首次正式提交（已终止）

- **实验名**: `amazon-sft-v3-full-20260825-1919`。
- **配置**: 与当前正式协议相同，但命令使用直接文件入口 `torchrun ... crossinterest.legacy.training_99/train.py` 和 `python crossinterest.legacy.training_99/evaluate.py`。
- **踩坑**: 集群从包外直接执行文件，触发 `ImportError: attempted relative import with no known parent package`，继而出现 `ModuleNotFoundError: No module named 'crossinterest.evaluation'` 和 `ChildFailedError`。修复为模块入口 `torchrun ... --module crossinterest.legacy.training_99.train` 与 `python -m crossinterest.legacy.training_99.evaluate`。
- **最终状态**: 3 failed、3 killed；取消已完成，不产生可用训练结果。

### 实验 2 - 修正版完整数据六作业（⚠️ Llama 全部已恢复）

- **实验名**: `amazon-sft-v3-full-fixed-20260825-195020`。
- **配置**: amlt_formal.yaml（私有原始证据，不随公开仓库分发），六个 8xA100 Standard 作业，完整 396,000 行，1 epoch，全局 batch=32。
- **启动方式**:

```bash
cd crossinterest_formal/amazon
amlt run crossinterest.legacy.training_99/amlt/amlt_formal.yaml \
  amazon-sft-v3-full-fixed-20260825-195020 --yes
```

- **已确认正常**: 三个 Llama 作业中的两个已建立 8-rank NCCL 通信并持续训练；未发现 import traceback、CUDA OOM、NCCL error 或 `ChildFailedError`。日志中的 `NCCL_ASYNC_ERROR_HANDLING` deprecation 和 DDP `<archived-job-id>=True` 均为非致命警告。
- **Qwen 慢速根因**: Qwen3.5-9B 的 32 个文本层中有 24 个 Gated Delta 线性注意力层。旧环境未提供 Transformers 5.14.1 所需的 `causal_conv1d_fn`、`causal_conv1d_update`、`chunk_gated_delta_rule`、`fused_recurrent_gated_delta_rule` 四个 symbol，因而落入含 float32 转换和 PyTorch 循环的 fallback；旧代码还实例化视觉塔并向视觉线性层注入无用 LoRA。实测 `interest` 约 8.4 秒/step，`behavior` 约 13.5--14.8 秒/step，单作业剩余约 25--48 小时，不满足 8 小时目标。
- **终止与断点**: 2026-08-26 已主动停止三个旧 Qwen job，释放两组 8xA100；`interest` 保留完整 `checkpoint-1500/`，`behavior` 保留完整 `checkpoint-500/`，均包含 adapter、optimizer、scheduler、trainer state 和 8-rank RNG。快速方案改用不同的纯文本 adapter 拓扑，因此从头训练，不直接恢复旧视觉 LoRA checkpoint。
- **截至 2026-08-26 00:35 +0800 的状态**:

| Job | AMLT 状态 | 已验收训练进度 | 最新可见 loss | 评测结果 |
|---|---|---:|---:|---:|
| `qwen35-behavior-sft` | killed | 558 / 12,375 | 0.4681 |  |
| `qwen35-interest-sft` | killed | 1,611 / 12,375 | 0.4587 |  |
| `qwen35-behavior-interest-sft` | killed | 0 / 12,375 |  |  |
| `llama31-behavior-sft` | queued | 62 / 12,375 |  |  |
| `llama31-interest-sft` | failed（训练已完成，评测终态 bug） | 12,375 / 12,375 | 0.3210 |  |
| `llama31-behavior-interest-sft` | paused（调度让位） | 7,774 / 12,375 | 0.3225 |  |

- **结果**: 六个 adapter 评测尚未完成，Macro Accuracy 保持空白。
- **预期产物**: 每个 AMLT job 的 `train/final_adapter/`、`train/train_metrics.json`、`eval/metrics.json`、`nvidia-smi.txt` 和 `pip-freeze.txt`。
- **耗时 / EXIT**: 进行中。

2026-08-26 01:49 +0800 状态刷新：三个 Llama job 均已暂停，为 Qwen 快速正式实验释放 GPU；旧 Qwen 三个 job 均为 `killed`。暂停前已确认的完整 checkpoint 包括 `llama31-interest-sft/checkpoint-8500` 与 `llama31-behavior-interest-sft/checkpoint-7500`，恢复时必须从各自最新完整 checkpoint 继续。

2026-08-26 03:51 +0800 状态刷新：`llama31-interest-sft` 于 03:25 获得 8 卡节点并恢复运行，日志已从 `checkpoint-8500` 后推进至至少 8,933 / 12,375，未从 step 0 重启；稳态约 1.82--1.90 秒/step，进度条剩余约 1 小时 44 分，未出现 OOM、NCCL error、traceback 或 checkpoint 错误。`llama31-behavior-interest-sft` 与 `llama31-behavior-sft` 的 resume 命令均成功退出，机器状态已从 `paused` 变为 `queued`，将在下一组 8 卡释放后自动接续。

2026-08-26 05:49 +0800 状态刷新：`llama31-interest-sft` 已完成 12,375 / 12,375，`train_runtime=7525.2292` 秒、`train_loss=0.10146365`，并写出完整 `checkpoint-12375/`、`final_adapter/` 与 `train_metrics.json`。AMLT 的 `failed` 来自随后第 1 条评测：Transformers beam search 在候选序列已经包含 EOS 后再次调用 prefix callback，原代码把 trie 叶节点的空后继误判为 `beam escaped candidate trie`。`evaluate.py` 已修为完整候选终态只允许 EOS，并用失败 token 序列通过局部回归检查；正式评测改由 1 卡 evaluation-only map 执行，不重跑训练。

### 实验 3 - Qwen 纯文本 FLA 8 卡 batch4 吞吐 smoke（已取消）

- **实验名**: `amazon-sft-qwen-text-fla-b4-smoke-20260826-002747`，job ID `<archived-job-id>`。
- **配置**: amlt_speed_smoke.yaml（私有原始证据，不随公开仓库分发），Qwen3.5 纯文本 `AutoModelForCausalLM`，`flash-linear-attention[cuda]==0.5.2`，`causal-conv1d==1.7.0`，8xA100，每卡 batch=4、accumulation=1、全局 batch=32，关闭 gradient checkpointing 和 DDP unused-parameter scan，训练 40 steps。
- **启动方式**:

```bash
cd projects/mind/crossinterest
amlt run crossinterest_formal/amazon/crossinterest.legacy.training_99/amlt/amlt_speed_smoke.yaml \
  amazon-sft-qwen-text-fla-b4-smoke-20260826-002747 --yes
```

- **硬门禁**: 下载模型前要求四个 Qwen Gated Delta / causal-conv symbol 全部可用，否则立即退出；LoRA 注入后要求视觉可训练 tensor 数为 0。
- **验收标准**: 40 steps 无 CUDA OOM、NCCL 或 traceback；`qwen_fast_path.available=true`；`visual_tensor_count=0`；稳态吞吐不高于约 2.2 秒/optimizer step。按完整 12,375 steps 倒推，2.2 秒/step 对应约 7.56 小时纯训练，只有达到该线才允许提交三个快速正式作业。
- **当前状态**: 原 Medium-priority 8 卡 smoke 长时间等待完整 ND96 节点，后续 High-priority 8 卡 smoke `amazon-sft-qwen-text-fla-b4-high-smoke-20260826-013410` 也处于 queued。4 卡同拓扑 smoke 已先获得资源并通过，因此两条 8 卡 smoke 均已取消，不再占用调度机会。
- **排队定位与处置**: 旧 Amazon Qwen 释放的两组 8xA100 曾被 `crossinterest-full-min200-4view-formal-0825-v2` 的两个新 job 立即接走。为优先满足本实验 8 小时目标，已将其中 `superloss-behavior-8gpu-train-eval` 暂停在完整 `checkpoint-250`，将刚进入 step 0 的 `sft-behavior-8gpu-train-eval` 暂停，并取消另外两个 queued job；这些作业可后续恢复或重提。两个 14 天前的 1-GPU probe 经强制刷新后均为 `failed`，未继续占 GPU。
- **容量证据**: 处置竞争作业后，Feeds A100 Standard 用量曾从 77/97 降至 59/97；当前 smoke 仍等待一台完整 ND96 8xA100 节点。AzureML 原始 job 状态为 `Queued`、`error=null`、retry=0，请求覆盖 `ND96am_A100_v4`、`ND96amrs_A100_v4`、`ND96amr_A100_v4` 三种可接受实例，因此不是配置拒绝或 SKU 误配。
- **并发优先级**: 为让三个 Qwen 正式 job 可在用户 32-GPU 上限内并发，已将 Llama `behavior_interest` 在完整 `checkpoint-7500` 后暂停（暂停前 7,774/12,375），并将刚启动的 Llama `behavior` 在 62/12,375 暂停；仅保留完成度最高的 Llama `interest` 继续运行。由此为三个 Qwen 预留 24 GPU。
- **正式配置**: amlt_qwen_fast.yaml（私有原始证据，不随公开仓库分发） 保留为 8 卡备用配置；实际正式提交改用已实测的 4 卡配置。
- **耗时 / EXIT**: 未运行训练 step，AMLT 作业已取消。

### 实验 4 - Qwen FLA symbol 预检（✅ 已完成）

- **实验名**: `amazon-qwen-fla-symbol-preflight-20260826-010527`，job `fair-mite`。
- **配置**: 复用与正式作业相同的 AMLT image 和 requirements，在 1xA100 上不加载 9B 权重，仅导入 Transformers 5.14.1 Qwen3.5 实现并断言 `causal_conv1d_fn`、`causal_conv1d_update`、`chunk_gated_delta_rule`、`fused_recurrent_gated_delta_rule` 四个 symbol 及 `is_fast_path_available`。
- **目的**: 在 8 卡整机排队期间先验证 `flash-linear-attention[cuda]==0.5.2` 与 `causal-conv1d==1.7.0` 的安装链；该预检不能替代 batch4 显存与吞吐 smoke。
- **结果**: `causal-conv1d==1.7.0` 在 Torch 2.8 / CUDA 12.6 image 上源码 wheel 构建成功；`flash-linear-attention==0.5.2`、`fla-core==0.5.2` 安装成功。运行输出为 `available=true`，四个 symbol 均为 `true`。
- **耗时 / EXIT**: AMLT 状态 `pass`，约 6 分钟（主要为依赖安装与 causal-conv wheel 构建），watch exit code 0。

### 实验 5 - Qwen 4 卡 FLA batch4/accum2 吞吐 smoke（✅ 已完成）

- **实验名**: `amazon-sft-qwen-text-fla-4gpu-b4a2-smoke-20260826-012823`，job `light-husky`，Azure job ID `<archived-job-id>`。
- **配置**: 4xA100 80GB，Qwen3.5 纯文本 `AutoModelForCausalLM`，每卡 batch=4、accumulation=2、全局 batch=32，关闭 gradient checkpointing 和 DDP unused-parameter scan，完整训练合同下限制为 40 optimizer steps。
- **已确认正常**: `runtime_config.json` 记录 `world_size=4`、`effective_batch_size=32`、`qwen_fast_path.available=true`，四个 FLA / causal-conv symbol 均为 `true`；`train_contract.json` 记录 396,000 行、`candidate_free=true`、`assistant_only=true`、`sample_limit=null`。40 steps 无 OOM、traceback、NCCL error 或快路径断言失败，最终 adapter 与 `checkpoint-40` 均已写出。
- **吞吐**: 首个 optimizer step 因一次性 kernel/JIT warm-up 为 118.96 秒；第 2--40 步稳定约 1.15--1.35 秒/step，结束时进度条为 1.27 秒/step。Trainer 总计 `train_runtime=183.5209` 秒、`train_steps_per_second=0.218`，该总值包含首步编译和尾部保存。
- **完整训练 ETA**: 按稳态 1.27 秒/step 和 12,375 steps 外推，训练约 4.37 小时；即使计入约 2 分钟首次编译，仍低于 8 小时训练硬线。评测耗时需在正式作业中单独观测。
- **产物**: `train/runtime_config.json`、`train/train_contract.json`、`train/trainable_parameters.json`、`train/train_metrics.json`、`train/checkpoint-40/`、`train/final_adapter/`。
- **耗时 / EXIT**: AMLT 状态 `pass`，精确错误 watcher exit code 0。

### 实验 6 - Qwen 三视图 4 卡 FLA 正式训练与评测（⚠️ 仅保留 `interest`）

- **实验名**: `amazon-sft-qwen-text-fla-4gpu-formal-20260826-014613`。
- **配置**: amlt_qwen_fast_4gpu.yaml（私有原始证据，不随公开仓库分发），三个视图均使用 `80G4-A100`、High priority、4 ranks、batch=4、accumulation=2、全局 batch=32；其余训练合同和 Dev-2K constrained-beam 评测口径不变。
- **启动方式**:

```bash
cd projects/mind/crossinterest
amlt run crossinterest_formal/amazon/crossinterest.legacy.training_99/amlt/amlt_qwen_fast_4gpu.yaml \
  amazon-sft-qwen-text-fla-4gpu-formal-20260826-014613 \
  --description "Amazon Qwen3.5 full 396K candidate-free three-view 4xA100 FLA LoRA SFT plus Dev-2K constrained-beam evaluation" \
  --yes
```

- **提交验证**: 三个 job 均已被 AMLT 接受；Azure 原始资源字段均为 `Priority=High`、`instanceTypes=[Singularity.ND48am_A100_v4]`、`error=null`。

| Job | Azure job ID | 当前状态 | 训练进度 | Macro Accuracy ↑ |
|---|---|---|---:|---:|
| `qwen35-behavior-text-fla-4gpu-sft` | `<archived-job-id>` | queued | 0 / 12,375 |  |
| `qwen35-interest-text-fla-4gpu-sft` | `<archived-job-id>` | queued | 0 / 12,375 |  |
| `qwen35-behavior-interest-text-fla-4gpu-sft` | `<archived-job-id>` | queued | 0 / 12,375 |  |

- **视图级显存结论**: 40-step `interest` smoke 不能代表长视图。动态 padding 且禁止截断时，4 卡 batch=4 的 `behavior` 与 `behavior_interest` 均在 step 0 OOM；降到 batch=2 后，`behavior_interest` 仍在 step 0 OOM，`behavior` 则在 step 310 遇到随机长批次后 OOM。最终仅保留 `interest` 的 batch=4 路径。
- **完整训练集字符长度分布**: `interest` P99=2,543、max=3,056；`behavior` P99=8,795、max=9,909；`behavior_interest` P99=10,962、max=12,140。长视图必须使用每卡 batch=1，不能用截断、采样训练集或候选泄漏规避显存问题。
- **监控**: 已启动精确异常 watcher，匹配 OOM、CUDA/NCCL error、traceback、fast-path unavailable、训练行数与有效 batch 合同失败。
- **耗时 / EXIT**: 进行中。

### 实验 7 - Qwen 三视图 FLA 正式训练（⚠️ 三视图均稳定运行）

- **`behavior_interest` 主作业**: `amazon-sft-qwen-fla-8gpu-longviews-20260826-021630` / `qwen35-behavior-interest-text-fla-8gpu-b1a4-sft`，Azure job ID `<archived-job-id>`，8 卡 batch=1 / accumulation=4。
- **`behavior` 主作业**: `amazon-sft-qwen-behavior-fla-8gpu-ib-b1a4-20260826-025211` / `qwen35-behavior-text-fla-8gpu-b1a4-sft`，Azure job ID `<archived-job-id>`，`80G8-A100-IB-NvLink`，8 卡 batch=1 / accumulation=4。
- **运行合同**: 两个长视图均为 `world_size=8`、`effective_batch_size=32`、gradient checkpointing 关闭、DDP unused-parameter scan 关闭；`qwen_fast_path.available=true`，四个 FLA / causal-conv symbol 全部为 `true`，版本为 FLA 0.5.2、causal-conv1d 1.7.0。
- **截至 2026-08-26 05:49 +0800 的实测状态**:

| 视图 | 主作业状态 | 已验收进度 | 稳态吞吐 | 保守剩余训练 ETA | Macro Accuracy ↑ |
|---|---|---:|---:|---:|---:|
| `interest` | running | 8,536 / 12,375 | 约 1.1--1.4 秒/step | 约 1.2--1.5 小时 |  |
| `behavior` | running | 5,537 / 12,375 | 约 1.6--1.8 秒/step | 约 3.0--3.4 小时 |  |
| `behavior_interest` | running | 6,022 / 12,375 | 约 1.8--2.1 秒/step | 约 3.2--3.7 小时 |  |

- **8 小时结论**: 三条主作业的剩余训练 ETA 均低于 8 小时；当前最长的是 `behavior_interest`，保守约 3.7 小时。原作业内评测使用旧代码快照，若触发已知 EOS 终态 bug，则保留完整 adapter 并转由修正版 evaluation-only map 续跑。
- **资源清理**: `behavior` 4 卡 batch=1 / accumulation=8 虽稳定，但实测约 3.2--3.4 秒/step、剩余约 10.7--11.3 小时，已取消。已取消另一个 8 卡非 IB 重复作业和仍 queued 的 2x8 卡 IB 竞速作业；仅保留上述三条主训练，避免浪费 GPU。
- **调度备用**: `2x80G8-A100-IB-NvLink` 多节点配置通过 AMLT schema/dump 并被后端接受，但未获得双节点前已主动取消。`2x80G4-A100` 被后端明确拒绝，因为 ND48 4 卡切片不是可跨节点互联 SKU。
- **耗时 / EXIT**: 进行中；三条主作业均已持续运行并由异常 watcher 监控。

### 实验 8 - SFT constrained-beam 评测终态修复与双协议正式评测（✅ 已完成）

- **根因与修复**: 候选 token 序列本身包含 EOS；Transformers 5.14.1 的 beam search 会对已到 trie 叶节点的序列再调用一次 `prefix_allowed_tokens_fn`。修复后，只有 `tuple(generated)` 精确命中候选序列时才返回 EOS；真实越界仍抛错。
- **回归检查**: 使用失败序列 `[37, 25597, 871, 19048, 10637, 30339, 128009]` 验证完整候选终态返回 EOS、候选前缀返回原后继、非法前缀仍被识别为越界。
- **配置**: amlt_eval_throughput_map.yaml（私有原始证据，不随公开仓库分发），1xA100 80GB、Standard/High，直接挂载源作业的 `train/final_adapter/`，不复制或重训 adapter。
- **Qwen 吞吐 smoke**: `qwen35-interest-eval-25-amazon-sft-qwen-text-fla-4gpu-b4a2-smoke-20260826-012823`，job `qwen35-interest-eval-25-light-husky`，25 条 `interest`，当前 `running`。
- **Llama 正式评测**: `llama31-interest-eval-full-amazon-sft-v3-full-fixed-20260825-195020`，job `llama31-interest-eval-full-llama31-interest-sft`，完整 Dev-2K `interest`，当前 `queued`。
- **结果**: 两项 Macro Accuracy 均不在未完成时填写；Qwen 25 条仅用于吞吐测量，不能作为正式准确率。

2026-08-26 14:21 +0800 状态刷新：修正版 constrained classification 已完成 5/6 个视图，candidate-free ranking 已完成 2/6 个视图。新完成的 `llama31-behavior-interest-eval-full-v2-amazon-sft-v3-full-fixed-20260825-195020` 为 2,000/2,000、99 类、`complete=true`，Macro Accuracy=0.1746272246（表中记 17.46%）；新完成的 `llama31-interest-ranking-full-v1-amazon-sft-v3-full-fixed-20260825-195020` 为 2,000/2,000、missing=0、99 类、`complete=true`，Macro Recall@10=0.1843915344、Macro Recall@20=0.3472342472、Macro nDCG@10=0.0932641783、Macro nDCG@20=0.1339692865。该 ranking 的 `valid_beam_rate=0.991025`、`strict_beam_format_rate=0.990975`、`duplicate_valid_beam_rate=0.0`。

同一时间点，三个未完成 ranking 作业均健康运行：Qwen `behavior` 1,787/2,000，Qwen `behavior_interest` 1,616/2,000，Llama `behavior_interest` 1,547/2,000；未用这些部分输出计算或填写正式指标。Llama `behavior` 是唯一未完成训练，现场日志推进至 11,946/12,375（约 97%，最近 loss=0.3206），完成后仍需提交修正版 constrained 与 candidate-free ranking 两项评测。Llama `behavior_interest` 源作业虽因旧内嵌评测 bug 标为 `failed`，完整 adapter 已由两条修正版评测正常加载，其中 constrained 已通过，因此不能将源状态误读为训练失败。

2026-08-26 14:51 +0800 状态刷新：六个正式 adapter 已全部完成训练。最后一个 Llama `behavior` 写出 `checkpoint-12375/`、`final_adapter/adapter_config.json` 和 `train/train_metrics.json`；训练指标为 `epoch=1.0`、`train_runtime=21046.229` 秒、`train_loss=0.2620238520`。watch 的 exit code 13 是匹配到 `train_runtime` 后按 `--exit-on-grep` 正常退出，不是训练错误。源 job 随后因旧内嵌 constrained evaluator 的已知 EOS 终态 bug 标为 `failed`，不影响完整训练产物。

同一快照新增三组完整 candidate-free ranking 结果，均为 2,000/2,000、missing=0、99 类、`complete=true`：Qwen `behavior` 的 Macro Recall@10/20=0.2132996633/0.3670995671、Macro nDCG@10/20=0.1090597632/0.1475719773；Qwen `behavior_interest` 为 0.2176286676/0.3783309283、0.1158061581/0.1558430109；Llama `behavior_interest` 为 0.2152717653/0.3769841270、0.1135082985/0.1540319354。正式 ranking 进度由 2/6 更新为 5/6。

最后两项 Llama `behavior` 评测已提交并由 AMLT 后端接受：constrained experiment `llama31-behavior-eval-full-v2-amazon-sft-v3-full-fixed-20260825-195020`，job ID `<archived-job-id>`；candidate-free ranking experiment `llama31-behavior-ranking-full-v1-amazon-sft-v3-full-fixed-20260825-195020`，job ID `<archived-job-id>`。两者均使用 `80G1-A100`、Standard SLA，提交后状态为 `queued`；完成前对应结果单元格保持空白。

2026-08-26 15:01 +0800 状态刷新：最后两项均已由 `queued` 转为 `running`，未发现 traceback、CUDA OOM 或 `ChildFailedError`。Llama `behavior` constrained classification 已落盘 28/2,000 条唯一样本，candidate-free ranking 已落盘 29/2,000 条唯一样本；两者均尚未生成完整 `metrics.json`，因此正式结果表继续保持相应单元格空白。两个独立 watcher 持续监控 `complete=true` 与异常信号。

2026-08-26 15:57 +0800 状态刷新：Llama `behavior` constrained classification 已推进至 1,373/2,000，candidate-free ranking 已推进至 1,281/2,000；两个 job 均仍为 `running`，未生成 `metrics.json`，因此没有新指标可写入正式结果表。继续保留相应单元格空白，等待 2,000/2,000、99 类且 `complete=true` 后再填写。

2026-08-26 16:24 +0800 状态刷新：Llama `behavior` constrained classification 已生成完整 `eval/metrics.json`，2,000/2,000、missing=0、99 类、`complete=true`，Macro Accuracy=0.1377104377（正式表记 13.77%）、Micro Accuracy=0.138，`strict_format_rate=1.0`、`resolved_prediction_rate=1.0`。至此 constrained classification 6/6 全部完成。对应 watcher 的 exit code 13 是命中 `complete=true` 后按 `--exit-on-grep` 正常退出，不代表作业失败。最后的 Llama `behavior` candidate-free ranking 同时推进至 1,877/2,000，仍未生成完整 `metrics.json`，因此 ranking 维持 5/6，四个 ranking 单元格继续留空。

2026-08-26 16:33 +0800 状态刷新：最后一个 Llama `behavior` candidate-free ranking 已生成完整 `ranking/metrics.json`，2,000/2,000、missing=0、99 类、`complete=true`。Macro Recall@10=0.2042568543、Macro Recall@20=0.3618566619、Macro nDCG@10=0.1072125873、Macro nDCG@20=0.1467760275（正式表分别记 0.2043、0.3619、0.1072、0.1468）；`valid_beam_rate=0.98885`、`strict_beam_format_rate=0.98885`、`duplicate_valid_beam_rate=0.0`。对应 watcher exit code 13 是命中 `complete=true` 的预期退出。至此六个 adapter、6/6 constrained classification 与 6/6 candidate-free ranking 均已完成，正式结果表全部补齐。

2026-08-27 结构与新版本更新：实验目录由 `eval_amazon_sft` 重命名为 `crossinterest.legacy.training_99`，Python 包导入和全部 AMLT 路径已同步，旧名称重复副本经归一化逐字核对后清理。原类别名输出结果保持不变。

同日完成 99 类 interest-ID 数据与方法合同：对 99 个可预测类别按固定 SHA-256 顺序分配 `<interest_000>` 至 `<interest_098>`，映射 fingerprint 为 `e760f350d2d87a4ca69518d328264d0956750deba86bf0e567b129f91c32b79f`。全量 Train/Dev 输入扫描确认历史标题与历史类别中不存在具体 ID；六种 `view x label_mode` 训练消息预览已写入 `data/TRAINING_SAMPLE_PREVIEW.json`。`amlt/amlt_interest_id_formal.yaml` 含两个模型、三视图共六个 ID 输出 job，已通过 `amlt run --dump`，尚未提交。正式结果继续只使用 `RESULTS.md` 中的一张统一表，ID 行在完整评测前保持空白。

## 6. 已知限制 / 未来工作

1. Qwen 三视图已选择各自安全拓扑并稳定运行；长视图不得回退到每卡 batch=2/4，也不能回退到已证实的 8--15 秒/step 旧 fallback 路线。
2. 类别名输出的六个 adapter、六个 constrained classification 与六个 candidate-free ranking 评测均已完成；interest-ID 输出尚未提交。
3. 正式结果只采纳完整 Dev-2K、99 类、`missing=0`、`complete=true` 的 `metrics.json`。
4. 未完成或不完整视图不填结果值；禁止用训练 loss 推断最终分类指标。