# Amazon 112 类训练数据量实验记录

最近更新：2026-09-01

## 目标

固定 112 类、B+I 输入、category-name 输出、冻结 Dev-2K、训练超参数和评测口径，只改变嵌套 Train 前缀大小，测量 Qwen3.5-9B 与 Llama-3.1-8B-Instruct 在 5K 至 112K 之间的性能饱和曲线。

本实验与 `crossinterest.legacy.data_scaling_99/` 的 99 类 / 396K 实验是两套独立协议：Train、标签空间和 Dev-2K 均不同。结果分别维护，不合并成同一条数据量曲线，也不把相同 Train rows 当作严格控制变量点直接比较。

## 方法

- 8 个规模：5K、10K、20K、40K、60K、80K、100K、112K（all）。
- 每个完整 112 行轮次每类恰取一条；轮内类别顺序按固定 seed 做 SHA-256 排序。
- 所有规模始终保留完整 112 类，不通过减少类别数控制数据量。对总量 $N$，令 $q=\lfloor N/112\rfloor$、$r=N\bmod112$，严格分配为 $r$ 类各 $q+1$ 条、其余类各 $q$ 条，因此总行数恰为 $N$ 且类间差最多 1；112K 包含原 Train 全部样本。
- 每个规模训练 1 epoch、effective batch=32，因此训练步数随数据量增加。这是数据量曲线，不是固定 optimizer-step 曲线。
- classification 与 candidate-free 20-beam ranking 沿用 `crossinterest.training` 的正式口径。

## 状态

112K 端点直接复用同协议正式实验 `amazon-sft-112-category-20260827-134729` 的 B+I/category 完整结果：Qwen Macro Accuracy=18.68%、Recall@10/20=0.2059/0.3632、nDCG@10/20=0.1005/0.1397；Llama 对应为 16.23%、0.1923/0.3512、0.0973/0.1375。该来源的 Train、prompt、1 epoch、effective batch=32 和冻结评测集与本曲线完全一致，因此不重复消耗两个 112K 训练 job。

2026-08-31 10:50 +0800 正式提交其余 7 个规模 x 2 模型共 14 个 Feeds job。实验名为 `amazon-datasize-bi-category-20260831-105001`，代码包哈希为 `b86b1646e91670b41fbe02efe892474ef634a1868b70eeae5a9d8b4985b543eb`；全部使用 Feeds Standard `80G4-A100`。提交返回 4 `queued` + 10 `preparing`，一分钟后的机器可读状态为 14 `queued`，`status_message` 均为空，后端已接受全部唯一 job：

- Qwen 5K/10K/20K/40K/60K/80K/100K：`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。
- Llama 5K/10K/20K/40K/60K/80K/100K：`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。

机器可读命令逐项确认 `VIEW=behavior_interest`、`LABEL_MODE=category`、`TRAIN_SELECTION_SIZE` 与 job 名中的规模一致，且均使用 4 ranks、per-device batch 1、gradient accumulation 8。当前所有 job 仍在排队，尚无训练日志或新指标；正式结果表除复用的 112K 端点外继续留空。

2026-08-31 11:07 +0800 运行时门禁：状态已推进为 7 `running` + 7 `queued`，无失败。Qwen 10K job `<archived-job-id>` 的 `train/train_contract.json` 确认 source Train SHA、selection SHA、`selection_size=10000`、B+I/category、1 epoch 与 effective batch 32 均匹配；实际训练总步数为 313，已推进至至少 128/313，loss 从 0.8954 降至约 0.47，grad norm 持续非零，未见 traceback、CUDA OOM、NCCL error、selection 或 contract 错误。

2026-08-31 14:00 +0800 类别平衡专项复核：使用冻结 source Train 的真实 `target_interest` 按 selection ID 独立重算，八个规模均为 112 类且总行数严格命中。5K 为 40 类 x 44 + 72 类 x 45；10K 为 80 x 89 + 32 x 90；20K 为 48 x 178 + 64 x 179；40K 为 96 x 357 + 16 x 358；60K 为 32 x 535 + 80 x 536；80K 为 80 x 714 + 32 x 715；100K 为 16 x 892 + 96 x 893；112K 为 112 x 1,000。四卡 `DistributedSampler` 模拟确认每个规模均为 0 重复、0 丢弃。当前已提交实验使用同一 selection SHA `756903e03407a5a342e4966ef06ef2470a8d4f1853cb7bb425cec9a6d940a61d`，无需取消或重提。训练入口已新增 112 类、精确行数、最大类差 1 和 world-size 可整除硬门禁，并显式设置 `dataloader_drop_last=False`。

## 2026-09-01 完成状态

正式实验 `amazon-datasize-bi-category-20260831-105001` 的 14/14 个新任务均为 `pass`，`status_message` 全空。下载并验证 28 份 classification/ranking `metrics.json` 后确认：每个任务均为 2,000/2,000 样本、112 类、B+I/category、`complete=true`、无 missing sample。加上复用的两个 112K 端点，`RESULTS.md` 的 16/16 行已全部完成。

本次补齐的五行结果为：

| 模型 | Train | Macro Accuracy (%) ↑ | Recall@10 ↑ | Recall@20 ↑ | nDCG@10 ↑ | nDCG@20 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 40K | 8.91 | 0.1836 | 0.3330 | 0.0925 | 0.1300 |
| Llama-3.1-8B-Instruct | 5K | 7.74 | 0.1366 | 0.2639 | 0.0660 | 0.0979 |
| Llama-3.1-8B-Instruct | 20K | 9.99 | 0.1697 | 0.3212 | 0.0816 | 0.1193 |
| Llama-3.1-8B-Instruct | 40K | 12.87 | 0.1884 | 0.3318 | 0.0929 | 0.1287 |
| Llama-3.1-8B-Instruct | 80K | 11.97 | 0.1950 | 0.3404 | 0.0967 | 0.1330 |

112 类曲线存在局部非单调，不能仅凭单个规模点断言饱和：Qwen Macro Accuracy 在 40K 低于 20K，Llama 在 60K/80K 的 Accuracy 低于 40K，而部分 ranking 指标仍继续提高。当前结论应以完整曲线和不同指标共同判断，不与 99 类曲线混合解释。
