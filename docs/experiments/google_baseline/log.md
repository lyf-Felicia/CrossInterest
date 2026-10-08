# Google VIE Amazon 公开机制复现实验记录 (Experiment Log)

> 项目：`crossinterest.baselines.google/`，在 Amazon Reviews'23 上复现 Google VIE 的 K=2 兴趣簇转移 SFT 机制，并接入统一 112 类评测。
> 关键约束：无官方代码、私有训练数据、工业兴趣簇、Gemini-Pro checkpoint 或线上系统，只能声称公开机制复现。
最近更新：2026-08-31

---

## 1. 实验目标

1. 忠实实现论文中“两个最近唯一兴趣簇 -> 一个未见新兴趣簇”的候选无关 SFT 合同。
2. 按每个目标兴趣簇选择频次最高的 10 个有序转移 pair，构建目标严格均衡的公开替代训练集。
3. 在同一 112 类标签空间上评测 10-way Macro Accuracy 与 candidate-free Macro Recall/nDCG@10/20。
4. 预注册数据规模等比例的 442-step 主设置，并把论文原始 3,000-step 预算隔离为敏感性实验。
5. 清楚区分论文事实、Amazon 适配和当前 benchmark 扩展，不把公开实验描述为私有工业系统的精确复跑。

## 2. 模型与方法

论文：Jianling Wang 等，*LLMs for User Interest Exploration in Large-scale Recommendation Systems*，arXiv:2405.16363v2。论文使用 761 个 cluster、每个 cluster top-10 transition pairs，共 7,610 条 SFT 数据；参考设置为 Gemini-Pro、batch 16、约 3,000 steps。

| 设置 | 模型 | Train | Global batch | Updates | 定位 |
|---|---|---:|---:|---:|---|
| `scaled-442` | Qwen3.5-9B / Llama-3.1-8B-Instruct | 1,120 | 16 | 442 | 主设置，保持论文约 6.31 次数据遍历 |
| `paper-3000` | Qwen3.5-9B / Llama-3.1-8B-Instruct | 1,120 | 16 | 3,000 | 仅敏感性设置，不用于事后替换主结果 |

公开适配使用 Amazon `L2 > L3` 标签作为 interest cluster；cluster 描述为 `{L2, L3}`。每条输入只保留最近两个不同且属于 112 类支持集的 cluster，并保持时间顺序。目标必须未出现在完整目标前历史中。训练为 candidate-free、assistant-only LoRA SFT：`r=16`、`alpha=32`、`dropout=0.05`、`all-linear`、LR `1e-4`、cosine scheduler、warmup ratio `0.03`。

10-way 分类只在评测时附加冻结候选并约束 decoder 到 10 个完整 `New cluster: ...` 输出。排序评测不提供候选、不使用标签约束，直接返回 20 beams 后映射到 112 类并计算 Macro Recall/nDCG。

## 3. 仓库结构

```text
crossinterest.baselines.google/
├── build_dataset.py              # 5.37M 母池流式 pair 计数与冻结
├── prompts.py                    # K=2、novelty、Figure-2 风格 prompt
├── train.py                      # LoRA assistant-only SFT
├── evaluate.py                   # 10-way Macro Accuracy
├── evaluate_ranking.py           # Recall/nDCG@10/20
├── run_train_eval.sh             # 训练后串行执行两种评测
├── amlt/
│   ├── amlt_formal.yaml          # 442-step 主设置
│   ├── amlt_sensitivity_3000.yaml # 3,000-step 敏感性
│   └── <archived-job-id>.yaml # Llama ranking 单卡断点恢复
├── data/
│   ├── train_top10_pairs.parquet # 1,120 条
│   ├── dev_2k_pair.parquet       # 1,997 条
│   ├── pair_grid.parquet         # 12,544 条
│   ├── MANIFEST.json             # v2 数据合同与 SHA
│   └── LEGACY_V1_AUDIT.md        # v1 重建差异审计
├── tests/test_contract.py        # 11 个合同测试
├── README.md
└── RESULTS.md
```

## 4. 运行环境

- 本地系统：Linux；工作目录 `crossinterest_formal/amazon`。
- 测试环境：conda `exp`；PyArrow 24 写出正式 parquet；pytest 合同测试通过。
- 母池：`prep/artifacts/home_and_kitchen/data/unbalanced/train.jsonl`，SHA-256 `db66726e3d8cad5c06a984620f14843e9e7709bee956f19f3f17dd2c8f5ecf36`，约 18 GB。
- AMLT：Recall target / `${AZURE_CONFIGURATION}` workspace；主实验每个 job 使用 4×80 GB A100，ranking 恢复使用 1×80 GB A100，均为 Standard SLA。配置最初由 `amlt template -t Feeds` 生成，ACPT 镜像 smoke 后按资源隔离要求迁移到 Recall；workspace 主身份为 `${AZURE_CONFIGURATION}`。
- 依赖：见 `requirements_amlt.txt`；正式 parquet writer 精确固定为 PyArrow 24.0.0，因为文件字节 SHA 属于冻结合同。442-step 主实验已提交到 Recall；3,000-step 敏感性尚未提交。

## 5. 实验进度

### 实验 1 — 论文与公开资源审计（✅ 已完成）

- **配置**：arXiv:2405.16363v2，正文 Figure 2 prompt 以 `vie_prompt.pdf` 为准。
- **启动方式**：读取 arXiv 正文、源文件与 BibTeX；检索 Google 官方公开代码和数据。
- **踩坑**：归档中的 `prompt.pdf` 文件名具有误导性 -> 回溯正文引用确认实际 Figure 2 使用 `vie_prompt.pdf` -> 旧文件不是最终论文 prompt。
- **已确认正常**：完整 12 人作者列表、v2 日期、K=2 prompt、top-10 pair、761/7,610、batch 16、约 3,000 steps 均已审计；未发现官方实现或数据。
- **结果**：仅确定复现边界，无模型指标。
- **产物**：`README.md` 的 Reproduction Scope 与 Citation。
- **耗时 / EXIT**：资料审计完成；不涉及训练 EXIT。

### 实验 2 — v1 冻结数据重建审计（✅ 已完成）

- **配置**：`google-interest-v1`，完整 5,372,964 行母池。
- **启动方式**：

```bash
python -m crossinterest.baselines.google.build_dataset \
  --prompt-schema google-interest-v1 \
  --verify-frozen crossinterest.baselines.google/data
```

- **踩坑**：旧文件由一次性脚本生成，正式 builder 未落盘 -> 流式重建 pair 统计并逐列比较 -> 确认只有 Train 的代表 `source_id` 规则未保留。
- **已确认正常**：全部聚合统计、1,120 个 target/pair/frequency/rank、1,997 个 Dev 样本和 12,544 个 pair grid 均复现；Dev 与 grid 语义完全一致。Train 只有 1,099 个 provenance `source_id` 不同，prompt、target、频次和训练 multiplicity 均一致。
- **结果**：旧 v1 SHA 与差异见 `data/LEGACY_V1_AUDIT.md`。
- **产物**：临时重建目录运行结束后自动删除，不覆盖冻结数据。
- **耗时 / EXIT**：比较器按预期因 provenance 不完全一致返回非零；核心语义审计完成。

### 实验 3 — v2 正式数据冻结与合同测试（✅ 已完成）

- **配置**：`google-interest-v2`；代表样本固定为母池 JSONL 顺序中的首次匹配；pair 以频次降序、pair 字典序升序破平。
- **启动方式**：

```bash
python -m crossinterest.baselines.google.build_dataset \
  --prompt-schema google-interest-v2 \
  --output-dir crossinterest.baselines.google/data
python -m crossinterest.baselines.google.build_dataset \
  --prompt-schema google-interest-v2 \
  --verify-frozen crossinterest.baselines.google/data
conda run -n exp python -m pytest -q crossinterest.baselines.google/tests
```

- **已确认正常**：扫描 5,372,964 行；支持目标 5,365,828 行；有效转移 5,359,952 行；唯一 target-pair 484,741 个。Train 1,120 条、112 类各 10 条；Dev 1,997 条、112 类各 17 或 18 条；pair grid 12,544 条。所有 prompt、novelty、rank 和 manifest SHA 已逐行验证，9 个测试全部通过。
- **结果**：Train SHA `886c45e521be91f0c4d1d2a56efbe06483401e6905bfc6381884ba30c97ca218`；Dev SHA `1efe9f221e3249c318fc279487cd96d7445990ce23664faa18c778f1d787744c`；grid SHA `a61dd1af5a9cdd362f902d1b57cefec2dd2339d26049598d77c8c23e032c33ec`。第二次完整母池重建中，三份 parquet 的 byte SHA、完整语义和全部列逐行比较均完全一致。
- **产物**：`data/*.parquet`、`data/MANIFEST.json`。
- **耗时 / EXIT**：正式构建 EXIT 0；临时目录字节级重建验证 EXIT 0；pytest `9 passed`。

### 实验 4 — 442-step 主设置（✅ 已完成）

- **配置**：`amlt/amlt_formal.yaml`，Qwen/Llama 各 4×A100，global batch 16，442 updates。
- **启动方式**：

```bash
cd crossinterest.baselines.google/amlt
amlt run amlt_formal.yaml google-vie-amazon-scaled-442-recall-20260828
amlt map <archived-job-id>.yaml :llama31-ranking-recovery \
  google-vie-amazon-scaled-442-recall-20260828 :llama31-scaled-442 \
  google-vie-llama-ranking-recovery-r3-20260831 -y
```

- **踩坑**：Llama 正式 job 已产生完整 classification 和 1,745 条 ranking 明细，但终态为 `failed`；后续执行从 checkpoint 再进入 runner 时触发 `classification: existing run contract differs`，不能直接重跑整条链。r1 在排队阶段因发现 map 挂载路径别名而主动取消；r2 的严格门禁发现自动重启重写了 `final_adapter/adapter_config.json`，仅 `target_modules` 顺序变化但 byte SHA 已不同。r3 改用保留原配置的 `checkpoint-442`，同时校验训练步数、adapter 权重 SHA、语义合同和导入行数。
- **已确认正常**：Qwen job `<archived-job-id>` 和 Llama classification 均通过 1,997 条完整性门禁。Llama ranking 恢复后为 1,997 条唯一 ID、`missing=0`、`class_count=112`、`complete=true`；原 1,745 条除新 run fingerprint 外逐字段不变，新增 252 条恰好覆盖原缺失 ID。独立重算七项 ranking 指标与正式 `metrics.json` 一致，`predictions.jsonl` 也与 details 投影一致。合同测试 11 个全部通过。
- **结果**：Qwen Macro Accuracy `11.248%`、Macro Recall@10/20 `0.1350/0.2271`、Macro nDCG@10/20 `0.0658/0.0891`；Llama 对应为 `11.394%`、`0.1373/0.2213`、`0.0669/0.0881`。Llama ranking valid/strict/duplicate-valid beam rate 为 `0.8812/0.8800/0.0011`，详见 [RESULTS.md](results.md)。
- **产物**：原实验 `google-vie-amazon-scaled-442-recall-20260828`；正式恢复实验 `google-vie-llama-ranking-recovery-r3-20260831`，job ID `<archived-job-id>`；结果包含 `ranking/details.jsonl`、`ranking/metrics.json`、`ranking/predictions.jsonl`、`ranking/resume_provenance.json`。
- **耗时 / EXIT**：Qwen 约 3 小时并 `pass`；原 Llama 约 2 小时后 `failed`，但 classification 产物完整；r3 恢复约 16 分钟并 `pass`。Feeds 无 baseline 活动作业。

### 实验 5 — 3,000-step 敏感性设置（⬜ 待办）

- **配置**：`amlt/amlt_sensitivity_3000.yaml`，与主设置仅训练 updates 和 checkpoint 间隔不同。
- **启动方式**：

```bash
cd crossinterest.baselines.google/amlt
amlt run amlt_sensitivity_3000.yaml google-vie-amazon-paper-3000
```

- **已确认正常**：`amlt run amlt_sensitivity_3000.yaml --dump` 通过；主设置与敏感性输出不会混用。
- **结果**：见 `RESULTS.md`，当前为空。
- **产物**：计划输出同实验 4。
- **耗时 / EXIT**：尚未提交。

## 6. 已知限制 / 未来工作

- 不能复现私有工业 cluster、Gemini-Pro checkpoint、低层 item grounding recommender 或线上 A/B 指标。
- Amazon taxonomy 实体是公开 metadata substitute，不等价于论文自动抽取的 salient keyword/entity 集合。
- 三个缺少两个不同支持 cluster 的冻结 Dev-2K 样本被排除，因此本 baseline 的样本数为 1,997，不与三视图 2,000 行结果做逐样本全覆盖宣称。
- 442-step 是主结论来源；3,000-step 只报告敏感性，不能事后选择更优结果替代主设置。
- 当前 AMLT 配置使用 ACPT Torch 2.8 / CUDA 12.6 镜像，但镜像 digest 与 HF 模型 revision 尚未固定；正式产物应保留 `nvidia-smi.txt`、`pip-freeze.txt` 和模型运行合同。
- Qwen 2-step smoke 的 candidate-free beams 尚未收敛到冻结 112-label 精确空间，`valid_beam_rate=0`；这是训练量不足的健康度信号，必须在 442-step 正式结果中重新审计，不能把零值直接解释为模型能力结论。
