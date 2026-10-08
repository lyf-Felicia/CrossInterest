# Amazon Data Preparation

本目录集中保存 Amazon `Home_and_Kitchen` 新兴趣数据的处理方法、报告、测试和中间产物。上一级 `amazon/` 只保存可直接使用的正式 Parquet 数据：`train.parquet` 和 `dev.parquet`，与 MIND 目录的发布格式一致。

## 先读什么

1. `docs/BUILD_REPORT.md`（私有原始证据，不随公开仓库分发）：完整构建报告。第 4 节集中解释历史如何截取、正样本如何得到。
2. `docs/DATA_CONTRACT.md`（私有原始证据，不随公开仓库分发）：不可变的数据合同与审计条件。
3. `artifacts/home_and_kitchen/MANIFEST.json`（私有原始证据，不随公开仓库分发）：数据版本、参数和最终统计的机器可读快照。

## 目录

```text
amazon/
├── train.parquet                   # 396,000 条平衡 Train
├── dev.parquet                     # 592,887 条自然闭集 Dev
├── crossinterest.evaluation/            # 无训练 Amazon 评测代码、数据与正式结果
├── crossinterest.legacy.training_99/             # quota=4,000、99 类、双输出 SFT 协议
├── crossinterest.training/            # quota=1,000、112 类、双输出 SFT 协议及独立数据
└── prep/
  ├── README.md                   # 数据处理唯一入口
  ├── docs/                       # 构建报告与数据合同
  ├── scripts/                    # 构建、导出、分析、绘图和验证
  ├── tests/                      # 合同测试
  └── artifacts/home_and_kitchen/ # raw、母池、OOV、索引和分析
```

根层 `train.parquet` / `dev.parquet` 是 99 类协议的正式数据。112 类协议的
`train.parquet` / `dev.parquet` 与冻结三视图位于
`crossinterest.training/data/`，不会覆盖根层 99 类数据。

## 核心数据语义

- 正向事件：`rating >= 4.0`。
- 用户按 SHA-256 稳定划入 90% Train / 10% Dev，用户零重叠。
- 同一毫秒交互共享严格事件前历史，不虚构组内先后顺序。
- 完整未截断历史用于判定 `L2 > L3` 是否为新兴趣。
- 目标前至少有 5 条合法正向交互；模型只看到最近 50 条。
- 同一事件组有多个新类别时，固定 seed 均匀选一个类别，再选一个商品作为唯一 gold。
- 先生成样本，后将当前事件组追加到历史。
- 只平衡 Train；Dev 保留自然分布。

## 复现命令

以下命令均在本 `amazon/prep/` 目录执行。JSONL 是保留全部审计字段的处理阶段格式，位于 `artifacts/home_and_kitchen/data/formal/`；最后使用 PyArrow 按 10,000 行批次和 Zstandard 压缩导出根层 Parquet。

```bash
python -m scripts.build \
  --work-dir artifacts/home_and_kitchen \
  --formal-dir artifacts/home_and_kitchen/data/formal

python -m scripts.analyze_balance \
  --work-dir artifacts/home_and_kitchen \
  --quotas 500 750 1000 1500 2000 2500 3000 4000 5000

python -m scripts.build_eval_views \
  --work-dir artifacts/home_and_kitchen \
  --formal-dir artifacts/home_and_kitchen/data/formal
python -m scripts.<archived-job-id> --work-dir artifacts/home_and_kitchen
python -m scripts.verify \
  --work-dir artifacts/home_and_kitchen \
  --formal-dir artifacts/home_and_kitchen/data/formal
python -m scripts.export_parquet \
  --input-dir artifacts/home_and_kitchen/data/formal \
  --dataset-dir ..
python -m pytest tests/test_contract.py -q
```

首次完整构建会流式处理约 11 GiB metadata 和约 3.6 GiB rating-only CSV。已有正式产物时，不要仅为重画图或重新分析 quota 而重建历史。

Parquet 保留 JSONL 的全部 22 个字段：字符串数组继续使用 Arrow list，`timestamp` 和 `history_timestamps` 转为 `timestamp[ms, UTC]`，其余审计字段保持显式数值或字符串类型。转换不会修改样本集合、历史内容或顺序。