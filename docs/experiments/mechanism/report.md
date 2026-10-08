# 结构与重构监督的机制消融

> 交接整理：2026-10-08。比较结构、重构预热及两者组合，不将整个方法差值解释成单组件因果效应。

## 研究问题与设置

三臂分别为 `structure`、`reconstruction`、`combined`。训练与辅助损失累积的归一化在第二版修正，因此正式引用应使用 `results_v2`。N阶段保持完整112K监督及原无候选20-beam评测；具体辅助预算和可训练参数取各臂合同。

## 状态与解释边界

| 正式条件 | Recall@10 ↑ | Recall@20 ↑ | nDCG@10 ↑ | nDCG@20 ↑ |
|---|---:|---:|---:|---:|
| structure | 0.221289 | 0.364904 | 0.112793 | 0.148969 |
| reconstruction | 0.240780 | 0.381215 | 0.124903 | 0.160060 |
| combined | 0.239233 | 0.377480 | 0.123248 | 0.157685 |

均完成Dev2000/112类；重构臂四项高于组合臂，当前结果不支持结构与重构叠加后进一步提升。

三臂正式结果与独立原始排名审计保存在同一版本目录。smoke中严格ID格式为0不等于最终结果为0；反过来，smoke执行成功也不是效果证据。必须将结构臂的额外参数和重构臂的额外训练预算分别解释。

## 复现与证据

- 配置：amlt_formal.yaml（私有原始证据，不随公开仓库分发）。
- 实现：[模型](../../../src/crossinterest/methods/mechanism/model.py)、[训练](../../../src/crossinterest/methods/mechanism/train.py)、[执行](../../../src/crossinterest/methods/mechanism/runner.py)、[收集](../../../src/crossinterest/methods/mechanism/audit.py)。
- 第二版原始验收：RAW_RESULT_AUDIT.json（私有原始证据，不随公开仓库分发）。
- 三臂结果：结构（私有原始证据，不随公开仓库分发）、重构（私有原始证据，不随公开仓库分发）、组合（私有原始证据，不随公开仓库分发）。

返回[总实验文档](../overview.md)。