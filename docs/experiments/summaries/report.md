# LLM摘要与检索上下文探索

> 交接整理：2026-10-08。本目录是历史输入与教师信号探索，不归入当前禁止改数据的结构实验。

## 研究问题与流程

考察LLM历史摘要、检索上下文、兴趣转移画像和教师排序能否提供新的预测信号。输入材料生成、数据物化、canary验证和最终评测分别保存，不能把教师输出质量当作学生排名效果，也不能把pilot结果作为完整正式实验。

## 状态与交接边界

目录包含 `pilot`、`teacher` 和 `results`，本次未重新查询云端作业或对这些结果执行全量独立验收，不将它们宣称为新的正式完成结果。引用前先核对生成来源、无未来泄漏、样本覆盖及门禁结论。

## 复现与证据

- 配置：摘要评测（私有原始证据，不随公开仓库分发）、rationale canary（私有原始证据，不随公开仓库分发）。
- 流程：[生成摘要](../../../src/crossinterest/experiments/summary/generate_summaries.py)、[构建检索上下文](../../../src/crossinterest/experiments/summary/build_retrieval_context.py)、[教师排序](../../../src/crossinterest/experiments/summary/generate_teacher_rankings.py)、[数据物化](../../../src/crossinterest/experiments/summary/materialize_dataset.py)。
- 健康与决策入口：[analyze_gate.py](../../../src/crossinterest/experiments/summary/analyze_gate.py)。
- 该探索与[去噪实验](../denoising/report.md)分开解释。

返回[总实验文档](../overview.md)。