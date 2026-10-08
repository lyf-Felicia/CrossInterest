# SuperLoss损失方案历史交接

> 交接整理：2026-10-08。设计、实现和smoke配置保留；本次没有发现该目录的完整正式结果汇总。

## 研究问题

探索根据样本损失调整训练贡献是否改善兴趣预测。具体目标、分组和公式见原实验设计。本方案属于损失与样本贡献研究，不能作为当前“原数据、原目标函数不变”的结构实验组件。

## 交接内容与使用边界

代码包含数据准备、SuperLoss、训练、分片评测与结果合并。没有正式结果表就不将其描述成效果已证实，也不把设计文档的预期数字填写为实际指标。DDP与速度smoke只验证执行和吞吐。

## 复现入口

- [实验设计](design.md)、[实际Prompt](prompts.md)。
- [损失实现](../../../src/crossinterest/losses/superloss/superloss.py)、[数据构建](../../../src/crossinterest/losses/superloss/prepare_data.py)、[训练](../../../src/crossinterest/losses/superloss/train.py)。
- [评测](../../../src/crossinterest/losses/superloss/evaluate.py)、[分片合并](../../../src/crossinterest/losses/superloss/merge_evaluation_shards.py)、[结果分析](../../../src/crossinterest/losses/superloss/analyze_results.py)。
- 正式配置（私有原始证据，不随公开仓库分发）、DDP烟测（私有原始证据，不随公开仓库分发）。

已有配置不意味着授权新提交；接手者应先确认是否存在外部保存的完成产物。返回[总实验文档](../overview.md)。