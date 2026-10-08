# 独立兴趣编码与分组控制

> 交接整理：2026-10-08。C/R/N完整训练与评测已完成，健康通过、预设效果门槛失败。

## 研究问题与设置

检验按兴趣分组后独立编码是否改善新兴趣预测。C按真实类别分组并做CIR预热，R保留组大小但随机分组且同样预热，N真实分组但不做CIR。CIR预算83,888条、3轮、7,866更新，N预算112K、1轮、3,500更新。三组共享生成器和原完整词表20-beam评测。

## 状态与结论

三组已完成规定预算、Dev2000及本地独立验收，未达到预注册效果门槛。训练布局在恢复中调整，端到端约38小时21分钟，两次8小时期限均未达到；不能将小样本吞吐外推成正式完成承诺。

该负结果不能证明分组编码永远无效，但足以说明当前实现未提供所要求的收益。后续理论讨论属于新候选分析，不是本实验的成功证据。

## 复现与证据

- 全部配置、来源、指标与恢复时间线：[EXPERIMENT_LOG.md](log.md)。
- 当前科学合同：PREFLIGHT.json（私有原始证据，不随公开仓库分发）。
- 实现：[composition_run.py](../../../src/crossinterest/methods/composition/runner.py)、[composition_monitor.py](../../../src/crossinterest/methods/composition/monitor.py)。
- 文献与条件性理论：[LITERATURE_REVIEW_20260923.md](literature.md)。

`launch-v*`、随机命名的提交暂存与`.venv-qwen`仍承担历史来源或本地验证职责；本次不批量删除。返回[总实验文档](../overview.md)。