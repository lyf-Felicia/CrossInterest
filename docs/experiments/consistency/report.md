# 双目标与一致性约束

> 交接整理：2026-10-08。对照普通CE与一致性条件，解释辅助任务和预测任务的耦合方式。

## 研究问题与设置

两臂以 `ce` 和 `consistent` 标识。给定历史R、已有兴趣A和新兴趣B，监督A|R、B|R、A|R+B、B|R+A四视图；一致性臂在四视图CE上加入前向与反向条件log-probability增量差的平方惩罚，系数1，CE臂系数0。该实验属于历史任务设计探索，不能把它的额外目标默认带入后续只允许原N损失的结构实验。

## 状态与解释边界

| 正式条件 | Recall@10 ↑ | Recall@20 ↑ | nDCG@10 ↑ | nDCG@20 ↑ |
|---|---:|---:|---:|---:|
| ce | 0.223477 | 0.371353 | 0.115646 | 0.152846 |
| consistent | 0.223769 | 0.367880 | 0.117204 | 0.153364 |

均为完整Dev2000/112类。一致性臂的nDCG略升但Recall@20下降，没有全面改善。

两臂均保留正式原始结果验收。`checkpoint_io_launch` 用于检查点写盘恢复，`launch` 保留原提交；I/O修复记录是复现证据，不是可删除的临时日志。只在样本、初始化、训练预算及生成合同对齐后解释方法差值。

## 复现与证据

- 配置：双目标实验（私有原始证据，不随公开仓库分发）。
- 损失与执行：[cir_dual_loss.py](../../../src/crossinterest/methods/consistency/loss.py)、[cir_dual_train.py](../../../src/crossinterest/methods/consistency/train.py)、[cir_dual_run.py](../../../src/crossinterest/methods/consistency/runner.py)。
- 原始验收：CE（私有原始证据，不随公开仓库分发）、一致性（私有原始证据，不随公开仓库分发）。
- 恢复合同：IO_RECOVERY_MANIFEST.json（私有原始证据，不随公开仓库分发）。

返回[总实验文档](../overview.md)。