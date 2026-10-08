# 冻结CIR与新建N LoRA

> 交接整理：2026-10-08。检验预热知识保留与预测阶段参数解耦。

## 研究问题与设置

冻结已训练的CIR adapter，为N阶段新建LoRA，而不是继续更新全部CIR参数。方法标识为 `frozen-cir-fresh-n-lora-v1`；保留原112K / 3,500步监督和完整词表排序。源权重及参考N权重分别记录，不能把二者SHA混用。

## 正式结果

| 条件 | Recall@10 ↑ | Recall@20 ↑ | nDCG@10 ↑ | nDCG@20 ↑ |
|---|---:|---:|---:|---:|
| 冻结CIR、新建N LoRA | 0.249767 | 0.394345 | 0.126828 | 0.163251 |

完整正式结果已审计。PREFLIGHT中的历史G0为0.251255 / 0.401786 / 0.127867 / 0.165582；当前分支低于这组参考，但N训练布局不同，不能将差值全部归因于冻结策略。

## 复现与证据

- 来源与协议：PREFLIGHT.json（私有原始证据，不随公开仓库分发）。
- 配置：amlt_cir_residual_20260920.yaml（私有原始证据，不随公开仓库分发）。
- 实现：[cir_residual_model.py](../../../src/crossinterest/methods/residual/model.py)、[cir_residual_run.py](../../../src/crossinterest/methods/residual/runner.py)。
- 结果：RESULT_AUDIT.json（私有原始证据，不随公开仓库分发）、原始复算（私有原始证据，不随公开仓库分发）。

返回[总实验文档](../overview.md)。