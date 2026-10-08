# Amazon 112 类 No-Training 实验记录

最近更新：2026-08-28

## 目标

在与 112k category-name SFT 完全相同的冻结 Dev-2K 上重新评测旧 no-training 表中的全部八个模型/配置。Qwen3.5-9B 与 Llama-3.1-8B-Instruct 为训练前后变化建立严格主对照；四个 GPT reasoning 配置和两个 Gemini reasoning 配置补齐同协议横向比较。

## 协议

- 三个视图各 2,000 条、2,000 个唯一 sample ID、112 类。
- 直接引用 `crossinterest.training/data/eval/dev_2k_{behavior,interest,behavior_interest}.parquet`，不重新抽样，不重建候选。
- category-name prompt schema v3；提供 10 个自然语言候选；10-beam candidate-constrained decoding；`do_sample=false`。
- 仅报告 Macro Accuracy；不做 candidate-free ranking，不评 Interest ID。
- 无训练、无 adapter。直接复用 `crossinterest.training.evaluate` 的 prompt render、category prompt 一致性断言、constrained beam trie 和 `AutoModelForCausalLM` base loader；Qwen 显式关闭 thinking。
- API 模型复用旧 no-training 表的模型 registry、reasoning effort、请求参数、重试、解析和 Macro Accuracy 实现，只将 data dir 与 run root 切到本实验。
- runner 在加载模型前验证三份文件 SHA-256；每个输出合同还会再次记录实际 dataset SHA-256。

## 建立记录

2026-08-27 19:34 +0800：建立独立目录、严格数据合同、结果表、runner 与 6-job AMLT 配置。三份冻结数据本地复核均为 2,000 行、2,000 个唯一 ID、112 类，SHA-256 与 `crossinterest.training/data/eval/MANIFEST.json` 一致。配置使用 Feeds `80G1-A100`、Standard SLA，每个模型和视图独立成 job；提交前需通过 AMLT dump 和最小样本 smoke。

2026-08-27 19:40 +0800：将首次提交的 `amazon-notrain-112k-20260827-1934` 六个 job 全部主动取消，最终状态为 6 `killed`、每个 duration=1 秒且无结果。取消原因是提交后复核发现旧 `crossinterest.evaluation.evaluate_hf_choice_beam` 的 Qwen base loader 和 beam EOS 处理与 112k SFT evaluator 不完全相同；即使数据相同，也不满足“除 adapter 外执行路径完全一致”的严格条件。该实验不得作为结果来源。

同一时间完成 v2 修正：新增专用 `crossinterest.experiments.no_training.evaluate`，直接复用 `eval_amazon_sft_112.evaluate.render_prompt_ids` 与 `predict`，并使用相同 `AutoModelForCausalLM` base loader，只省略 `PeftModel.from_pretrained`。Python 编译、CLI 导入、runner `bash -n`、AMLT `--dump` 和编辑器诊断均通过。正式实验 `amazon-notrain-112k-v2-20260827-1940` 已提交，代码包哈希为 `0f29d51b60bb7a9647455755bcbf74e0f13d0a785f08c7ad82a3def3121618ea`；机器可读状态确认六个唯一 job 均为 `queued`、`80G1-A100`、Standard，`status_message` 为空。只有该 v2 实验允许写入 [RESULTS.md](results.md)。

2026-08-27 19:49 +0800：按旧 [crossinterest.evaluation/RESULTS.md](../zero_shot/results.md) 扩展模型范围为八个模型/配置，并保留 Random Baseline=10%。TRAPI 重新探测确认 `redmond/interactive` 上存在 `gpt-5.6-sol_2026-07-09` 与 `gpt-5_2025-08-07`；Azure CLI 登录和 `azure.identity`/`openai` 依赖正常。使用 GPT-5 None 对真实 112k Behavior 样本完成 1 条 smoke，记录数=1、错误=0、`complete=true`、strict format=1.0。Gemini 不存在于当前 TRAPI；两个 Gemini 配置始终沿用旧正式实验的 Copilot 反代，不允许切换到 TRAPI 或其他模型。

2026-08-27 19:55 +0800：按历史精确命令 `npx --yes copilot-api@latest start --port 4141 --rate-limit 1 --wait` 恢复 Copilot 反代，登录用户为 `lyf-Felicia`，`/v1/models` 返回 35 个模型且明确包含 `gemini-3.6-flash`。Gemini Minimal 对真实 112k Behavior 样本的单条 smoke 成功，错误=0、strict format=1.0。随后在六个独立 tmux 会话中启动全部 API 正式评测：GPT-5.6 Sol Medium/None、GPT-5 Medium/None 走 TRAPI，Gemini Medium/Minimal 走该 4141 Copilot 反代。首次健康检查六个 Python 进程和代理均存活，无 traceback；Behavior 初始完成数依次为 63、240、8、240、9、107/2,000。输出写入 `runs/formal/<model>/<view>/`，日志写入 `logs/formal/`，均可按 sample ID 恢复。

2026-08-27 21:10 +0800：API 正式评测已有 14/18 个视图完成。GPT-5.6 Sol Medium、GPT-5.6 Sol None、GPT-5 None 与 Gemini 3.6 Flash Minimal 三视图全部完成；GPT-5 Medium 与 Gemini 3.6 Flash Medium 各完成 Behavior，Interest 分别推进至 233/2,000 与 299/2,000。14 份 `metrics.json` 逐份校验均满足 requested=recorded=successful=2,000、request errors=0、missing=0、112 类、`complete=true`；对应 Macro Accuracy 已写入 [RESULTS.md](results.md)。两个 Gemini 配置仍由 `127.0.0.1:4141` Copilot 反代提供服务，持续请求均返回 HTTP 200；1 秒限流等待会自动恢复，不是交互阻塞或请求失败。

同次机器可读 AMLT 状态检查显示 `amazon-notrain-112k-v2-20260827-1940` 为 5 `queued` + 1 `paused`，尚无 HF job 开始执行、失败或产出指标；`paused` 项为 Qwen Behavior，`status_message` 为空。排队等待时间无法由当前状态可靠外推，开始执行后的 ETA 应按实际吞吐另算。

2026-08-28 07:36 +0800：no-training 正式矩阵已全部完成。API 18/18 个视图均有正式 `metrics.json`；新增完成的 GPT-5 Medium Interest/B+I 为 10.48%/13.92%，Gemini 3.6 Flash Medium Interest/B+I 为 10.25%/15.05%。新增四份指标与此前 14 份一样，均满足 requested=recorded=successful=2,000、request errors=0、missing=0、112 类、`complete=true`。两个 Gemini 配置的全部正式请求均通过 Copilot reverse proxy 完成；本次状态扫描时 `127.0.0.1:4141` 已不再监听，但不影响已落盘结果与路由审计。

同次 AMLT 机器可读状态确认 `amazon-notrain-112k-v2-20260827-1940` 六个 HF job 全部为 `pass`：Qwen Behavior/Interest/B+I 的 Macro Accuracy 为 12.79%/12.79%/12.79%，Llama 为 10.06%/11.43%/10.90%。六份指标均为 2,000/2,000、112 类、错误与 missing 均为 0、`complete=true`。至此 [RESULTS.md](results.md) 中八个模型/config × 三视图共 24 个正式单元格全部有值；no-training 任务无剩余运行项，ETA=0。

2026-08-28 08:36 +0800：针对 Qwen 三视图均显示 12.79% 的异常进行逐层审计。原始 Macro Accuracy 并非严格相同：Behavior/Interest/B+I 分别为 0.12785947712418297、0.12785947712418294、0.1279178338001867，但三个视图均恰好答对 256/2,000。AMLT 完整产物确认三个 `run_contract.json` 分别记录正确的 view、三份不同的冻结数据 SHA-256 和三个不同 fingerprint；三份数据的 2,000 个同源样本均有互不相同的 prompt。逐样本预测也不是重复产物：Behavior–Interest、Behavior–B+I、Interest–B+I 分别有 995、708、705 个预测不同，对应仅一侧答对的样本数分别为 124/124、83/83、92/92。因此当前没有发现 view 传参、数据选择、结果串写或 scoring 复用错误；相同总正确数更符合一次罕见巧合，但在独立复测完成前保留审计标记。

同一时间已用新实验名 `amazon-notrain-112k-qwen-retest-20260828-083653` 提交独立 deterministic 全量复测，只运行 Qwen Behavior/Interest/B+I 三个 job，代码包哈希为 `25af4dfe952849e658e333912488157fa17b61d27c10530348d3bb1b2ca1f985`。机器可读状态确认三个 `80G1-A100`、Standard SLA job 均已接受并处于 `queued`：Behavior=`<archived-job-id>`，Interest=`<archived-job-id>`，B+I=`<archived-job-id>`。新实验使用独立结果根目录，不覆盖原产物；完成后需比较 exact metrics、逐样本 prediction 和 sequence score。

## 解释边界

该实验解决旧 99-label no-training 表与 112-label SFT 表之间样本、候选和标签空间不同的问题，可以直接回答“同一 112-label 分类任务上，加载 SFT adapter 后是否提升”。它仍不能把 99→112 的类别扩展效应与每类训练量 4,000→1,000 的 quota 效应拆开；后者需要单独的训练消融。