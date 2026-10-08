# Amazon 112 类三视图 SFT 实验记录

最近更新：2026-08-31

## 目标

在 quota=1,000 自然得到的 112 类标签空间上，对 Qwen3.5-9B 与 Llama-3.1-8B-Instruct 分别完成 Behavior、Interest、Behavior + Interest 三视图 SFT。每个视图包含自然语言类别名输出和单级 Interest ID `<interest_NNN>` 输出两个版本；Interest ID 使用子标签语义初始化。

## 数据状态

数据已完成构建和合同校验；category-name 正式训练已提交，修订后的 Interest-ID 训练等待方法确认：

- Train 为 112,000 条，112 类严格各 1,000 条。
- Dev 为 596,449 条自然闭集样本，不重采样。
- 三个冻结评测视图各 2,000 条，每类 17 或 18 条。
- Train 与 Dev 来自既有用户隔离母池，支持集一致。
- 三视图共享 source sample、候选集合、候选顺序和 gold 位置。
- 输入历史全量检查未发现具体 `<interest_...>` ID。

数据 SHA、构建规则和评测视图统计记录在 `data/DATASET_MANIFEST.json`；双输出训练合同、ID 映射哈希和六条完整训练消息记录在 `data/MANIFEST.json` 与 `data/TRAINING_SAMPLE_PREVIEW.json`。

## 方法状态

训练和评测代码已建立；category-name 矩阵已提交，Interest-ID 矩阵仅完成本地准备：

- 两种输出模式使用相同 Train 行和相同三视图输入。
- 类别名模式沿用 v3 prompt。
- Interest-ID 模式使用 v7-single-interest-id prompt。112 个唯一子标签分别对应一个新增原子 token `<interest_000>` 至 `<interest_111>`；父标签不参与 ID 排序、语义表示或 embedding 初始化。
- 每个新 token 的输入与输出 embedding 均以该子标签文本在 base model 中的原 token embedding 均值初始化；训练时只保存 112 行输入 token delta，并通过 `lm_head` LoRA 学习输出，不保存完整 embedding 矩阵。
- 10-way 分类候选继续显示自然语言类别名，decoder 输出候选对应的类别名或 ID。
- candidate-free ranking 不提供候选，也不使用 decoder 标签约束；20 beams 在后处理映射回类别。
- 只有完整 2,000/2,000、112 类、missing=0、`complete=true` 的结果允许写入正式表。

历史 `amlt/amlt_formal.yaml` 含 category 与错误 ID 方法的 12 个 job，仅作审计保留。修订后的 `amlt/amlt_interest_id.yaml` 隔离定义 6 个单级 Interest-ID job，已通过只解析的 `amlt run --dump`，但尚未提交。全部 job 均显式通过 `bash crossinterest.training/run_train_eval.sh` 启动，不依赖上传文件的 executable bit。

## 提交状态

2026-08-27 按 interest-ID 优先顺序完成提交：

- interest-ID 正式实验：`amazon-sft-112-interest-id-v2-20260827-133837`。先提交 `qwen35-interest-interest-id` 作为 4-GPU 健康门禁；该 job 已完成 4-rank NCCL 初始化并推进至至少 43/3,500，最新可见 loss=0.535，随后向同一实验追加其余五个 job。最终矩阵为 1 `running` + 5 `queued`，共 6 个唯一 job。
- category-name 正式实验：`amazon-sft-112-category-20260827-134729`。在上述 ID job 确认真实训练后提交，两个模型、三个视图共 6 个唯一 job 均已由后端接受，初始状态为 `queued`，`status_message` 为空。
- 两个正式实验复用代码包哈希 `eded974ee70d9bcf7f52afc3f914689fef01deabeb3f89a65e1bf68e419454e0`；每个 job 训练完成后依次执行 10-way constrained classification 和 candidate-free direct 20-beam ranking。

2026-08-27 16:03 +0800 状态刷新：两个正式实验合计 5 `running`、6 `queued`、1 AzureML `Paused`，无 `failed` 或 `pass`。当前运行的 Qwen `interest` ID、Qwen `interest` category、Qwen `behavior` category 与 Llama `interest` category 均已完成 3,500/3,500 训练，且各自写出 `train/train_metrics.json`、`train/final_adapter/adapter_config.json` 和完整 adapter 权重；四者已由原 runner 自动进入 2,000 条 constrained classification，最新日志分别推进到 1,060、1,040、990、1,190 条，因此不另行重复提交评测。Llama `behavior_interest` category 仍在训练，最新为 2,227/3,500，已有 `checkpoint-2000`。

尚未运行完的作业中，Qwen `behavior` ID 与 Llama `behavior_interest` ID 均保留了 `checkpoint-1000`，后续默认自动恢复；其余排队作业尚无训练产物。Qwen `behavior_interest` ID 的 AMLT 列表和 AzureML 原生查询均显示 `Paused`，输出目录为空；尝试恢复时 API 曾返回“当前状态为 Queued”，随后又显示 `Paused`，说明该作业存在后端状态切换，但没有训练失败证据，继续保留而不重复提交。四个进行中 classification 均尚未生成最终 `classification/metrics.json`，ranking 也尚未开始，因此 [RESULTS.md](results.md) 继续保持空白。

同次审计再次确认评测合同：classification prompt 提供自然语言 10 候选并使用 10-beam decoder 候选约束；ID 模式只约束到候选对应的 `<interest_NNN>`，随后映射回自然语言类别计分。ranking 使用无候选训练 prompt、20 个 direct beams、无 decoder 标签约束，完整输出 Macro Recall@10/20 与 Macro nDCG@10/20。只有 2,000/2,000、112 类、`missing=0`、`complete=true` 的最终指标允许入表。

2026-08-27 16:45 +0800 状态刷新：总状态仍为 5 `running`、6 `queued`、1 `Paused`，无失败。此前四个 classification 均已达到 2,000/2,000、112 类、`missing=0`、`complete=true`、`strict_format_rate=1.0`，并进入 candidate-free ranking：Qwen `interest` ID 为 Macro Accuracy=0.1135620915、ranking 600/2,000；Qwen `interest` category 为 0.1259337068、ranking 480/2,000；Qwen `behavior` category 为 0.0894899627、ranking 480/2,000；Llama `interest` category 为 0.1524276377、ranking 1,340/2,000。四个 Macro Accuracy 已写入 [RESULTS.md](results.md)，ranking 指标继续留空。Llama `behavior_interest` category 也已完成 3,500/3,500 训练，`train_runtime=6,201.5675` 秒，当前 classification 100/2,000。Feeds A100 Standard 同期占用 63/97，个人 32-GPU 运行配额仍满载。

2026-08-27 17:51 +0800 完成 99/112 训练效果与数据方法审计。112 Train 的 99 个旧类样本是 99 Train 的严格子集：旧类各保留 1,000/4,000 条，共 99,000 条，另加入 13 个新类各 1,000 条；因此总监督量从 396,000 降至 112,000（28.3%），optimizer steps 从 12,375 降至 3,500。112 Dev 母集完整继承 99 Dev 的 592,887 条并新增 3,562 条，但重新分层抽取的 Dev-2K 只有 1,769/2,000 个同源样本，且只有 519/2,000 同时保持完全相同的 10 候选集合，所以两张总表不能直接解释为纯类别数效应。

在上述 519 条完全相同 prompt 上做训练前 base 与 112 category-SFT 的严格配对：Qwen `behavior` Macro Accuracy 从 0.1640 降至 0.1069（-5.70 pp；wrong→right 33、right→wrong 65，双侧 McNemar exact p=0.00160），确认显著负迁移；Qwen `interest` 从 0.1252 降至 0.1001（-2.51 pp；42/48，p=0.598），方向为负但证据不足；Llama `interest` 从 0.0752 升至 0.1278（+5.26 pp；64/30，p=0.000588），确认显著提升。因此不是全模型、全视图统一失效，而是 Qwen 尤其 `behavior` 的模型特异负迁移。

数据完整性检查未发现结构错误：112,000 个 Train ID 全部唯一，每类严格 1,000 条；`target_interest` 从未出现在 `<archived-job-id>`；标题与历史兴趣长度逐行一致；无空历史；Train/Dev 用户交集为 0。99/112 的 `train.py` 除期望行数与标签数常量外逐字一致，`prompts.py` 完全一致，runner 除包名外一致；均为 candidate-free、assistant-only、LoRA r=16/alpha=32/dropout=0.05、LR=1e-4、global batch=32、1 epoch。112 Qwen 最后 100-step loss 约 0.43，99 约 0.41，未发现数值发散或单标签塌缩。

当前方法的主要问题是实验设计与数据利用率，而不是构建 bug：`quota=1000` 同时改变类别数和训练量，无法单独归因；它跑完了生成的 112k 文件，但不是 112 类母池“全部可用训练样本”。母池中这 112 类共有 5,365,828 条可用 Train，最小类也有 1,368 条；最大严格平衡无替换集可达 153,216 条。若目标是与 99 类训练量接近并保留旧知识，更合适的主实验是保留旧 99 类各 4,000 条，加上 13 个新类全部 30,820 条，共 426,820 条，并使用加权采样或 class-weighted loss处理不平衡；另建共同 99 类、共同样本、共同候选的配对评测作为严格 control。当前 12-job 矩阵继续完成，结果标记为 `quota=1000` 探索实验，不覆盖为最终训练配方。

2026-08-27 19:34 +0800 状态刷新：两个正式实验当前共 4 `pass`、4 `running`、4 `queued`，无失败。已完成的 Qwen `behavior` category、Qwen `interest` category、Llama `interest` category 与 Qwen `interest` ID 均重新从 AMLT 下载最终 `classification/metrics.json` 和 `ranking/metrics.json`；8 份指标全部满足 2,000/2,000、112 类、`missing=0`、`complete=true`，已写入 [RESULTS.md](results.md)。对应 ranking 结果依次为：Qwen `behavior` category Recall@10/20=0.1841/0.3280、nDCG@10/20=0.0888/0.1250；Qwen `interest` category=0.1714/0.3098、0.0833/0.1181；Llama `interest` category=0.1739/0.3093、0.0887/0.1227；Qwen `interest` ID=0.1321/0.2385、0.0673/0.0941。其余单元格继续留空。

同一时间建立 `crossinterest.experiments.no_training/` 严格训练前对照：直接复用本实验三份冻结 Dev-2K、自然语言 10 候选、category-name v3 prompt、candidate-constrained 10-beam decoding，只报告 Macro Accuracy，不做 ranking 或 Interest-ID 基线。专用 evaluator 直接复用本目录 classification 的 prompt render、prompt 一致性断言、beam trie、`AutoModelForCausalLM` base loader 和 scoring，只省略 adapter；正式来源 `amazon-notrain-112k-v2-20260827-1940` 已提交 6 个 Standard `80G1-A100` job，初始状态均为 `queued`。该对照固定样本、候选、顺序、gold 位置、prompt、模型加载和解码，可作为 category-SFT 训练前后主比较；它不替代 quota/类别数训练消融。

2026-08-28 07:36 +0800 状态刷新：原 12-job SFT 矩阵在本轮恢复前为 7 `pass` + 5 `failed`，无仍在运行或排队的旧实例。三个新增通过的 Qwen job 已完整产出 classification 与 ranking：category B+I 的 Macro Accuracy=18.68%，Recall@10/20=0.2059/0.3632，nDCG@10/20=0.1005/0.1397；Interest-ID Behavior 为 10.89%、0.1200/0.2323、0.0578/0.0862；Interest-ID B+I 为 10.79%、0.1351/0.2518、0.0663/0.0957。全部指标均满足 2,000/2,000、112 类、missing=0、`complete=true`，已写入 [RESULTS.md](results.md)。

五个旧失败项均为 Llama。Category Behavior、Category B+I、Interest-ID Behavior、Interest-ID Interest 已分别完成 classification，Macro Accuracy=13.03%、16.23%、14.34%、12.77%，四份指标均严格完整，故已正式入表；它们的 ranking 分别停在 176/2,000、990/2,000、1,157/2,000、1,229/2,000，明细 sample ID 均唯一且 fingerprint 与各自合同一致，但 `complete=false`，因此 partial Recall/nDCG 不入表。Interest-ID B+I 在训练 checkpoint-1500 缺少 `trainer_state.json` 时失败，尚无正式 classification/ranking 产物；最后有效训练点为 checkpoint-1000。

恢复根因与修复：四个已完成训练的作业在预占恢复后重复执行零步 `trainer.train()` 并重新保存 `final_adapter`，PEFT 将 `target_modules` 集合以不同列表顺序写回 `adapter_config.json`，使字节 SHA-256 改变，进而触发已有 evaluation `run_contract` 不同；实际模块集合、权重来源、数据 SHA 与评测参数未变。代码现已只在旧 checkpoint 配置与当前 adapter 配置除 `target_modules` 顺序外完全等价时复用旧合同/fingerprint，其他差异仍硬失败；runner 检测完整 `train_metrics.json` 和 final adapter 后直接跳过重复训练；checkpoint 选择忽略缺少 `trainer_state.json` 的半成品目录。行为级恢复检查、Python 编译、Bash 语法和编辑器诊断均通过。

07:34–07:36 +0800 首次使用 116 MB 最小恢复代码包提交五个作业，旧结果全部保留；五个实例上机后 6–10 分钟内均因打包清单漏掉 `crossinterest.evaluation/build_views.py` 而在 prompt import 阶段退出，没有开始新的训练或评测。该问题只影响恢复包，不影响已落盘正式指标。随后补齐运行时模块，并将 adapter 绝对路径加入受内容身份约束的可迁移合同字段：只有旧 checkpoint 配置哈希命中、且规范化 adapter 配置完全等价时才允许跨 rerun results 路径复用旧 fingerprint。工作区外干净导入、迁移行为、真实差异拒绝、Python 编译和 Bash 语法均通过。

2026-08-28 08:24 +0800 第二轮恢复已使用代码包 `cadcbe4cba6a6e481d2f38a0ce669d858cdc970815b49e7f94fc6bfa6a93cc8a` 提交，机器可读状态为 7 `pass` + 5 `queued`。Category Behavior/B+I 新 job ID 为 `<archived-job-id>`、`<archived-job-id>`；Interest-ID Behavior/Interest/B+I 为 `<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。排队等待无法可靠估计；按已有每样本真实 latency 计算，上机后的剩余执行 ETA 依次约 56、37、21、12 分钟。Interest-ID B+I 从 checkpoint-1000 恢复，按同模型训练步速加完整 classification/ranking 估计上机后约 3 小时。所有 ETA 均不含 Feeds 排队时间。

2026-08-28 09:16 +0800 状态刷新：Llama Interest-ID Behavior（`<archived-job-id>`）与 Interest（`<archived-job-id>`）恢复作业均已 `pass`，原有 classification Macro Accuracy 保持为 0.1433531746 与 0.1276552288。新完成的 ranking 均为 2,000/2,000、112 类、missing=0、`complete=true`；20-beam 的 `valid_beam_rate` 与 `strict_beam_format_rate` 均为 1.0，`duplicate_valid_beam_rate` 均为 0。Behavior 的 Macro Recall@10/20=0.1663165266/0.3007703081、Macro nDCG@10/20=0.0843828454/0.1181237409；Interest 对应为 0.1718020542/0.2883111578、0.0849477109/0.1138693406，已写入 [RESULTS.md](results.md)。

其余三个 Llama 恢复项仍为 `running`：Category Behavior（`<archived-job-id>`）、Category B+I（`<archived-job-id>`）与 Interest-ID B+I（`<archived-job-id>`）；未获得完整新指标前对应单元格继续留空。Qwen no-training 三视图独立复测也均已上机并处于 `running`，尚无可替换正式对照表的完整新结果。

2026-08-28 10:10 +0800 方法纠正：此前 `<interest_NNN>` 是类别到普通字符串的无语义映射，且未向 tokenizer 引入原子 token，因此 `amazon-sft-112-interest-id-v2-20260827-133837` 的全部 Interest-ID 指标在方法上无效，已从 [RESULTS.md](results.md) 主表撤下。仍在运行的错误方法 Llama B+I job `<archived-job-id>` 已单独取消并确认为 `killed`；未提交任何替代实验。

按最新确认要求，Interest ID 不采用父类+子类两级结构，而采用 112 个子标签的单级结构。112 个子标签文本经审计全部唯一；按子标签文本词法序冻结为 `<interest_000>` 至 `<interest_111>`，父标签不参与排序或初始化。完整 taxonomy path 仅作为与现有数据集 `target_interest` 对接的内部 key。新 mapping 使用 schema 4、namespace `amazon-child-interest-id-semantic-v1`，fingerprint=`d21096f188c9e2b757299469cd1a400332444ed3c87d6e17068b97bbbf00414c`，可硬拒绝旧随机和中断前两级 mapping。编号只承担稳定唯一索引作用，语义来自对应子标签文本的 embedding 初始化，而不是数字大小或邻近关系。

本地验证已完成：Qwen tokenizer 从 248,077 增至 248,189，Llama tokenizer 从 128,256 增至 128,368，二者均恰新增 112 个唯一原子 token；三个训练 preview 均为 candidate-free 且 assistant 只输出一个 `<interest_NNN>`。小型 untied Llama 验证了子标签 embedding 均值初始化、112 行 `trainable_token_indices`、`lm_head` LoRA、稀疏 checkpoint 保存与 adapter 重载；adapter 不含完整词表 embedding。classification 仍为自然语言 10 候选约束解码，ranking 仍为无候选、无标签约束 direct 20-beam，并按单 token ID 精确解析。runner 在复用训练产物前还会按训练数据 SHA、模型、视图、prompt schema、mapping SHA/fingerprint 校验合同，旧随机或两级 adapter 无法混入。6-job 配置保持“待确认、禁止提交”状态。

2026-08-28 10:27 +0800 命名修订：对外统一称 Interest ID，输出 token 从 `<sid_NNN>` 改为 `<interest_NNN>`，CLI 使用 `LABEL_MODE=interest_id`，结果表行名也改为 Interest ID；“semantic”只用于描述子标签 embedding 初始化方法。新 mapping 文件为 `data/interest_id_semantic_mapping.json`，SHA-256=`5d9d9e1863b54f458ea175eac90a1c7a537240e706d05d210188778d238be5da`。旧随机 `data/interest_id_mapping.json` 保留作审计，但 schema 4/namespace/fingerprint 校验会拒绝它。待审 6-job 配置更名为 `amlt/amlt_interest_id.yaml`，仍未提交。

2026-08-28 10:48 +0800 经确认开始逐步启动修订后的 Interest-ID 完整训练。提交前复核 Train SHA=`43e099050249a8b7746c96044e5db44d3d2ea0d8690a019d1e7d54688ca8c8dd`，三份 Dev-2K SHA 仍分别为 Behavior `890f3ef105e2b85da6adf4e3c283a34f60b4636b8fe29ca5e491e29e5ac7e502`、Interest `766eeeb971f3eb82b940084581a9f07a6cc814645a2d734d5b074e01c994e5cb`、B+I `aebd86d67243022c2dd76a77dccf30146124d54b437baf72788ac1ca98192cb3`；mapping fingerprint=`d21096f188c9e2b757299469cd1a400332444ed3c87d6e17068b97bbbf00414c`。抽查 50 条 Train x 3 views，category 与 Interest-ID prompt 的 history 前缀逐字相同；输出改为单原子 `<interest_NNN>`。三份评测集仍严格共享 source、target、自然语言 10 候选及顺序、gold 位置。

第一阶段只提交 Qwen Interest 视图作为完整健康门禁：实验 `amazon-sft-112-interest-id-semantic-v1-20260828-104049`，job `qwen35-interest-interest-id-semantic-init`，后端 ID `<archived-job-id>`，Feeds Standard `80G4-A100`，初始状态 `queued`。该 job 运行完整 112,000 条、1 epoch、3,500 steps，并依次执行自然语言 10 候选约束 classification 与无候选 direct 20-beam ranking。其余五项暂未提交；门禁需先确认 schema 4 mapping、112-token 注册、非零训练步和无 traceback/OOM/NCCL 错误，再按 Qwen 其余两视图、Llama 三视图逐批追加。

2026-08-28 11:04 +0800 第一阶段门禁通过并启动第二阶段。Qwen Interest job 的 `train_contract.json` 确认 `label_mode=interest_id`、prompt schema=`7-single-interest-id`、Train SHA 与 mapping SHA/fingerprint 均匹配；tokenizer `248077 -> 248189`，恰新增 112 个原子 Interest ID，模型 embedding 保持 248320 行，112 个 ID 均由对应子标签文本 token 均值初始化。可训练参数包含 112 行稀疏 `trainable_tokens_delta` 和 `lm_head` LoRA。训练已稳定推进至至少 step 104/3500；step 40/50/60/70/80/90/100 的 loss 从 2.588 降至 1.636，grad norm 全部非零，未见 traceback、OOM、NCCL error 或 mapping/contract 错误。

随后使用 42-file、115.5 MB 代码包（hash=`073fb9360fc60254b82835166bc30e2aebabfb65a804a5b635d7474ad82be8d2`）向同一实验追加 Qwen Behavior 与 B+I。Behavior job ID=`<archived-job-id>`，SKU=`80G8-A100-IB-NvLink`；B+I job ID=`<archived-job-id>`，SKU=`80G8-A100-NvLink`。机器可读状态为 Interest 门禁 `running`、新增两项 `queued`。Llama 三项仍未提交，待 Qwen 第二阶段至少有一项完成启动合同检查和非零训练步后再逐步放行。

2026-08-28 11:31 +0800 Interest-ID 进度刷新并启动 Llama 门禁。Qwen Interest 已推进至至少 step 1554/3500（44.4%），最近 loss=1.550、grad norm=7.599；Qwen B+I 已推进至至少 step 582/3500（16.6%），最近 loss=1.555、grad norm=9.515；两项吞吐稳定，watcher 未命中 traceback、OOM、NCCL error、mapping/contract mismatch 或 unknown token。Qwen Behavior 仍为 `queued`。

Qwen B+I 的启动合同已逐字段通过：`model=Qwen/Qwen3.5-9B`、`view=behavior_interest`、`label_mode=interest_id`、prompt schema=`7-single-interest-id`、112,000 Train、Train/mapping SHA 与 fingerprint 匹配、恰新增 112 个 Interest ID、初始化仅使用子标签文本 token 均值、输出侧使用 `lm_head` LoRA。由此放行单个 Llama Interest 门禁；使用 42-file、115.5 MB 代码包（hash=`761837ec488edce5c8ed72623995ad2ab6afbaf6e2ae8e7611a602ff76b85991`）提交 `llama31-interest-interest-id-semantic-init`，后端 ID=`<archived-job-id>`，Feeds Standard `80G8-A100-NvLink`，初始状态 `preparing`。当前正式矩阵为 4/6 已提交；Llama Behavior 与 B+I 仍未提交，待该门禁出现合同产物和非零训练步后再放行。

2026-08-28 12:33 +0800 Interest-ID 正式矩阵已完成 6/6 提交。Llama Interest 门禁已稳定推进至至少 step 1738/3500（49.7%），最近 loss=0.7730、grad norm=4.501、约 1.17 秒/step；启动合同确认 `model=NousResearch/Meta-Llama-3.1-8B-Instruct`、`view=interest`、prompt schema=`7-single-interest-id`、Train/mapping SHA 与 fingerprint 匹配，Llama tokenizer `128256 -> 128368`，恰新增 112 个原子 Interest ID，子标签文本均值初始化与 `lm_head` LoRA 均正确，watcher 未命中关键错误。因此使用 42-file、115.5 MB 代码包（hash=`58a3bfd1de045f124686562402f03355932d911566dbe95e25d151191ec9587d`）向同一实验追加剩余两项：Llama Behavior 后端 ID=`<archived-job-id>`，Llama B+I 后端 ID=`<archived-job-id>`；两项均为 Feeds Standard `80G8-A100-NvLink`，机器可读状态均为 `queued`。

同一状态快照中，其余四项全部为 `running`：Qwen Interest 已完成 3,500-step 训练并进入 classification（最新 `completed=1/2000`）；Qwen Behavior 已上机并处于依赖安装/分布式启动阶段；Qwen B+I 至少 step 2027/3500（57.9%，最近 loss=1.498、grad norm=7.636）；Llama Interest 如上已近半程。六项 `status_message` 均为空，尚无方法有效的完整 classification/ranking 指标可写入主表。

2026-08-28 11:18 +0800 Category-name 矩阵补全：正式来源 `amazon-sft-112-category-20260827-134729` 的 Llama Behavior 恢复 job `<archived-job-id>` 与 B+I 恢复 job `<archived-job-id>` 均已 `pass`。两项 classification/ranking 指标均为 2,000/2,000、112 类、missing=0、`complete=true`，模型/视图/输出模式身份分别为 `llama31_8b` + `behavior`/`behavior_interest` + `category`。Behavior 的 Macro Recall@10/20=0.1873541083/0.3209908964、Macro nDCG@10/20=0.0948393692/0.1282664325；B+I 对应为 0.1923144258/0.3512196545、0.0972922938/0.1374815030。四项四舍五入到 4 位小数后已补入 [RESULTS.md](results.md)；已有 Macro Accuracy 13.03% 与 16.23% 经新 metrics 复核一致，未改动。至此两个模型的 Category-name 三视图 classification 与 ranking 主表均已完整。

提交前两轮 ID 启动检查均未产生训练结果：`amazon-sft-112-interest-id-20260827-131407` 因上传包未保留 runner 执行位而为 4 `failed` + 2 `killed`；改用显式 `bash` 后的 `amazon-sft-112-interest-id-fixed-20260827-132638` 又暴露裸 `train.parquet` ignore pattern 误排除嵌套 112 Train，六个 job 均已取消。两个根因均已在最终配置修复，这两轮不得作为正式结果来源。

2026-08-28 16:28 +0800 Interest-ID 状态刷新：六项当前为 2 `pass` + 4 `running`，无 `failed`、`queued` 或 `killed`。Qwen Behavior 与 B+I 的 AMLT 状态已为 `pass`；Qwen Interest 的 AMLT 状态仍为 `running`，但其 classification 与 ranking 最终文件均已完整落盘。三项 Qwen classification/ranking 均为 2,000/2,000、112 类、missing=0、`complete=true`；ranking 的 `valid_beam_rate` 与 `strict_beam_format_rate` 均为 1.0，`duplicate_valid_beam_rate` 均为 0，因此按结果门槛全部写入 [RESULTS.md](results.md)。

Qwen Interest-ID 三视图依次为：Behavior Macro Accuracy=0.1422443978、Recall@10/20=0.2129726891/0.3437791783、nDCG@10/20=0.1067981477/0.1396379628；Interest 为 0.1215861345、0.1904178338/0.3180438842、0.0987365865/0.1309691490；B+I 为 0.1484010271、0.2286998133/0.3725198413、0.1174483663/0.1535877348。

Llama Interest 已由同一后端 job ID `<archived-job-id>` 自动重试并恢复运行：训练 adapter 与 2,000/2,000 classification 被复用，ranking 从此前 403 条继续，最新为 490/2,000。Llama Behavior 正在训练，最新 step 2,537/3,500（72.5%）；Llama B+I 已完成训练并进入 classification，最新 910/2,000，尚无最终 `classification/metrics.json`。三个 Llama 单元格均继续留空，直至同一视图 classification 与 ranking 同时满足正式完整性门槛。

2026-08-31 状态刷新：Llama Interest-ID 三视图均已 `pass`。Behavior、Interest 与 B+I 的 classification/ranking 均为 2,000/2,000、112 类、missing=0、`complete=true`；classification 的 `strict_format_rate` 与 `resolved_prediction_rate` 均为 1.0，ranking 的 `valid_beam_rate` 与 `strict_beam_format_rate` 均为 1.0，`duplicate_valid_beam_rate` 均为 0。身份字段均严格匹配 `model_key=llama31_8b`、对应三视图、`label_mode=interest_id`；classification 使用 10 beams，ranking 使用 20 beams。因此三项均已正式写入 [RESULTS.md](results.md)。

Llama Interest-ID 三视图依次为：Behavior Macro Accuracy=0.1299019608、Recall@10/20=0.2065534547/0.3428454715、nDCG@10/20=0.1035480834/0.1377993644；Interest 为 0.1232784781、0.1909138655/0.3143965920、0.0968471253/0.1279075144；B+I 为 0.1391514939、0.2217261905/0.3665674603、0.1146750099/0.1510818332。至此两个模型、两种输出标签、三个输入视图的正式 classification/ranking 主表全部完整。

2026-08-31 新增第三种输出模式 `[Interest ID] Category name`，CLI 名为 `<archived-job-id>`，字面格式冻结为 `[<interest_NNN>] Parent Category > Child Category`，prompt schema=`8-interest-id-category`。输入三视图、112K Train、冻结 Dev-2K、LoRA 与语义 token 初始化均不变；训练仍无候选且只监督 assistant 输出。Classification 仍展示自然语言 10 候选，并把 decoder 约束到候选对应的 10 个完整复合字符串；ranking 仍无候选、无 decoder 标签约束、direct 20-beam，后处理只接受 ID 与类别严格一致的完整 token 序列。真实三视图 prompt 已写入 `data/TRAINING_SAMPLE_PREVIEW.json`，方法和完整指令见 [README.md](README.md)。

补充矩阵配置为 `amlt/amlt_interest_id_category_recall.yaml`，目标 Recall，两个模型 x 三视图共 6 个 Standard `80G4-A100` job。提交前本地 round-trip、错配拒绝、112 输出 token collision、最大输出长度和 `amlt run --dump` 检查均已通过。

2026-08-31 10:50 +0800 正式提交复合输出矩阵。实验名为 `amazon-sft-112-id-category-recall-20260831-105059`，代码包哈希为 `5ca73e617382fb534a46febea2aab138d4d236867b181bf60b5b47c7e8e5074f`；全部使用 Recall Standard `80G4-A100`。提交返回 6 `preparing`，随后机器可读状态为 6 `queued`，`status_message` 均为空，后端已接受全部唯一 job：

- Qwen Behavior/Interest/B+I：`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。
- Llama Behavior/Interest/B+I：`<archived-job-id>`、`<archived-job-id>`、`<archived-job-id>`。

机器可读命令逐项确认三视图、`LABEL_MODE=<archived-job-id>`、112,000 条 Train、semantic mapping、4 ranks、per-device batch 1 和 gradient accumulation 8。当前所有 job 仍在排队，尚无运行时合同或新指标；未获得完整 classification 与 ranking 指标前 [RESULTS.md](results.md) 对应行保持空白。

2026-08-31 11:07 +0800 运行时门禁：状态已推进为 4 `running` + 2 `queued`，无失败。Qwen Behavior job `<archived-job-id>` 的训练合同确认 prompt schema=`8-interest-id-category`、`label_mode=<archived-job-id>`、112,000 条 Train、mapping SHA/fingerprint、assistant-only 与 candidate-free 均匹配。`semantic_token_setup.json` 确认 tokenizer `248077 -> 248189`、恰新增 112 个原子 token、112 组标签文本 source token、按 taxonomy text token 的 base-model 行均值初始化、输出 embedding 使用 LoRA；模型 embedding 保持 248,320 行。训练已推进至至少 112/3,500，loss 从 3.281 降至 0.3774，grad norm 持续非零；仅见 NCCL 环境变量弃用警告，未见 traceback、CUDA OOM、NCCL error、mapping 或 contract 错误。

2026-09-01 最终验收：复合输出实验 `amazon-sft-112-id-category-recall-20260831-105059` 的 6/6 个 job 均为 `pass`。逐项下载并校验 `classification/metrics.json` 与 `ranking/metrics.json`：全部为 2,000/2,000、112 类、`missing_sample_count=0`、`complete=true`，模型、视图与 `label_mode=<archived-job-id>` 身份严格匹配；classification `strict_format_rate=1.0`，ranking `strict_beam_format_rate=0.773475-0.779400`，`duplicate_valid_beam_rate=0`。此前空缺的 Qwen Interest 指标为 Macro Accuracy=0.1223155929、Recall@10/20=0.1767623716/0.2649393091、nDCG@10/20=0.0825646306/0.1056988309；Qwen B+I 为 0.1607726424、0.2025268441/0.2878443044、0.0999562665/0.1222299321。两项已按主表精度写入 [RESULTS.md](results.md)，至此两个模型、三种输出模式、三种输入视图的 18/18 个结果组合全部完整。

## 结果管理

本目录只使用 [RESULTS.md](results.md) 中的一张统一正式结果表。行维度为模型和输出标签模式，列维度为五组指标与三个输入视图；未完成结果保持空白。

## 比较限制

99 类协议与 112 类协议具有不同的目标支持集和自然闭集 Dev。跨目录结果可用于观察协议变化，但不能称为固定样本、固定类别空间下的纯 quota 消融。如需严格配对比较，应另建共同 99 类与共同 Dev 样本评测，不覆盖本目录正式口径。
