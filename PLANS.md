# 当前阶段与开发路线图

更新时间：2026-08-28。本文只保留当前阶段、优先级、验收标准和下一步，不记录历史开发过程。代码真相以 `ai_glasses_memory_assistant/`、`android/`、`static/`、`tests/`、`evals/` 为准。

## 当前阶段

当前项目是 Stage 2 可运行原型，核心闭环已经形成：

- Web/语音 demo 入口。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 记忆写入门控、召回、纠错、evidence、删除和 debug/audit。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- `GlassesChatService` 是主 service 调度入口，保留聊天主流程、最终 memory gate、后台 job 生命周期、audit、timeline evidence 和写库编排。
- 纯 helper 按主题放在 `document_helpers.py`、`import_helpers.py`、`memory_job_helpers.py`、`timeline_management_helpers.py`、`source_summary_helpers.py`、`explanation_helpers.py`、`response_timing.py`、`llm_runtime.py`、`report_helpers.py`、`memory_recall_arbitration.py`、`capture_helpers.py`、`conversation_helpers.py` 和 `conversation_candidate_helpers.py`。
- 启发式周报草稿和手动提醒候选检查。
- 统一音频 session、16 kHz PCM、VAD、KWS-only 唤醒、partial/final ASR、ambient capture 和保守声纹；模型和 runtime 都在 `audio_engine/`，`audio_processing.py` 只保留导入兼容。
- speaker-labeled transcript 先经过结构化 extraction plan：用户本人、命名人物和临时 speaker 分别拥有独立 subject；候选继续经过最终 memory gate，非敏感的第三方事实可写入其人物记忆，但不能串入用户本人或其他人物。
- 主 LLM runtime 走 DeepSeek/OpenAI-compatible API，配置解析和 client 创建集中在 `llm_runtime.py`，不保留本地模型默认值或 LLM legacy fallback。
- 包内 `ai_glasses_memory_assistant/README.md` 提供所有 Python 文件职责速查，和 `docs/context/code-map.md` 互补：前者按文件名反查，后者按任务找入口。
- `docs/context/assets/system-overview-pipeline.png` 提供系统总览；统一音频细节由 `frontend-audio-data-flow.mmd/.svg` 维护。

当前不是生产级硬件眼镜 runtime、完成签名和耐久验收的原生手机 App、已验证 24 小时的 always-on audio runtime、主动提醒推送系统、可靠 worker 队列或多租户服务。

Android 本地 demo 已可构建并安装到 XREAL X4000；Android 只新增平台外壳，继续复用同一套 Python 记忆、planner、SQLite、LLM 和 audit 核心。真机已通过 DeepSeek 文字聊天、模型原子安装、五组件 sherpa 自检、短时锁屏收音和停止释放，但还不能据此宣称 24 小时可交付。

当前 Android WIP 已具备 arm64 APK、前台麦克风服务、持久化 final 队列、联网恢复、partial UI、三段原生声纹录入、保守重叠判断、原生 TTS、模型自检、sherpa 1.13.4 五模型包和手动加密诊断导出。诊断包包含脱敏数据库快照、audit、设备/电池/内存和版本状态，不包含 API key、原始 PCM 或声纹向量。尚未完成的发布门禁是：真人三段声纹、真人唤醒/问答/TTS 回声、断网与权限等异常矩阵、签名分发、8/24 小时耐久和多厂商验证。

单元测试定位为核心保险丝：默认覆盖启动配置、存储、聊天主链路和 fake backend 音频契约；真实模型慢测不进入默认门禁。

私人多人开发骨架包括：`CONTRIBUTING.md` 说明分支协作和提交前验证，`.env.example` 提供本地配置模板，`.github/workflows/ci.yml` 在 GitHub 分支/PR 上运行最小核心门禁。

- LongMemEval 评测本地化已落地（2026-08-20）：所有 longmemeval 跑法（含 `resume_eval_500.sh` 续跑）已统一走 A100 本地 qwen3.8，零 DeepSeek 费用；新增硬约束——“新增/修改评测脚本必须导出 `AI_GLASSES_LLM_*` 本地值，禁止回退 deepseek”。评测链路与 Android app / 桌面 Web 配置独立、互不影响；完整离线评测步骤见 `AGENTS.md → LongMemEval 离线评测`。

## 文档规则

`docs/context` 只保留三份当前态开发文档：

- `docs/context/README.md`
- `docs/context/code-map.md`
- `docs/context/system-flow-current.md`

这三份文档只写当前系统怎么启动、怎么改、怎么验证、边界在哪里；不写历史开发过程、迁移流水账、调研过程或复盘材料。

包内文件职责速查放在 `ai_glasses_memory_assistant/README.md`，不计入 `docs/context` 三份文档上限。

`docs/context/assets/` 只保存当前架构图资产：`system-overview-pipeline.png` 是统一总览，其他图片只作为总览节点的细节展开，不新增专题说明文档。

## 结构借鉴边界

参考外部 agent memory 项目时，只借鉴适合当前 Python 本地 demo 的工程边界，不照搬发布型插件仓库结构。

- 暂不把当前包迁到 `src/` 布局。当前 `ai_glasses_memory_assistant/` 已经是正式 Python 包入口，`pyproject.toml`、根目录兼容 `server.py`、测试和文档都围绕这个路径工作；现在整体迁入 `src/` 只会制造大规模 import/启动/打包 diff。
- 可以保留并规范根目录 `scripts/`。只放可重复执行的本地诊断、数据迁移、benchmark 准备、清理扫描等工具；不放一次性补丁、临时 bugfix 流水账、旧调研材料或需要长期阅读的设计说明。`scripts/scan_cleanup_candidates.py` 是当前仓库内的只读清理体检入口，用于分级输出可再生缓存、需人工复核候选和禁止自动清理边界。
- 如果未来确实需要 `src/` 布局，必须先有发布/安装/多包隔离的明确需求，再单独开迁移计划和兼容验证，不混入普通结构治理刀。

## 当前优先级

### 实施完成、真实模型运行待执行：通用背景音频记忆闭环评测 V2（2026-08-31）

- [x] 唯一正式入口 `scripts/run_eval_ali_offline.py` 已改为 V2-only：第 1 通道 WAV 仍以 256 ms PCM16 小帧走真实 `start_audio_session` / `push_audio_session` / `stop_audio_session`，但每个环境源隔离为独立用户和虚拟日期，不构造跨源“每日回顾”。
- [x] 版本化人工金标 `data/benchmarks/eval_ali/ambient_memory_v2_gold.json` 固定 8 个 75 秒高语音密度窗口与 16 个后续问题；运行时校验窗口、TextGrid 证据文本和金标哈希，金标绝不输入 ASR、归档或 chat。
- [x] `smoke` 给出关键事实 ASR、topic/day overview、Timeline→topic evidence ID 溯源、真实 `GlassesChatService.chat()` 问答和 `memory_saved == 0` 闭环硬分；`full` 额外给出 8 段完整源音频的 CER、VAD、匿名说话人诊断、归档和吞吐健康报告。AliMeeting 仅是当前环境音来源，成绩不等同官方 M2MeT，也不代表真机声学效果。
- [x] 归档等待以 `ready` / `failed` 为终态；超时再读一次后报告 `ready_late` 或 `incomplete`，两者均不计入每日回顾或问答成功。本地 Qwen judge 只输出诊断，不影响硬分，也拒绝云端配置。
- [x] V2 resume 与三次 full-run 基线要求相同 manifest、金标哈希、源哈希、运行模式和 runtime；基线同时锁 health 中位数与 closure 结果。专项单测覆盖金标、关键证据 ASR、归档终态、resume 和基线。
- [ ] 用明确本地 Qwen 配置分别运行 `smoke`、三次 `full` 锁基线和 `--realtime smoke`；真实 run 只能证明本地软件链路，不能作为 Android 麦克风验收。

### 实施完成、候选模型实跑与真机安装待执行：Android 多语 VAD / ASR 轻量替换与 Eval_Ali 对比（2026-08-31）

- [x] 环境音 profile 已显式拆为 `legacy`、`sherpa_2024`、`sherpa_ten_2024`、`sherpa_silero_2025`、`candidate`：Mac 仍可复现旧 Python Silero + FunASR 基线；Sherpa profile 以相同 ONNX 工件分别完成原 Android 2024、Ten VAD-only、SenseVoice 2025-only 与组合的 A–E 消融。profile 会连同 VAD/ASR 文件 SHA-256、关键参数和 sherpa 版本写入 V2 manifest；resume 与三次 full 基线会拒绝不同 profile。
- [x] Android `SherpaVadAdapter` 已支持 `silero_vad` / `ten_vad` manifest engine，ambient ASR 接口保持 `sense_voice`、`language=auto`、ITN，不影响 online ASR、KWS、声纹或个人记忆策略。候选包组装工具对 ambient INT8 ASR 实施 300 MiB 上限；尚未将候选模型设为默认包。
- [x] 新增 `scripts/prepare_eval_ali_android_replay.py` 生成同一 8 个金标窗口、channel 0 的 16 kHz 单声道调试 WAV；Android 设置页可只导出最终事件时间/文本/profile/耗时 JSON。`--android-events-jsonl` 用同一 scorer 只生成 Android VAD/ASR health，禁止 PCM/WAV，绝不执行 Timeline、归档、聊天或个人记忆，不能宣称闭环或麦克风验收。
- [ ] 下载并校验 SenseVoice 2025 INT8 工件后，以 `scripts/assemble_eval_ali_android_candidate_pack.py` 生成新本地包；在真机确认自检、无队列溢出和回放 JSON 无音频，再覆盖安装。
- [ ] 用户前台运行固定 channel 0 的 A–E smoke（legacy、Sherpa 2024、Ten-only、ASR-only、combined）和 Android F health；只有 E 比 B 的 CER 或关键事实通过数更好、VAD recall 不降、归档/隐私/QA 不回归，才把 Ten VAD + 2025 SenseVoice 设为 Android 默认。

### 实施完成、真机覆盖安装验收待完成：Android 完整使用数据快照与 Agent 分析入口（2026-08-28）

- [x] 保留 Android 设置页原有的加密、脱敏诊断导出；删除仅能拉取脱敏诊断的旧 ADB Python 工具，改为 `android/tools/pull_device_usage_snapshot.py`。
- [x] Debug APK 新增独立的完整使用数据快照桥接：导出前工具停止持续收音并等待事件队列、记忆任务和讨论归档的终态；超时不生成部分快照。
- [x] 完整快照一致性备份 `timeline.db`、`events.db`、`sessions.db`，保留原问题/回复、最终 ASR、回复反馈、讨论日/话题和结构化记忆，以及完整 audit；不改动手机的真实数据库。
- [x] 包内生成 `manifest.json`（版本、哈希、记录数、排除项）、`feedback_index.json`（需改进优先、按 `turn_id` 关联原问题/回复/备注）和 `analysis_request.md`（Agent 取证与归因合同）。
- [x] 仅精确移除 API-key/authorization 值、原始 PCM/编码音频载荷、声纹 profile/enrollment/embedding；不再宽泛脱敏正常产品文本。Mac 端产物设为当前用户私有，工具不会上传或自动发送数据。
- [x] Python 导出、ADB 路径/损坏 ZIP/临时文件清理与 Kotlin Debug 路径策略测试通过；相关文档改为新的使用命令。
- [x] 已在连接的 Android 真机以 `adb install -r` 覆盖安装 Debug APK，并实际导出完整快照；快照包含 29 条原始对话、124 段原文、5 条回复反馈（3 条需改进）、1 份每日回顾和 6 个讨论话题。手机端三份数据库与 audit 仍存在，临时导出缓存已清理。

### 实施完成、Android 覆盖安装验收待完成：聊天内“回答依据”（2026-08-27）

- [x] 主聊天不再把“查看本次依据”渲染到隐藏的记忆管理面板；改为回复气泡下方的只读“回答依据”卡片。
- [x] 首屏展示最多 3 条原始片段和时间/来源/说话人提示；单条展开后才显示完整转写，并明确提示 ASR 错误和未知说话人不代表用户本人。
- [x] 依据加载中、失败重试、原文过期或已删除均有可见状态；聊天入口不提供删除操作，记忆管理中的原文维护能力保持不变。
- [x] Debug APK 已构建，并已从 APK 内确认带有 `mobile-ui-4`、内联依据卡和 ASR 提示。
- [ ] 以 `adb install -r` 覆盖安装到当前 Android 调试机，确认既有模型包和已收录数据保留，并人工点击验证内联展开、收起和错误状态。

### 实施完成、官方子集门槛通过：LongMemEval Reader96 第一批执行可靠性（2026-08-24）

- [x] 第一批只处理 27 道 Reader 账本执行/输出失败题，不改写入、召回、证据筛选、`PreReplyDecision` 或第二至第四批语义能力。
- [x] complete-set 中间批次只分类来源和提取事实；模型临时总数与临时最终答案不再作为校验条件，最终值由共享 `Decimal` 账本重算，最终措辞必须包含核算值。
- [x] 新增独立 `source_decisions`：每个来源只分类一次，而事实项可在同一来源明确包含多项事实时复用该来源；旧 item-only 账本继续兼容。
- [x] 产品与 LongMemEval Reader 统一区分 `answered`、`insufficient_evidence`、`execution_failed`，格式纠正最多重试一次；provider/JSON/账本/最终输出失败不再伪装成“没有记忆证据”。debug 记录失败阶段、选中来源、每次输入长度、耗时和校验错误。
- [x] 新任务计划内提供只读动态回放工具：依据旧归因和旧回放错误自动选第一批，不写死题号；按题型动态抽取 30 道既有正确对照题，Reader 与官方协议 judge 固定本地 qwen3.8。
- [x] 用户完成第一批本地 Qwen Reader + 官方协议 judge：27 道目标从 0/27 提升到 14/27，30 道正确对照保持 30/30，运行错误 0；第一批已独立提交为 `b1d207c`。

### 实验未过门槛、业务代码已回滚：LongMemEval Reader96 第二批聚合与去重（2026-08-25）

- [x] 第二批只处理归因为 `aggregation_or_dedup_error` 的 17 题，不改写入、召回、数据库、HTTP 或 `PreReplyDecision`；冻结 pre-baseline 为目标 4/17、同组对照 29/30、第一批相邻组 13/27，74/74 完成且运行错误 0。
- [x] 首轮实现曾新增 `entity|event|action|fact` 字段和 `source_group_id`；post 失败后的代码审查确认：basis 没有确定性执行语义，source group 只是可能包含多事实的 provenance 关联，不能当事实身份。该能力不得描述为已完成，整改时应撤回无效 schema/prompt/debug，而不是继续补规则。
- [x] 首轮曾把新账本 `count` 固定为每键计 1，并允许 item 级状态修订；回放证明它会丢失紧凑数量/分组频次，并让来源相关性与事实状态互相冲突。该语义已撤回，完整取消、纠正和当前状态处理推迟到第四批。
- [x] 收缩重写曾尝试保留 canonical-key 去重、`Decimal count|sum|average`、单位归一及共享 prompt builders；这些行为未通过官方 paired gate，现已随第二批业务代码全部撤销，不能描述为当前产品能力。
- [x] 用户明确授权 Codex 复现测试报错后，定向测试 `102 passed`、全量测试 `392 passed, 2 skipped`、编译检查与 `git diff --check` 均通过。五来源 average 用例的失败来自测试夹具按四来源分批却只提供一批模拟响应，已仅修正夹具；并行用例的 macOS 沙箱权限报错在沙箱外复验通过，未为环境限制修改生产代码。
- [x] 用户完成 paired post-replay：74/74、运行错误 0，但目标组 `4/17 -> 2/17`，第一批相邻组 `13/27 -> 8/27`；对照组虽保持 `29/30`，实际发生 1 道翻正和 1 道翻错。目标执行失败 `2 -> 8`，相邻组 `3 -> 10`，因此除运行错误外其余门槛全部失败，当前修改不得提交。
- [x] 收缩重写修复首轮两类主要执行回归：目标组执行失败从 8 降到 1，目标分数从 2/17 恢复到 4/17；第一批相邻组从 8/27 恢复到 13/27。但这只是恢复冻结基线，不是净提升。
- [x] 完成并实施过收缩性代码审查，但最终回放仍无净提升；该实现只作为失败实验记录保留在 planning/reports，不保留在生产、评测或测试代码中。
- [x] 用户完成 `post-consolidated` paired replay：目标组仍为 `4/17`（1 翻正、1 翻错），对照组 `29/30 -> 28/30`（1 翻正、2 翻错），第一批相邻组仍为 `13/27`（1 翻正、1 翻错）但执行失败 `3 -> 6`。只有运行错误为 0 的门槛通过，因此不得提交或进入第三批。
- [x] 已按锁定兜底方案精确恢复 11 个业务、测试和当前态文档文件到 `b1d207c`；未使用 `reset`，未移动历史，三轮 74 题回放、分析脚本和 planning 记录全部保留。
- [x] 17 题重新分为：4 道第三批义务/计算候选、9 道第四批时间/状态候选、4 道冻结正确对照；其中 5 道增加 `identity_overlap` 次级标签。身份关系合同只形成设计与通用测试规格，本轮不实现新 schema、prompt 或运行时规则。
- [x] 回退验收完成：恢复清单与 `b1d207c` 无差异，conda 编译通过，全量测试 `382 passed, 2 skipped`，`git diff --check` 通过；三份回放仍各 74 行且 SHA-256 未变化。当前跟踪修改只剩本计划结论。

### 规划完成、待授权：LongMemEval Reader96 第三批回答义务与结构化计算（2026-08-25）

- [x] 已区分两个同为 37 题的历史集合：早期 37 题是 36 道充分证据拒答加 1 道 provider 失败；第一批之后的剩余 37 题是 10 道拒答、10 道偏好义务遗漏、7 道综合遗漏和 10 道无证据却作答。
- [x] 当前合同审计确认：`PreReplyDecision` 是回答意图和义务的唯一语义权威，`TurnPlan` 只做非语义执行映射，`AnswerDirective`/Reader 不得自行增加身份、限定条件、时长或计算义务；现有账本只支持 `count|sum`，没有经过授权的 average 路由。
- [ ] 第三批按独立门禁推进：3A 已完成现有 `answer_obligations` 传递；旧 3B 的全局充分性自证已判定失败并回退；3B-R 改为端到端来源证据链；3C 仍须另行批准，才增加由 `PreReplyDecision` 所有的向后兼容计算 operation，验证 average。
- [x] 实施前先由动态脚本冻结 99 题 pre：剩余 37、重新归类 4、`681a1674` 实体/计数对照、第一批相邻 27、既有正确对照 30。题号只存在 planning/eval 选择中，不进入生产代码、prompt 或通用测试。
- [x] 3A 只有在 20 道义务目标净提升、翻正多于翻错、`681a1674` 保持正确、30 道正确对照零回归、第一批相邻组分数和执行状态均不下降、全部运行错误为 0 时才可进入 3B；任一门槛失败即停止并按 paired 证据归因，不叠加补丁。
- [x] 用户已授权 3A 边界并完成本地实现：只完善现有 `PreReplyDecision.answer_obligations` 组合规则和固定合同传递；未新增第二 planner、关键词 fallback、隐藏召回、benchmark 特判，也未修改写入、召回、数据库或 HTTP 接口。
- [x] 已完成 eval-only 99 题回放工具：严格校验 17/20/3/1/1/27/30 七组数量与零重叠；pre 保存当前 PPD 合同与 combined 结果，post 对 20 道义务目标额外复用冻结 pre 合同跑 Reader-only，区分合同生成和 Reader 执行收益。静态选择、conda 编译和 `git diff --check` 通过，生产代码仍未修改。
- [x] 用户重启本地 llama 服务后已用不变脚本完成 resume：append-only pre 为 105 行、99 个唯一题号，最新状态 99/99 完成、运行错误 0、PPD 错误 0、模型配置一致。冻结哈希为脚本 `161cbf4...`、pre JSONL `2f749b1...`、summary `9b3267c...`；不得再次续跑或改写该 pre，3A 生产实现门禁现已开放。
- [x] 3A 代码与通用测试已完成：PPD 用非 benchmark 组合示例区分纯计数、对象、规格和时间义务；产品/eval Reader 为四类义务补齐等价执行提示；complete-set 最终措辞接收完整固定合同并在证据支持时覆盖所需标签，不新增 schema 或严格字符串校验。定向测试 100 passed，已知并行用例沙箱外 1 passed；全量测试 386 passed、2 skipped，conda 编译与 `git diff --check` 通过。
- [x] 用户已完成本地 Qwen paired post，3A 全部门禁通过：99/99、运行/PPD 错误 0、召回哈希不变；20 道义务目标 `1/20 -> 3/20`（2 翻正、0 翻错，执行失败 `1 -> 0`）；Reader-only 冻结合同同样 `1/20 -> 3/20`（乐器、模型套件翻正，0 翻错）；`681a1674` 保持正确，正确对照 `24/30 -> 26/30` 无回归，第一批相邻 `19/27` 且执行失败仍为 2。无 byte-identical judge 翻转。3A 形成独立未提交检查点，不叠加补丁、不自动进入 3B。
- [x] 旧 3B 已完成 paired post，但正式门禁失败：所谓“充分证据”两侧都停在 `1/10`，无证据侧虽由 `1/10 -> 6/10`，仍出现正确对照和第一批相邻题回归。根因是一个全局 `support_status` 仍允许模型笼统自证，且旧充分组中 9 题的最终 Reader 输入实际缺少答案关键维度；该实验不得提交或继续叠补丁。
- [x] 已将失败 3B 的生产、eval、测试和当前态文档精确恢复到已提交的 3A `e39ef07`；保留本计划、任务 planning、3A/3B pre/post 产物，并把旧 3B 作为失败实验留档。
- [ ] 3B-R 第一检查点：在唯一一次 PPD 内新增向后兼容的内部 `answer_requirements`，由解析器按顺序分配 `r1...r8`，原序传过 `TurnPlan`、`AnswerDirective` 和 eval answer task；当前不进入 Reader prompt、不触发补充召回，先由用户本地 Qwen 冻结 99 题 `contract-pre`。
- [x] 首次 `contract-pre` 已冻结且结构门禁通过（99/99、运行/PPD 错误 0、188 项 requirement、最大 4 项），但语义门禁未通过：1 行含被解析器丢弃的非法 requirement、1 个检索 query 含未解析占位符，且 16 道个性化推荐中至少 14 道把可选偏好/限制/近期历史拆成全部必需项，会系统性制造过度拒答。该产物只保留为失败合同证据，不进入来源水合或 Reader 实施。
- [ ] 仅修订通用 PPD requirement 语义后重新冻结：事实/计算维度继续逐项必需；个性化推荐用“至少一个直接支持的相关偏好、限制、已有物品或既往经历”作为最小个性化锚点，额外偏好不在当前全必需 schema 中拆成可选项；检索 query 必须独立、自包含且不得含待替换占位符，非法 requirement 必须使 contract-pre 门禁失败。
- [x] 上述 v2 通用合同修订和 eval-only 冻结校验已实现：未新增字段、枚举、planner 或运行时关键词规则；新增非 benchmark 测试覆盖合取语义、单一推荐锚点、非法 requirement warning、ID 顺序和占位符拒绝。等待新路径 `contract-pre-v2` 后再勾选重新冻结项。
- [ ] `contract-pre-v2` 首次运行 98/99 完成：15 道个性化推荐全部收敛为单一最小锚点，语义修复通过；唯一错误是模型仍生成了依赖起始日期的方括号占位 query，新结构门禁已正确拒绝。保持脚本不变、同路径 `--resume` 仅重跑该题，完成后再冻结最终哈希。
- [ ] `contract-pre` 冻结且人工审计通过后，才实施来源水合、逐 requirement 授权检索、来源/原文绑定验证和 Reader support map；缺记录、空 ledger 或数据库扫描完成均不得自行推出现实世界为零。

### 实施中：iPhone 全功能记忆助手基础（2026-08-03）

- [x] 新增独立 `AIGlassesMemoryAssistant` iOS 16+ Target，保留 `AIGlassesMicProbe` 作为严格 HFP 路由验证器。
- [x] 新 Target 已具备用户显式启动、唯一实际 Bluetooth HFP 输入验证、路由/中断 fail-closed 停止、后台 audio 声明和 TTS 期间暂停收音的原生骨架；当前帧不会写入业务存储。
- [x] iOS 已改用 Android 同款 `model_pack.v1`：S22 已验证的 `x4000-sherpa-1.13.4-v2` 12 个文件在本机开发包构建时校验大小与 SHA-256 后打入 App；`sherpa-onnx v1.13.4` 和 `onnxruntime 1.27.1`（API 27）已完成依赖锁定和校验脚本验证。
- [x] 新增 Keychain API key 存储、嵌入 Python 运行时 fail-closed 边界，以及 Swift 本机 SQLite POC：只有通过说话人、重叠、隐私和敏感词门控的 `final` 才能写入 Timeline/显式记忆；`partial` 不写 Timeline、记忆或 audit。
- [x] 新增本机脱敏诊断快照，以 Android 相同的 `AIGDIAG1`、PBKDF2-HMAC-SHA256 和 AES-GCM 格式加密，并通过系统分享面板导出；快照不含 API key、PCM、声纹或 embedding。
- [x] 首页已复用 Android `static/` 网页并通过 `WKScriptMessageHandlerWithReply` 接入 iPhone 原生异步桥接：独立 Keychain owner ID、TTS、原生定位、模型/路由状态和右上角/网页高级设置均走同一原生配置页。App 启动或用户保存有效配置后，后台启动 `mobile_runtime.start()` 并通过 WKHTTPCookieStore 设置 `ai_glasses_local_token`，WebView 加载 Python localhost HTTP 地址而非 `file://` 页面。
- [x] iOS 原生音频事件已接入共享 Python runtime API：`start_capture`、`set_device_state`、`ingest_audio_event`、`wait_audio_event`、`stop_capture`，网页聊天、音频 final、memory job、Timeline 和 audit 使用与 Android 相同的 HTTP/数据路径。
- [x] 前台收音骨架使用非主线程加载 5 个 sherpa-onnx 模型（~290 MB），增加 `idle/loading/ready/failed` 状态机防止重复点击启动多个管线；模型创建失败转成可见错误返回网页，不让 C/C++ 依赖错误打崩 App。模型包校验结果已缓存，`webStatus()` 不再每次同步计算 SHA-256。
- [x] 暂停、路由变化、中断、媒体服务重置和离开前台时统一停止 pipeline、AudioEngine 和 Python capture。
- [x] MIC 输入策略：新增 Keychain 持久化 `allowPhoneMicFallback`，默认关闭。存在唯一 `bluetoothHFP` 时优先使用；无 HFP 且开关关闭时拒绝启动并说明原因；开关打开时使用 iPhone 内置麦克风。状态明确返回 `input_device_type`、`input_device_source`、`input_device_name`，网页显示"蓝牙 HFP"或"iPhone 内置麦克风"。
- [x] 依赖与构建：固定 `sherpa-onnx v1.13.4` 对应 ONNX Runtime API 27（`onnxruntime 1.27.1`），移除旧 ORT Embed Frameworks 引用，`onnxruntime.xcframework` 仅作链接输入不重复嵌入。新增 `ios/tools/verify_ios_dependencies.py` 校验 sherpa 版本、ORT `ORT_API_VERSION`、`CFBundleVersion`、arm64 slice 和关键文件 SHA-256。`build_and_install.sh` 使用完整 Xcode 路径、显式 iPhone destination、独立 DerivedData 路径，构建后检查 App 实际链接的 ORT 版本。
- [x] `ios/tools/build_numpy_ios.sh` 已实际交叉编译 arm64 `numpy 1.26.2`（19 个 native extension，含 `_multiarray_umath`），构建阶段自动复制到 PythonRuntime 的 `site-packages`；verifier 同时检查依赖目录、App bundle 和 arm64 slice。
- [ ] 仍需在可用的 iOS Python 环境执行 `import numpy` smoke check、完成五组件设备自检、签名安装和 Insta360 Mic Pro 三次真人验收。当前仅有主机 iPhoneOS Debug 构建和 bridge 行为契约测试，不能宣称真机聊天/记忆/HFP 闭环完成。

### 实现完成、真机模型验收待完成：Android 离线音频 VAD/ASR 测试（2026-07-27）

- [x] Android 高级设置可选择本地 PCM16 WAV 或 Lark 常见 AAC M4A；M4A 通过 Android 系统解码器输出 PCM16。输入只在内存中下混、重采样为 16 kHz 单声道，单次最长 15 分钟。
- [x] 用户默认使用原音频；可明确启用 `-24 dB` 到 `+24 dB` 的 1 dB 步进增益。页面显示增益前后峰值和削波警告，不生成增益后的音频文件。
- [x] 离线测试直接复用原生 Silero VAD 和 SenseVoice：逐段显示时间、语言与文本，并显示合并转写；不会进入 KWS、声纹、`audio_event.v1`、Python、Timeline、SQLite、audit、memory job 或诊断导出。
- [x] JVM 测试覆盖 PCM16 WAV 解码、共享 PCM 规范化、单/立体声、重采样、截断/不支持/超时长输入和可选增益限幅；`lintDebug` 与 `assembleDebug` 已通过。
- [ ] 在已安装完整模型包的 Android 真机上分别使用原音和增益的 Lark AAC M4A 验证：系统解码、VAD 片段、逐段转写、复制结果、取消，以及 Timeline、记忆、audit 和诊断包均无本次测试内容。

### 实现完成、蓝牙独占真机验收待完成：Android 收音设备路由（2026-07-27）

- [x] 全天收音、唤醒、声纹录入与设置页测试共用 16 kHz、单声道 PCM16、`VOICE_RECOGNITION` 和输入路由策略；不启动新的 Python、Timeline、SQLite、audit 或 memory job 路径。
- [x] 没有蓝牙输入时允许系统/手机收音；恰好一个蓝牙输入时必须通过 `setPreferredDevice()` 和实际 `routedDevice.id` 双重确认，无法路由时停止收音而不回退到手机麦克风；多个蓝牙输入拒绝启动并列出候选设备。
- [x] 设备增删会重建录音器并重新应用策略；前台通知、原生状态和 Debug 显示当前实际设备名称、类型与蓝牙/系统来源，原始 PCM 仍只在内存中处理。
- [x] JVM 单测覆盖无蓝牙系统输入、单蓝牙选择、双蓝牙拒绝、偏好设备后路由不一致、蓝牙竞态接入、无音频帧、声音不足和权限/初始化错误。
- [x] 设置页收音测试固定采样五秒，显示本地在线 ASR 的增量/最终文字；路由正确且有 PCM 时，识别到文字或 ASR 未就绪均不会把硬件收音误判为失败。最多五秒的测试 PCM 只在设置页内存中保留到用户播放一次、重新测试或离开页面，随后清零；音频和转写均不进入 Python、Timeline、SQLite、audit 或诊断包。
- [x] 新增 `android/tools/build_and_install_debug.sh`，自动检查 JDK 17、Android SDK、ADB 设备，构建 Debug APK、覆盖安装并启动；多设备时要求显式 `--serial`，不猜测目标设备。
- [ ] 在至少一台连接蓝牙收音设备的 Android 真机上验证：无蓝牙系统输入、唯一蓝牙强制路由、蓝牙断连后重建、双蓝牙拒绝、路由失败停止服务、全天收音/声纹录入/设置测试三入口一致。三星 S22 与 Insta360 Mic Pro 另需保留三份 `android_acceptance.v2` 报告：仅系统蓝牙、Insta360 App BLE 并发、官方接收器 USB；每份都必须以实际 `input_device_source` 判定，手机麦克风回退即使可听也失败。该测试只证明当前收音链路，不替代真人唤醒、ASR、记忆、回复和 TTS 验收。

### 已完成：Android 测试数据自动拉取（2026-07-22）

- [x] 新增 Mac ADB 命令行工具，自动选择单设备或要求显式 serial，正常停止全天收音，等待音频队列和对应 memory job，再拉取脱敏完整快照。
- [x] Debug WebView 原生桥接只在 app 私有缓存生成一次性 ZIP；Release 构建拒绝调用，路径固定校验，Mac 拉取后优先桥接删除并以受限 `run-as rm` 兜底。
- [x] Mac 目录保留 bundle、解压数据库/audit、采集元数据、Codex handoff 和 `latest.json`；目录已由现有 `android/captures/` 规则排除 Git，不自动删除历史证据。
- [x] 拉取元数据不重复保存 final query 或 partial 文本；诊断包继续移除 API key、raw PCM、声纹/录入样本和 embedding，同时保留问题定位所需文字与门控原因。
- [x] Python 目标测试覆盖设备选择、停止终态、失败、二进制传输、ZIP/路径安全、远端清理和 latest 指针；Android 纯策略测试覆盖 Debug-only 与临时路径规则。
- [x] 已覆盖安装最新 Debug APK 到三星 SM-S9010，并在正在收音状态运行工具：capture 正常停止且未重启，队列归零，三个 SQLite `integrity_check` 与 ZIP 校验通过，手机临时 ZIP 已删除；本次零片段短测正确走无 memory job 分支。

### 实现与浏览器验收完成、真机验收待完成：Android 与移动端界面优化（2026-07-22）

- [x] 手机聊天页顶部收敛为标题和齿轮；原有体验者、声纹、播报、定位、记忆和 Debug 控件复用同一套事件逻辑，在手机端进入全屏应用设置中心。
- [x] 手机记忆管理改为带返回箭头的全屏二级页面；设置、记忆、Debug 和声纹共用前端返回栈，Android 系统返回键优先关闭当前二级界面。
- [x] 声纹录入移除 Hermes 展示，三段样本分别显示不同朗读内容；重新录入从第 1 段开始，不改变 embedding、三样本聚合或 API 协议。
- [x] 全天待机主区域只保留短状态、片段数量和启停按钮；capability、capture、片段 ID、音量和 VAD 等详情移入设置中心与 Debug。
- [x] 原生高级设置页增加可见返回工具栏、中文字段与账户/模型/诊断分组，保存、自检、下载和加密诊断导出行为不变。
- [x] 完成 `390x844`、`412x915` 和 `1280x720` 浏览器截图与交互验收；设置、记忆、声纹和返回交互无重叠或横向溢出，桌面双栏保持不变。
- [ ] XREAL X4000 已覆盖安装 2026-07-22 新构建，并通过设置中心展示、系统返回键回到聊天和主界面布局检查；原生高级设置入口、三段真人声纹切换及全天待机运行时可读性仍待人工验收。

### 实现完成、真机真人验收待完成：Android 记忆召回与环境音频写入门控（2026-07-22）

- [x] 非时间型个人 `specific_fact` 同轮检索相关 profile 和 event；无关 profile 不再阻断 event，明确时间/事件/计划查询仍保持 event-focused。
- [x] `ambient_audio_text` 正常 stop 不再因空 `speaker_label` 退回元数据缺失的旧文本导入；匿名片段继续携带 `audio_event_id`、chunk evidence、speaker、overlap 和 `memory_eligible`。
- [x] 未知/其他/环境说话人、重叠不明、memory ineligible、语义 noise、低置信和语义 backend fallback 均 fail closed；ASR final 仍逐条保留在 Timeline，不持久化 raw PCM。
- [x] recall debug 增加 `cross_kind_recall`，音频 memory job 增加逐片段 `unit_gate_results`，可从 audio event 和 chunk 定位 saved/rejected 原因。
- [x] 新增跨类型事实、无证据/冲突证据、时间查询、Android 未标注/缺元数据说话人、混合片段和语义质量回归测试；最终专项结果 `7 passed`。
- [ ] 当前开发环境缺少 Java 17、Android SDK/ADB，Gradle 构建、X4000 owner 解析、结构化测试记忆软删除和真人验收待在设备工具链可用后完成。
- [ ] 在 XREAL X4000 上重置当前 owner 的结构化测试记忆，重新加入“我买车了 / 我的车停在楼下”，完成人真声纹、唤醒、ASR、召回、回复和 TTS 验收。
- [ ] 现有 3 条 subject recall 失败仍需单独修复：self 默认范围、`all` scope 优先级和歧义 provisional debug；本次不扩大到该既有问题。

### 实现完成、真实模型验收待完成：统一音频处理核心（2026-07-16）

- [x] 将 `.external/voice-recording` 的 VAD、KWS、流式/整段 ASR 和声纹能力抽取到主包，不接入外部 `main.py` 或记忆系统。
- [x] 外部源码复用基线记录为提交 `f63fb179466584fbb2f7c131997adec761d4ff85`；`.external/` 整体忽略，不进入主仓库提交。
- [x] 增加浏览器 PCM audio session、统一结构化事件和 service 消费边界；partial 只进 UI，final 才能进入聊天、capture 或声纹录入。
- [x] 收敛旧 blob、实时 PCM 和声纹录入的模型所有权，修复 `.env` 初始化、句首/句尾完整性和主动 session 回收。
- [x] 将网页收敛为单一全天待机按钮和 KWS 两段式唤醒，删除手动唤醒与直接语音提问主路径。
- [x] 修复同一端口并存 HTTP/HTTPS 时 localhost 和局域网命中不同协议的问题；server 改为独占绑定，并明确本机 HTTP 与局域网 HTTPS 的使用边界。
- [ ] 完成 fake backend 门禁后保持“实现完成、真实模型验收待完成”；真实 KWS/Paraformer 和现场麦克风/耳机通过后再标记完成。

### 代码加固完成、真实模型与设备验收待完成：持续音频输入闭环（2026-07-17）

- [x] P1-1：streaming sequence 响应缓存和 dispatch 幂等缓存共用集中上限；全天 session 长时间 push 不再让 service cache 无限增长，窗口内重复 sequence 仍只消费一次。
- [x] P1-2：final 聊天 dispatch 改为可查询 job，PCM push 不等待完整聊天生成；同 session 串行，重复 sequence 复用同一 job，stop/close/失败均有明确终态。
- [x] P1-3：唤醒回应和正式回答用 tokenized playback 状态同步服务端；合法播放期间不误回收，迟到 finished 不串台，断网漏 finish 仍会超时回收。
- [x] P1-4：service 层原子保证同一用户只有一个 active 音频 session；新 ambient/enrollment 接管旧 session，旧 token 失效且旧 capture 不触发记忆 job，不同用户隔离。
- [x] P2-1：分别计算并展示收音、全天转写、唤醒问答和声纹录入能力；模型缺失时对应入口明确不可用，API 不暴露绝对路径。
- [x] P2-2：声纹录入前 flush 尾包并以 pause reason 停止待机；capture 文本保留但不创建记忆 job，录入完成、取消或失败后只恢复原用户待机。
- [x] P2-3：流式 PCM、旧 blob 和声纹录入请求体使用集中上限；读取前校验 Content-Length，超限返回 413，非法/缺失/伪造长度不会无限读取。
- [x] P2-4：补齐匿名声纹从流式 final 到跨 capture 持久匹配、隔离和删除流程测试；`PRED_SPKxxxx` 默认保持匿名 provisional subject，只有用户显式命名时才合并为 named subject，API/debug/audit 不暴露 embedding。
- [x] 真实验收补丁：energy fallback 只表示“可以收音”，不再把全天转写、唤醒问答或声纹录入标为可用；service 在 takeover/capture 创建前拒绝不可用模式，避免环境噪声假片段和停止队列积压。
- [x] 当前环境可完成的真实模型/浏览器验收与降级修复已完成；fake backend 音频专项 46 passed，全量 110 passed，仍只保留 3 条既有 subject recall 失败。运行库和模型配置已补齐，需真人配合的声学项按下方边界继续保留，不宣称通过。

### 实现与新手机模型验收完成、真人唤醒待完成：Android 连说唤醒与可见问句（2026-07-22）

- [x] 原生音频状态明确区分持续记录、已唤醒、等待问题、正在听问题、问题已发送和唤醒超时；10 秒只约束独立唤醒后的提问等待窗口。
- [x] 同时支持“你好小忆，问题”连说和“你好小忆”->“我在，请说”->问题；连说时不播放确认语，不丢弃关键词后的 PCM。
- [x] 只有 final 问句进入聊天；界面立即显示去掉唤醒词的用户问句，并按 `event_id` 避免前台轮询和后台回复恢复造成重复气泡。
- [x] 保持 Python HTTP API、SQLite、`audio_event.v1`、声纹/隐私门控和 partial UI-only 边界不变。
- [x] Android 单测、lint、assemble、前端静态契约和音频专项通过；全量 Python 仍只有 3 条已记录的 subject recall 既有失败。
- [x] 新构建和 `x4000-sherpa-1.13.4-v2` 已安装到三星 SM-S9010；五项真实模型自检均为 `ok`，KWS 改为 `num_trailing_blanks=0`，score/threshold 不变。
- [ ] 新手机填写 API Key 后，真人分别验收连说、两段式、10 秒超时恢复和 final 问句只显示一次；随后在 XREAL X4000 复验同一组场景。

#### 真实模型与浏览器验收记录（2026-07-20）

已完成：

- 仓库 `.env` 中的 `AI_GLASSES_ASR_MODEL_DIR`、`AI_GLASSES_SPEAKER_MODEL_DIR` 已配置且目录存在；未自动下载模型。SenseVoice 和 Cam++ 均完成一次真实 CPU 加载/推理，Cam++ 返回 192 维 embedding；这只证明模型可运行，不代表真人语音准确率已通过。
- `hermes` 已安装 `silero-vad 6.2.1`、`sherpa-onnx 1.13.4`；Sherpa 中文 KWS 模型和 Paraformer streaming 模型已下载到 `/Users/huyaokai/Documents/model`，并在仓库 `.env` 配置。Sherpa 官方样例可真实命中，`你好小忆` 合成语音可命中自定义关键词；Paraformer 官方样例转写为“欢迎大家来体验达摩院推出的语音识别模型”。
- 浏览器 1280x720 和 390x844 均无横向溢出、关闭抽屉不遮挡聊天、控制台无 warning/error。补齐模型后的真实 capability 中五个后端均为 `ready`，`ambient_transcription_ready`、`speaker_enrollment_ready`、`assistant_query_ready` 和 `assistant_wake_ready` 均为 `true`；localhost 页面实际取得麦克风并完成全天待机 start/stop。
- 局域网 HTTPS 已使用覆盖 `10.2.30.121`、`127.0.0.1`、`localhost` 和本机 `.local` 主机名的 mkcert 证书启动；严格证书校验访问 `https://10.2.30.121:8765` 返回 200。浏览器在该 LAN 地址上实际取得麦克风，进入全天待机并缓存 3 段现场语境，正常停止后控制台无 warning/error。该地址只代表当前 DHCP 租约；长期固定需要网管或路由器做 DHCP reservation，同事设备需信任同一 mkcert 根证书。
- 收紧门禁前已用本机浏览器实际取得麦克风并完成 start/stop。该测试暴露 energy fallback 在约 20 秒环境噪声中产生 11 个假 final、停止等待约 80 秒；最终 `/stop` 为 200、capture 为 `stopped`、长期记忆 0、无 raw embedding 和音频文件。当前补丁已阻止缺 Silero 时再次进入这条不稳定路径。

仍需人工验收：

- 真实耳机回声、长问题、长回答、本人/他人/低置信声纹、疑似重叠、刷新/断网恢复需要佩戴者和第二位说话人现场配合；fake tests 已覆盖状态与副作用门禁，但不能替代真人声学结果。重叠语音仍只称“疑似声纹冲突”，不宣称完整 diarization。
- 本次浏览器停止待机时处理了 4 个现场片段，约 30 秒后正常停止；需要在真人长语音场景继续判断 CPU 推理队列耗时是否可接受，不能仅凭模型样例宣称实时性通过。

已执行的补齐步骤（完成）：

```bash
conda run -n hermes python -m pip install silero-vad sherpa-onnx

export AI_GLASSES_STREAMING_ASR_MODEL_DIR=/Users/huyaokai/Documents/model/paraformer-zh-streaming
export AI_GLASSES_KWS_MODEL_DIR=/Users/huyaokai/Documents/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
export AI_GLASSES_KWS_KEYWORDS_FILE=/Users/huyaokai/Documents/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/keywords_ai_glasses.txt

conda run -n hermes python -m ai_glasses_memory_assistant.server --host 127.0.0.1 --port 8765
curl http://127.0.0.1:8765/api/audio/capabilities
```

依赖安装、模型下载和路径配置已完成；capability 中 VAD/KWS/streaming ASR 均为 `ready`，四个工作流 readiness 均为 `true`。下一步录入佩戴者 3 段声纹，戴耳机依次验证“你好小忆”唤醒回应、10 秒内开始长问题、长回答 TTS、第二人/低置信/疑似重叠拒绝、说话中 stop 尾包、刷新、断网和恢复；最后检查无重复 chat/capture/memory job、无 raw PCM 文件、audit 无 embedding 向量。

### 已完成：全天讨论归档与回顾 V1（2026-07-20）

- [x] 保留最近 6 段作为“刚才”即时上下文；新增可恢复的处理切片、当天话题和每日概览，全天待机未停止时也能回顾早中晚内容。
- [x] ambient final 继续先脱敏写入 Timeline；后台按静音、时长、片段数、跨日、停止或回顾请求封段，不让 PCM push 等待摘要模型。
- [x] 同一天的同一话题允许跨多个时间段合并；每日概览按首次出现时间排序，并保留可核对的 chunk evidence ID。
- [x] 环境音频转写原文默认保留 30 天后物理删除，讨论摘要长期保留；用户可按日期分别删除原文、摘要或两者。
- [x] 增加 discussion recall、按日管理 API、每日回顾侧栏和“查看本次依据”，不改变现有音频 API 或长期记忆门控。
- [x] 验收覆盖 300 段、运行中回顾、跨切片同话题、重启/异常恢复、30 天证据过期、删除范围、隐私、用户隔离和桌面/移动端页面。

## 纯音频 AI 记忆眼镜长期路线（2026-07-20）

### 产品北极星与对标口径

本产品只通过声音理解现实，不增加摄像头或其他视觉输入。目标不是复制某个未公开的专有引擎，而是在公开可观察行为上形成并验证完整闭环：

```text
稳定听见
-> 正确分段和转写
-> 判断谁说了什么
-> 区分临时上下文、原话证据、长期记忆和动作候选
-> 按人、事、时间、任务组织
-> 在正确问题中召回并说明依据
-> 经用户授权后提醒或执行
```

达到“媲美”至少要求：同一组真实音频场景下，能完成自动记录、纪要、待办、长期召回、数据管理和办公动作闭环，并有可复现的质量证据。达到“超越”还要求：

- 每条重要结论能区分原话、摘要、模型推断和不确定信息，并能查看对应 evidence。
- 敏感信息、他人信息、低置信声纹和重叠语音默认 fail closed，不静默归入用户长期记忆。
- 用户纠正后能 supersede 旧结论，既不继续使用错误信息，也保留可解释的变更依据。
- 多人记忆在身份明确前保持隔离；同名、匿名和跨场景合并必须可解释、可确认、可撤销。
- 外部执行不绑定单一办公平台，统一支持预览、授权、幂等、状态查询、失败恢复、审计和撤销。
- 所有“效果更好”必须落到音频、记忆、召回、隐私、动作和长时间运行指标，不能只用演示或功能名称证明。

### 必须保持的机制

- `audio_event.v1` 是音频输入契约；`partial` 永远只进 UI/debug，只有合格 `final` 才能产生证据、记忆或动作副作用。
- 原始 PCM 继续只在有界内存中短暂存在，不默认落盘；脱敏 final 才能进入 Timeline。
- Timeline 原话证据、结构化长期记忆、discussion 派生摘要、动作候选四层分开管理，各有独立保留、删除和解释语义。
- 所有长期记忆仍经过 `MemoryWriteCandidate -> should_write_memory_candidate()`，新音频能力不得绕过门控、用户隔离、subject 隔离或 audit。
- 复用当前 `GlassesChatService`、`EventMemoryStore`、`TimelineStore` 和统一音频引擎，不另起一套 memory runtime、数据库或 HTTP owner。

### R0：完成当前全天讨论归档，建立可信基线

- [ ] 完成当前 discussion slice/topic/day、运行中回顾、30 天原文保留、按日删除和 evidence 展示的测试、文档与浏览器验收。
- [ ] 清除后台 worker/SQLite 生命周期告警，验证服务 close、临时目录、重启恢复、失败重试和并发写入不会留下悬空线程或半完成状态。
- [ ] 修复现有 3 条 subject recall 失败和多人待闭环清单，确保本人、命名人物、临时 speaker、`all` scope 和歧义行为一致。
- [x] P0：讨论归档 complete-set 已把 `PreReplyDecision` 的来源范围、图闭包和覆盖要求贯穿到确定性回答；`self/non_self/uncertain`、capture-local `spk_N` 与关联 capture 强制限制个人活动、人数和逐人发言结论。普通 complete-set Reader 失败会保留候选证据数量与授权范围，回放输出已增加脱敏的失败阶段和校验诊断。
- [ ] P0 验收：由用户运行相关 Python 测试和本地 Qwen 快照重放，确认三条 `needs_improvement` 分别实现 capture 全量环境总结、拒绝伪造参与人数/逐人发言、以及个人活动与未归属环境主题分离。
- [ ] 形成固定回归基线：300 段早中晚内容、跨时段同话题、运行中回顾、跨日、断网、重启、删除和过期均有自动测试。

用户效果：全天待机尚未停止时，用户问“今天上午讨论了什么”，也能得到按时间排序、可查看原话依据的回答，而不是只看到最近 6 段。

### R1：真实声学质量与长时间音频稳定性

- [ ] 建立真实音频验收集，覆盖安静室内、街道、车内、会议室、远近说话、方言口音、耳机回声、TTS 回灌、长问题、抢话和静音。
- [ ] 所有入口明确显示收音中、静音、暂停、断线、降级和处理积压状态；提供一键暂停、私密模式和停止后清理当前未归档内容的用户控制。
- [ ] 分别测量 VAD 漏检/误切、KWS 唤醒率/误唤醒率、ASR CER/WER、句首句尾丢失、final 延迟、队列等待和 CPU/内存；基线与发布阈值写入 eval，不散落到业务 hard code。
- [ ] 增加回声与播放感知、barge-in、网络抖动/乱序/重复/断线续传、背压和过载降级；任何降级都要公开 capability/reason，不能悄悄产生低质量记忆。
- [ ] 验证至少 10 小时连续待机：无未界定内存增长、无重复/丢失 final、无原始 PCM 文件、停止和恢复时间可接受。

用户效果：眼镜播报回答时用户插话，系统能停止播报并听清新问题；网络短暂中断后不会把同一句话记两次。

### R2：多人对话与“谁说了什么”

- [ ] 完成本人声纹真人录入和阈值校准；本人、他人、未知和低置信结果保持明确状态，不能用一个固定阈值假装适配所有人。
- [ ] 引入可评测的 speaker diarization/overlap 能力，但匿名 voice group 仍只是临时身份；跨 capture 合并和实名绑定必须经过用户确认。
- [x] Android 已有可单测的 capture-local 匿名轨道：同 capture 内可把相同匿名声音标为 `spk_01`，stop 时清空；重叠、低质量和本人不分配，陌生人 embedding 不落库。Java/Android SDK 可用环境仍需运行 JVM 单测和真机多人录音验收。
- [ ] 同一原话片段只能成为对应 speaker 的证据；重叠或归属不明片段进入待确认区，不自动污染任一人物记忆。
- [ ] 补齐人物改名、合并、拆分、撤销、同名歧义和“这是我说的/不是我说的”纠错流程及 audit。

用户效果：三个人开会后询问“谁承诺周五交方案”，系统能给出说话人和原话；无法判断时明确说“不确定”，而不是随便归给佩戴者。

### R3：记忆策略引擎升级

> 详见 `docs/memory-system-audit.md` 的差距分析。该文档在 iPhone App 开发完成后作为 R3 启动参考。

- [ ] 对每个 final/讨论摘要统一输出处理决策：临时上下文、Timeline 证据、长期记忆候选、动作候选或丢弃，并记录价值、置信度、敏感性、subject、时间、来源和原因。
- [ ] 在现有语义分类基础上增强人物、事件、时间、任务、决定、风险、偏好和关系变化抽取；不要恢复散落中文业务词表或把规则堆成第二个分类器。
- [ ] 增加重要性、重复度、新颖度、时效性和证据强度判断，支持同义合并、冲突并存、过期/衰减、correction、supersede 和 task 状态流转。
- [ ] 明确“记住”“不要记”“忘掉刚才”“以后提醒我”等用户指令的最高优先级，同时保持敏感信息确认和他人隐私边界。
- [ ] 建立记忆写入 precision、漏记率、错误人物率、敏感误存率、重复率和纠错后残留率 eval；敏感误存和未授权人物合并必须为零容忍门禁。

用户效果：“我以前爱喝咖啡，但医生让我戒了”不能留下两个同时有效的偏好；系统应保留变化历史，并在当前回答中使用“现在不喝咖啡”。

### R4：可核对的长期召回与记忆推理

- [ ] 按“刚才、今天、某天、跨日话题、某个人、某项目、某个任务”分别选择短上下文、discussion、Timeline、结构化记忆或文档，不把所有数据塞入 prompt。
- [ ] 回答中区分直接原话、派生摘要、结构化事实和模型推断；证据冲突、过期或缺失时主动降置信，不补写不存在的细节。
- [ ] 支持跨天同话题延续、任务承诺追踪、关系/偏好变化和事件时间线，同时保证查询范围、subject scope 与用户隔离。
- [ ] 运行 LongMemEval 和本项目纯音频场景集，衡量 recall precision/coverage、时间准确率、人物归属、evidence 支撑率、幻觉率和延迟。
- [x] 已完成 LongMemEval Oracle 兼容 runner、公平会话导入和通用 Subject Recall 修复：逐题隔离、按原始 session 保留 user/assistant Timeline 证据、仅让用户原话片段通过生产记忆门控、生产召回、统一 Reader、两字段 JSONL、独立召回字符详情、实时进度，以及 unresolved 名称在明确第一人称语境下回退 self 的安全规则；2026-07-27 已完成 500 题 Oracle 诊断运行，500/500 完成且无导入/运行失败，本地 answer hit 46.8%、recall hit 48.2%、空上下文 32 条；该结果只用于组件根因分析，尚未执行官方 judge。
- [x] LongMemEval 组件归因第一轮（2026-07-28）：建立仅离线使用的 routing/write-type/coverage-ranking/Reader/expected-abstention 分类；用非 benchmark 合成对话验证“按既有偏好给建议”会以 self profile 作为有界背景并保持普通建议零召回。生产链路只新增通用语义契约 `mixed + profile + summary`，debug 标记 `profile_context_for_llm`，不使用数据集字段或特判。30 条 Oracle 偏好切片全部完成，profile route 4->17、无召回空路由 19->11，但本地 answer hit 1/30->0/30、recall hit 3/30->0/30；因此只确认路由修复，不能宣称端到端评测提升，下一步独立审计偏好写入类型和候选覆盖。
- [x] 会话导入偏好类型修复（2026-07-28）：通用历史导入对已通过既有门控、且旧规则恰为 `event:event` 的片段，使用短生命周期统一语义分类做受限类型覆盖；只改 `kind`/`memory_type`，保留 subject、证据、时间、来源和全部既有门控，语义不可用、低置信、非写入、内容不对齐或安全门控不通过时回退旧类型。非 benchmark 回归验证稳定工作偏好、一次性事件、跨用户隔离、敏感信息和失败回退。Oracle 偏好切片仅作诊断：30/30 完成，44 条 active profile、profile route 17->18、非空 profile context 5->13、空上下文 23->10、无召回空路由 11->7，但本地 answer/recall hit 仍为 0/30，平均耗时 22.35s->125.95s。下一步应固定已有 profile 证据检查排序/上下文与 Reader，不扩大召回或新增 benchmark 规则。
- [x] LongMemEval 可恢复逐题诊断（2026-07-28）：runner 现为每题原子保存完整结果/debug、脱敏 audit、脱敏 SQLite 证据、离线阶段分类和受限候选排序追踪；`--resume` 仅在数据集、非密钥配置、选择顺序和源码快照完全一致时跳过已完成题，原两字段 evaluator JSONL 不会重复。`--max-new-cases` 可让自动化每次只执行一条未完成题而不改变同一 manifest。候选追踪只记录已有有界候选的 ID/分数/筛选原因，不改变召回或 Reader 输入；分类器只位于 `evals`，不参与生产请求。下一步运行新的固定快照 30 条偏好诊断，整轮结束后才根据通用合成回归决定是否实施一个生产修复。
- [x] LongMemEval 单次 PreReplyDecision 兼容与完整偏好诊断（2026-07-29）：修正离线分类只读取最终 planner flags，避免把 `skipped_by_planner` debug 字典误判为已召回；仅对高置信 `mixed`、self profile、主 LLM 回复且多个 recall 枚举非法的决策，恢复现有 `recall/profile/summary` 契约，明确 `none` 不覆盖。非 benchmark 回归覆盖饮食限制/午餐建议、工作偏好/会议建议、普通建议零召回、跨用户隔离和有界候选追踪。完整 Oracle 偏好切片 30/30 完成、0 运行失败，但本地 answer hit `1/30 -> 0/30`、recall hit `1/30 -> 1/30`、空上下文 `13 -> 18`、平均 `136.75s -> 137.55s`，且恢复逻辑触发 `0/30`；因此不能宣称改善。校正后为 13 条 routing miss、5 条空路由 coverage loss、12 条证据不足，下一步只审计同一次 `PreReplyDecision` 如何判断隐式个性化建议，禁止新增第二 planner、隐藏召回回退或 benchmark 特判。
- [x] 依据 `longmemeval_oracle_20260727_102934` 的失败样本修复通用复合记忆召回：低置信且只读的召回决策保留为无副作用降级，无关字段枚举错误改写入 warning；时间范围无命中时受限回退到同用户同 subject 文本检索；具体事实补充相关 Timeline 原话；Reader 上下文携带来源与时间标签；Runner 记录 fragment 失败详情且不让不完整历史进入 Reader。会话导入把 observation reflection 延后至全部 fragment 顺序导入成功后再触发，修复同一 SQLite 连接的后台并发写入。新增 60 项相关回归通过（另有 1 条既有身份 subject 隔离失败单独保留）。2026-07-27 重跑 27 个原始失败样本：27 完成、0 个导入错误、0 个空上下文、27 个带时间标签上下文、12 个 Reader 拒答；answer hit 33.33%，recall hit 29.63%。剩余拒答属于证据选择或时间推理质量，未使用 LongMemEval 题名、答案或 session id 特判。
- [x] LongMemEval 全量结果驱动的第二轮通用修复（2026-08-03）：统一报告互斥题型与 abstention overlay；为结构化事件/Timeline 召回增加 bounded candidate trace；隐式个性化建议仍由单次 `PreReplyDecision` 决定，并在依赖用户历史时同时打开有界 event text-search；assistant-history 只走 Timeline evidence；Reader/主模型提示要求在有直接相关证据时综合回答、证据不足才拒答。合成回归通过；单项 live 诊断确认 context 从 117 提升至 2026-2123 字，但 DeepSeek Reader 仍拒答，因此尚未宣称端到端准确率提升，正式结论仍需 V1 Cleaned full-history/common Reader/official judge。
- [x] LongMemEval 偏好题最终修复方案（2026-08-04，已实施）：按 `.planning/2026-08-04-longmemeval-preference-remediation-hando/` 将普通问题收敛为"确定性 preflight -> 单次 `PreReplyDecision` -> 非语义 ExecutionPlan"。修复显式 `text_search` 被词法 upcoming-plan 和无条件 `temporal.usable_range` 二次覆盖，移除 baseline discussion 等开放语义继承；保留敏感信息、音频 final、隐私、subject 隔离和确定性 fast path。补齐 Reader 选证据 trace、隐式个性化/realtime+memory 合成契约及 import/recall/Reader 分阶段计时。89 项测试通过（仅 1 条既有 identity subject 隔离失败单独保留）。Oracle 仅作诊断，正式结论仍需 common/official judge 与 V1 Cleaned full-history 配对验证。
- [x] LongMemEval 30 题拒答稳定性修复（2026-08-11，已实施）：`classify_pre_reply_decision` 在空响应/JSON 解析失败/异常时重试一次（合法 JSON 决策绝不重试，两次失败仍走 fail-closed 兜底，`warnings` 记录 `ppd_retry_used`）；timeline 召回在决策已授权 `needs_timeline_recall` 但 `timeline_query` 为空时，按"显式 query > specific_fact message > discussion_query"补齐搜索词，debug 新增 `query_source`。新增 8 项回归测试（classifier 重试 4 例 + timeline 映射 4 例），`pytest tests -q` 305 passed/2 skipped。
- [ ] LongMemEval 30 题整批复测（2026-08-12，HEAD 666e6ab，诊断+单题复验完成）：旧 6 道拒答全部清零（含 32260d93 两轮 judge 均判 1），拒答 6→4、空上下文 5→3、24/30 开了 profile 召回；但出现 4 道新拒答。单题复验结果：195a1a1b/1c0ddc50 单跑仍拒答（2/2 复现，分类器稳定误判 generic，真问题）；09d032c9 单跑正常但 judge 0（批量拒答是 API 连续空响应，偶发）；8a2466db 单跑正常且 judge 1（批量拒答是 Reader 证据选择翻车，偶发）。官方 judge 两轮 12/30 与 8/30（解析失败 4→7 条，评测侧波动）。结论：修复方向正确但官方命中率仍在噪声范围内；下一轮按"稳定复现"优先修 195a1a1b/1c0ddc50 的 generic 误判，再补 Reader"有上下文必须用证据"的规则。
- [x] 拒答与评测稳健性三项修复（2026-08-12，已实施）：①判断员 prompt 契约收紧（capsule 改为可信近期上下文证据；advice 规则新增 case e——第一人称推荐无需点名物品，有相关历史即开有界 self profile+event 召回）；②Reader 有上下文但未选证据时重试一次（扩展 refusal_retry 触发条件）；③judge 空响应/解析失败重试 3→5 次并追加"Return JSON only"指令（不改评分语义）。单测新增 11 项（含 judge 脚本 importlib 测试），`pytest tests -q` 315 passed/2 skipped。活体验收：195a1a1b 3/3 开召回且 judge 3/3、1c0ddc50 3/3 开召回（judge 2/3）、8a2466db 2/2 不拒答（judge 2/2）。评测回归（同批假设重判 3 轮）：10/17/14，parse error 1/1/0（修复前 4/7）。30 题整批需用新 prompt 重跑以测端到端。
- [x] LongMemEval 30 题偏好整批端到端复测（2026-08-12，HEAD 43a351f）：30/30 完成，拒答 6→4→1（仅 0a34ad58）、空上下文 1、29/30 开 profile 召回；官方 judge 3 轮 17/15/16（中位数 16），parse error 0/0/0；对比旧 prompt 整批中位数 14/30、解析失败 1-1-0。稳定判对：195a1a1b/1c0ddc50/32260d93/8a2466db/09d032c9/a89d7624 等；稳定判错 9 道（06878be2、1da05512、35a27287、54026fce、75f70248、afdc33df、b6025781、caf03d32、fca70973）全部有证据，属 C 类 Reader 合成，为下一轮专项样例。残留：0a34ad58 判 generic（分类时 capsule 为空，模型看不到已存东京行程记忆，需扩展 case e 至第一人称旅行建议）；answer_intent 仍常为 direct_answer（义务未传 Reader）。
- [x] 判断员可见内容 + 第一人称旅行建议 + Reader 义务三项优化（2026-08-12，已实施，双目标+零 hard code）：①D——判断员决定前用当前消息做本地 SQLite 文本搜索，把 5 条相关结构化记忆（含 profile/preference）并入判断员可见 capsule（对最近窗口去重），最近结构化段也纳入偏好类记忆，解决长对话里旧相关记忆不在最近窗口导致的误判；②E——classifier 规则新增 case f（第一人称 travel/commute/itinerary tips，合成 Barcelona 正例、无历史通用建议负例），通用化修复 0a34ad58 类问题；③F——personalized_recommendation 必须输出对应 answer_intent/answer_obligations，`answer_intent` 纳入路由契约并传到 AnswerDirective/主模型/Reader，Reader prompt 强化否定约束/增量下一步/比较覆盖。单测新增 7 项，`pytest tests -q` 322 passed/2 skipped；零 hard code 审计通过（生产/测试新增行无 benchmark 题号/答案/原文/字段）。白天未跑 30 题整批（按用户决定）；待提交后夜间跑 500 Oracle 全量 + judge 诊断，偏好 30 子集 judge 3 轮中位数对比基线 16/30，正式结论仍留 V1 Cleaned full-history 配对评测。
- [x] LongMemEval Oracle 评测提速基建（2026-08-13，已实施；全量 500 验收见下一条）：按 `.planning/2026-08-13-longmemeval-speedup-handoff/` 落地三个零精度损失手段，未做任何准确率修复、未碰 S Cleaned。①`chat()` 新增 `skip_reply_synthesis` 默认 False（默认值 + 单测断言 + 注释警告三层锁死），仅评测 runner 显式传 True：跳过 correction 检测、answer directive 与主模型回复（reply 置空），保留 PreReplyDecision/召回/时间解析/web/location/Timeline 写入；②runner 拆为 `build_question_memory_context()`（import+wait+recall，异常转 error dict）与 `answer_question_from_memory_context()`（Reader+打分）两阶段，新增 `--workers`（默认 1，Reader 保持主进程串行防限流，checkpoint/resume/max_new_cases 语义不变）；③两级缓存默认开启（`--cache-dir`/`--no-cache`）：L1 每题隔离 app home、L2 recall 结果，key=数据集 SHA-256+`LONGMEMEVAL_CACHE_VERSION`+history_mode，任一不匹配整层重建，损坏缓存按未命中重建，checkpoint schema 升 v2；④报告汇总只显示官方 judge，本地 substring 命中移出汇总、只保留每题 details 调试字段。单测新增 11 项，`pytest tests -q` 333 passed/2 skipped；`py_compile`/`git diff --check` 通过。冒烟：同一题冷跑 vs 缓存热跑，热跑 import/recall 相位为 0、recall_context 逐字节一致、Reader 输入逐字节一致（hypothesis 偶有改写属 temp=0 远端抖动）；3 题 `--workers 4` vs 串行活体：题号顺序/recall_context/计数完全一致（1 题 hypothesis 语义等价改写、Reader 输入相同），并行 1m52s vs 串行约 2m30s。
- [ ] Oracle 全量 500 `--workers 4` 验收（2026-08-13，待夜间运行，视耗时约 10-20 小时）：`PYTHONUNBUFFERED=1 conda run --no-capture-output -n hermes python -u -m ai_glasses_memory_assistant.evals.longmemeval_runner --dataset-path data/benchmarks/longmemeval/longmemeval_oracle.json --limit 0 --workers 4`。成功标准 500/500、0 导入失败、报告生成；记录总耗时与 mean phase_seconds 对比基线（2026-07-27 全量 28.8s/题，当时无逐分句语义分类），不做准确率结论；完成后 `scripts/judge_longmemeval.py --report-dir <report_dir> data/benchmarks/longmemeval/longmemeval_oracle.json` 仅做串行等价性核验。可中断后用 `--resume` 断点续跑。
- [x] 最新 Oracle 500 题官方协议判分（2026-08-14，报告 `pref500-20260812-e31d5c`）：先逐行核对 LongMemEval 上游 `src/evaluation/evaluate_qa.py`，将本地统一 JSON rubric 校准为上游按题型 prompt、user-only 消息、temperature 0 与 yes/no 标签规则，并移除固定未知回答强制记 0 的非官方 shortcut。按用户指定用 `deepseek-v4-flash` judge；因该模型先输出 reasoning，上游 GPT-4o 的 `max_tokens=10` 会截断在推理阶段，本次显式用 512 并在报告记录模型替代与 token 差异。500/500 判定完成、0 parse error，官方协议分数 386/500（77.2%）；普通题 361/470（76.81%），不可回答题 25/30（83.33%）。报告仅把官方 judge 总分作为成绩，本地 substring 不进入成绩。
- [ ] LongMemEval multi-session 最终机制方案（2026-08-17，机制代码与本地验证完成，官方门槛待测）：已实现“单一 `PreReplyDecision` answer contract -> 非语义 complete-set 执行合同 -> 范围穷尽的 structured+Timeline EvidenceSet -> 去重/状态/Decimal 校验账本”。数字/金额/日期切分、稳定游标分页、evidence-linked 原话及相邻上下文、RRF 非破坏排序、来源穷尽审计、最多九批证据加最终措辞、失败安全拒答均已进入主链路；DeepSeek 账本调用仅对固定 JSON 核算关闭 thinking 并启用官方 JSON Output，普通聊天和 PPD 不变。禁止 how many/total/check back 等短语路由、分类失败全量放开、benchmark ID/答案/label 特判。完整本地测试与首题真实 DeepSeek 账本通过后，仍需从 `v3` 干净缓存完成 59 失败集、74 回归集、133 multi-session 及官方 judge；未达到至少翻正 46 题且 74 题零回归前不标记完成。详细方案见 `reports/longmemeval/pref500-20260812-e31d5c/multi-session-final-mechanism-solution.md`。

用户效果：用户问“上个月和 Alex 讨论发布时最后决定了什么”，系统能找到对应决定、说明后来是否被修改，并展示相关日期与原话，而不是仅做关键词搜索。

### R5：从记录升级为个人回顾与认知辅助

- [ ] 在全天讨论 V1 上增加可配置的日/周回顾，按事件、决定、待办、风险、未决问题和重要人物组织，不把普通闲聊包装成成果。
- [ ] 自动发现重复承诺、长期未完成任务、相互冲突安排和需要再次确认的信息，但只生成建议，不静默改变任务状态。
- [ ] 用户可以编辑摘要、纠正人物、固定重要记忆、调整保留期限、批量导出和按原文/摘要/长期记忆/动作分别删除。
- [ ] 每条洞察都能回到来源；来源已过期时明确标注“依据已删除”，不能伪装成仍可核验。

用户效果：每天回顾不是一段泛泛总结，而是列出“今天做了什么、决定了什么、答应了谁、还有什么没完成”，每一项都能查看依据。

### R6：经授权的办公执行与主动服务

- [ ] 先定义 provider-neutral action contract，再接日历、任务、文档/PPT、邮件或其他办公适配器；核心记忆系统不直接依赖某一家云服务。
- [ ] 默认流程为“识别动作候选 -> 生成预览 -> 用户确认 -> 幂等执行 -> 查询结果 -> audit/撤销”；只有用户明确配置的低风险规则可自动执行。
- [ ] 任务、日历和文档动作必须保存外部对象 ID、权限范围、执行状态和错误，不允许模型只回复“已经创建”却没有真实副作用证据。
- [ ] 在动作执行稳定后再增加主动提醒：需 opt-in、安静时段、频率上限、延迟/重复抑制、送达状态和关闭入口。
- [ ] 建立未授权动作率、重复执行率、字段准确率、失败恢复率和撤销成功率门禁；未授权外部写入必须零容忍。

用户效果：听到“周五前把方案发给张经理”后，系统先展示待办和时间供确认；确认后创建真实任务，并能回答“任务建在哪里、是否成功、怎么撤销”。

### R7：眼镜运行时与生产可靠性

- [ ] 在不改变 `audio_event.v1` 业务语义的前提下定义硬件/手机音频 adapter，处理设备鉴权、麦克风状态、离线缓冲、网络恢复和版本兼容；不引入视觉接口。
- [ ] 将进程内 job 演进为可恢复的持久 worker，补齐事务并发、数据库 close/migration、崩溃恢复、幂等重放和失败队列。
- [ ] 补齐本地数据加密、密钥生命周期、备份/恢复、配额、物理删除验证、设备丢失处置和多设备同步冲突策略，再决定是否需要新增依赖。
- [ ] 建立音频健康、处理积压、模型耗时、记忆写入、召回和动作执行的脱敏观测指标；日志不得包含密钥、原始 PCM 或 embedding。
- [ ] 通过 10 小时真实待机、30 天数据生命周期、重启/升级/断网/磁盘不足/模型不可用和多设备冲突验收后，才从“Web 原型”升级产品阶段表述。

用户效果：手机或服务重启后，眼镜能恢复待机和未完成任务，不丢记忆、不重复执行，也不会因模型不可用而静默记录错误内容。

### 统一验收与宣称边界

- 每个阶段按独立提交推进：`检查 -> 最小实现 -> 聚焦测试 -> 全量回归 -> 审查 -> 文档/PLANS 更新`，不把多个阶段混成一次大重构。
- 建立固定纯音频验收包，包含音频文件、期望 transcript、speaker、时间、记忆候选、召回答案、隐私负例和动作预期；同一数据用于回归与竞品黑盒对照。
- 能接触竞品真机时，用同一批场景对比，不以发布稿作为准确率证据；无法验证的能力明确标注“公开宣称、未实测”。
- 只有音频、记忆、召回、隐私、动作和长时间运行门禁全部通过，才可使用“达到同类产品完整机制”；只有关键指标稳定领先，才可使用“超越”。

### 推荐实施顺序

1. 先完成 R0 全天讨论归档与现有失败清零。
2. 再完成 R1 真实声学验收和 R2 多人归属，确保输入可信。
3. 之后实施 R3 记忆策略和 R4 长期召回，确保记忆可信。
4. 在此基础上做 R5 回顾洞察，形成音频记忆产品核心体验。
5. 最后开放 R6 外部执行和主动服务，并以 R7 生产可靠性支撑真实眼镜接入。

### 其他并行维护项

1. 继续做工程可读性治理：只在三文档中维护当前态入口、代码地图和系统架构。
2. 公开 benchmark 评测：500 题 Oracle 已完成诊断，下一步先按“planner 路由漏检、写入类型、检索覆盖/排序、Timeline、Reader 证据利用”建立离线失败分类，并逐项用非 benchmark 文本的回归测试验证；再运行无标签泄漏的 V1 Cleaned S 全历史配对评测和官方 judge。详细任务计划位于 `.planning/2026-07-28-longmemeval-memory-foundation-audit/`。
3. 独立化后续修复：优先处理 correction fallback 重复保存、用户偏好 kind 归一化、`sqlite3 readonly database` 后台 job 生命周期问题。
4. 文字主线继续观察真实 audit 缺口；出现新问题时补最小核心测试或 target。
5. 音频方向继续按 R1 补真人 KWS、0.512 秒 partial、VAD final、声纹、耳机回声、长语音和 HTTPS 场景验收；真实模型慢测单独运行，不放进默认 CI。
6. 文件清理方向：用 `scripts/scan_cleanup_candidates.py` 先做只读候选分级，再按结果小步清理；不引入一次性补丁目录，不做 `src/` 大迁移。
7. 多人独立长期记忆已形成首个完整功能提交边界；后续按下方未闭环清单继续硬化，不恢复本地中文业务词表。`agent_bridge.py` 暂不做行数型清理，音频模块先保持稳定。

## 多人独立记忆待后续闭环

以下问题已在本次提交前复现或审查确认，因当前 token 额度止损保留到后续独立修复提交；当前提交不得宣称全量验收通过：

- identity fast path、周报和提醒默认只读取用户本人记忆。
- `recall_subject_scope=all` 优先于消息中提到的单个人名。
- 多个同名 provisional speaker 必须返回歧义，不得静默选择第一个。
- [x] 多人 transcript evidence 已按说话人 fragment/chunk 隔离，避免关联包含其他人物内容的整段原文。
- `EventMemoryStore` 共享 SQLite 连接增加完整事务同步，并让各 store 的 `close()` 真正关闭连接。
- import 只有 `subject_name` 时归为 named；迁移只更新不一致记录且不重复 rebuild FTS。
- 低置信度声纹 fail closed，不参与自动人物合并；未接入 runtime 的声纹 centroid 接口需删除或补齐明确策略。
- capture 部分 chunk 缺 speaker label 时不能整批退回 flat import；第三方敏感领域继续改为结构化 policy，不能扩充中文例词表。
- timeline/API/debug/audit 继续验证所有 embedding-like 字段均不会公开。
- 补齐 deferred correction、并发写入、迁移/关闭、部分标签和真实 API 路径回归测试，再运行 9 条 active 多人 strict eval 与桌面/390x844 浏览器验收。

## 当前阶段暂不做

- 摄像头、图片、视频、OCR、人脸识别、视觉场景理解或任何视觉记忆能力；这是长期产品边界，不是等待排期的缺口。
- 正式硬件眼镜 runtime、生产级音频上传/转写服务，以及 Android demo 的公开商店分发；当前 Android 只按内部测试 APK 和本地模型包继续验收。
- 主动提醒推送；只在 R6 动作候选、授权、幂等和撤销闭环通过后实施。
- 可靠 worker 队列、跨进程调度、多设备同步和完整审计后台；这些属于 R7 产品化阶段，不混入当前功能开发。
- 新向量库、Graph、外部索引或另起一套记忆系统。
- 为了看起来更像发布型插件仓库而迁移到 `src/` 布局。
- 在 `scripts/` 下沉淀一次性补丁、历史 bugfix 目录或临时实验脚本。
- 恢复旧专题文档、调研文档、HTML 汇报或历史流水账。

## 推荐阅读顺序

1. `AGENTS.md`
2. `README.md`
3. `CONTRIBUTING.md`
4. `docs/context/README.md`
5. `docs/context/code-map.md`
6. `ai_glasses_memory_assistant/README.md`
7. `docs/context/system-flow-current.md`
8. `docs/memory-system-audit.md`（记忆系统设计审计：已完成 vs 待优化，iPhone App 开发完成后启动）
9. 本文件

## 验收口径

一刀完成至少满足：

- 只改当前任务相关文件。
- 没有绕过记忆门控、用户隔离或敏感信息边界。
- 相关测试、静态检查或无法验证原因已说明。
- 如改变开发者入口或系统边界，同步更新三份 context 文档。

## 验证基线

文档改动：

```bash
cd /path/to/ai_glasses_memory_assistant
git diff --check
conda run -n hermes python -m pytest tests/test_core_startup.py -q
```

Python 改动：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

需要产品链路验证：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```

### Batch-3B-R Contract-Pre V2 Resume Failed; V3 Prepared

- The resumed row repeated the same unresolved, self-dependent temporal retrieval placeholder. The v2 artifact remains a failed freeze with 98 latest complete rows; preserve it for attribution.
- The repair is generic PPD guidance: derived temporal ranges must be expressed as a direct relationship, never as an unresolved formula requiring another requirement's answer. No benchmark matching, Reader, retrieval, API, database, average, or time-calculation code changed.
- Run v3 to a new immutable output path without `--resume`; mixing PPD prompt versions would invalidate the contract audit.

### Batch-3B-R Contract-Pre V3 Approved

- V3 passed: 99/99 complete, zero run/PPD/contract-validation errors, 148 requirements, and at most three requirements per contract. Preserve its immutable JSONL/summary artifacts and hashes in the task planning record.
- The generic derived-time requirement rule and one-minimum-anchor recommendation rule passed manual semantic review. No PPD semantic change is authorized during the next checkpoint.
- Proceed only with generic source hydration, requirement-specific retrieval, and verified source/quote binding under the frozen contract; do not add question matching, a second planner, average, deterministic time calculation, or closed-world zero inference.

### Batch-3B-R Evidence-Chain Checkpoint Implemented

- Requirement queries supplement only PPD-authorized structured-memory and Timeline sources, capped at five candidates per requirement per source. Linked Timeline evidence is explicitly hydrated; debug separates the requirement additions from ordinary recall.
- The eval Reader now requires exactly one support record per frozen requirement. A supported/conflicted record must cite a currently visible source ID and a literal source substring; malformed records retry once and then fail as execution errors, while valid unsupported records abstain under the existing policy.
- The product prompt receives the same immutable requirements and direct-source constraint without changing the public response shape. No database/API/schema, second planner, keyword fallback, average, deterministic time calculation, or closed-world zero inference was added.
- Local verification: focused suite `107 passed`; full suite `395 passed, 2 skipped`; conda compile and `git diff --check` passed. The remaining gate is user-owned paired replay with local Qwen; do not commit or start 3C before it passes.

### Batch-3B-R Paired Replay Tool Ready

- The eval-only replay tool now supports a paired mode: all 99 cases run combined and frozen-contract reader-only, while the dynamically selected truly-sufficient group also runs fixed-evidence. It records support binding/debug, answers, official judge results, status and immutable context hashes.
- Fixed-evidence uses only direct snippets already preserved in the historical counterfactual artifact and labels them with eval-only source IDs; it does not change production recall or create a hidden source.
- Verification passed: focused tool tests `64 passed`; full suite `397 passed, 2 skipped`; conda compile and `git diff --check` passed. Run the user-owned local-Qwen post before any commit or 3C work.

### Batch-3B-R First Paired Post Attribution

- The first post is preserved as an invalid harness result: 99/99 completed and frozen recall hashes were unchanged, but historical text-only contexts omitted source IDs required by the new binding contract, creating widespread `execution_failed` results.
- The repair is eval-only. It derives a visible source directory from the already saved recalled-memory/Timeline payload and appends it only to the Reader input; original recall context bytes, hash, product code, PPD and retrieval behavior remain unchanged.
- Re-run to a new path without `--resume`; do not interpret the old post as a product regression or commit the batch until the corrected gate completes.

### Batch-3B-R Proof-First Harness Gate

- Do not request another 99-case Qwen run until the eval harness passes a zero-Qwen preflight. The first paired post is retained only as proof that opaque source IDs were absent from the historical Reader input.
- The eval Reader now cites stable short `S1...Sn` aliases; alias-to-real-source mappings remain internal/debug and are checked against literal source text. Frozen `recall_context` bytes/hashes are retained, while derived Reader-context and alias-map hashes are separately auditable.
- `replay_batch3br.py --mode preflight` validates all 99 inputs, legal and illegal support bindings, three replay modes, frozen hashes, and a dynamically selected 12-case pilot without PPD, Reader, or judge calls. Only a passing report may lead to the pilot.
- The pilot is 4 saved-direct-evidence recovery cases + 4 unsupported-answer cases + 4 baseline-correct controls, all selected from stored attribution/score artifacts. It runs combined and reader-only, plus fixed-evidence for the first four. A final 99-case replay remains blocked until this smaller gate passes.
- Fixed-evidence coverage must report its real availability. The historical artifact currently contains direct snippets for 10 of 13 targets; the three missing inputs are a harness observation, never a Reader abstention or score.

### Batch-3B-R Pilot Result: Stop Before 99 Cases

- The 12-case Qwen pilot completed 12/12, but failed its gate in both combined and reader-only: correct controls fell 4/4 -> 1/4, and the unsupported side added one `execution_failed`. It did have one improvement on each safety side and no byte-identical judge flip, but those facts do not override the regression gate.
- Do not run final 99 cases, commit, or stack another Reader patch. Preserve the pilot artifacts for attribution.
- The failure is not one generic Reader defect. The completed zero-Qwen audit found 44/98 requirement-bearing contexts lose full alias-to-source-text equivalence because multi-line evidence is parsed as a single labeled line; 25 of those were baseline-correct. Separately, 98/99 frozen contracts change existing fields relative to 3A, with coverage routing changing on 15 rows (12 baseline-correct), so current Reader-only results cannot isolate Reader behavior. Complete-set also returns before ordinary requirement binding and has 7 incomplete-coverage plus 6 empty-source-ID contracts.
- Therefore do not layer another patch or run 99 cases. The evidence-backed next decision is to preserve all artifacts/diagnostics and restore uncommitted 3B-R production/eval/test/current-doc changes to committed 3A `e39ef07`, then redesign source serialization, PPD proof semantics, and complete-set as independently gated efforts. Await explicit authorization before this rollback.

### Batch-3B-R Rollback Completed

- User authorized the rollback. The explicit 3B-R production/eval/test/current-doc allowlist has been restored to committed 3A baseline `e39ef07`; untracked `evidence_binding.py` and its two dedicated tests were removed.
- Reports, contract freezes, paired/pilot/audit artifacts, planning tools and records, Android changes, and unrelated scripts remain preserved. Do not interpret the failed 3B-R experiments as product behavior, run another 99-case replay, or commit this batch.
- Any future Reader96 work starts from 3A and must separately prove source serialization, proof-contract compatibility, and complete-set behavior with zero-Qwen gates before a pilot is requested.

### Reader96 Source-Envelope Foundation (Phase A) Completed

- Added a standalone, unused source-envelope utility. It preserves each source's full text, stable alias, source ID/type, and integrity hash in a structured JSON record; embedded newlines are escaped data rather than record boundaries.
- The existing 3A Reader, PPD, planner, complete-set, API, and replay paths neither import nor call it. This is a serialization foundation, not a Reader behavior or score claim.
- Generic local regressions passed (`30 passed` with focused Reader-contract coverage), along with `py_compile`, `git diff --check`, and an exact 3A allowlist comparison to `e39ef07`. The wider suite reached `390 passed, 2 skipped` with one unrelated existing audio-reaper lifecycle failure; no Qwen, judge, replay, or pilot was run.
- Stop after Phase A. Any PPD proof-semantics or complete-set work requires new, separate user authorization and zero-Qwen gates.

### LongMemEval 146 Official-Judge Zero-Qwen Attribution Completed

- The eval-only analyzer `ai_glasses_memory_assistant.evals.longmemeval_failure_attribution` selects only the latest report's 146 `official_judge.per_question == 0` rows. It does not reuse the older local-`passed` taxonomy, which covers a different 234-row set.
- The immutable output `reports/longmemeval/pref500-20260831-182840-17656-attribution-20260901/` pins source hashes, source snapshot, 500 detail/judge IDs, and all 146 case-artifact paths. It reports `qwen_calls=0`, `judge_calls=0`, and `network_calls=0`.
- Direct artifact evidence confirms only 12 `retrieval_empty` and 11 `route_not_requested` failures. The remaining 123 are explicitly `insufficient_artifact_evidence`; 61 Reader refusals and 11 expected-abstention errors remain orthogonal behavior overlays, not a Reader diagnosis.
- Do not modify PPD, writing, ranking, complete-set, or Reader behavior from this batch. A next repair still requires a separately authorized, general root-cause evidence pack and a zero-Qwen gate.

### LongMemEval Named-Subject Evidence Boundary Corrected

- A named person who is not registered as a stored subject must remain an empty structured-memory subject scope. The system must not reinterpret a question about that person as permission to search the user's own structured memories.
- For complete-set recall only, an unambiguous but unregistered named subject may trigger a same-user raw Timeline scan. This preserves source evidence for events mentioned in conversation history without changing structured-memory ownership; ambiguous names remain fail-closed.
- PreReplyDecision distinguishes answer source rather than matching fixed phrases: a requested earlier assistant reply uses raw Timeline evidence, while a requested user action or personal event uses self event recall. Assistant text never becomes a personal memory candidate.
- Zero-Qwen regression coverage includes unresolved names, ambiguous names, resolved named subjects, same-user Timeline evidence, cross-user isolation, and assistant-history recall. `115` focused core-chat/classifier tests, `py_compile`, and `git diff --check` passed. No Qwen replay or score claim has been made; a frozen local-Qwen small replay is the next gate.

### Text Questions No Longer Masquerade as Continuous Audio Capture

- Structural length and punctuation are not semantic evidence that a message is a continuous audio capture. Ordinary `chat` text now always reaches the single PreReplyDecision authority, regardless of length.
- The continuous-capture fast path is retained only when existing real-audio provenance (`audio_event_id`) is present. This keeps long microphone transcripts available for capture/archival while preserving typed long questions as questions.
- Regression coverage verifies long text reaches PreReplyDecision, long audio retains continuous capture, and the existing named-subject/Timeline boundaries remain intact. This batch has no Reader change and no score claim until a separate frozen local-Qwen replay completes.

### Text-Query Routing Replay Confirmed

- The frozen `42c8391` replay completed all 19/19 selected cases with cache disabled and one local-Qwen worker. The input manifest contains the 11 original `route_not_requested` failures, two named-subject regression targets, and six baseline-correct controls; completed artifacts match the frozen IDs exactly.
- Every route target now records `pre_reply_decision_applied=true`. The local-Qwen official-protocol substitute judge changed 8/11 route targets from wrong to correct; all 6/6 controls stayed correct. This confirms the general text-versus-audio provenance boundary, but is not a 500-case score claim.
- The three route cases still judged wrong are now post-routing issues: two refused despite reaching the Reader (`0e5e2d1a` had no recalled context; `c7cf7dfd` had recalled context but no selected evidence), and one temporal case answered with evidence but omitted the required order (`gpt4_18c2b244`). The remaining named-subject target also reached recall and answered but made an incorrect temporal ordering. These four artifacts do not yet establish a shared five-case Reader or retrieval mechanism.
- Next work must mine direct evidence across the remaining 146 official-judge failures for a general state/time/evidence-selection mechanism before changing Reader or retrieval behavior. Do not add a Reader patch from this four-case replay.

### Zero-Qwen Direct-Source Coverage Audit

- The evaluator-only `longmemeval_coverage_audit` reads a frozen run's judge map, detail records, redacted case SQLite exports, and recall candidate trace. It makes no product imports or model/network calls. Its answer-reference comparison is diagnostic only: it never influences runtime retrieval.
- On the immutable `pref500-20260831-182840-17656-coverage-audit-20260902-v2/` artifact, 13 wrong cases contain a non-query direct source whose reference terms are absent from the Reader context. Six independently show the Timeline source in `candidate_trace.dropped_candidate_ids` while the trace selected only five candidates, across knowledge-update, multi-session, single-session-user, and temporal-reasoning questions.
- This is direct evidence for one general ranking defect: English stopwords still participate in Timeline FTS and lexical scoring, unlike structured-memory search, so generic question wording dilutes distinctive content terms and drops answer-bearing Timeline candidates. It authorizes a narrow Timeline search-term parity repair, not a larger Top-K, hidden fallback, Reader prompt, or benchmark-specific change.

### Timeline English-Stopword Ranking Repair Confirmed

- `bf1b30a` makes Timeline FTS and fallback scoring use the same English stopword policy already used by structured-memory retrieval. It changes only query terms, preserving user isolation, the five-chunk bound, PPD ownership, source scope, and all product interfaces.
- Zero-Qwen re-evaluation of the six direct candidate-drop artifacts put supporting source evidence back into the five-item Timeline result for 5/6. The remaining case stayed outside the bound, so this repair has an explicit known limit rather than a claim of complete coverage.
- The frozen local-Qwen replay completed 12/12 with cache disabled and one worker: five direct-source targets changed from baseline judge-wrong to correct, the sixth stayed wrong, and all six baseline-correct controls remained correct. Reader context contained the reference terms for five targets; no execution failures occurred. This is a 5/6 target slice result, not a 500-case score.
- Continue with a new direct-evidence batch for the remaining source-not-candidate, missing-query, state/time, or type-loss mechanisms. Do not increase Top-K merely because one target remained wrong, and do not infer a full-run gain from this slice.
