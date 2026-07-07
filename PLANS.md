# 当前阶段与开发路线图

更新时间：2026-06-24。本文只保留当前阶段、优先级、验收标准、下一步开发顺序和明确的暂不做范围。代码真相仍以 `agent_bridge.py`、`turn_planner.py`、`turn_semantic_classifier.py`、`memory_store.py`、`timeline_store.py`、`server.py`、`app.py`、`evals/` 和 `tests/` 为准。

## 当前阶段

当前项目处于 Stage 2 可运行原型阶段，已经验证了：

- Web/语音 demo 入口。
- `/api/chat` 主链路闭环。
- SQLite 结构化记忆和原文 timeline。
- reply-first + 后台 memory job。
- 基础 debug、audit、tests、evals。
- 文字输入下的记忆机制第一版闭环。

当前还不是：

- 生产级硬件眼镜 runtime。
- 原生手机 App。
- 可靠音频上传/ASR/diarization runtime。
- 主动提醒系统。
- 完整审计后台。
- 多设备同步和生产级权限体系。
- 独立于 `hermes-agent` 的可开源产品包；当前仍存在 `HERMES_HOME` 迁移期兼容、Hermes `AIAgent` fallback、Hermes runtime/env fallback 和 eval 迁移期兼容。

## 已完成能力摘要

以下能力已完成当前最小闭环，不再作为开放缺口重复书写：

- `llm_first` 主聊天链路和 PreReplyDecision 主权收口。
- 结构化记忆写入门控、低置信度 fallback 和敏感信息拦截。
- 自然偏好、事件、任务、decision、project_state、observation 的基础写入与召回。
- 记忆生命周期、evidence 生命周期、多来源召回仲裁、strength cap/decay。
- task `open/completed/cancelled` 状态和待办召回边界。
- Markdown 文档归档、文档细节召回、timeline evidence 查看与删除。
- continuous capture 长输入基础链路。
- `source_summary`、recent audit 回放和基础 explainability。
- active / target eval 的当前最小体系。
- 按钮模拟唤醒、wake session、真实音频片段临时处理入口、本地 ASR v1、基础声学情绪 metadata、3 段用户声纹校准与 `speaker_hint=user|other|unknown` 保守判定。
- 声纹录入不再作为网页文字聊天的阻塞前置条件；未录声纹时也能直接用输入框测试 demo，之后有条件再通过“声纹录入”按钮补做语音校准。
- 24 小时无感佩戴、多人数音频记忆与唤醒式问答已形成专项规划入口；当前仍只是规划，不代表真实后台常驻收音、生产级 diarization 或多人会话长期记忆模型已经完成。

详细机制不要继续堆在这里，统一回到对应 `docs/context/*.md`。

本轮补记（2026-06-23，历史状态；2026-06-24 已由下方新补记覆盖旧 extractor 方案）：

- 统一 LLM 语义判定层继续推进了一个最小切片：`turn_semantics.flags.correction` 现在会作为 correction 检测入口门控。LLM semantics 明确 `correction=false` 时，主链路跳过后置 correction LLM 检测；`correction=true`、`rule_fallback`、语义失败或缺字段时仍保留旧 `_detect_correction()` 兜底。这个改动只迁移“是否需要 correction 检测”的入口，不让 unified semantics 直接写候选、不负责找旧记忆、不负责 supersede。
- 已同步 `docs/context/unified-semantic-classifier-plan.md`，标明 correction gate 进入保守消费阶段；验证命令为 `conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m unittest ai_glasses_memory_assistant.tests.test_agent_bridge_policy -q`，当前通过。
- 同日继续推进 memory extraction 入口门控：`turn_semantics.memory_action` / `flags.transient` / `flags.do_not_remember` 现在会决定是否进入前台 `classify_turn_intent()`；当 LLM semantics 明确非写入且没有 candidate content 时，也会让 reply-first 后台 job 直接跳过候选抽取。这个改动仍不让 unified semantics 直接生成 `MemoryWriteCandidate`，只是减少明确非写入 turn 的重复 LLM 抽取；`write`、`candidate_content`、correction、`rule_fallback`、语义失败或缺字段仍走旧 extractor 兜底。
- 同日继续推进 candidate generation shadow 对齐：当 memory extraction 仍运行时，`debug.routing.unified_semantic_candidate_shadow` 会记录 `turn_semantics.candidate_content / memory_kind / memory_type` 与 `classify_turn_intent()` 实际候选的内容、kind、type 是否一致。这个切片只做观测，不用 unified semantics 创建、覆盖或保存 `MemoryWriteCandidate`，为后续是否迁移候选生成提供证据。
- 同日继续把 shadow 对齐固化为可回放机制证据：chat audit 保留 `debug.routing.unified_semantic_candidate_shadow`，reply-first 后台 memory job 会把同类摘要写入 `extraction_trace.unified_semantic_candidate_shadow`，并提供本地统计 helper 汇总 exact / similar / different / semantic-only / extractor-only / fallback。后续补漏洞先看统计和 audit，按 unified semantics、extractor、write gate、recall 等层级定位，不再优先补单点短语规则。
- 同日继续把统计 helper 从“只计数”升级为“迁移建议”：`summarize_unified_semantic_candidate_shadow()` 会根据 fallback、semantic-only、extractor-only、different、kind/type mismatch 和全量对齐情况输出 `recommended_next_action` 与 `migration_readiness`。这一步仍不改变主链路行为，只把“下一刀该修 semantic、extractor、typing，还是可考虑 typing hint”变成可观测判断。
- 同日新增 deterministic “shadow 小考场”：测试 helper 用固定样例模拟 `turn_semantics` 与 extractor 输出，覆盖偏好、任务、项目状态、纠错、临时内容、不要记、召回和普通聊天 8 类语义，并复用 shadow summary 产出考试结果和下一步建议。这个小考场不接 live LLM、不写长期记忆，只用于判断下一刀是否可以进入 typing hint，还是应先修 semantic / extractor / typing。
- 同日继续进入保守 semantic typing overlay 阶段：`intent_classifier.py` 仍负责摘出候选内容，unified semantics 只在 `backend=llm`、无 error、extractor 已有单条候选且内容与 `candidate_content` 对齐时，修正候选 `kind / memory_type`，并在 `debug.routing.unified_semantic_typing_hint` 记录 applied / skipped / fallback。大白话说，就是“老抽取器摘句子，新语义层只帮它贴更准的标签”；它不创建 semantic-only 候选、不改 content、不绕过 write gate，敏感内容仍会被原安全门拒绝。
- 同日继续把 semantic typing hint 从前台局部逻辑收口为共享候选后处理机制：前台聊天仍写 `debug.routing.unified_semantic_typing_hint`，reply-first 后台 memory job 也会在保存前走同一个 post-processing 入口，并把结果写到 `extraction_trace.unified_semantic_typing_hint`。这仍然只修正 `kind / memory_type`，不创建候选、不改 content、不绕过写入或安全门。
- 同日开始阶段 2 的准备切片：semantic typing hint debug 增加 `typing_source`，用来区分候选类型标签来自 unified semantics、extractor 已对齐、extractor fallback，还是 semantic 不可用；同时补充 typing hint 统计 helper，统计 applied / skipped / fallback 和来源分布。这个改动只增强 provenance 和 audit 统计，不删除 `intent_classifier.py`，也不让 unified semantics 生成候选内容。

本轮补记（2026-06-24）：

- unified semantics 已从“只 gate + 贴标签”升级为短 turn 记忆候选生成主路径：`turn_semantics.memory_action/write + memory_kind + memory_type + candidate_content` 会直接生成 `MemoryWriteCandidate`，再进入原有写入门控、去重、纠错和 evidence 路径。大白话：新语义层现在直接说“这轮要记什么、属于什么类型”，而不是等旧抽取器再摘一遍。
- 旧 `intent_classifier.py` 已移除；共享数据结构迁到 `memory_candidate.py`。前台聊天、reply-first 后台 job、长输入分段抽取都不再调用 `classify_turn_intent()`。长输入仍先走 segment semantic cleaner，随后对可抽取片段调用 unified semantics；规则候选仅保留为本地兜底。
- debug/audit 增加 `unified_semantic_candidate_authority`，记录候选是由 unified semantics 创建、覆盖已有本地候选，还是因 transient / do-not-remember / semantic error 等原因跳过或 fallback。旧 `unified_semantic_typing_hint` 字段暂时兼容保留，内容来自同一份 authority debug。
- 验证：`conda run -n hermes python -m py_compile agent_bridge.py turn_planner.py turn_semantic_classifier.py memory_candidate.py tests/test_agent_bridge_policy.py` 通过；`conda run -n hermes pytest tests/test_agent_bridge_policy.py -k "unified_semantic or llm_first_saves_profile_preference or llm_first_rejects_sensitive"` 通过。
- unified semantics 已继续升级为单次 pre_reply_decision：非 fast path 不再串行调用 `classify_turn_semantics()` + 独立 `route_turn()` LLM；现在只调用 `classify_pre_reply_decision()` 一次，同时决定 reply_mode、recall、web/location、correction/explanation flags 和记忆候选。大白话：以前回复前要问两个内部 LLM“这轮什么意思、该走哪条路”，现在只问一次，得到一张完整执行单。
- router 独立概念已删除：`turn_router.py` 被移除，`StructuredRouteDecision` / `route_from_pre_reply_decision()` / `debug.structured_router` 不再作为兼容壳存在；`TurnPlan.apply_pre_reply_decision()` 直接消费 `PreReplyDecision`。大白话：不再有“先得到一张完整执行单，再翻译成旧路由单”的中间层，系统只认这一张执行单。
- 用户可感知变化：普通非 fast path、记忆召回、天气/location、当前时间兜底、长期记忆候选生成，回复前少一次内部 LLM 路由调用；审计里也能看到“这轮要怎么答、要不要查记忆、要不要联网、要不要保存什么”来自同一个 decision。
- 验证：`conda run -n hermes python -m py_compile agent_bridge.py turn_semantic_classifier.py turn_planner.py tests/test_agent_bridge_policy.py` 通过；`conda run -n hermes pytest tests/test_agent_bridge_policy.py -k "pre_reply_decision or unified_semantic or pre_reply or weather or recall"` 通过；`conda run -n hermes pytest tests/test_agent_bridge_policy.py` 通过，结果为 394 passed、11 skipped。
- 架构继续收口：删除旧 `_apply_unified_semantic_recall_override()` 执行路径，recall 不再由 `turn_semantics.recall_type` 在后面二次覆盖 planner；profile/event/timeline/observation recall 统一由同一份 `PreReplyDecision.memory_recall_type / needs_* / recall_goal` 进入 `TurnPlan.apply_pre_reply_decision()`。大白话：不再“一张执行单发完后，后面又有个旧 recall 小补丁再改一次路线”，而是同一张执行单直接决定查什么。
- 命名债同步清理：主链路统一使用 `pre_reply_decision` 命名，session/debug step 改为 `created_pre_reply_decision_session` / `pre_reply_decision_applied`，旧兼容命名不再作为独立决策中心。
- 验证：`conda run -n hermes python -m py_compile agent_bridge.py turn_semantic_classifier.py turn_planner.py tests/test_agent_bridge_policy.py` 通过；`conda run -n hermes pytest tests/test_agent_bridge_policy.py` 通过，结果为 394 passed、11 skipped。
- 架构终态继续收口：`turn_planner.py` 非 fast path 现在只保留 baseline / deterministic temporal / fast path，不再基于短语判断 recall type、memory write intent、weather/web/location intent 或开放 reply mode；`agent_bridge.py` 删除旧后台 planner event fast path 写入入口和位置短语预判 helper，weather debug 只把 `PreReplyDecision` 已给出的 `needs_web_search / needs_location / location_text / web_reason` 映射成展示字段。大白话：planner 不再“猜用户这句话是什么意思”，只负责把唯一执行单落成可执行步骤。
- 测试桩同步改成直接模拟 `PreReplyDecision`：默认 pre-reply payload 不再借 `plan_turn()` 生成开放语义，只保留当前时间/世界时间这类 fast reply 的确定性模拟；旧 `event_record_ack/profile_statement` 相关断言已迁移到 pre-reply candidate / planner baseline 语义。
- 验证：`conda run -n hermes python -m py_compile agent_bridge.py turn_planner.py turn_semantic_classifier.py tests/test_agent_bridge_policy.py` 通过；focused policy 集合 `conda run -n hermes pytest tests/test_agent_bridge_policy.py -k "weather or pre_reply or planner or weekly_report or action_item or profile_statement or event_record_ack or chat_saves_event_statement_with_question_word_filler or uncertain_memory_command_shell_falls_back_to_llm_extraction or llm_first_new_event_statement_does_not_fall_into_empty_evidence_guard"` 通过，结果为 69 passed；全量 `conda run -n hermes pytest tests/test_agent_bridge_policy.py` 当前 374 passed、20 failed、11 skipped，剩余失败主要是旧后台 event-record job、默认 FakeAgent 语义 fixture 和同步/后台写入预期尚未完全迁移。
- 迁移收尾已完成：补齐本地确定性 identity statement 写入、同步写库失败降级为可解释 failed memory job、同步写入成功后的 job/audit 查询、observation reflect pending/primary write 状态表达、长输入 LLM 候选不足时的规则 fallback 补充，并同步测试桩默认 `PreReplyDecision` 语义。大白话：现在“我叫 Jack”不用多开一次内部 LLM 就能保存；“记一下…”同步保存后也能被 `/memory-job` 和“为什么没保存”解释查到；长输入里 LLM 只抽到一条时，会用规则兜底补足明显任务/风险/决定，但含 token/API key 的敏感长输入不会被规则兜底误保存。
- 验证：`conda run -n hermes pytest tests/test_agent_bridge_policy.py` 通过，结果为 394 passed、11 skipped。上一条记录中的 20 个迁移遗留失败已收尾。

## 当前真正未完成的优先级

### P0 独立化迁移准备（前十七刀完成，迁出后 eval baseline 已首次落盘）

目标：把当前 demo 从 `hermes-agent` 内嵌原型逐步迁移为独立产品、独立部署、独立开源项目。当前阶段按小步迁移原则逐个替换 home/env/LLM/session/web search 等 Hermes 依赖点，不改记忆主链路。

当前重点：

- 已新增 `docs/context/standalone-migration.md`，列出当前 Hermes 依赖点、风险等级、替换方向和效果不退化验收。
- 已完成独立化第一刀：新增 `app_home.py`，让 `AI_GLASSES_HOME` 成为本项目 home；`memory_store.py`、`timeline_store.py`、`agent_bridge.py` 的 demo 数据目录和 `evals/runner.py` 临时隔离目录已改走本项目 data dir。
- 为避免旧测试和旧本地数据立刻失效，迁移期保留 `HERMES_HOME` fallback：如果未设置 `AI_GLASSES_HOME` 但设置了 `HERMES_HOME`，数据目录仍使用 `HERMES_HOME/ai_glasses_memory_assistant`。
- 已完成独立化第二刀：新增 `env_loader.py`，让本项目先加载 `AI_GLASSES_HOME/.env`，再加载包目录 `ai_glasses_memory_assistant/.env`；加载时不覆盖当前进程已有环境变量。
- `agent_bridge.py::_new_session()` 和 `evals/runner.py` 已接入本项目 `load_app_dotenv()`，现有 `AI_GLASSES_LLM_PROVIDER`、`AI_GLASSES_LLM_MODEL`、`AI_GLASSES_LLM_BASE_URL`、`AI_GLASSES_LLM_API_KEY`、`AI_GLASSES_LLM_API_MODE` 等配置仍按原逻辑进入 Hermes `AIAgent`。
- 已完成独立化第三刀：新增 `llm_client.py`，定义本项目最小 `LLMClient` 协议、`HermesLLMClient` 适配器和 `create_hermes_llm_client()` 工厂；`agent_bridge.py::_new_session()` 不再直接构造 `AIAgent`，而是通过本项目 adapter 创建主模型客户端。
- 第三刀只是 LLMClient 对照层，运行时仍使用 Hermes `AIAgent`，不代表已经摆脱 Hermes LLM 依赖；`HermesLLMClient` 会透传 `run_conversation`、`chat` 和回调/属性，保持现有主链路、计时 debug、测试 patch 兼容。
- 已完成独立化第四刀：新增 `OpenAICompatibleLLMClient`，默认 backend 改为 `openai_compatible`；当前主验证目标是 DeepSeek `deepseek-v4-flash`。`AI_GLASSES_LLM_BACKEND=hermes` 时仍可手动回退第三刀的 Hermes adapter。
- 默认 OpenAI-compatible backend 不再调用 Hermes `AIAgent`、`runtime_provider` 或 Hermes `.env` loader；缺少 `AI_GLASSES_LLM_MODEL`、`AI_GLASSES_LLM_BASE_URL` 或 API key 时会清晰报错。provider 为 `deepseek` 时，API key 可来自 `AI_GLASSES_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`。
- 已完成独立化第五刀：新增 `session_store.py` 的 `AppSessionStore`，`agent_bridge.py` 不再直接 import `hermes_state.SessionDB`；默认 OpenAI-compatible 后端本来不依赖 session store，显式 `AI_GLASSES_LLM_BACKEND=hermes` fallback 也会收到本项目自己的最小 session store。
- 第五刀只替换/隔离 Hermes `SessionDB` 依赖，不改变多轮对话 history 拼接、记忆写入、召回、解释、web search 或 SQLite memory/timeline schema。
- 已完成独立化第六刀：新增/完善 `web_search.py` 的项目自有搜索边界，`agent_bridge.py` 默认不再 import 或优先调用 Hermes `tools.web_tools`；联网上下文现在优先使用已在 `hermes` conda 环境验证可用的 `ddgs`，DuckDuckGo HTML 只作为最后兜底，并保持 `debug.tools[].name/query/reason/backend/results_count/results` 等 audit 字段兼容。
- 第六刀只迁移 web search 默认依赖，不代表已经完全脱离 Hermes；Hermes `AIAgent` fallback、`runtime_provider`、Hermes env loader fallback 和 `HERMES_HOME` 迁移期兼容仍未移除。
- 已完成独立化第七刀：验证默认 OpenAI-compatible backend 下 `_new_session()` 不触发 Hermes `AIAgent`、`runtime_provider` 或 Hermes env loader；`AI_GLASSES_HOME` 能承载本项目 data、audit 和 sessions 路径；`AI_GLASSES_LLM_BACKEND=hermes` 仍作为显式迁移期 fallback 保留。
- 第七刀只是独立启动验证和 fallback 边界清点，不代表已经删除 Hermes fallback；默认路径和显式 `hermes` fallback 的职责已经通过测试固定。
- 已完成独立化第八刀：新增 `docs/context/standalone-startup.md`，写清 `AI_GLASSES_HOME`、`.env`、默认 `openai_compatible` backend、DeepSeek/OpenAI-compatible 配置、标准库 server 和 FastAPI 启动命令，以及 Hermes fallback 的 legacy/迁移期边界；新增文档一致性测试，避免 env 变量和启动入口说明漂移。
- 第八刀只是独立启动说明和文档一致性保护，不删除 Hermes fallback，也不代表当前源码已经完成独立打包；当前仍建议从外层 `hermes-agent` checkout 执行启动和测试命令。
- 已完成独立化第九刀：冻结默认启动路径和 Hermes fallback 的硬边界。`agent_bridge.py` 不再顶层 import `hermes_constants`，`hermes_constants` / `hermes_cli.runtime_provider` / Hermes env loader / `run_agent.AIAgent` 都只允许在显式 `AI_GLASSES_LLM_BACKEND=hermes` 分支触发；新增静态测试防止 Hermes fallback import 回到默认模块加载阶段。
- 第九刀没有删除 Hermes fallback，也没有迁出目录；它只是把默认 OpenAI-compatible 路径和 legacy Hermes 后门的边界钉得更死。新部署仍不推荐使用 `AI_GLASSES_LLM_BACKEND=hermes`。
- 已完成独立化第十刀：新增 `docs/context/extraction-check.md`，做临时目录抽离预演和剩余 Hermes 线分组；`evals/runner.py` 的旧 Hermes home/env fallback 已从顶层 import 改成延迟 best-effort helper，缺少 Hermes 主项目时也能导入 eval runner。
- 第十刀结论：默认 `server.py` 路径和 `evals/runner.py` 在 `python -S` 的临时复制目录中都能导入，关键文件可 `py_compile`；`app.py` 的 FastAPI 入口在当前 `hermes` 环境缺 `fastapi` 时不能导入，这是独立部署依赖问题，不是 Hermes 主项目依赖问题。
- 已完成独立化第十一刀：新增 `docs/context/standalone-deployment.md`，把独立仓库部署前的搬家清单和 dependency manifest 草案写清。默认入口建议 `python -m ai_glasses_memory_assistant.server`；FastAPI 入口作为可选入口，需要 `fastapi`、`pydantic`、`uvicorn`；默认 demo 依赖草案先收口到 `openai`，TTS / 本地 ASR / 测试依赖分别列为 `edge-tts`、`funasr`、`pytest`。
- 第十一刀没有新增正式 `pyproject.toml`，没有删除 Hermes fallback，也没有迁出目录；`AI_GLASSES_LLM_BACKEND=hermes` 仍是 legacy / 迁移期备用，不推荐新部署使用。
- 已完成独立化第十二刀：新增 `pyproject.standalone.toml` 和 `docs/context/standalone-packaging.md`，把独立仓库打包草案、optional dependencies、console scripts 和 package data 写清；README 也补了 `AI_GLASSES_HOME`、`.env`、默认 `openai_compatible`、FastAPI extra、Hermes legacy fallback 和迁出前后 eval 对比提醒。
- 第十二刀仍没有正式迁出目录，也没有把草案改名为正式 `pyproject.toml`；它只是让后续独立仓库 packaging dry-run 有一个可检查起点。
- 已完成独立化第十三刀：用临时目录复制 `ai_glasses_memory_assistant/`，把 `pyproject.standalone.toml` 临时复制成 `pyproject.toml`，再用 `pip install --no-index --no-deps --no-build-isolation --target ...` 做独立打包 dry-run，不联网、不发布、不污染当前 conda 环境。
- 第十三刀发现并修复了一个真实打包问题：原草案没有把 `ai_glasses_memory_assistant.evals` 声明为 Python 子包，导致临时安装后 `ai_glasses_memory_assistant.evals.runner` 无法 import；现在 `pyproject.standalone.toml` 已包含该子包，文档一致性测试也补了防漂移断言。
- 第十三刀 dry-run 结论：默认 server console script、FastAPI optional console script、`static/*.html/css/js`、`evals/scenarios.jsonl`、`docs/context/*.md`、默认 server import 和 eval runner import 都已在临时安装 target 中验证；FastAPI 入口仍保持 optional，不让默认 dry-run 因缺 `fastapi` 失败。
- 第十三刀没有正式迁出目录、没有删除 Hermes fallback、没有改变 LLM backend 行为、没有跑 live 网络测试，也没有固化迁出前 eval baseline。
- 已完成独立化第十四刀：Hermes legacy fallback 选择“彻底封存但不删除”。只设置 `AI_GLASSES_LLM_BACKEND=hermes` 不再启用旧后端，会直接报错；必须同时设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 才会进入 sealed legacy 分支。
- 第十四刀后默认 OpenAI-compatible 路径仍不 import、不初始化、不触发 Hermes `AIAgent`、`runtime_provider` 或 Hermes env loader；`HermesLLMClient`、`create_hermes_llm_client()` 和 `run_agent.AIAgent` legacy adapter 暂时保留为迁移期对照/最终删除前缓冲。
- 第十四刀不改记忆写入、召回、解释逻辑，不改 SQLite schema，不改 web_search，不固化迁出前 eval baseline。
- 已完成独立化第十五刀：新增 `docs/context/standalone-eval-baseline.md`，固化正式迁出前 baseline 的场景批次、runner 命令、报告字段和 `baseline-notes.md` 模板；明确当前只是方案固化，没有执行正式 baseline，也没有落盘 baseline 结果。
- 第十五刀把 baseline 分成核心 active 门禁、效果回放与主线压力、target 缺口快照三批；报告要求对比 `active_pass_rate`、`active_failed_turns`、`target_failed_turns`、failed scenario ids、`response.debug`、`new_memories`、`recalled_memories`、web/location/weather/timing/memory_jobs 字段和 LLM backend 配置摘要。
- 已完成独立化第十六刀：按第十五刀方案首次执行迁出前 baseline，并把三批报告落盘到 `ai_glasses_memory_assistant/reports/standalone-migration-baseline/`；索引 notes 为 `pre-extraction-20260706-baseline-notes.md`。
- 第十六刀使用默认 `openai_compatible` backend，临时命令环境补齐 `AI_GLASSES_LLM_PROVIDER=deepseek`、`AI_GLASSES_LLM_MODEL=deepseek-v4-flash`、`AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com`，API key 来自已有 `DEEPSEEK_API_KEY` 且未写入明文；未启用 Hermes legacy fallback。
- 第十六刀 baseline 摘要：核心 active 门禁 `active_pass_rate=0.6167`、`active_failed_turns=23`；主线压力 `active_pass_rate=0.6364`、`active_failed_turns=4`、`target_failed_turns=9`；target 快照 `active_failed_turns=0`、`target_failed_turns=3`。这些是迁出前老环境现状，不是全绿质量证明。
- 第十六刀执行观察：首次未补齐 `AI_GLASSES_LLM_MODEL` / `AI_GLASSES_LLM_BASE_URL` 时配置报错且未生成报告；补齐非密钥 DeepSeek 配置后三批完成。target 批结束时观察到后台线程 `sqlite3.OperationalError: attempt to write a readonly database`，报告已生成且命令返回 0，迁出后复跑时需同口径观察。
- 当前仓库已切到 `main`，并绑定 GitHub remote `origin=git@github.com:huyaokai-nreal/ai_glasses_memory_assistant.git`，`main` 已跟踪 `origin/main`。第十六刀 notes 中的 `dev_ykhu / bda74fc31` 仍保留为当时 baseline 执行的历史快照，不回写成当前分支。
- 已完成独立化第十七刀：在当前独立仓库 `main` 上首次执行迁出后 baseline，三批报告落盘到 `reports/standalone-migration-baseline/post-extraction-20260706-*`；索引 notes 为 `post-extraction-20260706-baseline-notes.md`。
- 第十七刀 post baseline 摘要：核心 active 门禁 `active_pass_rate=0.6`、`active_failed_turns=24`；主线压力 `active_pass_rate=0.6364`、`active_failed_turns=4`、`target_failed_turns=9`；target 快照 `active_failed_turns=0`、`target_failed_turns=5`。
- 第十七刀迁出前后初步对比：mainline / replay 批失败数量和失败 scenario 集合与迁出前一致；active gate 比迁出前新增 1 个 active failed turn，新增失败场景为 `memory_mechanism_correction_target_preference_supersede`；target 快照比迁出前新增 `public_dialogue_preference_write`，`target_failed_turns` 从 3 增至 5。
- 第十七刀执行观察：target 批仍复现 `sqlite3.OperationalError: attempt to write a readonly database` 后台异常，说明该问题不是迁出后才出现的新现象；后续应作为独立后台 job 生命周期 bugfix 小刀处理，不和迁出 baseline 对比混在一起修。
- 第十七刀新增差异已抽查：两条新增失败都集中在 unified semantics live 输出波动，而不是 remote/branch/report-dir/package 路径问题。`memory_mechanism_correction_target_preference_supersede` 迁出后已经生成正确 `profile/preference`，但同时因 `flags.correction=true` 触发 correction pipeline 额外保存了 `event/event`，且 correction target backend 变成 `none`；`public_dialogue_preference_write` 迁出后把用户咖啡口味偏好保存成 `assistant_preference/preference`，导致下一轮 profile recall 查不到。下一刀若修，应只收敛 correction fallback 重复保存或用户偏好 kind 归一化，不扩大到主链路重构。
- 独立化完全迁出前必须保留“迁出前基线”和“迁出后结果”对比：在外层 `hermes-agent` 当前路径先跑 eval 离线/target 测试并保存报告，迁出到独立工程后用同一批场景复跑，对比通过率、target_failed_turns、关键 debug/audit 字段和代表性回复差异；不能只以“能启动”作为迁出完成标准。
- 后续真正架构迁移应在高推理强度下按能力小步推进：下一刀优先抽查第十七刀新增差异 scenario 的 debug/audit，确认是 live LLM 波动、路径/配置差异还是真实行为退化；也可以单独做 `sqlite3 readonly database` 后台 job 生命周期修复。不要把删除 sealed Hermes fallback、记忆写入、召回、解释逻辑混在一起改。

最小验收：

- 后续 Codex 接手独立化任务时，先读 `docs/context/standalone-migration.md`。
- 每次只替换一个 Hermes 能力点，并用相关测试或 eval 证明记忆写入、召回、解释依据、隐私门控和会话连续性没有退化。
- 真正迁出工作开始前，先在迁出前 checkout 生成 eval 基线；迁出后必须复跑同一批离线 eval / target 场景并写出差异结论。
- 不把 Hermes 内部模块整块复制进本项目；只做本系统真正需要的最小实现。

验证：

- home/data 第一刀已补 `tests/test_app_home.py` 覆盖 `AI_GLASSES_HOME` 优先、旧 `HERMES_HOME` fallback 和默认独立目录。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_app_home.py -q` 通过，3 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_timeline_store.py -q` 通过，16 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_server_config.py -q` 通过，5 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py -k "new_session or chat_audit or memory_job or audit_path or session_db or read_memory_job" -q` 通过，11 passed、1 skipped。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_evals_metrics.py -q` 通过，30 passed。
- 已验证：`conda run -n hermes python -m py_compile ai_glasses_memory_assistant/app_home.py ai_glasses_memory_assistant/memory_store.py ai_glasses_memory_assistant/timeline_store.py ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/evals/runner.py ai_glasses_memory_assistant/tests/test_app_home.py` 通过。
- env loader 第二刀已补 `tests/test_env_loader.py` 覆盖 `AI_GLASSES_HOME/.env`、显式路径加载、当前环境变量优先和包目录候选路径。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_env_loader.py -q` 通过，4 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py -k "new_session" -q` 通过，4 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_evals_metrics.py -q` 通过，30 passed。
- 已验证：`conda run -n hermes python -m py_compile ai_glasses_memory_assistant/env_loader.py ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/evals/runner.py ai_glasses_memory_assistant/tests/test_env_loader.py ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py` 通过。
- LLMClient 第三刀已补 `tests/test_llm_client.py` 覆盖 `HermesLLMClient` 的 `run_conversation` / `chat` / 回调属性透传，以及 `create_hermes_llm_client()` 保持原 `AIAgent` 参数不变。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_llm_client.py -q` 通过，2 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py -k "new_session or llm_first or pre_reply" -q` 通过，49 passed、1 skipped、6 subtests passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_evals_metrics.py -q` 通过，30 passed。
- 已验证：`conda run -n hermes python -m py_compile ai_glasses_memory_assistant/llm_client.py ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/evals/runner.py ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py ai_glasses_memory_assistant/tests/test_llm_client.py` 通过。
- OpenAI-compatible 第四刀已补 `tests/test_llm_client.py` 覆盖返回结构、history 拼接、`persist_user_message` 和 `system_message` 覆盖；`tests/test_agent_bridge_policy.py` 覆盖默认 OpenAI-compatible backend、DeepSeek API key fallback、缺配置报错和显式 Hermes fallback。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_llm_client.py -q` 通过，4 passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py -k "new_session or llm_first or pre_reply" -q` 通过，52 passed、1 skipped、6 subtests passed。
- 已验证：`conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_evals_metrics.py -q` 通过，30 passed。
- 已验证：`conda run -n hermes python -m py_compile ai_glasses_memory_assistant/llm_client.py ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/env_loader.py ai_glasses_memory_assistant/evals/runner.py ai_glasses_memory_assistant/tests/test_llm_client.py ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py` 通过。
- Session store 第五刀已补 `tests/test_agent_bridge_policy.py` 覆盖显式 Hermes fallback 收到本项目 `AppSessionStore`，以及最小 session/message/token 记录能力。
- web search 第六刀已补 `tests/test_agent_bridge_policy.py` 覆盖 `agent_bridge.py` 调用本项目 `web_search.search_web()` 边界；测试使用 stub 搜索结果，不依赖 live 网络。
- web search 线上搜索后端已补 `ddgs` 优先级验证：从常用启动目录 `/Users/huyaokai/Desktop/workspace/hermes-agent` 导入的是外层包 `ai_glasses_memory_assistant/web_search.py`，底层 `search_web("今天最新新闻", limit=3)` 返回 `backend=ddgs` 且结果数为 3；服务层 FakeAgent 链路在 `needs_web_search=True` 时返回 `debug.tools[0].backend=ddgs`、`results_count=5`，并把 `Web/tool context` 注入主模型输入。大白话：不是只有底层脚本能搜，真实 chat service 走搜索工具时也会拿到 `ddgs` 结果。
- web search 回答约束已补工具状态注入：`agent_bridge.py` 会从 `debug.tools` 派生 `<tool-state>` 放进主模型上下文，区分 `not_triggered`、`completed_without_results`、`completed_with_results` 三种状态；未触发搜索时不允许声称搜索正在进行，搜索空结果时要求明确没有可用结果并禁止编实时事实，有结果时要求基于 `Web/tool context` 回答。大白话：不是按“请稍等”等短语做替换，而是告诉模型“搜索这一步已经完成或没有发生”，让新闻、天气等实时问题共用同一条工具状态规则。
- 已验证：`conda run -n hermes python -m py_compile ai_glasses_memory_assistant/web_search.py ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py` 通过。
- 独立启动验证第七刀已补 `tests/test_agent_bridge_policy.py` 覆盖默认 OpenAI-compatible backend 不触发 Hermes fallback 模块，以及 `AI_GLASSES_HOME` 下 data/audit/sessions 路径归属；测试使用 fake OpenAI 和临时 home，不依赖真实 API key。
- 独立启动说明第八刀已补 `tests/test_standalone_startup_docs.py`，校验文档中的 `AI_GLASSES_LLM_*`、backend 名称、DeepSeek key fallback 和 server/app 启动入口仍匹配代码常量。
- Hermes fallback 边界第九刀已补 `tests/test_agent_bridge_policy.py::test_agent_bridge_has_no_top_level_hermes_fallback_imports`，防止默认 import `agent_bridge.py` 时重新引入 `hermes_constants`、`hermes_cli` 或 `run_agent` 顶层依赖。
- 仓库抽离预检查第十刀已补 `tests/test_evals_metrics.py::test_eval_runner_has_no_top_level_hermes_fallback_imports`，防止 eval runner 重新顶层依赖 `hermes_constants` / `hermes_cli`；临时复制目录的 `python -S` 导入/编译结果记录在 `docs/context/extraction-check.md`。
- 独立部署文档第十一刀已扩展 `tests/test_standalone_startup_docs.py`，校验 `standalone-deployment.md` 至少包含默认 server 入口、FastAPI 可选入口、OpenAI-compatible backend、Hermes legacy fallback、依赖分组和迁出前后 eval 对比口子。
- 独立打包草案第十二刀继续扩展 `tests/test_standalone_startup_docs.py`，校验 README、`standalone-packaging.md` 和 `pyproject.standalone.toml` 至少包含默认/可选入口、依赖分组、package data、`AI_GLASSES_HOME`、Hermes legacy fallback 和 eval baseline 提醒。
- 独立打包 dry-run 第十三刀继续扩展 `tests/test_standalone_startup_docs.py`，校验 `pyproject.standalone.toml` 明确包含 `ai_glasses_memory_assistant.evals` 子包；临时安装 target 已验证 package metadata、package data、console scripts、默认 server import 和 eval runner import。
- Hermes legacy fallback 第十四刀已补 `tests/test_agent_bridge_policy.py::test_new_session_hermes_backend_requires_legacy_opt_in`，验证单独 `AI_GLASSES_LLM_BACKEND=hermes` 会被封存门挡住且不触发 Hermes runtime import；文档一致性测试同步校验 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK` 出现在启动/部署说明中。
- 迁出前 eval baseline 第十五刀已补 `tests/test_standalone_startup_docs.py::test_standalone_eval_baseline_doc_keeps_runner_contract_visible`，防止 baseline 文档里的 runner 命令、关键 category、报告字段和“未执行正式 baseline”声明漂移。

### P0 外部记忆系统借鉴准备（第一阶段完成，转观察）

目标：把 Graphiti/Zep、Mem0、A-MEM、Letta 等外部记忆系统的可借鉴点，收束成当前项目可验证的小实验，而不是直接替换存储或引入新平台。

当前重点：

- 已新增 `docs/context/external-memory-systems-review.md` 作为外部资源对照和第一阶段范围锚点。
- 第一阶段只考虑轻量 `provenance + supersession`：记忆来源、有效状态、改口覆盖、可解释召回。
- 暂不接入外部服务、图数据库、向量库或新的 agent 平台。
- 第一刀已补完整用户链路测试：统一语义层识别 correction，`correction target resolver` 选中旧饮品偏好，旧记忆进入 `superseded`，后续偏好召回只使用新记忆，解释追问锚到稳定画像来源。
- 第一刀不引入饮品词表，不靠本地枚举饮料名；“具体覆盖哪条旧记忆”由带候选旧记忆上下文的 target resolver 判断。
- 第二刀已补 `status=target` live eval：固定 unified semantics / correction writer，只让真实 `correction target resolver` 在候选画像记忆中选择旧记忆；当前验证显示它能把“用户最近不喝咖啡”覆盖到“用户喜欢冰美式”，并且后续召回只看到 replacement memory。
- 同步补了 eval runner 的承载能力：`correction_target_payloads` 与 `dedupe_payloads` 分离，场景可用 `correction_target_resolver: "live"` 验证真实 target resolver，而不是误用 fake dedupe payload 伪造选中结果。
- 第三刀已增强 explanation 的具体记忆依据：解释追问现在会从上一轮 audit 的 `recalled_memories` / `saved_memories` 里带出具体记忆内容和 `source_trace`，先做到“我依据这条记忆：用户最近不喝咖啡”，不反查原始 timeline 文本、不改 schema。
- 第一阶段 lightweight `provenance + supersession` 已形成最小闭环：覆盖、召回、解释 trace 均有最小测试或 target 托底；暂不继续扩 schema，后续转为观察真实缺口。
- 已补可选下一阶段的小切片：explanation 在解释 `profile` / `structured_memory` 依据时，会根据上一轮记忆的 `evidence_ids` 只读反查 active timeline chunk，并补一句“它来自你当时这句原话”；如果 evidence id 是 source surrogate、chunk 已删除或查不到，则只保留结构化记忆和 trace，不编造原话。
- 验证：`conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "explanation_reply"` 通过，结果 26 passed；`conda run -n hermes python -m py_compile agent_bridge.py timeline_store.py tests/test_agent_bridge_policy.py` 通过。

最小验收：

- 先读当前 `memory_store.py`、`timeline_store.py`、`agent_bridge.py` 和相关测试，确认现有 `source/evidence/status/confidence` 是否足够。
- 先补或确认显式写入来源、拒绝写入、改口覆盖、可解释召回的最小测试/target。
- 实现时仍必须经过 `PreReplyDecision`、`should_write_memory_candidate()`、dedupe、correction 和 safety gate，不为单个 query 加硬编码短语补丁。

验证：

- `conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "preference_correction_recall_and_explanation_use_replacement_memory"` 通过。
- `conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "correction_target_resolution or llm_dedupe_supersedes_conflicting_profile_preference or preference_correction_recall_and_explanation_use_replacement_memory"` 通过。
- `conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --scenario-id profile_preference_correction_target_resolver_live_target --background-wait 15 --report-dir ai_glasses_memory_assistant/reports/correction-target-live` 通过，`target_failed_turns=0`。
- `conda run -n hermes python -m pytest ai_glasses_memory_assistant/tests/test_agent_bridge_policy.py -k "preference_correction_recall_and_explanation_use_replacement_memory"` 通过，验证 explanation debug 带出 recalled memory trace，回复包含具体 replacement memory。

### P0 文字主线质量收敛（第一轮收敛完成，转观察）

目标：把“文字输入、ASR 后文字、Markdown 文档 follow-up”这条主线压到可稳定演示和继续观察的状态，减少误存、误召回、recent context 污染和解释串台。

当前重点：

- 当前已达到“第一轮收敛完成、进入观察补漏”的状态，不再把这条线当作高频大范围改动点。
- 保留 `recent_context_capsule` 与 source explanation 的回放批次，后续只在出现新 audit 失败时补最小 target。
- 失败解释链路先维持现状观察，优先等真实回放再决定是否继续扩更多细分失败阶段。
- 已补真实 audit 待办第一刀：`2026-06-30 17:21` 的“我今天要做什么呢”暴露出计划类召回边界问题。当前已在 `upcoming_plan` / `ambiguous_recent_upcoming_plan` 以及计划类 `observation_review` 的回答组织里，把“时间窗内确定事件”和“无时间但相关的兜底线索”分区：有 `start_at/end_at/occurred_at` 且落入召回窗口的事件作为确定安排；无明确时间的旧事件/原文（如“用户决定去吃肉丝面”）仍可召回，但必须作为“时间不确定的相关记录”呈现，并提示不能确定属于今天或当前时间段。该修复不靠单条“肉丝面”短语补丁，不改 schema。
- 验证：`conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "today_plan or upcoming_plan or untimed" -q` 通过，7 passed；`conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "test_today_plan_separates_untimed_related_memory or test_plan_recall_partition_separates_timed_untimed_and_out_of_window_events" -q` 通过，2 passed；`conda run -n hermes python -m py_compile agent_bridge.py tests/test_agent_bridge_policy.py` 通过。全量 `conda run -n hermes python -m unittest tests.test_agent_bridge_policy -q` 当前仍有 2 个长输入语义清洗相关失败，分别是 `test_chat_long_input_rule_candidates_keep_project_launch_plan_object` 和 `test_chat_long_low_value_chatter_keeps_timeline_without_memory_pollution`，不属于本次计划类召回路径。
- 已补真实 audit 待办第二刀：事件记忆查询会复用 `turn_planner.py::resolve_temporal_local()` 已解析出的明确本地时间窗，条件限定为 `backend=local`、`usable_range=true`、`granularity=hour/day` 且 `start_at/end_at` 有效；这类“今天/昨天/今天下午3点”不再额外调用 temporal LLM，debug 记录 `temporal_backend=local_reused` 和 `temporal_llm_skipped_reason=usable_local_temporal_scope`。复杂时间仍保持 LLM 兜底，例如“上周做什么了”和“昨天的现在”不会走 local_reused。顺手把主模型 memory context 的宽泛时间段展示与本地回复对齐，只有“下午/晚上”这类宽泛标签时不再暴露 `12:00-18:00` 内部范围。
- 验证：`conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "temporal or today_plan or upcoming_plan" -q` 通过，19 passed；`conda run -n hermes python -m py_compile agent_bridge.py tests/test_agent_bridge_policy.py` 通过。全量 `conda run -n hermes python -m unittest tests.test_agent_bridge_policy -q` 仍是 2 个长输入语义清洗既有失败：`test_chat_long_input_rule_candidates_keep_project_launch_plan_object` 和 `test_chat_long_low_value_chatter_keeps_timeline_without_memory_pollution`，不属于本次 temporal 复用路径。
- 已补真实 audit 待办第三刀：真实 audit 第 228 条 `今天做什么` 证明“无时间相关记录分区”仍会把无时间长转写/新闻材料带进计划回答。当前已把明确时间窗口内的计划/安排问题收窄为只展示有 `start_at/end_at/occurred_at` 且落入查询时间窗的记忆；无时间记忆不进入回复，只在 debug 中记录为被排除项：`excluded_untimed_plan_count`、`excluded_untimed_plan_policy=explicit_time_window_requires_timed_memory`、`excluded_untimed_plan_memory_ids`。如果只有无时间候选，回复会直接说“没有查到这个时间段内的确定安排”，不会把肉丝面或香港长转写交给主模型再组织。开放式计划/想法问题（如“我有什么长期计划”“我之前说过想做什么”）暂不套这条硬限制，后续再允许有时间和无时间分区展示。
- 验证：`conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "today_plan or upcoming_plan or untimed or temporal" -q` 通过，21 passed；`conda run -n hermes python -m py_compile agent_bridge.py tests/test_agent_bridge_policy.py` 通过。全量 `conda run -n hermes python -m unittest tests.test_agent_bridge_policy -q` 仍是 2 个长输入语义清洗既有失败：`test_chat_long_input_rule_candidates_keep_project_launch_plan_object` 和 `test_chat_long_low_value_chatter_keeps_timeline_without_memory_pollution`，不属于本次计划类召回收窄路径。
- 第三刀后续优化方向：补写入侧时间解析能力，尤其是“等会儿/一会儿/待会儿去拿快递”这类相对短期安排，尽量在保存时落成明确 `start_at/end_at`；同时在统一语义层继续区分“明确时间窗口安排查询”和“开放式长期计划/愿望查询”，避免为了修 `今天做什么` 牺牲长期计划回忆。

最小验收：

- 普通百科题、文档细节题不会因 recent context 抢答。
- source explanation 始终跟随最新业务来源，不回退旧来源。
- 新增 target 和相关最小单测通过。
- 明确时间窗口的计划类问题（如“我今天要做什么”）只把查询时间窗内的有时间事件作为确定安排；无 `start_at/end_at/occurred_at` 的召回结果不进入回复，只在 debug 中作为 excluded 解释保留。
- 明确本地 day/hour 时间窗会直接复用并跳过 temporal LLM；复杂时间表达仍走 temporal parser，不能为了省耗时牺牲时间范围准确性。
- 开放式长期计划问题仍允许后续设计无时间分区展示，避免为了修 `今天做什么` 牺牲“我之前说过想做什么”这类回忆。

本轮补记（2026-06-22）：

- 已复跑 `recent_context_capsule` 相关最小单测，确认普通问答、文档细节题、弱指代 specific-fact 场景仍不会把 recent context 注入主回答。
- 新增跨文档比较单测：`metadata compare` 场景下 recent context 也不会误注入主 LLM，避免“刚导入过材料”抢占两份文档高层对比的主证据。
- 已收紧 `summary/review` 场景的 recent-context 注入边界：弱指代（如“这个项目现在主要状态是什么”）不再自动注入 recent context；只有 `想一想/继续说/刚才那个/这个材料` 这类更强的继续上下文信号才允许注入。
- 已补 explanation replay 回归：当 recent context 确实参与过一条 summary 型业务回答时，即使中间插入两轮无来源闲聊，后续“为什么这么说”仍会回到那条真正的业务回答，并保留“recent context 这轮有参与主回答”的解释。
- 已把上述两类 recent-context 边界补进 `evals/scenarios.jsonl`：一条是 `weak_summary_recent_context_not_injected_target`，一条是 `audit_replay_recent_context_summary_after_chitchat_detour_target`。现在这条 P0 线不只靠 unittest，也有可复跑 target 回放入口。
- 已实跑这两个新 target：`conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id weak_summary_recent_context_not_injected_target --scenario-id audit_replay_recent_context_summary_after_chitchat_detour_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`。
- 已继续补更长 mixed replay：新增 `audit_replay_document_none_weak_summary_structured_target`，先实跑打出 1 个 target 缺口，定位到 `recent_context_capsule` 的 summary/document 判定顺序会把 `injection_reason` 错记成 `document_detail_without_recent_reference`，随后已修正并复跑通过。
- 已继续把 recent-context summary 分支接进更长 mixed replay：新增 `audit_replay_document_none_recent_context_summary_target`，验证 `document -> none -> recent-context summary -> explanation` 链路在 live runner 下通过，说明 explanation 在经历 none detour 后仍能回到最新那条 recent-context 参与过的业务回答。
- 已把 recent-context 相关 P0 target 作为一个小批次一起 live 回归：`ordinary_question_recent_context_not_injected_target`、`document_detail_recent_context_not_compete_target`、`weak_summary_recent_context_not_injected_target`、`audit_replay_recent_context_summary_after_chitchat_detour_target`、`audit_replay_document_none_weak_summary_structured_target`、`audit_replay_document_none_recent_context_summary_target`；当前整批 `target_failed_turns=0`。
- 已继续补 recent-context 的反向来源切换缺口：新增单测 `test_explanation_reply_does_not_fallback_to_recent_context_summary_when_latest_business_turn_has_none` 和 live target `audit_replay_recent_context_none_explanation_no_fallback_target`，专门验证 `recent-context summary -> none -> explanation` 时 explanation 会跟随最新那条 `none` 业务回答，明确返回“当前没有可用来源”，不会串回更早那条 recent-context summary。
- 已实跑这条新回放：`conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id audit_replay_recent_context_none_explanation_no_fallback_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`；同时对应最小单测也已通过。
- 已继续把 explanation 自身也纳入 recent-context 反向切换验证：新增单测 `test_explanation_reply_does_not_reuse_recent_context_after_prior_explanation_when_latest_business_turn_has_none` 和 live target `audit_replay_recent_context_explain_none_explanation_no_fallback_target`，专门验证 `recent-context summary -> explanation -> none -> explanation` 时，后一次 explanation 会跳过前一次 explanation 记录，仍然只跟随最新那条 `none` 业务回答，明确返回“当前没有可用来源”，不会被更早的 recent-context explanation 带偏。
- 已实跑这条更细 replay：`conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id audit_replay_recent_context_explain_none_explanation_no_fallback_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`；对应最小单测也已通过。
- 已继续验证 recent-context 与其他真实来源的正向切换：新增单测 `test_explanation_reply_switches_from_recent_context_summary_to_latest_document_turn` 和 live target `audit_replay_recent_context_explain_then_document_target`，专门验证 `recent-context summary -> explanation -> document -> explanation` 时，后一次 explanation 会跳过 earlier recent-context explanation，改为跟随最新文档业务回答，明确解释“上传文档原文”，不会继续沿用“观察总结 / summary_reference_signal”。
- 已实跑这条 recent-context -> document 切换 replay：`conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id audit_replay_recent_context_explain_then_document_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`；对应最小单测也已通过。
- 已继续验证 recent-context 切回 structured-memory 的正向来源切换：新增单测 `test_explanation_reply_switches_from_recent_context_summary_to_latest_structured_turn` 和 live target `audit_replay_recent_context_explain_then_structured_target`，专门验证 `recent-context summary -> explanation -> structured-memory summary -> explanation` 时，后一次 explanation 会跳过 earlier recent-context explanation，改为跟随最新结构化项目状态回答，明确解释“结构化任务 / 文档只作为背景补充”，不会继续沿用“观察总结 / summary_reference_signal”。
- 已实跑这条 recent-context -> structured-memory 切换 replay：`conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id audit_replay_recent_context_explain_then_structured_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`；对应最小单测也已通过。
- 已继续验证 recent-context 切回 raw-timeline 的正向来源切换：新增单测 `test_explanation_reply_switches_from_recent_context_summary_to_latest_raw_timeline_turn` 和 live target `audit_replay_recent_context_explain_then_timeline_target`，专门验证 `recent-context summary -> explanation -> raw-timeline recall -> explanation` 时，后一次 explanation 会跳过 earlier recent-context explanation，改为跟随最新 timeline 原话回答，明确解释“timeline 原话”，不会继续沿用“观察总结 / summary_reference_signal”。
- 已实跑这条 recent-context -> raw-timeline 切换 replay：`conda run -n hermes env PYTHONPATH=/Users/huyaokai/Desktop/workspace/hermes-agent python -m ai_glasses_memory_assistant.evals.runner --mode live --scenario-id audit_replay_recent_context_explain_then_timeline_target --repeat 1 --background-wait 15`，当前 `target_failed_turns=0`；对应最小单测也已通过。
- 已继续把 recent-context 起头的 explanation 链路扩成更长 mixed replay：新增单测 `test_explanation_reply_recent_context_mixed_chain_always_follows_latest_business_turn` 和 live target `audit_replay_recent_context_mixed_reanchor_target`，专门验证 `recent-context summary -> explanation -> none -> explanation -> raw-timeline -> explanation -> structured-memory -> explanation` 时，每一次 explanation 都会重新锚到最新业务回答，不会被更早 recent-context summary 或中间 explanation 自身带偏。
- 已把“为什么没记住”这条失败解释链路也补进 P0 live replay：新增单测 `test_explanation_reply_reports_transient_context_rejection_when_not_saved` 和 target `audit_replay_transient_context_not_saved_explanation_target`，专门验证用户明确说“先别写成偏好、等确认再说”时，这轮会被判成临时上下文，不写长期记忆；后续追问“为什么没记住”时，解释会明确说明“这次被判定为不应写入长期记忆”，不会误说成安全门控。
- 已继续扩失败解释的 live replay 覆盖面：新增 target `audit_replay_sensitive_memory_block_explanation_target`，专门验证用户直接口述 API key 这类敏感内容时，这轮不会进入长期记忆；后续追问“为什么没记住”时，解释会明确说明是安全门控拒绝，不会误说成临时上下文或普通未命中。
- 已继续补“用户误以为没保存”这类反直觉解释回放：新增单测 `test_explanation_reply_reports_memory_already_saved_when_user_asks_why_not_saved` 和 target `audit_replay_memory_already_saved_explanation_target`，专门验证用户先说“记一下…”、后台 job 实际已完成后，再追问“为什么没保存？”时，解释会明确说明“这轮其实已经保存了 1 条记忆”，不会误报成 pending 或失败。
- 已继续补真实故障型失败解释回放：给 live runner 增加 `memory_store_mode="failing_write"` 支撑后，新增单测 `test_explanation_reply_reports_background_write_failure_when_user_asks_why_not_saved` 和 target `audit_replay_memory_write_failed_explanation_target`，专门验证用户先说“记一下…”、后台写库失败后，再追问“为什么没保存？”时，解释会明确说明候选已经走到写入阶段但后续流程失败，并带出 `RuntimeError`，不会误报成 pending、已保存或临时上下文跳过。
- 已做一轮代表性收敛复核：复跑 `ordinary_question_recent_context_not_injected_target`、`audit_replay_recent_context_mixed_reanchor_target`、`audit_replay_transient_context_not_saved_explanation_target`、`audit_replay_sensitive_memory_block_explanation_target`、`audit_replay_memory_already_saved_explanation_target`、`audit_replay_memory_write_failed_explanation_target`，以及对应关键单测 `test_explanation_reply_recent_context_mixed_chain_always_follows_latest_business_turn`、`test_explanation_reply_reports_background_write_failure_when_user_asks_why_not_saved`，当前整批通过，`target_failed_turns=0`。
- 到目前为止，这条 P0 线已经明确进入“第一轮收敛完成，继续观察”的状态：recent-context 污染、source explanation 串台、以及 `为什么没记住/为什么没保存` 的核心失败解释都已有单测 + live replay 托底。后续策略改为观察真实新缺口，再补最小 target，而不是继续无上限扩 replay。

### P0 真实语境压力测试

目标：继续把真实聊天、ASR 后文字、`md` 文档追问沉淀成 target 场景，而不是靠单轮 demo 手敲判断好坏。

当前重点：

- 更长多轮 replay。
- 更复杂 recall + write 混合链路。
- 更多普通闲聊夹任务、模糊追问、否定和改口场景。

最小验收：

- 每次新修复都能落成 target 或单测。
- target 按语义族扩展，不为单个 query 继续堆 patch。

### P1 文本清洗 Phase C（第一轮完成，转观察）

目标：继续校准真实口语和转写文本的高价值 span 提取，减少噪声、临时上下文、口头禅和局部“不要记”误入长期记忆。

当前重点：

- `filler + task`
- 局部“不要记”
- 自我纠正
- 临时上下文
- 敏感误听
- 本轮已先复用现有 target 去重检查，没有新增重复样例；复跑暴露出一个真实链路缺口：segment cleaner 已正确产出 `candidate_span`，但后续 PreReplyDecision 未产候选时，长输入后台 job 会把可抽取 span 丢掉。
- 已补最小桥接：长输入里 `SegmentSemanticDecision.should_extract=true` 且有 `candidate_span/candidate_hint` 时，可作为 `semantic_span_fallback` 候选继续进入原有 `should_write_memory_candidate()`、dedupe、task 状态和 evidence 路径；并列 task span 会拆成多条任务，便于单独召回和完成/取消。
- 验证：`tests/test_text_cleaning.py` 通过；`tests/test_agent_bridge_policy.py -k "semantic_span_when_pre_reply_misses_candidate or semantic_span_splits_parallel_tasks or text_cleaning or long_input_rule_fallback or sensitive_asr_like or question_word_filler"` 通过；7 条现有文本清洗 target 小批次复跑通过，`target_failed_turns=0`。

最小验收：

- `SegmentSemanticClassifier`/相关 trace 能解释保留与拒绝原因。
- 新 target 能证明不是只靠低置信度侥幸挡住。

### P1 审计与解释长链稳定性（局部不要记解释切片完成，继续观察）

目标：让“为什么这么答/为什么没记住”在长链路、多来源切换和失败场景里仍然可解释。

当前重点：

- 更长 explanation replay。
- `none -> document -> timeline -> structured` 及逆向切换。
- memory job 各阶段解释稳定。
- 本轮补了局部 `do_not_remember_scope` 的解释小切片：memory job 会把 semantic cleaning 里的局部不要记范围摘要到 `memory_processing.local_do_not_remember_scopes`；用户追问“为什么没保存物流电话？”时，优先解释这是局部“不要记/不用记”范围，而不是误说成安全门控、写入失败或系统漏掉。
- 验证：`tests/test_agent_bridge_policy.py -k "local_do_not_remember_scope or explanation_reply_reports"` 通过；`py_compile agent_bridge.py tests/test_agent_bridge_policy.py` 通过；7 条文本清洗 target 小批次复跑通过，`target_failed_turns=0`。

最小验收：

- Debug 面板和对话内解释能稳定给出主依据、未采用来源和失败阶段。

### P1 持续收音/唤醒专项的下一刀

目标：如果继续沿眼镜音频方向推进，只做最小可信切片，不跳到完整 always-on runtime。当前产品方向已从“持续收音 + 唤醒式现场问答”扩展为“24 小时无感佩戴 + 多人会话记忆 + 唤醒式复盘”，专项规划见 `docs/context/ambient-audio-wakeword-plan.md`。

当前重点：

- 优先做“多人转写文本输入 -> 会话模型 -> eval/门控”的最小切片，先验证谁说了什么、用户与谁达成了什么、哪些旁人内容不能保存。
- 多人长期记忆必须以用户为主体，例如“用户和张三约定了某事”，而不是把旁人当作系统用户或保存旁人的私人画像。
- `AudioSegmentProcessor` 失败/超时/清理测试。
- 情绪模型高低置信与冲突时的 reply/use-debug 行为。
- `speaker_hint=user|other|unknown` 对长期记忆门控的实际保护。
- wake word 路径保持“唤醒前是背景，唤醒后才是问题”。
- 本轮补记（2026-07-07）：阶段 B 已形成开源方案调研文档 `docs/context/ambient-audio-v2-open-source-research.md`。结论是：先做多人转写文本输入和会话记忆模型，不先接完整 always-on runtime；近期默认不新增音频大依赖，继续复用现有 `/api/capture/*`、timeline chunk、SenseVoice/FunASR/cam++ 路线。VAD 优先调研 Silero VAD / FunASR 动态 VAD，ASR 主线继续 SenseVoiceSmall/FunASR，diarization 以 pyannote.audio 作 baseline、FunASR/cam++ 作本项目优先落地路径，openWakeWord 作为真实唤醒词候选，WeSpeaker/SpeechBrain 作为 speaker embedding 评测参考。下一步阶段 C 应把调研结论拆成严格执行步骤，先落文本级 `ConversationSession / ConversationTurn / Participant` 和 target eval。
- 本轮补记（2026-07-07）：阶段 A 已补齐“持续收音改进 V2”指导思想文档。`docs/context/ambient-audio-wakeword-plan.md` 现在明确四层边界：原始音频层随用随抛，ASR 多人原文时间线层不只保留用户本人且不等同于最近 6 条，唤醒问答上下文层只注入当前 query 需要的窄窗口，长期结构化记忆层只保存以用户为主体、有后续价值的多人会话事实。核心规则是“记录层可以长，注入层必须窄，长期记忆必须精”。本阶段没有实现功能代码；下一阶段应先做 VAD、ASR、diarization、speaker clustering、联系人命名、多人记忆建模和隐私删除机制调研。
- 本轮补记（2026-07-07）：已把 `docs/context/ambient-audio-wakeword-plan.md` 从“持续收音待机与唤醒式现场问答”扩展为“24 小时无感佩戴、多人数音频记忆与唤醒式问答”。新增目标模型包括授权后的常驻收音、多人数 `ConversationSession/ConversationTurn`、`speaker_id/speaker_role/speaker_label`、以用户为主体的多人长期记忆门控、以及先做多人转写文本输入和 eval、后接真实 diarization 的 MVP 顺序。大白话：下一步先证明“分清谁说了什么以后系统该怎么记”，再证明“真实音频里怎么分清谁说了什么”。
- 本轮补记（2026-07-07）：修正局域网 HTTP 页面上的本地 ASR 提示顺序。浏览器在 `http://<局域网 IP>` 下会先隐藏 `getUserMedia`，前端现在先判断非安全上下文，再判断浏览器能力，避免把“需要 HTTPS 或 localhost”误报成“浏览器不支持麦克风录音”。同轮继续修复标准库 HTTPS server 的 TLS 握手阻塞：不再把监听 socket 整体 `wrap_socket`，而是在线程 worker 内对每个连接单独握手，避免 Chrome 的慢/半开连接卡住后续 `127.0.0.1` 和局域网请求。验证：`env PYTHONPATH=/Users/huyaokai/Desktop/workspace conda run -n hermes python -m pytest tests/test_server_config.py -q` 通过，7 passed；`conda run -n hermes python -m py_compile server.py tests/test_server_config.py` 通过；Chrome 实测 `https://10.2.30.128:8765/` 约 291ms 打开，页面标题为 `AI Glasses Memory Assistant`。

最小验收：

- 真实音频入口只保留派生结果，不泄露临时路径或原始音频。
- 多人转写文本 target 能验证用户/已知联系人/未知说话人的归因、保存和拒绝保存边界。
- 清理测试覆盖 success / failure / timeout。
- 背景第三方说话不会误进长期记忆。

## 当前下一刀顺序

默认按下面顺序推进，除非新的代码证据或 audit 明确推翻：

1. 公开 benchmark 评测第一刀：LongMemEval 已新增本地数据目录、Git 忽略规则、adapter 和独立 runner；下一步把已下载的 `longmemeval_oracle.json` / `longmemeval_s_cleaned.json` 放入 `data/benchmarks/longmemeval/` 后，先用 oracle `--limit 20 --history-mode timeline` 跑 smoke report，再根据失败样本决定是否需要 `chat` 导入模式或官方 judge 对齐。
2. 独立化迁移下一刀：基于第十七刀差异抽查结果，做最小修复二选一：优先收敛 correction fallback 对已存在 `profile/preference` correction 候选的重复 `event` 保存，或收敛用户消费/口味偏好被标成 `assistant_preference` 后无法 profile recall 的 kind 归一化问题。
3. 单独处理 `sqlite3 readonly database` 后台 job 生命周期问题；迁出前后都复现，不应和 baseline 差异分析混在一起修。
4. 继续补文字主线 target 和 recent context 污染边界。
5. 外部记忆系统借鉴第一阶段已完成，暂时转观察；如后续真实缺口需要，再做 `evidence_ids` 反查 timeline 原话的最小实验，不直接接入外部系统。
6. 在统一真实语境压测前，先补文本清洗 Phase C 的高价值机制切片：局部“不要记”、filler 混任务、自我纠正范围、多说话人归因和敏感误听。
7. 继续补审计与解释长链稳定性，重点看 memory job 失败阶段、未采用来源和解释依据是否能从 debug/audit 讲清楚。
8. 如果转向音频专项，先做多人转写文本输入的会话模型和 target eval；通过后再做音频入口清理测试、speaker/emotion/wake 边界验证和真实 diarization 接入。
9. 机制收敛后再继续补真实语境压测 replay，优先来源切换和失败阶段。
10. 持续收音改进 V2 的阶段 A/B 已完成；阶段 C 根据 `ambient-audio-v2-open-source-research.md` 拆执行计划，先做文本级多人会话 schema 和 target eval，再决定何时接片段级音频、VAD、diarization 和 wake word。

每次只做一个最小切片：先补 target 或最小测试，再做窄修改，再写回结果。

## 暂不做

当前明确 out-of-scope：

- 真实硬件常驻收音 runtime。
- 原生手机 App。
- 生产级音频上传/转写服务。
- 主动提醒推送系统。
- 可靠 worker 队列和跨进程调度。
- 多设备同步。
- 完整审计后台。
- 新向量库、Graph、外部索引或另起一套记忆系统。
- 把学习/汇报 HTML 当成当前开发主线。

## 推荐阅读顺序

接手开发前默认按这个顺序看：

1. `AGENTS.md`
2. `docs/context/README.md`
3. 本文件
4. `docs/context/current-status-and-gaps.md`
5. 按任务进入 `pipeline.md`、`memory-mechanism.md`、`eval-coverage.md` 或专题文档

## 验收口径

只有满足以下条件，某一刀才算完成：

- 主链路真实行为已改变，不只是文档或 helper。
- 有最相关单测、target eval 或可复现对话证据。
- debug/audit 能解释关键决策。
- 相关文档已同步到正确位置。
- `PLANS.md` 只更新状态、下一步和验收结论，不回退成长篇流水账。

## 验证基线

文档改动最少执行：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent/ai_glasses_memory_assistant
git diff --check
```

Python 改动至少执行：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py
conda run -n hermes python -m unittest discover ai_glasses_memory_assistant/tests -q
```

需要产品链路验证时执行：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
