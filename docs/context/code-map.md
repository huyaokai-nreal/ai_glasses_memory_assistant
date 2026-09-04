# 代码地图

本文只说明当前代码入口和高风险修改区域，帮助开发者少走弯路。不要根据旧架构名猜逻辑；当前主链路以代码为准。

如果是看到某个 `.py` 文件不知道它负责什么，先看 `ai_glasses_memory_assistant/README.md`；本文更适合按“我要改什么功能”来找入口。

## 主聊天链路

入口：

- 标准库 HTTP：`ai_glasses_memory_assistant/server.py`
- 业务 service：`ai_glasses_memory_assistant/agent_bridge.py`

核心路径：

```text
POST /api/chat
-> GlassesChatService.chat()
-> plan_turn()
-> classify_pre_reply_decision()
-> TurnPlan.apply_pre_reply_decision()
-> 按需召回 profile/event/timeline/discussion/document/location/web
-> 本地回复或 OpenAI-compatible LLM
-> 同步保存或后台 memory job
-> TimelineStore / chat_audit.jsonl
-> response
```

当前主权决策是 `PreReplyDecision`。不要恢复旧 router，也不要把开放语义继续堆到 planner 短语规则里。

## 改功能先看哪里

| 任务 | 先读文件 | 注意点 |
| --- | --- | --- |
| 聊天行为 | `agent_bridge.py`、`turn_planner.py`、`turn_semantic_classifier.py`、`response_timing.py`、`llm_runtime.py`、`explanation_helpers.py` | 优先改 service 层；HTTP 入口只做薄包装。`response_timing.py` 只负责 assistant response timing/debug trace；`llm_runtime.py` 只负责 demo LLM 配置解析和 client 创建；`explanation_helpers.py` 只负责解释类问题识别和解释回复展示，不改聊天控制流。`chat(skip_reply_synthesis=False)` 是评测专用提速开关：只跳过 correction/answer directive/主模型回复，保留 PreReplyDecision/召回/时间解析/web/location/Timeline；默认必须为 False，生产入口永不传。 |
| 记忆写入 | `memory_candidate.py`、`intent_policy.py`、`memory_store.py` | 候选必须经过 `should_write_memory_candidate()`，敏感信息不能静默保存。 |
| 记忆召回 | `memory_store.py`、`timeline_store.py`、`memory_recall_arbitration.py`、`source_summary_helpers.py`、`explanation_helpers.py` | 召回只服务当前 turn，不改 system prompt，不默认读取全部历史；召回源仲裁和 debug reason 补全都在 `memory_recall_arbitration.py`，来源摘要和 audit 摘要 payload 在 `source_summary_helpers.py`；解释类回复的展示文案和 evidence trace 格式化在 `explanation_helpers.py`，上一轮 audit/job/timeline evidence 读取仍在 service。 |
| 文档导入/召回 | `agent_bridge.py`、`document_helpers.py`、`import_helpers.py`、`memory_store.py` | `agent_bridge.py` 保留 service 调度和文档召回编排；标题/摘要、文档查询识别和上下文拼装在 `document_helpers.py`；文本/JSON 导入拆分、导入条目分类和候选包装在 `import_helpers.py`；Markdown 文档保存完整原文，文档细节问题必须读原文或片段。 |
| 原话证据 | `timeline_store.py`、`agent_bridge.py`、`timeline_management_helpers.py` | timeline 是原始证据，不等于长期结构化记忆；管理接口的 chunk payload、删除结果说明和 redaction debug 在 `timeline_management_helpers.py`。 |
| 全天讨论归档 | `discussion_archive.py`、`timeline_store.py`、`agent_bridge.py`、`turn_planner.py`、`turn_semantic_classifier.py`、`server.py`、`static/app.js` | final ambient transcript 增量形成切片、当天话题和每日概览；最近 6 段只服务“刚才”。摘要不等于长期记忆，异常恢复不能触发正常 stop 才允许的记忆任务。 |
| 时间与计划 | `turn_planner.py`、`temporal_parser.py`、`memory_store.py` | 简单 day/hour 时间优先本地解析；复杂表达走 LLM fallback。 |
| Web/位置 | `web_search.py`、`agent_bridge.py`、`static/app.js` | 位置是当前 turn 临时状态；实时问题必须基于工具状态，不要编结果。 |
| 后台 job | `agent_bridge.py`、`memory_job_helpers.py`、`timeline_store.py`、`static/app.js` | `agent_bridge.py` 管 job 生命周期、锁和持久化；`memory_job_helpers.py` 管公开 payload、processing payload 和阶段说明；当前是 demo 级 job 状态持久化，不是可靠 worker 队列。 |
| Import/Capture | `agent_bridge.py`、`import_helpers.py`、`capture_helpers.py`、`conversation_helpers.py`、`conversation_candidate_helpers.py`、`server.py` | 新输入来源应复用统一候选、门控、去重、写库流程；capture 生命周期和最终 memory gate 仍在 service。speaker-labeled transcript 先由 `conversation_helpers.py` 解析角色和 alias，再由 `conversation_candidate_helpers.py` 生成结构化隐私 extraction plan；它不使用本地中文业务词表生成语义候选。 |
| 流式音频/speaker | `audio_engine/{settings,contracts,backends,offline,runtime}.py`、`audio_processing.py`、`turn_planner.py`、`intent_policy.py`、`agent_bridge.py`、`server.py`、`static/audio-worklet.js`、`static/app.js` | session 内 VAD/KWS/ASR cache 和所有模型所有权都在 `audio_engine`；环境 ASR 可由显式 profile 选择 legacy Python 或 Sherpa ONNX（Silero/Ten VAD + SenseVoice），不会更改默认桌面/Android 行为。`audio_processing.py` 只重导出旧名称。partial 不进入聊天或记忆，final 才由 `plan_audio_event()` 分发。 |
| Android 本地 demo | `android/app/src/main/`、`android/tools/install_local_model_pack.py`、`android/tools/DecryptDiagnosticBundle.java`、`android/README.md`、`android_runtime.py`、`static/app.js` | Kotlin 只负责权限、前台录音、模型、TTS、加密导出和 WebView 适配；`SherpaVadAdapter` 按 model-pack manifest 支持 `silero_vad` / `ten_vad`。设置页离线回放可输出不含音频的 Eval_Ali 最终事件 JSON，但不会创建 audio event、Timeline、记忆或 audit。记忆、队列、声纹聚合、诊断快照脱敏和 audit 仍由共享 Python 核心负责。模型、密钥和私有数据不进 Git。 |
| 通用背景音频记忆闭环评测 V2 | `ai_glasses_memory_assistant/evals/eval_ali.py`、`scripts/run_eval_ali_offline.py`、`scripts/p0_oracle_interval_ablation.py`、`scripts/prepare_eval_ali_android_replay.py`、`scripts/assemble_eval_ali_android_candidate_pack.py`、`data/benchmarks/eval_ali/ambient_memory_v2_gold.json`、`scripts/lock_eval_ali_offline_baseline.py` | runner 将第 1 通道 WAV 以 PCM16 小帧直接推入共享流式音频会话。`--audio-profile` 固化模型与实际 VAD 参数指纹；health 分别汇总删除/替换/插入、VAD、重叠负担与加权 DER。oracle interval 工具用金标边界区分重叠暴露，但不执行声纹分离。`--android-events-jsonl` 只评分 Android 原生最终事件健康，绝不标为闭环或麦克风验收。 |
| iPhone 本地 demo | `ios/AIGlassesMicProbe/AIGlassesMemoryAssistant/`、`ios/tools/verify_model_pack.py`、`mobile_runtime.py`、`static/app.js` | Swift 负责 WebKit 异步桥接、前台 HFP 路由、TTS、定位、Keychain 和模型包校验；网页继续复用 `static/`。CPython/numpy loopback 与 sherpa 五组件推理仍未完成时，必须拒绝待机和声纹录入，不能把打包网页资源描述为共享 Python 服务。 |
| 周报/提醒 | `agent_bridge.py`、`report_helpers.py`、`evals/runner.py` | 周报是启发式草稿；提醒是手动检查接口，不是主动 runtime。`agent_bridge.py` 保留查询、user/time 窗口和 audit；`report_helpers.py` 负责周报草稿、attention items 展示文案、项目归类和背景 observation 判断。 |
| LongMemEval 评测 | `evals/longmemeval_runner.py`、`evals/longmemeval_adapter.py`、`scripts/judge_longmemeval.py` | Oracle 仅作诊断，不做正式排名。runner 拆成 context 构建（import+recall，可 `--workers` 并行）与 Reader 答题两阶段；每题两级缓存默认开启，recall 阶段传 `skip_reply_synthesis=True`。complete-set 与产品共用 `evidence_set.py` 校验：中间账本不信任模型总数/最终措辞，来源分类与事实 provenance 分离，debug 区分证据不足和执行失败。成绩只看官方 judge。 |
| 前端 | `static/index.html`、`static/app.js`、`static/styles.css` | 检查移动端文本、debug 展示、job 轮询、语音/定位失败状态。 |
| 本地运行配置 | `app_home.py`、`env_loader.py`、`llm_runtime.py`、`llm_client.py` | 默认 home 是 `AI_GLASSES_HOME`；`llm_runtime.py` 管 demo LLM 环境变量解析、DeepSeek fallback 和 client 创建；主 LLM runtime 只走 DeepSeek/OpenAI-compatible API。 |
| 本地工具脚本 | `scripts/` | 只放可重复诊断、迁移、benchmark 准备、清理扫描工具；不要放一次性补丁流水账。当前清理入口是只读的 `scripts/scan_cleanup_candidates.py`，用于输出可清缓存、需复核候选和禁止自动清理边界。 |

## 高风险区域

- `agent_bridge.py` 很大，包含聊天、capture 生命周期、音频 service API、speaker enrollment、周报、提醒、audit、解释上下文读取、job 生命周期，以及仍与多人 transcript 的最终 privacy/memory gate 强绑定的导入调度。主题 helper 入口：文档看 `document_helpers.py`，导入看 `import_helpers.py`，memory job payload/stage 看 `memory_job_helpers.py`，timeline 管理 payload/说明看 `timeline_management_helpers.py`，来源摘要和 audit 摘要 payload 看 `source_summary_helpers.py`，解释问题识别和解释回复展示看 `explanation_helpers.py`，assistant response timing/debug trace 看 `response_timing.py`，demo LLM 配置解析和 client 创建看 `llm_runtime.py`，周报展示、attention items、项目归类和背景 observation 判断看 `report_helpers.py`，召回仲裁与 debug reason 补全看 `memory_recall_arbitration.py`，capture 纯 helper 看 `capture_helpers.py`，speaker-labeled transcript 结构解析看 `conversation_helpers.py`，多人 transcript 结构化隐私 extraction plan 看 `conversation_candidate_helpers.py`，音频 runner 和片段处理实现看 `audio_processing.py`。
- `server.py` 是唯一 HTTP 包装入口。新增 API 时保持薄包装，把业务逻辑放在 service 层。
- `memory_store.py` 和 `timeline_store.py` 管 SQLite schema、搜索、删除和 evidence。不要随意改字段或删除逻辑。
- 当前正式 Python 包就是 `ai_glasses_memory_assistant/`，暂不迁到 `src/` 布局；只有出现明确发布/安装/多包隔离需求时，才单独规划大迁移。
- `tests/` 只保留核心保险丝。新增功能由负责同事补专项测试，不把历史功能回归重新堆回默认门禁。
- `evals/scenarios.jsonl` 是 live eval 场景。不要把临时验证样例直接写成 strict 门禁。

## 常用测试

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

默认单元测试包括 `tests/test_audio_engine.py` 的 fake backend 契约，以及原有 startup/storage/chat 保险丝。

清理前先跑只读候选扫描；报告会把薄转发 helper 分成零引用、仅内部引用、测试引用三组，删除前仍需人工复核调用链：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python scripts/scan_cleanup_candidates.py
```

live eval：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```

LongMemEval Oracle 诊断跑（默认缓存开启，`--workers` 只并行记忆构建、Reader 保持串行）：

```bash
cd /path/to/ai_glasses_memory_assistant
PYTHONUNBUFFERED=1 conda run --no-capture-output -n hermes python -u \
  -m ai_glasses_memory_assistant.evals.longmemeval_runner \
  --dataset-path data/benchmarks/longmemeval/longmemeval_oracle.json \
  --limit 0 --workers 4
# 需要绝对干净的全量重跑：加 --no-cache
```
