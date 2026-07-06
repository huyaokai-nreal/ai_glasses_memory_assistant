# 代码地图

本文按任务说明应该先读哪些文件和函数，避免 Codex 只凭文件名猜逻辑。

## 主聊天链路

入口：

- `server.py` 的 `GlassesHandler.do_POST()` 中 `/api/chat`
- `app.py` 的 `/api/chat`
- `agent_bridge.py` 的 `GlassesChatService.chat()`

主流程：

```text
HTTP body
-> message/user_id/session_id/location/defer_memory_writes
-> GlassesChatService.chat()
-> planner/router
-> recall + reply + memory processing
-> response dict
```

改聊天行为时，优先改 service 层，不要只改前端或只改一个 HTTP 入口。

## 记忆写入

入口：

- 聊天候选和 fast path：`turn_planner.py`
- 写入门控：`intent_policy.py`
- 保存：`memory_store.py`
- 后台处理：`agent_bridge.py`
- 手动新增：`POST /api/memories`
- 批量导入：`POST /api/memory/import`
- 连续输入：`POST /api/capture/stop`

必须检查：

- 候选是否有 `kind`、`memory_type`、`confidence`、`reason`。
- 是否经过 `should_write_memory_candidate()`。
- 是否按 `user_id` 写入。
- 是否处理相似记忆合并或证据合并。
- debug/audit 是否记录 saved/rejected/requires_confirmation。

## 记忆召回

入口：

- `turn_planner.py` 决定 `needs_profile_memory` / `needs_event_memory`。
- `agent_bridge.py` 执行 profile/event 召回。
- `memory_store.py` 提供 list/search/range 查询。
- `timeline_store.py` 提供 raw chunk 全文搜索，服务原文回忆和 evidence。
- `documents` 相关问题由 `agent_bridge.py` 先查 `EventMemoryStore.search_documents()` / `list_documents()`。
- `memory_recall_arbitration.py` 在召回后、回复前决定 document、raw timeline、structured event、observation、profile 谁作为主证据。

注意：

- 普通问题不应默认读取个人记忆。
- 时间范围使用半开区间。
- 近期计划问题可以回退 untimed event。
- 召回结果只注入当前 turn，不改 system prompt。
- 文档细节问题必须注入 `<document-context>` 的原文或原文片段，不能只用 summary。
- 原文回忆问题才应查 timeline，普通问答不要默认注入 raw chunk。

## Timeline 原文回忆

优先读：

- `timeline_store.py`
- `agent_bridge.py` 的 timeline 写入、`_recall_timeline_chunks()`、`_message_with_recall()`
- `turn_planner.py` 的 `needs_timeline_recall` 和 `local_timeline_recall`
- `server.py` / `app.py` 的 `/api/timeline/search`
- `tests/test_timeline_store.py`

检查重点：

- raw turn 和 capture chunk 是否按 `user_id` 隔离。
- timeline 写入失败是否只降级到 debug，不阻断实时回复。
- 结构化记忆的 `evidence_ids` 是否关联当前 timeline chunk。
- `/api/timeline/search` 是否只用于调试/原文搜索，不被写成每轮默认上下文。
- `/api/timeline/chunks` 是否只用于 evidence 查看和批量删除决策，不支持直接编辑 raw chunk 原文。

## 时间和计划

优先读：

- `turn_planner.resolve_temporal_local()`
- `temporal_parser.py`
- `memory_store.py` 中按时间范围查询的函数
- `evals/scenarios.jsonl` 中 schedule、temporal、pressure 场景

本地解析覆盖高频表达：今天、昨天、明天、今晚、周几、最近、接下来。复杂表达可延迟给 LLM fallback，不要在同步路径硬猜。

## Web 和位置

优先读：

- `turn_planner.py` 的 web/location 判断。
- `web_search.py`
- `agent_bridge.LocationContext`
- `static/app.js` 的 geolocation 请求。

位置是单轮临时状态，不默认保存为长期记忆。缺少位置时必须说明无法获取当前位置，不能猜城市。

## 后台 job

优先读：

- `agent_bridge.py` 的 `_create_memory_job()`、`_update_memory_job()`、`read_memory_job()`
- `/api/memory/jobs`
- `static/app.js` 的 job 轮询

当前 job 先存在 `_memory_jobs` 进程内字典里，同时公开 payload 会写入 SQLite `memory_jobs`。`read_memory_job()` 可以在服务重启后恢复查询 pending/saved/rejected/skipped/failed 等公开状态。不要把它写成可靠 worker 队列，除非当前任务明确要求任务重放、超时补偿或跨进程调度。

## Import 和 Capture

优先读：

- `GlassesChatService.import_memory_events()`
- `start_capture()`、`append_capture_chunk()`、`stop_capture()`
- `server.py` / `app.py` 的对应 API

原则：

- 普通文本、JSON items 和 capture 输入进入统一候选、门控、去重、写库流程。
- Markdown 上传先归档完整 document，返回安心确认，再由后续聊天按需召回原文。
- “当前记忆”面板通过 `/api/memories` 同时读取记忆和文档 metadata，文档编辑/删除走 `/api/documents/{id}`。
- 不让某个输入端绕过敏感信息门控。
- `confirm=true` 才能保存需要确认的候选。

## 周报和提醒

优先读：

- `GlassesChatService.weekly_report()`
- `GlassesChatService.check_reminders()`
- `evals/runner.py` 中 `reminder_check` action

当前边界：

- 周报是启发式草稿。
- 提醒是手动检查接口。
- 主动提醒 runtime 尚未实现。

## 前端

优先读：

- `static/index.html`
- `static/app.js`
- `static/styles.css`

改前端后要检查：

- 移动端和桌面端文本不重叠。
- debug 面板仍显示原始字段或中文解释。
- memory job 状态 saved/rejected/failed/timeout 都能展示。
- 语音和定位失败有明确状态，不误导为后端 bug。

## 测试入口

| 目标 | 命令 |
| --- | --- |
| 核心 service/planner | `conda run -n hermes python -m unittest ai_glasses_memory_assistant.tests.test_agent_bridge_policy -q` |
| eval 指标 | `conda run -n hermes python -m unittest ai_glasses_memory_assistant.tests.test_evals_metrics -q` |
| 全部子项目测试 | `conda run -n hermes python -m unittest discover ai_glasses_memory_assistant/tests -q` |
| live eval | `conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict` |
