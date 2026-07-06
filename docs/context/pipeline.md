# Pipeline 工程说明

本文记录当前真实调用链，给修改代码和排查问题使用。不要把这里写成产品愿景；每条流程都应该能在代码里找到。

## 总览

长期记忆的统一方向：

```text
聊天 / 手动新增 / 文本导入 / JSON 导入 / Markdown 上传 / continuous_capture
-> 标准化输入
-> raw timeline/chunk 记录原文证据
-> MemoryWriteCandidate
-> should_write_memory_candidate()
-> 去重或合并证据
-> EventMemoryStore.add_memory()
-> debug/audit
```

任何新输入来源都应该进入这条管道，不要直接绕过门控写 SQLite。未来的眼镜语音、App 文档、App 音频转写文本和手动补充，也只能先标准化为文本、文档或候选记忆，再复用同一套门控、去重、写库和 audit。

## `/api/chat`

入口：

- `server.py`：`GlassesHandler.do_POST()` 的 `/api/chat`
- `app.py`：`chat()`
- service：`agent_bridge.py` 的 `GlassesChatService.chat()`

主流程：

```text
message/user_id/session_id/location/defer_memory_writes
-> validate message
-> TimelineStore.add_turn() 记录用户原话 chunk
-> plan_turn() 处理本地 fast path 和召回计划
-> classify_pre_reply_decision() 对非 fast path 做单次回复前决策
-> TurnPlan.apply_pre_reply_decision() 把 `PreReplyDecision` 应用到 recall/web/location/reply path
-> profile/event/timeline/location/web/document 准备
-> 本地 fast path 或 AIAgent.run_conversation()
-> TimelineStore.update_turn_reply() 补齐助手回复
-> 处理 memory_write_candidates
-> 同步保存或后台 job
-> response + chat_audit.jsonl
```

常见分支：

| 场景 | 预期行为 |
| --- | --- |
| 问候 | planner fast path 本地回复，不读写记忆，不创建 agent session。 |
| 身份查询 | planner fast path 读取 profile/assistant_preference 后本地回答。 |
| 统一回复前决策 debug | `turn_semantic_classifier.py` 产出 `debug.pre_reply_decision`；本轮的 reply、recall、web/location、记忆候选和 correction/explanation flags 来自同一份决策。 |
| 明确画像陈述 | `PreReplyDecision` 生成 profile 候选，过门控后保存。 |
| 明确事件记录 | `PreReplyDecision` 生成 event 候选，过门控、去重和 evidence 路径保存。 |
| 当前时间问题 | planner fast path 或 `PreReplyDecision.reply_mode` 判断后本地回答。 |
| 位置相关问题 | 前端提供定位则注入临时位置；缺失时不能猜地点。 |
| 原文回忆问题 | timeline 召回 raw chunk 后本地回答或注入 `<timeline-context>`。 |
| 普通事实问答 | 不默认读写个人记忆，进入主模型。 |

## 回复优先与后台写入

前端默认可传 `defer_memory_writes=true`。这会让非关键记忆写入进入后台 job：

```text
create job: pending
-> worker starts: running
-> process candidate / LLM extraction
-> saved / rejected / failed / skipped
-> audit record_type=background_memory_write
```

查询入口：

```text
GET /api/memory/jobs?user_id=<user>&job_id=<job>
```

边界：

- job 状态首先存在 `GlassesChatService._memory_jobs` 进程内字典，同时公开 payload 会写入 `timeline_store.memory_jobs`。
- `read_memory_job()` 可以在服务重启后从 SQLite 恢复 pending/saved/rejected/skipped/failed 等公开状态。
- 这仍只是 demo 级持久化，适合 UI 轮询和排查；不是可靠 worker 队列，不承诺后台任务重放、抢占、超时补偿或跨进程调度。

## 记忆写入

候选来源：

- `turn_planner.py` 从聊天中生成。
- `POST /api/memories` 手动新增。
- `POST /api/memory/import` 文本/JSON 导入。
- `POST /api/capture/stop` 汇总 capture 文本后导入。
- Markdown 上传先保存为 document；只有显式抽取长期价值时才生成候选。

保存流程：

```text
candidate
-> should_write_memory_candidate()
-> rejected / requires_confirmation / saved
-> find_similar_memory()
-> merge_memory_evidence() 或 add_memory()
-> event_to_dict()
```

不要把“生成候选”当作“已经保存”。保存前必须经过门控。聊天 turn 产生的结构化记忆应优先把本轮 timeline chunk 写入 `evidence_ids`，这样后续能追溯“这条记忆来自哪句原话”。

## 记忆召回

召回由 `turn_planner.py` 决定，不是每轮默认发生：

- `fast_path`
- `fast_path_kind`
- `needs_profile_memory`
- `needs_event_memory`
- `needs_timeline_recall`
- `temporal_scope`
- `event_recall_strategy`

执行位置在 `agent_bridge.py`，实际查询由 `memory_store.py` 提供：

- `list_memories(user_id, kind="profile")`
- `list_events_between(user_id, start, end)`
- `list_recent_untimed_events(user_id)`
- `search(user_id, query)`

召回结果只作为当前 turn 的背景上下文注入，不修改 system prompt，不混入 Hermes 自身 memory。

## Timeline 原文回忆

入口：

- 每轮 `/api/chat` 会先写入 `TimelineStore.add_turn()`。
- `POST /api/capture/append` 会把 capture 片段写成 timeline chunk。
- `GET /api/timeline/search?user_id=<user>&q=<query>` 只用于调试和原文搜索。

数据流：

```text
用户原话 / capture 片段
-> raw_turns / chunks
-> chunks_fts 或 LIKE fallback 搜索
-> turn_planner 识别原文回忆请求，或非 fast path 下由 PreReplyDecision 输出 recall_goal=raw_evidence
-> TimelineStore.search_chunks()
-> 本地原文回答或 <timeline-context>
```

边界：

- timeline 保存的是原始输入证据，不等于结构化长期记忆。
- timeline 按 `user_id` 隔离；`/api/timeline/search` 也必须带用户边界。
- 搜索只返回 active chunk；LIKE fallback 会过滤低信息词并按命中分数排序。
- `recall_goal=summary` 的历史回顾不会直接套用“没找到原话”文案，而是把召回上下文交给主 LLM 总结。
- `llm_first` 下 summary 回顾会先生成 `answer_directive`，要求主 LLM 区分直接证据、observation 线索和背景 profile；空 timeline query 不再直接召回最近 query 列表。
- 删除结构化记忆时，未被其他 active 记忆引用的 `evidence_ids` chunk 会同步软删除。
- 普通问题不应默认召回 timeline，避免把全文库当成每轮上下文。
- 当前保存文本原话和 capture chunk，不保存原始音频。

## Import

入口：

```text
POST /api/memory/import
```

支持：

- `text`：纯文本、会议纪要、摘要、粘贴内容。
- `items`：结构化 JSON 条目。
- `source="markdown_upload"`：Markdown 文件上传，对应未来 App 文档入口的后端雏形。

普通文本/JSON 导入流程：

```text
payload
-> _import_items_from_payload()
-> infer kind/memory_type/source_id/ingestion_id
-> should_write_memory_candidate()
-> saved / pending_confirmation / rejected
-> audit record_type=memory_import
```

Markdown 上传流程：

```text
payload text + context(filename)
-> save full document into documents
-> derive title/summary for acknowledgement only
-> return reply + document metadata
-> audit record_type=document_import
```

Markdown 不再按每一行拆成 event memory。用户之后问文档细节时，聊天链路会召回原始 Markdown 或命中的原文片段，而不是只读摘要。

文档管理入口：

- `GET /api/memories` 返回 active 记忆和文档 metadata，供当前记忆面板混合展示。
- `GET /api/documents/{id}` 返回单个文档和原文。
- `PATCH /api/documents/{id}` 更新文件名、标题、摘要或 Markdown 原文。
- `DELETE /api/documents/{id}` 默认软删除文档，删除后不再进入搜索和聊天召回；显式 `purge=true` 时物理删除文档记录。第一版 document purge 不联动 timeline，除非后续建立 document chunk evidence 关系。

`confirm=true` 只用于明确确认导入；敏感内容不能因为来自 import 就绕过规则。

未来 App 音频上传不应直接新增另一套记忆写入路径。音频先经过 ASR 转写，转写文本再按普通文本或 continuous_capture 进入 import/candidate/gate；原始音频是否保存属于隐私和产品授权问题，不是当前 demo 能力。

## Continuous Capture

入口：

- `POST /api/capture/start`
- `POST /api/capture/append`
- `POST /api/capture/stop`

流程：

```text
start -> capture_id
append -> chunks
stop -> 合并文本 -> import_memory_events()
```

当前 capture 状态会同时保存在进程内 `_captures` 和 SQLite `captures/chunks`；服务重启后，`stop_capture()` 可以从 SQLite 恢复 capture chunks 并继续 import。它仍不是硬件 ASR runtime，也不是 App 音频上传管道。append 的片段会进入 timeline chunk，stop 时再汇总文本进入 import 和记忆候选流程。它验证的是“长输入或音频转写文本如何进入统一 ingestion pipeline”。

## Weekly Report

入口：

```text
GET /api/weekly-report?user_id=<user>
```

当前实现：

```text
查询最近 7 天 event
-> 按项目名启发式分组
-> 根据 memory_type 分 completed/decisions/tasks/risks
-> 输出 draft 和 evidence_ids
```

边界：这是周报草稿，不是完整项目知识图谱，也不是最终自动回顾系统。

## Reminder Check

入口：

```text
GET /api/reminders/check?user_id=<user>
```

当前实现：

```text
读取未来 24 小时 event
-> 过滤 memory_type=task
-> 返回 reminders
-> audit record_type=reminder_check
```

边界：这是手动查询，不是主动提醒 runtime。主动提醒需要授权、触发窗口、静音/抑制策略和审计。

## Audit

审计文件位于当前 app data 目录下：

```text
AI_GLASSES_HOME/data/chat_audit.jsonl
```

它应该回答：

- 本轮读了什么记忆。
- 为什么没读记忆。
- 本轮是否写入 timeline，以及召回了哪些 raw chunk。
- 候选为什么保存、拒绝或等待确认。
- 后台 job 最终状态。
- 是否调用主模型、web、location。
- 各阶段耗时。

不要把 API key、token、密码等敏感凭据写入 audit。
