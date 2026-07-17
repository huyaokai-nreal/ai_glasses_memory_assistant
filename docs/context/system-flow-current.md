# 当前系统架构

本文只记录当前系统怎么工作、数据怎么流转、边界在哪里。代码是真相；如果本文和代码冲突，先读代码，再修文档。

## 一句话

这是 AI 眼镜个人记忆助手的本地 Web/语音原型：用户输入先被回复，再按需沉淀为长期记忆；后续问题可以按需召回画像、事件、任务、文档和原话证据，并通过 debug/audit 解释为什么读、写或拒绝保存。

## 系统总览图

![AI 眼镜个人记忆助手全系统 Pipeline](assets/system-overview-pipeline.png)

总图只保留新人必须先理解的职责边界：`server.py` 是薄 HTTP 层，`GlassesChatService` 是统一编排层，`PreReplyDecision` 决定非 fast path 的回复与召回方向，长期记忆候选必须经过 `should_write_memory_candidate()`，而 `sessions.db`、`timeline.db`、`events.db` 和 `chat_audit.jsonl` 分别承担不同状态。音频录制、唤醒和兼容入口的细节由 `assets/frontend-audio-data-flow.mmd/.svg` 展开。

## 系统边界

当前已经具备：

- Web 文字聊天、按体验者 ID 隔离的浏览器流式语音、TTS、定位、debug 面板。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- 启发式周报草稿和手动提醒候选检查。
- 统一音频 session、VAD、KWS-only 两段式唤醒、partial/final ASR、ambient capture、声纹参考和匿名 voice group。

当前不是：

- 生产级硬件眼镜 runtime。
- 原生手机 App。
- always-on audio runtime。
- 生产级 diarization 或联系人归因系统。
- 主动提醒推送系统。
- 可靠跨进程 worker 队列。
- 多租户生产服务。

## 主流程

```text
前端或 API 请求
-> server.py
-> GlassesChatService
-> 本地 baseline 和 fast path
-> PreReplyDecision
-> TurnPlan.apply_pre_reply_decision()
-> 按需读取记忆、原话、文档、位置或 web
-> 本地回复或 OpenAI-compatible LLM
-> 生成记忆候选
-> 写入门控
-> 去重、合并或 supersede
-> SQLite / audit / response
```

`PreReplyDecision` 同时决定回复模式、召回类型、web/location 需求、记忆候选字段和 correction/explanation flags。旧 router 和旧 `intent_classifier.py` 不再是当前架构的一部分。

## Service 和 Helper 分工

`agent_bridge.py` 是业务 service 调度层：负责把聊天、导入、capture、文档、后台 job、audit、timeline 和 memory gate 串起来。纯 helper 叶子逻辑按主题放在独立 helper 模块。

当前 helper 边界：

- `response_timing.py`：只负责 assistant response timing 和 debug trace payload。
- `llm_runtime.py`：只负责 demo LLM 环境变量解析、DeepSeek fallback 和 OpenAI-compatible client 创建。
- `document_helpers.py`：负责文档标题/摘要、文档查询识别、标题匹配评分和文档上下文拼装。
- `import_helpers.py`：负责文本/JSON 导入条目拆分、分类和候选包装。
- `memory_job_helpers.py`：负责后台 memory job 公开 payload、processing payload 和阶段说明。
- `timeline_management_helpers.py`：负责 timeline 管理接口的 chunk payload、删除结果说明和 redaction debug。
- `source_summary_helpers.py`：负责 source summary 和 audit summary 的来源计数、主来源说明和删除来源 debug payload。
- `explanation_helpers.py`：负责解释类问题识别、解释回复展示文案、局部不保存范围说明和 evidence trace 格式化；上一轮 audit/job/timeline evidence 读取仍在 service。
- `report_helpers.py`：负责周报草稿、attention items 展示文案、项目归类和背景 observation 判断；查询、时间窗口和 audit 仍在 service。
- `capture_helpers.py`：负责 continuous capture 摘要和确认回复文案。
- `conversation_helpers.py`：负责 speaker-labeled transcript 结构解析。
- `conversation_candidate_helpers.py`：生成多人 transcript 的结构化隐私 extraction plan，处理 user/known/unknown、alias 和敏感 fragment；不使用本地业务词表生成语义候选。

大多数 helper 只拆分实现归属；多人 transcript 的结构化隐私 plan 是例外，它会在普通 `PreReplyDecision` 前把输入切换到 reply-first 安全路径。该路径不改变 HTTP API 字段、数据库 schema 或最终 memory gate。

## 记忆写入

长期记忆不是自动保存所有聊天。写入流程是：

```text
MemoryWriteCandidate
-> should_write_memory_candidate()
-> rejected / requires_confirmation / saved
-> find similar memory
-> merge / supersede / add
-> EventMemoryStore
```

必须保留的边界：

- 敏感信息不能静默保存，包括密码、验证码、银行卡、API key、token、证件号等。
- 普通问题、临时上下文、低价值闲聊不应写长期记忆。
- 位置是当前 turn 临时状态，除非用户明确要求保存，否则不写长期记忆。
- 结构化记忆应尽量关联本轮 timeline evidence，方便后续解释来源。

## 召回与证据

系统按当前 turn 需要召回，不是每轮默认读全部历史：

- profile / preference：用户画像、长期偏好。
- event / task / decision / risk：事件、任务、决定、风险。
- timeline：原话证据和全文回忆。
- document：上传文档原文和片段。
- observation：后台轻量归纳。

召回结果只注入当前 turn，不改 system prompt。多来源同时出现时，由 recall arbitration 决定本轮主证据。

## Import、Capture 和文档

- `POST /api/memory/import` 支持文本和 JSON items，最终进入同一套候选、门控、去重和写库流程。
- Markdown 上传保存完整文档原文，不按每行拆成长期记忆。
- `/api/capture/start|append|stop` 用于连续文本或转写片段汇总；stop 后复用 import 管道。
- speaker-labeled transcript 在普通 `PreReplyDecision` 之前进入 reply-first 结构化隐私路径；安全片段逐说话人进入统一语义分类，候选携带 speaker 来源后再经过最终 memory gate。
- 新输入来源不要绕过写入门控，也不要绕过 `user_id` 隔离。

## Job、周报和提醒

- memory job 用于 reply-first 后台保存，生命周期仍由 `agent_bridge.py` 管；公开 payload 和阶段说明在 `memory_job_helpers.py` 里生成。
- `read_memory_job()` 可在服务重启后读取公开状态，但这不是可靠 worker 队列。
- `weekly_report()` 是启发式周报草稿，不是完整项目管理系统。
- `check_reminders()` 是手动检查未来 task 候选，不是主动推送 runtime。

## Web、位置和工具状态

实时问题应通过工具状态约束回答：

- 未触发 web search 时，不能声称“正在搜索”或编造实时结果。
- 搜索完成但无结果时，应明确没有可用结果。
- 有搜索结果时，应基于工具上下文回答。
- 定位缺失时不能猜城市或当前位置。

## 音频边界

![统一音频处理核心数据流](assets/frontend-audio-data-flow.svg)

```mermaid
flowchart LR
    Browser["浏览器麦克风"] --> Worklet["AudioWorklet 重采样<br/>16 kHz mono PCM16"]
    Worklet -->|"约 256 ms / sequence"| API["server.py<br/>audio session API"]
    API --> Session["AudioSession<br/>每用户独立 VAD/KWS/ASR cache"]
    Session --> Registry["AudioBackendRegistry<br/>唯一模型持有者"]
    Registry --> Models["共享模型 + 分离推理锁"]
    Session --> Events["audio_event.v1"]
    Events -->|"partial"| UI["实时 UI / debug"]
    Events -->|"wake_detected"| Ack["暂停上传 + 播放 ack_text"]
    Ack -->|"wake_ack_finished"| API
    Events -->|"final"| Planner["plan_audio_event()"]
    Planner -->|"chat / wake"| Chat["GlassesChatService.chat()"]
    Planner -->|"ambient"| Capture["Timeline capture"]
    Planner -->|"enroll"| Speaker["本人参考声纹"]
    Capture -->|"显式正常 stop"| Candidate["MemoryWriteCandidate"]
    Chat --> Candidate
    Candidate --> Gate["intent_policy + 敏感信息门控"]
    Gate --> Memory[("events.db")]
    Legacy["旧 blob / speaker API"] --> API
    API --> Registry
    Session -.-> Retention["PCM 仅进程内<br/>final/stop/error/timeout 释放"]
```

- `server.py` 仍是唯一 HTTP 入口；外部 `main.py`、`memory_runtime.py` 和外部数据库不进入运行时。
- 网页只有“开启/停止全天待机”一个日常音频控制。浏览器按约 256 ms POST PCM；引擎内部按 16 kHz、512 samples/32 ms 运行 VAD，流式模型可用时约每 0.512 秒产生 partial。
- KWS 命中后丢弃唤醒词片段，前端暂停麦克风上传并播放配置的回应；`wake_ack_finished` 后恢复同一 session，10 秒内等待用户开始提问，开口后不设固定时长，直到 VAD 静音 final。
- `transcript_partial` 只更新 UI/debug，不调用聊天、不追加 capture、不写 audit final、不创建 `MemoryWriteCandidate` 或 memory job。
- final 由 `plan_audio_event()` 分为 chat、capture、enroll 或 drop。ambient 逐个 `speech_end` 处理，正常 stop 才进入长期记忆候选；异常/超时标记 `interrupted`。
- 声纹只提供 `user/other/unknown` 和匿名 voice group 证据。`PRED_SPKxxxx` 是 session/capture 内临时标签，不是实名身份；API 不返回 embedding。
- `overlap=suspected/unknown`、他人、未知说话人、环境声和低置信 final 默认不能自动归人或写长期记忆。
- `/api/audio/segment/process`、`/api/speaker/enroll` 和 `/api/capture/*` 保持外部调用兼容，并与实时 session 共享同一 `AudioBackendRegistry`。网页不再运行旧 blob fallback；不支持 AudioWorklet 时明确提示浏览器不支持连续音频。

当前未实现操作系统后台常驻收音、真实眼镜设备鉴权、完整 diarization、重叠语音模型或联系人实名归因。

## API 入口

主要 API：

- `POST /api/chat`
- `GET /api/runtime`
- `GET /api/memories`
- `POST /api/memories`
- `DELETE /api/memories/{memory_id}`
- `GET /api/documents/{document_id}`
- `PATCH /api/documents/{document_id}`
- `DELETE /api/documents/{document_id}`
- `GET /api/memory/search`
- `GET /api/timeline/search`
- `GET /api/timeline/chunks`
- `DELETE /api/timeline/chunks`
- `GET /api/memory/jobs`
- `POST /api/memory/import`
- `POST /api/capture/start|append|stop`
- `GET /api/audio/capabilities`
- `POST /api/audio/session/start|push|control|stop`
- `POST /api/audio/segment/process`
- `POST /api/speaker/enroll`
- `GET /api/speaker/groups`
- `DELETE /api/speaker/groups/{group_id}`
- `GET /api/weekly-report`
- `GET /api/reminders/check`
- `GET /api/debug/audit`
- `POST /api/tts`

标准库 server 是唯一 HTTP 入口；业务行为应保持在 `GlassesChatService` 等 service 层。
