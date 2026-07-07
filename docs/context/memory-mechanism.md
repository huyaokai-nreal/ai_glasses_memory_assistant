# 记忆机制工程说明

当前 demo 的记忆目标不是保存所有聊天，而是把长期有用的个人上下文整理成可查看、可召回、可更新、可追溯的结构化记忆。

当前底层文字记忆机制已经形成第一版闭环：raw timeline 脱敏、候选抽取、写入门控、去重/冲突更新、生命周期、evidence 引用、召回仲裁、纠错、删除和机制级 eval 都有当前实现。它可以作为后续眼镜 ASR、App 音频和多来源输入的统一后端管道，但还不是生产级完整 AI 眼镜记忆系统。生产级差距见 `production-readiness.md`。

## Memory Kernel contract

当前记忆内核以 `memory_kernel.py` 为轻量 contract，不引入新依赖，只统一长期记忆的分层、来源追踪和召回证据。

四层边界：

| 层 | 作用 | 删除/更新原则 |
| --- | --- | --- |
| `raw_timeline` | 保存聊天原话、capture chunk 和全文搜索证据。 | 默认软删除 chunk；显式 hard purge 可物理删除未共享 chunk 和空父 turn/capture。 |
| `structured_memory` | 保存 profile、event、task、decision、assistant_preference 和 document metadata。 | 默认软删除；显式 `purge=true` 可物理删除同用户目标 memory/document。 |
| `reflection` | 保存后台 observation 归纳。 | 纠错时优先 `superseded`，不静默覆盖原始证据。 |
| `runtime_recall` | 当前 turn 里注入的少量 top-k 背景上下文。 | 只服务本轮回复，不持久化为新记忆。 |

统一追踪字段：

- `source_trace`：说明一条记忆、文档、timeline chunk 或 job 来自哪里、属于哪层、有哪些 evidence、隐私和删除策略是什么。
- `recall_trace`：说明一次召回由哪一层、哪种策略触发，召回了多少条、对应哪些 evidence。

这些 trace 是 debug/audit/API contract，不改变核心存储模型，也不把更多内容默认注入给主模型。

## 与 Hermes 记忆的边界

本 demo 的结构化个人记忆由 `memory_store.py` 的 `EventMemoryStore` 管理，默认落在 `AI_GLASSES_HOME/data/events.db`；未设置 `AI_GLASSES_HOME` 时使用独立默认目录 `~/.ai-glasses-memory-assistant/data/events.db`。原始时间线由 `timeline_store.py` 的 `TimelineStore` 管理，落在同目录 `timeline.db`，用于全文回忆和证据追溯。迁移期仍兼容旧 `HERMES_HOME`：如果只设置 `HERMES_HOME`，数据目录保持为 `HERMES_HOME/ai_glasses_memory_assistant`。

它不是：

- Hermes 内置 `MEMORY.md` / `USER.md`
- Hermes memory provider recall
- Hermes `session_search`
- Hermes 聊天历史全文库

所有读写都必须带 `user_id`，跨用户不可见。

## 数据模型

核心表是 SQLite `memories`；上传的 Markdown 文档另存到同一个库里的 `documents` 表。

顶层类型：

| kind | 含义 | 例子 |
| --- | --- | --- |
| `profile` | 稳定用户画像、偏好、习惯 | 用户喜欢安静靠窗的位置。 |
| `event` | 已发生或计划中的事件、任务、会议、承诺 | 明天下午 3 点和 Alex 开周会。 |
| `assistant_preference` | 用户对助手行为的偏好 | 以后回答短一点。 |

细分类：

| memory_type | 用途 |
| --- | --- |
| `fact` | 稳定事实。 |
| `task` | 任务、待办、承诺、提醒候选。 |
| `decision` | 决策和结论。 |
| `preference` | 用户偏好。 |
| `project_state` | 项目进展、风险、卡点。 |
| `observation` | 后台长期归纳结果，例如最近关注点、工程偏好或项目状态。 |
| `event` | 无法更细分时的默认事件。 |

重要字段：

| 字段 | 作用 |
| --- | --- |
| `user_id` | 用户隔离边界。 |
| `content` | 展示、搜索、召回使用的记忆文本。 |
| `source` / `source_id` | 来源和来源内 ID。 |
| `ingestion_id` | 一次导入或一次处理链路的追踪 ID。 |
| `evidence_ids` | 支撑这条记忆的证据。 |
| `privacy_level` | `normal`、`sensitive`、`requires_confirmation`。 |
| `status` | `active`、`superseded`、`stale`、`deleted`。 |
| `confidence` | 候选置信度。 |
| `access_count` | 这条结构化记忆实际进入回复上下文的次数。普通列表读取不会增加。 |
| `last_accessed_at` | 最近一次被召回并带入回复上下文的时间戳。 |
| `strength` | 轻量基础强度分，初始来自 `confidence` / `memory_type`，后续随访问次数小幅增强并受类型上限约束。 |
| `start_at` / `end_at` | 事件或计划时间范围。 |
| `temporal_text` | 用户原始时间表达。 |

observation 额外使用轻量 tag 表达主题边界：

- `observation_scope:recent_activity`：最近在忙什么、主要投入方向。
- `observation_scope:engineering_preference`：工程偏好、长期偏好、回答风格和证据偏好。
- `observation_scope:project_state`：项目状态、进展、风险、卡点、主线。
- `observation_scope:general`：暂时无法归类的长期归纳。

这是写在现有 `tags` 字段里的兼容信息，不新增表；旧 observation 没有 scope tag 时会按内容推断。

访问强度只用于召回排序和 debug 解释，不触发自动删除或保留分淘汰。旧库启动时会增量补列，并给缺少强度的旧记忆补一个初始 `strength`。

排序时不会直接使用持久化 `strength`。`memory_store.py` 会按 `memory_type` 计算运行时 `effective_strength = capped_strength * decay_factor`：

- `preference` 衰减较慢，保留稳定偏好的长期价值，但仍受 cap 限制。
- `task`、`project_state`、`decision` 更重视近期性，旧状态不应长期压过新状态。
- `observation` 轻度衰减，避免旧总结替代新的 source memory。

`effective_strength` 只影响排序和 debug，不反写数据库。文本搜索 ranking 会暴露 `base_strength_score`、`effective_strength_score`、`strength_cap`、`decay_factor` 和 `strength_policy_reason`，用于解释例如“半年前高频靠窗偏好为什么没有压过近期吧台偏好”。

### confidence 解释边界

`confidence` 不是一个全局可直接比较的“事实真实度”。当前用 `memory_confidence.py` 集中定义各链路阈值和 debug 解释：

| purpose | 含义 | 低置信度处理 |
| --- | --- | --- |
| `memory_write_candidate` | 候选事实/价值抽取置信度。 | 低于 `0.6` 拒绝保存。 |
| `question_profile_write` | 从问题式输入中抽到稳定画像的置信度。 | 低于 `0.85` 拒绝保存，避免把用户问题写成记忆。 |
| `dedupe_relationship` | 新候选和旧记忆之间的重复/冲突关系置信度，不代表事实本身真实性。 | 低于对应阈值回退为 `new`，不误合并、不误 supersede。 |
| `correction_detection` | 这句话是否在纠错/替换旧记忆的意图置信度，不代表新事实真实性。 | 低于 `0.75` 忽略纠错候选，普通聊天继续走原链路。 |
| `observation_update_relationship` | 新 observation 和旧 observation 的合并/替换关系置信度，不代表总结真实性。 | 低于 `0.75` 回退为 `new`，后台 reflect 不被阻断。 |
| `correction_target_resolution` | 新纠错记忆要替换哪条旧 active memory 的定位置信度，不代表新事实真实性。 | 低于 `0.75` 不 supersede 旧记忆，只保存新候选并暴露 fallback。 |

相关 debug/audit payload 会带 `confidence_policy`，包含 `purpose`、`meaning`、`confidence`、`min_confidence`、`passed`、`treatment`、`backend` 和 `band`。例如 dedupe 返回 `duplicate` 但置信度只有 `0.5` 时，系统会保存为新记忆，并在 debug 中标明 `treatment="fallback_to_new"`，而不是悄悄把两条可能不同的记忆合并。

### documents 表

`documents` 保存用户上传文档的原文和元数据，不是长期事件记忆。

| 字段 | 作用 |
| --- | --- |
| `filename` | 用户上传时的文件名。 |
| `title` / `summary` | 用于识别文档主题，不作为细节问答的唯一依据。 |
| `content` | 完整 Markdown 原文。 |
| `content_hash` | 原文哈希，用于排查和去重。 |
| `source` / `ingestion_id` | 来源和一次导入链路追踪。 |
| `created_at` | 上传归档时间。 |
| `updated_at` / `status` / `deleted_at` | 文档编辑和软删除状态，普通列表和召回只读 active 文档。 |

`GET /api/memories` 会返回文档 metadata 供“当前记忆”面板展示，但不返回 `content`。编辑文档时才通过 `GET /api/documents/{id}` 读取完整 Markdown；`PATCH /api/documents/{id}` 更新原文后会重新计算 `content_hash`，后续细节问答使用更新后的原文。

### timeline 表

`timeline_store.py` 维护独立的 raw timeline，不把大段原话塞进 `memories`。

| 表 | 作用 |
| --- | --- |
| `raw_turns` | 保存每轮用户原话、助手回复、session/interaction 线索。 |
| `chunks` | 保存可搜索的原文片段，来源可以是 chat turn 或 capture。 |
| `captures` | 保存连续输入的 capture 元数据，service 重启后可从 SQLite 恢复 stop/import。 |
| `memory_jobs` | 保存后台记忆 job 的公开 payload，重启后仍可查询 pending/saved/rejected/skipped/failed 状态；不是可靠 worker 队列。 |
| `chunks_fts` | SQLite 支持 FTS5 时用于全文搜索；不可用时走 LIKE fallback。 |

timeline 的用途：

- 跨 session 回忆用户“之前原话说过什么”。
- 给结构化记忆提供 `evidence_ids`。
- 调试导入、capture 和记忆抽取来源。
- 通过 `/api/timeline/chunks` 查看 evidence chunk、引用计数和 active/retained 保留原因，并执行批量软删除或受保护的彻底删除。
- 默认删除结构化记忆时，未被其他 active 记忆引用的 evidence chunk 会同步软删除。
- 显式 hard purge 结构化记忆时，未被其他 active 记忆引用的 evidence chunk 会被物理删除；如果对应 `raw_turn/capture` 已没有任何 chunk，父记录也会被物理删除。

timeline 不是：

- 每轮默认注入的大上下文。
- 结构化 `profile/event/assistant_preference` 记忆替代品。
- 可直接编辑 raw chunk 原文的审计后台。
- 原始音频存储。

## 多输入来源原则

最终产品会有眼镜语音、App 文档、App 音频、手动文本等多个入口。入口可以不同，但长期记忆写入规则必须一致：

| 来源 | 当前或未来处理方式 |
| --- | --- |
| `/api/chat` | 由 planner 生成候选，再过门控。 |
| `/api/memory/import` 文本/JSON | 标准化后生成候选，再过门控。 |
| `source="markdown_upload"` | 先保存为 document；不能把每一行静默拆成 event。 |
| `/api/capture/append` / `stop` | append 写入 timeline chunk；stop 合并长输入文本后走 import。 |
| 聊天内长输入 | planner 命中 `continuous_capture` 后低打扰回复，后台分段抽取候选并走同一写入门控。 |
| 未来眼镜 ASR / App 音频 | 先转写为文本，再走 import 或 capture；原始音频不是长期事件记忆。 |
| 未来 App 文档 | 先按 document 归档，需要长期价值时再抽候选并过门控。 |

不要因为来源是音频、文档或 App 上传，就跳过 `should_write_memory_candidate()`。文档和音频可以作为证据来源，但长期记忆仍应是经过抽取、可删除、可追溯的结构化事实。

阶段 C 第一/二刀已支持一种文本级多人转写输入：明确 `[时间][speaker] 原话`、`[speaker] 原话`、`speaker：原话` 或 `speaker: 原话` 的文本先解析成 `ConversationSession / ConversationTurn`，再生成以用户为主体的 `MemoryWriteCandidate`。例如“用户和张三约定客户拜访分工：张三带合同，用户准备 PPT”。旁人的私人偏好、unknown speaker 敏感片段和验证码不会写成用户画像或长期事件，只会在 `conversation_session.rejected_turns`、memory job extraction trace 或 import debug 中解释。Debug 还会暴露 `parsed_turns`、`candidate_turn_indices` 和参与人角色，方便判断某个 turn 为什么进入候选或被拒绝。这个能力复用现有 import/capture、`_save_memory_candidates()` 和 `should_write_memory_candidate()`，没有新增音频依赖，也没有做 DB schema migration。

### 聊天内长输入整理

聊天内长输入不等于会议纪要。`turn_planner.py` 会用组合信号判断长输入形态：长度和句子/分句数量只是入口，还会参考时间、人物、地点、项目、动作、状态变化和第一人称经历等信号。普通长问题、知识解释和单一主题写作请求仍走普通 LLM 回复。

命中 `reply_mode="continuous_capture"` 后：

- 本轮先写入脱敏 raw timeline/chunk，并立即返回低打扰确认。
- 后台 `memory_job` 按分段调用内部 LLM 分类器抽取候选，失败时只用轻量规则兜底。
- 候选类型可以是 `task`、`decision`、`project_state`、`event`、`preference` 等。
- 每条候选仍走 `should_write_memory_candidate()` 和 `_save_memory_candidates()`；敏感 token、密码、证件号等不会默认保存。
- 长输入保存后如果触发 `observation_reflect`，当前使用本地规则归纳，不让低打扰整理链路依赖后台 LLM 网络。
- 低价值长闲聊没有高价值候选时，job 可以是 `skipped/rejected`，只保留 raw timeline 供原话检索和证据追溯。

debug/audit 只记录 `reply_mode`、分段数量、job 状态、保存/拒绝数量和拒绝原因；用户原文类字段会经过敏感脱敏。

## 写入原则

写入长期记忆必须经过候选和门控：

```text
用户输入
-> MemoryWriteCandidate
-> should_write_memory_candidate()
-> saved / rejected / requires_confirmation
-> add_memory() 或 merge_memory_evidence()
```

可以保存：

- 用户明确自我介绍、稳定偏好、长期习惯。
- 明确事件、任务、会议、承诺。
- 用户要求记住的非敏感信息。
- 导入文本中结构清晰且有长期价值的条目。

不应保存：

- 问候、闲聊、普通百科问答。
- 临时状态，例如“刚刚路过”“现在在这里”。
- 用户的问题本身。
- 置信度低、内容空泛或无法判断价值的候选。
- 密码、验证码、银行卡、API key、token、证件号等敏感信息。

需要确认：

- 候选显式标记为 `requires_confirmation`。
- 隐私等级高但可能有长期价值的内容。

## 自然偏好

自然偏好是当前 P0。

应保存的例子：

```text
我不喜欢排队很久的餐厅。
我喜欢安静靠窗的位置。
我喝过 coco 的杨枝甘露，那是我觉得最好的饮品。
```

目标结果：

- `kind=profile`
- `memory_type=preference`
- 后续推荐、订座、选择问题能召回。

实现时不要只加单句 hard code。优先补同类偏好表达识别，并保留敏感信息负例。

## 去重与更新

当前 store 支持最小更新能力：

- `find_similar_memory()`：找相似 active 记忆。
- `merge_memory_evidence()`：相似命中时合并证据和置信度。
- `mark_superseded()`：新事实覆盖旧事实。
- `mark_stale()`：保留可追溯记录，但不再参与 active 召回、访问增强或 observation source。
- `delete_memory()`：软删除结构化记忆；服务层会同步清理未共享的 timeline evidence chunk。
- `purge_memory()`：显式彻底删除结构化记忆；只清理当前用户、不再被其他 active memory 引用的 evidence chunk，并清理空父级 `raw_turn/capture`。
- `purge_document()`：显式彻底删除文档归档；第一版只物理删除 `documents` 行，不联动 timeline。
- `purge_audit_records()`：hard purge 后按同用户目标 id 递归命中删除整条 JSONL audit 行；其他用户记录和无法解析的 JSONL 行保留。

原则：重复输入不应无限堆积；纠正和更新不应直接删除历史证据，除非用户明确删除。普通删除仍保留软删除记录用于排查；只有 API 或前端传入 `purge=true` 时才进入不可恢复清理。

语义去重分三层，全部复用 `_save_memory_candidates()` 主写入路径：

- 第一道仍是 `find_similar_memory()` 精确去重，命中后直接 `merge_memory_evidence()`，不调用 LLM。
- 精确去重没命中、且后台写入或当前链路已有 agent 时，才对特定类型做 LLM 辅助判断。同步文字写入会复用已有 `session.agent` 做语义去重；没有可用 agent 时不为去重额外创建主模型会话。
- LLM 只能返回 `duplicate`、`conflict` 或 `new`。低置信度、非法 JSON、无 agent、memory id 不合法时都回退为 `new`，不阻断写入。
- `duplicate` 会合并 evidence/source/confidence 到旧记忆，不新增 active 行。
- `conflict` 会先保存新记忆为 active，再把旧记忆标记为 `superseded`，旧记忆仍可追溯但不参与 active recall、access 更新或 observation source。
- `memory_processing`、memory job 和 background audit 会带出 `dedupe_decisions` 与 `superseded_memory_ids`，用于解释为什么合并或替换。

已覆盖类型：

- `kind=profile` 且 `memory_type=preference`：和当前用户 active profile preference top 20 比较。比如“我喜欢安静靠窗的位置”和旧的“用户喜欢靠窗座位”可合并；“我更喜欢吧台位置”可替换旧座位偏好。
- `kind=event` 且 `memory_type` 为 `event/task/decision/project_state`：只和当前用户同类型 active 结构化记忆比较。`event/task` 有明确时间时优先比较相近时间窗口内的候选；无明确时间时取最近 active top 20。`decision/project_state` 按最近更新时间取 top 20。
- `memory_type=observation`：不走通用 dedupe；由 observation 更新决策层单独处理。

结构化事件类例子：

- 用户先说“周五前补 memory eval”，后说“这周五之前把记忆机制评估补上”，在 LLM 判断为 `duplicate` 时只合并 evidence，不新增第二条 active task。
- 用户先说“项目决定先做本地记忆门控”，后说“结论更新为先补结构化去重”，在 LLM 判断为 `conflict` 时保存新 decision，并把旧 decision 标记为 `superseded`。
- 用户先说“项目状态是先做南太行攻略”，后说“现在主线是 AI 眼镜长期记忆”，在 LLM 判断为 `conflict` 时旧 project_state 会退出 active 回顾。

这里的 classifier `confidence` 只表示“更新关系判断置信度”，不表示候选事实本身真实性。候选是否能保存仍由 `should_write_memory_candidate()` 的敏感信息、问题文本、候选置信度等写入门控决定；更新关系低置信度时回退为 `new`，避免误合并或误替换。

task 业务状态是独立于 lifecycle status 的二级语义：

- lifecycle status 仍只表示记录是否进入 active surface：`active/stale/superseded/deleted`。
- task 业务状态写在 tags 中：`task_status:open`、`task_status:completed`、`task_status:cancelled`。没有 task status tag 的历史 task 按 open 兼容处理。
- 普通新 task 默认写入 `task_status:open`。包含“完成/做完/搞定/取消/不做了”等明确表达的 task 候选会写成 completed/cancelled 状态事实。
- 当 completed/cancelled 候选被 dedupe 判断为 conflict 时，旧 open task 会被标记为 `superseded`；新 completed/cancelled fact 仍保持 active，方便之后回答“我完成了什么/取消了什么”。
- 待办召回、未来计划召回和手动 `check_reminders()` 只读取 open task；completed/cancelled 不再作为未完成待办或提醒出现，但不等于删除，仍可通过审计、周报和相关状态问题追溯。
- `memory_processing`、memory job 和 background audit 暴露 `task_status_updates`，用于解释哪个旧 open task 因完成或取消退出了待办面。

task 状态例子：

- 用户先说“周五前补 memory eval”，后说“周五前补 memory eval 已完成”，后续问“我还有什么待办？”不会再召回这条旧 task。
- 用户先说“周五下午复盘会”，后说“取消周五下午复盘会”，手动提醒检查不会再提醒这个会。
- 周报会把 `task_status:open` 放到“待办/计划”，把 `task_status:completed` 放到“已完成任务”，把 `task_status:cancelled` 放到“已取消任务”。

纠错识别与目标定位：

- 显式纠错仍走本地规则，例如 `纠正一下`、`说错了`、`不是...而是...`。
- 自然表达规则覆盖 `其实我现在主要在...`、`更准确地说...`、`改一下...`、`我之前说的不对，应该是...`，以及 `前面那个座位偏好更新一下`、`项目主线说错了`、`周五前补 memory eval 不是完成，是取消了` 这类隐式对象纠错。
- 普通偏好表达不会因为带有“其实”就进入纠错模式，例如 `其实我喜欢低糖拿铁` 仍按普通 preference 写入。
- 已有 `session.agent` 的普通链路可以用 LLM 兜底判断自然纠错；低置信度、非法 JSON、空内容、问题句或敏感候选都会回退为非纠错，不阻断聊天。
- 新纠错候选保存成功后，会先尝试定位当前用户同类 active 旧记忆：`profile/preference`、`event/task`、`decision`、`project_state`。本地规则使用旧对象线索、scope 词、task 状态、项目 tag 和时间字段；目标不明确且已有 agent 时才调用 LLM target resolver。
- 目标定位低置信度、非法 id、跨类型或无明确旧线索时，不 supersede 旧记忆；新候选仍可按普通写入门控保存，debug 中标明 `fallback_no_supersede`。敏感候选如果被 `should_write_memory_candidate()` 拒绝，不允许替换任何旧记忆。
- 保存后的 `source="correction"` 候选仍会触发相关 active observation `superseded`，避免旧 observation 继续影响“最近在忙什么”这类回顾。
- debug、memory job 和 background audit 会暴露 `correction_detection` 与 `correction_target_resolution`。后者包括 `candidate_count`、`matched_memory_ids`、`superseded_memory_ids`、`resolution_backend`、`fallback_reason` 和可选 `confidence_policy`。

纠错目标定位例子：

- 旧记忆是“用户喜欢靠窗座位”，用户说“我前面那个座位偏好更新一下，我喜欢吧台位置”，旧靠窗偏好会变为 `superseded`，active 画像只剩吧台。
- 旧 task 是“周五前补 memory eval”，用户说“周五前补 memory eval 不是完成，是取消了”，旧 open task 退出待办召回，新取消事实仍可追溯。
- 用户只说“那个信息更新一下，我现在喜欢吧台位置”，LLM target resolver 低置信度时不会误替换旧记忆，避免把不明确纠错误伤到 unrelated 画像。

observation 更新第一版：

- 后台 `observation_reflect` 生成候选后，不再直接新增 active observation，而是先和当前用户 active observation 做本地更新判断。
- `merge`：新候选和旧 observation 的 `evidence_ids` 高度重叠，或主题词高度相似时，只合并 evidence/source/confidence 并更新旧 observation 的 `updated_at`，不新增重复 active 行。
- `supersede`：新候选明显替代旧总结时，先保存新 observation，再把旧 observation 标记为 `superseded` 并写入 `superseded_by`。比如旧总结是“用户最近主要在做南太行项目”，新证据显示“用户最近主要在做 AI 眼镜长期记忆和 observation 更新机制”，旧总结会退出 active recall。
- `new`：无 active observation、无足够 overlap、非法 id、低置信度或无法判断时，按新 observation 保存，不阻断后台 reflect。
- `skip`：只作为判断层的低置信度信号，实际写入链路会回退为 `new`，避免后台 job 因判断不稳而静默丢失候选。
- `memory_job`、`read_memory_job()` 和 background audit 会暴露 `observation_update_decisions`，每条包含 `action`、`observation_id`、`reason`、`confidence` 和 `backend`。

召回治理第一版：

- `agent_bridge.py` 在 `profile_memories` / `event_memories` 初步召回后、写入 `debug["memory"]` 和 `_message_with_recall()` 前执行本地 `drift_guard`。
- `strength`、`access_count` 和 ranking 只决定排序，不再允许高强度旧 profile 绕过当前 query 相关性。比如用户问 FastAPI 路由机制时，旧的“用户喜欢低糖拿铁”不会进入主 LLM prompt，也不会因为这次被过滤而继续增加 access。
- profile 记忆默认要求和当前问题主题相关；饮品、餐厅、座位等偏好只在对应主题问题中保留。`关于我 / 我是谁 / 身份` 这类宽泛或身份查询保留已有行为。
- 当前表达出现 `不要`、`不考虑`、`不按`、`别` 等覆盖信号时，和旧偏好冲突的 profile 不进入本轮回复上下文。比如“这次订座不要靠窗”会挡住旧的“用户喜欢靠窗位置”。
- event `text_search` / `observation_review` 保留已有事件过滤，只补 guard debug；时间范围事件召回、未来计划召回和 timeline 原文召回不改时间或原文优先语义。
- access 记录只针对 guard 后真正进入回复上下文的记忆。debug 会暴露 `debug["memory"]["drift_guard"]`，包括 `enabled`、`checked_count`、`kept_count`、`filtered_count`、`filtered_memory_ids`、`filtered_reasons` 和 `skipped_reason`。

多来源召回仲裁第一版：

- `memory_recall_arbitration.py` 在召回后、回复前统一处理 document、raw timeline、structured event、observation 和 profile 的优先级；它是纯策略层，不读写数据库、不调用 LLM、不改变 API 响应形状。
- 文档细节问题以 archived document 原文为主，相关 observation/profile/event 不进入主回复上下文。比如问“攻略里红旗渠门票多少钱”，即使 observation 里提过南太行，也不能替代文档原文。
- `recall_goal="raw_evidence"` 时以 timeline chunk 为主，observation 不能替代原话。比如问“我之前有没有说过语音识别不稳定的原话”，系统只返回 raw timeline 命中的原话。
- `recall_goal="specific_fact"` 时只保留直接 profile/event 证据，默认丢弃 observation；如果没有直接证据，会触发 `empty_evidence_guard` 本地保守回复，避免主 LLM 编造具体个人细节。
- `recall_goal="summary"` 时 observation 可以作为高层线索，但“最近/近期/这段时间”类活动总结会过滤稳定 profile 背景，避免把“用户名字叫...”说成最近活动。
- debug 会暴露 `debug["memory"]["recall_arbitration"]`，包括 `primary_source`、`kept_counts`、`dropped_counts`、`decisions` 和 `empty_evidence_guard`，方便排查为什么某个来源被保留或过滤。

当前状态语义：

| status | 当前行为 |
| --- | --- |
| `active` | 当前有效，能展示、召回、访问增强和作为 observation source。 |
| `superseded` | 被新记忆替代，可追溯但不参与普通召回。 |
| `stale` | 过期或不再当前有效，可追溯但不参与普通召回、访问增强和 observation source。 |
| `deleted` | 用户删除后的软删除状态，不参与普通召回；未共享 evidence chunk 会同步软删除。 |

## 删除与保留策略

当前删除策略分两层：

- 默认 `DELETE /api/memories/{id}?user_id=...` 和 `DELETE /api/documents/{id}?user_id=...` 仍是软删除，响应保持 `{"deleted": true}`，用于兼容现有 UI、测试和排障习惯。
- 显式 `purge=true` 是不可恢复 hard purge：`DELETE /api/memories/{id}?user_id=...&purge=true`、`DELETE /api/documents/{id}?user_id=...&purge=true` 会返回 `deleted`、`purged`、`purged_chunk_count`、`purged_parent_count`、`audit_records_removed`。

memory hard purge 的边界：

- 先读取当前用户任意状态的目标 memory，再物理删除 `memories` 行；因此已软删除的 memory 仍可继续彻底清理。
- 只物理删除不再被其他 active memory 引用的 `evidence_ids` 对应 timeline chunks。共享 evidence 仍被其他 active memory 使用时会保留。
- 如果被删除 chunk 的父级 `raw_turn/capture` 已没有任何剩余 chunk，则物理删除父记录；如果仍有 chunk，则保留父记录并刷新父级状态。

document hard purge 的边界：

- 物理删除 `documents` 行，文档列表、读取和搜索都不再可见。
- 第一版不影响 timeline；除非后续建立 document evidence 关系，否则删除文档不会清理 raw timeline。

audit 清理：

- hard purge 会原子重写 `chat_audit.jsonl`，删除同用户且递归包含目标 memory/document id、被 purge chunk id 或被 purge parent id 的整条 audit 行。
- audit 不做局部改写；一条 audit 如果同时包含目标 id 和其他信息，会整行删除。
- 其他用户记录保留，无法解析的 JSONL 行保留，避免误删未知数据。
- hard purge 不追加新的 audit 行，避免刚删除的目标 id 又出现在 audit 中。

当前不做自动 retention、strength decay、用户级全清理、后台 hard purge worker 或自动 `VACUUM`。SQLite 物理文件可能不会因为行删除立刻缩小；如需释放磁盘空间，后续应单独设计明确的 vacuum/维护入口。

## 召回原则

召回由 `turn_planner.py` 决定，不是每轮默认发生。

常见召回：

- 用户问“我是谁”“我喜欢什么” -> profile。
- 用户问“昨天吃了什么”“接下来有什么安排” -> event + 时间范围。
- 用户问“我最近在忙什么”“我的工程偏好是什么”“这个项目现在什么状态” -> observation。
- 用户问“我之前原话怎么说的”“我上次提到过什么” -> timeline chunk。
- 用户问“我什么时候上传过什么文档” -> documents metadata。
- 用户问上传文档里的细节 -> 召回原始 Markdown 或命中的原文片段。
- 用户问普通事实 -> 不读个人记忆。
- 用户问位置相关 -> 使用当前 turn 的 location，不读长期位置记忆。

时间查询使用半开区间：

```text
memory_start < query_end AND memory_end > query_start
```

近期计划类问题可以补充查询没有明确时间的近期 event，避免“已保存但没时间字段”完全召回不到。

## 后台长期归纳 observation

`observation` 是第一阶段新增的轻量 reflect 层，用来补足 profile/event 只能保存离散事实、但不会主动归纳长期模式的问题。

触发和写入边界：

- 聊天、fast path 或 import 完成 profile/event 写入后，后台检查是否已有至少 3 条带 `evidence_ids` 的 active `profile/event`。
- 满足阈值且距离上一条 observation 足够久时，创建 `mode="observation_reflect"` 的 memory job。
- 后台只读取当前用户最近 active 的 `profile/event`，排除已有 `observation`，不读取文档全文、位置状态或 Hermes core 记忆。
- 已 `deleted/stale/superseded` 的 source memory 不参与归纳；observation 自身也不作为下一轮 observation source。
- 归纳候选仍写入同一张 `memories` 表：`kind="event"`、`memory_type="observation"`、`source="observation_reflect"`。
- 新 observation 会带 `observation_scope:*` tag；旧 observation 没有 tag 时按内容推断 scope。
- 候选必须保留 `evidence_ids`；没有证据不保存。
- 候选继续经过 `should_write_memory_candidate()`，敏感信息仍会被拒绝或要求确认。
- observation update 只在同一 scope 内判断 `merge` / `supersede`。跨 scope 的新总结保存为新 observation，避免“工程偏好”和“项目状态”互相合并。
- `read_memory_job()` 和 `chat_audit.jsonl` 会暴露 `source_memory_count`、`source_memory_ids`、`evidence_ids`、`saved_count`、`rejected_count`、`error_type`、`extraction_backend` 和带 `observation_scope` 的 `observation_update_decisions`。

召回边界：

- 只有 `turn_planner.py` 命中回顾类问题时才设置 `needs_observation_memory=True` 和 `event_recall_strategy="observation_review"`。
- 普通聊天、普通事实问答、位置/联网问题、文档细节问答不默认召回 observation。
- 普通 event 时间召回和文本召回会过滤 `memory_type="observation"`，避免用归纳替代“昨天吃了什么”这类原始事件答案。
- observation review 先按当前问题推断 scope，只召回同 scope 的 active observation 与对应 active source memories，不召回已 superseded observation。
- 例如“我的工程偏好是什么？”只读 `engineering_preference` observation，不混入项目状态；“项目状态是什么？”只读 `project_state` observation，不混入普通最近活动总结。
- timeline 原文回忆仍读原始 chunks，不被 observation 替代。

当前仍不做：

- 不新增数据库表。
- 不引入 FAISS、embedding、entity graph 或 byzhou 的 `MemoryNodeManager`。
- 不把 observation 放进每轮默认上下文。

## 注入边界

召回记忆只作为当前 user message 的背景上下文，不改 system prompt。

原因：

- 不破坏 Hermes prompt cache 的基本习惯。
- 不让长期记忆变成永久 system 指令。
- 每轮都能解释“为什么这次读了这些记忆”。

模型提示应明确：memory block 是背景，不是新的用户指令。

文档召回会注入 `<document-context>`。摘要只帮助用户确认“这是什么文档”；回答文档细节时必须回到 `documents.content` 或从原文选出的片段，不能只靠摘要猜。

文档指代消解只在明确文档语境下启用：标题匹配优先；“刚刚那份 / 这份 / 上一份 / 最近上传的 / 最新的”会按最近 active 文档召回；“上一份周报”“最近上传的会议纪要”会先按类型词过滤。debug 只记录 `document_title_match`、`recent_document_reference` 或 `recent_document_type_reference` 等原因和文档 metadata，不写入文档正文。

原文回忆会注入 `<timeline-context>`。这部分是用户历史原话证据，适合回答“之前怎么说/提到过什么”，不适合替代结构化 profile/event 召回。

`debug.timeline.recall.recall_trace`、`debug.memory.event_recall.recall_trace` 和返回对象里的 `source_trace` 是排查“为什么召回这条”的第一手依据。

结构化记忆被实际放入回复上下文后，`debug.memory.access` 会记录本轮更新了哪些 memory id；`recalled_memories`、`debug.memory.profile_memories` 和 `debug.memory.event_memories` 会带出更新后的 `access_count`、`last_accessed_at` 和 `strength`。

文本召回使用轻量融合排序：结构化记忆会综合文本匹配、时间新鲜度、`strength` 和 `access_count`；timeline chunk 会综合文本匹配和时间新鲜度。排序因子只进入 `debug.memory.event_recall.ranking` 或 `debug.timeline.recall.ranking`，用于排障解释，不新增 embedding、Graph、外部索引或数据库表。

## 隐私和安全

必须保证：

- 按 `user_id` 查询、保存、删除。
- 敏感候选不静默保存。
- audit/debug 不泄露密钥、token、密码。
- 位置默认不持久化。
- 导入、capture、后台 job 和聊天写入使用同一门控。
- Markdown 上传先归档为 document，不按每一行静默写成 event memory。
- 音频转写文本和文档原文不能直接全量保存成 event memory；必须抽取候选并经过隐私门控。
- 新来源必须记录 `source`、`source_id`、`ingestion_id` 或 `evidence_ids`，方便用户追溯、删除和纠正。
- timeline 原文可用于证据和全文回忆，但不应在普通问答里默认注入。
- 删除记忆后，对应未共享 timeline chunk 不应继续被原文搜索或原文回忆召回。

新增记忆字段或新来源时，先回答：

- 来源是什么。
- 是否有用户授权。
- 是否可能包含敏感内容。
- 如何删除或纠正。
- debug/audit 如何解释。

## 验证重点

改记忆逻辑至少验证：

- profile 写入和召回。
- event 写入和时间召回。
- 自然偏好写入。
- 闲聊和普通问答不保存。
- 敏感信息拒绝或确认。
- 多用户隔离。
- 后台写入 saved/rejected/failed 可见。
- Markdown 上传只新增一个 document，后续细节问题能从原文召回。
- timeline 原话按 `user_id` 隔离，跨 session 原文回忆能召回相关 chunk。
- 机制级 eval 门禁覆盖 active 记忆面和 debug 决策，不只检查本轮回复文本。

当前 P1.M 机制级 eval 门禁：

- `evals/metrics.py` 支持 `active_memory_contains`、`active_memory_not_contains`、`active_memory_count` 和 `debug_contains`，用于检查 active 记忆最终状态和关键决策痕迹。
- `evals/runner.py` 的 fake agent 支持 `intent_payloads`、`dedupe_payloads`、`correction_payloads`、correction target resolver 和 `observation_update_payloads`，并提供真实 `reminder_check` / `weekly_report` action，可稳定模拟 LLM 分类器和机制级非聊天入口，避免机制回归依赖真实网络。
- `category=memory_mechanism` 覆盖重复偏好合并、冲突偏好 supersede、结构化 task 合并、task 完成/取消、提醒跳过 closed task、项目 scope 周报分组、project_state/decision 替换、低置信度 dedupe 回退、自然纠错后旧 observation 退出 active 回顾、隐式纠错定位旧 preference/task/project_state、低置信度 target fallback，以及当前意图压过历史偏好的 drift guard。
- hard purge、timeline evidence 清理和 SQLite 物理删除仍放在单元/API 测试中验证，不强塞进聊天 eval。
