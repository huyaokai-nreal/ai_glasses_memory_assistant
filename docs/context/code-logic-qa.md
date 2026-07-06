# 代码逻辑 Q&A

> 状态：学习材料。用于帮助理解当前代码，不替代 `pipeline.md`、`memory-mechanism.md`、`code-map.md` 这类正式上下文文档。

本文用于收集阅读和开发过程中出现的高密度代码逻辑解释。目标是把“当时听懂了、以后还会忘”的关键理解沉淀下来，方便后续继续追代码、改功能和复盘设计边界。

使用规则：

- 按大章节归类，优先围绕真实调用链、数据流和边界条件记录。
- 每个问题尽量包含“短答案”“调用链”“例子”。
- 只记录当前代码真实行为，不把计划能力写成已完成。
- 如果代码后来变化，直接更新对应 Q&A，并保留关键差异说明。

## 1. 记忆召回链路

### 1.1 用户 query 是不是由 LLM 直接查 SQLite？

短答案：不是。用户输入不会直接交给 LLM 去查 SQLite，而是先被规划成召回类型，再由本地存储层查询。

核心链路：

```text
用户输入 message
-> Planner baseline / PreReplyDecision 先决定召回类型
-> memory_store / timeline_store 用时间范围、FTS、LIKE、排序和过滤去查
-> 查到的结果作为上下文交给回复层
```

换句话说：

```text
不是 LLM 直接查 SQLite；
是 Planner baseline / PreReplyDecision 先决定召回类型；
再由 memory_store / timeline_store 用时间范围、FTS、LIKE、排序和过滤去查；
查到的结果再作为上下文交给回复层。
```

主要代码入口：

| 阶段 | 文件 / 函数 | 作用 |
| --- | --- | --- |
| 本地规划 | `turn_planner.py::plan_turn()` | 判断是否需要读 profile/event/timeline、是否写记忆、是否走本地 fast path。 |
| LLM 回复前决策 | `turn_semantic_classifier.py::classify_pre_reply_decision()` | 在 `llm_first` 下用一次 LLM 输出 `PreReplyDecision`，同时判断 `location/web/recall/reply_mode` 和记忆候选，不直接回答用户。 |
| 聊天主流程 | `agent_bridge.py::GlassesChatService.chat()` | 串起 planner baseline、PreReplyDecision、召回、回复、后台记忆写入和 debug/audit。 |
| 结构化记忆查询 | `memory_store.py::search_with_ranking()` / `list_events_between()` | 从 SQLite 的 `memories` / FTS 表中查 active 结构化记忆。 |
| 原文证据查询 | `timeline_store.py::search_chunks_with_ranking()` | 从 raw timeline chunk 中查原话和证据片段。 |

典型例子：

```text
用户：我明天有什么安排？
```

执行理解：

```text
turn_planner.py 识别为 event recall query
-> resolve_temporal_local() 解析“明天”的时间范围
-> agent_bridge.py::_recall_event_memories()
-> memory_store.py::list_events_between(user_id, start_at, end_at)
-> SQLite 查询当前用户 active event 记忆
-> 回复层基于召回结果回答
```

再比如：

```text
用户：我之前原话是怎么说的？
```

执行理解：

```text
turn_planner.py 打开 needs_timeline_recall
-> agent_bridge.py::_recall_timeline_chunks()
-> timeline_store.py::search_chunks_with_ranking()
-> SQLite FTS / LIKE 查询 raw timeline chunks
-> 回复层用原文证据回答
```

需要记住的边界：

- 普通聊天、百科问答、建议类问题不应默认读取个人记忆。
- 结构化记忆和 raw timeline 是两层不同数据：前者服务 profile/event/observation，后者服务原文证据和全文回忆。
- Planner baseline / PreReplyDecision 决定“查哪一层”，SQLite 查询函数负责“怎么查”。

### 1.2 Planner 和 PreReplyDecision 的区别是什么？

短答案：Planner 是本地确定性规划器，`PreReplyDecision` 是 `llm_first` 下唯一的 LLM 回复前决策对象。

Planner 不只是短语匹配。它会生成完整 `TurnPlan`，包括：

- 是否需要 profile / event / observation / timeline 召回。
- 是否需要 web search 或 location。
- 是否有 memory write candidates。
- 是否可以用本地 fast path 回复。
- event 召回策略，例如 `temporal_range`、`text_search`、`upcoming_plan`、`observation_review`。

`PreReplyDecision` 不回答用户，只输出结构化 JSON。在当前 `llm_first` 主链路里，它会一次性给出统一决策，例如：

- 这轮是否需要 location。
- 这轮是否需要 web。
- 这是不是 profile recall。
- 这是不是 event recall。
- 这是不是 raw timeline recall。
- `recall_goal` 是 `summary`、`specific_fact` 还是 `raw_evidence`。
- 最终 `reply_mode` 应该是什么。

当前理解：

```text
系统固定走 llm_first；
Planner 只保留强规则入口和 baseline/fallback，PreReplyDecision 是唯一 LLM 回复前决策；
最终是否查 SQLite、是否查 web、是否需要位置，仍由 service 层根据 PreReplyDecision 应用后的 TurnPlan 执行。
```

`llm_first` 下还有几个 LLM 辅助层，但它们不是平级裁判：

- `turn_semantic_classifier.py` 主导 PreReplyDecision，同时覆盖 reply、web/location、recall 和记忆候选字段。
- `temporal_parser.py` 只在 PreReplyDecision/TurnPlan 打开 event recall 后补时间范围。
- `answer_synthesizer.py` 只组织回答表达，不能重新决定是否召回、联网或需要定位。
- `intent_policy.py` 是写入安全门控，只决定候选能不能保存，不改变本轮回复路线。

一句话记忆：

```text
Planner：本地规则调度员，快、稳定、可控。
PreReplyDecision：llm_first 下唯一的 LLM 回复前决策，一次判断 location / web / recall / reply mode / memory candidate。
apply_pre_reply_decision：llm_first 下用 PreReplyDecision 覆盖 baseline 的执行判断。
```

### 1.3 具体从哪里开始召回记忆？召回后会不会进入 LLM prompt？

短答案：Planner baseline / PreReplyDecision 只决定“要不要召回、召回哪类记忆”，真正查询 SQLite 的位置在 `agent_bridge.py::GlassesChatService.chat()` 的 memory retrieval 阶段。召回结果不一定进入主 LLM；如果本地回复已经能完成，就跳过主 LLM。只有需要主 LLM 组织回答时，召回结果才会被 `_message_with_recall()` 包进上下文。

具体链路：

```text
Planner baseline / PreReplyDecision
-> 产出 planner.needs_profile_memory / needs_event_memory / needs_timeline_recall
-> agent_bridge.py::GlassesChatService.chat() 进入 memory_retrieval 阶段
-> 按 planner 打开的 gate 调 memory_store / timeline_store
-> drift_guard 和 recall_arbitration 再过滤、仲裁召回结果
-> 如果 local_reply 能回答，直接返回用户
-> 否则 _message_with_recall() 把召回结果包进 <memory-context>
-> session.agent.run_conversation(agent_message) 调主 LLM
```

关键代码位置：

| 阶段 | 文件 / 函数 | 说明 |
| --- | --- | --- |
| 召回 gate 生效 | `agent_bridge.py::GlassesChatService.chat()` | 只按 planner 打开的门读取相关记忆，避免每轮把所有长期记忆塞给模型。 |
| profile 召回 | `memory_store.list_memories(user_id, kind="profile")` | 读取当前用户 active profile 记忆，再按 strength 排序。 |
| event 召回 | `agent_bridge.py::_recall_event_memories()` | 根据策略走时间范围、未来计划、observation review 或文本搜索。 |
| timeline 召回 | `agent_bridge.py::_recall_timeline_chunks()` | 查 raw timeline chunks，用于原文和证据召回。 |
| 上下文注入 | `agent_bridge.py::_message_with_recall()` | 把结构化记忆、timeline、document、location、web 等包装成受控上下文。 |
| 主 LLM 调用 | `session.agent.run_conversation(agent_message)` | 只有本地回复没有完成时才调用。 |

例子 1：用户问个人偏好。

```text
用户：我喜欢什么座位？
```

可能流程：

```text
Planner 判断 needs_profile_memory = true
-> agent_bridge.py 调 list_memories(kind="profile")
-> 查到“用户喜欢安静靠窗的位置”
-> 如果本地 profile reply 足够，就本地回答
-> 如果需要主 LLM 组织表达，_message_with_recall() 把 profile 放进 <memory-context>
```

例子 2：用户问复杂总结。

```text
用户：你从之前我跟你聊过的内容看，我这段时间的重心是什么？
```

可能流程：

```text
Planner 本地规则不一定覆盖完整语义
-> PreReplyDecision 用 LLM 补充 memory_recall_type=observation / recall_goal=summary
-> agent_bridge.py 查 observation 和相关 source memories
-> _message_with_recall() 把 observation 放进 <memory-context>
-> 主 LLM 基于上下文组织总结
```

需要记住的边界：

- Planner baseline / PreReplyDecision 不直接查 SQLite。
- `memory_store` / `timeline_store` 才是真正的数据查询层。
- 召回结果不是无条件塞进 prompt；本地回复能完成时会跳过主 LLM。
- 进入主 LLM 时，召回内容被包在 `<memory-context>`，并标注“不是新的用户输入”，避免被误解成新指令。

### 1.4 当前最小代码地图是什么？

短答案：先记住 8 个文件，基本就能定位聊天、召回、写入、存储和安全边界。

| 文件 | 一句话职责 |
| --- | --- |
| `agent_bridge.py` | 主调度中心。聊天、召回、回复、后台写入、debug/audit 都在这里串起来。 |
| `turn_planner.py` | 本地 Planner。判断是否读记忆、写候选、查 web/location、走本地回复。 |
| `turn_semantic_classifier.py` | LLM 回复前决策。输出 `PreReplyDecision`，统一补充 Planner baseline 漏掉的召回、web/location、reply mode 和记忆候选。 |
| `intent_policy.py` | 写入门控。判断长期记忆候选能不能保存。 |
| `memory_store.py` | 结构化记忆存储。保存和搜索 profile/event/assistant_preference/observation。 |
| `timeline_store.py` | 原文时间线存储。保存和搜索 raw turn / timeline chunk。 |
| `memory_recall_arbitration.py` | 召回仲裁。决定 document、timeline、event、profile、observation 谁作为主证据。 |
| `privacy_filter.py` | 敏感信息脱敏。保护 raw timeline 入库前的密码、token、证件号等。 |

按阶段记忆：

```text
入口和编排：agent_bridge.py
判断：turn_planner.py / turn_semantic_classifier.py
写入门控：intent_policy.py
结构化记忆：memory_store.py
原文证据：timeline_store.py
召回仲裁：memory_recall_arbitration.py
隐私脱敏：privacy_filter.py
```

### 1.5 为什么查到记忆后不一定会用？

短答案：召回只是找到候选证据，不代表这些证据一定适合当前回答。召回后还要检查当前 query 相关性、当前意图是否冲突，以及多个证据来源谁优先。

核心判断点：

```text
drift_guard
recall_arbitration
当前 query 相关性
当前意图是否冲突
证据优先级
```

可以把流程理解为：

```text
Planner baseline / PreReplyDecision 打开召回门
-> memory_store / timeline_store 查出候选证据
-> drift_guard 过滤不相关或和当前意图冲突的记忆
-> recall_arbitration 决定哪个来源作为主证据
-> 最终才进入本地回复或 <memory-context>
```

`drift_guard` 主要防止“历史记忆漂移”。比如：

```text
旧 profile：用户喜欢靠窗座位
当前 query：这次订座不要靠窗，我应该优先考虑什么？
```

错误行为：

```text
因为查到“喜欢靠窗”，所以继续推荐靠窗。
```

正确行为：

```text
当前 query 明确说“不要靠窗”，历史偏好不能压过当前意图。
旧靠窗偏好即使被查到，也应该被过滤或弱化。
```

`recall_arbitration` 主要处理“多个来源同时召回时谁优先”。比如：

```text
用户问：这份攻略里红旗渠门票多少钱？
```

如果同时存在：

```text
profile：用户喜欢自驾游
document：攻略原文里写了门票价格
timeline：用户之前提过红旗渠
```

这时主证据应该是 document 原文，因为用户问的是文档细节，不是用户画像或原话回忆。

一句话记忆：

```text
查到了不等于要用；
要先看当前问题问的是什么、当前意图有没有和旧记忆冲突、哪个来源最能回答问题。
```

典型优先级直觉：

| 用户问题 | 更应该优先的证据 |
| --- | --- |
| “我喜欢什么座位？” | profile 画像 |
| “我明天有什么安排？” | event 时间范围召回 |
| “我之前原话怎么说？” | timeline raw chunk |
| “这份文档里门票多少钱？” | document 原文 |
| “我最近主要在忙什么？” | observation / 多个 event 总结 |
| “这次不要靠窗” | 当前 query 意图优先于旧 profile |

## 2. Observation、Chunk 和长输入

### 2.1 后台 observation 自动生成的具体触发逻辑是什么？

短答案：observation 不是每轮都生成，也不是用户一说“总结”就立刻写入。它由 `agent_bridge.py::_maybe_start_observation_reflect()` 在记忆写入后尝试触发，只读取当前用户已经保存成功、仍为 active、带 evidence 的 profile/event。

关键常量：

```text
OBSERVATION_REFLECT_MIN_SOURCE_MEMORIES = 3
OBSERVATION_REFLECT_SOURCE_LIMIT = 12
OBSERVATION_REFLECT_MIN_INTERVAL_SECONDS = 60 * 60
```

触发条件：

```text
当前用户 active profile/event source memories 数量 >= 3
-> 这些 source memory 必须带 evidence_ids
-> 如果已有 observation，距离最近一次 observation 更新时间至少 1 小时
-> 如果当前 evidence_ids 已经完全包含在最近 observation 里，不重复生成
-> 创建 mode="observation_reflect" 的后台 memory job
```

source memory 的筛选逻辑：

```text
memory.kind in {"profile", "event"}
memory.memory_type != "observation"
memory.status == "active"
memory.evidence_ids 非空
```

生成方式：

```text
优先：用内部 LLM reflection summarizer 基于 source memories 生成一句保守 observation
兜底：用本地规则拼接最近事件线索和稳定画像
保存：kind="event", memory_type="observation", source="observation_reflect"
```

observation 的使用方式：

```text
用户问“我最近在忙什么 / 项目现在什么状态 / 我的工程偏好是什么”
-> turn_planner.py 命中 observation recall
-> needs_observation_memory = true
-> event_recall_strategy = "observation_review"
-> agent_bridge.py::_recall_event_memories() 召回同 scope 的 active observation 和 source memories
```

一句话记忆：

```text
observation 的生成靠后台 reflect；
observation 的使用靠用户问总结/回顾类问题时召回；
它是 profile/event 的长期归纳，不是原始事实替代品。
```

### 2.2 chunk 到底是什么？怎么切分？主要用途是不是原文召回？

短答案：chunk 是 timeline 里的可搜索原文片段，主要用于原文召回、evidence 追溯和删除审计。它不是结构化长期记忆。

`timeline_store.py::TimelineChunk` 的核心字段：

```text
id
user_id
parent_type
parent_id
text
chunk_index
start_offset / end_offset
timestamp
source
metadata
status
```

来源：

```text
/api/chat 用户原话
-> TimelineStore.add_turn()
-> parent_type="turn" 的 chunks

/api/capture/append 连续输入片段
-> TimelineStore.add_capture_chunk()
-> parent_type="capture" 的 chunks
```

是否每输入一句话都会切 chunk：

```text
只要 /api/chat 正常进入主链路，系统会先调用 TimelineStore.add_turn() 保存用户原话；
add_turn() 会调用 _chunk_text()，因此这轮用户原话会至少形成 1 个 parent_type="turn" 的 chunk；
如果这句话很短，通常就是 1 个 chunk；
如果很长，才会按 max_chars=900 左右切成多个 chunk。
```

例外和边界：

```text
如果请求在进入 chat 主链路前就被校验拒绝，就不会形成正常 timeline turn/chunk。
chunk 只是原文证据，不等于这句话一定会保存成长期 structured memory。
```

切分逻辑在 `timeline_store.py::_chunk_text(text, max_chars=900)`：

```text
1. 先清理空白。
2. 用 _split_text_units() 把文本拆成较小语义单元。
3. 按顺序合并单元，当前 chunk 超过 900 字符就切出一个 chunk。
4. 记录 start_offset / end_offset，保留原文位置。
5. 如果仍有超大 chunk，再用 _split_oversized_chunks() 继续切。
```

搜索逻辑：

```text
优先 chunks_fts 全文搜索；
FTS 不可用或中文短词不适合时，fallback 到 LIKE；
LIKE 会从 query 中提取有用词，中文长词还会拆 4 字/3 字片段；
排序会综合 text_score 和 recency_score。
```

主要用途：

```text
1. 原文召回：用户问“我之前原话怎么说？”
2. evidence：结构化 memory 的 evidence_ids 指向 chunk
3. 删除审计：删除 memory 时可以同步清理未共享 evidence chunk
```

一句话记忆：

```text
chunk 是原文证据颗粒；
profile/event/observation 是结构化记忆；
chunk 主要服务原文召回和 evidence，不替代结构化记忆。
```

### 2.3 如何区分连续长输入？

短答案：`turn_planner.py::_long_input_signal()` 不是只看长度，而是组合判断“这是不是个人经历、会议纪要、项目复盘或转写文本”。

基础门槛：

```text
文本长度 >= 90
句子数或行数 >= 3
不是长问题 / 生成请求
```

主题信号：

```text
project_hits：项目、进展、风险、卡点等
daily_hits：日常经历、去了、见了等
action_hits：负责、截止、计划、启动、交付等
temporal_hits：今天、昨天、下周、周几等
person_hits：英文人名数量
first_person_hits：我 / 我们
```

命中条件大意：

```text
topic_score >= 3
并且 action/project/daily/person/first_person 等信号足够
并且至少有第一人称或项目线索
```

命中后：

```text
reply_mode = "continuous_capture"
fast_path_kind = "continuous_capture"
先低打扰回复“收到，我先整理”
后台分段抽取高价值候选
```

不会命中的典型情况：

```text
“请详细解释 Transformer 注意力机制……”这种长知识问题。
“帮我写一篇……”这种长生成请求。
```

### 2.4 capture 和 chunk 是什么关系？

短答案：capture 是一次连续输入会话；chunk 是这次会话里的具体文本片段。

关系：

```text
captures 表：保存 capture_id、context、started_at、status 等元数据
chunks 表：保存每次 append 进来的文本片段
```

对应父级：

```text
/api/chat 用户原话 -> parent_type="turn"
/api/capture/append -> parent_type="capture"
```

所以：

```text
capture 本身不是 chunk；
capture append 的文本会写成 timeline chunk；
stop_capture() 时再汇总 chunks 文本进入 import / 候选 / 门控 / 写入流程。
```

### 2.5 长输入里的闲聊如何判断？是否应该让 LLM 凝练去除闲话？

当前实现分两层。

第一层：普通低价值闲聊 fast path。

```text
turn_planner.py::_is_low_value_chitchat_statement()
```

它主要通过短语判断，例如：

```text
累、困、疲惫、辛苦、人好多、太挤、堵、差点睡着、站着睡着
```

命中后通常不沉淀长期记忆。

第二层：长输入分段后抽高价值候选。

```text
agent_bridge.py::_segment_long_input()
agent_bridge.py::_long_input_rule_candidates()
```

长输入不会把整段都当记忆，而是分段后找 marker：

```text
preference：喜欢、不喜欢、偏好、习惯
task：负责、截止、待办、计划、交付
decision：决定、结论、确定、先做
project_state：风险、卡点、失败、不稳定、进展
event：今天、昨天、下周、去了、见了
```

当前边界：

```text
闲聊判断确实偏规则化、偏保守；
低价值片段通常不进 structured memory；
但 raw timeline chunk 仍可保留，用于原文检索和证据追溯。
```

是否应该用 LLM 凝练去除闲话？

```text
方向上合理，尤其适合真实 ASR / 长会议 / 走路口述，因为闲话、噪声、口头禅、重复和误听很多。
但不能让 LLM 直接自由改写并保存，必须仍然输出结构化候选，再走 should_write_memory_candidate()、敏感门控、去重/冲突和 evidence 追溯。
```

更稳妥的改进形态：

```text
长输入 raw chunk
-> LLM 分段抽取候选，去除闲话/口头禅/重复
-> 输出 JSON candidates：content、kind、memory_type、confidence、reason、source span/chunk id
-> 本地门控 should_write_memory_candidate()
-> 去重 / conflict / supersede
-> 保存 structured memory，并保留 evidence_ids 指向原始 chunk
```

一句话判断：

```text
用 LLM 去除闲话是合理优化；
但 LLM 只能做候选提炼，不能绕过门控直接写长期记忆。
```
