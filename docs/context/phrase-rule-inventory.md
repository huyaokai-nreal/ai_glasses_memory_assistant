# 短语匹配规则盘点

本文对应 `PLANS.md` 中“短语规则治理清单”的专题落点。目标不是把所有字符串、关键词或正则一次性删掉，而是先给规则定角色：哪些可以做硬安全边界，哪些只能做弱信号，哪些是旧 fallback，避免每遇到一种自然说法就继续补短语。

## 分类标准

| 分类 | 允许动作 | 典型例子 | 治理原则 |
| --- | --- | --- | --- |
| `hard_safety` | 可以硬拒绝、脱敏或要求确认 | API key、token、密码、验证码、证件号、银行卡号 | 保留，尽量集中在 `privacy_filter.py` 和写入安全 policy。 |
| `format_parser` | 可以确定性解析封闭格式 | URL、文件扩展名、Markdown 标题、JSON/schema、状态 tag | 保留，但不能参与开放语义裁决。 |
| `weak_signal` | 只能作为 debug/prompt/context 信号 | `啥/什么/吗/刚才/这份/项目/喜欢/提醒/周报` | 不应单独 veto LLM 结构化候选。 |
| `retrieval_hint` | 只能扩大候选召回或排序 | 文档标题词、timeline 查询词、profile topic 词 | 不直接决定最终回答或写入。 |
| `legacy_fallback` | 只在 LLM 不可用或低风险兜底时启用 | `extract_memory_candidates()` 旧规则抽取、长输入规则候选 | 必须有 debug/audit 标记，不能静默覆盖主路径。 |

## 当前规则清单

| 位置 | 规则类型 | 当前动作 | 风险 | 分类 | 治理建议 |
| --- | --- | --- | --- | --- | --- |
| `privacy_filter.py::_REDACTION_PATTERNS` | token、JWT、验证码、银行卡、身份证等正则 | timeline/文本清洗时脱敏 | 低 | `hard_safety` | 保留硬规则；后续只补安全类 pattern，不补开放语义。 |
| `intent_policy.py::_sensitive_reason()` | 密钥、验证码、长数字、证件等敏感候选检测 | 拒绝或要求确认长期记忆写入 | 低 | `hard_safety` | 保留；候选写入门控已暴露 `safety_policy.role=hard_safety`，与 `privacy_filter.py` 的覆盖范围可后续合并或去重。 |
| `intent_policy.py::_looks_like_question()` | `什么/啥/吗/怎么/...` 问句 marker | 粗略问句信号 | 中 | `weak_signal` | 不再直接拦截候选文本；实际写入 veto 使用直接问题形态判断，避免候选里有口语 filler `啥` 就被拒绝。 |
| `intent_policy.py::_looks_like_direct_source_question()` | 直接问句形态和最终子句问句后缀 | 仅当候选正文像问题，或原始 message 像直接问题且没有记忆请求时，拒绝写入 | 中 | `weak_signal` | 已补 `question_policy`：拒绝时暴露 `source/treatment/overrides_llm=false`；这是候选级 question guard，不再作为全局意图判断器。 |
| `turn_planner.py::_is_question_like()` | 问句 marker + `is_question()` | 召回/query 判断和部分 fast path hint | 高 | `weak_signal` | 已从本地写入候选硬跳过里拆出：写入候选/fast path 使用直接问题 guard，`啥/什么/吗` 这类词不再单独 veto event 候选。 |
| `turn_planner.py::_memory_write_candidates()` / `_strip_memory_prefix()` | 问句不生成规则候选、记忆命令、画像/事件短语 | 生成或跳过本地候选 | 高 | `legacy_fallback` | 已先支持剥掉“你能帮我记一下…吗”这类记忆请求外壳，并把直接问题 guard 与候选级事件清洗分开；后续规则候选仍只能做低风险 fallback。 |
| `turn_planner.py` 的 recall/topic 判断 | `最近/接下来/未来/待办/提醒/安排/周报/原话/喜欢/推荐/风险/卡点/...` | 决定 `recall_goal`、profile/event/timeline 召回 | 高 | `weak_signal` / `retrieval_hint` | 保留召回 hint，但最终 evidence 选择应依赖 router/relevance 与 arbitration debug；`周报`、`风险/卡点`、`原话` 已从单词级 workflow trigger 收窄为生成/本周进展、当前项目注意事项或历史原文证据语境；`喜欢/推荐/偏好` 的 profile gate 已收窄为个人决策语境；`提醒/安排/待办/接下来/未来/近期` 概念问题已排除 event/upcoming recall。 |
| `agent_bridge.py::_recall_documents_for_query()` 及文档 helper | `文档/文件/上传/周报/这份/刚刚那份` 等 | 路由文档召回、选择 metadata/full document | 高 | `weak_signal` / `retrieval_hint` | 已先收紧类型词：`周报/日报/会议纪要/攻略` 需要和 `上一份/这份/里面/刚上传` 等文档语境同现才触发文档召回；普通生成/总结请求不因类型词查旧文档。`debug.document_recall.phrase_policy` 会暴露类型词/指代词是被忽略的 weak signal 还是 retrieval hint。 |
| `agent_bridge.py::_rule_correction_content()` / `_correction_kind_type_for_text()` | `说错了/不对/更准确地说/...` 以及 correction 内容里的 `喜欢/待办/项目/...` 类型词 | 生成明确纠错候选，LLM correction 之前执行 | 高 | `weak_signal` / `legacy_fallback` | 已先移除裸 `不是 X，而是 Y`、隐式 `其实我现在主要在...`、`改一下...` 和“前面/刚才/那个 + 更新/改”类本地强纠错资格；rule correction 的本地类型分类已改为可观测 fallback，未命中类型语境时默认 `event` 而不是静默 `project_state`。 |
| `agent_bridge.py::_correction_target_hints()` / `_local_correction_target_ids()` | `那个/前面/之前/刚才/座位/偏好/任务/项目/...` | 纠错写入后定位要 supersede 的旧记忆 | 高 | `weak_signal` / `retrieval_hint` | 已补 `correction_target_resolution.target_hint_policy`：具体目标词只作为候选级 target 打分 hint；不具体的“那个信息更新一下”标为 `weak_signal` 并交给 LLM target resolver，不能静默用本地短语锁旧记忆。 |
| `agent_bridge.py::_looks_like_question()` | `什么/啥/吗/如何/...` | LLM correction 结果二次拦截 | 中 | `weak_signal` | 保留候选级 question-text 拦截；不要扩展成原句级 veto。 |
| `agent_bridge.py::_long_input_rule_candidates()` | 长输入中 `喜欢/负责/决定/项目/今天/...` | 长输入无结构化候选时兜底生成候选 | 高 | `legacy_fallback` | 已改为有 agent 时优先 semantic cleaner + LLM，规则只在 LLM 无候选或无 agent 时 fallback；事件类规则需时间 + 具体动作，避免只靠时间词污染记忆。 |
| `agent_bridge.py::_local_reply_for_plan()` / `_event_reply_prefix()` | 本地回复模式、空证据 guard、`吃/运动/提醒` 回复前缀 | 跳过主 LLM 或组织本地短答 | 中 | `retrieval_guard` / `retrieval_hint` / `reply_style_only` | 已新增 `debug.local_reply_policy` 暴露本地回复 role/reason/phrase_match_role；`吃/运动/提醒` 只影响前缀文案，标记为 `reply_style_only`，不能决定召回或证据。 |
| `agent_bridge.py::_observation_scope_for_query()` / `_observation_source_fallback_memories()` | `工程偏好/项目状态/最近主要/...` observation scope 词 | 窄化 observation review，必要时回退 source memories | 高 | `retrieval_hint` / `legacy_fallback` | 已新增 `debug.memory.event_recall.observation_scope_policy` 和 `source_fallback_policy`：scope 词只作为 retrieval narrowing；没有同 scope observation 时 source memory 回退必须显式标记 `legacy_fallback`。 |
| `agent_bridge.py::_observation_scope_for_content()` | `项目状态/风险/最近主要/工程偏好/...` | 给 observation 内容打 scope tag，辅助同 scope 更新/召回 | 中 | `retrieval_hint` | 已补 `observation_update_decisions[].observation_scope_policy`：内容短语只影响 observation scope tag，不改写候选 `kind/memory_type`，并暴露 `affects_memory_type=false`。 |
| `agent_bridge.py::_filter_event_memories_for_query()` | `工作/任务/待办/项目/未完成/进展/卡点/风险` | event recall 后过滤非工作类事件 | 中 | `retrieval_narrowing` | 已新增 `debug.memory.event_recall.filter_policy` 暴露 markers、输入/输出数量和 filtered_count；这些词只用于召回窄化，不能静默丢弃而无 debug。 |
| `agent_bridge.py::_task_status_for_content()` | `完成/取消/不做了/done/cancelled` | task 候选保存后的状态 tag | 中 | `format_parser` | 已补 `task_status_policies`：这些词只在候选已是 `memory_type=task` 后解析 `task_status:*`，并暴露 `affects_memory_type=false`、`overrides_llm=false`，不能决定写入或二级分类。 |
| `agent_bridge.py::_profile_query_scope()` / `_filter_profile_memories()` | `喜欢/偏好/咖啡/餐厅/座位/...` | profile 召回过滤和本地回复 | 中 | `retrieval_hint` | 已把通用偏好查询改为信任 `memory_type=preference`，topic 词只做窄化；drift guard 继续挡明显无关问题，并在 `profile_topic_policy` 中暴露 topic mismatch。 |
| `memory_store.py::classify_memory_type()` | `喜欢/决定/任务/提醒/风险/卡点/进展/状态/...` | 导入或 fallback 候选分类 | 中 | `legacy_fallback` | 结构化来源应显式给 `memory_type`；`add_memory()` 和候选保存路径已不再用正文短语表补默认 `memory_type`，导入/legacy fallback 命中时需通过 `classification_decisions` 或 candidate reason 暴露。裸 `项目/project` 已不再触发 `project_state`；`喜欢/提醒` 概念问题不再触发 `preference/task`，只有个人偏好或提醒动作语境才可作为 fallback。 |
| `memory_store.py::extract_memory_candidates()` | 旧触发词 + 问句跳过 | LLM classifier 不可用时抽取候选 | 高 | `legacy_fallback` | 已先允许显式问句形式记忆请求剥壳后抽取，并给候选标记 `source=rule_fallback`、`reason=legacy_memory_extractor`、`classification`；普通问题仍保守跳过。 |
| `memory_recall_arbitration.py::_summary_profile_scope()` | `最近/刚才` 标记 recent activity，`关于我/我是谁` 标记 profile background | summary 召回前决定 profile 是否降级 | 中 | `weak_signal` | 已改为短语只决定 scope 信号；只有同时存在 event/timeline/document 动态证据时才 drop stable profile，否则保留并在 `summary_profile_policy` 中暴露裁决。 |
| `timeline_store.py::_LOW_INFORMATION_TERMS` | `什么/怎么/相关/提到/...` | timeline 查询词过滤 | 低 | `retrieval_hint` | 保留，属于搜索质量过滤，不参与写入裁决。 |
| `text_cleaning.py` marker patterns | filler、噪声、纠错、否定、do-not-remember | 分段、debug、LLM segment 输入信号 | 中 | `weak_signal` | 保持为 marker/debug；不要直接整段丢弃，除非 LLM 或候选级 policy 支持。 |
| `segment_semantic_cleaner.py::fallback_segment_semantic_decision()` | marker-only fallback | LLM segment cleaner 失败时跳过噪声/别记段 | 中 | `legacy_fallback` | fallback 必须显式 `backend=rule_fallback`；命中 filler 不应默认跳过含事实片段。 |
| `temporal_parser.py` prompt examples | `最近要做什么/接下来有什么/...` | 指导 LLM temporal parser 解析 future plan | 低 | `weak_signal` | 这是 prompt 示例，不是生产硬规则；保留但不要在代码层重复扩词表。 |

## 已开始治理

- `intent_policy.py::should_write_memory_candidate()` 已把“原始 message 含问句 marker -> 非 profile 候选硬拒绝”改成候选级策略：候选文本本身像问题仍拒绝；原始 message 只有在像直接问题且没有记忆请求时才拒绝写入。
- `intent_policy.py::should_write_memory_candidate()` 已补 `question_policy`：`candidate_is_question_text` 和 `source_question_without_memory_request` 不再只是黑盒 reason，而会暴露 `role=weak_signal`、命中来源、处理方式和 `overrides_llm=false`。
- `turn_planner.py::_strip_memory_prefix()` 已把问句形式的记忆请求外壳降级为解析步骤：`你能帮我记一下明天下午3点和 Mina 开会吗？` 的本地候选会变成 `明天下午3点和 Mina 开会`，避免把整句问句送进写入门控。
- `turn_planner.py::_memory_request_content()` 已补 mixed recall + write 截取：`我喜欢什么咖啡？另外帮我记一下我喜欢低糖拿铁` 会保留 profile recall，同时只把 `用户喜欢低糖拿铁` 作为 `profile/preference` 写入候选，前半句问题不会进入候选正文。
- `turn_planner.py::_memory_write_candidates()` 已把“问题词 -> 不生成候选”的硬跳过收窄成直接问题 guard，并为本地 event 候选做轻量口语片段清洗：`我忘记要说啥了...好像要和 Mina 开会` 会抽出含 `Mina 开会` 的事件候选，而 `今天三点和 Mina 开会是啥意思` 仍不会生成候选。
- `agent_bridge.py::_corrected_memory_content()` / `_rule_correction_content()` 已把裸 `不是 X，而是 Y`、隐式 `其实我现在主要在...`、`改一下...` 和“前面/之前/刚才/那个 + 更新/改”从本地强纠错规则中移除；普通对比句、新当前状态、编辑命令或进度文档更新不会再仅凭短语生成 correction 候选，显式 `纠正一下/说错了/之前说的不对/更准确地说` 仍保留，开放语义纠错交给 LLM detector。
- `agent_bridge.py::_correction_kind_type_for_text()` 已把 rule correction 的类型裁决暴露为 `debug.correction_detection.classification_decisions`：明确项目/偏好/任务/决定语境仍可命中 `legacy_phrase_classifier`，没有类型语境的纠错默认 `event`，例如“纠正一下，我今天下午3点是和 Mina 开会，不是和 Bob”不会再默认为 `project_state`。
- `agent_bridge.py::_correction_target_hints()` 已补 `target_hint_policy`：`前面那个座位偏好更新一下` 里的 `座位/偏好` 是 `retrieval_hint`，只用于旧记忆候选打分；`那个信息更新一下` 这类不具体指代是 `weak_signal`，需要 LLM target resolver，不走本地强 supersede。
- correction 候选写入时不再用 temporal parser 的 `normalized_text` 覆盖正文；时间解析仍可提供 metadata，但不能改写 LLM detector 已确认的 `corrected_content`。
- `memory_store.py::extract_memory_candidates()` 这个 legacy fallback 也已同步处理问句形式记忆请求：`你能帮我记一下明天下午3点和 Mina 开会吗？` 会抽成干净 event 候选，而 `明天下午3点和 Mina 开会是啥意思？` 仍跳过。
- `memory_store.py::classify_memory_kind_with_reason()` / `classify_memory_type_with_reason()` 已把导入和旧 fallback 的短语分类来源显性化：结构化导入显式给 `memory_type` 时不再标记 legacy 词表；未给 type 时，`classification_decisions` 会在 import result/audit 中暴露是 `legacy_phrase_classifier` 还是 `legacy_default`。
- `memory_store.py::classify_memory_kind_with_reason()` 已把身份问句从 profile fallback 中排除：`我是什么身份` 不再因为 `我是` 触发 `profile_trigger`，会暴露 `question_text_no_kind_marker` 并交给写入门控拒绝；`我是 XREAL 的员工` 仍可作为明确 profile fallback。
- `memory_store.py::add_memory()` 和 `agent_bridge.py::_candidate_memory_type()` 已把默认 `memory_type` 从正文短语表迁到结构化 `kind`：`kind=event` 缺省保存为 `event`，`kind=profile/assistant_preference` 缺省保存为 `preference`。例如 `AI 眼镜项目继续补 debug 可观测性` 作为 event 候选保存时不会只因 `项目` 被静默归成 `project_state`；如果确实是项目状态，需由 LLM/导入结构显式给出 `memory_type=project_state`。
- `memory_store.py::classify_memory_type_with_reason()` 的 legacy `project_state` 词表已继续收窄：裸 `项目/project` 不再让导入缺省条目升为 `project_state`；`风险/卡点/进展/状态/延期` 这类更具体状态词仍保留为可观测 fallback。
- `memory_store.py::classify_memory_type_with_reason()` 的 `喜欢/提醒` legacy 词表已继续收窄：`喜欢是什么意思`、`提醒是什么意思` 这类概念问题默认 `event` 并暴露 `question_text_no_type_marker`；`我喜欢低糖拿铁`、`提醒我明天下午3点和 Mina 开会` 这类有个人偏好或提醒动作语境的输入仍可命中可观测 fallback。
- `intent_policy.py::_sensitive_reason()` 和敏感凭据 fast path 已补 `safety_policy`：API key、token、密码、验证码、银行卡/证件号等封闭安全规则仍可硬拒绝，但会标明 `role=hard_safety`、`treatment` 和是否覆盖最终裁决。
- `agent_bridge.py::_is_document_query()` 已把文档类型词降级为 retrieval hint：`帮我写一份周报` 不会因为 `周报` 查旧文档；`上一份周报里有什么风险？` 仍会按最近文档类型召回。
- `agent_bridge.py::_recall_documents_for_query()` 已补 `debug.document_recall.phrase_policy`：文档类型词、最近文档指代词和显式文档词都会暴露 role/treatment；`帮我写一份周报` 中的 `周报` 标为 `weak_signal` 且不加载文档，`上一份周报...` 标为 `retrieval_hint`。
- `turn_planner.py::_is_weekly_summary_query()` 已把 `周报` 从单词级 workflow trigger 收窄为生成/进展语境 trigger：`帮我写一份周报`、`这周进展如何` 仍可进入 weekly report；`周报是什么？`、`周报怎么写？` 回到普通 LLM 问答，不会直接查项目记忆生成草稿。
- `turn_planner.py::_is_attention_items_query()` / `_is_observation_recall_query()` 已把 `风险/卡点/项目风险` 从单词级 workflow trigger 收窄为当前事项语境：`我接下来有什么要注意的？`、`这个项目有什么风险要注意？` 仍可召回任务和项目风险；`风险是什么？`、`项目风险是什么？`、`卡点是什么意思？` 回到普通 LLM 问答。
- `turn_planner.py::_is_timeline_recall_query()` 已把 `原话` 从单词级 raw timeline trigger 收窄为历史原文证据语境：`我之前有没有说过语音识别不稳定的原话？` 仍可查 raw timeline；`原话是什么意思？`、`怎么写原话？`、`原话格式是什么？` 回到普通 LLM 问答。
- `turn_planner.py::_profile_context_would_help()` 已把 `喜欢/推荐/偏好` 从裸词 profile gate 收窄为个人决策语境：`推荐算法是什么？` 不会读取 profile；`以后推荐餐厅时要避开什么？`、`我喜欢喝什么？` 仍可使用个人偏好。
- `turn_planner.py::_is_event_query()` / `_is_upcoming_plan_query()` 已把 `提醒/安排/待办/接下来/未来/近期` 概念问题从本地 event/upcoming recall 中排除：`提醒是什么意思？`、`安排是什么意思？`、`待办是什么意思？`、`接下来是什么意思？` 回普通 LLM；`我有什么待办事项`、`最近有什么安排` 仍可查个人未来计划。
- `memory_recall_arbitration.py` 已把 summary 场景里的 `最近/刚才` 类短语从 profile 硬丢弃降级为 scope 信号：有 event/timeline/document 动态证据时，稳定 profile 会被压低；没有动态证据时，profile 不会只因为“最近”这个词被清空，debug 会写出 `summary_profile_policy`。
- `agent_bridge.py::_local_reply_for_plan()` 已补 `debug.local_reply_policy`：空证据保护会标记为 `retrieval_guard`，事件召回本地短答会标记为 `retrieval_hint`，`_event_reply_prefix()` 里的 `吃/运动/提醒` 只作为 `reply_style_only` 的前缀策略暴露，不参与召回裁决。
- `agent_bridge.py::_observation_scope_for_query()` 已补 `observation_scope_policy`：`工程偏好/项目状态/最近主要` 等词只标记为 `retrieval_hint`，用于窄化 observation review；`_observation_source_fallback_memories()` 命中时会在 `source_fallback_policy` 中标记 `legacy_fallback`，说明这是无同 scope observation 时的 source memory 回退。
- `agent_bridge.py::_observation_scope_for_content()` 已补 observation update 里的 `observation_scope_policy`：`项目状态/风险/最近主要/工程偏好` 等词只给 observation 打 scope tag，用于同 scope 更新或召回，不影响候选 `kind/memory_type`。
- `agent_bridge.py::_filter_event_memories_for_query()` 已补 `filter_policy`：`工作/任务/项目/风险` 等词只标记为 `retrieval_narrowing`，并暴露被过滤前后的数量，例如“我最近有什么工作”会说明生活事件被过滤，而不是静默丢弃。
- `agent_bridge.py::_task_status_for_content()` 已补 `task_status_policies`：`完成/取消/不做了` 只在候选已经是 task 时解析成 `task_status:*` tag；例如 `取消周五复盘会` 不会因为 `取消` 决定写入，只会在 LLM/planner 已给出 task 后标为 `cancelled`。
- `agent_bridge.py` 的 profile topic 过滤已收窄职责：`咖啡/座位/餐厅` 等词仍用于具体偏好问题的召回 narrowing；但“我有什么偏好 / 我的判断偏好是什么”这类通用画像问题会信任 `memory_type=preference`，不会因为内容没命中特定 topic 词就说没有记忆。
- `agent_bridge.py::_long_input_rule_candidates()` 已降级为真正 fallback：有 agent 时不再因为规则候选数量够多而绕过 semantic cleaner + LLM；规则候选数量会写入 `extraction_trace.rule_fallback`，事件类规则必须同时有时间和动作。
- 这能覆盖 audit 中的误拦截族：`我忘记要说啥了，想起来了，今天三点和 Mina 开会` 这类口语句不会再因为 `啥` 误杀 event 候选，并且保存正文不会带入“忘记要说啥”这类 filler。
- 真问题仍不保存：`今天三点和 Mina 开会是啥意思？` 或无问号的 `今天三点和 Mina 开会是啥意思` 仍会被拒绝。

## 治理结果与后续观察

第一轮执行结果与 `PLANS.md` 的 P1.11-A 到 P1.11-G 对齐。后续如果真实 audit 暴露新问题，不要直接从 `rg` 命中删代码，也不要为单句 query 扩词表；先判断这条规则当前是否在 block/reject/route/drop/覆盖 LLM，再把它归入安全/格式/hint/fallback/policy 边界。

| 任务 | 优先级 | 当前重点 | 不做什么 |
| --- | --- | --- | --- |
| P1.11-A 写入门控收口 | 高 | 继续确认 `intent_policy.py`、`turn_planner.py` 和 correction candidate gate 只看候选级事实，不用原始 message 问句词粗暴 veto | 不再新增 `啥/什么/吗` 的例外词补丁。 |
| P1.11-B Planner 召回触发降级 | 高 | 继续把 `周报/风险/原话/推荐/提醒/未来` 等自然词从 workflow trigger 降为语境信号或召回 hint | 不让单个概念词决定用户是在问个人记忆、项目状态还是通用知识。 |
| P1.11-C `agent_bridge.py` 本地语义规则清理 | 高 | 第一轮已覆盖文档指代、纠错、correction target hint、长输入 fallback、profile topic、本地回复前缀、source/observation fallback、task 状态 tag；后续观察真实 audit | 不让本地短语规则覆盖 LLM detector/router 的结构化输出。 |
| P1.11-D `memory_store.py` legacy 分类治理 | 高 | 第一轮已覆盖 `喜欢/提醒/项目/风险/身份问句` 等高风险 type/kind marker；后续观察真实导入语料 | 不用正文词表静默改写结构化 `kind/memory_type`。 |
| P1.11-E 安全/格式规则集中保留 | 中 | 保留 token、密码、证件号、URL、JSON/schema 等封闭规则，并尽量集中在安全/格式模块；敏感写入拒绝已暴露 `safety_policy.role=hard_safety` | 不把开放语义词混进 safety 或 parser 层。 |
| P1.11-F 可观测性与 audit 对齐 | 高 | 第一轮已补 `question_policy`、`safety_policy`、`target_hint_policy`、`task_status_policies`、`observation_scope_policy`、`classification_decisions` 和 fallback trace；后续抽样真实 audit | 不接受“规则命中了但 audit 看不出来它有没有影响最终结果”。 |
| P1.11-G 真实语境压力测试 | 高 | 第一轮已覆盖问句词、文档类型词、纠错表达、概念问题、future/upcoming、长输入和可观测字段；后续继续补真实失败 audit replay 和混合 recall+write 长周期样本 | 不为了某一句话继续扩散短语表。 |
