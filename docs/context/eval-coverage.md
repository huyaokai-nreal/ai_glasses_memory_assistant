# Eval 覆盖矩阵

更新时间：2026-06-15。本文基于当前 `evals/scenarios.jsonl`、`evals/runner.py`、`evals/metrics.py` 和最近 `reports/eval-latest.md` 整理，用来判断离线 eval 是否覆盖 AI 眼镜真实使用场景。

## 当前结论

当前 eval 更像“文字记忆机制回归测试”，还不是完整“AI 眼镜真实使用场景测试集”。

当前场景统计：

| 指标 | 数量 |
| --- | ---: |
| 总场景 | 127 |
| active 场景 | 60 |
| target 场景 | 67 |
| `memory_mechanism` 场景 | 34 |
| `continuous_input` 场景 | 2 |
| `public_dialogue_*` target 场景 | 5 |
| `text_cleaning_semantics` target 场景 | 3 |
| `llm_first_answer_quality` target 场景 | 2 |
| `document_text_scope` target 场景 | 1 |
| `source_control` target 场景 | 2 |
| `real_context_pressure` target 场景 | 5 |
| `ambient_emotion_metadata` target 场景 | 3 |

这说明底层记忆状态机、去重、纠错、召回仲裁、evidence 和长输入基础链路已经有比较强的回归保护；但当前主线应只围绕“文字输入、ASR 后文字、`md` 文档输入”继续收敛。2026-06-12 已验证 P0.1 三个真实语境 target：`real_context_prefix_not_in_task_body_target`、`real_context_correction_does_not_create_fragment_memories_target`、`real_context_natural_correction_task_supersede_target`，证明口语命令壳层和自然纠错时间碎片这两个写脏库源头已有最小回归保护；这不代表真实音频 runtime、原生 App、主动提醒和更完整的产品化输入闭环已经完成。

## 覆盖矩阵

| 用户场景 | 当前状态 | 已有代表场景 | 缺口 | 建议新增 target 场景 |
| --- | --- | --- | --- | --- |
| 日常闲聊和低价值输入 | weak | `greeting_no_memory`, `public_dialogue_chitchat_negative`, `real_context_chitchat_tasks_transient_preference_target`, `audit_replay_document_source_explanation_target`, `audit_replay_raw_timeline_source_explanation_target`, `audit_replay_structured_memory_source_explanation_target`, `audit_replay_followup_switches_to_latest_business_source_target`, `audit_replay_latest_none_source_does_not_fallback_target`, `audit_replay_long_chain_respects_latest_source_each_time_target`, `audit_replay_document_followup_after_explanation_target`, `audit_replay_document_followup_then_none_does_not_fallback_target`, `audit_replay_document_followup_then_structured_switch_target`, `audit_replay_document_then_raw_timeline_switch_target`, `audit_replay_structured_then_document_switch_target`, `audit_replay_long_chain_structured_document_none_timeline_target`, `audit_replay_structured_then_document_followup_target`, `audit_replay_timeline_then_none_followup_target`, `audit_replay_document_none_followup_then_timeline_target`, `audit_replay_timeline_none_followup_then_structured_target`, `audit_replay_mixed_chain_document_none_timeline_structured_target`, `audit_replay_mixed_chain_document_followup_none_timeline_structured_target`, `audit_replay_mixed_chain_document_none_emotional_timeline_structured_target` | 已新增连续闲聊夹任务与临时偏好 target，也把“正常回答后追问依据”的 audit replay 扩到 document、timeline 和 structured-memory 三类主来源。当前已经形成 3 组较稳定的自然语义族：1）来源切换族：`document -> timeline`、`timeline -> structured`、`timeline -> none -> structured` 等；2）文档 follow-up 族：同文档细节 follow-up、structured 后接 document follow-up；3）mixed chain 族：`structured -> document -> none -> raw_timeline`、`document -> none -> raw_timeline -> structured`、`document follow-up -> none -> raw_timeline -> structured`，以及中间再插入情绪/闲聊空转轮的长回环。这些场景都在验证同一个产品规律：解释回复只跟随最新业务回答，不回退旧来源，也不把普通 follow-up、情绪闲聊或旧 explanation 自己当锚点。当前仍缺更贴近真实聊天的更长链路和更多来源交错变体。 | 后续补更多长回环与自然 follow-up 变体 |
| 随口记事和后台价值提取 | weak | `passive_background_capture`, `glasses_walk_fragmented_day_plan`, `text_cleaning_filler_plus_task_target`, `real_context_chitchat_tasks_transient_preference_target`, `real_context_cross_turn_task_confirmation_target`, `real_context_multi_speaker_reported_tasks_target`, `multi_speaker_labeled_transcript_target`, `audit_replay_document_source_explanation_target`, `audit_replay_raw_timeline_source_explanation_target`, `audit_replay_structured_memory_source_explanation_target`, `audit_replay_followup_switches_to_latest_business_source_target`, `audit_replay_latest_none_source_does_not_fallback_target`, `audit_replay_long_chain_respects_latest_source_each_time_target`, `audit_replay_document_followup_after_explanation_target`, `audit_replay_document_followup_then_none_does_not_fallback_target`, `audit_replay_document_followup_then_structured_switch_target`, `audit_replay_document_then_raw_timeline_switch_target`, `audit_replay_structured_then_document_switch_target`, `audit_replay_long_chain_structured_document_none_timeline_target`, `audit_replay_structured_then_document_followup_target`, `audit_replay_timeline_then_none_followup_target`, `audit_replay_document_none_followup_then_timeline_target`, `audit_replay_timeline_none_followup_then_structured_target`, `audit_replay_mixed_chain_document_none_timeline_structured_target`, `audit_replay_mixed_chain_document_followup_none_timeline_structured_target` | 走路碎片口述、filler+task、连续闲聊夹任务、临时偏好、跨轮任务确认、多人转述和 speaker label 多人转写已有 target；真实 audit replay 已经不只是“多加几条 case”，而是开始形成自然 follow-up / source-switch 语义族，稳定覆盖 document、raw timeline、structured-memory、none 四类来源，以及它们之间的普通追问、文档内追问、和更长 mixed chain 切换。当前这些场景共同验证的是：中间插入普通问答或普通 follow-up 时，系统仍应只跟随最新业务回答，不回退旧来源。 | 后续补更长口语 replay 和更复杂 recall/write 混合链路 |
| 会议和长口述 | covered | `continuous_input_meeting_notes`, `continuous_input_daily_project_recap`, `public_dialogue_long_context`, `glasses_long_speech_with_noise_and_tasks`, `text_cleaning_local_do_not_remember_target`, `text_cleaning_self_correction_target` | 基础长输入已进 active；噪声/重复/闲聊夹任务、局部不要记和自我纠正已先落 target，并用 deterministic `intent_payloads` / `segment_payloads` 隔离外部 LLM 网络波动。2026-06-03 诊断通过。 | 后续从 target 结果中挑稳定场景升 active |
| 个人画像和偏好 | covered | `preference_memory_write`, `memory_mechanism_preference_duplicate_merge`, `memory_mechanism_drift_guard_current_intent_override`, `glasses_temporary_preference_not_profile` | 机制覆盖较好；临时偏好不误存也已有 target，并在 2026-06-04 诊断通过。后续重点是更多口语化变体。 | 后续补更多临时偏好口语变体 |
| 日程、任务和提醒 | weak | `upcoming_schedule_recall`, `memory_mechanism_reminder_skips_closed_task`, `public_dialogue_schedule_event` | 手动提醒检查已覆盖；主动提醒授权、触发窗口、静音/抑制仍不是 runtime。 | `proactive_reminder_authorization_window_target` |
| 文档和文字材料输入 | weak | `observation_not_used_for_document_detail`, `memory_mechanism_multi_source_document_detail_wins`, `app_audio_transcript_ingestion_target`, `app_document_and_voice_weekly_report_target`, `document_detail_recent_context_not_compete_target`, `md_document_state_vs_background_target`, `document_update_delete_followup_target`, `cross_document_compare_metadata_target`, `cross_document_compare_summary_target`, `cross_document_compare_risk_summary_target`, `cross_document_compare_advice_summary_target`, `cross_document_compare_conclusion_summary_target`, `cross_document_compare_applicability_summary_target` | 当前主线只保留 Markdown 文档细节和转写后文字导入；最近文档指代 + 文档细节 + recent context、`md` 背景 vs 结构化项目状态、文档更新/删除后追问、跨文档比较 metadata guard，以及 metadata 级高层差异摘要/风险差异/建议差异/结论差异/适用场景差异 都已落 target，并在 2026-06-04 诊断通过。仍缺更多文档多段细节和更复杂比较型回答组织。 | 后续补更复杂跨文档对比问法 |
| 原话回忆和 timeline evidence | covered | `memory_kernel_trace_contract`, `memory_mechanism_multi_source_raw_timeline_wins`, `memory_mechanism_inactive_evidence_not_recalled_from_active_timeline` | 文本原话和 evidence 生命周期较好；本次新增 `source_summary` 和内置 audit 回放，但仍缺音频证据。 | 后续补音频证据和更细 source 过滤 |
| 总结、回顾和 observation | covered | `observation_recent_work_recall`, `memory_mechanism_observation_scope_engineering_vs_project`, `llm_first_recent_activity_answer_quality`, `cross_week_life_review_with_corrections`, `alignment_drift_summary_source_mix_target` | scope 和 evidence 已有门禁，跨周纠错和 source mix 已先落 target；缺更复杂项目状态结构化。 | 后续从 target 结果中挑稳定场景升 active |
| 纠错、删除和用户控制 | weak | `memory_mechanism_correction_target_preference_supersede`, `memory_mechanism_last_retained_evidence_hard_purge_cleans_parent`, `user_delete_source_chunk_after_recall_target`, `document_update_delete_followup_target` | 机制级纠错/删除已覆盖；用户删除来源 chunk、文档更新/删除后追问已先落 target，但还不是完整批量纠正 workflow。 | 后续补完整纠正流程 |
| 隐私和敏感信息 | weak | `passive_background_capture`, `asr_sensitive_noise_not_saved_target` | ASR 敏感误听已先落 target，2026-06-03 诊断通过；当前断言检查 timeline 脱敏和 redaction category，不把真实 ASR runtime 伪装成已完成。 | 后续补更多敏感类型和确认流 |
| 位置、天气和导航上下文 | weak | `location_missing_no_guess`, `observation_not_used_for_weather_location` | 仍有组合场景缺口，但不是当前文字主线第一优先级。 | `location_distance_no_memory_target` |
| 失败、慢响应和可观测性 | weak | `reports/eval-latest.md` slowest turns, `chat_turn_failed` 代码路径, `source_summary` / `memory_processing` debug target, `failed_turn_audit_memory_retrieval_target`, `failed_turn_audit_assistant_response_target`, `failed_turn_audit_memory_snapshot_target` | eval 汇总有耗时统计；runner 已开始等待后台 memory job 终态再读 `new_memories`，Debug 面板和对话内解释已能展示来源与 memory job 阶段；记忆召回失败、主回答阶段失败、memory snapshot 失败后写出 `chat_turn_failed` audit 的 target 都已落地并在 2026-06-04 诊断通过。仍缺更广的慢回复阶段定位和更多失败阶段覆盖。 | 后续补更细失败阶段与慢回复阶段定位 |
| 音频/ASR/真实眼镜 runtime | out-of-scope-now | `glasses_long_speech_with_noise_and_tasks`, `asr_sensitive_noise_not_saved_target`, `app_audio_transcript_ingestion_target` | 当前只把“ASR 后文字”当文字输入处理，不在本轮设计真实音频上传、原始 ASR runtime、断句、说话人和原始音频证据。 | later |
| 主动服务 runtime | out-of-scope-now | `memory_mechanism_reminder_skips_closed_task` 只覆盖手动 check | 缺授权、触发、抑制、勿扰和推送通道；本轮不纳入当前文字主线实现。 | later |

状态含义：

- `covered`：已有 active 或强机制 eval 覆盖，能作为当前回归门禁。
- `weak`：有零散覆盖，但不能代表真实 AI 眼镜使用场景。
- `missing`：当前 eval 没有直接覆盖。
- `not product-ready`：产品能力本身未实现，只能作为 target/known gap 记录。
- `out-of-scope-now`：当前仓库里有相关背景或 target，但这轮 LLM-first 文字主线不继续推进它。

## 场景晋级和新增优先级

已稳定的对话式周报/注意事项场景可以进入 `--strict` active gate；当前优先补文字主线 target，真实硬件、主动服务和更复杂用户控制仍先保留 target 或标记 out-of-scope：

| 优先级 | 建议 scenario id | 场景目标 | 关键断言 |
| --- | --- | --- | --- |
| P0 | `glasses_walk_fragmented_day_plan` | 已新增。走路时碎片口述一天安排，系统应低打扰整理并保存高价值日程/任务；候选抽取使用 deterministic `intent_payloads`，重点测后台写入和门控。 | `saved_count_min`、`saved_contains`、`memory_processing_status`、planner/debug。 |
| P0 | `glasses_long_speech_with_noise_and_tasks` | 已新增。长语音转写里有重复句、口头禅、闲聊和任务，系统应过滤低价值片段并沉淀任务/风险；候选抽取使用 deterministic `intent_payloads`。 | `reply_contains_any`、`saved_count_min`、`saved_not_contains`、`debug.planner.reply_mode`。 |
| P0 | `asr_sensitive_noise_not_saved_target` | 已新增。ASR 文本里出现疑似密码、验证码、证件号，不能静默长期保存。 | `saved_not_contains`、timeline `redacted`、`redaction_categories`。 |
| P1 | `app_audio_transcript_ingestion_target` | 已新增。App/ASR 转写后文本通过 `memory_import` action 进入统一导入管道。 | `saved_count_min`、`saved_contains`、`source_summary`。 |
| P1 | `app_document_and_voice_weekly_report_target` | 已升 active。App/Markdown 文档和 `app_audio_transcript` 口述项目状态共同参与聊天内周报草稿。 | `conversation_action=weekly_report`、`reply_contains`、`document` + `structured_memory` source summary、`recalled_document_contains`。 |
| P1 | `conversation_weekly_progress_summary_target` | 已升 active。用户自然问“这周进展如何”时，复用启发式周报草稿，而不是掉到普通 LLM。 | `reply_contains`、`api_calls=0`、`debug.planner.conversation_action`、`source_summary`。 |
| P1 | `conversation_attention_items_target` | 已升 active。用户自然问“我接下来有什么要注意的”时，召回 open task 和项目风险，跳过 completed/cancelled task。 | `recalled_contains`、`recalled_not_contains`、`debug.memory.event_recall.strategy`、`source_summary`。 |
| P1 | `user_delete_source_chunk_after_recall_target` | 已新增。用户基于某次原文召回要求删除来源证据，删除后 active timeline 不再召回。 | `timeline_search`、`delete_timeline_chunks`、`recalled_timeline_not_contains`。 |
| P1 | `document_update_delete_followup_target` | 已新增。Markdown 文档更新后细节追问应使用新版原文；文档删除后同一细节题不再召回旧文档。 | `document_update`、`document_delete`、`document_recall.strategy`、`source_summary.primary_source`。 |
| P1 | `cross_document_compare_metadata_target` | 已新增。两份标题相近文档并存时，跨文档比较应停在 metadata 级别，不误进单文档全文细节回答。 | `document_recall.reason=cross_document_compare_query`、`document_count=2`、`metadata_only_for_cross_document_compare`。 |
| P1 | `cross_document_compare_summary_target` | 已新增。跨文档比较在保持 metadata guard 的前提下，应能根据两份文档的高层摘录/摘要组织出“侧重点差异”，而不是只报文件名。 | `reply_contains` 两份文档的高层差异、`reply_not_contains` 正文细节、`document_recall.strategy=metadata`。 |
| P1 | `cross_document_compare_risk_summary_target` | 已新增。跨文档比较在保持 metadata guard 的前提下，应能按问法优先比较 `风险/卡点` 这类高层段落，而不是退回通用摘要或正文细节。 | `reply_contains` 风险差异、`reply_not_contains` 建议/正文细节、`document_recall.strategy=metadata`。 |
| P1 | `cross_document_compare_advice_summary_target` | 已新增。跨文档比较在保持 metadata guard 的前提下，应能按问法优先比较 `建议/做法` 这类高层段落，而不是退回通用摘要或风险细节。 | `reply_contains` 建议差异、`reply_not_contains` 风险细节、`document_recall.strategy=metadata`。 |
| P1 | `cross_document_compare_conclusion_summary_target` | 已新增。跨文档比较在保持 metadata guard 的前提下，应能按问法优先比较 `结论/决定` 这类高层段落，而不是退回通用摘要或建议细节。 | `reply_contains` 结论差异、`reply_not_contains` 建议细节、`document_recall.strategy=metadata`。 |
| P1 | `cross_document_compare_applicability_summary_target` | 已新增。跨文档比较在保持 metadata guard 的前提下，应能按问法优先比较 `适用场景/适合谁/适用对象` 这类高层段落，而不是退回通用摘要或建议细节。 | `reply_contains` 适用场景差异、`reply_not_contains` 建议细节、`document_recall.strategy=metadata`。 |
| P1 | `cross_week_life_review_with_corrections` | 已新增。跨天/跨周生活流里有旧事实和纠错，回顾时应使用新事实。 | `recalled_not_contains`、`active_memory_not_contains`、`debug_contains`。 |
| P1 | `alignment_drift_summary_source_mix_target` | 已新增。summary 同时面对 profile、observation 和 timeline source 时，稳定画像不污染最近活动。 | `reply_not_contains`、`recalled_not_contains`、`recall_arbitration` debug。 |
| P1 | `ordinary_question_recent_context_not_injected_target` | 普通知识问答即使前面刚导入/刚聊过材料，也不应把 recent context 塞进主回复。 | `debug.recent_context_capsule.injected_to_main_llm=false`、`reply_not_contains`、`main prompt` 无 recent capsule。 |
| P1 | `document_detail_recent_context_not_compete_target` | 文档细节题即使命中最近文档背景，也应保持 document 为主证据，recent context 只做指代消歧。 | `document_recall.strategy=full_document`、`recall_arbitration.primary_source=document`、`debug.recent_context_capsule.injected_to_main_llm=false`。 |
| P1 | `text_cleaning_filler_plus_task_target` | 夹杂口头禅和背景噪声的文字/转写输入，应只提取真正任务。 | `saved_contains`、`saved_not_contains`、`semantic_cleaning.extractable_segment_count`。 |
| P1 | `text_cleaning_local_do_not_remember_target` | 局部“不要记”不能误伤后半句任务。 | `saved_contains`、`saved_not_contains`、`do_not_remember_scope`。 |
| P1 | `text_cleaning_self_correction_target` | 自我纠正片段应在 segment 层被视为 correction 或 extractable span。 | `saved_contains`、`debug_contains`、`semantic role`。 |
| P1 | `md_document_state_vs_background_target` | `md` 文档里的背景描述不应自动盖过结构化项目状态/任务。 | `reply_contains`、`reply_not_contains`、`document` vs `structured_memory` source summary。 |
| P1 | `real_context_chitchat_tasks_transient_preference_target` | 连续闲聊、吐槽、临时偏好和真实任务混在一段 ASR 后文字里时，只保存任务，不保存闲聊和临时偏好。 | `saved_count_min`、`saved_contains`、`saved_not_contains`、`semantic_cleaning.extractable_segment_count`。 |
| P1 | `glasses_temporary_preference_not_profile` | 已新增。用户明确说某个偏好只是临时试试、先别沉淀时，应在 `source_transient_context_only` 阶段直接跳过长期写入，不误生成 profile preference。 | `no_saved`、`memory_processing.status=not_needed`、`skip_policy.role=ephemeral_context`。 |
| P1 | `real_context_cross_turn_task_confirmation_target` | 跨轮对话里先说临时想法不保存，后续明确确认任务后再保存，并能在下一轮待办追问中召回。 | `no_saved`、`skip_policy.role=ephemeral_context`、`saved_count_min`、`recalled_contains`。 |
| P1 | `real_context_multi_speaker_reported_tasks_target` | 多人会议转述里只保存明确项目任务，不把旁人的偏好或闲聊误写成用户画像/任务。 | `saved_count_min`、`saved_contains`、`saved_not_contains`、`do_not_remember_scope`。 |
| P1 | `multi_speaker_labeled_transcript_target` | 阶段 C 第一刀 target。明确 speaker label 的多人转写文本应解析出 `user/known_person/unknown_speaker`，保存用户参与分工，过滤旁人偏好和 sensitive/unknown 片段，并支持按参与人复盘。 | `saved_contains`、`saved_not_contains`、`conversation_session` debug、`event_recall.strategy=text_search`、`source_summary.primary_source=structured_memory`。 |
| P1 | `multi_speaker_no_user_no_memory_target` | 阶段 C 第二刀 target。只有旁人对话、没有用户参与时，不允许退回普通 import 写长期记忆。 | `no_saved`、`conversation_session.rejected_turns.reason=no_user_turn`。 |
| P1 | `multi_speaker_unknown_task_not_named_target` | 阶段 C 第二刀 target。`speaker_2` 这类 unknown speaker 的任务不能写成确定联系人事实。 | `no_saved`、`participants.speaker_2=unknown_speaker`、`unknown_speaker_not_saved_as_memory_fact`。 |
| P1 | `multi_speaker_sensitive_fragment_filtered_target` | 阶段 C 第二刀 target。多人分工里夹杂验证码等敏感片段时，保留安全分工，过滤敏感分句。 | `saved_contains`、`saved_not_contains`、`sensitive_fragment_filtered`、`candidate_turn_indices`。 |
| P1 | `multi_speaker_multi_task_private_preference_target` | 阶段 C 第二刀 target。同一会话多个参与人和多个任务时，只保存用户参与协作事实，不保存旁人私人偏好。 | `saved_contains`、`saved_not_contains`、`third_party_preference`、`candidate_turn_indices`。 |
| P1 | `multi_speaker_alias_assignment_target` | 阶段 C 第三刀 target。用户显式说“speaker_2 是李四”后，当前批次内协作任务可以归因到李四，但不代表真实联系人系统完成。 | `saved_contains`、`saved_not_contains`、`speaker_aliases`、`alias_applied_turns`。 |
| P1 | `multi_speaker_alias_privacy_boundary_target` | 阶段 C 第三刀 target。speaker alias 命名后仍不能倒灌旁人私人偏好或敏感片段。 | `saved_contains`、`saved_not_contains`、`rejected_reasons`。 |
| P1 | `multi_speaker_split_quote_recall_target` | 阶段 C 第三刀 target。多人任务候选按参与人拆分后，可按“客户拜访 + 报价”召回李四相关事项。 | `saved_count_min`、`candidate_facts`、`recalled_contains`、`event_recall.strategy=text_search`。 |
| P1 | `ambient_emotion_chitchat_wake_query_target` | 已验证。按钮模拟唤醒里，环境原话仍会注入主上下文，但底层 emotion metadata 不再直接驱动回复；没有高置信融合结果时，`reply_emotion_used=false`。 | `ambient_wake_query`、`reply_contains`、`debug.ambient_context.emotion_fusion.reply_emotion_used=false`、`no_saved`。 |
| P1 | `ambient_emotion_unknown_low_confidence_target` | 已验证。情绪 metadata 为 `unknown` 或低置信时，回复不能强行断言用户情绪，且融合层必须保守回退。 | `reply_contains_any` 不确定性措辞、`reply_not_contains` 强断言、`debug.ambient_context.emotion_fusion.reply_emotion_used=false`。 |
| P1 | `ambient_emotion_do_not_remember_short_context_target` | 已验证。环境片段里用户说“别记这个”时，只作为当前语境回答，不写长期记忆；情绪只作为短期回复提示，不改变长期写入边界。 | `reply_contains` 当前语境/不写长期、`no_saved`、`debug.ambient_context`。 |
| P2 | `location_distance_no_memory_target` | 距离/位置问题只使用本轮临时定位，不保存长期位置画像。 | `api_calls`、`no_saved`、location debug。 |
| P2 | `failed_turn_audit_memory_retrieval_target` | 已新增。记忆召回失败时应写出可诊断 `chat_turn_failed` audit，并暴露失败阶段、异常类型和失败时的 planner/debug。 | `chat_expect_failure`、`audit_record_type`、`failed_stage/error_type/debug.failure`。 |
| P2 | `failed_turn_audit_assistant_response_target` | 已新增。主回答阶段失败时应写出可诊断 `chat_turn_failed` audit，并明确失败发生在 `assistant_response`。 | `chat_expect_failure`、`audit_record_type`、`failed_stage=assistant_response`、`debug.failure.stage`。 |
| P2 | `failed_turn_audit_memory_snapshot_target` | 已新增。主回复后 memory snapshot 失败时应写出可诊断 `chat_turn_failed` audit，并明确失败发生在 `memory_snapshot`。 | `chat_expect_failure`、`audit_record_type`、`failed_stage=memory_snapshot`、`debug.failure.stage`。 |
| P2 | `audit_replay_document_source_explanation_target` | 已新增。用户先问文档细节、再追问“你为什么这么说”时，应基于上一轮 audit/source summary 解释主要依据来自文档原文。 | `debug.explanation_context.record_found`、`source_summary.primary_source=document`、解释回复不自由编造。 |
| P2 | `audit_replay_raw_timeline_source_explanation_target` | 已新增。用户先追问自己之前说过的原话、再追问“依据是什么”时，应解释主要依据来自 timeline 原话 chunk，而不是 observation。 | `debug.explanation_context.record_found`、`source_summary.primary_source=raw_timeline`、解释回复明确 `timeline 原话`。 |
| P2 | `audit_replay_structured_memory_source_explanation_target` | 已新增。用户先问项目当前状态、再追问“你的依据是什么”时，应解释主要依据来自结构化任务/项目状态，文档只作为背景。 | `debug.explanation_context.record_found`、`source_summary.primary_source=structured_memory`、解释回复明确结构化记忆优先。 |
| P2 | `audit_replay_latest_none_source_does_not_fallback_target` | 已新增。用户先问一条有明确来源的问题，再切到普通无来源问答，随后追问“你为什么这么说”时，解释必须跟随最新那条无来源回答并明确“当前没有可用来源”，不能回退到更早旧来源。 | `debug.explanation_context.record_found`、`source_summary.primary_source=none`、`reply_contains=当前没有可用来源`、`reply_not_contains=timeline 原话`。 |
| P2 | `audit_replay_long_chain_respects_latest_source_each_time_target` | 已新增。更长多轮 replay 在 `raw_timeline -> none -> structured_memory` 连续切换时，每次 explanation 都必须跟随最新业务回答来源，不串回更早旧来源。 | 三次 explanation 分别命中 `raw_timeline`、`none`、`structured_memory`，且 `reply_not_contains` 旧来源说明。 |
| P2 | `audit_replay_document_followup_after_explanation_target` | 已新增。解释过一次后，用户继续追问同一份文档另一处细节，再追问依据时，系统应切到最新那条文档业务回答，并优先裁出相关文档小节，而不是继续粘在更早那条费用回答上。 | `reply_contains=第一天从林州出发`、`document_recall.reason=recent_document_followup`、`document_recall.strategy=sections`、解释回复继续保持 `primary_source=document`。 |
| P2 | `audit_replay_document_followup_resume_after_none_detour_target` | 已新增。用户先问文档细节、追问一次依据，中间插入一句普通无来源问答，再回到“那路线怎么写的？”这类同文档 follow-up 并再次追问依据时，系统仍应继续接住刚才那份文档，而不是因为中途岔开一句普通问答就丢掉文档锚点。 | 中间普通问答保持 `primary_source=none`；回到文档 follow-up 时仍命中 `document_recall.reason=recent_document_followup`、`document_recall.strategy=sections`、`source_summary.primary_source=document`；最后 explanation 继续明确 `上传文档原文`。 |
| P2 | `audit_replay_document_followup_then_none_does_not_fallback_target` | 已新增。文档细节、解释、同文档 follow-up、再次解释之后，如果用户又切到普通无来源问答再追问依据，系统仍应只跟随最新那条无来源业务回答，不回退去引用前面的文档依据。 | `reply_contains=当前没有可用来源`、`reply_not_contains=上传文档原文/第一天从林州出发`、`source_summary.primary_source=none`。 |
| P2 | `audit_replay_document_followup_then_structured_switch_target` | 已新增。文档细节、解释、同文档 follow-up、再次解释之后，如果用户又切到结构化项目状态问题再追问依据，系统仍应切到最新那条结构化业务回答，不回退去引用前面的文档依据。 | `reply_contains=结构化任务/文档只作为背景补充`、`reply_not_contains=上传文档原文/第一天从林州出发`、`source_summary.primary_source=structured_memory`。 |
| P2 | `audit_replay_document_then_raw_timeline_switch_target` | 已新增。用户先问文档背景细节、追问依据、再切到自己之前说过的原话回忆、最后再次追问依据时，系统应切到最新那条 raw timeline 业务回答，不继续沿用前面的文档依据。 | 文档解释轮保持 `primary_source=document`；原话回忆轮命中 `source_summary.primary_source=raw_timeline`；最后 explanation `reply_contains=timeline 原话` 且 `reply_not_contains=上传文档原文`。 |
| P2 | `audit_replay_structured_then_document_switch_target` | 已新增。用户先问项目当前状态、追问一次依据、随后切到项目文档里的背景细节、最后再次追问依据时，系统应切到最新那条 document 业务回答，不继续沿用前面的 structured-memory 依据。 | 项目状态解释轮保持 `primary_source=structured_memory`；文档细节轮命中 `source_summary.primary_source=document`；最后 explanation `reply_contains=上传文档原文` 且 `reply_not_contains=结构化任务`。 |
| P2 | `audit_replay_long_chain_structured_document_none_timeline_target` | 已新增。更长多轮 replay 在 `structured_memory -> document -> none -> raw_timeline` 连续切换时，每次 explanation 都必须跟随最新业务回答来源，不串回更早旧来源。 | 四次业务回答分别命中 `structured_memory`、`document`、`none`、`raw_timeline`；对应 explanation 依次只保留当前来源说明，并排除旧来源文案。 |
| P2 | `audit_replay_structured_then_document_followup_target` | 已新增。用户先问项目状态、追问一次依据、再切到文档细节、继续普通 follow-up 追问同一文档另一处细节、最后再次追问依据时，系统应跟随最新那条文档 follow-up 业务回答，而不是把普通 follow-up 或更早结构化解释当成新的锚点。 | 文档 follow-up 轮命中 `document_recall.reason=recent_document_followup`、`document_recall.strategy=sections`、`source_summary.primary_source=document`；最后 explanation `reply_contains=上传文档原文` 且 `reply_not_contains=结构化任务`。 |
| P2 | `audit_replay_timeline_then_none_followup_target` | 已新增。用户先问 timeline 原话、追问一次依据、再切到普通无来源问答、继续普通 follow-up、最后再次追问依据时，系统应跟随最新那条 none 业务回答，而不是把普通 follow-up 或更早的 timeline 解释当成新的锚点。 | timeline 轮命中 `primary_source=raw_timeline`；两条普通问答都保持 `primary_source=none`；最后 explanation `reply_contains=当前没有可用来源` 且 `reply_not_contains=timeline 原话`。 |
| P2 | `audit_replay_document_none_followup_then_timeline_target` | 已新增。用户先问文档背景细节、追问一次依据、再切到普通无来源问答、继续普通 follow-up、最后切到 timeline 原话回忆并再次追问依据时，系统应切回最新那条 raw_timeline 业务回答，不被中间的 none/follow-up 干扰。 | 文档轮保持 `primary_source=document`；中间两条普通问答保持 `primary_source=none`；timeline 轮命中 `primary_source=raw_timeline`；最后 explanation `reply_contains=timeline 原话` 且 `reply_not_contains=当前没有可用来源/上传文档原文`。 |
| P2 | `audit_replay_timeline_none_followup_then_structured_target` | 已新增。用户先问 timeline 原话、追问一次依据、再切到普通无来源问答、继续普通 follow-up、最后切到结构化项目状态并再次追问依据时，系统应切回最新那条 structured-memory 业务回答，不被中间的 none/follow-up 干扰。 | timeline 轮命中 `primary_source=raw_timeline`；中间两条普通问答保持 `primary_source=none`；structured 轮命中 `primary_source=structured_memory`；最后 explanation `reply_contains=结构化任务/文档只作为背景补充` 且 `reply_not_contains=当前没有可用来源/timeline 原话`。 |
| P2 | `audit_replay_mixed_chain_document_none_timeline_structured_target` | 已新增。更长 mixed replay 在 `document -> none -> raw_timeline -> structured_memory` 连续切换时，每次 explanation 都必须跟随最新业务回答来源，不串回更早旧来源。 | 文档轮命中 `primary_source=document`；none 段 explanation 明确 `当前没有可用来源`；timeline 段 explanation 明确 `timeline 原话`；最终 structured 段 explanation 明确 `结构化任务/文档只作为背景补充`，且每一段都排除前一段旧来源文案。 |
| P2 | `audit_replay_timeline_none_then_document_followup_target` | 已新增。用户先问 timeline 原话、追问一次依据，再切到普通无来源问答，随后切到文档细节并继续普通 document follow-up、最后再次追问依据时，系统应能在经历 `raw_timeline -> none` 后重新把锚点建立到最新 document follow-up 业务回答，而不是继续沿用旧的 timeline 或 none 说明。 | timeline 轮命中 `primary_source=raw_timeline`；中间普通问答保持 `primary_source=none`；文档 follow-up 轮命中 `document_recall.reason=recent_document_followup`、`document_recall.strategy=sections`、`source_summary.primary_source=document`；最后 explanation `reply_contains=上传文档原文` 且 `reply_not_contains=timeline 原话/当前没有可用来源`。 |
| P2 | `audit_replay_document_none_timeline_none_then_document_followup_target` | 已新增。用户先问文档细节、追问一次依据，再切到普通无来源问答、切到 timeline 原话回忆并再追问一次依据、随后再次插入一句普通无来源问答、最后回到同一份文档继续普通 follow-up 并再次追问依据时，系统仍应把锚点重新建回最新 document follow-up 业务回答，而不是停留在旧的 timeline 或 none 说明。 | 首轮文档解释保持 `primary_source=document`；中间 timeline 解释轮保持 `primary_source=raw_timeline`；再次插入的普通问答保持 `primary_source=none`；最后 document follow-up 轮命中 `document_recall.reason=recent_document_followup`、`document_recall.strategy=sections`、`source_summary.primary_source=document`；最后 explanation `reply_contains=上传文档原文` 且 `reply_not_contains=timeline 原话/当前没有可用来源`。 |
| P2 | `audit_replay_structured_with_emotional_chitchat_detour_target` | 已新增。用户先问结构化项目状态，随后插入两句情绪表达/轻量闲聊，再追问“那你这次为什么这么说？”时，系统应跳过中间这些空转闲聊轮，继续回到最近那条真正的业务回答，而不是因为中间两轮 `primary_source=none` 就退化成“当前没有可用来源”。 | 首轮项目状态轮命中 `primary_source=structured_memory`；中间两轮情绪/闲聊都保持 `primary_source=none`；最后 explanation 继续命中 `primary_source=structured_memory`，`reply_contains=结构化任务/文档只作为背景补充`，且 `reply_not_contains=当前没有可用来源`。 |
| P2 | `audit_replay_document_with_emotional_chitchat_detour_target` | 已新增。用户先问文档细节，随后插入两句情绪表达/轻量闲聊，再追问“那你这次为什么这么说？”时，系统应跳过中间这些空转闲聊轮，继续回到最近那条真正的文档业务回答，而不是因为中间两轮 `primary_source=none` 就退化成“当前没有可用来源”。 | 首轮文档细节轮命中 `primary_source=document`；中间两轮情绪/闲聊都保持 `primary_source=none`；最后 explanation 继续命中 `primary_source=document`，`reply_contains=上传文档原文`，且 `reply_not_contains=当前没有可用来源`。 |
| P2 | `audit_replay_timeline_with_emotional_chitchat_detour_target` | 已新增。用户先问 timeline 原话回忆，随后插入两句情绪表达/轻量闲聊，再追问“那你这次为什么这么说？”时，系统应跳过中间这些空转闲聊轮，继续回到最近那条真正的 timeline 业务回答，而不是因为中间两轮 `primary_source=none` 就退化成“当前没有可用来源”。 | 首轮原话回忆轮命中 `primary_source=raw_timeline`；中间两轮情绪/闲聊都保持 `primary_source=none`；最后 explanation 继续命中 `primary_source=raw_timeline`，`reply_contains=timeline 原话`，且 `reply_not_contains=当前没有可用来源`。 |
| P2 | `audit_replay_mixed_chain_document_none_emotional_timeline_structured_target` | 已新增。用户先问文档背景并追问一次依据，再切到普通无来源问答，随后插入两句情绪表达/轻量闲聊，再切到 timeline 原话回忆并追问依据，最后再切到结构化项目状态并再次追问依据时，系统应在 document、none、emotional chitchat、raw_timeline、structured_memory 交错出现时仍只跟随最新业务回答，不把中间情绪闲聊轮错当成新的解释锚点。 | 文档轮 explanation 保持 `primary_source=document`；普通问答和两轮情绪/闲聊都保持 `primary_source=none`；none 段 explanation 明确 `当前没有可用来源`；timeline 段 explanation 明确 `timeline 原话`；最终 structured 段 explanation 明确 `结构化任务/文档只作为背景补充`，且每一段都排除前面旧来源文案。 |

### Audit Replay 语义族索引

| 语义族 | 目标 | 当前代表 target |
| --- | --- | --- |
| `source-switch` | 验证来源在 document / raw_timeline / structured_memory / none 间切换时，解释始终只跟随最新业务回答。 | `audit_replay_followup_switches_to_latest_business_source_target`, `audit_replay_latest_none_source_does_not_fallback_target`, `audit_replay_document_then_raw_timeline_switch_target`, `audit_replay_structured_then_document_switch_target`, `audit_replay_timeline_none_followup_then_structured_target` |
| `document-followup` | 验证同一份文档里的自然追问不会把更早 explanation 或其他来源错当成新的解释锚点；即使中间岔开一句普通无来源问答，甚至经历一段 timeline 回忆回环，也还能回到刚才那份文档。 | `audit_replay_document_followup_after_explanation_target`, `audit_replay_document_followup_resume_after_none_detour_target`, `audit_replay_document_followup_then_none_does_not_fallback_target`, `audit_replay_document_followup_then_structured_switch_target`, `audit_replay_structured_then_document_followup_target`, `audit_replay_document_none_timeline_none_then_document_followup_target` |
| `mixed-chain` | 验证 document / none / raw_timeline / structured_memory 多段连续交错时，解释能连续多次切到最新来源，包括在经历 timeline 和 none 之后重新进入 document follow-up，以及中间插入情绪/闲聊空转轮时继续跳过这些非业务锚点。 | `audit_replay_long_chain_respects_latest_source_each_time_target`, `audit_replay_long_chain_structured_document_none_timeline_target`, `audit_replay_mixed_chain_document_none_timeline_structured_target`, `audit_replay_mixed_chain_document_followup_none_timeline_structured_target`, `audit_replay_timeline_none_then_document_followup_target`, `audit_replay_document_none_timeline_none_then_document_followup_target`, `audit_replay_structured_with_emotional_chitchat_detour_target`, `audit_replay_document_with_emotional_chitchat_detour_target`, `audit_replay_timeline_with_emotional_chitchat_detour_target`, `audit_replay_mixed_chain_document_none_emotional_timeline_structured_target` |

## Active 晋级规则

target 场景满足以下条件后，才考虑升为 active：

- 同类场景至少连续两次 live eval 通过，且失败原因不是外部 LLM/API 网络波动。
- 断言不能只靠回复关键词，至少覆盖写入、召回、debug、audit、job 终态或 active memory surface 之一。
- 不把未实现的产品能力伪装成已完成。例如主动提醒 runtime、音频上传/ASR、原始音频管理只能保留 target/known gap。
- 场景文案应是产品行为类型改写，不能为通过测试硬编码某一句话。

## 运行建议

先跑 target 诊断，不阻断当前 strict 门禁：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category public_chitchat_negative --category public_preference_write --category public_schedule_event --category public_temporal_recall --category public_long_context --background-wait 15
```

新增 target 后按新分类逐类跑，观察 `reports/eval-latest.md` 的 `target_failed_turns`、慢场景和失败样本，再决定是否修主链路或继续只记录产品缺口。

本轮已验证的文字主线 target：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category llm_first_answer_quality --background-wait 15
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category text_cleaning_semantics --background-wait 15
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category document_text_scope --background-wait 15
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category real_context_pressure --background-wait 15
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category source_control --background-wait 15
```

2026-06-03 验证结果：以上五组均为 `target_failed_turns=0`。这只证明当前文字主线 target 通过，不代表真实音频 runtime、原生 App 或生产级压力测试已经完成。

第一批 AI 眼镜 target 场景：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 1 --category glasses_fragmented_input --category glasses_noisy_long_speech --category glasses_asr_privacy --background-wait 15
```

2026-06-03 验证结果：第一批 AI 眼镜 target 三组均为 `target_failed_turns=0`，覆盖走路碎片口述、噪声长转写和 ASR 敏感误听；仍保持 target，不升 active strict 门禁。
