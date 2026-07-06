# 当前状态与缺口

更新时间：2026-05-27。本文是给 Codex 的代码审阅快照，用来判断“现在做到哪一步”和“下一步优先修什么”。如果和最新代码冲突，以代码为准。

## 当前结论

当前项目已经达到 Stage 2 可运行原型：

- 有 Web/语音 demo 入口。
- 有统一 `/api/chat` 主链路。
- 有 SQLite 用户记忆库。
- 有 SQLite 原文 timeline/chunk 库。
- 有轻量 Memory Kernel contract 和统一 `source_trace` / `recall_trace`。
- 有 profile/event/assistant_preference 基础分类。
- 有轻量 observation 后台长期归纳。
- 有本地 planner、fast path、结构化路由和主模型 fallback。
- 有回复优先后台记忆写入和 job 查询。
- 有 debug、audit、tests、evals。
- 有 eval 覆盖矩阵，用于区分底层记忆机制门禁和真实 AI 眼镜使用场景覆盖。
- 有 import、capture、weekly report、reminder check 的最小扩展能力。
- 后台 memory job 和 capture 状态已做 SQLite 最小持久化，重启后可以恢复查询或 stop/import。
- raw timeline/capture 写入前已有隐私脱敏；结构化长期记忆候选仍走 `should_write_memory_candidate()` 门控。
- 自然偏好写入、访问计数/strength、轻量召回排序、文档指代和聊天长输入 fast path 已有当前最小实现。
- chat/import 现在返回 `source_summary`，Web Debug 面板能展示本次回答依据、最近 audit 回放和 App/ASR 转写后文本导入结果。
- 写入、dedupe、correction 和 observation update 的低置信度路径已通过 `confidence_policy` 暴露阈值、含义和处理结果。
- `memory_type=task` 已有第一版业务状态：open/completed/cancelled 写入 tags；完成或取消会让旧 open task 退出待办召回和手动提醒检查，但保留可追溯事实。
- 底层文字记忆机制已形成第一版闭环：生命周期、evidence 引用、语义去重/冲突更新、纠错对象定位、多来源召回仲裁、strength cap/decay、observation scope 和 timeline evidence 管理都有当前实现和测试/eval 覆盖。

这些能力是“无限 AI 眼镜 + 手机端 App”最终形态的早期后端验证：眼镜侧低摩擦语音输入、App 侧文档/音频/文本输入、统一记忆内化、总结/回忆/回顾输出。

但它还不是生产级 AI 眼镜助手：没有真实硬件 runtime、没有原生手机 App、没有音频上传/转写管道、没有主动提醒推送、没有可靠 job 队列、没有完整项目知识图谱，也没有生产级账号/权限系统。

## 生产级完整系统距离

生产级完整记忆系统至少需要同时满足：多来源可靠输入、隐私和权限门控、可追溯长期记忆、稳定召回和回答组织、用户可控管理、主动服务、生产级可靠性和账号安全。当前最强的是“文本进入后怎么沉淀和召回”；最弱的是“真实设备/App 输入、主动服务和生产运维”。

| 维度 | 当前状态 | 生产级差距 |
| --- | --- | --- |
| 文字记忆内核 | 写入、门控、dedupe、纠错、evidence、删除、召回仲裁和 eval 已形成第一版闭环。 | 继续补更长上下文、项目状态结构化和跨来源组合验证。 |
| 真实输入 | Web 文字、浏览器语音、import、Markdown、capture、转写后文本导入。 | 缺真实眼镜 runtime、原生 App、音频上传、ASR、断句和说话人处理。 |
| 用户管理 | 记忆/文档面板、timeline evidence 查看和批量删除、轻量 audit 回放、回答依据摘要。 | 缺完整审计后台、复杂纠正 workflow、来源权限和音频证据管理。 |
| 主动服务 | 手动提醒检查、启发式周报草稿。 | 缺授权、触发窗口、勿扰/抑制、推送通道和主动触发 audit。 |
| 可靠性 | SQLite 本地 demo，job/capture 有最小持久化公开状态。 | 缺可靠 worker、重试、幂等、跨进程调度、多设备同步、账号权限和监控恢复。 |

详细定义见 `production-readiness.md`。

## 已完成的主路径

```text
用户输入
-> /api/chat
-> planner/router
-> 按需召回 profile/event/timeline/location/web
-> 本地回复或 AIAgent
-> 原话 timeline 和结构化记忆候选
-> 记忆候选门控
-> 同步保存或后台 job
-> 必要时后台 observation reflect
-> debug/audit
```

关键证据：

- `GlassesChatService.chat()` 是唯一主聊天 service。
- `EventMemoryStore` 管理 SQLite `memories` 表。
- `TimelineStore` 管理 raw turn、capture chunk 和全文搜索。
- `turn_planner.py` 决定 fast path、profile/event/web/location 和 `reply_mode`。
- `observation_reflect` 复用后台 memory job，把多条带证据的 profile/event 归纳成 `memory_type=observation`。
- `intent_policy.py` 集中长期记忆写入门控。
- `/api/memory/jobs` 能查询后台写入状态。
- `chat_audit.jsonl` 记录聊天、导入、后台写入和提醒检查。

## 当前最重要缺口

| 优先级 | 缺口 | 影响 | 下一步 |
| --- | --- | --- | --- |
| P1 | 主动提醒 runtime 缺失 | 手动 check 已接入真实 `check_reminders()` eval，但不能真正主动出现。 | 后续再设计授权、触发窗口、静音/抑制和 audit；不要先做后台定时器。 |
| P1 | 对话式周报和注意事项入口已进 active 门禁 | “这周进展如何”会复用启发式 `weekly_report()`；“我接下来有什么要注意的”会召回 open task 和项目风险，并通过 `source_summary` 解释 structured memory 来源。 | 继续补更多真实口语变体和跨周回顾 target；不要把它写成完整项目知识图谱或主动提醒 runtime。 |
| P1 | App 文档 + 口述周报已进 active 门禁 | App/Markdown 文档归档和 `app_audio_transcript` 转写文本导入后，自然问“这周进展如何”能合并文档背景、任务、决策、风险，并在 `source_summary` 同时标出 document 和 structured memory。 | 仍是启发式周报，不代表真实音频上传/ASR、说话人分离或完整项目图谱已完成。 |
| P1 | 长输入 active 门禁仍需继续扩展 | 聊天内 `continuous_capture` fast path 已有 active eval 覆盖会议记录和项目复盘，能阻断低打扰整理、批量沉淀和 debug 路由退化。 | 继续补更长上下文 source 组合、公开素材长口述 target 和误判边界，不要重复实现已入门禁的长输入基础链路。 |
| P1 | AI 眼镜场景覆盖矩阵仍需落到 target 场景 | `docs/context/eval-coverage.md` 已标注 covered/weak/missing/not product-ready，但大部分真实口语、ASR 噪声、App 音频和用户控制流程还只是计划。 | 先新增 target，不直接升 active；稳定后再把少量高价值场景纳入 strict 门禁。 |
| P1 | App 音频入口还未成型 | 最终产品不能只靠 Web 聊天输入；当前已有转写后文本导入入口和 target 场景。 | 后续再设计真实音频上传、ASR、断句和说话人处理。 |
| P1 | 持续收音待机与唤醒式现场问答已有按钮模拟 MVP | 前端可开启收音待机，把浏览器 ASR final 文本写入 ambient capture；点击“唤醒提问”后，下一句 query 会引用最近现场原话。当前已加固 running capture 注入、跨用户隔离和停止后不注入；`/api/audio/segment/process` 已支持真实音频片段校验、临时处理后立即删除，并接入 `SenseVoiceSmall` 片段级本地 ASR、独立基础声学情绪 metadata 优先 + SenseVoice token fallback，以及基于 `cam++ + 3 段用户参考声纹校准` 的 `speaker_hint=user/other/unknown` 保守判定。仍没有后台常驻收音、streaming/VAD 或完整 diarization runtime。 | 规划见 `ambient-audio-wakeword-plan.md`；下一步应继续补情绪模型实测、失败清理和后续 diarization/runtime 校准，不要绕过该入口直接写 capture。 |
| P1 | 音频转写后的记忆门控未验证 | 已用 `app_audio_transcript_ingestion_target` 验证转写后文本走统一导入管道；真实 ASR 噪声仍需更多样本。 | 继续补真实转写样本 target，保留 gate/audit。 |
| P1 | timeline 原文管理仍是基础版 | 现有记忆面板已能查看 evidence 原文、搜索 timeline，并批量安全删除 chunk；Debug 面板已有轻量 audit 回放。 | 后续再设计批量纠正、source/time 过滤、document chunk 引用和完整审计后台。 |
| P1 | memory job/capture 仍非完整可靠队列 | 已能 SQLite 恢复公开状态，但没有 worker 重放、超时补偿或跨进程调度。 | 暂保持 demo 级持久化，后续再设计可靠任务队列。 |
| P2 | observation 质量仍是轻量归纳 | 已新增复杂项目状态 observation/source evidence active eval，task 完成/取消和项目 scope 也已进入机制门禁，但还不是完整项目知识图谱。 | 后续补更长上下文 source 组合、项目状态结构化和过期边界。 |
| P2 | confidence fallback eval 可继续扩展到新 classifier | dedupe、correction、observation update 的低置信度 fallback 已进入 active 机制 eval；后续新增 classifier 时仍要补同类门禁。 | 新增 classifier 或 pending/确认流时，必须同步补 `confidence_policy` 断言。 |
| P2 | 周报还是启发式草稿 | 项目周报已能合并结构化记忆和文档 metadata/source summary，但分组、状态、风险和证据追踪仍不是完整项目管理系统。 | 逐步引入更强项目上下文和证据规则，不直接跳到项目图谱。 |

## 不能误判为已完成的能力

- `/api/reminders/check` 是手动提醒候选检查，不是主动提醒 runtime。
- `/api/weekly-report` 是启发式周报草稿，不是完整项目管理系统。
- `/api/capture/*` 是连续输入 API，不是硬件 ASR runtime。
- `/api/memory/import` 和 Markdown 上传是 App 文档/文本入口的后端雏形，不是原生手机 App。
- `/api/timeline/search` 和 `/api/timeline/chunks` 已支撑基础原文搜索、证据查看和批量安全删除，但不是完整 timeline 审计后台，也不支持直接编辑 raw chunk 原文。
- `observation` 是后台轻量归纳，不是 embedding、FAISS、实体图谱或 byzhou 全量长期记忆系统。
- 当前没有后台常驻收音；`/api/audio/segment/process` 已支持真实音频片段的大小/格式/时长校验、临时处理后立即删除，并接入 `SenseVoiceSmall` 片段级本地 ASR，但仍不是 streaming/VAD/runtime。
- “持续收音待机 + 唤醒式现场问答”已有可运行原型，并已完成 demo 可信闭环加固、独立 wake detector 接入、ambient 情绪 metadata 验证、片段级本地 ASR v1，以及 3 段用户声纹校准 + `cam++` centroid 相似度判定；但当前 wake detector backend 仍是最小本地短语方案，还没有真实麦克风常驻 runtime、streaming/VAD、完整多说话人重建或更完整的多说话人 diarization runtime。
- memory job/capture 的公开状态已能从 SQLite 恢复，但仍不是可靠任务队列或常驻 capture runtime。
- 浏览器语音只是 demo 交互，不是最终眼镜硬件方案。

## 判断是否完成一个功能

完成标准必须至少包含：

- 代码行为已落在真实主链路，而不是只加 helper。
- 有单元测试、service 测试、eval 场景或可复现手动对话。
- debug/audit 能解释关键决策。
- 没有绕过 `user_id` 隔离和敏感信息门控。
- `server.py` 与 `app.py` 的公开 API 没有明显偏离。
- 相关 docs 和 `PLANS.md` 已同步。

## 推荐下一步

如果继续当前 demo 主线，下一步不要重复做已进入 active 门禁的复杂 observation/source evidence、task 完成/取消、项目 scope、confidence fallback、纠错对象定位、多来源召回仲裁、长输入基础链路、对话式周报和注意事项基础入口；优先补更多真实口语变体、跨周回顾纠错、alignment drift source mix、来源删除后的解释闭环和失败 audit 注入。若转向产品能力，按钮模拟唤醒 MVP 已完成第一版，下一步优先参考 `ambient-audio-wakeword-plan.md` 的真实音频临时缓存生命周期，先实现独立音频临时处理入口和清理测试，再设计真实唤醒词、音频 ASR 和主动提醒授权；当前只验证了手动 `check_reminders()` 和聊天内查询，不要把它描述成主动提醒 runtime 或 always-on audio 成品。公开素材长口述仍可继续作为 target 诊断，用来发现更复杂口语输入和误判边界。
