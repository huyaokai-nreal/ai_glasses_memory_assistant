# ai_glasses_memory_assistant 包内 Python 文件说明

这份 README 是给第一次接手本包的同事看的“文件门牌表”。当你看到某个 `.py` 文件不知道它该不该改时，先看这里；如果你是按任务找入口，再看 `docs/context/code-map.md`；如果你要理解完整调用链，再看 `docs/context/system-flow-current.md`。

本文覆盖包内当前 Python 模块，包括 `audio_engine/` 和 `evals/`。代码是真相；如果本文和代码冲突，先读代码，再修本文。

## 新人先看这几个入口

| 文件 | 为什么先看 |
| --- | --- |
| `server.py` | HTTP/HTTPS API 入口，负责把前端请求转给 service。 |
| `agent_bridge.py` | 主 service 调度入口，聊天、导入、capture、音频、写库、job、audit 都从这里编排。 |
| `turn_planner.py` | 本地快速判断和简单时间解析，决定哪些问题能不走大模型。 |
| `turn_semantic_classifier.py` | 回复前统一语义分类，决定召回、写入候选、web/location 和 reply mode。 |
| `intent_policy.py` | 长期记忆写入的最终门控，敏感信息、问题句、低置信度都在这里拦。 |
| `memory_store.py` | 结构化长期记忆和文档归档的 SQLite 存储。 |
| `timeline_store.py` | 原话 timeline、capture chunk、memory job、speaker profile 的 SQLite 存储。 |

## 顶层 Python 文件

| 文件 | 一眼看懂 | 什么时候看它 | 注意边界 |
| --- | --- | --- | --- |
| `__init__.py` | Python 包标记。 | 只在包元信息需要变化时看。 | 不放业务逻辑。 |
| `agent_bridge.py` | `GlassesChatService` 主流程编排器。 | 改 `/api/chat` 行为、记忆写入/召回编排、capture、音频 service 方法、job、audit、周报、提醒时先看。 | 不是纯 helper；不要绕过它直接写 DB 或绕过 memory gate。 |
| `answer_synthesizer.py` | 用 LLM 产出回复组织指令和文本情绪判断。 | 想调整“怎么基于证据组织最终回复”或文本情绪标签时看。 | 不负责召回、写库或 HTTP。 |
| `app_home.py` | 统一解析应用 home 和 data 目录。 | 改数据目录、测试隔离目录、`AI_GLASSES_HOME` 规则时看。 | 不要硬编码用户目录；`HERMES_HOME` 只是迁移期 fallback。 |
| `audio_processing.py` | 旧导入路径的薄兼容导出。 | 维护旧调用方 import 兼容时看。 | 不创建模型；实现位于 `audio_engine/offline.py`。 |
| `capture_helpers.py` | continuous capture 的摘要和确认文案 helper。 | 只改 capture 用户可见短回复时看。 | 不做 capture 生命周期或写库决策。 |
| `conversation_candidate_helpers.py` | 多人 transcript 的结构化隐私 extraction plan。 | 改 user/known/unknown 角色、alias、敏感 fragment 或语义抽取输入时看。 | 只决定哪些带来源的片段可以进入语义抽取，不用本地业务词表生成 memory candidates。 |
| `conversation_helpers.py` | 把带说话人标签的文本解析成会话、参与者、turn 和 alias。 | 处理 `A:`、`张三：` 这类多人文本结构时看。 | 只做结构解析，不判断哪些内容该保存。 |
| `document_helpers.py` | Markdown 文档标题/摘要、文档问题识别、文档上下文拼装。 | 改文档导入、文档追问、最近文档引用、跨文档比较时看。 | 文档保存和搜索仍在 `memory_store.py`，service 编排仍在 `agent_bridge.py`。 |
| `env_loader.py` | 加载 app-owned `.env`，并快照/恢复 LLM 相关环境变量。 | 改本地配置读取、eval 临时环境隔离时看。 | 不覆盖已存在环境变量，不放真实密钥。 |
| `explanation_helpers.py` | 识别“为什么这么回答/为什么没保存”并格式化解释回复。 | 改解释类问题、evidence 引用展示、上一轮依据说明时看。 | 不改变聊天主控制流或 memory gate。 |
| `import_helpers.py` | 把文本/JSON 导入整理成 import item 和 `MemoryWriteCandidate`。 | 改 `/api/memory/import` 的条目拆分、类型 fallback、候选包装时看。 | 候选仍必须交给 service 和 `intent_policy.py` 门控。 |
| `intent_policy.py` | 长期记忆写入最终门控和少量本地快速回复。 | 改“该不该保存”、敏感信息拦截、问题句拦截、短确认回复时看。 | 这是隐私边界，不能为了通过 demo 绕开。 |
| `llm_client.py` | OpenAI-compatible chat completions client 封装。 | 改 LLM 请求格式、错误信息、client protocol 时看。 | 不负责读取环境变量；配置在 `llm_runtime.py`。 |
| `llm_runtime.py` | demo LLM 环境变量解析和 client 创建。 | 改 DeepSeek/OpenAI-compatible 配置、启动缺参报错时看。 | 当前只支持 `chat_completions`，不恢复本地模型 legacy fallback。 |
| `memory_candidate.py` | 记忆候选和 intent 决策的共享 dataclass。 | 需要看候选字段、intent 字段契约时看。 | 只定义结构，不做策略。 |
| `memory_confidence.py` | 记忆置信度阈值和 debug payload。 | 改写入阈值、去重阈值、置信度解释时看。 | 不直接决定保存；最终门控在 `intent_policy.py`。 |
| `memory_evidence.py` | evidence id 规范化和 timeline evidence 清理计划。 | 删除/purge 记忆或 timeline chunk，需处理 evidence 引用时看。 | 不直接删库，只提供引用和清理计划 helper。 |
| `memory_job_helpers.py` | 后台 memory job 的内部/公开 payload 和 stage 说明。 | 改 job 状态展示、debug payload、processing payload 时看。 | job 生命周期和持久化仍由 `agent_bridge.py` 编排。 |
| `memory_kernel.py` | 记忆层契约、source trace、recall trace 的说明型 helper。 | 需要统一解释记忆层输入输出、来源追踪字段时看。 | 不是新的存储层，也不替代 `memory_store.py`。 |
| `memory_lifecycle.py` | 记忆状态和合法状态流转 helper。 | 改 `active/stale/superseded/deleted` 状态规则时看。 | 不直接操作 SQLite。 |
| `memory_recall_arbitration.py` | 召回后仲裁：选主来源、丢弃冲突来源、补 debug reason。 | 改 profile/event/timeline/document 多来源冲突处理时看。 | 不执行检索，只处理检索后的取舍。 |
| `memory_store.py` | 结构化长期记忆和文档归档的 SQLite store。 | 改 memory/document schema、搜索、软删除、purge、去重、strength 时看。 | 高风险：不要随意改字段、迁移、删除语义或用户隔离。 |
| `privacy_filter.py` | 敏感文本脱敏。 | 改 token、JWT、银行卡、证件号、验证码等 redaction 时看。 | 不判断是否保存，只负责把敏感片段遮掉。 |
| `report_helpers.py` | 周报草稿、attention items、项目归类和背景 observation 判断。 | 改 weekly report 文案、项目名归类、注意事项回复时看。 | 周报是启发式草稿，不是正式报表系统。 |
| `response_timing.py` | assistant response timing/debug trace。 | 排查慢回复、统计 LLM/tool 阶段耗时时看。 | 只做观测，不改变业务决策。 |
| `segment_semantic_cleaner.py` | 长输入分段后的语义清洗：记忆候选、噪声、纠错、别记范围。 | 改长文本导入或 capture 分段筛选策略时看。 | 结果仍要经过 service 候选合并和 memory gate。 |
| `server.py` | 标准库 HTTP/HTTPS server 和 API route handler。 | 新增/修改 API route、请求解析、响应包装时看。 | 保持薄包装；业务逻辑放 `GlassesChatService`。 |
| `server_config.py` | server 启动参数、host/port/TLS、局域网 IP 和启动提示。 | 改命令行参数或启动文案时看。 | 不创建 server，不处理 API。 |
| `session_store.py` | demo 会话和消息历史 SQLite store。 | 改当前会话历史、session metadata 时看。 | 不是长期个人记忆，别和 `memory_store.py` 混用。 |
| `source_summary_helpers.py` | 来源摘要、audit 摘要、primary source label 和解释文案。 | 改 debug/source summary、人类可读来源说明时看。 | 不做真实召回，只总结已有 debug/source payload。 |
| `temporal_parser.py` | 复杂自然语言时间解析的 LLM fallback。 | 本地 day/hour 解析不够用，需要解析复杂时间表达时看。 | 简单时间优先 `turn_planner.py` 本地解析。 |
| `text_cleaning.py` | 输入文本归一、敏感片段标记、填充词/纠错/否定 marker、分段。 | 改进入记忆候选前的文本清洗和切段时看。 | 不直接生成最终记忆。 |
| `timeline_management_helpers.py` | timeline chunk 管理接口的 payload、删除摘要、redaction debug。 | 改 timeline 管理页/API 的展示数据时看。 | 不直接搜索或删除，底层在 `timeline_store.py`。 |
| `timeline_store.py` | 原话 timeline、capture chunk、memory job、speaker profile 的 SQLite store。 | 改原话证据、capture、memory job 恢复、speaker profile、chunk 搜索/删除时看。 | 原话证据不等于结构化长期记忆。 |
| `tts_service.py` | Edge TTS 文本清洗、参数校验和语音合成。 | 改 `/api/tts`、voice/rate/pitch、Markdown 去除时看。 | 可选能力；不要让 TTS 影响主聊天结果。 |
| `turn_planner.py` | 本地 planner、fast path、简单时间解析、长输入初判。 | 改问候、身份问题、敏感凭据、本地时间、简单计划召回、长输入 baseline 时看。 | 不要继续堆开放语义规则；复杂语义交给 `turn_semantic_classifier.py`。 |
| `turn_semantic_classifier.py` | 回复前 LLM 统一语义分类，产出 `PreReplyDecision`。 | 改召回意图、写入候选、web/location、reply mode、recall goal 时看。 | 它决定方向，不直接回答用户，也不最终写库。 |
| `web_search.py` | 可选 web 搜索，优先 `ddgs`，再 DuckDuckGo HTML fallback。 | 改实时信息搜索、搜索结果格式、网络错误说明时看。 | 搜索结果只服务当前 turn，不进入长期记忆。 |

## audio_engine/ 文件

| 文件 | 一眼看懂 | 注意边界 |
| --- | --- | --- |
| `audio_engine/contracts.py` | `audio_event.v1` 和 `AudioEventPlan` 结构。 | partial 永远是 UI/debug-only；final 才能交给 planner。 |
| `audio_engine/settings.py` | 采样率、帧长、pre-roll、partial 周期、唤醒等待和回收时间的集中配置。 | 不在前端或 runtime 散落同类数字。 |
| `audio_engine/backends.py` | Silero、Sherpa KWS、Paraformer 的 session/共享模型适配和 capability。 | 不自动下载模型，不公开绝对模型路径。 |
| `audio_engine/offline.py` | 旧 blob 与 ambient 共用的 SenseVoice、Cam++、emotion 和临时音频输入适配。 | 原始音频只进临时文件或有界内存，处理后立即释放。 |
| `audio_engine/runtime.py` | 每用户 session 的 PCM framing、VAD/KWS/ASR cache、sequence 幂等、匿名 voice group 和 buffer 生命周期。 | 原始 PCM 只在有上限的进程内 buffer；异常、超时、stop 后释放。 |

## evals/ Python 文件

| 文件 | 一眼看懂 | 什么时候看它 | 注意边界 |
| --- | --- | --- | --- |
| `evals/__init__.py` | `evals` 子包标记。 | 几乎不用改。 | 不放评测逻辑。 |
| `evals/longmemeval_adapter.py` | 读取并规范 LongMemEval JSON 数据。 | 接外部 LongMemEval 数据集、解析 question/session/answer 时看。 | 只适配数据，不跑 service。 |
| `evals/longmemeval_runner.py` | 把 LongMemEval 样本跑进当前 service 并出报告。 | 跑长记忆 benchmark、比较 timeline/chat history 导入模式时看。 | 使用临时 app home，不是产品 runtime。 |
| `evals/metrics.py` | 项目自定义 eval 的硬判指标。 | 改 `reply_contains`、`saved_contains`、debug path、latency 等断言时看。 | 不使用 LLM 自评，避免评测结果被模型解释带偏。 |
| `evals/report.py` | 项目自定义 eval 的 JSON/Markdown 报告生成。 | 改 eval 报告结构、失败块、性能表时看。 | 只生成报告，不跑场景。 |
| `evals/runner.py` | 离线 live eval harness，直接调用真实 `GlassesChatService`。 | 跑 `evals/scenarios.jsonl`、预置记忆/文档/timeline、模拟失败、安装 eval agent 时看。 | 不是 HTTP server；会用临时 `AI_GLASSES_HOME` 隔离数据。 |

## 常见接手路径示例

### 持续收音、VAD、ASR demo 怎么嫁接

可以把新 demo 当作“输入适配层”，不要让它直接写 `memory_store.py`。

推荐路径：

```text
持续收音
-> VAD 判断有人声
-> ASR 得到 transcript
-> 选择现有入口：
   1. 需要像用户聊天一样立即回答：调用 /api/chat 或 GlassesChatService.chat()
   2. 只是持续记录环境片段：调用 capture start/append/stop
   3. 已经整理成可导入条目：调用 import_memory_events()
-> 统一经过 text_cleaning / MemoryWriteCandidate / intent_policy / EventMemoryStore
```

这样嫁接的好处是：新音频 demo 只负责“把声音变成文本和 metadata”，本系统继续负责隐私、门控、去重、timeline evidence、memory job 和 audit。大白话说，就是新人只接一根“进水管”，不要自己在旁边另挖一个水库。

### 文档问答入口怎么找

先看 `document_helpers.py` 判断问题是不是文档问题、最近文档追问或跨文档比较；再看 `agent_bridge.py` 如何召回文档；最后看 `memory_store.py` 的 document 存储和搜索。

### 为什么没保存、为什么这么回答怎么查

先看 `explanation_helpers.py` 的解释类问题识别和回复格式；再看 `agent_bridge.py` 里上一轮 audit/job/timeline evidence 的读取；如果是来源摘要文案，再看 `source_summary_helpers.py`。

### 要改长期记忆写入策略怎么走

先看 `memory_candidate.py` 明白候选结构，再看 `intent_policy.py` 的最终门控，最后才看 `agent_bridge.py` 的 `_save_memory_candidates()` 编排。不要从 `memory_store.py` 直接下手绕过门控。

## 不要误解的边界

- `audio_processing.py` 是音频片段处理器，不是持续录音和 VAD runtime。
- `session_store.py` 存当前 demo 会话历史，不是用户长期记忆。
- `timeline_store.py` 存原话证据、capture chunk、job 和 speaker profile，不是结构化长期记忆。
- `memory_store.py` 是长期记忆和文档的 DB 层，但“该不该保存”由 service 和 `intent_policy.py` 决定。
- `conversation_candidate_helpers.py` 只负责多人 transcript 的结构、来源和隐私 extraction plan；业务语义仍交给统一语义分类，最终保存仍由 `intent_policy.py` 和 service 决定。
- `evals/` 是验证工具，不是产品运行路径。
- 仓库根目录的 `server.py` 只是兼容薄入口，真正 HTTP server 在包内 `server.py`。
