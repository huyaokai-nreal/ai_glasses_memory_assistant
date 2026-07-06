# 持续收音待机与唤醒式现场问答规划

更新时间：2026-06-15。本文是专项产品/工程规划，不代表当前 demo 已实现后台常驻收音、真实 ASR 或硬件唤醒词。当前代码真相仍以 `agent_bridge.py`、`timeline_store.py`、`memory_store.py`、`server.py`、`app.py` 和 `static/` 为准。

## 一句话定位

目标不是“自动写日记”，而是：

```text
可见的持续收音待机
-> 短期原话时间线
-> 唤醒词后的用户 query
-> 基于最近现场语境、长期记忆和情绪线索自然回答
```

用户现在能感受到的理想变化：

> 以前 AI 像一个等你输入问题的聊天框；这个方向完成后，它更像一个在场的同伴。你叫它时，它知道刚刚发生了什么，也能听出你当时为什么不爽、紧张或开心。

## 产品判断

屏幕抓取和文本清洗可以验证底层文本处理能力，但不能代表 AI 眼镜的日常核心体验。

| 维度 | 屏幕抓取/文本清洗 | 日常收音/现场问答 |
| --- | --- | --- |
| 输入来源 | 外部页面、桌面、App 文本 | 用户和周围人的现场言语 |
| 用户意图 | 不一定明确，很多只是路过信息 | 唤醒后 query 通常有明确表达需求 |
| 关键价值 | 去噪、提取事实、整理内容 | 记住现场、理解情绪、自然接话 |
| 最大风险 | 把看到的内容误当成用户关心 | 把周围人的隐私误当成用户长期记忆 |
| 原文价值 | 可压缩为事实摘要 | 原话承载语气、委屈、玩笑、反讽和关系 |

核心原则：

> 屏幕文本可以压缩事实，但日常对话不能压扁情绪。

## 当前边界

当前 demo 已有：

- Web 文字聊天和浏览器语音输入。
- raw timeline/chunk 原文保存。
- `/api/capture/start|append|stop` 连续输入 API。
- 按钮模拟唤醒 MVP：前端可开启收音待机，把浏览器 ASR final 文本写入 `source=ambient_audio_text` 的 capture chunk；点击“唤醒提问”后，下一句语音作为 query，并通过 `ambient_capture_id` 把最近现场原话注入受控上下文。
- 文本/JSON/Markdown 导入。
- 长输入 `continuous_capture` fast path。
- 结构化记忆写入门控、去重、召回和 audit。

当前 demo 没有：

- 后台常驻收音 runtime。
- 真实眼镜麦克风接入。
- 原始音频上传和音频证据管理。
- 生产级 ASR、断句、说话人分离。
- 真实唤醒词检测。
- always-on audio 的权限、加密、删除和审计闭环。

因此本规划应指导后续拆解，不应被文档或汇报描述成“已经完成”。

## 目标能力

### 1. 可见的持续收音待机

Demo 打开后可以进入“收音待机态”，但必须可见、可暂停、可清空。

界面至少应表达：

- 麦克风开启/关闭状态。
- 当前是否在等待唤醒词。
- 最近缓存时长。
- 是否只做短期缓存。
- 一键暂停。
- 清空最近语境。

不建议做无提示自动开麦。产品要传达的是“在用户授权下保留最近语境”，不是“偷偷监控”。

### 2. 临时音频处理缓存

音频不是记忆本体，而是理解现场的临时传感器。

持续收音产生的音频片段应先进入临时音频缓存，同一段音频用于：

- ASR 生成原话或近原话文本。
- `emotion2vec` / `emotion2vec+` 提取基础声学情绪标签和分数。
- 可选提取语速、音量、停顿、语调、叹气等可解释语气特征。

处理完成后，原始音频应自动删除。删除策略必须覆盖：

- ASR 和情绪判断都完成后立即删除。
- 任一处理失败后按失败路径删除。
- 超过最长缓存窗口后删除。
- 进程重启或异常恢复时清理遗留临时音频。
- 不把原始音频写入 `audit`、debug payload 或长期存储。

后续真正进入系统的是：

```text
raw_text + time + source_type + speaker_hint
+ emotion_label / emotion_score / emotion_candidates
+ emotion_evidence / acoustic_features
+ audio_retention=discarded_after_processing
```

### 2.1 真实音频临时缓存生命周期

真实音频入口上线前，必须先把生命周期边界定死：原始音频只是临时传感器输入，不是记忆证据本体。

推荐状态机：

```text
received
-> processing
-> extracted
-> discarded

received / processing
-> failed
-> discarded
```

每个状态只做一件事：

| 状态 | 含义 | 唯一下一步 |
| --- | --- | --- |
| `received` | 服务端刚收到一段临时音频，只允许放在受控临时区。 | `processing` |
| `processing` | 正在执行 ASR、情绪判断或语气特征提取。 | 成功到 `extracted`，失败到 `failed` |
| `extracted` | 已拿到可保留的派生结果，例如文本、时间、来源、情绪 metadata。 | `discarded` |
| `failed` | ASR、情绪判断、格式校验或超时失败。 | `discarded` |
| `discarded` | 原始音频已删除，只允许保留派生结果或失败摘要。 | 终态 |

允许长期或短期文本系统继续使用的派生结果只有：

- ASR 原话或近原话文本。
- `start_time` / `end_time` / `source_type` / `speaker_hint`。
- `asr_confidence`。
- `emotion_label`、`emotion_intensity`、`emotion_score`、`emotion_candidates`、`emotion_evidence`。
- `acoustic_features` 的可解释摘要。
- `audio_retention=discarded_after_processing` 或 `discarded_after_failure`。

不得保留或传播：

- 原始音频二进制。
- 临时音频文件路径。
- 临时文件名、对象存储 key、录音设备底层标识。
- 能反推出音频存储位置的 debug 字段。

接口边界建议：

- 真实音频入口独立设计为 `/api/audio/segment/process`，不把原始音频塞进 `/api/capture/append`。
AI_GLASSES_ASR_MODEL_DIR- 当前已实现真实音频临时处理入口 + 本地 ASR v1：支持接收 base64 音频片段及 mime type / duration 做大小、格式、时长校验，临时落地后立即删除；当前正式主链路已切换到本地 `SenseVoiceSmall` 生成 transcript，失败时才按策略回退到 `transcript_hint`，仍未接真实情绪模型。
- `/api/audio/segment/process` 只负责临时音频处理，成功后返回文本和 metadata；如果请求带 `capture_id`，只把派生文本和 metadata 写入 capture chunk。
- `/api/capture/append` 继续只接受文本和 metadata，是进入 raw timeline/capture chunk 的边界。
- audit、debug、长期 memory 只能记录处理状态、错误类型和 `audio_retention`，不能记录原始音频路径或内容。

清理规则：

- 成功：ASR 和情绪判断完成后立即删除原始音频，再返回派生结果。
- 情绪判断失败但 ASR 成功：删除原始音频，返回 ASR 文本，metadata 标记 `emotion.enabled=false` 和失败类型。
- ASR 失败：删除原始音频，不生成 ambient chunk，只返回可诊断错误类型。
- 超时：进入失败清理，不允许留下悬挂音频。
- 进程启动或异常恢复时，应扫描临时区并删除过期音频；扫描结果只记录数量和错误类型，不记录文件名。

### 3. 短期原话时间线

持续收音转写后，先进入短期原话缓存。短期缓存服务唤醒后回答，不等同于长期记忆。

每段片段建议保留：

| 字段 | 含义 |
| --- | --- |
| `source_type` | `ambient_audio` / `wake_query` / `chat` / `app_audio_transcript` |
| `start_time` / `end_time` | 片段时间范围 |
| `raw_text` | 原话或近原话转写 |
| `clean_text` | 去噪后的辅助文本，不替代原话 |
| `speaker_hint` | 用户/周围人/未知等弱线索 |
| `asr_confidence` | 转写置信度 |
| `emotion_label` | 烦躁、委屈、开心、紧张、平静、不确定等轻量标签 |
| `emotion_intensity` | 1-5，作为回复辅助信号 |
| `emotion_model` | 情绪来源，例如 `emotion2vec_plus_base`、`llm_contextual_emotion` |
| `emotion_score` | 情绪模型 top label 置信分数 |
| `emotion_candidates` | 情绪模型 top candidates，例如 `neutral/sad/angry` 及分数 |
| `emotion_evidence` | 支撑情绪判断的词、语气或上下文 |
| `acoustic_features` | 可选语气特征，例如语速、音量、停顿、语调、叹气 |
| `audio_retention` | 原始音频处理结果，例如 `discarded_after_processing` |
| `importance` | 是否值得进入后续抽取 |
| `privacy_flags` | 敏感、第三方隐私、疑似误听等 |

### 4. 唤醒词后进入 query 模式

唤醒词之前：系统记录最近现场语境。  
唤醒词之后：用户的话作为 query。

回答时应检索：

- 最近 1-5 分钟原话时间线。
- 与 query 相关的 raw timeline evidence。
- 结构化长期记忆。
- 当前情绪状态。
- 必要时的文档或 App 导入内容。

示例：

```text
环境片段：
A：这个你应该早就知道吧。
用户：行，知道了。
用户小声说：服了，又来了。

唤醒后 query：
你觉得刚才他是不是在阴阳我？

理想回复：
我觉得你不爽的点主要是那句“你应该早就知道吧”，听起来像是在把责任压给你。再加上你后面说“服了，又来了”，这不像单纯讨论事情，更像你已经被这种语气烦过很多次了。
```

### 5. 长期记忆严格门控

一直收音不等于一直永久记录。

默认策略：

- 周围人的话可以进入短期时间线，但不能默认进入用户长期记忆。
- 明显第三方隐私不进长期记忆。
- 敏感内容先脱敏，再决定是否保留证据。
- 只有和用户强相关、稳定、明确、有后续价值的信息才进入结构化长期记忆。
- 用户必须能按时间段删除或清空最近语境。

长期记忆候选示例：

```text
原话：“服了，又开始甩锅。”
情绪：烦躁，强度 4
长期候选：用户最近对工作协作中的甩锅行为敏感
处理：可以作为候选，但需结合重复出现或用户后续明确表达再提高置信度
```

## 推荐架构

```text
麦克风/音频流
-> 本地权限和可见状态
-> VAD/断句
-> 临时音频缓存
-> ASR 生成原话文本
-> emotion2vec 基础声学情绪识别
-> 可选语气特征提取
-> 删除原始音频
-> 短期原话时间线
-> 隐私脱敏和第三方隐私标记
-> 文本 + 上下文的轻量语义情绪判断
-> 唤醒词检测
-> 唤醒后 query 截取
-> 最近现场语境召回
-> 长期记忆召回
-> 情绪感知回答
-> 记忆候选抽取和门控
-> raw timeline / structured memory / audit
```

与现有模块的关系：

| 现有模块 | 后续角色 |
| --- | --- |
| `timeline_store.py` | 承接原话时间线和证据 chunk；后续需要扩展音频片段 metadata。 |
| `memory_store.py` | 继续保存结构化长期记忆，不直接保存所有环境原话。 |
| `intent_policy.py` | 继续作为长期记忆 admission gate；需加强 ambient audio 和第三方隐私边界。 |
| `agent_bridge.py` | 唤醒后 query 的主编排入口；需在回答前注入最近现场语境。 |
| `/api/capture/*` | 可作为 MVP 的持续转写文本入口，不等同于生产级音频 runtime。 |
| `static/` | 先用浏览器麦克风和按钮模拟收音待机、唤醒和 query。 |
| `evals/` | 新增 target 场景，验证现场理解和情绪辅助回复。 |

## 情绪判断怎么加

需要加入，但先做轻量元数据，不做心理诊断。

情绪判断分两层：

| 层级 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| 基础声学情绪 | 临时音频片段 | `emotion2vec` / `emotion2vec+` 输出的 `angry`、`happy`、`neutral`、`sad`、`surprised`、`unknown` 等标签和分数 | 给出声音层面的情绪信号。 |
| 语义/上下文情绪 | 原话文本、最近现场、声学情绪 | 委屈、敷衍、压着火、阴阳怪气、松了一口气等更贴近日常对话的判断 | 帮助回答更像正常人，但必须保留不确定性。 |

第一版可优先评估 `emotion2vec_plus_base`，在 M2 Pro 本地按片段级 CPU 推理验证；`emotion2vec_plus_large` 更重，后续再评估。不建议 MVP 做逐帧实时情绪识别，先在 VAD/断句后的 1-10 秒片段上跑一次基础情绪判断。

情绪判断的职责：

- 帮助回复选择语气。
- 解释用户为什么可能在意某句话。
- 辅助判断某个片段是否与用户强相关。

情绪判断不应该：

- 替代原话。
- 单独成为长期结论。
- 给用户贴稳定人格标签。
- 对严肃心理健康问题给医疗化判断。

推荐最小字段：

```text
emotion_label: 烦躁 / 委屈 / 开心 / 紧张 / 平静 / 不确定
emotion_intensity: 1-5
emotion_model: emotion2vec_plus_base / llm_contextual_emotion
emotion_score: 0.72
emotion_candidates: [{"label": "neutral", "score": 0.55}, {"label": "sad", "score": 0.28}]
emotion_evidence: 支撑判断的原话片段、声学特征或上下文
audio_retention: discarded_after_processing
```

## MVP 拆解

### MVP 0：按钮模拟唤醒

状态：已完成可信闭环加固。当前实现只处理浏览器 ASR 文本，不保存原始音频，不接真实 wake word，不启用情绪模型。

目标：不接真实唤醒词，先验证“最近原话 + 唤醒后 query”是否成立。

- 打开页面后请求麦克风权限。
- 授权后显示“收音待机态”。
- 浏览器语音或手动输入片段进入短期缓存。
- 用户点击“唤醒”按钮。
- 按钮后的下一句话作为 query。
- 回答时注入最近 3-5 分钟原话。
- 不自动写长期记忆，只产出候选和 debug。

当前实现边界：

- ambient 片段只进入 raw timeline/capture chunk，不会因为待机本身自动导入长期记忆。
- 唤醒后的 query 仍是正常 `/api/chat`，会按现有聊天规则写入 raw timeline 和后台记忆门控。
- `debug.ambient_context` 会展示 capture 是否使用、chunk 数、chunk ids、是否注入主 LLM，以及情绪模型未启用。
- 只有 `running` 状态的 ambient capture 会作为唤醒现场语境注入；已停止或跨用户的 capture 只在 debug 中显示状态，不进入主回复上下文。
- 前端清空只清空本页最近语境引用，不物理删除已进入 raw timeline 的 capture chunk。

验收例子：

> 用户问“刚才那个人是不是有点阴阳怪气？”时，系统能引用最近原话，而不是泛泛回答。

### MVP 1：轻量情绪元数据

目标：让回复不再像冷冰冰的摘要。

- 状态：已完成 target eval 收尾验证。当前通过 3 个 `ambient_emotion_metadata` target，验证了高情绪线索先共情、低置信情绪不强断言、以及“别记这个”只用于当前回答不写长期记忆。这里仍是模拟 metadata，不是 `emotion2vec` 或真实音频模型。
- 用 `emotion2vec_plus_base` 对临时音频片段做基础声学情绪识别，先按片段级 CPU 推理验证。
- 给短期片段增加 `emotion_label`、`emotion_score`、`emotion_candidates`、`emotion_model`、`emotion_intensity`、`emotion_evidence`、`audio_retention`。
- 情绪语义不只依赖音频标签，还要结合原话和上下文，避免把 `neutral` 误读成“用户真的没情绪”。
- 回答 prompt 使用情绪字段调整语气。
- Debug 面板展示情绪判断依据。
- 验证原始音频处理后删除，失败/超时也删除。
- eval 检查回复是否先接住情绪，再给建议。

### MVP 2：真实唤醒词与短期缓存策略

目标：从按钮模拟走向真实唤醒。

- 接入本地或浏览器可用的 wake word 方案。
- 明确缓存窗口、过期策略和清空机制。
- 增加暂停、恢复、清空 UI。
- audit 记录唤醒前后窗口。

### MVP 3：长期记忆门控增强

目标：避免 ambient audio 污染长期记忆。

- 给 `source_type=ambient_audio` 的候选设置更严格门控。
- 区分用户明确表达、周围人表达和不确定说话人。
- 第三方隐私默认不写长期记忆。
- 增加删除某一时间段原话的入口。

## 评测方向

不要只测文本清洗准确率，应新增现场理解 target。

| 场景 | 应验证 |
| --- | --- |
| 刚才那句话是否阴阳怪气 | 能引用最近原话和语气，而不是泛泛建议 |
| 用户小声吐槽后唤醒 | 能识别用户真实情绪，不把“行吧”当成平静同意 |
| 音频随用随抛 | ASR 和 `emotion2vec` 处理完成后原始音频删除，只保留文本、时间和情绪结果 |
| 情绪模型低置信度 | 低分或 `unknown` 时回答保持谨慎，不强行断言用户情绪 |
| 周围人说出隐私 | 不进入长期记忆，必要时脱敏 |
| 用户明确说“别记这个” | 短期可用于当前回答，但不写长期 |
| ASR 噪声误听敏感信息 | 不写长期，debug/audit 可解释 |
| 唤醒词前后切分 | query 只取唤醒后的用户问题，回答可引用唤醒前现场 |

## 公开依据

这些公开资源可用于支撑产品论证和后续调研，不代表当前代码已引用或依赖。

| 资源 | 类型 | 支持点 |
| --- | --- | --- |
| LifeDialBench | 论文/benchmark | 连续生活场景记忆是独立问题，不能简单等同普通聊天或屏幕清洗；高保真上下文对 lifelog 记忆很重要。 |
| Memoro | 论文/系统 | 可穿戴音频记忆增强和低打扰现场辅助是合理方向。 |
| ES-MemEval | 论文/评测 | 长期个性化情绪支持需要记忆，不只是单轮聊天。 |
| FeatureSense | 论文/隐私 | always-on audio 必须有隐私保护、信任和边界设计。 |
| emotion2vec / emotion2vec+ | 开源语音情绪识别模型 | 可作为本地片段级基础声学情绪识别候选；输出标签和分数，不等同于完整人类语境情绪。 |
| openWakeWord / Howl | 开源 wake word | 可作为真实唤醒词检测参考。 |
| whisper_streaming | 开源 ASR | 可作为实时转写参考。 |

## 和 `PLANS.md` 的关系

`PLANS.md` 只保留本规划的入口和当前优先级，不展开全部细节。后续如果进入实现阶段，每个最小切片可以在 `PLANS.md` 写一行进度，并链接回本文。

建议入口文案：

> 持续收音待机与唤醒式现场问答：专项规划见 `docs/context/ambient-audio-wakeword-plan.md`。按钮模拟唤醒 MVP 已完成第一版；当前仍未实现真实常驻收音/ASR/唤醒词和 emotion2vec。下一步如继续推进，优先补轻量情绪元数据 target eval 和真实音频临时处理缓存设计。
