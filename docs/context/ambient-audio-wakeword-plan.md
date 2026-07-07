# 24 小时无感佩戴、多人数音频记忆与唤醒式问答规划

更新时间：2026-07-07。本文是专项产品/工程规划，不代表当前 demo 已实现后台常驻收音、真实 ASR、说话人分离或硬件唤醒词。当前代码真相仍以 `agent_bridge.py`、`timeline_store.py`、`memory_store.py`、`server.py`、`app.py` 和 `static/` 为准。

## 一句话定位

目标不是“自动写日记”，而是：

```text
用户授权下的 24 小时无感佩戴
-> 临时音频处理
-> ASR + 说话人分离
-> 以用户为主体的多人会话时间线
-> 多人事件、任务和关系上下文记忆
-> 唤醒词后的用户 query
-> 基于最近现场、长期记忆和参与人线索自然回答
```

用户现在能感受到的理想变化：

> 以前 AI 像一个等你输入问题的聊天框；这个方向完成后，它更像一个在场的随身记录员。它不只记“我说了什么”，还知道“我和谁聊了什么、谁答应了什么、我后续该跟谁复盘什么”。你叫它时，它能把刚才或过去某次多人对话重新串起来。

## 产品判断

屏幕抓取和文本清洗可以验证底层文本处理能力，但不能代表 AI 眼镜的日常核心体验。

| 维度 | 屏幕抓取/文本清洗 | 日常收音/多人现场问答 |
| --- | --- | --- |
| 输入来源 | 外部页面、桌面、App 文本 | 用户和周围人的现场言语 |
| 用户意图 | 不一定明确，很多只是路过信息 | 唤醒后 query 通常有明确表达需求 |
| 关键价值 | 去噪、提取事实、整理内容 | 分清谁说了什么、记住协作上下文、自然复盘 |
| 最大风险 | 把看到的内容误当成用户关心 | 把周围人的隐私误当成用户长期记忆 |
| 原文价值 | 可压缩为事实摘要 | 原话承载语气、委屈、玩笑、反讽和关系 |

核心原则：

> 屏幕文本可以压缩事实，但日常对话不能压扁说话人、关系和情绪。

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
- 多人会话级数据模型和联系人归因。
- 真实唤醒词检测。
- always-on audio 的权限、加密、删除和审计闭环。

因此本规划应指导后续拆解，不应被文档或汇报描述成“已经完成”。

## 目标能力

### 1. 授权后的 24 小时无感收音

长期目标是眼镜在用户授权后可以全天候接收外界音频，不要求每段对话都先说提示词或唤醒词。唤醒词只用于“主动问系统”，不是“开始记录”的前置条件。

产品上仍必须可见、可控、可暂停：

- 用户知道当前设备处于常驻收音或暂停状态。
- 用户可以按时间段、地点或会话删除记录。
- 用户可以关闭多人会话记忆，只保留主动对话模式。
- 第三方敏感内容默认不进入长期结构化记忆。

大白话：

```text
平时：眼镜像会议记录员，在授权范围内记录周围对话。
唤醒后：眼镜像助理，回答“刚才谁说了什么”“上次我和张三怎么约的”。
```

### 2. 可见的持续收音待机

Demo 打开后可以进入“收音待机态”，但必须可见、可暂停、可清空。

界面至少应表达：

- 麦克风开启/关闭状态。
- 当前是否在等待唤醒词。
- 最近缓存时长。
- 是否只做短期缓存。
- 一键暂停。
- 清空最近语境。

不建议做无提示自动开麦。产品要传达的是“在用户授权下保留最近语境”，不是“偷偷监控”。

### 3. 多人会话时间线

核心不是只判断 `speaker_hint=user|other|unknown`，而是把一段现场对话拆成多个人的 turn，并保留它们之间的上下文关系。

推荐最小结构：

```text
ConversationSession
  session_id
  user_id
  started_at / ended_at
  context_hint
  participants[]
  turns[]

ConversationTurn
  turn_id
  speaker_id
  speaker_role: user / known_person / unknown_speaker
  speaker_label: 用户 / 张三 / speaker_2
  raw_text
  start_time / end_time
  asr_confidence
  diarization_confidence
  emotion metadata
  privacy_flags
```

这里的“多人”仍以用户为主体：

- 不把旁人当成本系统的独立用户。
- 允许保存“用户和某个人聊过什么”。
- 允许保存“某个人在这次会话里承诺/提醒/反对了什么”。
- 用户后续可以问“我和某某聊过什么”“某某让我做什么”“那次会议谁负责什么”。
- 旁人的私人偏好、敏感信息和无关闲聊默认不沉淀为长期记忆。

示例：

```text
09:31 用户：下午三点我们去见客户。
09:32 张三：我带合同，你带方案。
09:33 用户：可以，那我负责 PPT。

长期候选：
用户和张三约定下午三点见客户；张三负责合同，用户负责 PPT。
```

### 4. 临时音频处理缓存

音频不是记忆本体，而是理解现场的临时传感器。

持续收音产生的音频片段应先进入临时音频缓存，同一段音频用于：

- ASR 生成原话或近原话文本。
- 说话人分离，识别不同人分别说了什么。
- 用户声纹比对，用于判断哪段是佩戴者本人。
- 已知联系人声纹或后续人工命名，用于把 `speaker_2` 归并为“张三”等联系人。
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
+ speaker_id / speaker_role / speaker_label
+ diarization_confidence / speaker_confidence
+ emotion_label / emotion_score / emotion_candidates
+ emotion_evidence / acoustic_features
+ audio_retention=discarded_after_processing
```

### 4.1 真实音频临时缓存生命周期

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
- `start_time` / `end_time` / `source_type` / `speaker_hint` / `speaker_id` / `speaker_role` / `speaker_label`。
- `asr_confidence`。
- `diarization_confidence`、`speaker_confidence`、`speaker_source`。
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
- 当前已实现真实音频临时处理入口 + 本地 ASR v1：支持接收 base64 音频片段及 mime type / duration 做大小、格式、时长校验，临时落地后立即删除；当前正式主链路已切换到本地 `SenseVoiceSmall` 生成 transcript，失败时才按策略回退到 `transcript_hint`，仍未接真实情绪模型。
- `/api/audio/segment/process` 只负责临时音频处理，成功后返回文本和 metadata；如果请求带 `capture_id`，只把派生文本和 metadata 写入 capture chunk。
- `/api/capture/append` 继续只接受文本和 metadata，是进入 raw timeline/capture chunk 的边界。
- audit、debug、长期 memory 只能记录处理状态、错误类型和 `audio_retention`，不能记录原始音频路径或内容。

清理规则：

- 成功：ASR 和情绪判断完成后立即删除原始音频，再返回派生结果。
- 情绪判断失败但 ASR 成功：删除原始音频，返回 ASR 文本，metadata 标记 `emotion.enabled=false` 和失败类型。
- ASR 失败：删除原始音频，不生成 ambient chunk，只返回可诊断错误类型。
- 超时：进入失败清理，不允许留下悬挂音频。
- 进程启动或异常恢复时，应扫描临时区并删除过期音频；扫描结果只记录数量和错误类型，不记录文件名。

### 5. 短期多人原话时间线

持续收音转写后，先进入短期多人原话缓存。短期缓存服务唤醒后回答，也作为长期多人事件抽取的候选来源，但不等同于长期记忆。

每段片段建议保留：

| 字段 | 含义 |
| --- | --- |
| `source_type` | `ambient_audio` / `wake_query` / `chat` / `app_audio_transcript` |
| `start_time` / `end_time` | 片段时间范围 |
| `raw_text` | 原话或近原话转写 |
| `clean_text` | 去噪后的辅助文本，不替代原话 |
| `speaker_hint` | 用户/周围人/未知等弱线索 |
| `speaker_id` | 稳定说话人 ID，例如 `user`、`person_zhangsan`、`speaker_2` |
| `speaker_role` | `user` / `known_person` / `unknown_speaker` |
| `speaker_label` | 用户可读名称，例如“我”“张三”“未知说话人 2” |
| `diarization_confidence` | 说话人分离置信度 |
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

### 6. 唤醒词后进入 query 模式

唤醒词之前：系统记录最近现场语境。  
唤醒词之后：用户的话作为 query。

回答时应检索：

- 最近 1-5 分钟原话时间线。
- 与 query 相关的 raw timeline evidence。
- 结构化长期记忆。
- 相关参与人和历史会话。
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

多人复盘示例：

```text
多人现场：
张三：我带合同，你带方案。
用户：可以，那我负责 PPT。
李四：报价别超过上次那版。

唤醒后 query：
刚才客户会前分工是什么？

理想回复：
刚才分工是：张三带合同，你负责 PPT，李四提醒报价不要超过上次那版。这里我会把它记成一次多人协作会话，而不是只记“你要做 PPT”。
```

### 7. 长期记忆严格门控

一直收音不等于一直永久记录。

默认策略：

- 周围人的话可以进入短期多人时间线，但不能默认进入用户长期记忆。
- 多人长期记忆必须以用户为主体，例如“用户和张三约定了某事”，而不是无边界保存张三自己的生活画像。
- 明显第三方隐私不进长期记忆。
- 敏感内容先脱敏，再决定是否保留证据。
- 只有和用户强相关、稳定、明确、有后续价值的信息才进入结构化长期记忆。
- 用户必须能按时间段删除或清空最近语境。
- 对未知说话人的长期保存要更保守；可先保存为会话事实，等用户后续命名或确认参与人。

长期记忆候选示例：

```text
原话：“服了，又开始甩锅。”
情绪：烦躁，强度 4
长期候选：用户最近对工作协作中的甩锅行为敏感
处理：可以作为候选，但需结合重复出现或用户后续明确表达再提高置信度
```

多人长期记忆候选示例：

```text
原话：
张三：“我带合同，你带方案。”
用户：“可以，那我负责 PPT。”

长期候选：
用户和张三约定客户拜访分工：张三带合同，用户准备 PPT。

处理：
可以保存，因为这是与用户强相关的协作任务，不是旁人的私人闲聊。
```

## 推荐架构

```text
麦克风/音频流
-> 本地权限和可见状态
-> VAD/断句
-> 临时音频缓存
-> ASR 生成原话文本
-> 说话人分离和用户声纹比对
-> 已知联系人归并或 unknown speaker 临时编号
-> emotion2vec 基础声学情绪识别
-> 可选语气特征提取
-> 删除原始音频
-> 短期多人原话时间线
-> 隐私脱敏和第三方隐私标记
-> 文本 + 上下文的轻量语义情绪判断
-> 唤醒词检测
-> 唤醒后 query 截取
-> 最近现场语境召回
-> 参与人和历史会话召回
-> 长期记忆召回
-> 情绪感知回答
-> 记忆候选抽取和门控
-> raw timeline / structured memory / audit
```

与现有模块的关系：

| 现有模块 | 后续角色 |
| --- | --- |
| `timeline_store.py` | 承接原话时间线和证据 chunk；后续需要扩展音频片段 metadata。 |
| `memory_store.py` | 继续保存结构化长期记忆，不直接保存所有环境原话；后续需要表达多人会话事实和参与人。 |
| `intent_policy.py` | 继续作为长期记忆 admission gate；需加强 ambient audio 和第三方隐私边界。 |
| `agent_bridge.py` | 唤醒后 query 的主编排入口；需在回答前注入最近现场语境、参与人和历史会话。 |
| `/api/capture/*` | 可作为 MVP 的持续转写文本入口，不等同于生产级音频 runtime。 |
| `static/` | 先用浏览器麦克风和按钮模拟收音待机、唤醒和 query。 |
| `evals/` | 新增 target 场景，验证多人归因、现场理解、隐私门控和情绪辅助回复。 |

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

### MVP 1：多人转写文本输入与会话记忆模型

目标：不急着接真实全天候音频，先用“已经转写好的多人文本”验证数据模型、归因和长期门控是否成立。

第一刀只处理类似下面的文本输入：

```text
[用户] 下午三点我们去见客户。
[张三] 我带合同，你带方案。
[用户] 可以，那我负责 PPT。
```

最小要验证：

- 能识别一段输入是多人会话，而不是用户一个人的口述。
- 能区分 `user`、`known_person`、`unknown_speaker`。
- 能把多人分工保存成“用户参与的会话事实/任务”，不是把旁人的闲聊写成用户画像。
- 能按参与人召回，例如“我和张三上次聊客户时怎么分工的？”
- Debug/audit 能解释为什么保存某条多人协作事实、为什么拒绝保存旁人隐私或无关闲聊。

这个 MVP 可以完全基于文本和 eval 完成，不依赖真实麦克风、diarization 模型或全天候 runtime。大白话：先证明“分清谁说了什么以后，系统应该怎么记”，再去接“音频里怎么分清谁说了什么”。

### MVP 2：轻量情绪元数据

目标：让回复不再像冷冰冰的摘要。

- 状态：已完成 target eval 收尾验证。当前通过 3 个 `ambient_emotion_metadata` target，验证了高情绪线索先共情、低置信情绪不强断言、以及“别记这个”只用于当前回答不写长期记忆。这里仍是模拟 metadata，不是 `emotion2vec` 或真实音频模型。
- 用 `emotion2vec_plus_base` 对临时音频片段做基础声学情绪识别，先按片段级 CPU 推理验证。
- 给短期片段增加 `emotion_label`、`emotion_score`、`emotion_candidates`、`emotion_model`、`emotion_intensity`、`emotion_evidence`、`audio_retention`。
- 情绪语义不只依赖音频标签，还要结合原话和上下文，避免把 `neutral` 误读成“用户真的没情绪”。
- 回答 prompt 使用情绪字段调整语气。
- Debug 面板展示情绪判断依据。
- 验证原始音频处理后删除，失败/超时也删除。
- eval 检查回复是否先接住情绪，再给建议。

### MVP 3：真实唤醒词与短期缓存策略

目标：从按钮模拟走向真实唤醒。

- 接入本地或浏览器可用的 wake word 方案。
- 明确缓存窗口、过期策略和清空机制。
- 增加暂停、恢复、清空 UI。
- audit 记录唤醒前后窗口。

### MVP 4：长期记忆门控增强

目标：避免 ambient audio 污染长期记忆。

- 给 `source_type=ambient_audio` 的候选设置更严格门控。
- 区分用户明确表达、周围人表达和不确定说话人。
- 第三方隐私默认不写长期记忆。
- 增加删除某一时间段原话的入口。

### MVP 5：真实多人音频 diarization 接入

目标：在文本模型和门控通过后，再把真实音频里的说话人分离接进来。

- 在 `/api/audio/segment/process` 的临时音频边界内接入 diarization。
- 把音频片段切成多个 `ConversationTurn`。
- 用用户声纹识别佩戴者本人。
- 对非用户说话人先用 `unknown_speaker`，允许后续用户命名或合并为联系人。
- 不把原始音频作为长期证据；长期证据仍是派生文本、时间、说话人和 metadata。

## 评测方向

不要只测文本清洗准确率，应新增现场理解 target。

| 场景 | 应验证 |
| --- | --- |
| 刚才那句话是否阴阳怪气 | 能引用最近原话和语气，而不是泛泛建议 |
| 多人客户会前分工 | 能保存“用户和张三/李四的协作事实”，不是只保存用户单句任务 |
| 按参与人复盘 | 用户问“我和张三上次聊了什么”时，能按参与人召回相关会话 |
| 旁人无关闲聊 | 不把旁人喜欢喝什么、空调冷等闲聊写成用户画像 |
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

> 24 小时无感佩戴、多人数音频记忆与唤醒式问答：专项规划见 `docs/context/ambient-audio-wakeword-plan.md`。按钮模拟唤醒、真实音频临时处理入口、本地 ASR v1 和基础声纹判定已有原型；当前仍未实现真实后台常驻收音、生产级 diarization、联系人归因和多人会话长期记忆模型。下一步如继续推进，优先做“多人转写文本输入 -> 会话模型 -> eval/门控”的最小切片，再接真实全天候音频。
