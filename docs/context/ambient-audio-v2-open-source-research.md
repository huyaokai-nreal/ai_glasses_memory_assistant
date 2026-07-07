# 持续收音改进 V2 开源方案调研

更新时间：2026-07-07。本文是阶段 B 调研结论，用于支撑 `ambient-audio-wakeword-plan.md` 后续拆解。它不是当前实现状态，也不代表已接入这些依赖。

## 结论摘要

推荐路线：

```text
先文本建模
-> 再本地片段级音频处理
-> 最后接真实 streaming/runtime
```

当前最稳妥的技术组合：

| 能力 | 推荐借鉴 | 理由 |
| --- | --- | --- |
| VAD / 断句 | Silero VAD；FunASR 动态 VAD 作中文链路备选 | Silero VAD 轻量、ONNX 可跑边缘端；FunASR 和现有 SenseVoice/FunASR 迁移线更近。 |
| ASR | 继续优先 SenseVoiceSmall / FunASR；WhisperX 只作长音频对齐参考 | 本机已配置 SenseVoiceSmall；FunASR 同时覆盖 ASR、时间戳、speaker diarization、emotion；WhisperX 强在 word-level timestamps 和长音频对齐，但依赖栈更重。 |
| 说话人分离 | 第一候选 pyannote.audio；中文/本地轻量链路继续观察 FunASR + cam++ | pyannote 是成熟 diarization 工具链；FunASR 已和本项目本地模型方向一致。 |
| 用户声纹 / speaker embedding | 继续使用 cam++；调研 WeSpeaker / SpeechBrain 作为替代或评测基线 | cam++ 已配置；WeSpeaker 和 SpeechBrain 都覆盖 speaker verification / embedding / diarization 相关能力。 |
| 唤醒词 | openWakeWord 作为真实 wake word 候选 | 它面向低复杂度 wake word 检测，支持流式音频和自定义唤醒词训练思路。 |
| 多人记忆建模 | 不直接套音频库；在本项目内建 `ConversationSession / ConversationTurn / Participant` 语义层 | 音频库只回答“谁在什么时候说话”，不回答“这件事如何以用户为主体写成记忆”。 |

核心判断：

- V2 不应该先接完整 always-on runtime；应先把多人转写文本的会话模型、归因、召回和门控跑通。
- 原始音频仍然随用随抛；可保留的是派生文本、时间戳、speaker metadata、情绪 metadata 和处理状态。
- 说话人分离和联系人命名要分两步：模型先产出 `speaker_0/speaker_1`，系统再通过用户声纹、历史聚类或人工命名映射到 `user/known_person/unknown_speaker`。

## 可借鉴做法

### 1. VAD / 断句

Silero VAD 适合作为真实持续收音第一候选：

- 支持 Python 和 ONNX runtime，模型轻，适合本地原型和未来边缘端。
- 只负责判断“哪里有人声”，不要让它决定记忆写入。
- 可用于把持续音频切成可处理片段，减少 ASR 空跑和 hallucination。

FunASR 的动态 VAD 适合作为中文链路备选：

- 如果后续继续沿 SenseVoice / FunASR 统一音频栈推进，可以优先评估 FunASR 内置 VAD + ASR + diarization 的一体化路径。
- 代价是更容易把多个能力绑在一个工具链里，需要 debug 字段把 VAD、ASR、speaker、emotion 分开记录。

建议：

```text
MVP 阶段：保留现有片段入口，补 VAD/断句 eval。
真实 runtime 前：比较 Silero VAD 和 FunASR VAD 在中文、多人、噪声、远场场景下的切段质量。
```

### 2. ASR / 时间戳

SenseVoiceSmall / FunASR 仍是本项目优先路线：

- 当前机器已经配置 SenseVoiceSmall 和 cam++，继续复用能降低集成风险。
- FunASR 提供 ASR、时间戳、speaker diarization、emotion、streaming 等相关能力，适合作为本地中文 demo 的主调研对象。

WhisperX 值得借鉴，但不建议第一刀替换：

- 它适合长音频、word-level timestamps、Whisper + alignment + diarization 的组合思路。
- 它依赖 faster-whisper、alignment model、pyannote token 等，第一刀会引入更多依赖和 GPU/显存约束。

建议：

```text
主线继续 SenseVoice/FunASR；
WhisperX 用作长音频对齐和 diarization 输出格式参考；
暂不把 WhisperX 作为默认 runtime。
```

### 3. 说话人分离 / diarization

pyannote.audio 是最值得重点借鉴的 diarization 方案：

- 它的核心输出就是“哪个时间段属于哪个 speaker”，正好对应 `ConversationTurn`。
- 社区版可本地运行，但需要接受 Hugging Face 模型条款和 token；这会影响离线部署和配置说明。
- 它不负责联系人语义命名，只给 `SPEAKER_00` 这类标签。

FunASR + cam++ 更贴近当前本地模型路径：

- FunASR 文档展示了 speaker id、timestamps、punctuation 一次性输出的思路。
- cam++ 已在本项目中作为用户声纹相似度判定基础，可继续用于 `user/other/unknown` 的保守门控。

建议：

```text
调研基线：pyannote.audio
本项目优先落地：FunASR/SenseVoice/cam++ 路线
对齐目标：都转成统一 ConversationTurn，不让上层依赖具体模型输出格式
```

### 4. Speaker embedding / 用户声纹 / 联系人聚类

当前 demo 已有 3 段用户声纹校准 + cam++ 相似度判定，因此 V2 不应推翻这条线。

WeSpeaker 和 SpeechBrain 的价值：

- WeSpeaker 面向 speaker embedding、speaker verification、diarization，有生产/研究双定位。
- SpeechBrain 覆盖 speaker recognition、ECAPA-TDNN、x-vector、diarization 等，可作为替代模型或评测基线。

V2 应采用两级身份策略：

| 层级 | 输出 | 说明 |
| --- | --- | --- |
| 声学身份 | `speaker_0 / speaker_1` | diarization 或 embedding 聚类产物，不代表真实联系人。 |
| 产品身份 | `user / known_person / unknown_speaker` | 由用户声纹、用户命名、历史聚类和确认流程决定。 |

不要让模型直接把未知人命名为“张三”。正确流程是：

```text
speaker_2 多次出现
-> 系统提示“这个人是否是张三？”
-> 用户确认
-> 后续映射为 known_person: 张三
```

### 5. Wake word

openWakeWord 适合作为真实唤醒词候选：

- 它面向 wake word / phrase detection，强调性能和简单性。
- 支持流式音频处理思路，也有 web streaming 示例。
- 唤醒词只决定“什么时候开始 query 模式”，不能决定“什么时候开始记录”。

建议：

```text
V2 保留唤醒词，但不把唤醒词作为持续记录的开关。
第一刀仍可保留按钮模拟唤醒；真实 wake word 放在多人会话模型和音频片段处理稳定之后。
```

## 推荐技术选型

### 近期默认

用于阶段 C 前几刀：

- 不新增音频大依赖。
- 先做多人转写文本输入和 `ConversationSession / ConversationTurn` 内部模型。
- 继续使用现有 `/api/capture/*` 和 timeline chunk 作为证据入口。
- 只新增 target eval，验证多人归因、旁人隐私拒写、按参与人召回。

### 中期候选

用于真实片段级音频：

- ASR：SenseVoiceSmall / FunASR。
- VAD：Silero VAD 或 FunASR 动态 VAD。
- 用户声纹：cam++。
- 基础情绪：SenseVoice token fallback + emotion2vec 按需评估。
- 输出统一成 `ConversationTurn`，不要让上层直接消费模型原始 JSON。

### 后期候选

用于更复杂的多人 diarization：

- pyannote.audio 作为 diarization baseline。
- WhisperX 作为长音频 word-level timestamp / alignment 参考。
- WeSpeaker / SpeechBrain 作为 speaker embedding 和聚类质量评测参考。

## 风险和约束

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| pyannote/WhisperX 依赖和 token 较重 | 可能拖慢本地 demo 集成 | 只作为 baseline/参考，不作为第一刀默认依赖。 |
| 重叠说话识别差 | 多人会话会丢 turn 或归错人 | debug 暴露 `diarization_confidence`，低置信只进原文时间线，不直接写长期。 |
| unknown speaker 被误命名 | 会污染联系人记忆 | 必须走用户确认或显式命名，不能模型自动命名。 |
| ASR 误听敏感信息 | 隐私风险 | 敏感内容先脱敏和拦截；原始音频不留；audit 只记错误类型和派生 metadata。 |
| 记录层和注入层混淆 | 再次出现“只能最近 6 条”的问题 | 保持 V2 四层边界：记录层长、注入层窄、长期记忆精。 |

## 阶段 C 前置步骤

阶段 B 完成后，阶段 C 应按下面顺序拆计划：

1. 已完成第一刀：文本级多人会话结构 `ConversationSession / ConversationTurn`，先不做 DB migration。
2. 已完成第一刀：`multi_speaker_labeled_transcript_target` 和单元测试覆盖多人分工、旁人闲聊、敏感误听、unknown speaker、按参与人复盘。
3. 已完成第一刀：service 层最小解析，从带 `[时间][speaker]` label 的文本生成会话 turns。
4. 已完成第一刀：写入门控只保存以用户为主体的多人事实，旁人偏好和 unknown/sensitive 片段只进 debug 拒绝解释。
5. 已完成第一刀：按参与人和主题走结构化事件 text search 复盘。
6. 已完成第二刀：支持 `[speaker]`、`speaker：`、`speaker:` 等明确文本 label；补齐无用户参与、unknown speaker、敏感夹杂、多参与人多任务和旁人私人偏好的 target/test；debug 暴露 parsed/candidate/rejected turn 边界。
7. 片段级音频适配：把 SenseVoice/FunASR/cam++ 输出映射为同一个 turn 模型。
8. 再评估 Silero VAD / pyannote / openWakeWord 是否接入 runtime。

第一/二刀边界：当前实现只支持明确 speaker label 的文本，不代表真实 VAD、ASR、diarization、speaker clustering、联系人命名或全天候音频 runtime 已完成。

## 参考入口

- Silero VAD: <https://github.com/snakers4/silero-vad>
- FunASR: <https://github.com/modelscope/FunASR>
- WhisperX: <https://github.com/m-bain/whisperX>
- pyannote.audio: <https://github.com/pyannote/pyannote-audio>
- WeSpeaker: <https://github.com/wenet-e2e/wespeaker>
- SpeechBrain: <https://github.com/speechbrain/speechbrain>
- openWakeWord: <https://github.com/dscripka/openWakeWord>
