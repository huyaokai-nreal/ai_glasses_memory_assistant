# Demo 能力与边界

本文说明当前 demo 能展示什么、API 有哪些、哪些仍是边界。面向工程接手，不写市场化说明。

## 当前能展示

- Web 文字聊天。
- Web 端麦克风录音 + 本地 ASR 转写，作为眼镜语音入口的早期交互替身。
- 按钮模拟唤醒 MVP：开启收音待机后，前端录音片段会送到 `/api/audio/segment/process` 走本地 ASR，转写文本先进入最近语境；点击“唤醒提问”后，下一句语音作为 query 并引用最近现场原话。
- 用户声纹录入按钮 + 3 步固定短句校准弹窗；完成 3 段录入后会在本地生成 1 份 active speaker centroid，供后续 `cam++` 片段 embedding 做 `speaker_hint=user/other/unknown` 保守比对。
- TTS 播报。
- 当前位置作为单轮临时上下文。
- profile/event/assistant_preference 记忆写入和召回。
- raw timeline/chunk 原文保存和跨 session 原文回忆。
- 记忆列表、搜索、手动新增、软删除、显式彻底删除。
- timeline evidence 查看、原文搜索、批量软删除和可清理 chunk 的彻底删除。
- 回复优先后台记忆写入。
- 后台 memory job 状态查询。
- 文本/JSON 导入和 Markdown 文档归档，作为未来 App 文档/文本入口的后端雏形。
- continuous_capture API，作为长语音或会议转写文本进入统一管道的验证。
- 启发式周报草稿，作为自动总结/回顾能力的早期形态。
- 手动提醒候选检查。
- debug 面板和 audit 记录。
- live-LLM 离线 eval。

## 最终产品需要展示

最终产品不是单独的 Web chat，而是“无限 AI 眼镜 + 手机端 App”组合：

- 眼镜端负责低摩擦语音/环境输入，并在用户授权范围内捕捉日常片段。
- 眼镜端理想交互是可见的持续收音待机：短期保留最近原话、时间和轻量情绪线索，等待唤醒词后再根据 query 回答；专项规划见 `ambient-audio-wakeword-plan.md`。
- 手机 App 负责文档上传、音频输入、手动文本补充、记忆管理和结果查看。
- 所有输入进入同一套候选抽取、敏感门控、去重、写库和 audit 管道。
- 输出侧重点是总结、回忆、回顾、周报、提醒和基于个人上下文的问答。

当前 demo 只验证其中的后端主链路和部分 UI 形态，不能把这些最终形态写成已完成能力。

## 主要 API

| API | 能力 | 备注 |
| --- | --- | --- |
| `POST /api/chat` | 主聊天入口 | 执行 planner、召回、回复、写入、audit。 |
| `GET /api/memories` | 查看 active 记忆和文档 metadata | 按 `user_id` 隔离，不返回文档全文。 |
| `POST /api/memories` | 手动新增记忆 | 直接创建一条结构化记忆。 |
| `DELETE /api/memories/{id}` | 删除记忆 | 默认软删除当前用户记忆，并清理未被其他 active 记忆引用的 timeline 证据；`purge=true` 时彻底删除可清理 evidence 和相关 audit。 |
| `GET /api/documents/{id}` | 查看单个文档 | 返回 Markdown 原文供编辑。 |
| `PATCH /api/documents/{id}` | 编辑文档 | 可改文件名、标题、摘要、原文。 |
| `DELETE /api/documents/{id}` | 删除文档 | 默认软删除后不再进入 UI 和召回；`purge=true` 时物理删除文档记录。 |
| `GET /api/memory/search` | 搜索记忆 | FTS5 可用时走 FTS，否则 LIKE fallback。 |
| `GET /api/timeline/search` | 搜索原始 timeline chunks | 调试原文回忆用，按 `user_id` 隔离，只返回 active chunk。 |
| `GET /api/timeline/chunks` | 查看 timeline evidence chunk | 按 ids 返回 chunk、引用计数、active/retained 原因和是否可彻底删除。 |
| `DELETE /api/timeline/chunks` | 批量删除 timeline chunk | 默认软删除；`purge=true` 时只物理删除没有 retained 引用的 chunk。 |
| `GET /api/memory/jobs` | 查询后台写入 job | 公开状态会写入 SQLite，可在重启后恢复查询；不是可靠 worker 队列。 |
| `POST /api/memory/import` | 文本/JSON 导入 | 进入统一门控。 |
| `POST /api/capture/start` | 开始连续输入 | 返回 `capture_id`。 |
| `POST /api/capture/append` | 追加片段 | 写入进程内 capture，同时落 SQLite chunk，便于重启后 stop/import。 |
| `POST /api/capture/stop` | 停止并导入 | 汇总文本后走 import。 |
| `POST /api/speaker/enroll` | 录入当前用户校准样本 | 原始音频临时处理后立即删除，只在本地 SQLite 保存 pending sample 或 finalized centroid。 |
| `GET /api/speaker/profile` | 查看当前声纹校准状态 | 不返回原始 embedding。 |
| `GET /api/weekly-report` | 周报草稿 | 启发式项目分组。 |
| `GET /api/reminders/check` | 提醒候选检查 | 手动查询未来 24 小时 task。 |
| `GET /api/debug/audit` | 查看 audit | 本地诊断用。 |
| `POST /api/tts` | 语音合成 | 不可用时前端可回退。 |

`server.py` 和 `app.py` 应保持这些 API 的核心行为一致。

## 典型演示

```text
用户：我叫 jack。
助手：我记住了，你叫 jack。

用户：我喜欢安静靠窗的位置。
助手：知道了，以后订座我会优先考虑安静靠窗。

用户：以后订座位优先考虑什么？
助手：你偏好安静、靠窗的位置。
```

```text
用户：记一下明天下午 3 点和 Alex 开周会。
助手：我先记下，明天下午 3 点和 Alex 开周会。

用户：我接下来有什么安排？
助手：你接下来有一项安排：明天下午 3 点和 Alex 开周会。
```

## 前端体验边界

Web UI 采用主聊天区、当前记忆侧栏和可折叠 Debug 抽屉。Debug 只是前端展示层，仍读取同一份 `/api/chat` debug payload 和 `/api/debug/audit` 记录。

助手回复支持受限 Markdown 展示，原始 HTML 按文本处理；按钮 tooltip 只改善前端可用性，不改变 API。

麦克风录音和定位受浏览器权限、HTTPS/localhost 安全上下文和设备环境影响。

排查顺序：

1. 先看浏览器是否授权麦克风/定位。
2. 局域网设备测试定位/语音时优先 HTTPS。
3. 如果语音失败，先区分是麦克风权限/录音失败，还是后端本地 ASR 模型、音频格式或服务错误。
4. 再看 `/api/chat`、`/api/tts`、debug 和 audit。

## 不是当前能力

- 真实眼镜硬件、蓝牙、摄像头、传感器。
- 原生手机 App。
- 音频上传、ASR 转写和原始音频管理。
- 后台常驻主动监听。
- 真实唤醒词检测、真实音频临时缓存和完整 diarization runtime；当前已切到本地 `SenseVoiceSmall` 片段转写待机，并增加 3 段用户参考声纹校准 + `cam++` 相似度判定，以及“独立基础声学情绪模型优先、SenseVoice token fallback”的真实 emotion metadata，但仍不是完整 always-on audio，也还没做逐句多说话人重建。
- 主动提醒推送 runtime。
- 生产级账号、鉴权、多租户权限后台。
- 可靠持久任务队列。
- 完整项目知识图谱。
- 外部向量数据库。
- Hermes gateway 或 Feishu bot 替代实现。
- 完整 timeline 审计后台、复杂批量纠正 workflow 和 raw chunk 原文直接编辑。

## 可观测性

每轮应该尽量能看到：

- planner 判断。
- planner 结果。
- structured router 结果。
- 是否读 profile/event。
- 召回了哪些记忆。
- 是否写入 timeline、召回了哪些 raw chunk。
- 候选保存、拒绝、等待确认原因。
- memory job 状态。
- location 状态。
- web search 状态。
- LLM provider/model/api calls。
- stage timing。

如果用户问“为什么这次答错/慢/没记住/乱记了”，优先查：

```text
AI_GLASSES_HOME/data/chat_audit.jsonl
```

然后再读 `agent_bridge.py` 对应分支。

## 评估边界

active eval 是当前门禁；target eval 是已知缺口。

当前 target 主要包括：

- 公开素材长口述和更复杂长上下文 source 组合。
- App 音频输入进入统一记忆管道。
- 总结、回忆、回顾输出的证据追踪。
- 随口表达的后台价值提取。
- 主动提醒触发和抑制。

不要为了让 target 暂时通过而把未设计好的能力硬编码到主链路里。
