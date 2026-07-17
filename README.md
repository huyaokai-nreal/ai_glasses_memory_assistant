# AI 眼镜个人记忆助手

这是一个本地 Web/语音原型，用来验证 AI 眼镜个人记忆助手的核心闭环：

```text
自然输入
-> 先回复用户
-> 按需沉淀长期记忆
-> 后续按需召回画像、事件、任务、文档和原话证据
-> 通过 debug / audit 解释为什么读、写或拒绝保存
```

## 一图看懂系统

![AI 眼镜个人记忆助手全系统 Pipeline](docs/context/assets/system-overview-pipeline.png)

这张图是新开发者的第一入口：先理解一轮回复和记忆门控，再到 `docs/context/system-flow-current.md` 查看可维护的统一音频处理核心 SVG 数据流图。

当前代码是真相。第一次接手开发建议按下面顺序读：

1. `AGENTS.md`
2. `PLANS.md`
3. `CONTRIBUTING.md`
4. `docs/context/README.md`
5. `docs/context/code-map.md`
6. `docs/context/system-flow-current.md`

## 当前定位

当前是 Stage 2 可运行原型，已经具备：

- Web 文字聊天、按体验者 ID 隔离的浏览器流式语音、TTS、定位、debug 面板。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- 启发式周报草稿和手动提醒候选检查。
- 统一音频 session：同一体验者单 active session、16 kHz PCM、32 ms VAD、KWS-only 两段式唤醒、partial/final ASR、TTS 播放保活、ambient capture、声纹参考和匿名 voice group。

当前不是：

- 生产级硬件眼镜 runtime。
- 原生手机 App。
- always-on audio runtime。
- 生产级 diarization 或联系人归因系统。
- 主动提醒推送系统。
- 可靠跨进程 worker 队列。
- 多租户生产服务。

## 目录

| 路径 | 作用 |
| --- | --- |
| `ai_glasses_memory_assistant/` | 正式 Python 包代码。 |
| `server.py` | 根目录兼容薄入口，转到包内标准库 HTTP server。 |
| `static/` | Web UI、语音、TTS、定位、debug、job 轮询。 |
| `tests/` | 核心保险丝测试：启动配置、存储、聊天主链路。 |
| `evals/` | eval 场景和 runner。 |
| `docs/context/` | 只保留三份当前态开发文档。 |
| `CONTRIBUTING.md` | 私人多人协作流程：分支、验证、提交前检查。 |
| `.env.example` | 本地配置模板，不含真实密钥。 |
| `certs/` | 本地 HTTPS 测试证书，已被 `.gitignore` 忽略，每个开发者自己生成，不提交到 Git。 |
| `reports/` | eval 运行产物。 |

## 文档

`docs/context` 只保留三份文档：

| 文件 | 用途 |
| --- | --- |
| `docs/context/README.md` | 开发者上手入口：启动、配置、依赖、测试和 LLM 边界。 |
| `docs/context/code-map.md` | 按任务找代码入口。 |
| `docs/context/system-flow-current.md` | 当前系统架构和真实调用链。 |

这些文档只写当前态，不写历史开发过程。

## 快速运行

下面命令里的 `/path/to/ai_glasses_memory_assistant` 请替换成你本机 clone 的仓库根目录。

推荐先设置本项目自己的 home：

```bash
export AI_GLASSES_HOME="$HOME/.ai-glasses-memory-assistant"
```

默认数据位置：

```text
$AI_GLASSES_HOME/data/events.db
$AI_GLASSES_HOME/data/timeline.db
$AI_GLASSES_HOME/data/sessions.db
$AI_GLASSES_HOME/data/chat_audit.jsonl
```

启动默认标准库 server：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

默认访问：

```text
http://127.0.0.1:8765
```

本机浏览器应使用这个 loopback 地址；它可以在 HTTP 下使用麦克风。若启动提示 `port is already used`，先用 `lsof -nP -iTCP:8765 -sTCP:LISTEN` 找到并停止旧 server，或显式传入其他 `--port`。不要让 HTTP 和 HTTPS server 同时占用 8765，否则同一个端口可能按访问地址命中不同协议。

局域网设备测试语音或定位时使用 HTTPS：

`http://<Mac 局域网 IP>:8765` 可以展示普通页面，但浏览器会禁止麦克风和定位，因此“开启全天待机”不能工作。启动 HTTPS 后，请使用启动日志输出的 `https://<Mac 局域网 IP>:8765`。

`certs/` 只放本机自签名证书，已被 `.gitignore` 忽略。每个开发者在自己机器上生成一份即可，不要把 `cert.pem` 或 `key.pem` 提交到 Git。

```bash
cd /path/to/ai_glasses_memory_assistant
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 \
  -subj "/CN=localhost"
```

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server --certfile certs/cert.pem --keyfile certs/key.pem
```

## LLM 配置

推荐把配置写到：

```text
$AI_GLASSES_HOME/.env
```

可以从示例文件开始：

```bash
mkdir -p "$AI_GLASSES_HOME"
cp .env.example "$AI_GLASSES_HOME/.env"
```

默认使用 DeepSeek/OpenAI-compatible API：

```bash
AI_GLASSES_LLM_PROVIDER=deepseek
AI_GLASSES_LLM_MODEL=deepseek-v4-flash
AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com
AI_GLASSES_LLM_API_KEY=<your-api-key>
AI_GLASSES_LLM_API_MODE=chat_completions
```

如果 `AI_GLASSES_LLM_PROVIDER=deepseek` 且未设置 `AI_GLASSES_LLM_API_KEY`，运行时会尝试读取 `DEEPSEEK_API_KEY`。缺少 model、base URL 或 API key 时应直接报错。

主 LLM runtime 不再保留本地模型默认值或 legacy fallback。

## 依赖

`pyproject.toml` 是当前 Python 包配置。默认依赖只有 `openai`；可选能力按 extra 分组：

| extra | 依赖 | 用途 |
| --- | --- | --- |
| `tts` | `edge-tts` | 可选语音播报 |
| `voice` | `funasr`、`torch`、`numpy`、`soundfile` | 可选本地整段/流式 ASR、声学情绪、声纹模型 |
| `voice-stream` | `silero-vad`、`sherpa-onnx` | 可选流式 VAD 和 KWS；未配置时 capability 明确降级 |
| `dev` | `pytest` | 测试 |

正式发布或部署前必须重新检查依赖 pin、package data 和安装流程。

## 主要 API

| API | 作用 |
| --- | --- |
| `POST /api/chat` | 主聊天入口。 |
| `GET /api/runtime` | 返回前端运行时信息。 |
| `GET /api/memories` | 查看记忆和文档 metadata。 |
| `POST /api/memories` | 手动新增记忆。 |
| `DELETE /api/memories/{memory_id}` | 删除记忆。 |
| `GET /api/documents/{document_id}` | 查看文档。 |
| `PATCH /api/documents/{document_id}` | 编辑文档 metadata 或原文。 |
| `DELETE /api/documents/{document_id}` | 删除文档。 |
| `GET /api/memory/search` | 搜索结构化记忆。 |
| `GET /api/timeline/search` | 搜索原话 timeline。 |
| `GET /api/timeline/chunks` | 查看 evidence chunks。 |
| `DELETE /api/timeline/chunks` | 删除 timeline chunks。 |
| `GET /api/memory/jobs` | 查询后台记忆写入 job。 |
| `POST /api/memory/import` | 导入文本或 JSON。 |
| `POST /api/capture/start|append|stop` | 连续输入采集。 |
| `GET /api/audio/capabilities` | 查看收音、全天转写、唤醒问答、声纹录入及各后端公开状态，不返回模型路径。 |
| `POST /api/audio/session/start|push|control|stop` | 统一流式 PCM 音频 session；stop 支持无损 `pause_for_enrollment`。 |
| `GET /api/audio/dispatch/jobs` | 查询 final 创建的异步聊天 job 和最终回答。 |
| `POST /api/audio/segment/process` | 处理音频片段。 |
| `POST /api/speaker/enroll` | 录入声纹参考。 |
| `GET /api/speaker/groups` | 查看匿名 voice group 元数据，不返回 embedding。 |
| `DELETE /api/speaker/groups/{group_id}` | 删除指定体验者的匿名声纹样本。 |
| `GET /api/weekly-report` | 启发式周报草稿。 |
| `GET /api/reminders/check` | 手动检查提醒候选。 |
| `GET /api/debug/audit` | 查看 audit。 |
| `POST /api/tts` | 生成语音音频。 |

## 验证

文档或小改动：

```bash
cd /path/to/ai_glasses_memory_assistant
git diff --check
```

Python 改动至少跑：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

默认单元测试包括核心保险丝和 fake backend 音频契约：`tests/test_core_startup.py`、`tests/test_core_storage.py`、`tests/test_core_chat.py`、`tests/test_audio_engine.py`。

## 流式音频模型

`AI_GLASSES_STREAMING_ASR_MODEL_DIR` 指向 Paraformer streaming 模型目录。未配置时不会下载模型：ambient 仍可用 SenseVoice 逐段转写，但语音唤醒问答会明确显示不可用，不提供手动降级入口。

KWS 模型必须放在仓库外。下载示例：

```bash
curl -L -o sherpa-kws.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
tar -xjf sherpa-kws.tar.bz2 -C /path/outside/repository
```

解压后再配置 `AI_GLASSES_KWS_MODEL_DIR` 和 `AI_GLASSES_KWS_KEYWORDS_FILE`。未配置或依赖缺失时 `/api/audio/capabilities` 返回 `degraded`，网页继续支持 ambient 记录，但会显示“语音唤醒问答不可用”。唤醒词命中后，助手先播放 `AI_GLASSES_WAKE_ACK_TEXT`（默认“我在”），再等待下一段问题语音。
