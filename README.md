# AI 眼镜个人记忆助手

这是一个本地 Web/语音原型，用来验证 AI 眼镜个人记忆助手的核心闭环：

```text
自然输入
-> 先回复用户
-> 按需沉淀长期记忆
-> 后续按需召回画像、事件、任务、文档和原话证据
-> 通过 debug / audit 解释为什么读、写或拒绝保存
```

当前代码是真相。第一次接手开发建议按下面顺序读：

1. `AGENTS.md`
2. `PLANS.md`
3. `docs/context/README.md`
4. `docs/context/code-map.md`
5. `docs/context/system-flow-current.md`

## 当前定位

当前是 Stage 2 可运行原型，已经具备：

- Web 文字聊天、浏览器语音、TTS、定位、debug 面板。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- 启发式周报草稿和手动提醒候选检查。
- 真实音频片段处理入口、本地 ASR v1、基础情绪 metadata、声纹参考和 `speaker_hint`。

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
| `tests/` | 单元测试和 service 级测试。 |
| `evals/` | eval 场景和 runner。 |
| `docs/context/` | 只保留三份当前态开发文档。 |
| `certs/` | 本地 HTTPS 测试证书，已被 `.gitignore` 忽略，每个开发者自己生成，不提交到 Git。 |
| `reports/` | eval 运行产物。 |

## 文档

`docs/context` 只保留三份文档：

| 文件 | 用途 |
| --- | --- |
| `docs/context/README.md` | 开发者上手入口：启动、配置、依赖、测试、legacy fallback 边界。 |
| `docs/context/code-map.md` | 按任务找代码入口。 |
| `docs/context/system-flow-current.md` | 当前系统架构和真实调用链。 |

这些文档只写当前态，不写历史开发过程。

## 快速运行

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
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

默认访问：

```text
http://127.0.0.1:8765
```

局域网设备测试语音或定位时使用 HTTPS：

`certs/` 只放本机自签名证书，已被 `.gitignore` 忽略。每个开发者在自己机器上生成一份即可，不要把 `cert.pem` 或 `key.pem` 提交到公共仓库。

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 \
  -subj "/CN=localhost"
```

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server --certfile certs/cert.pem --keyfile certs/key.pem
```

## LLM 配置

推荐把配置写到：

```text
$AI_GLASSES_HOME/.env
```

默认建议使用 OpenAI-compatible backend：

```bash
AI_GLASSES_LLM_BACKEND=openai_compatible
AI_GLASSES_LLM_PROVIDER=deepseek
AI_GLASSES_LLM_MODEL=deepseek-v4-flash
AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com
AI_GLASSES_LLM_API_KEY=<your-api-key>
AI_GLASSES_LLM_API_MODE=chat_completions
AI_GLASSES_LLM_REASONING_ENABLED=false
```

如果 `AI_GLASSES_LLM_PROVIDER=deepseek` 且未设置 `AI_GLASSES_LLM_API_KEY`，运行时会尝试读取 `DEEPSEEK_API_KEY`。缺少 model、base URL 或 API key 时应直接报错。

Hermes backend 只作为 legacy fallback，必须显式双开关：

```bash
export AI_GLASSES_LLM_BACKEND=hermes
export AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1
```

新开发和新部署默认走 `openai_compatible`。

## 依赖草案

`pyproject.standalone.toml` 是独立打包草案，不是正式发布配置。默认依赖只有 `openai`；可选能力按 extra 分组：

| extra | 依赖 | 用途 |
| --- | --- | --- |
| `tts` | `edge-tts` | 可选语音播报 |
| `voice` | `funasr` | 可选本地 ASR、声学情绪、声纹模型 |
| `dev` | `pytest` | 测试 |

正式打包前必须重新 dry-run。

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
| `POST /api/audio/segment/process` | 处理音频片段。 |
| `POST /api/speaker/enroll` | 录入声纹参考。 |
| `GET /api/weekly-report` | 启发式周报草稿。 |
| `GET /api/reminders/check` | 手动检查提醒候选。 |
| `GET /api/debug/audit` | 查看 audit。 |
| `POST /api/tts` | 生成语音音频。 |

## 验证

文档或小改动：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
git diff --check
```

Python 改动至少跑：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m unittest discover tests -q
```

文档一致性测试：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m pytest tests/test_startup_docs.py -q
```
