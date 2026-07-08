# 独立打包草案

本文是第十二刀产物：把第十一刀的 dependency manifest 草案进一步落成一个可讨论、可复制的独立打包草案。它仍然不是正式发布文件，也不代表当前目录已经完成迁出。

当前草案文件：

```text
pyproject.standalone.toml
```

注意：这个文件当前不会被外层 `hermes-agent` 自动使用。正式迁出时，应把它作为独立仓库的 `pyproject.toml` 起点，再按真实部署环境重新确认版本 pin、extras、README 和发布流程。

## 打包形态

当前草案按“仓库根目录保留项目入口和资源目录，`ai_glasses_memory_assistant/` 作为正式 Python 包目录”来写：

```toml
[tool.setuptools]
packages = [
  "ai_glasses_memory_assistant",
  "ai_glasses_memory_assistant.evals",
]
```

这表示核心 Python 模块位于 `ai_glasses_memory_assistant/` 下。这样从仓库根目录仍能使用：

```bash
python -m ai_glasses_memory_assistant.server
```

也可以使用 console script 草案：

```bash
ai-glasses-memory-assistant
```

FastAPI 可选入口草案：

```bash
ai-glasses-memory-assistant-fastapi
```

它需要安装 `fastapi` extra。

## 默认依赖

默认依赖只放当前主线必需项：

```text
openai
```

默认主线是：

```text
server.py
-> GlassesChatService
-> OpenAICompatibleLLMClient
```

标准库继续覆盖：

- `http.server`
- `sqlite3`
- `ssl`
- `urllib`
- 本项目 `env_loader.py` 的简单 `.env` 解析

## Optional Dependencies

草案分组：

| extra | 依赖 | 用途 |
| --- | --- | --- |
| `fastapi` | `fastapi`、`pydantic`、`uvicorn` | 只在使用 `app.py` / FastAPI 入口时需要。 |
| `tts` | `edge-tts` | 只在使用 Edge TTS 语音播报时需要。 |
| `voice` | `funasr` | 只在启用本地 ASR、声学情绪或声纹模型时需要。 |
| `dev` | `pytest` | 只在运行测试和文档一致性检查时需要。 |

对应环境变量：

```bash
AI_GLASSES_TTS_PROVIDER=edge
AI_GLASSES_TTS_VOICE=zh-CN-XiaoyiNeural
AI_GLASSES_ASR_MODEL_DIR=/path/to/asr-model
AI_GLASSES_EMOTION_MODEL_DIR=/path/to/emotion-model
AI_GLASSES_SPEAKER_MODEL_DIR=/path/to/speaker-model
```

## Package Data

第二阶段包结构迁移后，`static/`、`evals/scenarios.jsonl` 和 `docs/context/*.md` 暂时仍作为仓库根目录资源，而不是 Python 包内文件。

原因：

- `static/` 是标准库 server 和 FastAPI 入口都要服务的 Web UI。
- `evals/scenarios.jsonl` 是迁出前后做离线对比的核心场景文件。
- `docs/context/*.md` 是 Codex 后续接手和独立化迁移边界的主要上下文。

正式发布时需要单独决定这些资源是迁入包内，还是继续作为 wheel 外源码/部署资源存在；不要在未验证前声称它们已经作为 package data 进入安装产物。

## 第十三刀 dry-run 结果

本刀用临时目录验证 `pyproject.standalone.toml` 能否作为独立仓库 `pyproject.toml` 的起点。验证目录：

```text
/private/tmp/ai-glasses-packaging.XH6rEa
```

验证方法：

1. 复制 `ai_glasses_memory_assistant/` 到临时目录。
2. 删除临时副本中的 `.git`、`__pycache__` 等非打包产物。
3. 在临时副本中把 `pyproject.standalone.toml` 复制为 `pyproject.toml`。
4. 使用 `pip install --no-index --no-deps --no-build-isolation --target ...` 安装到临时 target，不联网下载依赖，也不污染当前 conda 环境。
5. 从安装 target 检查 metadata、console scripts、package data、默认 server import 和 eval runner import。

第一次 dry-run 发现的问题：

- `static/`、`evals/scenarios.jsonl` 和 `docs/context/*.md` 能作为 package data 进入安装 target。
- 但 `ai_glasses_memory_assistant.evals.runner` 不能 import。
- 原因是 `[tool.setuptools].packages` 只声明了 `ai_glasses_memory_assistant`，没有声明 `ai_glasses_memory_assistant.evals` 这个 Python 子包。

本刀已做的最小修复：

```toml
[tool.setuptools]
packages = [
  "ai_glasses_memory_assistant",
  "ai_glasses_memory_assistant.evals",
]
```

修复后第二次 dry-run 结论：

- 临时安装成功，package metadata 可解析。
- package data 覆盖 `static/index.html`、`static/styles.css`、`static/app.js`、`evals/scenarios.jsonl` 和 `docs/context/standalone-packaging.md`。
- console scripts 草案存在：
  - `ai-glasses-memory-assistant = ai_glasses_memory_assistant.server:main`
  - `ai-glasses-memory-assistant-fastapi = ai_glasses_memory_assistant.app:main`
- 默认 server 入口 `ai_glasses_memory_assistant.server:main` 可 import，并且 `main` 可调用。
- `ai_glasses_memory_assistant.evals.runner` 可 import。
- FastAPI 入口只做 `importlib` spec 检查，没有 import `app.py`，因此默认 dry-run 不会因为当前环境缺少 `fastapi` 失败。

第二阶段包结构迁移后，上述 dry-run 结果已经是历史记录：Python 模块现在位于内层 `ai_glasses_memory_assistant/` 包目录，`static/`、`evals/scenarios.jsonl` 和 `docs/context/*.md` 暂时仍留在仓库根目录。下一次正式打包前必须重新 dry-run，单独确认这些资源是否迁入包内或继续作为部署资源保留。

本刀限制：

- 没有发布包。
- 没有正式迁出目录。
- 没有联网安装依赖。
- 没有使用真实 API key。
- 没有跑 live 网络测试。
- 没有固化迁出前 eval baseline。
- 没有删除 Hermes fallback。

大白话：这次是把“包装盒草案”放到临时目录里试装了一次。发现盒子里漏装了 `evals/runner.py` 这条 Python 入口，于是补上 `ai_glasses_memory_assistant.evals` 子包声明；补完后，默认 server、eval runner、静态页面、场景文件和上下文文档都能从临时安装 target 里找到。

## README 草案同步点

本刀同步更新了 `README.md`，让独立部署用户能直接看到：

- `AI_GLASSES_HOME` 推荐路径。
- `$AI_GLASSES_HOME/.env` 示例。
- 默认 backend 是 `openai_compatible`。
- DeepSeek / OpenAI-compatible 配置方式。
- 标准库 `server.py` 是默认入口。
- FastAPI 是可选入口，需要额外依赖。
- Hermes fallback 是 legacy，不推荐新部署。
- 正式迁出前后必须做 eval baseline 对比。

## 当前仍不是正式迁出

本刀没有做：

- 不把 `pyproject.standalone.toml` 改名为正式 `pyproject.toml`。
- 不安装新依赖。
- 不删除 Hermes fallback。
- 不迁出目录。
- 不跑 live 网络或真实 API key 测试。
- 不跑或固化 eval baseline。

## 下一步

建议下一刀二选一：

1. 单独评估是否移除或彻底封存 `AI_GLASSES_LLM_BACKEND=hermes` legacy fallback。
2. 做正式迁出前的离线 eval baseline 方案和报告模板，保证后续迁出后能同场景对比。

正式迁出前仍要先跑迁出前离线 eval / target baseline，迁出后用同一批场景复跑对比。
