# 独立部署与依赖清单草案

本文是第十一刀产物：给 `ai_glasses_memory_assistant` 将来迁出为独立仓库准备“搬家清单”。它只记录当前代码需要带走什么、默认怎么启动、依赖大致怎么分组，以及还有哪些迁出前检查。

注意：这不是正式打包结果，也不是已经完成独立仓库迁出。当前仍不新增 `pyproject.toml`，不生成锁文件，不删除 Hermes fallback，不跑 live 网络或真实 API key 测试。

第十二刀已经在 `pyproject.standalone.toml` 和 `standalone-packaging.md` 中补了独立打包草案。本文继续作为部署和依赖清单总览。

## 当前结论

默认推荐路径已经可以按本项目自己的边界理解：

```text
AI_GLASSES_HOME/.env
-> env_loader.load_app_dotenv()
-> agent_bridge._demo_llm_config()
-> OpenAICompatibleLLMClient
-> GlassesChatService
-> server.py
```

默认不需要：

- Hermes `run_agent.AIAgent`
- Hermes `hermes_cli.runtime_provider`
- Hermes `hermes_cli.env_loader`
- Hermes `hermes_state.SessionDB`
- Hermes `tools.web_tools`

仍然保留：

- `AI_GLASSES_LLM_BACKEND=hermes` legacy fallback，但第十四刀后必须同时设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 才能启用。
- `HERMES_HOME` 迁移期路径兼容。
- `evals/runner.py` 的旧 Hermes env fallback best-effort 加载。

大白话：第十刀证明“拎起来不会马上断线”，第十一刀写的是“箱子里要装什么、默认从哪个开关启动、哪些旧线还没剪”。

## 推荐默认入口

独立部署优先使用标准库 server：

```bash
python -m ai_glasses_memory_assistant.server
```

原因：

- `server.py` 使用 Python 标准库 HTTP server。
- 默认 demo 静态页、聊天、记忆查询、timeline、audit、TTS 路由都从这里进入。
- 第十刀抽离预检查已经验证默认 `server.py` 路径不需要 Hermes 主项目顶层模块。

FastAPI 入口是可选入口：

```bash
python -m ai_glasses_memory_assistant.app
```

注意：

- `app.py` 和 `server.py` 复用同一套 `GlassesChatService`。
- `app.py` 需要额外安装 `fastapi`、`pydantic`、`uvicorn`。
- 当前 `hermes` conda 环境缺 `fastapi` 时，FastAPI 入口不能导入；这是独立部署依赖问题，不是 Hermes 依赖问题。

## Home、数据和 .env

推荐新部署显式设置：

```bash
export AI_GLASSES_HOME="$HOME/.ai-glasses-memory-assistant"
```

默认数据文件：

```text
$AI_GLASSES_HOME/data/events.db
$AI_GLASSES_HOME/data/timeline.db
$AI_GLASSES_HOME/data/sessions.db
$AI_GLASSES_HOME/data/chat_audit.jsonl
```

推荐 `.env` 放在：

```text
$AI_GLASSES_HOME/.env
```

示例：

```bash
AI_GLASSES_LLM_BACKEND=openai_compatible
AI_GLASSES_LLM_PROVIDER=deepseek
AI_GLASSES_LLM_MODEL=deepseek-chat
AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com
AI_GLASSES_LLM_API_KEY=your-api-key
AI_GLASSES_LLM_API_MODE=chat_completions
AI_GLASSES_LLM_REASONING_ENABLED=false
```

如果 provider 是 `deepseek`，也可以用：

```bash
DEEPSEEK_API_KEY=your-api-key
```

新部署不推荐依赖：

```bash
AI_GLASSES_LLM_BACKEND=hermes
AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1
HERMES_HOME=...
```

这些仍是迁移期兼容开关，不是独立部署主路径。只设置 `AI_GLASSES_LLM_BACKEND=hermes` 不再足够启用 Hermes fallback。

## Dependency Manifest 草案

这里先写草案，不直接创建正式 `pyproject.toml`。正式迁出时应再按目标 Python 版本和部署方式锁定版本。

### 默认 demo 依赖

用于标准库 server + OpenAI-compatible LLM backend：

```text
openai
```

标准库覆盖：

- HTTP server：`http.server`
- SQLite：`sqlite3`
- `.env` 解析：本项目 `env_loader.py` 简单解析
- web search fallback：`urllib` + DuckDuckGo HTML fallback

### FastAPI 可选依赖

只在使用 `app.py` 时需要：

```text
fastapi
pydantic
uvicorn
```

### TTS 可选依赖

只在使用 Edge TTS 路径时需要：

```text
edge-tts
```

代码 import 名是 `edge_tts`，包名按安装习惯写 `edge-tts`。

### 本地 ASR / 情绪模型可选依赖

只在使用本地语音转写、声纹/情绪模型相关路径时需要：

```text
funasr
```

同时需要本地模型目录配置：

```bash
AI_GLASSES_ASR_MODEL_DIR=/path/to/asr-model
AI_GLASSES_EMOTION_MODEL_DIR=/path/to/emotion-model
AI_GLASSES_SPEAKER_MODEL_DIR=/path/to/speaker-model
```

### 测试依赖

用于单元测试和文档一致性检查：

```text
pytest
```

### 不属于新部署默认依赖

以下只属于 legacy / 迁移期 fallback，不应写进新部署默认依赖：

```text
hermes-agent
run_agent.AIAgent
hermes_cli.runtime_provider
hermes_cli.env_loader
hermes_state.SessionDB
tools.web_tools
```

## 独立仓库需要带走的文件

正式迁出时至少需要带走：

- `ai_glasses_memory_assistant/*.py`
- `ai_glasses_memory_assistant/static/`
- `ai_glasses_memory_assistant/evals/`
- `ai_glasses_memory_assistant/tests/`
- `ai_glasses_memory_assistant/docs/context/`
- `ai_glasses_memory_assistant/README.md`
- `ai_glasses_memory_assistant/AGENTS.md`
- `ai_glasses_memory_assistant/PLANS.md`
- `ai_glasses_memory_assistant/.gitignore`

不应该直接带走：

- `ai_glasses_memory_assistant/.git/`
- `__pycache__/`
- `reports/` 下临时或本机生成的 eval 报告，除非明确要作为迁出基线归档
- Hermes 主项目的 `run_agent.py`、`hermes_cli/`、`tools/`、`toolsets.py`

是否需要带走 `docs/html/` 和 `docs/reports/`，取决于独立仓库是否要保留汇报材料。它们不是默认启动必需文件。

## Hermes Fallback 策略

当前策略：

- 保留 `AI_GLASSES_LLM_BACKEND=hermes`，但必须搭配 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 才能启用。
- 明确标记为 sealed legacy / 迁移期备用。
- 新部署不推荐使用。
- 不把 Hermes fallback 写入默认依赖清单。

第十四刀已选择“彻底封存但不删除”。后续真正迁出前仍需要做最终删除决策：

1. 删除 Hermes fallback，并同步删除相关测试/文档旧边界。
2. 继续保留 sealed legacy adapter，但不写入默认依赖清单，也不推荐新部署使用。

本刀不做这个决定。

## 迁出前后 Eval 口子

正式迁出前必须保留可对比基线：

1. 在当前外层 `hermes-agent` checkout 跑离线 eval / target 场景。
2. 保存报告、通过率、`target_failed_turns`、关键 debug/audit 字段和代表性回复。
3. 迁出后用同一批场景、同一类配置复跑。
4. 对比记忆写入、召回、解释、隐私门控、web/debug 字段是否退化。

第十五刀已把 baseline 执行方案和报告模板固化到 `docs/context/standalone-eval-baseline.md`。

第十六刀已首次执行迁出前 baseline，并把报告落到：

```text
ai_glasses_memory_assistant/reports/standalone-migration-baseline/
```

索引 notes：

```text
pre-extraction-20260706-baseline-notes.md
```

当前迁出前快照不是全绿证明，而是后续迁出后对比用的老环境现状：核心 active 门禁 `active_failed_turns=23`，主线压力 `active_failed_turns=4` / `target_failed_turns=9`，target 快照 `target_failed_turns=3`。迁出后不要把这些既有失败算作迁出回归；重点看新增失败和 debug/audit 字段是否漂移。

## 下一步缺口

正式迁出前还缺：

- 当前 `main` 已绑定 GitHub `origin=git@github.com:huyaokai-nreal/ai_glasses_memory_assistant.git`；下一步应基于该 remote 做临时迁出或独立仓库预演，并记录迁出后复跑 baseline 的 commit / branch / remote。
- 独立仓库真实 `pyproject.toml` 或等价依赖文件。
- 用户向 README：安装、配置、启动、常见错误。
- 是否删除 Hermes fallback 的最终决策。
- 是否把测试里的旧 `HERMES_HOME` 隔离写法逐步替换成 `AI_GLASSES_HOME`。
- 迁出后复跑同一批 baseline 场景并生成对比报告。

大白话：现在写的是“搬家清单草稿”。下一刀可以开始准备真正的包装箱，比如 `pyproject.toml` 草案和独立 README；也可以先决定旧 Hermes 备用通道到底删不删。
