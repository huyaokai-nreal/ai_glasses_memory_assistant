# 独立启动说明

本文记录 `ai_glasses_memory_assistant` 当前推荐的独立启动路径。这里的“独立”指默认运行时不再依赖 Hermes `AIAgent`、`SessionDB`、`tools.web_tools` 或 Hermes env loader；源码当前仍在外层 `hermes-agent` checkout 内，尚未完成完全独立打包。

注意：能按本文启动不等于迁出完成。真正迁出到独立工程前后，必须用同一批离线 eval / target 场景做效果对比，确认记忆写入、召回、解释和 debug/audit 没有退化。

如果要准备独立仓库部署、依赖清单或文件搬家范围，继续看 `standalone-deployment.md`。本文只聚焦当前怎么启动和启动边界。

## 当前边界

默认路径：

```text
AI_GLASSES_HOME/.env 或当前进程环境变量
-> env_loader.load_app_dotenv()
-> agent_bridge._demo_llm_config()
-> OpenAICompatibleLLMClient
-> GlassesChatService
```

默认不触发：

- `run_agent.AIAgent`
- `hermes_cli.runtime_provider`
- `hermes_cli.env_loader.load_hermes_dotenv`
- Hermes `SessionDB`
- Hermes `tools.web_tools`

仍保留但已封存的迁移期边界：

- `AI_GLASSES_LLM_BACKEND=hermes` 只有再设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 时才会启用 Hermes fallback。
- `agent_bridge.py` 只在双开关 legacy fallback 分支内延迟 import Hermes runtime/env/home 相关模块。
- `evals/runner.py` 仍有少量 Hermes 兼容 import，用于迁移期 eval fallback。
- 当前启动命令仍建议在外层 `/Users/huyaokai/Desktop/workspace/hermes-agent` 执行，确保包路径和现有测试环境一致。

## Home 和数据目录

推荐设置：

```bash
export AI_GLASSES_HOME="$HOME/.ai-glasses-memory-assistant"
```

默认数据落点：

```text
$AI_GLASSES_HOME/data/events.db
$AI_GLASSES_HOME/data/timeline.db
$AI_GLASSES_HOME/data/sessions.db
$AI_GLASSES_HOME/data/chat_audit.jsonl
```

如果没有设置 `AI_GLASSES_HOME`，默认 home 是：

```text
~/.ai-glasses-memory-assistant
```

迁移期兼容：如果只设置了旧 `HERMES_HOME`，数据目录仍会落到：

```text
$HERMES_HOME/ai_glasses_memory_assistant
```

新部署不建议依赖这个旧路径。

## .env 配置

本项目会按顺序加载：

1. `$AI_GLASSES_HOME/.env`
2. 仓库根目录 `.env`，仅在未显式设置 `AI_GLASSES_HOME` 或迁移期 `HERMES_HOME` 时作为本地开发兜底

加载时不覆盖当前 shell 已经设置的环境变量。

推荐 DeepSeek / OpenAI-compatible 配置：

```bash
AI_GLASSES_LLM_BACKEND=openai_compatible
AI_GLASSES_LLM_PROVIDER=deepseek
AI_GLASSES_LLM_MODEL=deepseek-v4-flash
AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com
AI_GLASSES_LLM_API_KEY=your-api-key
AI_GLASSES_LLM_API_MODE=chat_completions
AI_GLASSES_LLM_REASONING_ENABLED=false
```

如果 provider 是 `deepseek`，也可以不写 `AI_GLASSES_LLM_API_KEY`，改用：

```bash
DEEPSEEK_API_KEY=your-api-key
```

必需项：

- `AI_GLASSES_LLM_MODEL`
- `AI_GLASSES_LLM_BASE_URL`
- `AI_GLASSES_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`

`AI_GLASSES_LLM_API_MODE` 当前只支持：

```text
chat_completions
```

## 启动方式

标准库 server 是当前默认入口：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

默认监听：

```text
http://127.0.0.1:8765
http://<mac-lan-ip>:8765
```

只允许本机访问：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.server --host 127.0.0.1 --port 8765
```

同局域网设备测试定位或浏览器语音时，优先使用 HTTPS：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.server \
  --certfile certs/cert.pem \
  --keyfile certs/key.pem
```

FastAPI 入口复用同一套 `GlassesChatService`：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.app
```

注意：FastAPI 入口需要运行环境安装 `fastapi`；标准库 `server.py` 是当前默认入口，也是独立抽离预检查优先验证的入口。

## Hermes fallback

Hermes fallback 只用于迁移期对照，不推荐新部署默认使用。第十四刀后它已经被封存为双开关 legacy 路径；只设置 `AI_GLASSES_LLM_BACKEND=hermes` 会报错，不会 import Hermes runtime。

显式开启：

```bash
AI_GLASSES_LLM_BACKEND=hermes
AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1
```

开启后会触发：

- Hermes `runtime_provider`
- Hermes env loader fallback
- `HermesLLMClient`
- `run_agent.AIAgent`

即使走 Hermes fallback，本项目仍传入自己的 `AppSessionStore`，并保持：

```text
skip_context_files=True
skip_memory=True
enabled_toolsets=[]
```

大白话：默认启动已经走本项目自己的 OpenAI-compatible 模型入口；Hermes fallback 现在被锁在第二道门后面，只有迁移期故意做旧后端对照时才打开，不是新部署主路径。

## 最小验证

不需要真实 API key 的静态和单元测试：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m pytest tests/test_env_loader.py tests/test_app_home.py tests/test_server_config.py -q
conda run -n hermes python -m pytest tests/test_agent_bridge_policy.py -k "new_session or hermes" -q
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/agent_bridge.py ai_glasses_memory_assistant/llm_client.py ai_glasses_memory_assistant/server.py ai_glasses_memory_assistant/app.py server.py app.py
```

真实启动需要可用 API key。缺少配置时，默认 OpenAI-compatible backend 会明确报错，而不是静默回退 Hermes。
