# 独立化迁移清单

本文记录 `ai_glasses_memory_assistant` 从 `hermes-agent` 内嵌 demo 迁移为独立产品前的准备信息。当前阶段只做依赖识别和迁移边界说明，不改变运行代码。

## 目标

长期目标：

- demo 可以作为独立仓库、独立部署、独立开源项目维护。
- 日常开发只需要关注 `ai_glasses_memory_assistant` 自身。
- 不再要求从 `hermes-agent` 父目录启动。
- 不再依赖 Hermes 的 home、LLM agent、session、toolset 或 env loader。

当前低风险目标：

- 先列清 Hermes 依赖点。
- 区分低风险基础设施替换和高风险效果相关替换。
- 给后续高推理强度架构迁移留下明确入口。

## 当前 Hermes 依赖点

| 依赖点 | 当前位置 | 当前作用 | 风险等级 | 后续替换方向 |
| --- | --- | --- | --- | --- |
| `hermes_constants.get_hermes_home` | `agent_bridge.py`、`evals/runner.py` | 两处都已改成延迟 import；`agent_bridge.py` 只在双开关 sealed Hermes fallback 分支使用，`evals/runner.py` 只作为迁移期旧 env fallback 尝试加载 | 中 | 移除 Hermes fallback 或重做 eval env fallback 时再删除。 |
| `hermes_state.SessionDB` | 已移除直接 import | 第五刀后由本项目 `session_store.py::AppSessionStore` 提供最小 session store；默认 OpenAI-compatible backend 不依赖它，Hermes fallback 会收到本项目 store | 低 | 后续移除 Hermes fallback 后，再评估是否仍需要保留 `sessions.db`。 |
| `hermes_cli.env_loader.load_hermes_dotenv` | `agent_bridge.py`、`evals/runner.py` | 延迟 import；`agent_bridge.py` 第十四刀后只在 `AI_GLASSES_LLM_BACKEND=hermes` 且 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 时使用，`evals/runner.py` 仍作为迁移期旧 env fallback best-effort 加载 | 中 | 删除 Hermes fallback 或重做 eval env fallback 时再删除。 |
| `hermes_cli.runtime_provider.resolve_runtime_provider` | `agent_bridge.py` | 第十四刀后仅在双开关 sealed Hermes fallback 下使用 | 中 | 后续如果不再需要旧后端对照，可删除 Hermes fallback。 |
| `run_agent.AIAgent` | `llm_client.py` | 第四刀后仅由 `HermesLLMClient` fallback 持有；默认主模型已走本项目 OpenAI-compatible client | 中 | 后续如果 OpenAI-compatible backend 稳定，可移除 Hermes fallback。 |
| `tools.web_tools` | 已移除默认依赖 | 第六刀后 `agent_bridge.py` 不再 import 或优先调用 Hermes web tools；实时搜索收口到本项目 `web_search.py` | 低 | 后续如需增强搜索能力，应继续作为本项目能力接入，不回挂 Hermes toolset。 |
| `HERMES_HOME` 文档和测试约定 | `tests/` 以及少量历史文档 | 迁移期测试隔离和旧数据路径兼容 | 低到中 | `AI_GLASSES_HOME` 已作为新主路径；旧 `HERMES_HOME` 仅保留为过渡 fallback。 |

## 已完成切片

### home/data 第一刀

已新增 `app_home.py`：

- `AI_GLASSES_HOME` 优先生效。
- 未设置 `AI_GLASSES_HOME` 时，默认使用 `~/.ai-glasses-memory-assistant/data`。
- 迁移期如果只设置旧 `HERMES_HOME`，仍兼容使用 `HERMES_HOME/ai_glasses_memory_assistant`，避免旧测试和旧本地数据立刻失效。

已切换到本项目 data dir 的位置：

- `memory_store.py` 默认 `events.db`。
- `timeline_store.py` 默认 `timeline.db`。
- `agent_bridge.py` 的 `chat_audit.jsonl` 和 `sessions.db`。
- `evals/runner.py` 的临时隔离目录改为 `AI_GLASSES_HOME`。

未改变：

- Hermes `AIAgent`。
- `SessionDB` 类本身。
- Hermes provider/env loader。
- web search。
- SQLite 表结构、记忆写入、召回和解释逻辑。

### env loader 第二刀

已新增 `env_loader.py`：

- 优先加载 `AI_GLASSES_HOME/.env`。
- 再加载包目录 `ai_glasses_memory_assistant/.env`，用于本地开发兜底。
- 不覆盖当前进程已经存在的环境变量。
- 不引入新依赖，不读取 Hermes `config.yaml`。

当前保持兼容的 LLM 环境变量：

- `AI_GLASSES_LLM_PROVIDER`
- `AI_GLASSES_LLM_MODEL`
- `AI_GLASSES_LLM_BASE_URL`
- `AI_GLASSES_LLM_API_KEY`
- `AI_GLASSES_LLM_API_MODE`
- `AI_GLASSES_LLM_REASONING_ENABLED`

已接入的位置：

- `agent_bridge.py::_new_session()`：创建 Hermes `AIAgent` 前先加载本项目 `.env`，然后仍保留 Hermes `.env` loader 作为迁移期 provider key fallback。
- `evals/runner.py`：live eval 启动时先加载本项目 `.env`，再保留 Hermes `.env` fallback。

未改变：

- Hermes `AIAgent` 调用方式。
- `runtime_provider` 解析方式。
- `SessionDB`。
- web search。
- 记忆写入、召回、解释和 SQLite schema。

大白话：这一刀只是让“AI 眼镜 demo 自己去哪里找 `.env`”先独立出来，还没有把“怎么调用大模型”从 Hermes 里拆出来。

### LLMClient 对照层第三刀

已新增 `llm_client.py`：

- 定义本项目最小 `LLMClient` 协议，只覆盖当前真实需要的 `run_conversation()`。
- 新增 `HermesLLMClient`，内部仍持有 Hermes `AIAgent`。
- 新增 `create_hermes_llm_client()`，集中构造 Hermes `AIAgent`。
- `agent_bridge.py::_new_session()` 改为通过本项目工厂创建 client，不再直接写 `AIAgent(...)`。

兼容边界：

- 运行时仍然使用 Hermes `AIAgent`，这一步不代表已经移除 Hermes LLM 依赖。
- `HermesLLMClient` 透传 `run_conversation()`、`chat()` 和未知属性读写，确保现有 `session.agent` 调用、debug runtime 字段、回调计时和 eval patch 逻辑不变。
- `AIAgent` 参数保持原含义：`model`、`provider`、`base_url`、`api_key`、`api_mode`、`reasoning_config`、`quiet_mode=True`、`platform="ai_glasses_web"`、`session_id`、`session_db`、`user_id`、`skip_context_files=True`、`skip_memory=True`、`enabled_toolsets=[]`、`ephemeral_system_prompt=AI_GLASSES_SYSTEM_PROMPT`。

未改变：

- OpenAI-compatible backend 尚未实现。
- Hermes `runtime_provider` 仍在使用。
- `SessionDB` 仍在使用。
- web search 当时仍未迁移。
- 记忆写入、召回、解释和 SQLite schema 未改变。

大白话：这一刀只是先把“模型调用插口”变成本项目自己的名字，但插口背后还是接着 Hermes。这样下一刀替换后端时，有一个清楚的替换位置。

### OpenAI-compatible backend 第四刀

已在 `llm_client.py` 新增 `OpenAICompatibleLLMClient`：

- 默认 backend 改为 `openai_compatible`。
- 当前主验证目标是 DeepSeek `deepseek-v4-flash`。
- `agent_bridge.py::_new_session()` 默认不再调用 Hermes `AIAgent`、`runtime_provider` 或 Hermes `.env` loader。
- 只有显式设置 `AI_GLASSES_LLM_BACKEND=hermes` 时，才走第三刀保留的 `HermesLLMClient` fallback。

默认 OpenAI-compatible 配置：

- `AI_GLASSES_LLM_PROVIDER=deepseek`
- `AI_GLASSES_LLM_MODEL=deepseek-v4-flash`
- `AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com`
- `AI_GLASSES_LLM_API_KEY=<your key>`，或在 provider 为 `deepseek` 时使用 `DEEPSEEK_API_KEY`
- `AI_GLASSES_LLM_API_MODE=chat_completions`

缺少 `model`、`base_url` 或 API key 时会清晰报错，不再静默回退 Hermes。

第四刀当时未改变：

- `SessionDB` 仍在使用。
- web search 当时仍未迁移。
- 记忆写入、召回、解释和 SQLite schema 未改变。

大白话：这一刀开始让默认主模型调用真正走本项目自己的 OpenAI-compatible client；Hermes 还在旁边留作手动应急通道。

### Session store 第五刀

已新增 `session_store.py`：

- 新增 `AppSessionStore`，使用本项目 `data/sessions.db`。
- `agent_bridge.py` 不再直接 import `hermes_state.SessionDB`。
- 默认 OpenAI-compatible backend 本来不依赖 session store；显式设置 `AI_GLASSES_LLM_BACKEND=hermes` 时，Hermes `AIAgent` fallback 会收到本项目自己的最小 session store。

当前最小能力：

- 创建 session。
- 记录 message。
- 读取 session / messages。
- 更新 system prompt。
- 记录 token 和 API call 计数。
- 保留 `title/end/reopen` 等 Hermes fallback 可能触达的轻量兼容方法。

未改变：

- 多轮对话 history 拼接仍由现有 `ChatSession` / LLM client 返回 messages 维护。
- 记忆写入、召回、解释和 SQLite memory/timeline schema 未改变。
- web search 当时仍未迁移。
- Hermes `AIAgent` fallback、`runtime_provider` 和 Hermes `.env` loader fallback 仍保留。

大白话：这一刀不是重做“聊天记忆”，只是把原来挂在 Hermes 身上的会话数据库换成本项目自己的最小记录本；默认 DeepSeek/OpenAI-compatible 路径本来就不靠 Hermes `SessionDB`。

### web search 第六刀

已完善 `web_search.py`：

- 新增本项目自己的 `search_web()` 搜索边界。
- 继续复用原有无 key、无新依赖的 DuckDuckGo HTML fallback。
- 新增 `WebSearchResponse`，统一保留搜索结果和注入主模型的上下文文本格式。
- `agent_bridge.py::_maybe_search_web()` 不再 import `tools.web_tools`，也不再优先使用 Hermes web search。

兼容边界：

- `debug.tools[]` 仍使用 `name="web_search"`。
- `query`、`reason`、`results_count`、`results` 等字段继续保留，便于旧 audit/debug/eval 读法延续。
- 天气、地点、定位相关的 route 决策仍由 `PreReplyDecision` / planner 落地；这一刀不改 `web_query`、`web_reason`、`needs_location` 和 weather debug 策略。
- 测试环境不依赖 live 网络，使用 stub 搜索结果验证模块边界。

未改变：

- 不改 LLM backend。
- 不改 session store。
- 不移除 Hermes `AIAgent` fallback。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- Hermes `runtime_provider` 和 Hermes `.env` loader fallback 仍保留。

大白话：这一刀只是把“要联网查一下”这件事从 Hermes 工具箱里拿出来，改成本项目自己的小搜索入口；搜索失败时仍只是少一段实时上下文，不会让整轮聊天、记忆保存或定位保护崩掉。

### 独立启动验证第七刀

已验证默认启动边界：

- `GlassesChatService._new_session()` 会先加载本项目 `.env`，再读取 `_demo_llm_config()`。
- 默认 `AI_GLASSES_LLM_BACKEND` 是 `openai_compatible`。
- 默认 OpenAI-compatible 分支只创建本项目 `OpenAICompatibleLLMClient`，不调用 Hermes `AIAgent`、`runtime_provider` 或 Hermes env loader。
- `AI_GLASSES_HOME` 下的 `data/` 能承载 `chat_audit.jsonl` 和 `sessions.db`，不需要 Hermes `SessionDB`。
- `server.py` 和 `app.py` 都只是薄启动入口，核心行为仍进入同一个 `GlassesChatService`；启动入口本身不需要 Hermes toolset。

显式 fallback 边界：

- 只有设置 `AI_GLASSES_LLM_BACKEND=hermes` 时，`agent_bridge.py::_new_session()` 才 import `hermes_cli.runtime_provider` 和 `hermes_cli.env_loader`，并通过 `create_hermes_llm_client()` 创建 Hermes `AIAgent`。
- Hermes fallback 会收到本项目 `AppSessionStore`，并继续禁用 `skip_context_files=True`、`skip_memory=True`、`enabled_toolsets=[]`。
- Hermes env loader fallback 只服务这个显式 Hermes backend，加载后会恢复本项目 `AI_GLASSES_LLM_*` 环境变量，避免旧 Hermes `.env` 覆盖 app-owned 配置。

未改变：

- 没有删除 `HermesLLMClient`。
- 没有删除 `run_agent.AIAgent` fallback。
- 没有删除 Hermes `runtime_provider` / Hermes env loader fallback。
- 没有修改 LLM backend API、记忆写入、召回、解释逻辑、web search 或 SQLite schema。

大白话：第七刀不是拆拐杖，而是确认“默认走路不用拐杖”；只有用户明确把 `AI_GLASSES_LLM_BACKEND` 设成 `hermes`，系统才会去拿那根 Hermes 备用拐杖。

### 独立启动说明第八刀

已新增 `docs/context/standalone-startup.md`：

- 写清推荐 `AI_GLASSES_HOME` 和默认 data/audit/session 落点。
- 写清本项目 `.env` 加载顺序：`$AI_GLASSES_HOME/.env` 优先，包目录 `.env` 兜底，且不覆盖当前 shell 环境变量。
- 写清默认 backend 是 `openai_compatible`，以及 DeepSeek / OpenAI-compatible 必需配置。
- 写清标准库 server 与 FastAPI 两个启动入口。
- 写清 Hermes fallback 是 legacy/迁移期备用，只有 `AI_GLASSES_LLM_BACKEND=hermes` 才触发，不推荐新部署默认使用。

已补轻量文档一致性测试：

- `tests/test_standalone_startup_docs.py` 会检查文档里的 `AI_GLASSES_LLM_*`、backend 名称、`DEEPSEEK_API_KEY` 和启动命令是否仍匹配代码常量与入口。
- 测试不需要 live 网络或真实 API key。

未改变：

- 不删除 Hermes fallback。
- 不改 LLM backend API。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- 不改 web search。
- 不把当前状态写成已经完成独立打包；当前仍建议从外层 `hermes-agent` checkout 启动。

大白话：第八刀是给已经验证过的默认启动路径写一份“说明书”，并加一个小检查，防止以后变量名或启动命令改了，说明书还停在旧版本。

### Hermes fallback 边界冻结第九刀

已冻结默认路径和 Hermes fallback 的 import/初始化边界：

- `agent_bridge.py` 不再顶层 import `hermes_constants.get_hermes_home`。
- 默认 `openai_compatible` 分支仍只创建本项目 `OpenAICompatibleLLMClient`。
- `hermes_constants.get_hermes_home`、`hermes_cli.runtime_provider`、Hermes env loader 和 `run_agent.AIAgent` 都只允许在显式 `AI_GLASSES_LLM_BACKEND=hermes` 分支内触发。
- 新增 `test_agent_bridge_has_no_top_level_hermes_fallback_imports`，用静态 AST 检查防止 `hermes_constants`、`hermes_cli` 或 `run_agent` 回到 `agent_bridge.py` 顶层 import。

未改变：

- 不删除 `HermesLLMClient`。
- 不删除 `run_agent.AIAgent` legacy fallback。
- 不删除 Hermes `runtime_provider` / Hermes env loader fallback。
- 不改 LLM backend API、记忆写入、召回、解释逻辑、web search 或 SQLite schema。
- 不迁出目录，也不代表独立打包已经完成。

大白话：第九刀不是拔掉 Hermes 备用通道，而是把它锁到一扇明确的门后面；平时默认启动不会先摸这扇门，只有用户显式写 `AI_GLASSES_LLM_BACKEND=hermes` 才会打开。

### 仓库抽离预检查第十刀

仓库抽离预检查已完成，历史过程文档已删除；当前结论以本文和测试为准。

本刀检查结论：

- `rg` 静态清点显示 `ai_glasses_memory_assistant` 下已经没有 Hermes 主项目模块的顶层 import。
- `evals/runner.py` 的 `hermes_constants` / `hermes_cli.env_loader` 从顶层 import 改成延迟 best-effort helper；没有 Hermes 主项目时可以导入 runner。
- 临时复制 `ai_glasses_memory_assistant` 到 `/private/tmp` 后，用 `python -S` 禁用 site 初始化，默认 `agent_bridge.py` / `server.py` 可以导入。
- 同样的硬抽离预演下，`evals/runner.py` 可以导入，关键入口文件可以 `py_compile`。
- `app.py` 的 FastAPI 入口在当前 `hermes` conda 环境缺 `fastapi` 时不能导入；这是独立部署依赖问题，不是 Hermes 主项目依赖问题。标准库 `server.py` 是当前默认入口，不受影响。

剩余边界：

- 默认路径：未发现 Hermes 主项目顶层依赖。
- legacy fallback：`AI_GLASSES_LLM_BACKEND=hermes` 仍会触发 Hermes `AIAgent` / runtime_provider / env loader / home 兼容。
- eval/test 迁移期：旧 Hermes `.env` fallback 已可缺省；测试中仍有大量 `HERMES_HOME` 作为旧隔离写法，后续可逐步替换为 `AI_GLASSES_HOME`。

大白话：第十刀是“搬家前体检”。现在把包拎到临时目录，默认 server 和 eval runner 不会因为找不到 Hermes 主项目马上断；真正搬家前还要补独立依赖清单、README、baseline 对比，以及决定 Hermes fallback 是否删除。

### 独立部署文档与 dependency manifest 草案第十一刀

已新增 `docs/context/standalone-deployment.md`，把独立仓库部署前需要确认的内容先写成草案：

- 推荐默认入口是 `python -m ai_glasses_memory_assistant.server`。
- FastAPI 入口 `python -m ai_glasses_memory_assistant.app` 是可选入口，需要 `fastapi`、`pydantic`、`uvicorn`。
- 默认 demo 依赖草案先收口到 `openai`，标准库继续覆盖 HTTP server、SQLite、`.env` 简单解析和 DuckDuckGo HTML fallback。
- 可选依赖分组写清：`edge-tts` 用于 TTS，`funasr` 用于本地 ASR / 情绪模型，`pytest` 用于测试。
- 文件搬家清单写清哪些包文件、`static/`、`evals/`、`tests/`、`docs/context/` 和项目文档要带走，哪些 `.git/`、`__pycache__/`、Hermes 主项目模块不该直接带走。
- `AI_GLASSES_LLM_BACKEND=hermes` 仍保留为 legacy / 迁移期备用，但不写进新部署默认依赖。
- 正式迁出前仍必须做迁出前 eval baseline 和迁出后同场景对比。

未改变：

- 不新增正式 `pyproject.toml` 或锁文件。
- 不删除 Hermes `AIAgent` fallback。
- 不改 LLM backend 行为、记忆写入、召回、解释逻辑、SQLite schema 或 web search。
- 不正式迁出目录，不跑 live 网络或真实 API key 测试。

大白话：第十一刀不是搬家，而是写搬家清单；现在知道默认箱子里至少要放哪些文件、哪些依赖是必带/可选、以及旧 Hermes 备用线还没剪。

### 独立打包草案与 README 草案第十二刀

已新增 `pyproject.standalone.toml` 和 `docs/context/standalone-packaging.md`：

- `pyproject.standalone.toml` 当前只作为草案，不参与外层 `hermes-agent` 构建。
- 草案按“把 `ai_glasses_memory_assistant/` 作为独立仓库 root”来写，使用 `package-dir` 把当前目录映射为 `ai_glasses_memory_assistant` 包。
- 默认依赖只放 `openai`。
- optional dependencies 分为 `fastapi`、`tts`、`voice`、`dev`。
- console scripts 草案包括默认标准库入口 `ai-glasses-memory-assistant` 和可选 FastAPI 入口 `ai-glasses-memory-assistant-fastapi`。
- package data 草案纳入 `static/*.html/css/js`、`evals/*.jsonl` 和 `docs/context/*.md`。
- README 已补独立部署草案状态、`AI_GLASSES_HOME`、`.env` 示例、默认 `openai_compatible` backend、FastAPI extra、Hermes legacy fallback 和迁出前后 eval 对比提醒。

未改变：

- 不把草案改名为正式 `pyproject.toml`。
- 不安装新依赖。
- 不删除 Hermes fallback。
- 不改 LLM backend 行为、记忆写入、召回、解释逻辑、SQLite schema 或 web search。
- 不正式迁出目录，不跑 live 网络或真实 API key 测试，不固化 eval baseline。

大白话：第十二刀开始把第十一刀的“搬家清单”变成“包装盒草案”：依赖怎么分、入口怎么挂、静态页面怎么打包，都有了第一版，但还没真正搬家。

### 独立打包 dry-run 第十三刀

已用临时目录对 `pyproject.standalone.toml` 做独立打包 dry-run：

- 在 `/private/tmp/ai-glasses-packaging.XH6rEa` 复制当前 `ai_glasses_memory_assistant/`。
- 临时把 `pyproject.standalone.toml` 复制为 `pyproject.toml`。
- 使用 `pip install --no-index --no-deps --no-build-isolation --target ...` 安装到临时 target，不联网下载依赖，也不污染当前 conda 环境。
- 检查 package metadata、console scripts、package data、默认 server import 和 eval runner import。

本刀发现并修复的打包问题：

- 原草案只声明 `ai_glasses_memory_assistant` 包，临时安装后 `ai_glasses_memory_assistant.evals.runner` 无法 import。
- 已把 `ai_glasses_memory_assistant.evals` 加入 `pyproject.standalone.toml` 的 `[tool.setuptools].packages`。
- 文档一致性测试也补了对应断言，防止后续 packaging 草案再次漏掉 eval 子包。

修复后 dry-run 结论：

- `static/index.html`、`static/styles.css`、`static/app.js` 被打入 package data。
- `evals/scenarios.jsonl` 被打入 package data。
- `docs/context/*.md` 被打入 package data。
- 默认 console script 指向 `ai_glasses_memory_assistant.server:main`，可 import。
- FastAPI console script 指向 `ai_glasses_memory_assistant.app:main`，但 dry-run 不 import `app.py`，继续保持 FastAPI optional 边界。
- `ai_glasses_memory_assistant.evals.runner` 在临时安装 target 中可 import。

未改变：

- 不正式迁出目录。
- 不发布包。
- 不联网安装依赖。
- 不删除 Hermes fallback。
- 不改 LLM backend 行为、记忆写入、召回、解释逻辑、SQLite schema 或 web search。
- 不固化迁出前 eval baseline。

大白话：第十三刀是“把包装盒拿去临时试装”。试装时发现 eval runner 这条 Python 子包没装进去，已经补上；补完后默认 server、eval runner、静态页面、场景文件和上下文文档都能从临时安装 target 里看到。

### Hermes legacy fallback 封存第十四刀

本刀选择“彻底封存但不删除”：

- 默认 `openai_compatible` 路径继续不 import、不初始化、不触发 Hermes `AIAgent`、`runtime_provider` 或 Hermes env loader。
- `AI_GLASSES_LLM_BACKEND=hermes` 单独设置不再足够启用 Hermes fallback，会直接报错。
- 只有同时设置 `AI_GLASSES_LLM_BACKEND=hermes` 和 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1`，才会进入 sealed legacy fallback 分支。
- sealed legacy 分支仍会延迟 import Hermes `runtime_provider`、Hermes env loader、`hermes_constants`，并通过 `HermesLLMClient` 调用 `run_agent.AIAgent`。
- 现有 `HermesLLMClient` 和 `create_hermes_llm_client()` 暂时保留，作为迁移期对照和最终删除前的安全缓冲。

测试边界：

- 新增测试覆盖“只设置 `AI_GLASSES_LLM_BACKEND=hermes` 会被封存门挡住，且不会 import Hermes runtime 模块”。
- 旧 Hermes 参数兼容测试改为显式设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1`。
- 文档一致性测试覆盖新的 legacy opt-in 环境变量，避免 README / standalone docs 又把单开关写成可用路径。

未改变：

- 不删除 Hermes fallback 实现。
- 不改默认 OpenAI-compatible backend 行为。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- 不改 web_search。
- 不固化迁出前 eval baseline。

大白话：第十四刀不是把旧线剪断扔掉，而是把旧线收到盒子里并上了第二把锁。默认启动和新部署碰不到它；只有迁移期故意拿钥匙打开，才会走 Hermes 老后端。

### 迁出前 eval baseline 方案第十五刀

迁出前 eval baseline 方案已执行并落盘；历史方案文档已删除，后续对比以 `reports/standalone-migration-baseline/` 中的 notes 和报告为准。

未改变：

- 不执行正式 baseline。
- 不正式迁出目录。
- 不改 LLM backend 行为。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- 不改 web_search。
- 不删除 sealed Hermes fallback。

大白话：第十五刀是把“搬家前拍照怎么拍、拍哪些角度、照片上要看哪些细节”写成固定清单；这刀没有真的拍照，正式 baseline 结果还要下一刀按清单执行后再落盘。

### 迁出前离线 eval baseline 首次执行第十六刀

已按迁出前 baseline 三批方案首次执行迁出前 baseline；历史方案文档已删除，后续以实际报告为准：

- 核心 active 门禁。
- 效果回放与主线压力。
- target 缺口快照。

报告已落盘：

```text
ai_glasses_memory_assistant/reports/standalone-migration-baseline/
```

索引 notes：

```text
pre-extraction-20260706-baseline-notes.md
```

执行配置：

- 默认 `openai_compatible` backend。
- `AI_GLASSES_LLM_PROVIDER=deepseek`
- `AI_GLASSES_LLM_MODEL=deepseek-v4-flash`
- `AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com`
- API key 来自已有 `DEEPSEEK_API_KEY`，没有写入明文。
- 未启用 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK`。

首次 baseline 摘要：

| 批次 | active_pass_rate | active_failed_turns | target_failed_turns |
| --- | ---: | ---: | ---: |
| 核心 active 门禁 | 0.6167 | 23 | 0 |
| 效果回放与主线压力 | 0.6364 | 4 | 9 |
| target 缺口快照 | 1.0 | 0 | 3 |

结论边界：

- 第十六刀不修改 LLM backend 行为。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- 不改 web_search。
- 不删除 sealed Hermes fallback。
- 不正式迁出目录。
- 这次结果不是全绿质量证明，而是迁出前现状快照；迁出后必须同口径复跑并比较新增失败、失败原因变化、`target_failed_turns` 和关键 debug/audit 字段。

执行观察：

- 首次不带临时 `AI_GLASSES_LLM_MODEL` / `AI_GLASSES_LLM_BASE_URL` 的命令因配置缺失未生成报告。
- 补齐非密钥 DeepSeek 配置后完成三批报告。
- target 批报告已生成且命令返回 0，但结束时观察到后台线程 `sqlite3.OperationalError: attempt to write a readonly database`，已记录在 baseline notes 中，迁出后复跑时需要同口径观察。

当前 GitHub 仓库状态：

- 当前本项目已切到 `main` 分支。
- 当前 remote 为 `origin=git@github.com:huyaokai-nreal/ai_glasses_memory_assistant.git`。
- `main` 已跟踪 `origin/main`。
- 第十六刀 baseline notes 中记录的 `dev_ykhu / bda74fc31` 是当时执行 baseline 的历史状态，不应改写；后续迁出预演报告应记录当前 `main` / `origin/main` 状态和新的 commit。

### 迁出后离线 eval baseline 首次执行第十七刀

已在当前独立仓库 `main` 上按第十六刀同一批场景首次执行迁出后 baseline：

- 核心 active 门禁。
- 效果回放与主线压力。
- target 缺口快照。

报告已落盘：

```text
reports/standalone-migration-baseline/
```

索引 notes：

```text
post-extraction-20260706-baseline-notes.md
```

首次 post baseline 摘要：

| 批次 | active_pass_rate | active_failed_turns | target_failed_turns |
| --- | ---: | ---: | ---: |
| 核心 active 门禁 | 0.6 | 24 | 0 |
| 效果回放与主线压力 | 0.6364 | 4 | 9 |
| target 缺口快照 | 1.0 | 0 | 5 |

迁出前后初步对比：

- mainline / replay 批失败数量和失败 scenario 集合与迁出前一致。
- active gate 比迁出前新增 1 个 active failed turn，新增失败场景为 `memory_mechanism_correction_target_preference_supersede`。
- target 快照比迁出前新增 `public_dialogue_preference_write`，`target_failed_turns` 从 3 增至 5。
- target 批仍复现 `sqlite3.OperationalError: attempt to write a readonly database`；迁出前也出现过，因此先记录为既有后台 job 生命周期问题。

结论边界：

- 第十七刀不修改 LLM backend 行为。
- 不改记忆写入、召回、解释逻辑。
- 不删除 sealed Hermes fallback。
- 这次结果说明独立仓库已能跑完整 post baseline，但不能直接判定“完全无回归”；新增差异 scenario 需要下一刀抽查 debug/audit 后再定性。

## 不建议直接复制 Hermes 代码

后续迁移不应把 `run_agent.py`、`model_tools.py`、`toolsets.py`、`hermes_cli/`、`tools/` 等 Hermes 内部模块整块复制进本项目。

原因：

- 会把独立产品变成“小 Hermes”，长期维护边界仍然不清楚。
- 会引入当前 demo 不需要的工具、插件、CLI、provider 和 session 复杂度。
- 开源时解释成本更高，用户也更难判断哪些代码是 AI 眼镜助手核心能力。
- 复制后的 Hermes 组件可能仍依赖原有全局配置、目录结构和隐含初始化顺序。

推荐方式是按能力替换：路径、环境变量、LLM 调用、session、搜索分别做本项目最小实现。

## 迁移顺序建议

中等推理强度阶段只适合做准备工作：

- 维护本文依赖清单。
- 补充 `PLANS.md` 中的独立化迁移入口。
- 不碰 `AIAgent`、`SessionDB`、LLM provider、真实运行链路。

高推理强度阶段再开始真正迁移：

1. 新增本项目 `env_loader.py`。已完成。
2. 抽象 `LLMClient`，先保留 Hermes backend 做对照。已完成。
3. 实现 OpenAI-compatible LLM backend。已完成默认切换，Hermes backend 保留为手动 fallback。
4. 替换或移除 `SessionDB`。已完成最小隔离，默认后端不依赖 Hermes `SessionDB`。
5. 移除 Hermes `web_tools`。已完成默认依赖迁移，web search 收口到本项目 `web_search.py`。
6. 独立启动验证与 Hermes fallback 边界清点。已完成默认 OpenAI-compatible 路径和显式 Hermes fallback 的测试化边界。
7. 整理独立部署文档/启动说明。已完成 `standalone-startup.md` 和轻量文档一致性测试。
8. 冻结默认启动路径与 Hermes fallback 硬边界。已完成 `agent_bridge.py` 顶层 Hermes import 下沉和静态边界测试。
9. 仓库抽离预检查。已完成临时复制目录 `python -S` 导入/编译检查，历史过程文档已删除，当前结论保留在本文。
10. 独立部署文档 / dependency manifest 草案。已完成，结果见 `standalone-deployment.md`。
11. 独立打包草案与 README 草案。已完成，结果见 `pyproject.standalone.toml` 和 `standalone-packaging.md`。
12. 独立打包 dry-run 与临时目录安装预演。已完成，结果见 `standalone-packaging.md`。
13. Hermes legacy fallback 移除或彻底封存评估。已完成封存，不删除实现；结果见本文“第十四刀”。
14. 迁出前离线 eval baseline 方案与报告模板。已完成且历史方案文档已删除，后续以实际 baseline reports 为准。
15. 迁出前 baseline 首次执行与报告落盘。已完成，结果见 `reports/standalone-migration-baseline/pre-extraction-20260706-baseline-notes.md`。
16. 当前 GitHub `origin/main` 已建立，迁出后 baseline 已在当前 `main` 首次复跑完成；后续优先抽查新增差异 scenario，再决定是否继续删除 Hermes sealed fallback。

## 效果不退化验收

每次替换都要验证它没有削弱 demo 的核心效果。至少保住：

- 记忆写入：用户明确表达偏好、事件、任务时仍能保存。
- 记忆召回：后续问题能召回正确 profile、event、task 或 timeline evidence。
- 解释依据：用户问“为什么记得”时仍能指向具体记忆或原文证据。
- 会话连续性：同一 session 内的多轮上下文不丢失。
- 隐私门控：密码、token、验证码、银行卡等敏感信息仍不被静默保存。
- 工具边界：独立版不会意外启用 Hermes 工具或 Hermes 记忆。

完全迁出工程前后还必须做效果对比：

1. 在当前外层 `hermes-agent` checkout 先跑一轮离线 eval / target 场景，保存报告作为迁出前基线。
2. 迁出到独立工程后，用同一批场景、同一类配置和同一套判定指标复跑。
3. 对比至少包含通过率、`target_failed_turns`、关键 debug/audit 字段、记忆写入/召回/解释结果，以及代表性回复差异。
4. 如果迁出后结果退化，先定位是配置、路径、LLM backend、eval runner、记忆数据隔离还是真实行为变化，再决定是否继续删除 fallback。

大白话：独立化不是只让代码能离开 Hermes 文件夹，而是要让“记得准、解释清、不乱存、不串工具”这些用户能感知的能力保持住。

## 后续启动条件

开始真正代码迁移前，先确认：

- 当前工作区没有与迁移无关但会冲突的大量未提交代码改动。
- 已选定第一刀目标，例如只替换 home 目录，不同时碰 LLM。
- 已明确验证命令，至少包含相关单测、最小 service 测试和必要的离线 eval / target 场景。
- 已决定是否保留 Hermes backend 作为短期对照开关。
- 如果任务进入“完全迁出工程”阶段，必须先生成迁出前 eval 基线，并规划迁出后同场景复跑和差异报告。
