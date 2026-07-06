# 仓库抽离预检查

本文记录 `ai_glasses_memory_assistant` 离开外层 `hermes-agent` 前的第十刀预检查结果。它不是正式迁出记录，也不是效果基线报告。

## 检查目标

- 判断默认 OpenAI-compatible 启动路径是否还依赖 Hermes 主项目顶层模块。
- 区分剩余依赖属于默认路径、显式 legacy fallback，还是 eval/test 迁移期。
- 用临时目录做最小抽离预演，提前发现离开 `hermes-agent` 根目录后的 import / compile 问题。

## 预检查结论

默认路径当前可以脱离 Hermes 主项目顶层模块导入：

- `agent_bridge.py` 顶层不再 import `hermes_constants`、`hermes_cli` 或 `run_agent`。
- `server.py` 在临时复制目录中用 `python -S` 可以导入，说明不依赖外层 `hermes-agent` editable 安装钩子。
- `evals/runner.py` 已从顶层 Hermes import 改成延迟 best-effort fallback；没有 Hermes 主项目时也能导入。

仍不能说已经完成独立仓库：

- 第十四刀后，`AI_GLASSES_LLM_BACKEND=hermes` 必须同时搭配 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1` 才会触发 Hermes legacy fallback。
- `llm_client.py::create_hermes_llm_client()` 仍会在被调用时 import `run_agent.AIAgent`。
- `agent_bridge.py::_new_session()` 的 hermes 分支仍会在被调用时 import `hermes_cli.runtime_provider`、Hermes env loader 和 `hermes_constants`。
- `HERMES_HOME` 仍是旧数据路径兼容 fallback。
- 当前没有独立 packaging / dependency manifest / README 迁出说明。
- FastAPI 入口 `app.py` 需要运行环境安装 `fastapi`；当前 `hermes` conda 环境的硬抽离预演里缺少这个依赖。标准库 `server.py` 不受影响。

## 剩余依赖分组

| 分组 | 剩余线 | 当前结论 |
| --- | --- | --- |
| 默认启动路径 | `server.py -> GlassesChatService -> OpenAICompatibleLLMClient` | 未发现 Hermes 主项目顶层 import；缺配置会报 OpenAI-compatible 配置错误，不静默回 Hermes。 |
| legacy fallback | `AI_GLASSES_LLM_BACKEND=hermes`、`AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1`、`HermesLLMClient`、`run_agent.AIAgent`、`hermes_cli.runtime_provider`、Hermes env loader、`hermes_constants` | 第十四刀后改为双开关 sealed legacy 路径；默认和单独 `hermes` backend 都不会触发 Hermes runtime。 |
| eval/test 迁移期 | `evals/runner.py` 的旧 Hermes `.env` fallback、测试里的 `HERMES_HOME` 隔离习惯 | `evals/runner.py` 已改成可缺省延迟 fallback；测试里的 `HERMES_HOME` 主要是旧隔离写法，后续可逐步替换为 `AI_GLASSES_HOME`。 |
| 独立部署依赖 | `app.py` 的 `fastapi` | 不是 Hermes 依赖，但独立部署文档或 dependency manifest 需要写清。 |

## 抽离预演方法

本刀使用临时目录：

```text
/private/tmp/ai-glasses-extract-check.6tdjYk
```

预演步骤：

```bash
cp -R ai_glasses_memory_assistant /private/tmp/ai-glasses-extract-check.6tdjYk/
```

用 `python -S` 禁用 site 初始化，避免外层 `hermes-agent` editable 安装钩子把 Hermes 主项目重新挂进 `sys.path`：

```bash
conda run -n hermes python -S -c "import sys; sys.path.insert(0, '/private/tmp/ai-glasses-extract-check.6tdjYk'); import ai_glasses_memory_assistant.agent_bridge; import ai_glasses_memory_assistant.server; print('default-server-import-ok-without-site')"
```

结果：通过。

```bash
conda run -n hermes python -S -c "import sys; sys.path.insert(0, '/private/tmp/ai-glasses-extract-check.6tdjYk'); import ai_glasses_memory_assistant.evals.runner; print('eval-runner-import-ok-without-site')"
```

结果：通过。

```bash
conda run -n hermes python -S -m py_compile /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/agent_bridge.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/llm_client.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/env_loader.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/app_home.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/server.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/app.py /private/tmp/ai-glasses-extract-check.6tdjYk/ai_glasses_memory_assistant/evals/runner.py
```

结果：通过。

FastAPI 入口检查：

```bash
conda run -n hermes python -S -c "import sys; sys.path.insert(0, '/private/tmp/ai-glasses-extract-check.6tdjYk'); import ai_glasses_memory_assistant.app"
```

结果：失败，原因是当前环境缺少 `fastapi`。这是独立部署依赖问题，不是 Hermes 主项目依赖问题。

## 本刀改动边界

已改：

- `evals/runner.py` 的 Hermes home/env fallback 从顶层 import 改为延迟 best-effort helper。
- 增加测试防止 `evals/runner.py` 重新引入顶层 `hermes_constants` / `hermes_cli`。
- 更新 `PLANS.md`、`standalone-migration.md` 和本文。

未改：

- 不删除 Hermes AIAgent fallback。
- 不改 LLM backend 行为。
- 不改记忆写入、召回、解释逻辑。
- 不改 SQLite schema。
- 不改 web_search。
- 不做正式迁出。
- 不跑迁出前 eval baseline。

## 下一步

建议下一刀二选一：

- 继续做独立部署文档 / dependency manifest / README 草案，写清标准库 `server.py` 是默认入口，FastAPI 需要额外依赖。
- 后续最终决定是否直接删除 sealed Hermes legacy fallback，或在正式独立仓库中继续作为非默认 legacy adapter 保留。

正式迁出前仍必须先生成迁出前离线 eval / target baseline，迁出后用同一批场景复跑对比。
