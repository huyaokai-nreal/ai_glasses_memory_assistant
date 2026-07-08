# AI 眼镜个人记忆助手

这是一个面向“无限 AI 眼镜 + 手机端 App”组合形态的个人记忆助手本地原型。

最终产品希望让用户不用手动整理周报、会议纪要、生活记录和回顾材料：眼镜负责低摩擦语音/环境输入，手机 App 负责文档、音频、文本补充和结果查看，所有输入进入统一记忆系统，再服务总结、回忆、回顾和周报等自动化输出。

当前 Web/语音 demo 用来验证：

```text
眼镜语音/环境输入 + App 文档/音频/文本输入
-> 统一 ingestion 和记忆内化
-> 总结 / 回忆 / 回顾 / 周报等自动化输出
```

当前代码是真相。后续 Codex 接手开发时，先读 `AGENTS.md` 和 `PLANS.md`，再按任务进入 `docs/context/`。

## 主目录保留原则

仓库主目录只保留几类“非放不可”的内容：

- `ai_glasses_memory_assistant/` 正式 Python 包代码
- 少量根目录兼容入口，如 `server.py`、`app.py`
- 顶层说明和工程规则，如 `README.md`、`AGENTS.md`、`PLANS.md`
- 构建和环境配置，如 `pyproject.toml`、`.gitignore`、`.env`

其他内容按角色进入对应目录，方便后续 Codex 和人工查找：

- `certs/`：本地 HTTPS 测试证书
- `docs/`：长期上下文、人类说明、HTML 和报告草稿
- `tests/`：单元测试和 service 级测试
- `evals/`：评估 runner、adapter 和场景
- `static/`：前端静态资源
- `scripts/`：辅助脚本
- `data/`：基准数据和样例数据
- `reports/`：运行产物和评估报告

## 独立部署草案状态

当前默认启动路径已经不依赖 Hermes `AIAgent`、Hermes `SessionDB`、Hermes `tools.web_tools` 或 Hermes env loader；核心 Python 代码已搬入 `ai_glasses_memory_assistant/` 正式包目录。

第十二刀新增了独立打包草案：

```text
pyproject.standalone.toml
```

它只是草案，不会被当前外层 `hermes-agent` 构建自动使用。正式迁出时，再把它作为独立仓库 `pyproject.toml` 的起点。

## 当前定位

当前处于 Stage 2 可运行原型阶段：

- Web UI 模拟眼镜对话和手机端管理入口，支持文字、浏览器语音、TTS、定位。
- `/api/chat` 已形成 planner baseline、统一 turn decision pre_reply_decision、记忆召回、answer directive、回复、后台写入、debug/audit 的闭环。Web demo 只保留 `llm_first` 主链路，旧 `routing_mode` 请求字段会被忽略。
- 个人记忆落在 SQLite，按 `user_id` 隔离。
- 已支持 profile、event、assistant_preference、原始 timeline/chunk 全文回忆、timeline evidence 管理、文本/JSON 导入、Markdown 文档归档、continuous_capture、周报草稿和手动提醒候选检查。
- 这些能力分别对应最终产品里的眼镜语音入口、App 文档/音频入口、个人记忆管理和自动总结的早期验证。
- 纯文字输入下的底层记忆内核已经形成第一版闭环：写入门控、去重/冲突更新、纠错、生命周期、evidence、召回仲裁、删除和 eval 都有当前最小实现。
- 仍不是生产级硬件 runtime、原生手机 App、音频上传/转写管道、后台常驻收音、主动提醒系统、项目知识图谱或多租户服务。

## Codex 文档入口

文档分工遵守下面的原则：

- `AGENTS.md` 只放 Codex 工作规则、入口和验证命令。
- `PLANS.md` 只放当前阶段、P0/P1、下一刀和验收标准。
- `docs/context/*.md` 放长期机制、调用链、边界和专题知识。
- `docs/reports/*.md` 放压测、复盘和证据报告。
- `docs/html/*.html` 放人类阅读版可视化，不作为当前开发计划。

| 文件 | 用途 |
| --- | --- |
| `AGENTS.md` | Codex 工作规则、代码入口、验证命令和注意事项。 |
| `PLANS.md` | 当前阶段、P0/P1、验收标准、推荐开发顺序。 |
| `docs/context/README.md` | 文档地图。 |
| `docs/context/code-map.md` | 按任务定位文件和函数。 |
| `docs/context/current-status-and-gaps.md` | 当前代码审阅结论和缺口。 |
| `docs/context/pipeline.md` | `/api/chat`、import、capture、weekly report、reminder 的真实流程。 |
| `docs/context/memory-mechanism.md` | 记忆模型、写入门控、召回、隐私边界。 |
| `docs/context/eval-coverage.md` | eval 覆盖矩阵、AI 眼镜真实使用场景缺口和 target 场景规划。 |
| `docs/context/demo-features.md` | 当前 demo 能力、API 和边界。 |
| `docs/context/production-readiness.md` | 生产级完整记忆系统定义，以及当前底层机制和完整产品之间的距离。 |
| `docs/context/standalone-migration.md` | 独立化迁移清单、剩余 Hermes 依赖和迁移顺序。 |
| `docs/context/standalone-startup.md` | 当前推荐独立启动方式、`.env` 和 Hermes fallback 边界。 |
| `docs/context/standalone-deployment.md` | 独立部署和 dependency manifest 草案。 |
| `docs/context/standalone-packaging.md` | 独立打包草案、optional dependencies 和 package data。 |

## 快速运行

推荐先设置本项目自己的 home：

```bash
export AI_GLASSES_HOME="$HOME/.ai-glasses-memory-assistant"
```

数据默认落在：

```text
$AI_GLASSES_HOME/data/events.db
$AI_GLASSES_HOME/data/timeline.db
$AI_GLASSES_HOME/data/sessions.db
$AI_GLASSES_HOME/data/chat_audit.jsonl
```

从本仓库根目录运行，优先使用本地 conda 环境 `hermes`：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

默认访问：

```text
http://127.0.0.1:8765
```

同一局域网设备测试语音或定位时，浏览器通常需要 HTTPS 安全上下文：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server --certfile certs/cert.pem --keyfile certs/key.pem
```

## 本地 LLM 配置

Web demo 默认优先让 LLM 负责开放语义路由、记忆召回判断、工具判断和长期记忆候选抽取；本地 planner 主要保留寒暄、身份查询、显式记忆命令、敏感信息拒绝等确定性 fast path。

推荐把配置写到：

```text
$AI_GLASSES_HOME/.env
```

默认部署建议使用 OpenAI-compatible 接口。当前主验证目标是 DeepSeek `deepseek-v4-flash`：

```bash
AI_GLASSES_LLM_BACKEND=openai_compatible
AI_GLASSES_LLM_PROVIDER=deepseek
AI_GLASSES_LLM_MODEL=deepseek-v4-flash
AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com
AI_GLASSES_LLM_API_KEY=<your-deepseek-api-key>
AI_GLASSES_LLM_API_MODE=chat_completions
AI_GLASSES_LLM_REASONING_ENABLED=false
```

如果 `AI_GLASSES_LLM_PROVIDER=deepseek` 且未设置 `AI_GLASSES_LLM_API_KEY`，会尝试读取 `DEEPSEEK_API_KEY`。缺少 `model`、`base_url` 或 API key 时会直接报错，避免独立版静默回退到 Hermes。

需要临时对照旧 Hermes 后端时，必须同时打开 legacy opt-in：

```bash
export AI_GLASSES_LLM_BACKEND=hermes
export AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK=1
```

Hermes fallback 已封存为 legacy / 迁移期备用。只设置 `AI_GLASSES_LLM_BACKEND=hermes` 不会启用它；新部署默认应该使用 `openai_compatible`。

路由模式已经固定为 `llm_first`，不再支持通过环境变量或前端开关切回本地 planner 对照模式。`/api/runtime` 只返回当前固定运行信息。

## 独立依赖草案

`pyproject.standalone.toml` 当前按下面方式分组：

| 分组 | 依赖 | 用途 |
| --- | --- | --- |
| 默认 | `openai` | 标准库 server + OpenAI-compatible LLM backend。 |
| `fastapi` | `fastapi`、`pydantic`、`uvicorn` | 可选 FastAPI 入口 `app.py`。 |
| `tts` | `edge-tts` | 可选 Edge TTS 语音播报。 |
| `voice` | `funasr` | 可选本地 ASR、声学情绪和声纹模型。 |
| `dev` | `pytest` | 测试和文档一致性检查。 |

标准库 `ai_glasses_memory_assistant/server.py` 是默认入口。根目录 `server.py` 只是兼容薄入口。FastAPI 入口复用同一套 `GlassesChatService`，但需要安装 `fastapi` extra 后再使用：

```bash
python -m ai_glasses_memory_assistant.app
```

如果正式迁出为独立仓库，至少要保留 `static/`；它是 Web UI 和两个 server 入口都需要的 package data。

正式迁出前后必须用同一批离线 eval / target 场景做效果对比，至少对比通过率、`target_failed_turns`、关键 debug/audit 字段、记忆写入/召回/解释结果和代表性回复差异。

## 主代码入口

| 文件 | 职责 |
| --- | --- |
| `ai_glasses_memory_assistant/agent_bridge.py` | `GlassesChatService` 主 service：聊天、召回、回复、后台 job、导入、capture、周报、提醒、audit。 |
| `ai_glasses_memory_assistant/turn_planner.py` | 本地 planner：在 `llm_first` 下保留低风险确定性 fast path 和 baseline/fallback，不再作为可切换主路线。 |
| `ai_glasses_memory_assistant/intent_policy.py` | 长期记忆写入门控和短确认回复 helper。 |
| `ai_glasses_memory_assistant/turn_semantic_classifier.py` | 单次 pre_reply_decision：一次 LLM 判断同时输出 `reply_mode / location / web / recall / memory_action / memory_kind / memory_type / candidate_content / correction flags`，主导回复前准备和短 turn 记忆候选生成。 |
| `ai_glasses_memory_assistant/memory_candidate.py` | 记忆候选与兼容 intent debug 的轻量数据结构。 |
| `ai_glasses_memory_assistant/answer_synthesizer.py` | `llm_first` 的回答组织规划层，基于 route/temporal/证据摘要生成 `answer_directive`，只组织表达，不重新决定路由。 |
| `ai_glasses_memory_assistant/memory_lifecycle.py` | 结构化记忆 `active/stale/superseded/deleted` 生命周期策略。 |
| `ai_glasses_memory_assistant/memory_evidence.py` | timeline evidence 引用、active/retained 清理决策。 |
| `ai_glasses_memory_assistant/memory_recall_arbitration.py` | 文档、raw timeline、结构化记忆、observation、profile 的召回仲裁。 |
| `ai_glasses_memory_assistant/memory_confidence.py` | 写入、dedupe、纠错和 observation update 的 confidence 阈值与 debug 解释。 |
| `ai_glasses_memory_assistant/memory_store.py` | SQLite 记忆存储、搜索、去重、软删除、证据合并。 |
| `ai_glasses_memory_assistant/timeline_store.py` | SQLite 原始时间线存储，保存 raw turn/capture chunk，服务跨 session 原文全文回忆和 evidence。 |
| `ai_glasses_memory_assistant/server.py` | 标准库 HTTP demo 入口。 |
| `ai_glasses_memory_assistant/app.py` | FastAPI 入口，API 行为应与 `server.py` 对齐。 |
| `server.py`、`app.py` | 根目录兼容薄入口，只负责转发到正式包入口。 |
| `static/` | Web UI、语音、TTS、定位、debug、memory job 轮询。 |
| `evals/` | live-LLM 离线评估。 |
| `tests/` | 单元测试和 service 级测试。 |

## 主要 API

| API | 作用 |
| --- | --- |
| `POST /api/chat` | 主聊天入口；固定走 `llm_first`，旧客户端传入的 `routing_mode` 字段会被忽略。 |
| `GET /api/runtime` | 返回前端需要的运行时信息，目前包含固定 `routing_mode=llm_first`。 |
| `GET /api/memories` | 查看当前用户 active 记忆和文档 metadata。 |
| `POST /api/memories` | 手动新增记忆。 |
| `DELETE /api/memories/{memory_id}` | 默认软删除当前用户记忆；显式 `purge=true` 时彻底删除并清理可清理 evidence/audit。 |
| `GET /api/documents/{document_id}` | 查看单个文档，编辑时返回 Markdown 原文。 |
| `PATCH /api/documents/{document_id}` | 编辑文档名、标题、摘要和 Markdown 原文。 |
| `DELETE /api/documents/{document_id}` | 默认软删除当前用户文档；显式 `purge=true` 时彻底删除文档记录。 |
| `GET /api/memory/search` | 搜索当前用户记忆。 |
| `GET /api/timeline/search` | 搜索当前用户原始 timeline chunks，用于调试原文回忆。 |
| `GET /api/timeline/chunks` | 按 evidence ids 查看 timeline chunk、active/retained 引用和是否可彻底删除。 |
| `DELETE /api/timeline/chunks` | 批量软删除或显式 `purge=true` 彻底删除可清理的 timeline chunks。 |
| `GET /api/memory/jobs` | 查询后台记忆写入 job。 |
| `POST /api/memory/import` | 导入文本或 JSON 事件。 |
| `POST /api/capture/start|append|stop` | 连续输入采集并导入。 |
| `GET /api/weekly-report` | 生成启发式周报草稿。 |
| `GET /api/reminders/check` | 手动检查未来任务提醒候选。 |
| `GET /api/debug/audit` | 查看最近 audit。 |
| `POST /api/tts` | 生成语音音频。 |

## 验证

文档或小改动先跑：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent/ai_glasses_memory_assistant
git diff --check
```

Python 改动至少跑：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py app.py
conda run -n hermes python -m unittest discover tests -q
```

需要产品链路验证时跑 live eval：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```

## 当前优先级

最新优先级以 `PLANS.md` 为准。自然偏好写入、敏感信息门控、raw timeline 脱敏、轻量排序、文档指代、聊天长输入 fast path、纠错对象定位、confidence fallback、timeline evidence 管理和多来源召回仲裁已有当前最小实现；长会议/项目复盘长输入场景已进入 active 门禁。下一步优先补更长上下文 source 组合、公开素材长口述 target 诊断、项目状态结构化，或再设计主动提醒 runtime。不要为了让 target eval 临时通过而把未设计好的能力硬编码进主链路。
