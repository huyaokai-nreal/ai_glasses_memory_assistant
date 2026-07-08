# Codex 工程手册

本文件是 Codex 在 `ai_glasses_memory_assistant/` 中接手任务时的第一入口。当前代码是真相；如果本文和代码不一致，先读代码，再修文档。

## 当前项目边界

这是 AI 眼镜个人记忆助手的本地 Web/语音原型，不是 Hermes 主仓库、Feishu gateway、硬件眼镜 runtime 或生产级多租户服务。

核心目标：

- 用户通过网页文字或浏览器语音模拟眼镜对话。
- 助手先正常回复，再按需沉淀长期个人记忆。
- 后续问题能按需召回画像、事件、计划、任务和偏好。
- 用户能查看、搜索、手动新增、删除记忆。
- debug、audit、eval 能解释每轮为什么读记忆、写记忆、拒绝保存、查 web、用定位或走主模型。

## 先判断任务

编码前先明确：

- 要解决的具体问题是什么。
- 成功标准是什么，最好能转成测试、eval 场景或可复现对话。
- 需要读哪些入口文件。
- 是否会影响 API、数据结构、记忆门控、用户隐私或现有前端行为。
- 当前工作区是否已有无关修改；不要覆盖用户改动。

存在歧义时先说明假设。可以合理推进的小歧义不必停住，但必须在回复中写清楚取舍。

## 自动优化用户提示词

用户日常可能只用自然语言描述问题。接手任务时，默认按 GPT-5.5 prompt guidance 的结果导向方式，在心里把用户输入整理成可执行任务，不要求用户每次写长 prompt。

整理时优先补齐：

- `目标`：这次要解决、分析或验证什么。
- `成功标准`：什么现象代表完成，能否转成测试、eval、audit 证据或可复现对话。
- `上下文`：需要查看的代码入口、日志、配置、记忆数据、前端行为或文档。
- `约束`：最小修改、接口兼容、不 hard code、不改无关文件、隐私和实时信息边界。
- `验证`：最相关的测试、静态检查、人工复现或无法验证原因。
- `输出`：最后用中文说明根因、功能变化、用户可感知变化、验证结果和剩余风险。

不要把这一步变成冗长复述。简单任务直接执行；复杂、多文件、调试或高风险任务，先用简短中文说明整理后的目标、成功标准和计划，再继续。

如果用户说“按 GPT-5.5 prompt guide 优化我的需求”“帮我整理 prompt”“自动优化 prompt”，应主动把模糊输入改写为上面的结构，并指出缺失但关键的信息。能合理假设的，不要停住；会影响方向、数据安全、接口兼容或破坏性操作的，先问。

示例：

```text
用户：最近天气回答又不准，看看。

内部整理：
目标：分析最近天气相关对话不准的根因。
成功标准：区分定位、web 搜索、记忆召回、主模型臆测或前端传参问题，并给出证据。
上下文：先查 chat_audit.jsonl / debug payload，再按证据读 agent_bridge.py、turn_planner.py、前端定位传参。
约束：先分析，未经确认不做大重构；不把实时位置写入长期记忆。
验证：如有代码修改，跑最相关 unittest 或给出人工复现步骤。
```

## 必读代码入口

| 任务类型 | 先读文件 | 重点 |
| --- | --- | --- |
| 聊天主链路 | `ai_glasses_memory_assistant/agent_bridge.py` | `GlassesChatService.chat()`、本地 fast path、主模型调用、debug/audit、后台写入。 |
| 记忆存储 | `ai_glasses_memory_assistant/memory_store.py` | `EventMemoryStore`、SQLite schema、搜索、软删除、去重、证据合并。 |
| 本地规划 | `ai_glasses_memory_assistant/turn_planner.py` | 本地 fast path、是否读 profile/event、是否写候选、时间范围、web/location 需求、`reply_mode`。 |
| 写入门控 | `ai_glasses_memory_assistant/intent_policy.py` | `should_write_memory_candidate()`、问题/敏感信息/低置信度拦截和短确认回复 helper。 |
| 回复前决策 | `ai_glasses_memory_assistant/turn_semantic_classifier.py` | 非 fast path 只做一次 `PreReplyDecision`，统一决定回复模式、召回、web/location 和记忆候选，不直接回答用户。 |
| 时间解析 | `ai_glasses_memory_assistant/temporal_parser.py` | LLM 时间解析兜底；主路径优先本地 `resolve_temporal_local()`。 |
| HTTP 服务 | `ai_glasses_memory_assistant/server.py` | 标准库 HTTP 入口；根目录 `server.py` 只是兼容薄入口。 |
| 前端体验 | `static/app.js`、`static/index.html`、`static/styles.css` | 语音、TTS、定位、debug、memory job 轮询、记忆面板。 |
| 离线评估 | `evals/`、`tests/` | 场景门禁、target 缺口、单元测试和报告输出。 |

## 主要执行链路

`POST /api/chat` 的真实主路径：

```text
ai_glasses_memory_assistant/server.py
-> GlassesChatService.chat()
-> plan_turn()
-> classify_pre_reply_decision()
-> TurnPlan.apply_pre_reply_decision()
-> 按需召回 profile/event/location/web
-> 本地回复或 Hermes AIAgent.run_conversation()
-> 同步保存或创建后台 memory job
-> 返回 reply/debug/recalled/saved/memory_processing
-> 追加 chat_audit.jsonl
```

扩展功能都应尽量复用同一条 service 层能力：

- `POST /api/memory/import` -> `import_memory_events()`
- `POST /api/capture/start|append|stop` -> capture 片段汇总后进入导入管道
- `GET /api/weekly-report` -> 基于 event/task/decision/risk 记忆生成启发式周报
- `GET /api/reminders/check` -> 手动读取未来 task 候选，不是主动提醒 runtime
- `GET /api/memory/jobs` -> 查询后台写入公开状态；先读进程内状态，必要时从 SQLite `memory_jobs` 恢复查询结果

## 修改原则

- 小步、最小、可验证。不要顺手重构无关代码。
- 优先延续现有函数和数据结构，不新建框架、服务、依赖或大抽象。
- 不写散落 hard code。产品规则优先放在 `turn_planner.py`、`intent_policy.py` 或清晰命名的 helper 中，并让 debug 能解释。
- 重要函数或关键调用上一行保留简明注释，解释为什么存在，不写重复代码含义的空注释。
- 不破坏现有 API；新增响应字段优先放在 `debug` 或向后兼容结构中。
- `ai_glasses_memory_assistant/server.py` 是唯一 HTTP 入口；新增 API 时保持薄包装，把业务逻辑放在 service 层。
- 长期状态路径必须使用 `app_home.py` 的 `get_data_dir()` / `get_app_home()`，不要硬编码 `~/.hermes` 或项目目录。`HERMES_HOME` 只作为迁移期兼容 fallback。
- 不主动创建、查找或依赖 `.venv` / `venv`；使用本地 conda 环境 `hermes`。

## 推理强度建议

当用户准备开始一个优化、修复、审查或实现任务时，应在正式实施前给出建议推理强度：`低`、`中`、`高` 或 `超高`，用于帮助用户决定本次 GPT-5.5 / Codex 配置。判断目标是同时节省 token 和保证优化质量，不要默认建议高强度。

如果用户已经明确要求立即实施，可以在简短说明目标、成功标准或计划时一并给出推理强度建议；不要等任务完成后才首次给出。任务完成后的推理强度说明只作为“下次类似任务参考”，不能替代开始前建议。

建议规则：

- `低`：适合单文件小改、文案调整、明确的小 bugfix、只读解释、运行测试或补充很局部的文档说明。
- `中`：适合本项目多数真实功能优化，包括跨少量文件、需要先读调用链、涉及 `PLANS.md` 或 docs 同步、影响 planner / intent policy / 记忆召回 / 前端交互的小到中等改动。
- `高`：适合主聊天链路、记忆写入门控、统一语义分类、API 兼容、eval 指标、audit 根因分析、跨模块行为变化，或已经返工一次且副作用不确定的任务。
- `超高`：只建议用于长期疑难调试、大范围架构方案、完整代码审查、复杂迁移计划，或需要在多个方案之间做高风险取舍的任务。

开始任务前用一句话说明理由，例如：“建议推理强度：中，因为这次改动涉及 planner 和测试，但没有触碰 API 兼容或主链路架构。”

## 记忆与隐私边界

必须区分：

- 用户长期记忆：SQLite `memories` 表，按 `user_id` 隔离。
- 当前 turn 临时上下文：定位、planner、召回结果、web 结果、debug，不默认持久化。
- Hermes 自身记忆：`MEMORY.md`、provider recall、session search，不等于本 demo 的用户个人记忆。

写长期记忆前必须经过候选和门控：

```text
MemoryWriteCandidate
-> should_write_memory_candidate()
-> saved / rejected / requires_confirmation
-> add_memory() 或 merge_memory_evidence()
```

敏感信息默认不能静默保存，包括密码、验证码、银行卡、API key、token、证件号等。提升自然偏好和事件写入覆盖时，必须同时保留敏感信息负例。

位置是实时设备状态，只能用于当前 turn。除非用户明确表达要保存，否则不要把当前位置写入长期记忆。

## 验证命令

优先在当前独立仓库根目录执行：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

运行 live eval：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```

如果导入 `server.py` 或运行服务时遇到默认 home 不可写，先设置临时 `AI_GLASSES_HOME`：

```bash
AI_GLASSES_HOME=/tmp/ai-glasses-test conda run -n hermes python -m pytest tests -q
```

## 本地运行

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

默认地址：

```text
http://127.0.0.1:8765
```

需要局域网设备测试语音/定位时，优先用 HTTPS 参数启动。浏览器语音和定位失败时，先判断是否是浏览器权限或国内网络访问 speech service 问题，不要先改后端。

## 文档维护规则

文档分工：

- `AGENTS.md`：Codex 工作规则和验证入口。
- `PLANS.md`：当前阶段、优先级、验收标准和下一步开发顺序。
- `.planning/`：`planning-with-files` skill 的任务级 `task_plan.md`、`findings.md`、`progress.md`，一个任务一个子目录。
- `docs/context/README.md`：开发者上手入口，包含启动、配置、依赖、测试和 legacy fallback 边界。
- `docs/context/code-map.md`：按任务找代码入口。
- `docs/context/system-flow-current.md`：当前系统架构、真实调用链和能力边界。

`docs/context` 只保留这三份当前态文档。不要新增专题流水账、历史复盘、调研过程或 HTML 汇报文档；旧经验如果仍重要，必须转成当前规则或当前边界后写入三文档。

代码行为变化时同步更新相关文档。不要把计划中能力写成已完成；不要把 target eval 或 demo stub 写成生产能力。

使用 `planning-with-files` 时，不要在仓库根目录散放 `task_plan.md`、`findings.md`、`progress.md`。新任务用 skill 自带脚本创建 `.planning/<date>-<slug>/`，并通过 `.planning/.active_plan` 或 `PLAN_ID` 指定当前活跃计划。

## Git 工作习惯

除非用户明确要求，不要自动提交。

用户偏好“小而多次”。一个功能完成并验证后，主动提醒可以准备提交和 push，再继续下一个功能，避免工作区混入太多互不相关的改动。
