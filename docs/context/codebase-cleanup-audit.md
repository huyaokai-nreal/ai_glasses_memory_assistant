# 工程可读性与冗余治理审计

更新时间：2026-07-08。本文是只读工程审计报告，用来回答“当前仓库哪里最影响同事接手、哪些冗余可以安全清理、下一阶段应该怎么小刀推进”。本轮没有修改主链路代码，也没有删除生产逻辑。

## 审计目标

目标：

- 建立当前仓库的工程可读性标准。
- 标出正式入口、主链路核心、legacy / fallback / 实验性边界。
- 找出最影响同事接手的冗余、文档漂移和清理候选。
- 给出可以逐步执行、可测试、可回滚的下一阶段路线。

成功标准：

- 候选问题有证据来源，不靠“看起来没用”判断。
- 生产代码只记录候选，不直接删除。
- 低风险缓存和临时文件与真实运行产物分开。
- `PLANS.md` 能据此安排下一阶段小刀治理。

## 本轮只读证据

已查看：

- 顶层入口：`README.md`、`AGENTS.md`、`PLANS.md`、`server.py`、`app.py`。
- 主链路与 API 入口：`ai_glasses_memory_assistant/server.py`、`ai_glasses_memory_assistant/app.py`、`ai_glasses_memory_assistant/agent_bridge.py`。
- 上下文文档：`docs/context/README.md`、`code-map.md`、`pipeline.md`、`current-status-and-gaps.md`。
- 代码规模：`wc -l ai_glasses_memory_assistant/*.py tests/*.py server.py app.py PLANS.md docs/context/*.md`。
- 文档和 legacy 关键词：`rg "hermes-agent|HERMES_HOME|AIAgent|legacy|fallback|router|intent_classifier|planner/router"`。
- 工作区状态：`git status --short`、`git diff --check`，当前无未提交 diff，diff check 通过。

本轮尝试运行清理 skill 的只读扫描脚本，但脚本仍绑定旧的 `hermes-agent/ai_glasses_memory_assistant` 根目录，当前独立仓库根目录下返回：

```text
Not an ai_glasses_memory_assistant root: /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
```

这本身是一个治理信号：部分辅助工具仍带旧仓库路径假设，应作为后续工具清理项处理。

## 当前专业标准

正式入口：

- 默认本地运行入口：`python -m ai_glasses_memory_assistant.server`。
- 标准库 HTTP server：`ai_glasses_memory_assistant/server.py`。
- 可选 FastAPI 入口：`ai_glasses_memory_assistant/app.py`。
- 根目录 `server.py`、`app.py` 只是兼容薄入口。
- Web UI 静态资源在 `static/`，两个 server 入口都应复用同一套 service 行为。

主链路核心：

- `GlassesChatService.chat()` 是聊天主 service。
- `turn_planner.py` 负责本地 fast path、baseline、确定性时间范围和把 `PreReplyDecision` 落成执行计划。
- `turn_semantic_classifier.py` 是非 fast path 的单次回复前决策来源。
- `intent_policy.py` 是长期记忆写入门控。
- `memory_store.py`、`timeline_store.py` 分别负责结构化记忆和原文证据。
- `memory_recall_arbitration.py` 负责多来源召回后的主证据仲裁。

legacy / fallback / 实验性边界：

- Hermes fallback 仍是 sealed legacy 路径，只应在显式双开关下用于迁移期对照。
- `HERMES_HOME` 仍是迁移期路径兼容，不应作为新部署默认说明。
- `pyproject.standalone.toml` 是独立打包草案，不是当前正式构建入口。
- `/api/reminders/check` 是手动检查，不是主动提醒 runtime。
- `weekly_report()` 是启发式草稿，不是完整项目知识图谱。
- 音频、多人会话和 ambient capture 已有若干文本级或 demo 级切片，但不是生产级 always-on runtime。

测试门禁：

- 文档或只读治理至少跑 `git diff --check`。
- Python 改动至少跑 `py_compile` 和相关最小 unittest。
- 主链路行为改动优先跑 `tests/test_agent_bridge_policy.py` 的 focused 集合，再视风险跑全部 tests。
- eval 或 benchmark 只能在数据和配置齐备时声明实际结果，不能把计划写成已验证。

## 主要问题清单

### P0: Service 巨石影响接手

`ai_glasses_memory_assistant/agent_bridge.py` 约 13k 行，包含聊天主链路、音频处理、speaker enrollment、文档召回、导入、capture、weekly report、reminder、memory job、audit、纠错、observation 等多类职责。

影响：

- 新同事很难判断改一个功能应该读哪个区域。
- 审查 diff 时容易误碰无关路径。
- 同一个文件里混有主链路、实验能力和 legacy fallback，边界成本高。

建议：

- 暂不大拆。
- 先在文档中建立区域索引和“可动/慎动/暂不动”边界。
- 后续每次只抽一个低耦合区块，例如 audio segment processor、document recall helper 或 weekly report helper，并先补最小 API/测试保护。

### P0: 主测试文件同样过大

`tests/test_agent_bridge_policy.py` 约 15k 行，是当前主门禁，但已经承载了 planner、pre_reply、memory write、document、timeline、audio、explanation、job 等大量语义族。

影响：

- 运行 focused 测试时难以定位已有覆盖。
- 新增测试容易重复或放错位置。
- 审计“某行为有没有门禁”成本高。

建议：

- 不先搬迁测试。
- 先做测试索引文档或文件内分区清单。
- 后续按语义族拆出低风险测试文件，例如 `test_agent_bridge_explanation.py`、`test_agent_bridge_documents.py`，每刀只移动一组，并保持原断言不变。

### P0: 文档存在时间漂移

`docs/context/current-status-and-gaps.md` 更新时间仍是 2026-05-27，其中仍写 `planner/router` 和本地 planner / structured router 旧表述；`code-map.md` 也仍有 `planner/router` 流程描述。当前代码与 `PLANS.md` 已显示独立 router 删除、`PreReplyDecision` 收口。

影响：

- 新同事按文档读，会先找已经删除或降权的 router 概念。
- 容易把 planner 当成开放语义主权层继续堆规则。

建议：

- 下一刀优先做文档真相同步：更新 `current-status-and-gaps.md`、`code-map.md`、`pipeline.md` 中的旧 `router/AIAgent/hermes-agent checkout` 表述。
- 只改文档，不改代码。

### P1: 独立仓库与旧 Hermes 路径说明混杂

文档中同时存在“当前推荐从独立仓库根目录运行”和“仍建议从外层 `hermes-agent` checkout 执行”的历史表述。`PLANS.md` 已记录第二阶段第一刀把推荐运行路径切到当前独立仓库，但部分文档尚未完全收敛。

影响：

- 启动和测试命令容易跑错 cwd。
- 新同事可能误以为当前仍必须依赖外层 Hermes checkout。

建议：

- 统一启动说明：当前默认以 `/Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant` 为仓库根目录。
- 保留 Hermes fallback 说明，但明确为 legacy / 迁移期对照。

### P1: 辅助清理脚本仍绑定旧路径

`ai-glasses-code-cleanup` skill 的扫描脚本在当前独立仓库根目录拒绝执行，说明工具层仍带旧目录判断。

影响：

- 后续周期性清理不能直接复用。
- 容易把独立仓库和旧子目录状态混在一起。

建议：

- 后续单独修 skill 脚本，支持当前独立仓库根目录。
- 不把 skill 脚本修复混入业务代码治理刀。

### P1: 低风险可再生缓存存在

当前可见：

- `__pycache__/`
- `ai_glasses_memory_assistant/__pycache__/`
- `ai_glasses_memory_assistant/evals/__pycache__/`
- `tests/__pycache__/`
- `docs/.DS_Store`
- `.pytest_cache/`

影响：

- 不影响运行，但干扰仓库体感和扫描输出。

建议：

- 可以作为第一批低风险清理，但清理前确认 `.gitignore` 覆盖。
- 不删除 reports、数据库、audit、certs、`.env` 或 benchmark 数据。

### P1: API 双入口一致性需要长期门禁

`server.py` 和 `app.py` 都是薄 HTTP 层，但两边手写了大量相同 API。当前职责方向是正确的：都调用 `GlassesChatService`。风险在于未来新增 API 时只改一边。

影响：

- 标准库 server 与 FastAPI 入口可能慢慢漂移。

建议：

- 后续新增 API 时必须同步两个入口。
- 可以增加轻量文档一致性或路由清单测试，不必马上抽象成新框架。

### P1: legacy / fallback 命名噪声高

代码和文档中大量 `legacy`、`fallback` 既包含正常降级策略，也包含迁移期 Hermes fallback、旧语义路径、规则兜底和 debug 兼容字段。

影响：

- 新同事看到 fallback 不知道是“必须保留的安全降级”还是“可删除旧债”。

建议：

- 先给 fallback 分类：runtime safety fallback、semantic rule fallback、migration legacy fallback、debug compatibility fallback。
- 删除前必须先找调用链和测试保护。

### P2: `PLANS.md` 历史补记较长

`PLANS.md` 已承担大量已完成流水账，虽然能保留历史，但同事想找“下一刀”时需要越过很多内容。

影响：

- 当前行动焦点不够突出。

建议：

- 不删除已完成历史。
- 可以新增更靠前的“当前治理阶段 / 下一刀”小节，并把长历史逐步沉到专题 docs 中。

### P2: 报告、HTML、上下文文档数量多

`docs/context`、`docs/html`、`docs/reports` 已经形成多类材料，但部分文档的当前性不一致。

影响：

- 不知道哪些是正式上下文，哪些是历史报告或可视化材料。

建议：

- `docs/context/README.md` 已有地图，后续应增加“当前性状态”：current、needs-refresh、historical。
- 对 stale 文档只标状态，不急着全文重写。

## 清理候选排序

可以较安全清理：

- `__pycache__/`
- `.pytest_cache/`
- `docs/.DS_Store`
- 已合并到 `PLANS.md` 或 `docs/context` 的临时 planning 文件。本轮未发现根目录 `task_plan.md`、`findings.md`、`progress.md`。

暂留但要标记：

- Hermes sealed legacy fallback。
- `HERMES_HOME` 迁移期兼容。
- `pyproject.standalone.toml` 草案。
- reports 下的 eval 报告。
- certs 下的本地 HTTPS 证书。

需要测试保护后再动：

- `agent_bridge.py` 内 document recall、weekly report、audio processor、memory job、explanation、correction、observation 等 helper。
- `tests/test_agent_bridge_policy.py` 的语义族拆分。
- `server.py` / `app.py` 的重复 API 包装。
- `turn_planner.py` 中仍保留的 fast path / deterministic temporal / baseline guard。

本轮禁止删除：

- API 行为、数据库 schema、audit 格式、eval reports、真实用户数据、`.env`、证书、benchmark 数据、Hermes fallback 运行逻辑。

## 建议下一阶段路线

第一刀：文档真相同步。

- 更新 `current-status-and-gaps.md`、`code-map.md`、`pipeline.md` 的旧 router / AIAgent / hermes-agent cwd 表述。
- 目标是让同事第一小时读到的入口和当前代码一致。
- 验证：`git diff --check`，必要时跑文档一致性测试。

第二刀：低风险工作区清理。

- 删除可再生缓存和 `.DS_Store`。
- 确认 `.gitignore` 已覆盖。
- 不动 reports、certs、`.env`、数据库和 audit。

第三刀：测试门禁地图。

- 给 `tests/test_agent_bridge_policy.py` 建语义族索引。
- 暂不搬测试代码。
- 目标是新增测试前能快速判断放哪里、是否已有覆盖。

第四刀：service 巨石分区地图。

- 给 `agent_bridge.py` 建区域索引：主 chat、audio、documents、import/capture、jobs/audit、explanation、correction、observation。
- 暂不拆文件。
- 目标是降低审查和定位成本。

第五刀：选择一个低耦合 helper 做小拆分预演。

- 优先候选：weekly report helper、document recall helper、audio segment processor 三选一。
- 先补或确认测试，再搬迁。
- 每次只搬一个能力族。

## 大白话总结

这个仓库现在最大的问题不是“不能跑”，而是“能跑的东西太多都挤在几个大文件和一堆历史文档里”。同事接手时最容易迷路的地方有三个：不知道真正入口在哪里，不知道哪些 fallback 是保命降级还是旧债，不知道主链路现在到底由 planner 还是 `PreReplyDecision` 说了算。

下一阶段不要先删代码。先把路牌立好：入口路牌、主链路路牌、legacy 路牌、测试门禁路牌。等路牌清楚后，再一刀一刀拆废线。这样每次清理都能解释清楚：同事少读了哪段旧说明、少踩了哪个 cwd 坑、少在巨石文件里翻多久。
