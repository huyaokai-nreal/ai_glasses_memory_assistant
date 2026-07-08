# Codex 上下文地图

本目录给后续 Codex 快速接手当前 AI 眼镜记忆助手 demo 使用。源码仍是最终事实；文档只记录当前理解、调用链、边界和专题知识，不替代代码。

## 文档分工

当前仓库文档按下面分工协作：

| 位置                  | 作用         | 该写什么                            | 不该写什么                 |
| ------------------- | ---------- | ------------------------------- | --------------------- |
| `../../AGENTS.md`   | Codex 工作手册 | 入口文件、验证命令、工程边界、修改原则             | 长篇产品规划、实现流水账          |
| `../../PLANS.md`    | 当前路线图和交接面  | 当前阶段、P0/P1、下一刀、验收标准、暂不做         | 函数级细节、HTML 制作进度、长报告全文 |
| `docs/context/*.md` | 长期上下文和专题知识 | 调用链、机制、边界、专题方案、专题清单             | 临时任务进度、单次修复流水账        |
| `docs/reports/*.md` | 阶段性报告      | 压测证据、问题地图、复盘结论、汇报草稿             | 当前路线图入口               |
| `docs/html/*.html`  | 人类阅读版可视化   | 汇报、概览、交互式解释                     | Codex 唯一依赖事实源         |
| `.planning/*`       | 临时任务工作文件   | task plan / findings / progress | 长期保留知识；任务结束后应合并收尾     |

判断规则：

- 想知道“现在最该做什么”，看 `PLANS.md`。
- 想知道“系统实际上怎么工作”，看 `docs/context/*.md`。
- 想知道“这次压测具体发现了什么”，看 `docs/reports/*.md`。
- 想做人类演示或汇报，才看 `docs/html/*.html`。

## 目录职责

主目录默认只保留包根、项目入口、顶层说明和构建配置。后续看到下面这些目录时，可以直接按角色找：

| 位置 | 作用 |
| --- | --- |
| `ai_glasses_memory_assistant/` | 正式 Python 包代码，包含 service、store、planner、server、FastAPI 和 eval runner Python 模块。 |
| `certs/` | 本地 HTTPS 测试证书与密钥。 |
| `docs/` | 长期上下文、人类说明、HTML 和报告草稿。 |
| `tests/` | 单元测试和 service 级测试。 |
| `evals/` | 评估 runner、adapter、metrics 和场景文件。 |
| `static/` | Web UI 静态资源。 |
| `scripts/` | 辅助脚本。 |
| `data/` | 基准数据和样例数据。 |
| `reports/` | 运行产物和评估报告。 |

## 先读顺序

| 顺序 | 文档                                    | 什么时候读                                                                             |
| -- | ------------------------------------- | --------------------------------------------------------------------------------- |
| 1  | `../../AGENTS.md`                     | 每次开始改代码前先读，确认工作规则、环境和验证命令。                                                        |
| 2  | `../../PLANS.md`                      | 判断当前优先级、P0/P1、验收标准和暂不做范围。                                                         |
| 3  | `current-status-and-gaps.md`          | 需要知道当前代码做到哪一步、缺口在哪里时读。                                                            |
| 4  | `code-map.md`                         | 需要按任务定位入口文件和函数时读。                                                                 |
| 5  | `pipeline.md`                         | 改 `/api/chat`、import、capture、weekly report、reminder 时读。                           |
| 6  | `system-flow-current.md`              | 需要快速理解当前系统全景、单轮聊天、记忆流转、收音待机和数据落点时读。                                          |
| 7  | `memory-mechanism.md`                 | 改记忆 schema、写入门控、召回、隐私、去重时读。                                                       |
| 8  | `eval-coverage.md`                    | 需要判断 eval 是否覆盖 AI 眼镜真实使用场景、规划 target 场景时读。                                        |
| 9  | `phrase-rule-inventory.md`            | 治理短语匹配、关键词、正则和 rule fallback 时读，确认哪些规则可硬拦、哪些只能做弱信号。                               |
| 10 | `unified-semantic-classifier-plan.md` | 需要重构 planner/pre_reply_decision/legacy fallback 的语义职责，或讨论“是否统一让 LLM 做一次开放语义判定”时读。 |
| 11 | `codebase-cleanup-audit.md`           | 进入工程可读性、冗余治理、legacy/fallback 清理或同事接手标准建设时读；这是只读审计和下一阶段路线，不是已删除代码清单。             |
| 12 | `codex-hardcode-review-checklist.md`  | 需要快速判断某次 Codex 改动是不是在长 hard code / patch，而不想逐行全审时读。                               |
| 13 | `ambient-audio-wakeword-plan.md`      | 需要规划“持续收音待机、唤醒式现场问答、原话时间线、轻量情绪判断”时读；这是未来专项规划，不是当前已实现能力。                           |
| 14 | `ambient-audio-v2-open-source-research.md` | 需要为持续收音改进 V2 选择 VAD、ASR、diarization、speaker embedding、wake word 或多人记忆建模路线时读；这是阶段 B 调研结论，不是已接入依赖。 |
| 15 | `standalone-migration.md`             | 需要把 demo 拆成独立产品、独立部署、独立开源项目，或替换 Hermes 依赖时读。                                    |
| 16 | `standalone-startup.md`               | 需要独立启动 demo、配置 `AI_GLASSES_HOME` / `.env` / OpenAI-compatible backend，或确认 Hermes fallback 边界时读。 |
| 17 | `standalone-deployment.md`            | 需要准备独立仓库部署、dependency manifest 草案或文件搬家范围时读。                    |
| 18 | `standalone-packaging.md`             | 需要查看独立 `pyproject` 草案、optional dependencies、console scripts 或 package data 时读。           |
| 19 | `local-audio-model-migration.md`      | 迁出 Hermes 后需要确认本地 ASR、声学情绪、声纹模型依赖、模型目录和环境变量时读。                                 |
| 20 | `demo-features.md`                    | 改 API、前端展示、demo 边界或对外说明时读。                                                        |
| 21 | `production-readiness.md`             | 需要判断“当前离生产级完整 AI 眼镜记忆系统还有多远”时读。                                                   |

## 文档状态

默认把 `docs/context` 里的文档分成三类：

| 类型      | 文档                                                                                                                                                                                                                                                                                                                                                    | 用法                              |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 正式上下文   | `current-status-and-gaps.md`、`code-map.md`、`pipeline.md`、`system-flow-current.md`、`memory-mechanism.md`、`eval-coverage.md`、`demo-features.md`、`production-readiness.md`、`phrase-rule-inventory.md`、`codebase-cleanup-audit.md`、`ambient-audio-wakeword-plan.md`、`ambient-audio-v2-open-source-research.md`、`standalone-migration.md`、`standalone-startup.md`、`standalone-deployment.md`、`standalone-packaging.md`、`local-audio-model-migration.md`、`unified-semantic-classifier-plan.md`、`codex-hardcode-review-checklist.md` | 可以直接作为当前代码和当前路线的参考，但遇到冲突仍以代码为准。 |

## 当前一句话定位

这是一个 Stage 2 可运行原型：用 Web/语音模拟 AI 眼镜入口，验证“自然输入 -> 回复优先 -> 后台记忆沉淀 -> 后续按需召回 -> 可调试可管理”的个人上下文闭环。纯文字输入下的底层记忆内核已形成第一版闭环，但真实眼镜 runtime、原生 App、音频上传/ASR、主动提醒和生产级可靠性仍未完成。

新的“持续收音待机 + 唤醒式现场问答”方向已单独沉淀到 `ambient-audio-wakeword-plan.md`：它规划 demo 打开后可见收音待机、ASR 多人原文时间线、唤醒后 query、轻量情绪元数据和严格长期记忆门控。当前仍是规划，不能把它写成已完成的后台常驻收音或真实唤醒词能力。阶段 B 开源调研结论见 `ambient-audio-v2-open-source-research.md`。

## 代码真相入口

| 文件                             | 职责                                                                          |
| ------------------------------ | --------------------------------------------------------------------------- |
| `ai_glasses_memory_assistant/agent_bridge.py`              | 主 service：聊天、召回、回复、后台 job、import、capture、周报、提醒、audit。                       |
| `ai_glasses_memory_assistant/turn_planner.py`              | 本地执行编排器：fast path、少量确定性 guard、时间范围 baseline，并把 `PreReplyDecision` 落成执行计划。   |
| `ai_glasses_memory_assistant/intent_policy.py`             | 长期记忆写入门控和短确认回复 helper。                                                      |
| `ai_glasses_memory_assistant/turn_semantic_classifier.py`  | 单次 `PreReplyDecision`，同时输出回复模式、召回类型、`recall_goal`、web/location 和记忆候选字段。     |
| `ai_glasses_memory_assistant/memory_recall_arbitration.py` | 召回后仲裁 document、raw timeline、structured memory、observation 和 profile 谁作为主证据。 |
| `ai_glasses_memory_assistant/memory_store.py`              | SQLite 记忆表、搜索、去重、合并证据、软删除。                                                  |
| `ai_glasses_memory_assistant/timeline_store.py`            | SQLite 原文时间线、raw turn、capture chunk、全文搜索和 evidence chunk。                   |
| `ai_glasses_memory_assistant/server.py`                    | 标准库 HTTP demo 入口。                                                           |
| `ai_glasses_memory_assistant/app.py`                       | FastAPI 入口，API 行为应与 `server.py` 对齐。                                         |
| `server.py`、`app.py`                       | 根目录兼容薄入口，只负责转发到正式包入口。                                         |
| `static/`                      | Web UI、语音、TTS、定位、debug、job 轮询。                                              |
| `evals/`                       | live-LLM 离线评估场景、runner、report。                                              |
| `tests/`                       | 单元测试和 service 级测试。                                                          |

## 文档维护规则

- 不要把计划中能力写成已完成。
- 不要把 target eval 写成当前门禁已通过。
- 不要把进程内 job/capture 写成生产级持久队列。
- 不要把专题实现流水账或 persona 压测原文堆进 `PLANS.md`。
- 改代码行为时同步更新相关文档。
- 文档中文优先，短句、表格、调用链优先，避免长篇抽象描述。

## 常用验证

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m unittest discover tests -q
```

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
