# 代码地图

本文只说明当前代码入口和高风险修改区域，帮助开发者少走弯路。不要根据旧架构名猜逻辑；当前主链路以代码为准。

## 主聊天链路

入口：

- 标准库 HTTP：`ai_glasses_memory_assistant/server.py`
- 业务 service：`ai_glasses_memory_assistant/agent_bridge.py`

核心路径：

```text
POST /api/chat
-> GlassesChatService.chat()
-> plan_turn()
-> classify_pre_reply_decision()
-> TurnPlan.apply_pre_reply_decision()
-> 按需召回 profile/event/timeline/document/location/web
-> 本地回复或 OpenAI-compatible LLM
-> 同步保存或后台 memory job
-> TimelineStore / chat_audit.jsonl
-> response
```

当前主权决策是 `PreReplyDecision`。不要恢复旧 router，也不要把开放语义继续堆到 planner 短语规则里。

## 改功能先看哪里

| 任务 | 先读文件 | 注意点 |
| --- | --- | --- |
| 聊天行为 | `agent_bridge.py`、`turn_planner.py`、`turn_semantic_classifier.py` | 优先改 service 层；HTTP 入口只做薄包装。 |
| 记忆写入 | `memory_candidate.py`、`intent_policy.py`、`memory_store.py` | 候选必须经过 `should_write_memory_candidate()`，敏感信息不能静默保存。 |
| 记忆召回 | `memory_store.py`、`timeline_store.py`、`memory_recall_arbitration.py` | 召回只服务当前 turn，不改 system prompt，不默认读取全部历史。 |
| 文档导入/召回 | `agent_bridge.py`、`memory_store.py` | Markdown 文档保存完整原文；文档细节问题必须读原文或片段。 |
| 原话证据 | `timeline_store.py`、`agent_bridge.py` | timeline 是原始证据，不等于长期结构化记忆。 |
| 时间与计划 | `turn_planner.py`、`temporal_parser.py`、`memory_store.py` | 简单 day/hour 时间优先本地解析；复杂表达走 LLM fallback。 |
| Web/位置 | `web_search.py`、`agent_bridge.py`、`static/app.js` | 位置是当前 turn 临时状态；实时问题必须基于工具状态，不要编结果。 |
| 后台 job | `agent_bridge.py`、`timeline_store.py`、`static/app.js` | 当前是 demo 级 job 状态持久化，不是可靠 worker 队列。 |
| Import/Capture | `agent_bridge.py`、`server.py` | 新输入来源应复用统一候选、门控、去重、写库流程。 |
| 周报/提醒 | `agent_bridge.py`、`evals/runner.py` | 周报是启发式草稿；提醒是手动检查接口，不是主动 runtime。 |
| 前端 | `static/index.html`、`static/app.js`、`static/styles.css` | 检查移动端文本、debug 展示、job 轮询、语音/定位失败状态。 |
| 本地运行配置 | `app_home.py`、`env_loader.py`、`llm_client.py` | 默认 home 是 `AI_GLASSES_HOME`；Hermes backend 只允许显式 legacy fallback。 |

## 高风险区域

- `agent_bridge.py` 很大，包含聊天、导入、capture、音频、speaker、周报、提醒、audit、解释、job。改动前先定位具体方法，避免顺手重构。
- `server.py` 是唯一 HTTP 包装入口。新增 API 时保持薄包装，把业务逻辑放在 service 层。
- `memory_store.py` 和 `timeline_store.py` 管 SQLite schema、搜索、删除和 evidence。不要随意改字段或删除逻辑。
- `tests/` 只保留核心保险丝。新增功能由负责同事补专项测试，不把历史功能回归重新堆回默认门禁。
- `evals/scenarios.jsonl` 是 live eval 场景。不要把临时验证样例直接写成 strict 门禁。

## 常用测试

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

默认单元测试分三类：`tests/test_core_startup.py`、`tests/test_core_storage.py`、`tests/test_core_chat.py`。

live eval：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
