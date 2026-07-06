# 外部记忆系统借鉴对照

更新时间：2026-06-29。本文用于把近年论文和开源项目里的记忆机制，转成当前 AI 眼镜记忆助手可以验证的小实验。它不是接入方案，也不表示本项目已经采用这些外部系统。

## 目标

当前借鉴目标不是“换一个记忆库”，而是补强本项目已经在做的记忆生命周期治理：

- 记忆从哪一句话、哪个 turn、哪个 evidence 来。
- 用户改口后旧记忆如何降级、覆盖或保留为历史。
- 召回时为什么选择这条记忆，而不是旧记忆或临时上下文。
- 哪些输入只适合留在 raw timeline，不应进入长期结构化记忆。
- 借鉴点必须能落成最小测试、target eval 或 audit/debug 证据。

大白话：先学别人怎么给记忆贴来源、有效期、关系和检查项，不急着搬别人整套系统。

## 外部资源对照

| 资源 | 值得借鉴的机制 | 本项目对应位置 | 暂不采用 | 最小实验 |
| --- | --- | --- | --- | --- |
| Graphiti / Zep temporal knowledge graph | temporal facts、episode provenance、增量更新、混合召回 | `memory_store.py`、`timeline_store.py`、`source_trace`、audit | 不直接引入图数据库或外部服务 | 给结构化记忆补强来源、覆盖关系和有效状态表达 |
| Mem0 | 用户级、会话级、agent 级多层记忆；长期记忆产品化边界 | `memory-mechanism.md` 四层边界、`runtime_recall`、`structured_memory` | 不替换 `EventMemoryStore` | 明确 user profile、session context、runtime state 的字段/调试边界 |
| A-MEM | 记忆卡片、tag、相关记忆链接、演化更新 | `MemoryWriteCandidate`、`memories.tags`、dedupe/correction | 不做复杂自动知识图谱链接 | 先把候选和保存记录对齐到“内容 + 类型 + 来源 + 关系”的轻量卡片形态 |
| Letta / MemGPT | stateful agent、memory blocks、显式 agent 状态 | `PreReplyDecision`、system context、assistant/user preference | 不迁移到完整 agent 平台 | 只参考显式状态块的组织方式，避免隐式 prompt 堆叠 |
| MemoryAgentBench / LongMemEval 等评估 | 长期记忆能力分类：召回、更新、测试时学习、长程理解、选择性遗忘 | `evals/`、`tests/`、`chat_audit.jsonl` | 不把 benchmark 原样搬进 demo | 把第一阶段测试分成写入来源、改口覆盖、拒写、可解释召回 |

参考入口：

- Graphiti: <https://github.com/getzep/graphiti>
- Zep paper: <https://arxiv.org/abs/2501.13956>
- Mem0: <https://github.com/mem0ai/mem0>
- Mem0 paper: <https://arxiv.org/abs/2504.19413>
- A-MEM: <https://github.com/agiresearch/A-mem>
- A-MEM paper: <https://arxiv.org/abs/2502.12110>
- Letta: <https://github.com/letta-ai/letta>

## 第一阶段范围

第一阶段只做轻量 `provenance + supersession` 准备，不引入新依赖、不接外部服务、不换 SQLite 存储。

要验证的核心问题：

```text
这条记忆从哪来？
它现在是否仍是 active？
它有没有被后来的记忆覆盖？
回答时能不能解释为什么用新记忆，而不是旧记忆？
```

示例：

```text
旧输入：我喜欢冰美式。
新输入：我最近不喝咖啡了。

期望：
- 旧偏好不要被物理删除。
- 新状态应成为当前回答的优先依据。
- 旧记忆可以标记为 superseded/stale 或在召回排序中被压低。
- debug/audit 能解释这次回答采用了新状态。
```

## 当前项目映射

当前项目已经有一些可承接外部机制的基础：

| 已有基础 | 可承接的借鉴点 |
| --- | --- |
| `source` / `source_id` / `ingestion_id` / `evidence_ids` | provenance、episode/source trace |
| `status=active/superseded/stale/deleted` | supersession、有效状态 |
| `confidence` / `strength` / `effective_strength` | 排序解释、置信和时间衰减 |
| `TimelineStore` raw timeline | 原话证据、episode 来源 |
| `PreReplyDecision` | 写入、召回、临时/不要记、纠错的统一语义入口 |
| `chat_audit.jsonl` / debug payload | 可解释召回和失败解释 |

准备阶段应先盘点现有字段是否足够表达第一阶段，不默认新增 schema。只有当现有 `status/evidence/source` 无法支持清晰行为时，再考虑最小字段变更。

## 建议最小测试组

第一阶段实现前，先准备或确认这些测试/target：

| 场景 | 输入例子 | 期望 |
| --- | --- | --- |
| 显式写入来源 | “帮我记一下，我喜欢低糖拿铁。” | 保存结构化记忆，并能从 debug/audit 看到来源 turn/evidence。 |
| 拒绝写入 | “这个只是随口吐槽，别记。” | 不进入长期结构化记忆，但 raw timeline 可保留。 |
| 改口覆盖 | “我以前喜欢冰美式，但现在不喝咖啡了。” | 新状态优先，旧偏好不再作为当前答案主依据。 |
| 可解释召回 | “我现在喜欢喝什么？” | 回答能避免旧记忆误导，并能解释采用的新依据。 |
| 敏感拒写 | “帮我记一下 API key 是 ...” | 被安全门控拒绝，解释为 sensitive rejection。 |

## 暂不做

- 不直接接入 Graphiti、Mem0、Letta 或任何外部记忆服务。
- 不新增图数据库、向量库或后台 worker。
- 不把 raw timeline 全量注入主模型。
- 不绕过 `should_write_memory_candidate()`、dedupe、correction、safety gate。
- 不为了某个 query 写硬编码短语补丁。

## 下一步

1. 对照 `memory_store.py`、`timeline_store.py`、`agent_bridge.py` 和现有测试，确认第一阶段字段缺口。
2. 先补最小测试或 target：显式写入来源、改口覆盖、可解释召回。
3. 再决定是否需要 schema 最小增量，或只用现有 `status/source/evidence/tags` 完成。
4. 实现后同步 `memory-mechanism.md` 和 `PLANS.md`，并把验证命令写清楚。
