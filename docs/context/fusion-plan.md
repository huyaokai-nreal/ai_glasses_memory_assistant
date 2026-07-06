# 记忆系统融合方案（历史候选）

> 状态：历史候选方案，不是当前主线，不应默认视为现行实现或当前开发计划。
>
> 用途：保留当时关于“向量索引 / 实体图谱 / 认知层融合”的设计思路，供以后重新评估时参考。
>
> 当前主线请优先看 `PLANS.md`、`current-status-and-gaps.md`、`pipeline.md` 和 `memory-mechanism.md`。
>
> 当前约束：仓库现状并没有按本文落地 FAISS、实体图谱、外部向量库或新的多路检索基础设施。

---

## 一、融合后总体架构

```
用户输入
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Layer 0: Timeline 证据底座（已有，不动）          │
│   TimelineStore — 原始对话原文永不丢              │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Layer 1: 安全门控（从 Demo 保留，增强）            │
│   intent_policy — 敏感词/隐私/瞬时上下文拦截      │
│   plan_turn  — 本地规则快速路由（不调 LLM）        │
└─────────────────────────────────────────────────┘
  │
  ├── 本地回复路径（greeting/identity/profile/event_ack）
  │
  ▼ 需要 LLM 时
┌─────────────────────────────────────────────────┐
│ Layer 2: 记忆存储（融合双方）                      │
│   EventMemoryStore（已有一半）                     │
│   + FAISS 向量索引（从自研引入）                   │
│   + 实体知识图谱（从自研引入）                     │
│   + 去重/合并/软删除（Demo 已有）                  │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Layer 3: 多路检索（融合）                          │
│   FTS5（Demo 已有）+ FAISS 向量（自研引入）        │
│   + 实体图谱 BFS（自研引入）                       │
│   + 按 fact_type 分桶召回                          │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Layer 4: 认知归纳（从自研引入，异步）              │
│   Observation 生成 — 跨回合归纳稳定模式            │
│   reflect — 定期整理、时间衰减、实体合并            │
└─────────────────────────────────────────────────┘
```

---

## 二、分模块融合方案

### 2.1 Layer 0：Timeline 证据底座（保留 Demo）

**不动。** 当前 TimelineStore 已经做得很好了：
- 独立 `timeline.db`，不污染结构化记忆
- 原始对话原文 + 助手回复完整保留
- FTS5 支持"原话"检索

**微调：** 让自研系统的 narrative fact 通过 `evidence_ids` 关联回 timeline chunk，可追溯"这条记忆是从哪句话提炼的"。

---

### 2.2 Layer 1：安全门控（保留 Demo，小幅增强）

当前 Demo 的门控是核心优势，**全部保留**：

#### 保留项
| 组件 | 文件 | 作用 |
|---|---|---|
| 本地规划器 | `turn_planner.py` | 关键词规则快速路由，不调 LLM |
| 写入门控 | `intent_policy.py` | 敏感词/问题/瞬时上下文/低置信度拦截 |
| 门控结果 | `MemoryWriteGateResult` | allowed / requires_confirmation / rejected |

#### 增强项
1. **门控规则可配置化**：把 `POLICY_TERMS` 中的触发词移到配置文件，支持热更新
2. **自研的 LLM fact 提取进来时，同样过门控**：LLM 提炼的 narrative fact 不等于就能直接写库，必须经过 `should_write_memory_candidate` 过滤

---

### 2.3 Layer 2：记忆存储（融合核心）

#### 2.3.1 当前 EventMemoryStore 保留
- SQLite `memories` 表保持不动
- 去重 `find_similar_memory`、证据合并 `merge_memory_evidence`、软删除 全部保留
- 已有机读化（增量列迁移、兼容旧数据）

#### 2.3.2 新增：向量索引（从自研引入）

```
EventMemoryStore
  │
  ├── memories 表（SQLite，已有）
  ├── memories_fts（FTS5，已有）
  └── memories_faiss（新增）  ← 独立 FAISS 索引
```

**实现要点：**
- 每次 `add_memory` 后，异步生成 embedding 并写入 FAISS
- FAISS 索引文件落在 `events.db` 同目录的 `memories.faiss`
- 如果 FAISS 不可用（没有安装 faiss-cpu），降级为纯 FTS5——不影响现有功能
- embedding 模型：优先用本地模型（如 `all-MiniLM-L6-v2`），避免每次调 API

#### 2.3.3 新增：实体知识图谱（从自研引入，轻量版）

不需要完整的 Neo4j，用 SQLite 轻量实现：

```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,         -- 实体名，如 "小明父亲"
    type TEXT NOT NULL,         -- person / project / tool / place
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE entity_links (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation TEXT NOT NULL,     -- related_to / parent_of / works_on / prefers
    memory_id TEXT,             -- 关联到哪条 memory
    weight REAL DEFAULT 1.0,
    user_id TEXT NOT NULL
);
```

**实体提取时机：**
- `add_memory` 时，异步调用 LLM 提取实体（如果本地规则能识别人名/项目名，优先本地）
- 不阻塞主流程

**阶段策略：**
- Phase 1（本周）：先不做实体图谱，只加向量检索
- Phase 2（下一周）：加实体提取和图谱
- Phase 3：BFS 检索融合

---

### 2.4 Layer 3：多路检索（融合）

#### 当前 Demo 的检索流程
```
search(query) → FTS5 → 无结果 → LIKE fallback → 返回
```

#### 融合后的检索流程
```
search(query)
  │
  ├── 1. FTS5 关键词（已有，第一阶段）
  │
  ├── 2. FAISS 向量（新增，第二阶段）
  │
  ├── 3. 实体图谱 BFS（新增，第三阶段）
  │
  └── 4. 结果融合：按 fact_type 分桶 + 去重 + 排序
      │
      ├── [Observations]     ← 归纳出的稳定模式
      ├── [Profile facts]    ← kind=profile
      ├── [World facts]      ← kind=event, memory_type=fact
      ├── [Experience]       ← kind=assistant_preference
      └── [Tasks/Plans]      ← kind=event, memory_type=task
```

**检索策略：**
1. 先用 FTS5 和 FAISS 各自召回 top_k
2. 合并后按 `confidence * 时间衰减因子` 排序
3. 如果有实体图谱，对结果中的实体做 1-2 跳 BFS 扩展
4. 最终按 fact_type 分桶返回，便于 LLM 理解上下文结构

**调用时机（不破坏 Demo 的"按需召回"原则）：**
```python
# turn_planner.py 中已有的门控保持不变
if planner.needs_profile_memory:
    profile_memories = store.search(user_id, query, kind="profile")
if planner.needs_event_memory:
    event_memories = store.search(user_id, query, kind="event")
# 新增：向量检索作为 search() 内部增强，对外透明
```

---

### 2.5 Layer 4：认知归纳（从自研引入，异步）

这是自研系统最核心的优势——**跨回合归纳稳定模式**。

#### 2.5.1 Observation 类型

| 类型 | 例子 | 触发条件 |
|---|---|---|
| Insight | "用户偏爱简洁回复" | 同一 pattern 出现 3 次以上 |
| Task | "用户正在迭代记忆系统" | 持续有相关对话 |
| Preference | "用户不喜欢 round-trip 延迟" | 用户明确表达 |

#### 2.5.2 触发时机（改进自研的固定阈值）

自研系统用 `min_sources=3` 硬编码，改为**自适应阈值**：

```python
def should_reflect(store, user_id):
    """当有新证据但尚未归纳时触发"""
    new_events = store.count_recent_events(user_id, since=last_reflect_time)
    unreflected = store.count_unreflected_events(user_id)  # 未被 observation 引用的 event
    
    # 自适应：记忆越多、阈值越高
    total = store.total_active_memories(user_id)
    threshold = max(3, int(total * 0.1))  # 至少 3 条，最多 10% 的新增
    
    return unreflected >= threshold
```

#### 2.5.3 Reflect 流程（异步，不阻塞对话）

```
后台线程（每次对话后触发检查）
  │
  ├── 1. 检查是否达到 reflect 阈值
  │
  ├── 2. 取出未归纳的新 event
  │
  ├── 3. LLM 生成/更新 Observation
  │     "基于以下最近的对话，归纳出用户的稳定偏好或正在进行的任务"
  │
  ├── 4. Observation 写为 kind=observation 的 MemoryEvent
  │     （经由 should_write_memory_candidate 门控！）
  │
  ├── 5. 用 evidence_ids 关联回原始 event
  │     ┌────────────┐     ┌──────────────────┐
  │     │ event_001  │────▶│ observation_001   │
  │     │ event_002  │     │ "用户喜欢简洁回复" │
  │     │ event_003  │     │ confidence: 0.9    │
  │     └────────────┘     └──────────────────┘
  │
  ├── 6. 时间衰减：旧 observation 降低 confidence
  │     confidence *= 0.5 ^ (days_since_update / 30)
  │
  └── 7. 更新 user_profile.md（可选）
```

#### 2.5.4 Observation 的使用

Observation 在检索时优先级最高，作为 LLM 上下文的第一段：

```
[Observations]
- 用户偏爱简洁回复 (confidence: 0.9)
- 用户正在设计 AI 眼镜记忆系统 (confidence: 0.85)

[Profile facts]
- 用户叫Jack
- 用户喜欢散步

[Relevant events]
- 昨天讨论了 memory provider 方案
- ...
```

---

## 三、实施路线图

### Phase 1：向量检索（1-2 天）⭐ 最快见效

**目标：** 提升语义召回质量，不破坏现有功能。

1. 安装 `faiss-cpu`（`conda install -c conda-forge faiss-cpu`）
2. `MemoryEvent` 新增 `embedding` 字段或独立 FAISS 索引
3. `add_memory` 时异步生成 embedding
4. `search()` 增加 `use_vector=True` 参数，内部融合 FTS5 + FAISS 结果
5. 如果 FAISS 不可用，降级为纯 FTS5——完全向后兼容

**验收标准：**
- "我上次说的那个检索方案" 能用向量召回（即使关键词不匹配）
- 现有测试全部通过

---

### Phase 2：认知归纳（3-5 天）⭐⭐ 核心能力

**目标：** 让系统能跨回合总结用户偏好和任务。

1. 新增 `memory_type="observation"` 的 MemoryEvent
2. 实现 `reflect()` 方法：取未归纳 event → LLM 生成 observation → 写库
3. 实现自适应阈值 `should_reflect()`
4. 后台线程在每次对话后检查是否触发 reflect
5. retrieval 中 observation 排在最前面

**验收标准：**
- 连续 3 次说"不喜欢太长的回复"后，系统生成 observation "用户偏爱简洁回复"
- observation 在后续对话中被检索到并作为上下文

---

### Phase 3：实体图谱（5-7 天）⭐⭐⭐ 锦上添花

**目标：** 实体级别的关联和检索。

1. 创建 `entities` / `entity_links` 表
2. `add_memory` 时异步提取实体（人名/项目名/工具名）
3. 实体去重合并
4. 检索时 BFS 扩展 1-2 跳
5. 如"小明的父亲"能关联到之前存储的所有关于"小明父亲"的记忆

**验收标准：**
- "小明父亲喜欢什么" 能跨多条记忆聚合回答
- 实体合并：同一个人不同说法（"Jack"/"Jack"）归一化

---

## 四、不做的（重要！）

| 不做的事 | 原因 |
|---|---|
| 关系图（temporal/semantic/causal）| 自研系统的异步线程管理太简单，投入产出比低。等向量检索和认知归纳稳定后再考虑 |
| FTS5 替换为纯向量 | FTS5 对中文短词和精确匹配仍然比向量好，融合才是正解 |
| 改 TimelineStore | 当前已足够好 |
| 改门控规则 | 当前 `intent_policy` 是 Demo 的核心优势，不动 |
| Reranker（cross-encoder）| 向量+FTS5 融合已足够，加上 reranker 会增加延迟和依赖 |

---

## 五、文件变更清单

```
ai_glasses_memory_assistant/
├── memory_store.py          ← 改：add_memory 加 embedding + FAISS
│                               search() 融合向量结果
│                               add_memory 加 entity extraction hook
├── memory_retrieval.py      ← 新：多路检索融合逻辑
│                               (从 memory_store.search 抽出来)
├── observation_engine.py    ← 新：reflect 逻辑
│                               should_reflect / generate_observation / decay
├── entity_store.py          ← 新：实体图谱
│                               entities + entity_links + bfs_search
├── intent_policy.py         ← 不动
├── turn_planner.py          ← 微调：retrieval 结果分桶
├── agent_bridge.py          ← 微调：chat() 后触发 reflect 检查
└── docs/context/
    └── fusion-plan.md       ← 本文档
```

---

## 六、核心原则

1. **向后兼容**：新功能失败时降级到当前行为，不破坏现有对话链路
2. **异步不阻塞**：embedding 生成、实体提取、reflect 全部异步，对话回复延迟不受影响
3. **门控不改**：所有写入（包括 observation）必须经过 `should_write_memory_candidate`
4. **审计不丢**：所有变更通过 `chat_audit.jsonl` 可追溯
5. **小而多次提交**：每个 phase 独立 PR，验证后再进下一个
