# 记忆系统设计审计：已完成 & 待优化

> 审计时间：2026-07-30
> 状态：等待 iPhone App 开发完成后启动优化
> 注意：本文不是执行计划，而是差距清单和优先级参考。待 iOS 阶段完成后逐条评估启动。

---

## 一、已完成的正面能力

当前系统在**显式交互记忆**（用户主动对话 → 结构化存储）上已经扎实：

| 能力 | 实现位置 | 说明 |
|---|---|---|
| 四层记忆核 | `memory_kernel.py` | raw_timeline → structured_memory → reflection → runtime_recall，符合 CoALA 框架 |
| 结构化存储 | `memory_store.py` | SQLite 6 表 + FTS5 全文搜索，含 memories / subjects / aliases / voice_profiles / documents |
| 双层记忆分类 | `turn_semantic_classifier.py` | kind（profile/event/assistant_preference）+ memory_type（fact/event/task/preference/decision/project_state/observation） |
| 记忆门控（13 条拒绝规则） | `intent_policy.py` | 空内容、非本人、重叠不明、敏感信息、置信度 < 0.6、问题文本、transient marker 等 |
| 敏感信息检测 | `intent_policy.py:_sensitive_reason()` | 关键词黑名单 + 正则（API key、JWT、GitHub token、验证码、银行卡、身份证号等），含护照补办类豁免 |
| 去重与证据合并 | `memory_store.py:find_similar_memory()` + `merge_memory_evidence()` | 内容标准化比对 + evidence_ids 链 + 置信度取最大值 |
| 强度衰减系统 | `memory_store.py` | 按 memory_type 差异化 Ebbinghaus 式时间衰减（preference 180天、task 30天等） |
| 记忆生命周期 | `memory_lifecycle.py` | active → stale → superseded → deleted，含状态迁移校验 |
| 三层审计追溯 | `source_trace` / `recall_trace` / `chat_audit.jsonl` | 每次写入和召回都有完整可追溯的 evidence 链 |
| 人物主体管理 | `memory_subject_identity.py` | self / named / provisional 三种主体类型，含声纹向量和别名映射 |
| 本地 fast path | `turn_planner.py:plan_turn()` | 问候/身份/凭证 五个 fast path 场景无需调 LLM |
| 时间解析 | `temporal_parser.py` + `resolve_temporal_local()` | 本地解析"今天/明天/下周"等高频短语，避免每个时间表达都调用 LLM |
| 召回仲裁 | `memory_recall_arbitration.py` | 多源召回后的确定性优先级仲裁，空证据保护 |
| 讨论归档 | `discussion_archive.py` | 全天对话的话题追踪和摘要 |

---

## 二、需要优化的差距（按优先级排序）

### 差距 1：缺少"被动记忆形成"管道 🔴 高优先级

**现状**：`intent_policy.py` 直接拒绝 `ambient_audio` 来源。环境音频只进 timeline 留证据，不形成结构化记忆。

**问题**：对于"持续收音 + 唤醒提问"的 AI 眼镜，这意味着记忆系统的最大信息源被关掉了。大部分生活记忆应该是无意识形成的，不需要每次记忆都伴随一次主动对话。

**理想状态**：
```
全天音频流 → VAD → ASR → 声纹 → 门控
  → 本人说话 → 背景记忆提取 pipeline → 结构化记忆候选 → 门控 → 持久化
  → 他人说话 → 丢弃（或作为事件上下文保留）
唤醒词 → 用户提问 → chat turn → 回忆 + 回复 → 可能写记忆
```

**参考**：李未可 "Memory Policy Engine" 明确区分被动采集和主动交互两条路径。

---

### 差距 2：缺少"惊喜度/重要性"信号 🔴 高优先级

**现状**：系统只有通用的 `confidence` 阈值，没有衡量信息的新颖度、重要性或与已有记忆的冲突度。

**问题**：Google Titans 论文的核心发现——信息与已有记忆的偏差（surprise metric）是判断是否值得记住的最强信号。当前系统无法区分"重复信息"和"全新发现"。

**需要的能力**：
- 新候选 vs 已有记忆的语义相似度 → 低相似度 = 高新奇度 = 高写入价值
- 信息与已有记忆矛盾 → 标记冲突，可能触发更新或澄清
- 信息包含时间/地点/人物/决策标记 → 更高的重要性分

**参考**：Google Research "Titans + MIRAS" momentum-based surprise metric。

---

### 差距 3：记忆召回太保守 🟡 中优先级

**现状**：`memory_kernel.py` 第 66 行写死："ordinary chat and factual QA do not read long-term memory"。

**问题**：这对通用 chatbot 是正确的，但对记忆眼镜是反直觉的。用户在普通聊天时（"今晚吃什么"），系统应该自动想起相关记忆（"你上周说冰箱有剩咖喱"），而不是只在不明确的"帮我回忆"时才查记忆。

**需要的能力**：
- 普通对话中轻量语义匹配 → 注入低权重记忆上下文
- 区分"背景注入"（不影响回复结构）和"显式召回"（主导回复方向）
- 用户可感知的注入标记（"根据你上周的记录..."）

**参考**：Limitless Pendant 的对话中自动上下文注入。

---

### 差距 4：缺少"每日/每周记忆整合"管道 🟡 中优先级

**现状**：`reflection` 层使用简单计数触发（≥3 条新事件做 observation），但这不等同于真正的记忆整合。

**问题**：人类记忆的关键环节是睡眠期的海马体→皮层整合。系统需要离线处理来提取模式、更新偏好、检测冲突、清理冗余。

**需要的能力**：
- **每日整合**：当天 timeline → 日摘要 → 长期事实提取
- **每周整合**：本周事件 → 模式识别 → 习惯/偏好更新
- **冲突检测**：新记忆与旧记忆矛盾时，标记并澄清
- **冗余清理**：多条记忆高度重复时，合并为一条更准确的

---

### 差距 5：记忆索引缺少时空维度 🟡 中优先级

**现状**：位置信息只在当前 turn 使用（`needs_location`），不索引进记忆。时间维度已有时但缺少"今天/本周"等自然语言过滤。

**问题**：用户会问"我当时在哪儿说的？"、"附近发生过什么？"、空间和时间组合查询是生活记忆的核心需求。

**需要的能力**：
- 每条记忆关联位置标记（经纬度 + 语义地名）
- "附近发生过什么"的空间查询
- 时间+空间组合过滤

---

### 差距 6：Agent 记忆 vs 人类记忆增强的定位模糊 🟡 中优先级

**现状**：当前设计更接近 **Agent 记忆**（帮 AI 记住对话上下文），而不是 **Human Memory Augmentation**（帮人记住自己的生活）。

**关键差异**：

| 维度 | Agent 记忆（当前倾向） | 人类记忆辅助（目标） |
|---|---|---|
| 记忆主体 | AI 自身 | 用户这个人 |
| 召回触发 | LLM 自主决定 | 用户提问 / 场景触发 |
| 记忆内容 | 对话历史 / 偏好 | 生活事件 / 人际关系 / 决策 |
| 遗忘策略 | Token 优化 | 模拟人类遗忘曲线 |
| 产品形态 | API / SDK | 可穿戴硬件 |

**需要的能力**：
- 召回策略从"帮 AI 更好回复"转向"帮用户回忆生活"
- 用户对记忆的可见性、可控性增强
- 用户能查询"我跟张总周二说了什么？"而不是依赖 AI 自主判断

---

### 差距 7：缺少可解释的"为什么不记"层 🟢 低优先级

**现状**：audit 已有 `unit_gate_results`，但缺少面向用户的解释能力。

**问题**：当用户问"你刚才记了什么？"或"为什么没记住？"，系统应该能给出清晰回答。这对建立信任至关重要。

**需要的能力**：
- 每次写入/拒绝给出用户可理解的简短原因
- 用户可以追问"为什么没记刚刚那句话？"并得到具体门控原因

---

### 差距 8：Continuous Capture 定位需要重新审视 🟢 低优先级

**现状**：`continuous_capture` 被归为 `turn_planner` 的一个 fast path（长文本 ≥90 字），触达条件过于苛刻。

**问题**：对于全天眼镜，continuous capture 应该是**主要处理模式**，而不是一个"特殊情况"。大部分生活场景的音频不会满足"≥90字且多维度话题特征"的条件。

---

## 三、外部参考资源速查

| 资源 | 类型 | 最相关的是什么 |
|---|---|---|
| **李未可 X-AI 记忆眼镜** | 商业产品 | 跟你产品定位一致，Memory Policy Engine 四大能力，499元级硬件 |
| **MemGPT / Letta** | 开源项目 (24K ⭐) | 分层记忆 OS 模型，Agent 自主管理记忆 |
| **Google Titans + MIRAS** | 学术论文 | Surprise Metric 判断记忆重要性 |
| **CoALA (Princeton)** | 学术论文 | Agent 记忆分层理论基础 |
| **Limitless Pendant** | 商业产品 | 全天对话记录 + 对话式回忆 |
| **Mem0** | 开源项目 (60K ⭐) | 93% token 压缩率，即插即用记忆 API |
| **Zep / Graphiti** | 开源项目 | 时序知识图谱，天然处理时序状态变更 |

---

## 四、启动条件与执行顺序建议

### 启动条件
- [ ] iPhone App 核心功能开发完成并验收通过
- [ ] 现有 Python 核心测试全量通过

### 建议执行顺序

**第一批（理解调整，不改架构）**
1. 放松召回保守度：允许普通聊天中注入低权重记忆上下文
2. 开启 ambient→memory 通道：ambient_audio 经过说话人+隐私门控后可写记忆
3. 增加重要性评分字段和简单规则打分

**第二批（新增管道和索引）**
4. 实现每日/每周记忆整合管道
5. 增加位置索引到记忆
6. 加入惊喜度信号（候选 vs 已有记忆语义相似度对比）

**第三批（架构调整）**
7. 重构 ambient processing pipeline，让它成为一等公民
8. 双层召回策略：显式查询 vs 隐式注入
9. 面向用户的记忆解释层

---

> 本文在 iPhone App 开发完成前不进入执行阶段。每条差距在启动时需重新评估技术可行性和优先级。
