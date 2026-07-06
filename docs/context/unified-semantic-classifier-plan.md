# 统一 LLM 语义判定层方案

更新时间：2026-06-24。本文是统一 `PreReplyDecision` 的专题设计文档，用来指导“把分散在 planner/旧 router/fallback 的开放语义判断，收束成一次统一 LLM 回复前决策”的重构方向。当前已实现状态以本节为准；下方历史设计描述中如仍提到旧 router 独立概念，以当前实现状态覆盖。

## 当前实现状态（2026-06-24）

非 fast path 回复前已经从“两次内部 LLM 判断”收口为一次 `classify_pre_reply_decision()`：

```text
/api/chat
-> TimelineStore.add_turn()
-> plan_turn() 生成本地 baseline / fast path / fallback
-> classify_pre_reply_decision() 单次 LLM PreReplyDecision
   - reply_mode / answer_source / scope
   - needs_location / location_text
   - needs_web_search / web_query / web_reason
   - needs_profile_memory / needs_event_memory / needs_timeline_recall
   - memory_recall_type / recall_goal / timeline_query
   - memory_action / memory_kind / memory_type / candidate_content
   - transient / do_not_remember / correction / explanation flags
-> TurnPlan.apply_pre_reply_decision() 直接应用到 recall / web / location / reply path
-> 候选后处理 + should_write_memory_candidate() / dedupe / correction / evidence
-> recall / web / location / answer_directive / 本地回复或主 LLM
-> debug / audit / 后台 memory job
```

`debug.pre_reply_decision` 是唯一主权威字段。`debug.structured_router` 不再生成，`turn_router.py` 已删除，`StructuredRouteDecision` / `route_from_pre_reply_decision()` 不再作为兼容壳存在。

旧 `_apply_unified_semantic_recall_override()` 路径已删除。recall 不再从 `turn_semantics.recall_type` 后置二次覆盖 planner；统一由 `PreReplyDecision.memory_recall_type / needs_profile_memory / needs_event_memory / needs_timeline_recall / recall_goal` 进入 `TurnPlan.apply_pre_reply_decision()`。

## 为什么要单独做这一层

当前主链路里，和“这轮话到底在干嘛”相关的语义判断分散在几层：

- `turn_planner.py`：本地 fast path、baseline、部分时间语义清洗。
- 旧 router：曾经作为 LLM 结构化路由，已删除；相关职责并入 `PreReplyDecision`。
- `memory_store.py`：import/manual item 的 `legacy_fallback` phrase rule。
- `agent_bridge.py`：correction、task status、写入门控、召回后仲裁。

这会带来三个持续问题：

1. 新 target 一旦暴露缺口，很容易往“最近的一层”补 patch。
2. `planner / pre_reply_decision / fallback` 会各自猜一部分语义，职责越来越混。
3. 同一类开放语义会在多层重复表达，维护成本高，debug 也难解释。

本专题的目标不是“再加一层复杂中间件”，而是把开放语义收口成一个明确的单点：

> 不管用户这轮是什么 query，在 pre_reply_decision 真正决定执行路径之前，先由一次统一的 LLM 语义判定给出结构化结果；后面的 planner / pre_reply_decision / writer / recall 只消费这个结果，不再各自重新发明开放语义。

## 当前 pipeline vs 目标 pipeline

这一节只讲 `/api/chat` 主链路，不展开 import / capture / weekly report。

### 当前 pipeline

当前主链路的真实结构更接近：

```text
/api/chat
-> TimelineStore.add_turn() 记录用户原话
-> plan_turn() 只生成本地执行 baseline
   - fast path
   - 少量确定性 guard
   - temporal_scope
   - 非 fast path 默认 llm
-> classify_pre_reply_decision() 做单次 LLM pre_reply_decision
   - `debug.pre_reply_decision`
   - 一次性决定 recall 类型、reply path、web/location、correction/explanation flags、memory candidate
-> TurnPlan.apply_pre_reply_decision() 把同一份 decision 落成执行计划
-> memory candidate 按 `pre_reply_decision.memory_action / memory_kind / memory_type / candidate_content` 生成
-> correction gate 按 `pre_reply_decision.flags.correction` 决定是否进入 _detect_correction()
-> document/profile/event/timeline/web 准备
-> 本地回复或主 LLM 回复
-> _save_memory_candidates() / correction supersede / write gate
-> finalize response + debug + audit
```

当前生产主链路的开放语义主权已经收口到 `PreReplyDecision`。剩余本地逻辑只做执行编排、安全/写入门控、确定性 fast path、时间归一化和 evidence/recall 后处理。

### 目标 pipeline

统一语义判定层接入后，目标主链路应收口成：

```text
/api/chat
-> TimelineStore.add_turn() 记录用户原话
-> 本地硬边界预处理
   - text cleaning / redaction
   - temporal normalization
   - current location / recent context / document context 准备
   - 明确 do-not-remember / transient / safety 硬拦截
-> unified semantic classifier (一次 LLM 判定)
   - turn_intent
   - memory_action
   - memory_kind / memory_type
   - recall_type
   - correction / transient / explanation flags
   - candidate_content
   - reply_mode_hint
-> execution planner
   - 根据 unified result 决定本地答 / 查 recall / 调主 LLM / 走 correction / 是否写 memory
-> recall / correction target resolution / writer
-> finalize response + debug + audit
```

### 结构变化的重点

这个变化的重点不是“多一个文件”，而是把职责改清楚：

- `planner` 从“半语义层”退回“执行编排器”
- `PreReplyDecision` 是唯一回复前语义与路由决策对象
- `intent_classifier` 不再单独存在为第二份开放语义判断
- `correction detection` 变成统一语义结果里的一个 flag，而不是后置补判
- `memory_store` 不再承担主链路开放 task typing

## 字段和判断迁移表

下面这张表回答的是：当前系统里已经存在的那些判断，未来统一语义判定层接进来之后，应该迁到哪里。

| 当前来源 | 当前字段/判断 | 未来归属 | 说明 |
| --- | --- | --- | --- |
| `turn_planner.py` | `reply_mode` 中的开放语义部分 | `PreReplyDecision.reply_mode / memory_recall_type / conversation_action` | 问候、当前时间这类极确定 fast path 可以继续留本地；`reply_mode_hint` 仅保留 legacy debug 兼容，不再路由。 |
| `turn_planner.py` | `memory_write_candidates` 的开放 kind/type 推断 | `unified semantic classifier` 输出 `memory_action` + `memory_kind` + `memory_type` + `candidate_content` | planner 不再自己判断 task/preference/project_state。 |
| `turn_planner.py` | `temporal_scope` | 本地保留 | temporal normalization 仍是 planner/本地工具链职责。 |
| `turn_semantic_classifier.py` | `memory_recall_type` / `recall_goal` | `PreReplyDecision` 直接输出执行字段，`TurnPlan.apply_pre_reply_decision()` 直接消费 | recall 语义和执行决策来自同一次 LLM，不再有独立 router prompt 或兼容 adapter。 |
| `intent_classifier.py` | memory candidate extraction | 已移除，并入 `unified semantic classifier` | 当前短 turn 候选由 `turn_semantics.candidate_content / memory_kind / memory_type` 直接生成；前台和 reply-first 后台 job 共用候选后处理入口。旧 `intent_classifier.py` 不再存在，候选仍必须经过本地写入门控、去重、纠错和 evidence 路径。 |
| `agent_bridge.py::_detect_correction()` | correction 检测入口 | `unified semantic classifier.flags.correction` | 已进入保守消费阶段：LLM semantics 明确 `correction=false` 时跳过后置 correction LLM；`correction=true`、`rule_fallback`、语义失败或缺字段时仍保留现有 `_detect_correction()` / target resolution 兜底。 |
| `agent_bridge.py::_is_explanation_query()` | explanation query 检测 | `unified semantic classifier.flags.explanation_query` | 极确定本地短路也可以先保留，再逐步并入。 |
| `memory_store.py` | import/manual item 的 reminder fallback | `legacy_fallback` / 接近 `format_parser` | 不升级成主链路开放语义来源。 |
| `agent_bridge.py` | task status `已完成/取消/open` | `format_parser` 本地保留 | 这类格式性状态信号不必交给统一语义层。 |
| `intent_policy.py` | safety / sensitive / question gate | `hard_safety` 或后置 gate 本地保留 | 统一语义层可以给 hint，但最终执行仍在本地。 |

## 建议的 debug 结构变化

当前 debug 里，和语义相关的信息分散在：

- `debug.planner`
- `debug.pre_reply_decision`
- `debug.intent`
- `debug.correction_detection`

统一语义判定层接入后，主 debug 字段是 `debug.pre_reply_decision`，例如：

```json
{
  "debug": {
    "pre_reply_decision": {
      "backend": "llm",
      "turn_intent": "memory_write",
      "reply_mode": "llm",
      "memory_recall_type": "none",
      "recall_goal": "none",
      "memory_action": "write",
      "memory_kind": "event",
      "memory_type": "task",
      "recall_type": "none",
      "reply_mode_hint": "llm",
      "flags": {
        "transient": false,
        "do_not_remember": false,
        "sensitive": false,
        "correction": false,
        "explanation_query": false
      },
      "candidate_content": "周六下午把演示稿过一遍",
      "reason": "confirmed commitment with temporal scope"
    },
    "turn_semantics": {
      "source": "pre_reply_decision"
    },
    "pre_reply_decision": {
      "source": "pre_reply_decision"
    }
  }
}
```

这样后续排查任何缺口，都先问：

1. 统一语义判定层有没有判对？
2. 后面的 planner / writer / recall 有没有消费错？

而不是先去猜是 planner patch、pre_reply_decision patch、intent classifier 还是 fallback 在误伤。

## 这层要解决什么，不解决什么

这层要解决：

- 这轮是普通问答、记忆召回、待办追问、timeline 原话、解释追问，还是长期记忆写入。
- 如果涉及长期记忆，属于 `profile / event / task / preference / decision / project_state / observation` 哪种。
- 这轮是不是 transient、do-not-remember、sensitive、correction、explanation。
- 这轮是否需要后续 recall / write / supersede / temporal re-parse。

这层不解决：

- 最终怎么答用户：那是 reply synthesis / answer path 的事。
- SQLite 怎么落库：那是 writer/store 的事。
- 纠错后具体 supersede 哪条旧记忆：那是 correction target resolution 的事。
- 安全硬边界的最终执行：例如敏感信息强拦截、speaker gate，这些仍要本地硬执行。

## 建议的职责边界

### 1. 统一语义判定层

已采用一个明确的 LLM 回复前决策模块：`turn_semantic_classifier.py`。

它的职责是：对当前 turn 只做一次统一判定，输出结构化结果。

### 2. planner 的职责

`planner` 只保留这些确定性工作：

- fast path（问候、身份、当前时间等极确定本地场景）
- 本地 temporal normalization
- location/web 的低成本前置条件判断
- 对统一语义结果的执行编排

它不再承担新的开放 task / preference / project_state 语义。

### 3. pre_reply_decision 的职责

如果保留 `pre_reply_decision` 命名，它应该只负责：

- 根据统一语义结果，决定 reply path / recall path
- 不再自己追加一套开放 memory typing 逻辑

### 4. memory_store 的职责

`memory_store.py` 只保留：

- import/manual item 的 `legacy_fallback`
- reminder 这类接近格式解析的兜底
- 可观测 debug/classification 决策记录

它不再承担主链路开放 task typing。

## 建议输出结构

统一语义判定建议至少输出下面这些字段。字段名可以调整，但职责要稳定。

```json
{
  "turn_intent": "chat|memory_write|memory_recall|timeline_recall|explanation|correction|mixed",
  "reply_mode_hint": "llm",
  "memory_action": "none|write|recall|correction|explain",
  "memory_kind": "profile|event|assistant_preference|none",
  "memory_type": "fact|event|task|preference|decision|project_state|observation|none",
  "recall_type": "none|profile|event|observation|timeline|attention_items|weekly_summary",
  "temporal_intent": {
    "needs_temporal_resolution": true,
    "temporal_role": "event_time|recall_window|correction_fragment|none"
  },
  "flags": {
    "transient": false,
    "do_not_remember": false,
    "sensitive": false,
    "correction": false,
    "explanation_query": false
  },
  "candidate_content": "normalized semantic content if applicable",
  "reason": "short structured explanation"
}
```

`reply_mode_hint` 目前只作为旧 debug 兼容字段保留，不能再承担路由职责。真实回复路径、召回、web/location 和 conversation action 都以 `PreReplyDecision` 的执行字段为准。

重点不是字段一定长这样，而是：

- 一次判定把开放语义讲清楚
- 后面层只消费，不再重新猜
- debug/audit 能明确还原“LLM 当时把这轮判成了什么”

## 哪些本地规则必须保留

即使走统一 LLM 语义判定，也不应该把所有本地规则删掉。必须保留的，是那些“硬边界”而不是“开放语义补丁”。

### 保留为 `hard_safety`

- 敏感信息拦截
- speaker 不是用户时的长期记忆阻断
- 明确安全红线（token、密码、证件等）

### 保留为 `format_parser`

- task status：`已完成 / 取消 / open`
- reminder 类格式兜底
- 明确 temporal normalization

## 当前输入输出边界 vs 目标输入输出边界

这一节用大白话说明“系统结构到底会怎么变”。

### 当前

现在更像是多个人分别看同一句话，各自做一点判断：

- `planner` 先猜“像不像要写记忆、像不像 task、要不要本地答”
- `pre_reply_decision` 再猜“像不像 recall、该走什么答复路径”
- 旧 extractor 再单独抽一版 memory candidate
- `correction detection` 再补一次“是不是纠正前文”
- `memory_store` 的少量 fallback 还会兜一点 phrase-based typing

问题不是这些模块都没价值，而是：

- 同一句话被多次重复判语义
- 一旦 target 失败，很容易往最近模块补规则
- debug 时很难一句话说明“到底是哪一层把这句判成 task/recall/correction 的”

### 目标

目标不是把所有能力塞进一个超大分类器，而是把“开放语义判断”集中一次做完：

```text
原始 query
-> 本地预处理
-> 统一语义判定
-> 执行规划
-> 回复 / recall / 写入 / correction 落地
```

这样后面模块就不再各自重复猜语义，只做自己的执行工作：

- `planner` 负责“接下来走哪条执行路径”
- `writer` 负责“要不要写、怎么写”
- `recall` 负责“要查什么、查完怎么仲裁”
- `correction resolver` 负责“纠正命中哪条旧记忆”

可以把它理解成：

- 现在：多个模块分别“边理解边执行”
- 目标：先统一“理解”，后面模块只“执行”

## 统一语义层接入后，现有文件大致会变成什么角色

这部分不是承诺马上改成这样，而是给后续实现时一个稳定边界。

| 文件 | 现在主要角色 | 目标角色 |
| --- | --- | --- |
| `agent_bridge.py` | 主链路编排，同时夹带 correction / save gate / fallback 协调 | 继续做主链路编排，但尽量消费统一语义结果，而不是自己再补开放语义判断 |
| `turn_planner.py` | 本地 baseline + 部分开放语义猜测 | 收缩成执行 planner，本地只保留 hard boundary、fast path、temporal normalization |
| 旧 router | 路由 + 部分开放语义 | 已删除；职责并入 `turn_semantic_classifier.py::classify_pre_reply_decision()` |
| `intent_classifier.py` | 第二份 memory candidate 语义抽取 | 逐步并入统一语义层，最后不再独立承担开放语义判定 |
| `memory_store.py` | 存储 + import/manual fallback + 少量 phrase typing | 保留存储和 `legacy_fallback`，不再承担主链路开放语义来源 |
| `turn_semantic_classifier.py` | `PreReplyDecision` | 成为开放语义和回复前执行决策的单点入口 |

## 最值得先迁走的判断

如果后面真做这轮重构，不建议“一口气全迁”。性价比最高的是先迁那些最容易引起职责混乱、又最常暴露缺口的判断。

优先级建议如下：

1. `memory_action` 和 `memory_type`
   例如“这是普通闲聊，还是一条应写入的 task / preference / decision”。
2. `correction` flag
   例如“不是新任务，而是在纠正上一轮那条记忆内容”。
3. `recall_type`
   例如“用户是在问 profile、timeline 还是待办状态”。
4. `reply_mode_hint`
   例如“该本地短答、该 recall 后答、还是该直接走主 LLM”。

原因很简单：

- 这几项最容易在多个模块里重复判断
- 也是最容易因为单个 target 失败而去补 patch 的地方
- 一旦统一掉，后面很多 debug 都会明显简单

## 哪些东西不值得急着迁

不是所有逻辑都值得立刻塞进统一语义层。下面这些保留本地，通常更稳：

- 明确的 `fast path`
  例如问候、当前时间、简单寒暄。
- 明确的 `hard_safety`
  例如 API key、密码、证件号、非本人说话。
- 明确的 `format_parser`
  例如 `已完成`、`取消`、`open` 这类 task status。
- 明确的 `temporal normalization`
  例如“明天下午”“下周一”转标准时间窗口。

换句话说，统一语义层主要接管“开放解释空间很大”的部分，不是把所有 if/else 都扔给 LLM。

## 后续完整阶段路线图

这条路线的目标不是“让 unified semantics 一步接管所有记忆逻辑”，而是让旧层逐步退场。专业上叫 staged migration；大白话就是先让新判定层当“总理解员”，再让老模块少猜一点，最后只保留执行和硬边界。

### 阶段 0：契约和观测，已完成

目标：先把统一语义层跑起来，但不让它决定业务结果。

已完成：

- `debug.turn_semantics` 已进入主链路。
- explanation flag、recall_type override、correction gate、memory extraction gate 已开始保守消费。
- candidate shadow 已进入 chat audit / background memory job trace，可统计 exact / similar / different / semantic-only / extractor-only / fallback。
- deterministic shadow 小考场已用于固定样例对账。

验收口径：

- debug/audit 能回答“统一语义层当时怎么判”。
- 统一语义层出错、`rule_fallback`、缺字段时，旧链路仍兜底。

### 阶段 1：候选类型标签保守消费，当前阶段

目标：让 unified semantics 主导短 turn 候选内容和类型，旧独立 extractor 退场。

当前状态：

- `intent_classifier.py` 已删除；`MemoryWriteCandidate` / `IntentDecision` 数据结构迁到 `memory_candidate.py`。
- `turn_semantics.backend=llm` 且输出 `memory_action=write`、`candidate_content`、`memory_kind`、`memory_type` 时，系统直接生成 `MemoryWriteCandidate`。
- 前台聊天、reply-first 后台 memory job、长输入分段抽取都不再调用旧 `classify_turn_intent()`。
- `debug.routing.unified_semantic_candidate_authority` 和后台 `extraction_trace.unified_semantic_candidate_authority` 记录候选创建/覆盖/跳过/fallback 原因。
- 所有候选仍必须经过 `should_write_memory_candidate()`、去重、纠错/supersede 和 evidence 绑定。

验收口径：

- 前台/后台同一类语义输出生成一致候选。
- semantic-only 写入语义可以生成长期记忆候选。
- sensitive / transient / do-not-remember 内容仍被本地安全门或语义 flag 拒绝保存。

### 阶段 2：清理旧 extractor 残留

目标：确认旧 extractor 代码、测试和文档引用全部退出主路径。

推进顺序：

- 保留 `memory_candidate.py` 作为数据结构归属，不再恢复 `intent_classifier.py`。
- 将旧 `unified_semantic_typing_hint` 兼容字段逐步改名或移除，统一到 `unified_semantic_candidate_authority`。
- 继续压缩 `turn_planner.py` 中开放 kind/type 推断，只保留低风险本地 fast path 和 fallback。
- 文档、HTML explainer、eval fixtures 中提到 `intent_classifier.py` 的地方按当前实现更新。

验收口径：

- `rg "intent_classifier|classify_turn_intent"` 只剩历史说明或已明确标注为 removed 的文档引用。
- audit 能解释每条候选是 unified semantics 创建、本地 planner fallback，还是被语义 flag / write gate 拒绝。

### 阶段 3：候选内容迁移前的真实 audit 回放

目标：在考虑让 unified semantics 生成候选内容前，先用真实 audit 证明它稳定。

必须先做：

- 用 `summarize_unified_semantic_candidate_shadow()` 跑真实 chat audit，而不是只看 deterministic 小考场。
- 如果 semantic-only 多，先修 unified semantic prompt/schema。
- 如果 extractor-only 多，先修 unified semantics 漏判。
- 如果 kind/type mismatch 多，先修 typing 规则或 prompt。

暂不做：

- 不直接把 `candidate_content` 保存成长期记忆。
- 不因为一两个 case 对齐就删除 extractor。
- 不用短语 hard code 补 unified semantics 漏洞。

验收口径：

- 有真实 audit 统计说明 exact / contains_or_similar 占优，fallback 和 mismatch 可解释。
- 分歧 case 能按层定位：semantic 错、extractor 错、write gate 错、recall 错、correction resolver 错。

### 阶段 4：收缩 pre_reply_decision / planner 的开放语义判断

目标：让 `turn_planner.py` 和旧 router 路径不再继续吸收新的开放语义 patch。

迁移顺序：

- `turn_planner.py` 保留 fast path、temporal normalization、hard boundary、执行 baseline。
- 旧 router 已删除，不再承担 reply path / recall path。
- `reply_mode_hint` 最后迁，因为它直接影响用户可见回复，风险高于 memory gate。

验收口径：

- 新 target 失败时，默认先看 `debug.turn_semantics`，而不是先往 planner/pre_reply_decision 补词表。
- planner/pre_reply_decision 的新增逻辑主要是执行条件，不是开放语义分类。

### 阶段 5：候选生成主权迁移，最后阶段

目标：只有当 audit 证明 unified semantics 稳定后，才考虑让它创建或主导候选内容。

允许条件：

- 真实 audit 与 target eval 显示 `candidate_content` 与 extractor content 长期对齐。
- semantic error / fallback 回退链路稳定。
- write gate、dedupe、supersede、sensitive gate 仍在本地硬执行。

迁移方式：

- 先在 feature flag 或内部开关下试运行。
- 先只处理单候选、内容明确、非敏感、非 correction fragment 的场景。
- 保留 extractor fallback，直到足够多 replay 证明不退化。

验收口径：

- unified semantics 可以主导候选内容，但不能绕过本地写入门控。
- 删除或降权旧 extractor 之前，必须有 audit replay / unittest / target eval 三类证据支撑。

### 阶段 6：清理冗余本地语义规则

目标：旧层真正退场，而不是新旧两套语义长期并存。

清理对象：

- 已被 unified semantics 覆盖的 planner open-semantic patch。
- 不再需要的 pre_reply_decision recall/write 开放语义重复判断。
- 主链路里为了单个 query 加的 phrase rule。

必须保留：

- `hard_safety`：敏感信息、speaker gate、安全红线。
- `format_parser`：task status、reminder 格式兜底、temporal normalization。
- `legacy_fallback`：import/manual item 没有结构化结果时的最小兜底。

验收口径：

- 代码路径更少，但行为不退化。
- debug 更清楚：先看统一语义层是否判对，再看执行层是否消费对。

## 用 Codex 做这类重构，什么用法性价比最高

这类任务最怕的不是改不动，而是：

- 还没把边界讲清楚就开始搬代码
- 每次只为一个 target 修局部 bug
- 最后多出一个新层，但旧层逻辑没真正退场

对 Codex 来说，最高性价比的用法不是“直接让它大改一轮”，而是：

1. 先把专题文档定清楚
   让 Codex 先写清当前 pipeline、目标 pipeline、字段迁移表、暂不迁内容。
2. 再让它做 debug-only 接线
   只新增统一语义结果的观测，不接业务主权。
3. 再按字段一刀一刀迁
   每次只迁一个判断族，比如 `correction` 或 `memory_type`。
4. 每刀都要求 target / unittest / audit 证据
   避免“回答看起来更聪明了”就算完成。

最不划算的用法是：

- 直接要求“把整个 planner/pre_reply_decision/memory typing 统一重构”
- 或者“这个 target 失败了，随便补到最近模块里让它过”

前者容易失控，后者会继续把系统补胖。

## 对 PLANS.md 应该怎么表述

`PLANS.md` 里不应该把这件事写成“下一步立刻全面重构 pipeline”。更合适的表述是：

- 这是一条中期架构方向
- 当前主要作用是约束后续不要继续给 planner / legacy fallback 堆开放语义 patch
- 真正实施时按本文“阶段 0 -> 阶段 6”的路线推进，每次只记录当前完成的阶段切片

这样 `PLANS.md` 既能表达方向，又不会把当前 repo 写成已经进入大重构期。

### 不应该继续扩的

- 为单个 query / 单个 persona / 单个 replay 继续补 task 动词词表
- 把“确认型 task”长期塞在 planner 本地规则里
- 把开放 recall/write 语义散落到多层重复判断

## 对 eval 和 target 的影响

引入统一语义判定层后，target 的作用要改一下：

- target 继续用于暴露真实语境缺口
- 但不再默认导向“往 planner/fallback 补规则”
- 新 target 应优先回答两个问题：
  - 统一语义判定层是否把这轮判对了
  - 后续 writer/recall/supersede 是否消费对了这个结果

建议新增一类 debug 断言，直接看统一语义结果，而不是只看最终 reply/saved memory。

当前已补一层 deterministic shadow 小考场：用固定样例模拟 `turn_semantics` 与旧 extractor 输出，对齐统计只用于判断下一刀该修 semantic、extractor、typing，还是可考虑 typing hint；它不接 live LLM，也不驱动写库。

## 当前已确认的设计原则

- 不接受继续往 `memory_store.py` 补 task 动作词 marker。
- 不接受继续把开放 task 语义长期塞进 `turn_planner.py`。
- reminder phrase fallback 可以保留，但只作为 `legacy_fallback` / 接近 `format_parser` 的 import/manual 兜底，不是主链路语义来源。
- 目标不是“让 pre_reply_decision 也更胖”，而是“让统一语义判定成为唯一开放语义入口”，然后缩小其他层职责。

## 与现有文档的关系

- 当前调用链参考：`pipeline.md`
- 当前记忆机制边界参考：`memory-mechanism.md`
- 当前短语规则治理参考：`phrase-rule-inventory.md`
- 当前 eval 缺口和 target 语义族参考：`eval-coverage.md`

如果后续真的开始实现统一语义判定层，应优先更新本文，再同步 `PLANS.md` 中的下一刀顺序；不要只在 `PLANS.md` 里写结论。
