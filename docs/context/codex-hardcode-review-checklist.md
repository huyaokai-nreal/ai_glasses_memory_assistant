# Codex Hard Code 审查清单

更新时间：2026-06-29。本文不是新增一套产品规则，而是给后续 Codex 改代码时用的轻量审查清单。目标不是逐行盯代码，而是用少量高风险信号，尽早发现“为了过一个 case 临时补丁”的实现。

## 这份清单解决什么问题

当前 repo 已经明确要求不要散落 hard code，也不要继续给 planner / fallback / phrase rule 堆开放语义 patch。但只靠一句“不要 hard code”还不够，因为模型在局部压力大、target 很具体、代码理解不完整时，还是可能滑向：

- 为某个 query 单独补一条短语规则。
- 在最近的一层偷偷加一个 fallback 分支。
- 用看似通用、其实只服务当前例子的 marker / hint / regex。
- 让测试过了，但系统机制没有真正变清楚。

这份清单的作用，就是把“我不想逐行审”变成“我只查高风险点和自证项”。

## 一句话原则

如果一个改动主要是在回答“怎么让这句话过”，而不是回答“这类语义应该在哪一层处理”，它大概率就是 patch，不是机制修复。

## 哪些地方最容易长 hard code

优先看这些位置，它们是高风险区：

| 高风险区 | 为什么危险 | 常见伪装 |
| --- | --- | --- |
| `turn_planner.py` | 离执行很近，最容易为了当前 case 多加一条判断。 | 新增 `if message contains ...`、reply mode 特判、局部 recall 分支。 |
| `agent_bridge.py` | 主链路很长，容易混入“顺手补一个例外”。 | correction / explanation / recall / save gate 里加 case patch。 |
| `intent_policy.py` | 写入门控很敏感，容易把开放语义误塞成安全或问题规则。 | 某个具体表达被当成“禁止写入”硬拦。 |
| `memory_store.py` | fallback 分类、导入兜底天然容易长词表。 | `喜欢/项目/提醒/...` 一类 marker 越堆越多。 |
| `phrase-rule-inventory.md` 对应代码位 | 这里本来就是规则集中地，最容易在“已有规则”掩护下继续加 patch。 | 看起来像“小补充”，实际是在扩开放语义责任。 |

大白话：不用每一行都查，但这几个“最容易藏补丁的口袋”要重点看。

## 一眼识别 hard code 的红旗

只要改动里出现下面任意信号，就应该提高警惕：

1. 新增 `if` / `elif` 直接匹配某个短语、问法、语气词、固定 query。
2. 新增 marker、关键词、正则，但解释不清它属于 `hard_safety`、`format_parser`、`weak_signal`、`retrieval_hint` 还是 `legacy_fallback`。
3. 同一个问题在多层都补了逻辑，例如 planner 补一点、bridge 再补一点、fallback 再补一点。
4. 测试只覆盖一条具体句子，没有覆盖这一类行为的完整链路。
5. debug 字段看起来更丰富了，但主控制流没有更收口。
6. 理由是“为了让这个场景通过”，而不是“这类语义本来就应该由这一层负责”。

## 每次改动前必须自证的 5 件事

让 Codex 在动手前先回答这 5 个问题：

1. 这次问题属于哪一层？
   - `planner baseline`
   - `pre_reply_decision`
   - `write gate`
   - `dedupe / correction / lifecycle`
   - `recall arbitration`
   - `explanation / audit`

2. 为什么必须放在这一层，而不是别层？

3. 这次是否新增了 phrase rule / marker / regex / 特判分支？

4. 如果新增了，它属于哪类角色？
   - `hard_safety`
   - `format_parser`
   - `weak_signal`
   - `retrieval_hint`
   - `legacy_fallback`
   - `reply_style_only`

5. 如果不加这条规则，能不能用现有统一语义层、共享后处理、现有 gate 或现有 debug 解释完成？

如果这 5 个问题答不清，先不要改代码。

## 每次改动后最少要查什么

不逐行全审时，至少查下面 4 项：

1. 有没有新增短语、词表、marker、regex、局部 `if message contains ...`。
2. 改动是否落在原本负责这类问题的层，而不是最近的一层。
3. 测试测的是“机制链路”还是“单句过关”。
4. debug / audit 是否能解释这次改动带来的行为变化。

## 推荐测试口径

优先要求“行为链路测试”，不要只要“句子测试”。

坏测试例子：

```text
输入“我现在喜欢吧台位置”时保存成功。
```

更好的测试例子：

```text
旧偏好 active
-> 用户改口
-> 旧偏好 superseded
-> 新偏好 active
-> 后续召回答案使用新偏好
-> “为什么这么答”能解释到新来源
```

大白话：不要只测“这句话过没过”，要测“系统有没有真的学会这类事”。

## 什么时候可以接受少量规则

不是所有规则都不允许。以下类型通常可以接受，但必须可解释：

- `hard_safety`：token、密码、证件号、验证码等封闭安全边界。
- `format_parser`：日期、时间、结构化命令、显式状态词等确定性格式。
- `weak_signal`：只做提示，不直接决定最终语义。
- `retrieval_hint`：只帮助缩小召回范围，不直接替代主语义判断。
- `reply_style_only`：只影响文案，不改召回、写入、路由。

以下类型默认高风险：

- 为某个自然说法单独补开放语义规则。
- 用局部 marker 决定长期记忆写入/纠错/召回主路径。
- 在 fallback 层悄悄接管本应由统一语义层负责的事。

## 你可以怎样低成本抽查

如果工作量很大，不需要每次逐行审。可以只做这 3 步：

1. 先看 diff 里有没有新增：
   - 字符串词表
   - regex
   - `special_case`
   - `fallback`
   - `marker`
   - `hint`

2. 再看它落在不落在高风险文件：
   - `turn_planner.py`
   - `agent_bridge.py`
   - `intent_policy.py`
   - `memory_store.py`

3. 最后只问一句：
   - “这是在修机制，还是在修这句话？”

如果更像后者，就先停下来 review。

## 和本仓库现有文档的关系

- 想看规则角色边界：读 `phrase-rule-inventory.md`。
- 想看开放语义应该往哪里收口：读 `unified-semantic-classifier-plan.md`。
- 想看这次到底该做什么：看 `PLANS.md`。
- 本文只负责一件事：帮助你快速识别 Codex 改动是不是在偷偷长 patch。
