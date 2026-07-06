# 迁出前离线 Eval Baseline 方案

本文是独立化迁移第十五刀产物：为正式迁出 `ai_glasses_memory_assistant` 前固化 baseline 执行方案和报告模板。本刀只固化方案，不代表 baseline 已经执行，也不代表已经完成迁出。

## 目标

正式迁出前先在当前外层 `hermes-agent` checkout 内跑一轮 baseline。迁出后用同一批场景、同一类配置和同一套指标复跑，确认迁出没有让这些能力退化：

- 记忆写入。
- 记忆召回。
- 解释依据。
- timeline / document / observation 证据。
- web / location / weather debug 字段。
- 多用户隔离和隐私门控。
- 后台 memory job 状态。

大白话：先在老位置拍一张“效果照片”，搬到新仓库后再拍同角度照片。两张照片对得上，才能说迁出没有把 demo 效果搬坏。

## 当前 Runner 边界

当前 eval runner 是：

```bash
python -m ai_glasses_memory_assistant.evals.runner
```

已确认的参数边界：

- `--mode live`：当前唯一模式，走真实 `GlassesChatService` 和当前配置的 LLM backend。
- `--scenario-id`：按场景 id 过滤，可重复。
- `--category`：按 category 过滤，可重复。
- `--repeat`：重复次数。
- `--report-dir`：报告输出目录。
- `--background-wait`：等待后台 memory job 的时间。
- `--strict`：active 场景失败时返回非零；`status=target` 缺口不阻断。

因此本文说的“离线 baseline”含义是：不启动 Web UI、不跑浏览器、不依赖 live 网络搜索、不依赖真实定位设备；但 runner 当前仍可能调用真实 LLM backend。正式执行 baseline 时必须记录 LLM backend 配置摘要。

## Baseline 场景批次

迁出前 baseline 建议分三批跑，避免一次全量失败时不知道是哪类能力退化。

### 第一批：核心 active 门禁

这批优先确认当前已经稳定的基础闭环：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner \
  --mode live \
  --category identity \
  --category temporal_recall \
  --category schedule \
  --category memory_write \
  --category negative \
  --category location \
  --category isolation \
  --category memory_mechanism \
  --repeat 1 \
  --background-wait 15 \
  --strict \
  --report-dir ai_glasses_memory_assistant/reports/standalone-migration-baseline/pre-extraction-active
```

### 第二批：效果回放与主线压力

这批看解释、真实语境、多来源、长输入和文档相关链路：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner \
  --mode live \
  --category audit_replay \
  --category real_context_pressure \
  --category document_text_scope \
  --category observation \
  --category llm_first_answer_quality \
  --category text_cleaning_semantics \
  --category source_control \
  --repeat 1 \
  --background-wait 15 \
  --strict \
  --report-dir ai_glasses_memory_assistant/reports/standalone-migration-baseline/pre-extraction-mainline
```

### 第三批：target 缺口快照

这批记录还没作为阻断门禁的 target 缺口。`target_failed_turns` 允许存在，但迁出后必须同口径对比：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner \
  --mode live \
  --category public_chitchat_negative \
  --category public_preference_write \
  --category public_schedule_event \
  --category public_temporal_recall \
  --category public_long_context \
  --category glasses_fragmented_input \
  --category glasses_noisy_long_speech \
  --category glasses_asr_privacy \
  --category ambient_memory_gate \
  --repeat 1 \
  --background-wait 15 \
  --report-dir ai_glasses_memory_assistant/reports/standalone-migration-baseline/pre-extraction-targets
```

如果正式执行时资源允许，可以再补一轮全量：

```bash
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner \
  --mode live \
  --repeat 1 \
  --background-wait 15 \
  --strict \
  --report-dir ai_glasses_memory_assistant/reports/standalone-migration-baseline/pre-extraction-all
```

## 报告必须记录的字段

runner 会输出：

```text
eval-latest.md
eval-latest.json
```

baseline 报告至少要看这些字段：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `summary.pass_rate` | `eval-latest.json` | 总通过率。 |
| `summary.active_pass_rate` | `eval-latest.json` | active 门禁通过率。 |
| `summary.active_failed_turns` | `eval-latest.json` | active 失败轮次，`--strict` 主要看它。 |
| `summary.target_failed_turns` | `eval-latest.json` | target 缺口轮次，迁出后必须同口径对比。 |
| `summary.reply_fact_hit_rate` | `eval-latest.json` | 用户可见回复命中率。 |
| `summary.memory_recall_hit_rate` | `eval-latest.json` | 召回证据命中率。 |
| `summary.memory_write_hit_rate` | `eval-latest.json` | 写入证据命中率。 |
| `summary.exception_rate` | `eval-latest.json` | 异常率。 |
| `summary.completed_rate` | `eval-latest.json` | 完成率。 |
| `summary.api_calls_total` / `summary.api_calls_mean` | `eval-latest.json` | LLM/API 调用成本粗略对比。 |
| `summary.latency_seconds` / `summary.stage_latency_seconds` | `eval-latest.json` | 迁出后性能是否异常变慢。 |
| `runs[].turns[].failures` | `eval-latest.json` | failed scenario ids 和失败原因。 |
| `runs[].turns[].response.debug` | `eval-latest.json` | routing、memory、web、location、weather、timing、memory_jobs 等 debug 证据。 |
| `runs[].turns[].new_memories` | `eval-latest.json` | 本轮写入结果。 |
| `runs[].turns[].all_memories` | `eval-latest.json` | 最终 active memory 状态。 |
| `runs[].turns[].response.recalled_memories` | `eval-latest.json` | 结构化记忆召回。 |
| `runs[].turns[].response.recalled_timeline_chunks` | `eval-latest.json` | timeline 原文召回。 |
| `runs[].turns[].response.recalled_documents` | `eval-latest.json` | 文档召回。 |

特别要人工抽查这些 debug 路径：

- `debug.routing`
- `debug.memory`
- `debug.memory_processing`
- `debug.memory_jobs`
- `debug.tools`
- `debug.location`
- `debug.weather`
- `debug.timing`

## LLM Backend 配置摘要模板

正式执行 baseline 时，在报告旁边补一个 `baseline-notes.md`，至少写：

```markdown
# Pre-extraction Baseline Notes

- 日期：
- git commit / branch：
- 命令：
- report dir：
- scenario batch：
- repeat：
- background_wait：
- AI_GLASSES_HOME：
- AI_GLASSES_LLM_BACKEND：
- AI_GLASSES_LLM_PROVIDER：
- AI_GLASSES_LLM_MODEL：
- AI_GLASSES_LLM_BASE_URL：
- AI_GLASSES_LLM_API_MODE：
- 是否使用 `DEEPSEEK_API_KEY`：
- 是否设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK`：
- active_pass_rate：
- active_failed_turns：
- target_failed_turns：
- failed scenario ids：
- 人工抽查结论：
```

不要把 API key 明文写进报告。

## 对比规则

迁出后复跑时必须满足：

1. 使用同一份 `evals/scenarios.jsonl`，除非迁出本身修改了场景路径；如果修改，必须记录差异。
2. 使用同一组 category / scenario-id。
3. 使用同一类 LLM backend 配置。
4. 对比 `active_failed_turns`、`target_failed_turns` 和 failed scenario ids。
5. 抽查失败样本的 `response.debug`，区分是路径/配置变化，还是真实行为退化。
6. 不把 target 缺口伪装成 active 通过，也不把 baseline 已有失败归咎于迁出。

## 第十六刀首次执行结果

第十六刀已按本文三批方案首次执行迁出前 baseline，并落盘到：

```text
ai_glasses_memory_assistant/reports/standalone-migration-baseline/
```

索引 notes：

```text
pre-extraction-20260706-baseline-notes.md
```

三批报告：

- `pre-extraction-20260706-active/eval-latest.json` / `.md`
- `pre-extraction-20260706-mainline/eval-latest.json` / `.md`
- `pre-extraction-20260706-targets/eval-latest.json` / `.md`

执行时使用默认 `openai_compatible` backend，临时命令环境设置：

- `AI_GLASSES_LLM_PROVIDER=deepseek`
- `AI_GLASSES_LLM_MODEL=deepseek-v4-flash`
- `AI_GLASSES_LLM_BASE_URL=https://api.deepseek.com`
- API key 来自已有 `DEEPSEEK_API_KEY`，没有写入报告明文。
- 未设置 `AI_GLASSES_ENABLE_HERMES_LEGACY_FALLBACK`。

首次结果摘要：

| 批次 | active_pass_rate | active_failed_turns | target_failed_turns | 报告目录 |
| --- | ---: | ---: | ---: | --- |
| 核心 active 门禁 | 0.6167 | 23 | 0 | `pre-extraction-20260706-active` |
| 效果回放与主线压力 | 0.6364 | 4 | 9 | `pre-extraction-20260706-mainline` |
| target 缺口快照 | 1.0 | 0 | 3 | `pre-extraction-20260706-targets` |

注意：这是一张迁出前“老房子效果快照”，不是全绿质量证明。迁出后复跑时，不要把本次已经存在的失败算作迁出回归；重点看新增失败、失败原因变化、`target_failed_turns` 变化和关键 debug/audit 字段是否缺失。

target 批执行时观察到一个后台线程异常：`sqlite3.OperationalError: attempt to write a readonly database`。报告已生成且命令返回 0，但该异常也已记录在 notes 中，迁出后复跑时需要同口径观察。

## 本刀结论

第十五刀只固化 baseline 方案和报告模板，没有执行正式 baseline。正式 baseline 结果需要下一刀或人工按本文命令运行后再落盘记录。
