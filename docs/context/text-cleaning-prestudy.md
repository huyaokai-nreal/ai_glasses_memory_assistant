# AI 眼镜文本清洗能力预研与新方案

日期：2026-05-29  
目标：在真实语境压力测试前，为本项目形成一套自研文本清洗方案。方案要吸收开源和论文里的有效做法，但避免把系统再次做成短语规则打地鼠。

## 结论先行

本项目的文本清洗不应该继续走“看到某个词就判定整段是闲话/噪音/不用记”的路线。那会重蹈早期 local planner 的问题：每遇到一种真实说法就补一条规则，最后变成无限打地鼠。

新的原则是：

```text
短语规则 = 传感器
LLM-first 结构化判断 = 语义裁判
写入门控/eval = 最终验收
raw timeline = 永远保留的事实证据
```

也就是说，规则层可以标记“这里有嗯嗯/哈哈/有点吵/不用记”，但不能轻易决定“整段丢掉”。真正要判断的是：

- 口头禅后面是否仍有任务。
- “不用记”到底作用于哪一小段。
- 插话、转述和背景噪声是否影响候选抽取。
- 一句话是闲聊、临时上下文、长期记忆候选，还是敏感拒绝。

例子：

```text
嗯嗯，那个外卖电话不用记，但 Mia 说明天一定要验证语音按钮。
```

规则层只能标记：

```json
{
  "markers": ["filler", "do_not_remember"],
  "raw_text_kept": true
}
```

LLM 语义层才应该判断：

```json
{
  "segments": [
    {
      "text": "嗯嗯，那个外卖电话不用记",
      "role": "reject",
      "reason": "user says not to remember the food delivery phone"
    },
    {
      "text": "Mia 说明天一定要验证语音按钮",
      "role": "memory_candidate",
      "candidate_type": "task",
      "should_extract": true,
      "evidence_span": "Mia 说明天一定要验证语音按钮"
    }
  ]
}
```

用户能感知到的结果是：系统不会因为一句话里出现“嗯嗯”就跳过真实任务，也不会因为出现“不用记”就把后半句真正重要的安排一起丢掉。

## 检索证据与取舍

本轮重新看 Google、GitHub、arXiv 和工业实践资料后，重点不是找更多关键词，而是找“清洗系统如何避免规则打地鼠”的设计。能借鉴的共识很清楚：成熟做法通常把清洗拆成 pipeline/operator、结构化 schema、validation、trace 和 bad-case mining，而不是靠一个关键词列表做最终裁判。

| 资料 | 可以取的精华 | 要避开的糟粕或不适合点 | 本项目落法 |
| --- | --- | --- | --- |
| [Data-Juicer](https://github.com/datajuicer/data-juicer) | 数据清洗应做成可组合 operator pipeline，每步有输入、输出、统计和报告。 | 它面向大规模训练数据，不是实时个人记忆；不能照搬重型数据处理栈。 | 借鉴 operator 化：normalize、mark、semantic classify、extract、validate 分开。 |
| [DataFlow](https://github.com/OpenDCAI/DataFlow) | 用 pipeline 编排 LLM 数据处理算子，强调可替换、可观测和可复用。 | 不引入新框架或外部数据流服务，避免把 demo 复杂化。 | 借鉴“每个算子有 schema 和 trace”，先在现有 Python service 内实现。 |
| [DocETL operators](https://ucbepic.github.io/docetl/concepts/operators/) | LLM 处理非结构化文本时，用 prompt + output schema + validation，而不是自由改写。 | 它偏离线文档处理，不直接适配聊天实时链路。 | 照抄“结构化输出”思想：LLM cleaner 只输出 JSON，不覆盖原文。 |
| [Conversational Speech Reveals Structural Robustness Failures in SpeechLLM Backbones](https://arxiv.org/abs/2509.20321) | 口语里的插入、改口和旁枝会暴露模型结构修复能力问题；去 disfluency 本质上是受控删除任务。 | 不能把 LLM 当万能清洗器，让它直接生成“干净版事实”。 | LLM 只判断 role/span/should_extract，禁止无证据重写事实。 |
| [DisfluencyFixer](https://arxiv.org/abs/2305.16957) | 口语修正要保留 disfluency 类型、数量和处理痕迹。 | 它面向 ASR-TTS 流水线，我们不需要 TTS，也不能删除纠错证据。 | 保留 `cleaning_trace` / `extraction_trace`，把 filler、correction、negation 作为 trace。 |
| [NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing) / ITN | 数字、日期、单位、金额等格式转换适合规则或 WFST，要求可解释。 | Pynini/OpenFST 依赖重；当前 demo 还没有真实 ASR runtime。 | 暂时只借鉴“确定性格式处理要可控”，不引入重依赖。 |
| [Microsoft Presidio](https://github.com/microsoft/presidio) | PII 检测、脱敏和 recognizer 思路成熟，适合多层防线。 | 当前已有本地隐私过滤；暂不服务化引入 Presidio。 | 继续增强本地 `privacy_filter`，保持 raw timeline、LLM 输入和 memory 写入前的敏感边界。 |
| [deep-disfluency-detector](https://github.com/pariajm/deep-disfluency-detector) | 学界通常用模型检测 disfluent words，而不是纯手工关键词。 | 旧 TensorFlow/训练数据依赖不适合直接搬进 demo。 | 证明“纯规则不够”，后续可用本地 LLM 或轻模型替代关键词裁判。 |
| [whisper-punctuator](https://github.com/jumon/whisper-punctuator) / punctuation projects | ASR 后处理常把标点恢复作为独立派生步骤。 | 标点模型可能对走路口述、多人插话、中英混合不稳定。 | 标点/切句只能作为派生 segment 信号，不覆盖 raw chunk。 |
| [Cleanlab](https://github.com/cleanlab/cleanlab) | 用模型和报告发现数据质量问题，而不是人工枚举所有坏样本。 | Cleanlab 面向数据集质量，不是实时记忆写入。 | 借鉴 bad-case mining：压力测试失败样本反哺清洗 eval。 |

这些资料给本项目的说服力不是“某个项目也用了同样的规则”，而是“多个成熟方向都在把清洗做成可组合、可验证、可追溯的处理链”。所以我们可以比较放心地自研轻量版，而不是引入重框架：

- **可以照抄的设计**：operator 边界、schema output、validation/trace、bad-case mining。
- **只能借鉴不能照搬的实现**：大规模 Ray/数据集 pipeline、Pynini/WFST text normalization、独立 PII 服务、训练数据质量平台。
- **必须坚持的边界**：清洗结果不覆盖 raw timeline；LLM 不直接写长期记忆；短语规则只当传感器。

一句话取舍：

```text
取 operator pipeline、schema output、trace、validation、bad-case mining。
弃重框架、黑盒改写、纯规则裁判、直接覆盖原文。
```

## 新方案：AI 眼镜文本清洗七层 pipeline

### 1. Raw Evidence Layer：原文证据层

职责：保留原始输入证据，必要时先脱敏。

要求：

- raw timeline / capture chunk 仍是事实源。
- 清洗文本不能替代原文。
- 所有候选都必须能回到 `evidence_ids`。
- 敏感内容在 raw 存储、debug、audit 和 LLM 输入前都要有保护边界。

例子：

```text
原文：验证码 123456，啊别记这个
结果：timeline 脱敏；长期记忆不保存验证码；debug 说明命中 sensitive + do_not_remember
```

### 2. Deterministic Sensor Layer：确定性传感器层

职责：只做高确定性的规范化、脱敏和弱信号标注。

可以做：

- Unicode / 空白 / 重复标点规范化。
- token、验证码、证件号、银行卡、电话、email、URL 等敏感或格式化片段识别。
- 标记 filler cue：嗯、啊、那个、哈哈、怎么说呢。
- 标记 noise cue：有点吵、听不清、背景杂音。
- 标记 correction cue：哦不对、刚才说错、改成、应该是。
- 标记 do-not-remember cue：不用记、别保存、不要记这个。

不能做：

- 不能因为有 filler 就跳过整段。
- 不能因为有 “不用记” 就跳过后面所有内容。
- 不能把标点恢复、数字改写或日期改写当成事实替换。
- 不能把规则命中当成最终语义结论。

这一层输出的是 `markers`，不是最终判决。

### 3. Segment Proposal Layer：候选分段层

职责：给后续语义判断提供候选片段。

分段信号可以来自：

- 标点、换行、停顿词。
- 转折词：但是、不过、后来、对了、另外。
- correction cue 前后。
- do-not-remember cue 的局部作用范围。
- 长输入长度和主题切换。

关键原则：

```text
分段可以多切一点，不能少切到把“不用记”和真实任务绑死在一起。
```

例子：

```text
外卖电话不用记，但 Mia 说明天测语音按钮
```

应至少拆成：

```text
外卖电话不用记
Mia 说明天测语音按钮
```

### 4. LLM Semantic Cleaner：LLM-first 语义清洗层

职责：判断每个 segment 在真实语境里的作用。

这一层是新方案的核心，用来避免规则打地鼠。它不生成最终记忆，也不改写 raw text，只输出结构化 JSON。

建议 schema：

```json
{
  "segment_index": 1,
  "raw_span": "嗯嗯 Mia 明天要验证语音按钮",
  "semantic_role": "memory_candidate",
  "noise_level": "low",
  "contains_filler": true,
  "do_not_remember_scope": "",
  "should_extract": true,
  "candidate_span": "Mia 明天要验证语音按钮",
  "candidate_hint": "task",
  "reason": "filler does not invalidate the task"
}
```

`semantic_role` 建议收敛为：

| role | 含义 | 后续处理 |
| --- | --- | --- |
| `memory_candidate` | 可能值得长期保存 | 进入候选抽取 |
| `temporary_context` | 只服务当前回复 | 不写长期记忆 |
| `chitchat` | 闲聊/情绪/寒暄 | 默认不写 |
| `noise_only` | 背景噪声或无意义转写 | 不写，只保留 trace |
| `correction` | 纠错/改口 | 进入纠错候选流程 |
| `do_not_remember` | 用户明确不要保存的局部内容 | 不写，并记录范围 |
| `sensitive_reject` | 敏感信息或凭证 | 拒绝或确认，不进入候选抽取 |

这一层必须遵守：

- 不允许直接保存。
- 不允许输出没有原文依据的事实。
- 不允许删除 raw evidence。
- 输出必须带 `raw_span` / `candidate_span`。
- 低置信度时宁可交给后续门控拒绝，也不要静默写入。

### 5. Memory Candidate Extractor：结构化候选抽取层

职责：只从 `should_extract=true` 的 span 中抽取候选。

候选类型继续贴合现有记忆体系：

- `profile / preference`
- `event / task`
- `event / decision`
- `event / project_state`
- `assistant_preference`
- `correction`
- `reject`

建议输出：

```json
{
  "content": "Mia 明天要验证语音按钮",
  "kind": "event",
  "memory_type": "task",
  "confidence": 0.86,
  "evidence_span": "Mia 明天要验证语音按钮",
  "source_segment_index": 1,
  "reason": "explicit future task"
}
```

注意：这一层现在应交给 unified semantics 或专门的 segment-level extractor，避免把 web intent、router 和 memory extraction 混在一起。

### 6. Gate / Dedupe / Conflict Layer：写入门控层

职责：最终决定能不能进入长期记忆。

继续复用现有机制：

- `should_write_memory_candidate()`
- privacy gate
- confidence gate
- semantic dedupe
- correction target resolution
- task status
- `evidence_ids`
- source trace / audit

新方案特别强调：

```text
LLM semantic cleaner 只是裁判候选片段有没有价值，不是写库权限。
```

例子：

```text
我的 token 是 sk-xxx，别记
```

即使 LLM 错误判断成 `memory_candidate`，门控也必须拒绝。

### 7. Pressure Eval Layer：真实语境压力测试层

职责：用失败样本反推清洗能力，而不是继续凭感觉补规则。

每个压力测试样本要记录：

- `raw_input`
- `markers`
- `segments`
- `semantic_cleaner_output`
- `expected_candidates`
- `actual_saved_memories`
- `rejected_reasons`
- `evidence_ids`
- `failure_type`

建议优先覆盖：

| 场景 | 示例 | 要验证 |
| --- | --- | --- |
| filler 混任务 | “嗯嗯 Mia 明天测按钮” | 不因 filler 跳过任务 |
| do-not-remember 局部范围 | “外卖电话不用记，但 Mia 要测按钮” | 只拒绝外卖电话 |
| 自我纠正 | “周五交，哦不对周四晚” | 后者优先，保留 correction trace |
| 否定偏好 | “不是喜欢 coco，是喜欢喜茶” | 不保存旧偏好 |
| 多人转述 | “Alex 说 Bob 明天去医院” | 不误写成用户本人日程 |
| 敏感误听 | “验证码 123456，别记” | 脱敏、拒绝、可诊断 |
| 长口述漂移 | 30-100 轮模拟用户 | 检查污染、过删、漏写、召回漂移 |

## 推荐实现路线

### Phase A：保留现有 deterministic trace，但降级为传感器

状态：已完成第一版，后续需要调整语义定位。

当前已有：

- `text_cleaning.py`
- `normalized_text`
- `segments`
- `noise_markers`
- `correction_markers`
- `negation_markers`
- `redaction_categories`
- chat/import/capture/eval debug 中的 `cleaning_trace`

下一步要改的不是删除它，而是把文档和代码里的使用方式改清楚：

```text
markers 只能提示风险，不能充当最终跳过依据。
```

### Phase B：把长输入抽取改成 LLM semantic cleaner + extractor 两步

状态：已开始落地第一版。

当前已有：

- 长输入 `continuous_capture` 使用 `cleaning_trace.segments`。
- job/debug/audit 暴露 `extraction_trace`。
- LLM 分段候选继续走门控、去重和 evidence。
- 新增 `segment_semantic_cleaner.py`，提供 `SegmentSemanticClassifier` 的 prompt/schema 和 fallback。
- 后台长输入链路新增 `semantic_cleaning.segment_decisions`、`extractable_segment_count`、`skipped_segment_count` trace。
- 当有 agent 可用时，先由 semantic cleaner 判断 `should_extract` 和 `candidate_span`，再把 span 交给候选抽取；规则 marker 不再直接决定“整段是否送 LLM”。

当前第一版链路：

```text
segments
-> SegmentSemanticClassifier
-> only should_extract spans
-> MemoryCandidateExtractor
-> gate/dedupe/evidence
```

也就是：规则层只给 LLM 提供 markers，LLM 语义层负责判断局部作用范围和可抽取 span。

### Phase C：新增 schema 和 trace

状态：第一版已随 `SegmentSemanticClassifier` 落地，后续重点是扩大压力测试和校准。

当前调试结构应稳定为：

```json
{
  "text_cleaning": {
    "markers": [],
    "segments": []
  },
  "semantic_cleaning": {
    "backend": "local_llm",
    "segment_decisions": [],
    "skipped_segment_count": 0,
    "extractable_segment_count": 0
  },
  "memory_extraction": {
    "candidate_count": 0,
    "gate_rejected_count": 0
  }
}
```

这样以后排查问题时可以回答：

- 是规则没标出来？
- 是 LLM semantic cleaner 判断错了？
- 是 extractor 抽错了？
- 是 gate 拒绝了？
- 是 dedupe 合并了？

### Phase D：真实语境压力测试闭环

先不追求大而全，先做 target eval；每个失败样本都要反查它卡在哪一层，而不是直接补关键词：

1. `filler_with_task_scope_target`
2. `do_not_remember_scope_target`
3. `self_correction_scope_target`
4. `multi_speaker_attribution_target`
5. `long_chitchat_no_memory_pollution_target`
6. `sensitive_asr_false_positive_target`

稳定后再小批量升 active。

## 当前代码和新方案的差距

| 项目 | 当前状态 | 差距 |
| --- | --- | --- |
| raw timeline 保留 | 已有 | 继续保持 |
| 确定性 marker | 已有第一版 | 需要明确降级为传感器 |
| 长输入分段 | 已有第一版 | 需要更关注 do-not-remember 和 correction scope |
| LLM 分段抽取 | 已接入 semantic cleaner 第一版 | 继续扩更多语义角色和失败兜底 |
| 结构化 schema | 已有 `SegmentSemanticDecision` 第一版 | 需要继续稳定字段和 eval 覆盖 |
| 写入门控 | 已有 | 继续作为最终验收 |
| 压力测试 | 部分 target | 需要补范围作用、多说话人、过删/漏写场景 |

## 风险边界

- 不把 LLM 清洗输出当事实源；事实源仍是 raw timeline。
- 不让 LLM 自由改写成“干净文本”；只允许输出结构化判断和 span。
- 不让短语规则决定最终语义；规则只标记 cue。
- 不新增重依赖或外部清洗框架，先在现有 Python service 内实现 operator 化结构。
- 不把当前 Phase B 说成最终文本清洗能力；它只是第一版分段抽取。
- 不把真实 ASR、标点模型、Presidio 服务化、NeMo ITN 写成已完成。

## 建议下一步

1. 继续扩展 `SegmentSemanticClassifier` 的 target eval，优先覆盖 `do_not_remember` 局部范围、filler 混任务、多说话人归因、自我纠正范围。
2. 把 `semantic_cleaning` trace 接入更多报告视图，方便排查“规则没标出 / semantic cleaner 判断错 / extractor 抽错 / gate 拒绝”。
3. 继续校准 semantic cleaner prompt，避免 LLM 过删、漏抽或改写事实。
4. 稳定后再考虑是否把少量高确定性规则升级为 active gate。
