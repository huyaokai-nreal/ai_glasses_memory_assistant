# AI 眼镜个人记忆助手 - Demo 周会汇报草稿

> 状态：汇报草稿，不是当前开发计划，也不是当前代码规范。
>
> 用途：给人类做阶段汇报时复用；涉及当前真实能力、边界和路线时，仍以 `PLANS.md`、`demo-features.md`、`current-status-and-gaps.md` 和代码为准。
>
> 2026-05-14 · Jack

---

## 一、这是什么

这是一个运行在本地的 **AI 眼镜个人记忆助手 Stage 2 原型**。

当前 demo 验证的不是一套完整硬件产品，而是个人记忆助手最核心的一件事：用户自然说话、上传文档或补充文本后，系统能把有长期价值的信息整理成记忆，并在之后的问题里找回来。

和普通聊天机器人最大的不同：它不会只“问一句答一句”，而是把长期有用的信息沉淀为可查看、可搜索、可追溯的个人记忆。下次用户问“我接下来有什么安排”“我之前怎么说的”“我喜欢什么”，系统可以从本地记忆和原文证据里找答案。

可以先把它理解成：

> 用 Web UI 模拟眼镜交互，用 SQLite 保存个人记忆，用 Debug/Audit 解释每轮为什么这样回答。

---

## 二、怎么用

### 启动

从 Hermes repo parent 启动，优先使用本地 conda 环境 `hermes`：

```bash
cd /Users/huyaokai/Desktop/workspace/hermes-agent
conda run -n hermes python -m ai_glasses_memory_assistant.server
```

浏览器打开：

```text
http://127.0.0.1:8765
```

看到聊天界面、当前记忆侧栏和 Debug 入口，就可以开始演示。

### 基础操作

| 操作 | 怎么用 | 当前含义 |
| --- | --- | --- |
| 文字对话 | 在输入框打字回车 | 模拟眼镜或 App 的文本输入。 |
| 语音输入 | 点麦克风，浏览器识别后自动发送 | 只是浏览器语音能力，不是真实眼镜 ASR。 |
| TTS 播报 | 点播报或自动播放入口 | 验证回复可被语音读出。 |
| 看记忆 | 右侧当前记忆面板浏览、搜索、删除 | 查看结构化记忆和文档 metadata。 |
| Debug | 打开 Debug 面板 | 看 planner、召回、写入、耗时、LLM 调用等字段。 |
| 定位 | 浏览器弹窗点允许 | 只作为当前 turn 的临时位置上下文，不长期保存位置。 |

### 给同事演示的 5 轮脚本

这组话术贴近当前代码实现，适合 30 秒展示“写入、召回、原文回忆、计划查询”。

| 顺序 | 用户说 | 预期展示点 |
| --- | --- | --- |
| 1 | `我叫Jack。` | `turn_planner.py` 识别画像陈述，保存 `kind=profile`。 |
| 2 | `记一下明天下午3点跟产品开会。` | 先快速确认，后台 memory job 保存 `kind=event` / `memory_type=task` 候选。 |
| 3 | `我接下来有什么安排？` | 未来计划召回，查询 timed event 和近期 untimed event fallback。 |
| 4 | `我叫什么？` | 身份查询 fast path，本地读取 profile 后回答。 |
| 5 | `我之前提到产品会议时原话怎么说？` | timeline 原文搜索，返回 raw chunk，而不是只读结构化摘要。 |

演示时重点打开 Debug 面板看三件事：

- `planner.reply_mode`：这轮是本地回复，还是进入主 LLM。
- `debug.memory_processing`：候选是 pending、saved、rejected 还是 failed。
- `debug.timing.stages`：时间花在哪个阶段。

---

## 三、Pipeline 总览

一句话说完整个过程：**先留原文证据，再本地规划；能本地答就不调主模型，需要长期保存才过门控。**

> 在 HTML 展示页里，鼠标悬停蓝色概念词可以看到解释。Markdown 里也保留了 `title` 提示，支持的预览器同样可以悬停查看。

<div class="pipeline-flow">
  <div class="flow-node">
    <strong>1. 用户输入</strong><br>
    一次用户提问就是一个 <span title="turn = 一轮交互边界，包含用户这句话、后续回复、debug 和写入结果。">turn</span>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>2. <span title="TimelineStore.add_turn() = 先把用户原话写入 timeline.db，后面即使摘要不准，也能回到原文证据。">TimelineStore.add_turn()</span></strong><br>
    保存原话并拆成可搜索 <span title="chunk = 最小可搜索原文片段，也是结构化记忆 evidence_ids 可以指向的证据单元。">chunk</span>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>3. <span title="plan_turn() = 本地 planner 入口，把当前输入转成执行计划，不直接生成最终答案。">plan_turn()</span></strong><br>
    判断 <span title="fast path = 高频确定场景的本地快速回复，如问候、身份查询、画像陈述、事件记录确认。">fast path</span>、召回、web、写候选和时间范围
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>4. 上下文准备</strong><br>
    <span title="document recall = 用户问上传文档时，召回 documents 表里的原文或命中片段。">document</span> /
    <span title="structured router = planner 拿不准时的 LLM 结构化路由，只输出 JSON 决策，不负责回答。">router</span> /
    <span title="profile = 稳定用户画像、偏好和习惯。">profile</span> /
    <span title="event = 已发生或计划中的事件、任务、会议、承诺。">event</span> /
    <span title="timeline = 原文时间线，保存 raw turn 和 chunk，用于回忆用户之前怎么说。">timeline</span> /
    <span title="location = 当前 turn 临时定位上下文，不长期保存为记忆。">location</span> /
    <span title="web = 只在需要实时信息时查一次外部信息，并作为上下文注入。">web</span>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>5. 回复生成</strong><br>
    本地回复或 <span title="AIAgent.run_conversation() = Hermes 主对话循环，复杂问题会带召回上下文进入这里。">AIAgent.run_conversation()</span>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>6. 记忆写入</strong><br>
    <span title="should_write_memory_candidate() = 长期记忆门控，拒绝敏感、低置信度、问题文本和临时上下文。">Gate</span> 后同步保存或进入 <span title="memory job = 回复优先时的后台记忆写入任务，公开状态会写入 SQLite 供重启后查询。">memory job</span>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-node">
    <strong>7. <span title="chat_audit.jsonl = 每轮诊断账本，记录输入、回复、召回、写入、debug、耗时和记忆快照。">chat_audit.jsonl</span></strong><br>
    留下本轮可复盘证据
  </div>
</div>

### Pipeline 概念速查
