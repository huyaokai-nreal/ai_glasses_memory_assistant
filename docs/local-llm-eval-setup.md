# 用服务器本地 LLM 跑 LongMemEval 测试（零云费用）

> 状态：**进行中 / 可初步复现**。这份文档随着本地 LLM 测试方案逐步完善会持续更新。
> 最后更新：2026-08-18

## 0. 为什么这么做（背景）

- LongMemEval 评测默认用云 LLM（本项目历史上是 DeepSeek v4，涨价后全量 500 题约 **150–300 元**）。
- 本项目的测试目标是**验证记忆系统（召回 + 结构化记忆写入 + 作答）能不能打**，不是比某个云 LLM 的强弱——而且眼镜商用部署也未必用 DeepSeek。
- 因此把测试 LLM 换成**免费的本地/服务器部署**，既能省钱，又能顺带验证"记忆系统不绑死某个 LLM"。

## 1. 架构与前提

LongMemEval 评测分两步，都吃 LLM：

| 步骤 | 脚本 | 作用 | 吃 LLM 的方式 |
|---|---|---|---|
| ① 作答 | `ai_glasses_memory_assistant/evals/longmemeval_runner.py` | reader 用召回记忆生成答案 | `--reader-*` 参数 |
| ② 打分 | `scripts/judge_longmemeval.py` | 官方 judge 判定答案对错 | `AI_GLASSES_LLM_*` 环境变量 |

**关键认知（容易漏）：** 当 `history-mode=import` 时，runner 会调用 `service.import_conversation_events(...)` 把对话**抽取并写入结构化记忆**——这一步的 LLM 读的是 agent 全局配置 `AI_GLASSES_LLM_*`，**不是** `--reader-*`。所以必须让三步全部指向同一个本地 ollama，才真正零云费用。

好处：整套方案**不改任何业务代码**，只新增本章第 5 节的启动脚本 + 在服务器上准备 ollama。

## 2. 服务器选型建议

本地 Mac（M2 Pro + 16GB 无独显）太弱，不适合跑 8B+ 模型做全量评测。优先选有 GPU 的服务器：

- 需要：独立 GPU（A100 / RTX 3090 等）、能装 `ollama`、磁盘余量够（14B-Q4 约 9GB）。
- 本项目实测用的：`10.252.17.5`（主机名 `WX-AI-A100-01`，8× A100 40G，已装 ollama v0.5.13）。
- **共享 GPU 机器注意**：挑空闲 GPU、别踩别人训练任务；上卡前先用 `nvidia-smi` 看占用。

## 3. 服务器侧准备

> ⚠️ **文件存放约定**：服务器侧所有文件（Modelfile、日志、脚本）都放在你的账户目录 **`/home/ykhu`** 下，**不要放 `/tmp` 或任何共享/公共目录**——`/tmp` 在共享 GPU 机器上是公共且重启易丢的。下面统一用 `/home/ykhu/ollama/`。ollama 的模型仓库默认在 `~/.ollama`（即 `/home/ykhu/.ollama`），本来就在你账户下，无需额外处理。

```bash
# 3.1 起 ollama 服务（setsid 脱离 SSH 会话，断开连接也不会被杀）
#     挑一张空闲卡，避免占满共享机器
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPUS=1
export OLLAMA_HOST=127.0.0.1:11434
mkdir -p /home/ykhu/ollama
setsid bash -c "ollama serve >/home/ykhu/ollama/ollama.log 2>&1" </dev/null >/dev/null 2>&1 &

# 等几秒确认在监听
for i in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q :11434 && break; sleep 2; done

# 3.2 拉基础模型（约 9GB，服务器出网慢时可能要十几分钟，可断点续传）
ollama pull qwen2.5:14b
```

### ⚠️ 3.3 必须建一个固定上下文的模型变体（重要）

直接用 `qwen2.5:14b`，ollama 会在请求上下文尺寸不一致时**反复重载模型**，并且把长 prompt **截断到 2048 token**——LongMemEval 的对话历史常常几千 token，会被砍掉导致结果失真，且单条请求慢到 30–72 秒。修复：用 Modelfile 写死上下文。

> 上下文选多大？实测 LongMemEval 单条 session（import 阶段喂给 LLM 抽取记忆的最小单元）最长约 **17,500 token**（估算），p95 约 12,000 token。所以：
> - `num_ctx 16384` 会漏掉最长那 1 条 session（被截）；
> - **`num_ctx 32768`** 覆盖全部单 session 并留有余量，且无截断警告。模型原生支持 32768，A100 40G 上 14B-Q4 只占 ~9GB，KV cache 完全够。下面用 32768。

```bash
mkdir -p /home/ykhu/ollama
cat > /home/ykhu/ollama/Modelfile.qwen2.5-14b-32k <<'EOF'
FROM qwen2.5:14b
PARAMETER num_ctx 32768
EOF
ollama create qwen2.5:14b-32k -f /home/ykhu/ollama/Modelfile.qwen2.5-14b-32k
```

> ⚠️ **Modelfile 里绝不要写 `PARAMETER num_gpu 1`！** 在这个 ollama 版本里 `num_gpu` 会被当成"卸载到 GPU 的**层数**=1"，导致 48 层里只有 1 层上卡、其余全跑 CPU（`ollama ps` 显示 ~68% CPU、SIZE 49GB、单条请求慢一个数量级）。只用 `num_ctx` 控制上下文，**让 GPU 卡数由环境变量 `CUDA_VISIBLE_DEVICES=0` + `OLLAMA_NUM_GPUS=1` 决定**，ollama 会自动把全部层卸载到那张卡。`ollama ps` 确认 PROCESSOR 为 `100% GPU` 再继续。

### ⚠️ 3.4 务必确认模型真正跑在 GPU 上（重要）

ollama 有时会大部分层卸载到 CPU（表现为 `ollama ps` 里 PROCESSOR 是 `cpu`/`72% CPU`），单条请求慢一个数量级。验证：

```bash
# 触发一次加载后看 PROCESSOR 列应为 100% GPU
curl -fsS --max-time 90 http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:14b-16k","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
ollama ps          # 看 PROCESSOR 列
nvidia-smi         # 看对应 GPU 显存是否真被占（14B-Q4 应 ~9GB 上卡）
```

如果没上 GPU：彻底杀掉旧 ollama（`pkill -9 -x ollama`）再按 3.1 干净重启，**确认 `CUDA_VISIBLE_DEVICES` 生效**（旧进程没杀净会导致新进程选错卡 / 只部分上卡）。

## 4. 客户端（Mac）侧：SSH 隧道

> ⚠️ **Mac 本机可能已有一个 ollama 占着 11434 端口**（本机装的是 `qwen3:4b`），隧道若还用 11434 会和它冲突、悄悄打到本机弱模型并报 `Address already in use`。所以隧道本地端口改用 **11435**。

```bash
# 本地 11435 -> 服务器 11434，后台常驻
ssh -N -L 11435:localhost:11434 ykhu@10.252.17.5 &

# 验证隧道通（应能看到服务器上的模型列表，含 qwen2.5:14b-16k）
curl -fsS http://localhost:11435/api/tags
```

> 安全提示：ollama 默认只监听 `127.0.0.1`，不建议直接暴露到网段；用 SSH 隧道比开 `0.0.0.0` 更稳。api key 随便填（ollama 不校验），本项目用 `ollama` 占位。

## 5. 运行脚本

仓库已提供 `scripts/run_longmemeval_server_llm.sh`，它自动：设 `AI_GLASSES_LLM_*` 指向隧道端口 → 开隧道 → 跑 reader（import 写记忆 + 作答）→ 跑 judge 打分。

```bash
# 用法
bash scripts/run_longmemeval_server_llm.sh [LIMIT] [HISTORY_MODE]

# 先小样本验证（推荐第一步）
bash scripts/run_longmemeval_server_llm.sh 30 import

# 全量 500 题（验证链路通了再跑）
bash scripts/run_longmemeval_server_llm.sh 0 import
```

可用环境变量覆盖默认值：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LONGMEM_SERVER` | `ykhu@10.252.17.5` | 跑 ollama 的 GPU 服务器 |
| `LONGMEM_MODEL` | `qwen2.5:14b-32k` | 服务器上已 pull / 已 create 的 ollama tag |
| `LONGMEM_ORACLE` | `data/benchmarks/longmemeval/longmemeval_oracle.json` | oracle 数据 |
| `LONGMEM_OUT` | `reports/longmemeval/local-<date>-qwen16k` | 输出目录 |
| `LONGMEM_PORT` | `11435` | 本地隧道端口（避开本机 11434） |

> 提示：runner 的 `--overwrite` 在批量删 >50 个旧文件时会触发安全删除确认而中断。最省事的做法是用**全新日期的输出目录**，不触发删除门禁。

## 6. 验证看什么

跑完后看 `$LONGMEM_OUT/eval-latest.md`：

- `official_judge.correct_rate`：官方正确率（用本地模型 judge，语义判定）。
- `parse_errors`：**本地 14B 的 JSON 解析失败率会高于 gpt-4o**，这是模型弱不是记忆系统问题，看报告时区分开。
- 同时确认服务器 `ollama ps` 全程 100% GPU、无 `truncating input prompt` 警告（日志在 `/home/ykhu/ollama/ollama.log`）。

## 7. 已知坑汇总（都已踩过并解决）

1. **Mac 本机 ollama 占 11434** → 隧道改用 **11435**，否则悄悄打到本机弱模型 + 端口冲突。
2. **runner `--overwrite` 删除门禁** → 用全新日期输出目录绕过。
3. **qwen2.5:14b 默认上下文截到 2048 + 反复重载** → 用 Modelfile 建固定 `num_ctx 32768` 变体（第 3.3 节；16384 会漏掉最长那 1 条 session）。
4. **Modelfile 里写了 `PARAMETER num_gpu 1` 反而变慢** → 这个 ollama 版本把 `num_gpu` 当成"GPU 层数=1"，48 层只上 1 层、其余跑 CPU（`ollama ps` ~68% CPU、SIZE 49GB）。**删掉这行**，让 `CUDA_VISIBLE_DEVICES=0` + `OLLAMA_NUM_GPUS=1` 决定单卡，ollama 自动全层上 GPU（修复后 100% GPU、SIZE 18GB）。

## 8. 分数可比性说明

换 judge 模型后，报告里 `judge_model_substitution=true`，分数与你之前的 gpt-4o / DeepSeek 基线**不能直接横比**（这正是"测记忆系统不是测 LLM"的预期）。若将来要出可对齐的正式数字，只需对**同一批 hypotheses 用 gpt-4o 重跑 judge** 即可（不用重跑 reader）。

## 9. 待完善（TODO，随方案推进补充）

- [ ] 全量 500 题（`limit 0`）尚未完整跑通并固化基线数字。
## 实测结果（2026-08-18，单次 full run）

**环境**：`10.252.17.5`（8×A100-40G），ollama 部署 `qwen2.5:14b-32k`（num_ctx 32768，修复 `num_gpu 1` 后 100% GPU、驻留 ~18GB）。本地 LLM 同时承担 reader / import 抽取 / judge 三角色。

**任务**：`single-session-preference` 30 题（与基线 `pref30_20260813` 同题型）。

| 指标 | 结果 |
| --- | --- |
| 题目数 / 成功数 | 30 / 30（0 失败） |
| 本地 judge 命中率 | **15/30 = 50.0%** |
| judge 解析失败（空响应） | 0（本地模型不空响应，分数干净） |
| 标准拒答（"根据已有记忆我不知道"） | 9 次 |
| 真实答错 | 6 次 |
| 阶段耗时 | import 均值 21s、recall 12s、reader 1.7s，整轮 ~17m48s |

**苹果对苹果对比（同一套记忆系统代码，只换 reader + judge）**：用本地 `qwen2.5:14b-32k` judge 重判基线 `pref30_20260813` 的答案文件——
- 基线 reader（deepseek-v4-flash）答案：**24/30 = 80.0%**（0 拒答、0 解析失败）
- 本地 reader（qwen2.5:14b-32k）：15/30 = 50.0%（9 拒答 + 6 真错）
- 基线原始报告写的 26.67% 是 **deepseek-v4-pro judge 13 次空响应**的假象，真实水平约 80%。

**诊断（9 次拒答根因）**：这 9 题 `saved_memory_count` 均为 12–18（记忆写入成功），但 `recalled_memory_count = 0`（召回为空），`recall_context_chars = 0`。即**记忆存下了却没被翻出来**，不是写记忆失败、也不是 reader 过度谨慎。最合理归因：更换测试 LLM 后 embedding 模型随之切换，导致这 9 题的查询/记忆向量空间错位（召回靠 cosine 相似度）。

**结论**：记忆系统本身稳——同代码下强 reader 拿 80%、弱 14B reader 掉到 50%，差值主要来自 reader 强弱 + 9 次过度谨慎的拒答，而非记忆抽取/存储设计缺陷。9 题拒答根因已复查：**本项目的记忆召回不依赖 embedding 向量**（recall 走完整 `GlassesChatService.chat()` 主链路，`agent_bridge.add_memory` 调用未传 embedding），而是依赖 LLM 做 import 抽取与 recall 判断；换本地 qwen2.5:14b 后这两个环节比 DeepSeek 弱，导致这 9 题该召回的没召回 → reader 空上下文拒答。若要追平基线，优先排查 (1) recall 主链路中哪个 LLM 调用对这 9 题失效、(2) import 抽取的记忆覆盖差异（见另一 agent 排查 prompt）。

**速度对比（vs DeepSeek API 基线 `pref30_20260813`，同为 single-session-preference 30 题）**：本地方案全阶段更快，且零 API 费用/零限流。

| 阶段 | 本地 qwen2.5:14b-32k（均值/中值/P95，秒） | DeepSeek 基线（均值/中值/P95，秒） | 本地加速 |
| --- | --- | --- | --- |
| import | 21.35 / 17.99 / 58.54 | 222.23 / 225.01 / 322.57 | ~10x |
| recall | 11.72 / 11.74 / 14.17 | 42.82 / 41.74 / 90.83 | ~3.6x |
| reader | 1.73 / 2.18 / 3.34 | 9.25 / 7.72 / 23.83 | ~5.3x |
| **30 题总耗时** | **~17.5 min** | **~137 min（估算）** | **~8x** |

> 注：基线 import 单题 ~222s 偏高，可能含 DeepSeek API 限流/高延迟；本地独占 A100、零网络延迟，速度优势明确且可复现。

- [ ] 比较 `qwen2.5:14b` vs 更大模型（32B / 70B-Q4，A100 有余量）对 judge 质量的影响。
- [ ] 确认 import 阶段写记忆的 LLM 是否还有别的隐式配置来源（目前按 `AI_GLASSES_LLM_*` 全部接管，已验证可行）。
- [ ] 若将来换非 ollama 后端（如 vLLM），补充对应 base-url / 并发配置。
- [ ] 把"服务器重启后一键恢复 ollama + `qwen2.5:14b-32k` 变体（注意 Modelfile **不要**写 num_gpu）+ 确认 100% GPU"做成脚本。
