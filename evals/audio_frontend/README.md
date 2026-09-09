# evals/audio_frontend — 多通道重叠语音前端评测管线

## 目的

评估 AI 眼镜在"多人同时讲话"场景下，靠**多麦克风阵列 + 说话人活动检测 + 空间增强**，能否显著降低语音识别（ASR）错误率。这是一条**离线研究评测**管线，用于证明各组件指标是否达标，是端到端产品链路的**前置验证**——它本身不是可发货的连续推理服务。

## 整体链路

```
多通道音频 (AliMeeting 8ch far)
  → 说话人活动检测 (Sortformer diarization)
  → 空间增强 (MVDR beamforming, oracle 或 auto-mask 驱动)
  → SenseVoice ASR
  → CER / DER 评分 + 门禁判定
```

## 目录文件

### 数据准备
- `prepare_ali_multichannel.py` — 裁 8 个 75s 8ch 窗口 + 相对窗口 RTTM（仅匿名 speaker label，无 gold text）+ `prep-manifest.json`（含 4 通道变体、ASR hash、输入 SHA-256）。是 75s 窗口评测的入口。
- `prepare_ali_continuous.py` — 完整会议连续输入 manifest 准备（8 场、4.21h、6457 条 TextGrid 金标区间），供连续 diarization 使用。

### 说话人活动检测（自动分轨）
- `diarize_sortformer.py` — 调用锁定 revision 的 Streaming Sortformer，产出每会议说话人活动时间轴（CPU 推理，无需 GPU / 服务器）。
- `tune_sortformer_threshold.py` — 在 stratified 4/4 切分上扫描概率阈值（最终选定 0.30），输出 threshold-scan（含 tune / holdout 门禁）。
- `score_diarization.py` — 用金标 TextGrid 给预测说话人时间轴打分：DER、重叠帧召回/精度、逐 case 指标。

### 空间增强
- `enhance_mvdr_oracle.py` — 用**人工标签（oracle）**活动掩码驱动纯 NumPy MVDR 波束形成，产出逐说话人增强音轨（8 会议 × ch0/micA/micB/all8 四变体）。这是"空间信息价值"的**上限验证**。
- `enhance_mvdr_auto_mask.py` — 用 **Sortformer 自动活动掩码**驱动同一套 MVDR（替代人工标签），验证"不靠人工"时增益保留多少；内部复用 `enhance_mvdr_oracle`、`score_diarization`、`tune_sortformer_threshold`。

### 连续端到端（单场 smoke）

- `run_continuous_frontend.py` — 在 `py311` 中把一场完整会议的 8 通道 WAV + Sortformer 全会议概率列，按 **30 秒 core block + 3.2 秒右上下文**水位线推进，逐匿名轨道做 MVDR（或 max-energy fallback），输出对齐的 `all8` / `ch0` 片段。只处理 `--case-id` 显式指定的 case；prep manifest 字段经白名单过滤，**TextGrid 文本与金标区间永不进入推理**。
- `score_continuous_e2e.py` — 在 `hermes` 中用锁定的 SenseVoice 转写，计算**最多四轨置换不变拼接 CER（cpCER）**、按 TextGrid 区间重建音频后的 all / non_overlap / overlap CER、复用既有定义的 DER 与重叠帧召回，并生成 `audio_event.v1` 事件；可选 `--replay-service` 在临时 `AI_GLASSES_HOME` 中回放并断言零记忆写入、零网络调用。

- `run_full_loop_timed.py` — 用 subprocess **背靠背**拉起上面两个阶段（各自 `conda run --no-capture-output -n <env> python <script>`，子命令解释器固定为 `python`，绝不用父进程 `sys.executable`），以单一 `perf_counter` 从进程启动计时到进程退出，输出 `full-loop-timing.json`（`frontend_process_seconds` / `scorer_process_seconds` / `stage_seconds_sum` / `full_process_seconds` / `full_process_rtf`，阈值 ≤1.0）。两段由同一时钟测量且互不重叠，因此可相加；人工等待由构造方式排除。**已存在的 `--out` 目录一律拒绝**（退出码 5）。退出码：1 前端失败、2 评分失败、3 外层 RTF 超阈、4 读不到 `processed_seconds`、5 输出目录已存在。
- `verify_frontend_wavs.py` — 对已产出的前端目录做**磁盘级**校验（非 pytest 替代）：`wav_frames == sample_count == round((end_s-start_s)*16000)` 精确相等、`wav_sha256` 匹配、同轨无重叠无越界、30/60 秒 block 边界两侧严格首尾相接。
- 门禁范围：`score_continuous_e2e.py` 的 `--gate-scope {auto,smoke,full}`。只有 manifest 能**证明截短**（`--limit-seconds > 0` 且 `processed_seconds` 严格短于会议时长）时才允许 `smoke`（DER / 重叠帧召回仅作诊断）；缺失、不可解析或无法证明一律回退 `full`，显式 `smoke` 降级完整会议会被拒绝。`full` override 只能加强检查。

### 连续端到端（full8 八场批量层）

- `run_continuous_full8.py`（薄批量驱动，`summarize_continuous_full8.py` 为失败状态机 / 加权聚合 / 批次门禁的**唯一定义源**并被驱动 import）。**一条前台命令**跑完整批：preflight → 8 场串行（每场 = wrapper → verifier → 从真实产物分类）→ 原子 `batch-run-manifest.json` → 汇总 `full8-summary.json`。**必须由用户在前台亲自运行**（长任务，数小时级；下班前启动、次日复核是预期用法）：
  ```bash
  conda run --no-capture-output -n hermes \
    python evals/audio_frontend/run_continuous_full8.py \
      --batch-root reports/p4_continuous_e2e/<NEW-dir> \
      --prep-dir reports/p3_continuous_prep/20260907-full8 \
      --predictions reports/p3_sortformer_continuous/20260907-full8-thr030 \
      --source-run reports/eval_ali/<source-run> \
      --frontend-env py311 --scorer-env hermes --wrapper-env hermes
  ```
- **失败语义**：`evidence_fail`（网络/记忆/回放审计/verifier/WAV/hash 漂移/产物缺失，任一）→ 该场不入选、**批次立即中止、不重试**；逐场 result 失败（无合格 final、cpCER 相对改善失败、重叠 CER 相对改善失败、内部 RTF >1.0、外层 RTF >1.0）→ 仍入选、**批次继续跑完但最终必须红**（汇总门禁 `all_session_result_gates_passed`）；瞬时故障（驱动捕获的启动异常 / 预定义超时）→ 重试 ≤2 次；Ctrl-C/SIGTERM → 信号转发给当前子进程、记录 `interrupted`、**不重试不启下一场**。DER ≤25% 与重叠帧召回 ≥70% 是**聚合层专属门禁**，从不给单场打 quality 失败。
- **运行前硬约束（任一不满足即拒绝启动，不写任何文件）**：① `--batch-root` 目录必须尚不存在；② 6 个运行时源必须已 commit 且与 HEAD 一致；③ 工作树干净、HEAD 不漂移（每场后复核，漂移即停且不产绿色 summary）；④ prep 的 case 集合必须**恰好等于锁定 8 场且顺序一致**（不存在放宽为 7 场的入口）；⑤ 两份源 manifest 的模型 hash 必须等于固定锁（manifest 不得自我认证）；⑥ 磁盘可用空间 ≥ 输出预算 + 安全余量；⑦ `py311`/`hermes` 的 python 路径与版本必须可解析（写入 manifest `env`）。
- manifest `post_run` 记录 `finished_at` / `head_sha` / `status` / `code_drift` / 运行时源 hash；`locks` 记录 prep / predictions / source-run 三份输入 manifest 的 sha256 与逐 case 输入 hash。

#### full8 六项加固（汇总入口严格校验 + 运行约束）

针对 full8 批次的六条加固，全部 fail-closed：

1. **汇总入口严格校验**（`summarize_continuous_full8.check_manifest_integrity`）：`schema` 必须精确等于 `continuous_full8_batch.v3`、`plan_id` 必须精确等于 `2026-09-08-audio-continuous-e2e-full8`；`git.status` 必须是显式空字符串、`git.runtime_sources_match_head` 必须严格为 `True`；manifest 必须列**恰好锁定 8 场、有序、唯一**；每场都有 `selected_attempt_dir`；`post_run` 必须存在、为对象，且 `status=="completed"`、`finished_at` 为**非空字符串**、`code_drift` 必须是**显式的空列表**、`git_status` 必须是**显式的空字符串**——**缺失不等于干净**（`None` / 非列表 / 非字符串一律失败）；`runtime_sources_locked is True`（严格布尔真）；HEAD 前==后；`locks.model_hashes` 两个模型锁都**完整且等于固定常量**；**前后源码 hash 集合必须完整、集合相同、逐项值相同**（缺文件 / 缺 hash / hash 不一致 / 集合非 dict 均非零失败）。任一不满足 → 批次红、非零退出。生产逻辑已删除"纯单测可缺锁"分支——fixture 必须携带真实的 `runtime_source_sha256` 前后集合、`git.status`、`git.runtime_sources_match_head` 等必需字段，不能只放通过布尔值。
2. **数值真实性**（`check_numeric_consistency` + `check_rtf_authenticity`，`math.isfinite`）：拒绝 bool / NaN / Infinity / 非法计数；外层 RTF = `full_process_seconds / processed_seconds` 重新推导并对 `full_process_rtf_passed` 重新判定；内部 RTF = `frontend_rtf + scoring_rtf + replay_rtf` 与记录的 `full_loop_rtf` 核对（6 位小数容差 `RTF_TOL=1e-4`）；"RTF=9 但 passed=true" 一律判质量失败而非证据失败。
3. **源码锁（目录枚举，不再人工枚举 import）**：`enumerate_managed_sources()` 用 `git ls-files` + `git ls-files --others --exclude-standard` 枚举 `evals/audio_frontend`、`ai_glasses_memory_assistant`、`scripts` 三个运行代码目录下**全部受管 `*.py`**（当前 92 个：86 tracked + 6 untracked），对每一个做 HEAD 一致性检查（`git diff --quiet HEAD -- <path>`，覆盖 staged+unstaged）。**同时枚举 untracked** 是关键：只枚举 tracked 会漏掉尚未首次提交的新代码（例如 full8 的六个入口脚本本身），让"上锁的代码"逃过锁；untracked 文件仍写入 manifest hash 表，但 `matches_head` 判其不干净（不在 HEAD 里就不可能 HEAD-clean），因此**真实运行会拒绝启动，直到它们被提交**。运行前、每场后、结束时三次检查共用同一枚举与同一实现（`run_env_snapshot`）。每个文件的 sha256 保留在 manifest `git.runtime_source_sha256` 便于审计。任何受管源码（含 MVDR 实现 `enhance_mvdr_oracle.py` 与 `agent_bridge.py`）改动 → 运行前拒绝，或本场结束后**立即停止、不启动下一场**。
4. **超时终止整个子进程组**：wrapper 以 `start_new_session=True` 启动，`_kill_process_group` 用 `os.killpg` 杀整组（含 conda run → python → 解码器）；超时分支先 `killpg`+`communicate()` 确认退出再允许重试——不残留孤立后代。中断（SIGINT/SIGTERM）转发给进程组后记录 `interrupted`，**不重试**。
5. **磁盘预算按真实输出推导（已删除无依据的 5 GiB/场默认值）**：`derive_disk_budget()` 从八场 processed duration（取自 prep manifest 的 `total_audio_seconds`，真实值 **15138.215 s**）推导 WAV 上限。前端落盘的是**每个匿名轨道一个单声道 PCM16 WAV**，不是 8 通道 WAV；Sortformer 是固定 4 说话人模型（`diar_streaming_sortformer_4spk-v2.1`），因此

    `wav = 时长(s) × 16000Hz × 2B(PCM_16) × 4 轨 × 2 变体(all8+ch0)`

    全部统一使用 **GiB（除以 `1024**3`）**，常量、字段名、帮助文本与 manifest 键一律写 `_gib`，**不把 decimal GB 写成 GiB**。
    再加每场 0.05 GiB 的文本/日志余量、retry 留存（`× (max_retries+1)`）与 8 GiB 安全余量。**源输入已存在，不重复计入新增空间**。推导的每一项（`total_processed_seconds` / `sample_rate` / `bytes_per_sample` / `max_tracks` / `num_variants` / `wav_gib` / `text_log_gib` / `per_attempt_total_gib` / `retry_factor` / `n_sessions` / `safety_margin_gib` / `available_gib` / `total_needed_gib` / `budget_ok` / `params_ok` 及 `derivation` 文字）全部写入 manifest `disk_budget`；各项可逐项相加复核（`per_attempt × retry + margin == total`）。**可用空间不可读取 → `budget_ok=False`；预算参数非法（`None`/`<=0`/bool）或空间不足 → 真实运行（`enforce_git=True`）直接拒绝**；CLI 已移除 `--disk-budget-gb-per-session`，**不许临时降低预算绕过**；绝不删除任何现有产物腾空间。
    默认 `--max-retries 2` 的预算：`WAV 3.609 GiB + 日志 0.400 GiB = 4.009 GiB/attempt；×3 + 8 GiB = 20.028 GiB`。
6. **环境探测失败即停**：启动 wrapper 前先 `EnvProbe` 解析 `frontend_env` / `scorer_env` / `wrapper_env` 三套环境的 python 与版本并写入 manifest `env`；任一解析为 None → 拒绝启动、不写任何文件。已解析的 conda 路径显式传入 wrapper 命令（首位参数）。

#### full8 真实运行前置与 batch manifest 完整性（第五轮）

7. **真实运行要求整个工作树干净**：`enforce_git=True` 时除受管 Python 源码锁之外，还要求 `git status --porcelain` **整体为空**——**非 Python 文件的修改或未跟踪文件也必须在启动前拒绝**，不能等八场跑完才失败。dry-run 不拒绝（它不写任何文件），但会把真实 status 记进 manifest。受管 Python 源码锁与逐场漂移检查全部保留（运行前 / 每场后 / 结束，共三次）。
8. **batch manifest 完整性 + 常量单一权威**：`schema` 与 `plan_id` 在 `summarize_continuous_full8` 中**各只有一处权威定义**（`SCHEMA = "continuous_full8_batch.v3"`、`PLAN_ID = "2026-09-08-audio-continuous-e2e-full8"`），driver 从该模块 import 后 re-export，写入方与校验方不可能漂移（driver 侧另有两条 `assert` 兜底）。汇总入口要求：`schema` 与 `plan_id` **精确相等**（缺失 / 错误类型 / 错误值均失败）；`git.status` 必须是**显式空字符串**；`git.runtime_sources_match_head` 必须**严格 `is True`**（`1` / `"yes"` 等真值亦失败）。

### 评分与门禁
- `score_multichannel_cer.py` — 把增强音轨喂 SenseVoice，按 `segment_id` 映射金标文本，算 all / non_overlap / overlap_exposed CER 与删除/替换/插入；缺/重/未知 segment 与 ASR hash 漂移均 fail-closed，并自动判四项门禁。
- `score_auto_mask_retention.py` — 比较 oracle 与 auto-mask 的 CER 增益保留率（retention），综合 diarization tune/holdout 门禁给出组件级 `passed` / `false`。

### 测试
- `tests/test_audio_frontend_*.py` — 对应每个脚本的单测（`test_audio_frontend_score_diarization.py` 含 retention 字段断言）。`diarize_sortformer` 与 `tune_sortformer_threshold` 的测试为纯函数单测，不下载模型、不联网。
- `tests/test_audio_frontend_continuous_frontend.py` — 连续前端纯函数与合成信号测试：水位线、帧唯一归属、跨块长发言切分、两轨重叠、稳定 ID、完整性检查、MVDR 抑制干扰、fallback 原因、all8/ch0 对齐、hash 漂移 fail-closed（在 `py311` 与 `hermes` 均需通过）。
- `tests/test_audio_frontend_continuous_e2e.py` — 连续评分测试：四轨 cpCER、缺轨/多轨/空轨、片段拼接、重复与越界 fail-closed，以及 partial UI-only、匿名 final 进 capture 但记忆为 0、rejected 不持久化的服务边界（`hermes`）。
- `tests/test_audio_frontend_full8.py` + `tests/fixtures/full8/` — full8 驱动/汇总器单测（mock + 脱敏 fixture，不调 ASR、不联网、不读 `reports/`）：路径 resolve 与越界拒绝、gate 两层语义、schema 严格校验、隐私/回放证据、manifest 生命周期、信号语义、dry-run 零落盘。**必须由用户在前台运行**，不要用 WorkBuddy 沙箱（tmp_path 会被沙箱拦截）。
- `tests/test_audio_frontend_full_loop_timed.py` — 外层计时 wrapper 的命令构造 / 退出码契约 / 计时算术单测（不跑 conda/音频）。

## 运行环境

- **空间增强 / 分轨 / 阈值**：隔离 `conda env py311`（纯 NumPy / SciPy / SoundFile，不碰 `hermes`、不上 GPU、不用服务器 LLM）。
- **ASR 评分**：`conda run -n hermes`（复用 SenseVoice，ASR hash 锁定保证可复现）。
- 服务器（10.252.17.5:11438）仅运行 LLM API；本管线不在服务器上跑音频 / GSS。

## 范围边界（重要）

- 本管线是**离线研究验证**：固定 75s 窗口或整场一次性跑，证明"技术上行不行"。它**不等于**眼镜上的连续端到端推理。
- `oracle` 实验（人工标签）证明上限，不可直接推导自动方案收益。
- 自动掩码组件（Sortformer → MVDR → ASR）在 **75s 窗口** CER 门禁验证通过（retention 1.1135、重叠 CER 0.2226、8/8 改善、RTF 0.0077）。
- 连续端到端**单场 full（opt 口径，完整会议）已完成**：`R8001_M8004-full`（26 分钟，1573.85s）→ all8 cpCER 0.2455（ch0 0.4386）、区间 CER（all/non_overlap/overlap）0.2067/0.1583/0.2110、DER 0.1846、重叠帧召回 0.7202；外层 `full_process_rtf` 0.35194（≤1.0 通过，`full-loop-timing.json` 口径，含进程启动/导入/产物写出），内部 stage_sum `full_loop_rtf` 0.3144 仅作参考；隔离回放 0 网络 / 0 记忆 / capture 以 `interrupted` 收尾、audit 无 embedding 无 PCM（见 `reports/p4_continuous_e2e/20260908-R8001-M8004-all8-full-timed-opt/`）。完整 4.21h 自动掩码 CER **仍需 full8 八场聚合**：单场 non-overlap 15.8% 高于 ≤11.5% 目标，只能作诊断，不能宣称正式通过；亦不代表 Android 真机验收。DER/重叠帧召回在 full8 层只按八场加权聚合判定（本单场 DER 0.1846 / recall 0.7202 属单场诊断值）。
- 连续前端的轨道连续性来自 Sortformer 全会议运行的内部 speaker cache；**外部 `push()` 跨调用状态保持未验证**（产物中固定记录 `external_push_state_validated=false`）。
- 严禁把本离线结果称作 Android 真机验收。

## 评测产物

`reports/` 下的跑批产物（json / wav / db）已被 gitignore，不进 git，属一次性产物。
