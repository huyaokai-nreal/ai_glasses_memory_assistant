# 当前阶段与开发路线图

更新时间：2026-07-17。本文只保留当前阶段、优先级、验收标准和下一步，不记录历史开发过程。代码真相以 `ai_glasses_memory_assistant/`、`static/`、`tests/`、`evals/` 为准。

## 当前阶段

当前项目是 Stage 2 可运行原型，核心闭环已经形成：

- Web/语音 demo 入口。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 记忆写入门控、召回、纠错、evidence、删除和 debug/audit。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- `GlassesChatService` 是主 service 调度入口，保留聊天主流程、最终 memory gate、后台 job 生命周期、audit、timeline evidence 和写库编排。
- 纯 helper 按主题放在 `document_helpers.py`、`import_helpers.py`、`memory_job_helpers.py`、`timeline_management_helpers.py`、`source_summary_helpers.py`、`explanation_helpers.py`、`response_timing.py`、`llm_runtime.py`、`report_helpers.py`、`memory_recall_arbitration.py`、`capture_helpers.py`、`conversation_helpers.py` 和 `conversation_candidate_helpers.py`。
- 启发式周报草稿和手动提醒候选检查。
- 统一音频 session、16 kHz PCM、VAD、KWS-only 唤醒、partial/final ASR、ambient capture 和保守声纹；模型和 runtime 都在 `audio_engine/`，`audio_processing.py` 只保留导入兼容。
- speaker-labeled transcript 先经过结构化 extraction plan：用户本人、命名人物和临时 speaker 分别拥有独立 subject；候选继续经过最终 memory gate，非敏感的第三方事实可写入其人物记忆，但不能串入用户本人或其他人物。
- 主 LLM runtime 走 DeepSeek/OpenAI-compatible API，配置解析和 client 创建集中在 `llm_runtime.py`，不保留本地模型默认值或 LLM legacy fallback。
- 包内 `ai_glasses_memory_assistant/README.md` 提供所有 Python 文件职责速查，和 `docs/context/code-map.md` 互补：前者按文件名反查，后者按任务找入口。
- `docs/context/assets/system-overview-pipeline.png` 提供系统总览；统一音频细节由 `frontend-audio-data-flow.mmd/.svg` 维护。

当前不是生产级硬件眼镜 runtime、原生手机 App、always-on audio runtime、主动提醒推送系统、可靠 worker 队列或多租户服务。

单元测试定位为核心保险丝：默认覆盖启动配置、存储、聊天主链路和 fake backend 音频契约；真实模型慢测不进入默认门禁。

私人多人开发骨架包括：`CONTRIBUTING.md` 说明分支协作和提交前验证，`.env.example` 提供本地配置模板，`.github/workflows/ci.yml` 在 GitHub 分支/PR 上运行最小核心门禁。

## 文档规则

`docs/context` 只保留三份当前态开发文档：

- `docs/context/README.md`
- `docs/context/code-map.md`
- `docs/context/system-flow-current.md`

这三份文档只写当前系统怎么启动、怎么改、怎么验证、边界在哪里；不写历史开发过程、迁移流水账、调研过程或复盘材料。

包内文件职责速查放在 `ai_glasses_memory_assistant/README.md`，不计入 `docs/context` 三份文档上限。

`docs/context/assets/` 只保存当前架构图资产：`system-overview-pipeline.png` 是统一总览，其他图片只作为总览节点的细节展开，不新增专题说明文档。

## 结构借鉴边界

参考外部 agent memory 项目时，只借鉴适合当前 Python 本地 demo 的工程边界，不照搬发布型插件仓库结构。

- 暂不把当前包迁到 `src/` 布局。当前 `ai_glasses_memory_assistant/` 已经是正式 Python 包入口，`pyproject.toml`、根目录兼容 `server.py`、测试和文档都围绕这个路径工作；现在整体迁入 `src/` 只会制造大规模 import/启动/打包 diff。
- 可以保留并规范根目录 `scripts/`。只放可重复执行的本地诊断、数据迁移、benchmark 准备、清理扫描等工具；不放一次性补丁、临时 bugfix 流水账、旧调研材料或需要长期阅读的设计说明。`scripts/scan_cleanup_candidates.py` 是当前仓库内的只读清理体检入口，用于分级输出可再生缓存、需人工复核候选和禁止自动清理边界。
- 如果未来确实需要 `src/` 布局，必须先有发布/安装/多包隔离的明确需求，再单独开迁移计划和兼容验证，不混入普通结构治理刀。

## 当前优先级

### 实现完成、真实模型验收待完成：统一音频处理核心（2026-07-16）

- [x] 将 `.external/voice-recording` 的 VAD、KWS、流式/整段 ASR 和声纹能力抽取到主包，不接入外部 `main.py` 或记忆系统。
- [x] 外部源码复用基线记录为提交 `f63fb179466584fbb2f7c131997adec761d4ff85`；`.external/` 整体忽略，不进入主仓库提交。
- [x] 增加浏览器 PCM audio session、统一结构化事件和 service 消费边界；partial 只进 UI，final 才能进入聊天、capture 或声纹录入。
- [x] 收敛旧 blob、实时 PCM 和声纹录入的模型所有权，修复 `.env` 初始化、句首/句尾完整性和主动 session 回收。
- [x] 将网页收敛为单一全天待机按钮和 KWS 两段式唤醒，删除手动唤醒与直接语音提问主路径。
- [ ] 完成 fake backend 门禁后保持“实现完成、真实模型验收待完成”；真实 KWS/Paraformer 和现场麦克风/耳机通过后再标记完成。

1. 继续做工程可读性治理：只在三文档中维护当前态入口、代码地图和系统架构。
2. 公开 benchmark 评测：LongMemEval 数据放入本地数据目录后，先跑小样本 smoke，再看失败样本决定后续适配。
3. 独立化后续修复：优先处理 correction fallback 重复保存、用户偏好 kind 归一化、`sqlite3 readonly database` 后台 job 生命周期问题。
4. 文字主线继续观察真实 audit 缺口；出现新问题时补最小核心测试或 target。
5. 音频方向保持 demo 边界：模型路径就绪后运行真实 KWS、0.512 秒 partial、VAD final 和耳机/HTTPS 专项验收，不把慢测放进默认 CI。
6. 文件清理方向：用 `scripts/scan_cleanup_candidates.py` 先做只读候选分级，再按结果小步清理；不引入一次性补丁目录，不做 `src/` 大迁移。
7. 多人独立长期记忆已形成首个完整功能提交边界；后续按下方未闭环清单继续硬化，不恢复本地中文业务词表。`agent_bridge.py` 暂不做行数型清理，音频模块先保持稳定。

## 多人独立记忆待后续闭环

以下问题已在本次提交前复现或审查确认，因当前 token 额度止损保留到后续独立修复提交；当前提交不得宣称全量验收通过：

- identity fast path、周报和提醒默认只读取用户本人记忆。
- `recall_subject_scope=all` 优先于消息中提到的单个人名。
- 多个同名 provisional speaker 必须返回歧义，不得静默选择第一个。
- [x] 多人 transcript evidence 已按说话人 fragment/chunk 隔离，避免关联包含其他人物内容的整段原文。
- `EventMemoryStore` 共享 SQLite 连接增加完整事务同步，并让各 store 的 `close()` 真正关闭连接。
- import 只有 `subject_name` 时归为 named；迁移只更新不一致记录且不重复 rebuild FTS。
- 低置信度声纹 fail closed，不参与自动人物合并；未接入 runtime 的声纹 centroid 接口需删除或补齐明确策略。
- capture 部分 chunk 缺 speaker label 时不能整批退回 flat import；第三方敏感领域继续改为结构化 policy，不能扩充中文例词表。
- timeline/API/debug/audit 继续验证所有 embedding-like 字段均不会公开。
- 补齐 deferred correction、并发写入、迁移/关闭、部分标签和真实 API 路径回归测试，再运行 9 条 active 多人 strict eval 与桌面/390x844 浏览器验收。

## 暂不做

- 真实硬件常驻收音 runtime。
- 原生手机 App。
- 生产级音频上传/转写服务。
- 主动提醒推送系统。
- 可靠 worker 队列和跨进程调度。
- 多设备同步。
- 完整审计后台。
- 新向量库、Graph、外部索引或另起一套记忆系统。
- 为了看起来更像发布型插件仓库而迁移到 `src/` 布局。
- 在 `scripts/` 下沉淀一次性补丁、历史 bugfix 目录或临时实验脚本。
- 恢复旧专题文档、调研文档、HTML 汇报或历史流水账。

## 推荐阅读顺序

1. `AGENTS.md`
2. `README.md`
3. `CONTRIBUTING.md`
4. `docs/context/README.md`
5. `docs/context/code-map.md`
6. `ai_glasses_memory_assistant/README.md`
7. `docs/context/system-flow-current.md`
8. 本文件

## 验收口径

一刀完成至少满足：

- 只改当前任务相关文件。
- 没有绕过记忆门控、用户隔离或敏感信息边界。
- 相关测试、静态检查或无法验证原因已说明。
- 如改变开发者入口或系统边界，同步更新三份 context 文档。

## 验证基线

文档改动：

```bash
cd /path/to/ai_glasses_memory_assistant
git diff --check
conda run -n hermes python -m pytest tests/test_core_startup.py -q
```

Python 改动：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

需要产品链路验证：

```bash
cd /path/to/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
