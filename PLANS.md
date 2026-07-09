# 当前阶段与开发路线图

更新时间：2026-07-09。本文只保留当前阶段、优先级、验收标准和下一步，不记录历史开发过程。代码真相以 `ai_glasses_memory_assistant/`、`static/`、`tests/`、`evals/` 为准。

## 当前阶段

当前项目是 Stage 2 可运行原型，核心闭环已经形成：

- Web/语音 demo 入口。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 记忆写入门控、召回、纠错、evidence、删除和 debug/audit。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- 文档标题/摘要、文档查询识别、文档上下文拼装等 helper 已从 `agent_bridge.py` 迁到 `document_helpers.py`，未引用的文档叶子薄转发已从 service 中移除；文本/JSON 导入拆分、导入条目分类和候选包装 helper 已迁到 `import_helpers.py`；后台 memory job 的 payload 和阶段说明 helper 已迁到 `memory_job_helpers.py`；capture 摘要和 continuous capture 确认文案 helper 已迁到 `capture_helpers.py`；speaker-labeled transcript 结构解析 helper 已迁到 `conversation_helpers.py`，`GlassesChatService` 只保留仍被 service/tests 使用的 transcript 入口薄转发；`conversation_candidate_helpers.py` 只保留结构 debug 边界，不再用本地中文规则生成多人语义候选；`GlassesChatService` 仍保留最终 memory gate 和 service 调度入口。
- 启发式周报草稿和手动提醒候选检查。
- 音频片段处理入口、本地 ASR v1、基础情绪 metadata 和保守声纹参考；音频 runner 和片段处理实现已从主聊天文件抽到 `audio_processing.py`。
- 主 LLM runtime 已收敛到 DeepSeek/OpenAI-compatible API，不再保留本地模型默认值或 LLM legacy fallback。

当前不是生产级硬件眼镜 runtime、原生手机 App、always-on audio runtime、主动提醒推送系统、可靠 worker 队列或多租户服务。

单元测试已经瘦身为核心保险丝：默认只保留启动配置、存储和聊天主链路三类测试；音频、ambient、speaker、过细策略和历史 eval harness 单测不再作为默认门禁。

私人多人开发骨架已经补齐：`CONTRIBUTING.md` 说明分支协作和提交前验证，`.env.example` 提供本地配置模板，`.github/workflows/ci.yml` 在 GitHub 分支/PR 上运行最小核心门禁。

## 文档规则

`docs/context` 只保留三份当前态开发文档：

- `docs/context/README.md`
- `docs/context/code-map.md`
- `docs/context/system-flow-current.md`

这三份文档只写当前系统怎么启动、怎么改、怎么验证、边界在哪里；不写历史开发过程、迁移流水账、调研过程或复盘材料。

## 结构借鉴边界

参考外部 agent memory 项目时，只借鉴适合当前 Python 本地 demo 的工程边界，不照搬发布型插件仓库结构。

- 暂不把当前包迁到 `src/` 布局。当前 `ai_glasses_memory_assistant/` 已经是正式 Python 包入口，`pyproject.toml`、根目录兼容 `server.py`、测试和文档都围绕这个路径工作；现在整体迁入 `src/` 只会制造大规模 import/启动/打包 diff。
- 可以保留并规范根目录 `scripts/`。只放可重复执行的本地诊断、数据迁移、benchmark 准备、清理扫描等工具；不放一次性补丁、临时 bugfix 流水账、旧调研材料或需要长期阅读的设计说明。`scripts/scan_cleanup_candidates.py` 是当前仓库内的只读清理体检入口，用于分级输出可再生缓存、需人工复核候选和禁止自动清理边界。
- 如果未来确实需要 `src/` 布局，必须先有发布/安装/多包隔离的明确需求，再单独开迁移计划和兼容验证，不混入普通结构治理刀。

## 当前优先级

1. 继续做工程可读性治理：只在三文档中维护当前态入口、代码地图和系统架构。
2. 公开 benchmark 评测：LongMemEval 数据放入本地数据目录后，先跑小样本 smoke，再看失败样本决定后续适配。
3. 独立化后续修复：优先处理 correction fallback 重复保存、用户偏好 kind 归一化、`sqlite3 readonly database` 后台 job 生命周期问题。
4. 文字主线继续观察真实 audit 缺口；出现新问题时补最小核心测试或 target。
5. 音频方向保持 demo 边界：后续由接手同事按新方案重建专项测试，不沿用旧单测堆。
6. 文件清理方向：用 `scripts/scan_cleanup_candidates.py` 先做只读候选分级，再按结果小步清理；不引入一次性补丁目录，不做 `src/` 大迁移。
7. `agent_bridge.py` 后续只优先评估仍和 privacy/memory gate 强绑定的多人 transcript 后续策略是否值得恢复；文档 helper 已迁到 `document_helpers.py` 且未引用的文档叶子薄转发已从 service 中移除，低风险 import helper 已迁到 `import_helpers.py`，memory job payload/stage helper 已迁到 `memory_job_helpers.py`，capture 纯 helper 已迁到 `capture_helpers.py`，speaker-labeled transcript 结构解析 helper 已迁到 `conversation_helpers.py` 且未引用的叶子薄转发已从 service 中移除；`conversation_candidate_helpers.py` 当前只保留结构 debug，不生成语义候选；音频模块先保持稳定，不扩大重构范围。

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
6. `docs/context/system-flow-current.md`
7. 本文件

## 验收口径

一刀完成至少满足：

- 只改当前任务相关文件。
- 没有绕过记忆门控、用户隔离或敏感信息边界。
- 相关测试、静态检查或无法验证原因已说明。
- 如改变开发者入口或系统边界，同步更新三份 context 文档。

## 验证基线

文档改动：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
git diff --check
conda run -n hermes python -m pytest tests/test_core_startup.py -q
```

Python 改动：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m pytest tests -q
```

需要产品链路验证：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
