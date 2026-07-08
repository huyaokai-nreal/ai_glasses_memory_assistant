# 当前阶段与开发路线图

更新时间：2026-07-08。本文只保留当前阶段、优先级、验收标准和下一步，不记录历史开发过程。代码真相以 `ai_glasses_memory_assistant/`、`static/`、`tests/`、`evals/` 为准。

## 当前阶段

当前项目是 Stage 2 可运行原型，核心闭环已经形成：

- Web/语音 demo 入口。
- `/api/chat` 主链路。
- SQLite 结构化记忆和原话 timeline。
- reply-first 后台 memory job。
- 记忆写入门控、召回、纠错、evidence、删除和 debug/audit。
- 文本/JSON 导入、Markdown 文档归档、continuous capture。
- 启发式周报草稿和手动提醒候选检查。
- 音频片段处理入口、本地 ASR v1、基础情绪 metadata 和保守声纹参考。

当前不是生产级硬件眼镜 runtime、原生手机 App、always-on audio runtime、主动提醒推送系统、可靠 worker 队列或多租户服务。

## 文档规则

`docs/context` 只保留三份当前态开发文档：

- `docs/context/README.md`
- `docs/context/code-map.md`
- `docs/context/system-flow-current.md`

这三份文档只写当前系统怎么启动、怎么改、怎么验证、边界在哪里；不写历史开发过程、迁移流水账、调研过程或复盘材料。

## 当前优先级

1. 完成三文档重建后的断链清理和文档一致性测试。
2. 继续做工程可读性治理：只在三文档中维护当前态入口、代码地图和系统架构。
3. 公开 benchmark 评测：LongMemEval 数据放入本地数据目录后，先跑小样本 smoke，再看失败样本决定后续适配。
4. 独立化后续修复：优先处理 correction fallback 重复保存、用户偏好 kind 归一化、`sqlite3 readonly database` 后台 job 生命周期问题。
5. 文字主线继续观察真实 audit 缺口；出现新问题时补最小测试或 target。
6. 音频方向保持 demo 边界：先验证文本级多人会话和门控，再决定是否接更完整音频 runtime。

## 暂不做

- 真实硬件常驻收音 runtime。
- 原生手机 App。
- 生产级音频上传/转写服务。
- 主动提醒推送系统。
- 可靠 worker 队列和跨进程调度。
- 多设备同步。
- 完整审计后台。
- 新向量库、Graph、外部索引或另起一套记忆系统。
- 恢复旧专题文档、调研文档、HTML 汇报或历史流水账。

## 推荐阅读顺序

1. `AGENTS.md`
2. `README.md`
3. `docs/context/README.md`
4. `docs/context/code-map.md`
5. `docs/context/system-flow-current.md`
6. 本文件

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
conda run -n hermes python -m pytest tests/test_standalone_startup_docs.py -q
```

Python 改动：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m py_compile ai_glasses_memory_assistant/*.py ai_glasses_memory_assistant/evals/*.py server.py
conda run -n hermes python -m unittest discover tests -q
```

需要产品链路验证：

```bash
cd /Users/huyaokai/Desktop/workspace/ai_glasses_memory_assistant
conda run -n hermes python -m ai_glasses_memory_assistant.evals.runner --mode live --repeat 3 --strict
```
