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

### 评分与门禁
- `score_multichannel_cer.py` — 把增强音轨喂 SenseVoice，按 `segment_id` 映射金标文本，算 all / non_overlap / overlap_exposed CER 与删除/替换/插入；缺/重/未知 segment 与 ASR hash 漂移均 fail-closed，并自动判四项门禁。
- `score_auto_mask_retention.py` — 比较 oracle 与 auto-mask 的 CER 增益保留率（retention），综合 diarization tune/holdout 门禁给出组件级 `passed` / `false`。

### 测试
- `tests/test_audio_frontend_*.py` — 对应每个脚本的单测（`test_audio_frontend_score_diarization.py` 含 retention 字段断言）。`diarize_sortformer` 与 `tune_sortformer_threshold` 的测试为纯函数单测，不下载模型、不联网。

## 运行环境

- **空间增强 / 分轨 / 阈值**：隔离 `conda env py311`（纯 NumPy / SciPy / SoundFile，不碰 `hermes`、不上 GPU、不用服务器 LLM）。
- **ASR 评分**：`conda run -n hermes`（复用 SenseVoice，ASR hash 锁定保证可复现）。
- 服务器（10.252.17.5:11438）仅运行 LLM API；本管线不在服务器上跑音频 / GSS。

## 范围边界（重要）

- 本管线是**离线研究验证**：固定 75s 窗口或整场一次性跑，证明"技术上行不行"。它**不等于**眼镜上的连续端到端推理。
- `oracle` 实验（人工标签）证明上限，不可直接推导自动方案收益。
- 自动掩码组件（Sortformer → MVDR → ASR）的 CER 门禁仅在 **75s 窗口**验证通过（retention 1.115、重叠 CER 0.222、8/8 改善、RTF 0.007）；**完整 4.21h 连续会议**的自动掩码 CER 尚未测量（仅 diarization 在整场上过门禁）。这是当前唯一未补齐的离线组件指标，应在交棒端到端前补齐或显式标记。
- 严禁把本离线结果称作 Android 真机验收。

## 评测产物

`reports/` 下的跑批产物（json / wav / db）已被 gitignore，不进 git，属一次性产物。
