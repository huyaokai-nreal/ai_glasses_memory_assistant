# 当前阶段与开发路线图

更新时间：2026-07-21。本文只保留当前阶段、优先级、验收标准和下一步，不记录历史开发过程。代码真相以 `ai_glasses_memory_assistant/`、`android/`、`static/`、`tests/`、`evals/` 为准。

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

当前不是生产级硬件眼镜 runtime、完成签名和耐久验收的原生手机 App、已验证 24 小时的 always-on audio runtime、主动提醒推送系统、可靠 worker 队列或多租户服务。

Android 本地 demo 已可构建并安装到 XREAL X4000；Android 只新增平台外壳，继续复用同一套 Python 记忆、planner、SQLite、LLM 和 audit 核心。真机已通过 DeepSeek 文字聊天、模型原子安装、五组件 sherpa 自检、短时锁屏收音和停止释放，但还不能据此宣称 24 小时可交付。

当前 Android WIP 已具备 arm64 APK、前台麦克风服务、持久化 final 队列、联网恢复、partial UI、三段原生声纹录入、保守重叠判断、原生 TTS、模型自检、sherpa 1.13.4 五模型包和手动加密诊断导出。诊断包包含脱敏数据库快照、audit、设备/电池/内存和版本状态，不包含 API key、原始 PCM 或声纹向量。尚未完成的发布门禁是：真人三段声纹、真人唤醒/问答/TTS 回声、断网与权限等异常矩阵、签名分发、8/24 小时耐久和多厂商验证。

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

### 实现完成、真机模型验收待完成：Android 离线音频 VAD/ASR 测试（2026-07-27）

- [x] Android 高级设置可选择本地 PCM16 WAV 或 Lark 常见 AAC M4A；M4A 通过 Android 系统解码器输出 PCM16。输入只在内存中下混、重采样为 16 kHz 单声道，单次最长 15 分钟。
- [x] 用户默认使用原音频；可明确启用 `-24 dB` 到 `+24 dB` 的 1 dB 步进增益。页面显示增益前后峰值和削波警告，不生成增益后的音频文件。
- [x] 离线测试直接复用原生 Silero VAD 和 SenseVoice：逐段显示时间、语言与文本，并显示合并转写；不会进入 KWS、声纹、`audio_event.v1`、Python、Timeline、SQLite、audit、memory job 或诊断导出。
- [x] JVM 测试覆盖 PCM16 WAV 解码、共享 PCM 规范化、单/立体声、重采样、截断/不支持/超时长输入和可选增益限幅；`lintDebug` 与 `assembleDebug` 已通过。
- [ ] 在已安装完整模型包的 Android 真机上分别使用原音和增益的 Lark AAC M4A 验证：系统解码、VAD 片段、逐段转写、复制结果、取消，以及 Timeline、记忆、audit 和诊断包均无本次测试内容。

### 实现完成、蓝牙独占真机验收待完成：Android 收音设备路由（2026-07-27）

- [x] 全天收音、唤醒、声纹录入与设置页测试共用 16 kHz、单声道 PCM16、`VOICE_RECOGNITION` 和输入路由策略；不启动新的 Python、Timeline、SQLite、audit 或 memory job 路径。
- [x] 没有蓝牙输入时允许系统/手机收音；恰好一个蓝牙输入时必须通过 `setPreferredDevice()` 和实际 `routedDevice.id` 双重确认，无法路由时停止收音而不回退到手机麦克风；多个蓝牙输入拒绝启动并列出候选设备。
- [x] 设备增删会重建录音器并重新应用策略；前台通知、原生状态和 Debug 显示当前实际设备名称、类型与蓝牙/系统来源，原始 PCM 仍只在内存中处理。
- [x] JVM 单测覆盖无蓝牙系统输入、单蓝牙选择、双蓝牙拒绝、偏好设备后路由不一致、蓝牙竞态接入、无音频帧、声音不足和权限/初始化错误。
- [x] 设置页收音测试固定采样五秒，显示本地在线 ASR 的增量/最终文字；路由正确且有 PCM 时，识别到文字或 ASR 未就绪均不会把硬件收音误判为失败。最多五秒的测试 PCM 只在设置页内存中保留到用户播放一次、重新测试或离开页面，随后清零；音频和转写均不进入 Python、Timeline、SQLite、audit 或诊断包。
- [x] 新增 `android/tools/build_and_install_debug.sh`，自动检查 JDK 17、Android SDK、ADB 设备，构建 Debug APK、覆盖安装并启动；多设备时要求显式 `--serial`，不猜测目标设备。
- [ ] 在至少一台连接蓝牙收音设备的 Android 真机上验证：无蓝牙系统输入、唯一蓝牙强制路由、蓝牙断连后重建、双蓝牙拒绝、路由失败停止服务、全天收音/声纹录入/设置测试三入口一致。三星 S22 与 Insta360 Mic Pro 另需保留三份 `android_acceptance.v2` 报告：仅系统蓝牙、Insta360 App BLE 并发、官方接收器 USB；每份都必须以实际 `input_device_source` 判定，手机麦克风回退即使可听也失败。该测试只证明当前收音链路，不替代真人唤醒、ASR、记忆、回复和 TTS 验收。

### 已完成：Android 测试数据自动拉取（2026-07-22）

- [x] 新增 Mac ADB 命令行工具，自动选择单设备或要求显式 serial，正常停止全天收音，等待音频队列和对应 memory job，再拉取脱敏完整快照。
- [x] Debug WebView 原生桥接只在 app 私有缓存生成一次性 ZIP；Release 构建拒绝调用，路径固定校验，Mac 拉取后优先桥接删除并以受限 `run-as rm` 兜底。
- [x] Mac 目录保留 bundle、解压数据库/audit、采集元数据、Codex handoff 和 `latest.json`；目录已由现有 `android/captures/` 规则排除 Git，不自动删除历史证据。
- [x] 拉取元数据不重复保存 final query 或 partial 文本；诊断包继续移除 API key、raw PCM、声纹/录入样本和 embedding，同时保留问题定位所需文字与门控原因。
- [x] Python 目标测试覆盖设备选择、停止终态、失败、二进制传输、ZIP/路径安全、远端清理和 latest 指针；Android 纯策略测试覆盖 Debug-only 与临时路径规则。
- [x] 已覆盖安装最新 Debug APK 到三星 SM-S9010，并在正在收音状态运行工具：capture 正常停止且未重启，队列归零，三个 SQLite `integrity_check` 与 ZIP 校验通过，手机临时 ZIP 已删除；本次零片段短测正确走无 memory job 分支。

### 实现与浏览器验收完成、真机验收待完成：Android 与移动端界面优化（2026-07-22）

- [x] 手机聊天页顶部收敛为标题和齿轮；原有体验者、声纹、播报、定位、记忆和 Debug 控件复用同一套事件逻辑，在手机端进入全屏应用设置中心。
- [x] 手机记忆管理改为带返回箭头的全屏二级页面；设置、记忆、Debug 和声纹共用前端返回栈，Android 系统返回键优先关闭当前二级界面。
- [x] 声纹录入移除 Hermes 展示，三段样本分别显示不同朗读内容；重新录入从第 1 段开始，不改变 embedding、三样本聚合或 API 协议。
- [x] 全天待机主区域只保留短状态、片段数量和启停按钮；capability、capture、片段 ID、音量和 VAD 等详情移入设置中心与 Debug。
- [x] 原生高级设置页增加可见返回工具栏、中文字段与账户/模型/诊断分组，保存、自检、下载和加密诊断导出行为不变。
- [x] 完成 `390x844`、`412x915` 和 `1280x720` 浏览器截图与交互验收；设置、记忆、声纹和返回交互无重叠或横向溢出，桌面双栏保持不变。
- [ ] XREAL X4000 已覆盖安装 2026-07-22 新构建，并通过设置中心展示、系统返回键回到聊天和主界面布局检查；原生高级设置入口、三段真人声纹切换及全天待机运行时可读性仍待人工验收。

### 实现完成、真机真人验收待完成：Android 记忆召回与环境音频写入门控（2026-07-22）

- [x] 非时间型个人 `specific_fact` 同轮检索相关 profile 和 event；无关 profile 不再阻断 event，明确时间/事件/计划查询仍保持 event-focused。
- [x] `ambient_audio_text` 正常 stop 不再因空 `speaker_label` 退回元数据缺失的旧文本导入；匿名片段继续携带 `audio_event_id`、chunk evidence、speaker、overlap 和 `memory_eligible`。
- [x] 未知/其他/环境说话人、重叠不明、memory ineligible、语义 noise、低置信和语义 backend fallback 均 fail closed；ASR final 仍逐条保留在 Timeline，不持久化 raw PCM。
- [x] recall debug 增加 `cross_kind_recall`，音频 memory job 增加逐片段 `unit_gate_results`，可从 audio event 和 chunk 定位 saved/rejected 原因。
- [x] 新增跨类型事实、无证据/冲突证据、时间查询、Android 未标注/缺元数据说话人、混合片段和语义质量回归测试；最终专项结果 `7 passed`。
- [ ] 当前开发环境缺少 Java 17、Android SDK/ADB，Gradle 构建、X4000 owner 解析、结构化测试记忆软删除和真人验收待在设备工具链可用后完成。
- [ ] 在 XREAL X4000 上重置当前 owner 的结构化测试记忆，重新加入“我买车了 / 我的车停在楼下”，完成人真声纹、唤醒、ASR、召回、回复和 TTS 验收。
- [ ] 现有 3 条 subject recall 失败仍需单独修复：self 默认范围、`all` scope 优先级和歧义 provisional debug；本次不扩大到该既有问题。

### 实现完成、真实模型验收待完成：统一音频处理核心（2026-07-16）

- [x] 将 `.external/voice-recording` 的 VAD、KWS、流式/整段 ASR 和声纹能力抽取到主包，不接入外部 `main.py` 或记忆系统。
- [x] 外部源码复用基线记录为提交 `f63fb179466584fbb2f7c131997adec761d4ff85`；`.external/` 整体忽略，不进入主仓库提交。
- [x] 增加浏览器 PCM audio session、统一结构化事件和 service 消费边界；partial 只进 UI，final 才能进入聊天、capture 或声纹录入。
- [x] 收敛旧 blob、实时 PCM 和声纹录入的模型所有权，修复 `.env` 初始化、句首/句尾完整性和主动 session 回收。
- [x] 将网页收敛为单一全天待机按钮和 KWS 两段式唤醒，删除手动唤醒与直接语音提问主路径。
- [x] 修复同一端口并存 HTTP/HTTPS 时 localhost 和局域网命中不同协议的问题；server 改为独占绑定，并明确本机 HTTP 与局域网 HTTPS 的使用边界。
- [ ] 完成 fake backend 门禁后保持“实现完成、真实模型验收待完成”；真实 KWS/Paraformer 和现场麦克风/耳机通过后再标记完成。

### 代码加固完成、真实模型与设备验收待完成：持续音频输入闭环（2026-07-17）

- [x] P1-1：streaming sequence 响应缓存和 dispatch 幂等缓存共用集中上限；全天 session 长时间 push 不再让 service cache 无限增长，窗口内重复 sequence 仍只消费一次。
- [x] P1-2：final 聊天 dispatch 改为可查询 job，PCM push 不等待完整聊天生成；同 session 串行，重复 sequence 复用同一 job，stop/close/失败均有明确终态。
- [x] P1-3：唤醒回应和正式回答用 tokenized playback 状态同步服务端；合法播放期间不误回收，迟到 finished 不串台，断网漏 finish 仍会超时回收。
- [x] P1-4：service 层原子保证同一用户只有一个 active 音频 session；新 ambient/enrollment 接管旧 session，旧 token 失效且旧 capture 不触发记忆 job，不同用户隔离。
- [x] P2-1：分别计算并展示收音、全天转写、唤醒问答和声纹录入能力；模型缺失时对应入口明确不可用，API 不暴露绝对路径。
- [x] P2-2：声纹录入前 flush 尾包并以 pause reason 停止待机；capture 文本保留但不创建记忆 job，录入完成、取消或失败后只恢复原用户待机。
- [x] P2-3：流式 PCM、旧 blob 和声纹录入请求体使用集中上限；读取前校验 Content-Length，超限返回 413，非法/缺失/伪造长度不会无限读取。
- [x] P2-4：补齐匿名声纹从流式 final 到跨 capture 持久匹配、隔离和删除流程测试；`PRED_SPKxxxx` 默认保持匿名 provisional subject，只有用户显式命名时才合并为 named subject，API/debug/audit 不暴露 embedding。
- [x] 真实验收补丁：energy fallback 只表示“可以收音”，不再把全天转写、唤醒问答或声纹录入标为可用；service 在 takeover/capture 创建前拒绝不可用模式，避免环境噪声假片段和停止队列积压。
- [x] 当前环境可完成的真实模型/浏览器验收与降级修复已完成；fake backend 音频专项 46 passed，全量 110 passed，仍只保留 3 条既有 subject recall 失败。运行库和模型配置已补齐，需真人配合的声学项按下方边界继续保留，不宣称通过。

### 实现与新手机模型验收完成、真人唤醒待完成：Android 连说唤醒与可见问句（2026-07-22）

- [x] 原生音频状态明确区分持续记录、已唤醒、等待问题、正在听问题、问题已发送和唤醒超时；10 秒只约束独立唤醒后的提问等待窗口。
- [x] 同时支持“你好小忆，问题”连说和“你好小忆”->“我在，请说”->问题；连说时不播放确认语，不丢弃关键词后的 PCM。
- [x] 只有 final 问句进入聊天；界面立即显示去掉唤醒词的用户问句，并按 `event_id` 避免前台轮询和后台回复恢复造成重复气泡。
- [x] 保持 Python HTTP API、SQLite、`audio_event.v1`、声纹/隐私门控和 partial UI-only 边界不变。
- [x] Android 单测、lint、assemble、前端静态契约和音频专项通过；全量 Python 仍只有 3 条已记录的 subject recall 既有失败。
- [x] 新构建和 `x4000-sherpa-1.13.4-v2` 已安装到三星 SM-S9010；五项真实模型自检均为 `ok`，KWS 改为 `num_trailing_blanks=0`，score/threshold 不变。
- [ ] 新手机填写 API Key 后，真人分别验收连说、两段式、10 秒超时恢复和 final 问句只显示一次；随后在 XREAL X4000 复验同一组场景。

#### 真实模型与浏览器验收记录（2026-07-20）

已完成：

- 仓库 `.env` 中的 `AI_GLASSES_ASR_MODEL_DIR`、`AI_GLASSES_SPEAKER_MODEL_DIR` 已配置且目录存在；未自动下载模型。SenseVoice 和 Cam++ 均完成一次真实 CPU 加载/推理，Cam++ 返回 192 维 embedding；这只证明模型可运行，不代表真人语音准确率已通过。
- `hermes` 已安装 `silero-vad 6.2.1`、`sherpa-onnx 1.13.4`；Sherpa 中文 KWS 模型和 Paraformer streaming 模型已下载到 `/Users/huyaokai/Documents/model`，并在仓库 `.env` 配置。Sherpa 官方样例可真实命中，`你好小忆` 合成语音可命中自定义关键词；Paraformer 官方样例转写为“欢迎大家来体验达摩院推出的语音识别模型”。
- 浏览器 1280x720 和 390x844 均无横向溢出、关闭抽屉不遮挡聊天、控制台无 warning/error。补齐模型后的真实 capability 中五个后端均为 `ready`，`ambient_transcription_ready`、`speaker_enrollment_ready`、`assistant_query_ready` 和 `assistant_wake_ready` 均为 `true`；localhost 页面实际取得麦克风并完成全天待机 start/stop。
- 局域网 HTTPS 已使用覆盖 `10.2.30.121`、`127.0.0.1`、`localhost` 和本机 `.local` 主机名的 mkcert 证书启动；严格证书校验访问 `https://10.2.30.121:8765` 返回 200。浏览器在该 LAN 地址上实际取得麦克风，进入全天待机并缓存 3 段现场语境，正常停止后控制台无 warning/error。该地址只代表当前 DHCP 租约；长期固定需要网管或路由器做 DHCP reservation，同事设备需信任同一 mkcert 根证书。
- 收紧门禁前已用本机浏览器实际取得麦克风并完成 start/stop。该测试暴露 energy fallback 在约 20 秒环境噪声中产生 11 个假 final、停止等待约 80 秒；最终 `/stop` 为 200、capture 为 `stopped`、长期记忆 0、无 raw embedding 和音频文件。当前补丁已阻止缺 Silero 时再次进入这条不稳定路径。

仍需人工验收：

- 真实耳机回声、长问题、长回答、本人/他人/低置信声纹、疑似重叠、刷新/断网恢复需要佩戴者和第二位说话人现场配合；fake tests 已覆盖状态与副作用门禁，但不能替代真人声学结果。重叠语音仍只称“疑似声纹冲突”，不宣称完整 diarization。
- 本次浏览器停止待机时处理了 4 个现场片段，约 30 秒后正常停止；需要在真人长语音场景继续判断 CPU 推理队列耗时是否可接受，不能仅凭模型样例宣称实时性通过。

已执行的补齐步骤（完成）：

```bash
conda run -n hermes python -m pip install silero-vad sherpa-onnx

export AI_GLASSES_STREAMING_ASR_MODEL_DIR=/Users/huyaokai/Documents/model/paraformer-zh-streaming
export AI_GLASSES_KWS_MODEL_DIR=/Users/huyaokai/Documents/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20
export AI_GLASSES_KWS_KEYWORDS_FILE=/Users/huyaokai/Documents/model/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/keywords_ai_glasses.txt

conda run -n hermes python -m ai_glasses_memory_assistant.server --host 127.0.0.1 --port 8765
curl http://127.0.0.1:8765/api/audio/capabilities
```

依赖安装、模型下载和路径配置已完成；capability 中 VAD/KWS/streaming ASR 均为 `ready`，四个工作流 readiness 均为 `true`。下一步录入佩戴者 3 段声纹，戴耳机依次验证“你好小忆”唤醒回应、10 秒内开始长问题、长回答 TTS、第二人/低置信/疑似重叠拒绝、说话中 stop 尾包、刷新、断网和恢复；最后检查无重复 chat/capture/memory job、无 raw PCM 文件、audit 无 embedding 向量。

### 已完成：全天讨论归档与回顾 V1（2026-07-20）

- [x] 保留最近 6 段作为“刚才”即时上下文；新增可恢复的处理切片、当天话题和每日概览，全天待机未停止时也能回顾早中晚内容。
- [x] ambient final 继续先脱敏写入 Timeline；后台按静音、时长、片段数、跨日、停止或回顾请求封段，不让 PCM push 等待摘要模型。
- [x] 同一天的同一话题允许跨多个时间段合并；每日概览按首次出现时间排序，并保留可核对的 chunk evidence ID。
- [x] 环境音频转写原文默认保留 30 天后物理删除，讨论摘要长期保留；用户可按日期分别删除原文、摘要或两者。
- [x] 增加 discussion recall、按日管理 API、每日回顾侧栏和“查看本次依据”，不改变现有音频 API 或长期记忆门控。
- [x] 验收覆盖 300 段、运行中回顾、跨切片同话题、重启/异常恢复、30 天证据过期、删除范围、隐私、用户隔离和桌面/移动端页面。

## 纯音频 AI 记忆眼镜长期路线（2026-07-20）

### 产品北极星与对标口径

本产品只通过声音理解现实，不增加摄像头或其他视觉输入。目标不是复制某个未公开的专有引擎，而是在公开可观察行为上形成并验证完整闭环：

```text
稳定听见
-> 正确分段和转写
-> 判断谁说了什么
-> 区分临时上下文、原话证据、长期记忆和动作候选
-> 按人、事、时间、任务组织
-> 在正确问题中召回并说明依据
-> 经用户授权后提醒或执行
```

达到“媲美”至少要求：同一组真实音频场景下，能完成自动记录、纪要、待办、长期召回、数据管理和办公动作闭环，并有可复现的质量证据。达到“超越”还要求：

- 每条重要结论能区分原话、摘要、模型推断和不确定信息，并能查看对应 evidence。
- 敏感信息、他人信息、低置信声纹和重叠语音默认 fail closed，不静默归入用户长期记忆。
- 用户纠正后能 supersede 旧结论，既不继续使用错误信息，也保留可解释的变更依据。
- 多人记忆在身份明确前保持隔离；同名、匿名和跨场景合并必须可解释、可确认、可撤销。
- 外部执行不绑定单一办公平台，统一支持预览、授权、幂等、状态查询、失败恢复、审计和撤销。
- 所有“效果更好”必须落到音频、记忆、召回、隐私、动作和长时间运行指标，不能只用演示或功能名称证明。

### 必须保持的机制

- `audio_event.v1` 是音频输入契约；`partial` 永远只进 UI/debug，只有合格 `final` 才能产生证据、记忆或动作副作用。
- 原始 PCM 继续只在有界内存中短暂存在，不默认落盘；脱敏 final 才能进入 Timeline。
- Timeline 原话证据、结构化长期记忆、discussion 派生摘要、动作候选四层分开管理，各有独立保留、删除和解释语义。
- 所有长期记忆仍经过 `MemoryWriteCandidate -> should_write_memory_candidate()`，新音频能力不得绕过门控、用户隔离、subject 隔离或 audit。
- 复用当前 `GlassesChatService`、`EventMemoryStore`、`TimelineStore` 和统一音频引擎，不另起一套 memory runtime、数据库或 HTTP owner。

### R0：完成当前全天讨论归档，建立可信基线

- [ ] 完成当前 discussion slice/topic/day、运行中回顾、30 天原文保留、按日删除和 evidence 展示的测试、文档与浏览器验收。
- [ ] 清除后台 worker/SQLite 生命周期告警，验证服务 close、临时目录、重启恢复、失败重试和并发写入不会留下悬空线程或半完成状态。
- [ ] 修复现有 3 条 subject recall 失败和多人待闭环清单，确保本人、命名人物、临时 speaker、`all` scope 和歧义行为一致。
- [ ] 形成固定回归基线：300 段早中晚内容、跨时段同话题、运行中回顾、跨日、断网、重启、删除和过期均有自动测试。

用户效果：全天待机尚未停止时，用户问“今天上午讨论了什么”，也能得到按时间排序、可查看原话依据的回答，而不是只看到最近 6 段。

### R1：真实声学质量与长时间音频稳定性

- [ ] 建立真实音频验收集，覆盖安静室内、街道、车内、会议室、远近说话、方言口音、耳机回声、TTS 回灌、长问题、抢话和静音。
- [ ] 所有入口明确显示收音中、静音、暂停、断线、降级和处理积压状态；提供一键暂停、私密模式和停止后清理当前未归档内容的用户控制。
- [ ] 分别测量 VAD 漏检/误切、KWS 唤醒率/误唤醒率、ASR CER/WER、句首句尾丢失、final 延迟、队列等待和 CPU/内存；基线与发布阈值写入 eval，不散落到业务 hard code。
- [ ] 增加回声与播放感知、barge-in、网络抖动/乱序/重复/断线续传、背压和过载降级；任何降级都要公开 capability/reason，不能悄悄产生低质量记忆。
- [ ] 验证至少 10 小时连续待机：无未界定内存增长、无重复/丢失 final、无原始 PCM 文件、停止和恢复时间可接受。

用户效果：眼镜播报回答时用户插话，系统能停止播报并听清新问题；网络短暂中断后不会把同一句话记两次。

### R2：多人对话与“谁说了什么”

- [ ] 完成本人声纹真人录入和阈值校准；本人、他人、未知和低置信结果保持明确状态，不能用一个固定阈值假装适配所有人。
- [ ] 引入可评测的 speaker diarization/overlap 能力，但匿名 voice group 仍只是临时身份；跨 capture 合并和实名绑定必须经过用户确认。
- [ ] 同一原话片段只能成为对应 speaker 的证据；重叠或归属不明片段进入待确认区，不自动污染任一人物记忆。
- [ ] 补齐人物改名、合并、拆分、撤销、同名歧义和“这是我说的/不是我说的”纠错流程及 audit。

用户效果：三个人开会后询问“谁承诺周五交方案”，系统能给出说话人和原话；无法判断时明确说“不确定”，而不是随便归给佩戴者。

### R3：记忆策略引擎升级

- [ ] 对每个 final/讨论摘要统一输出处理决策：临时上下文、Timeline 证据、长期记忆候选、动作候选或丢弃，并记录价值、置信度、敏感性、subject、时间、来源和原因。
- [ ] 在现有语义分类基础上增强人物、事件、时间、任务、决定、风险、偏好和关系变化抽取；不要恢复散落中文业务词表或把规则堆成第二个分类器。
- [ ] 增加重要性、重复度、新颖度、时效性和证据强度判断，支持同义合并、冲突并存、过期/衰减、correction、supersede 和 task 状态流转。
- [ ] 明确“记住”“不要记”“忘掉刚才”“以后提醒我”等用户指令的最高优先级，同时保持敏感信息确认和他人隐私边界。
- [ ] 建立记忆写入 precision、漏记率、错误人物率、敏感误存率、重复率和纠错后残留率 eval；敏感误存和未授权人物合并必须为零容忍门禁。

用户效果：“我以前爱喝咖啡，但医生让我戒了”不能留下两个同时有效的偏好；系统应保留变化历史，并在当前回答中使用“现在不喝咖啡”。

### R4：可核对的长期召回与记忆推理

- [ ] 按“刚才、今天、某天、跨日话题、某个人、某项目、某个任务”分别选择短上下文、discussion、Timeline、结构化记忆或文档，不把所有数据塞入 prompt。
- [ ] 回答中区分直接原话、派生摘要、结构化事实和模型推断；证据冲突、过期或缺失时主动降置信，不补写不存在的细节。
- [ ] 支持跨天同话题延续、任务承诺追踪、关系/偏好变化和事件时间线，同时保证查询范围、subject scope 与用户隔离。
- [ ] 运行 LongMemEval 和本项目纯音频场景集，衡量 recall precision/coverage、时间准确率、人物归属、evidence 支撑率、幻觉率和延迟。
- [x] 已完成 LongMemEval Oracle 兼容 runner、公平会话导入和通用 Subject Recall 修复：逐题隔离、按原始 session 保留 user/assistant Timeline 证据、仅让用户原话片段通过生产记忆门控、生产召回、统一 Reader、两字段 JSONL、独立召回字符详情、实时进度，以及 unresolved 名称在明确第一人称语境下回退 self 的安全规则；2026-07-27 已完成 500 题 Oracle 诊断运行，500/500 完成且无导入/运行失败，本地 answer hit 46.8%、recall hit 48.2%、空上下文 32 条；该结果只用于组件根因分析，尚未执行官方 judge。
- [x] LongMemEval 组件归因第一轮（2026-07-28）：建立仅离线使用的 routing/write-type/coverage-ranking/Reader/expected-abstention 分类；用非 benchmark 合成对话验证“按既有偏好给建议”会以 self profile 作为有界背景并保持普通建议零召回。生产链路只新增通用语义契约 `mixed + profile + summary`，debug 标记 `profile_context_for_llm`，不使用数据集字段或特判。30 条 Oracle 偏好切片全部完成，profile route 4->17、无召回空路由 19->11，但本地 answer hit 1/30->0/30、recall hit 3/30->0/30；因此只确认路由修复，不能宣称端到端评测提升，下一步独立审计偏好写入类型和候选覆盖。
- [x] 会话导入偏好类型修复（2026-07-28）：通用历史导入对已通过既有门控、且旧规则恰为 `event:event` 的片段，使用短生命周期统一语义分类做受限类型覆盖；只改 `kind`/`memory_type`，保留 subject、证据、时间、来源和全部既有门控，语义不可用、低置信、非写入、内容不对齐或安全门控不通过时回退旧类型。非 benchmark 回归验证稳定工作偏好、一次性事件、跨用户隔离、敏感信息和失败回退。Oracle 偏好切片仅作诊断：30/30 完成，44 条 active profile、profile route 17->18、非空 profile context 5->13、空上下文 23->10、无召回空路由 11->7，但本地 answer/recall hit 仍为 0/30，平均耗时 22.35s->125.95s。下一步应固定已有 profile 证据检查排序/上下文与 Reader，不扩大召回或新增 benchmark 规则。
- [x] LongMemEval 可恢复逐题诊断（2026-07-28）：runner 现为每题原子保存完整结果/debug、脱敏 audit、脱敏 SQLite 证据、离线阶段分类和受限候选排序追踪；`--resume` 仅在数据集、非密钥配置、选择顺序和源码快照完全一致时跳过已完成题，原两字段 evaluator JSONL 不会重复。`--max-new-cases` 可让自动化每次只执行一条未完成题而不改变同一 manifest。候选追踪只记录已有有界候选的 ID/分数/筛选原因，不改变召回或 Reader 输入；分类器只位于 `evals`，不参与生产请求。下一步运行新的固定快照 30 条偏好诊断，整轮结束后才根据通用合成回归决定是否实施一个生产修复。
- [x] LongMemEval 单次 PreReplyDecision 兼容与完整偏好诊断（2026-07-29）：修正离线分类只读取最终 planner flags，避免把 `skipped_by_planner` debug 字典误判为已召回；仅对高置信 `mixed`、self profile、主 LLM 回复且多个 recall 枚举非法的决策，恢复现有 `recall/profile/summary` 契约，明确 `none` 不覆盖。非 benchmark 回归覆盖饮食限制/午餐建议、工作偏好/会议建议、普通建议零召回、跨用户隔离和有界候选追踪。完整 Oracle 偏好切片 30/30 完成、0 运行失败，但本地 answer hit `1/30 -> 0/30`、recall hit `1/30 -> 1/30`、空上下文 `13 -> 18`、平均 `136.75s -> 137.55s`，且恢复逻辑触发 `0/30`；因此不能宣称改善。校正后为 13 条 routing miss、5 条空路由 coverage loss、12 条证据不足，下一步只审计同一次 `PreReplyDecision` 如何判断隐式个性化建议，禁止新增第二 planner、隐藏召回回退或 benchmark 特判。
- [x] 依据 `longmemeval_oracle_20260727_102934` 的失败样本修复通用复合记忆召回：低置信且只读的召回决策保留为无副作用降级，无关字段枚举错误改写入 warning；时间范围无命中时受限回退到同用户同 subject 文本检索；具体事实补充相关 Timeline 原话；Reader 上下文携带来源与时间标签；Runner 记录 fragment 失败详情且不让不完整历史进入 Reader。会话导入把 observation reflection 延后至全部 fragment 顺序导入成功后再触发，修复同一 SQLite 连接的后台并发写入。新增 60 项相关回归通过（另有 1 条既有身份 subject 隔离失败单独保留）。2026-07-27 重跑 27 个原始失败样本：27 完成、0 个导入错误、0 个空上下文、27 个带时间标签上下文、12 个 Reader 拒答；answer hit 33.33%，recall hit 29.63%。剩余拒答属于证据选择或时间推理质量，未使用 LongMemEval 题名、答案或 session id 特判。

用户效果：用户问“上个月和 Alex 讨论发布时最后决定了什么”，系统能找到对应决定、说明后来是否被修改，并展示相关日期与原话，而不是仅做关键词搜索。

### R5：从记录升级为个人回顾与认知辅助

- [ ] 在全天讨论 V1 上增加可配置的日/周回顾，按事件、决定、待办、风险、未决问题和重要人物组织，不把普通闲聊包装成成果。
- [ ] 自动发现重复承诺、长期未完成任务、相互冲突安排和需要再次确认的信息，但只生成建议，不静默改变任务状态。
- [ ] 用户可以编辑摘要、纠正人物、固定重要记忆、调整保留期限、批量导出和按原文/摘要/长期记忆/动作分别删除。
- [ ] 每条洞察都能回到来源；来源已过期时明确标注“依据已删除”，不能伪装成仍可核验。

用户效果：每天回顾不是一段泛泛总结，而是列出“今天做了什么、决定了什么、答应了谁、还有什么没完成”，每一项都能查看依据。

### R6：经授权的办公执行与主动服务

- [ ] 先定义 provider-neutral action contract，再接日历、任务、文档/PPT、邮件或其他办公适配器；核心记忆系统不直接依赖某一家云服务。
- [ ] 默认流程为“识别动作候选 -> 生成预览 -> 用户确认 -> 幂等执行 -> 查询结果 -> audit/撤销”；只有用户明确配置的低风险规则可自动执行。
- [ ] 任务、日历和文档动作必须保存外部对象 ID、权限范围、执行状态和错误，不允许模型只回复“已经创建”却没有真实副作用证据。
- [ ] 在动作执行稳定后再增加主动提醒：需 opt-in、安静时段、频率上限、延迟/重复抑制、送达状态和关闭入口。
- [ ] 建立未授权动作率、重复执行率、字段准确率、失败恢复率和撤销成功率门禁；未授权外部写入必须零容忍。

用户效果：听到“周五前把方案发给张经理”后，系统先展示待办和时间供确认；确认后创建真实任务，并能回答“任务建在哪里、是否成功、怎么撤销”。

### R7：眼镜运行时与生产可靠性

- [ ] 在不改变 `audio_event.v1` 业务语义的前提下定义硬件/手机音频 adapter，处理设备鉴权、麦克风状态、离线缓冲、网络恢复和版本兼容；不引入视觉接口。
- [ ] 将进程内 job 演进为可恢复的持久 worker，补齐事务并发、数据库 close/migration、崩溃恢复、幂等重放和失败队列。
- [ ] 补齐本地数据加密、密钥生命周期、备份/恢复、配额、物理删除验证、设备丢失处置和多设备同步冲突策略，再决定是否需要新增依赖。
- [ ] 建立音频健康、处理积压、模型耗时、记忆写入、召回和动作执行的脱敏观测指标；日志不得包含密钥、原始 PCM 或 embedding。
- [ ] 通过 10 小时真实待机、30 天数据生命周期、重启/升级/断网/磁盘不足/模型不可用和多设备冲突验收后，才从“Web 原型”升级产品阶段表述。

用户效果：手机或服务重启后，眼镜能恢复待机和未完成任务，不丢记忆、不重复执行，也不会因模型不可用而静默记录错误内容。

### 统一验收与宣称边界

- 每个阶段按独立提交推进：`检查 -> 最小实现 -> 聚焦测试 -> 全量回归 -> 审查 -> 文档/PLANS 更新`，不把多个阶段混成一次大重构。
- 建立固定纯音频验收包，包含音频文件、期望 transcript、speaker、时间、记忆候选、召回答案、隐私负例和动作预期；同一数据用于回归与竞品黑盒对照。
- 能接触竞品真机时，用同一批场景对比，不以发布稿作为准确率证据；无法验证的能力明确标注“公开宣称、未实测”。
- 只有音频、记忆、召回、隐私、动作和长时间运行门禁全部通过，才可使用“达到同类产品完整机制”；只有关键指标稳定领先，才可使用“超越”。

### 推荐实施顺序

1. 先完成 R0 全天讨论归档与现有失败清零。
2. 再完成 R1 真实声学验收和 R2 多人归属，确保输入可信。
3. 之后实施 R3 记忆策略和 R4 长期召回，确保记忆可信。
4. 在此基础上做 R5 回顾洞察，形成音频记忆产品核心体验。
5. 最后开放 R6 外部执行和主动服务，并以 R7 生产可靠性支撑真实眼镜接入。

### 其他并行维护项

1. 继续做工程可读性治理：只在三文档中维护当前态入口、代码地图和系统架构。
2. 公开 benchmark 评测：500 题 Oracle 已完成诊断，下一步先按“planner 路由漏检、写入类型、检索覆盖/排序、Timeline、Reader 证据利用”建立离线失败分类，并逐项用非 benchmark 文本的回归测试验证；再运行无标签泄漏的 V1 Cleaned S 全历史配对评测和官方 judge。详细任务计划位于 `.planning/2026-07-28-longmemeval-memory-foundation-audit/`。
3. 独立化后续修复：优先处理 correction fallback 重复保存、用户偏好 kind 归一化、`sqlite3 readonly database` 后台 job 生命周期问题。
4. 文字主线继续观察真实 audit 缺口；出现新问题时补最小核心测试或 target。
5. 音频方向继续按 R1 补真人 KWS、0.512 秒 partial、VAD final、声纹、耳机回声、长语音和 HTTPS 场景验收；真实模型慢测单独运行，不放进默认 CI。
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

## 当前阶段暂不做

- 摄像头、图片、视频、OCR、人脸识别、视觉场景理解或任何视觉记忆能力；这是长期产品边界，不是等待排期的缺口。
- 正式硬件眼镜 runtime、生产级音频上传/转写服务，以及 Android demo 的公开商店分发；当前 Android 只按内部测试 APK 和本地模型包继续验收。
- 主动提醒推送；只在 R6 动作候选、授权、幂等和撤销闭环通过后实施。
- 可靠 worker 队列、跨进程调度、多设备同步和完整审计后台；这些属于 R7 产品化阶段，不混入当前功能开发。
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
