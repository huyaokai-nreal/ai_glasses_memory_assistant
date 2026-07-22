const androidNativeBridge = window.AiGlassesAndroid || null;

function isAndroidNative() {
  return Boolean(androidNativeBridge && androidNativeBridge.platform?.() === "android");
}

function callAndroidBridge(method, ...args) {
  if (!isAndroidNative() || typeof androidNativeBridge[method] !== "function") {
    throw new Error("Android 原生桥接不可用");
  }
  const result = androidNativeBridge[method](...args);
  if (typeof result !== "string" || !result.trim()) return result;
  try {
    return JSON.parse(result);
  } catch {
    return result;
  }
}

const state = {
  sessionId: null,
  userId: "",
  location: null,
  voiceEnabled: true,
  listening: false,
  speaker: {
    enrolled: false,
    updatedAt: null,
    speakerModel: "",
    speakerSource: "",
    modalOpen: false,
    sampleCount: 0,
    targetSampleCount: 3,
    calibrationStatus: "not_enrolled",
    speakerProfileVersion: null,
    enrollmentSessionId: "",
    resumeAmbientAfterEnrollment: false,
    resumeAmbientUserId: "",
  },
  ambient: {
    enabled: false,
    wakePending: false,
    captureId: null,
    chunkCount: 0,
    status: "idle",
    lastSegmentId: "",
    lastCapturedAt: 0,
    chunks: [],
    wakeSession: null,
  },
  audio: {
    active: null,
    capabilities: null,
    workletLoaded: false,
    dispatchJobs: new Map(),
    nativePartialSequence: -1,
    nativeFinalQuerySequence: -1,
    nativeQueryEventIds: new Set(),
    nativeEnrollmentCompletionSession: "",
  },
  memory: {
    subjects: [],
    memories: [],
    documents: [],
    activeSubjectId: "all",
    activeView: "long-term",
    discussionDays: [],
  },
};

const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const memoryFormEl = document.querySelector("#memory-form");
const memoryInputEl = document.querySelector("#memory-input");
const memoryKindEl = document.querySelector("#memory-kind");
const memorySubjectEl = document.querySelector("#memory-subject");
const memoryNewSubjectFieldEl = document.querySelector("#memory-new-subject-field");
const memoryNewSubjectNameEl = document.querySelector("#memory-new-subject-name");
const memorySubjectFilterEl = document.querySelector("#memory-subject-filter");
const memoryListEl = document.querySelector("#memory-list");
const refreshMemoryEl = document.querySelector("#refresh-memory");
const memoryCountEl = document.querySelector("#memory-count");
const memoryPaneTitleEl = document.querySelector("#memory-pane-title");
const longTermMemoryViewEl = document.querySelector("#long-term-memory-view");
const dailyDiscussionViewEl = document.querySelector("#daily-discussion-view");
const dailyDiscussionListEl = document.querySelector("#daily-discussion-list");
const memoryViewLongTermEl = document.querySelector("#memory-view-long-term");
const memoryViewDiscussionsEl = document.querySelector("#memory-view-discussions");
const timelineEvidencePanelEl = document.querySelector("#timeline-evidence-panel");
const sendButtonEl = document.querySelector("#send-button");
const toastEl = document.querySelector("#toast");
const memoryPaneEl = document.querySelector("#memory-pane");
const memoryToggleEl = document.querySelector("#memory-toggle");
const memoryCloseEl = document.querySelector("#memory-close");
const memoryBackdropEl = document.querySelector("#memory-backdrop");
const settingsToggleEl = document.querySelector("#settings-toggle");
const settingsCloseEl = document.querySelector("#settings-close");
const settingsBackdropEl = document.querySelector("#settings-backdrop");
const settingsAudioDetailsEl = document.querySelector("#settings-audio-details");
const nativeSettingsButtonEl = document.querySelector("#native-settings-button");
const locationRefreshEl = document.querySelector("#location-refresh");
const locationSettingSummaryEl = document.querySelector("#location-setting-summary");
const debugToggleEl = document.querySelector("#debug-toggle");
const closeDebugEl = document.querySelector("#close-debug");
const debugBackdropEl = document.querySelector("#debug-backdrop");
const debugOutputEl = document.querySelector("#debug-output");
const buttonTooltipEl = document.querySelector("#button-tooltip");
const clearDebugEl = document.querySelector("#clear-debug");
const viewAuditEl = document.querySelector("#view-audit");
const exportAuditEl = document.querySelector("#export-audit");
const voiceStatusEl = document.querySelector("#voice-status");
const voiceToggleEl = document.querySelector("#voice-toggle");
const userSwitchEl = document.querySelector("#user-switch");
const userSelectModalEl = document.querySelector("#user-select-modal");
const userSelectFormEl = document.querySelector("#user-select-form");
const userIdInputEl = document.querySelector("#user-id-input");
const userSelectErrorEl = document.querySelector("#user-select-error");
const ambientModeLabelEl = document.querySelector("#ambient-mode-label");
const ambientStatusEl = document.querySelector("#ambient-status");
const ambientStandbyToggleEl = document.querySelector("#ambient-standby-toggle");
const speakerEnrollButtonEl = document.querySelector("#speaker-enroll-button");
const markdownImportButtonEl = document.querySelector("#markdown-import-button");
const markdownImportInputEl = document.querySelector("#markdown-import-input");
const speakerEnrollModalEl = document.querySelector("#speaker-enroll-modal");
const speakerEnrollSummaryEl = document.querySelector("#speaker-enroll-summary");
const speakerEnrollProgressEl = document.querySelector("#speaker-enroll-progress");
const speakerEnrollStatusEl = document.querySelector("#speaker-enroll-status");
const speakerEnrollPhraseLabelEl = document.querySelector("#speaker-enroll-phrase-label");
const speakerEnrollPhraseEl = document.querySelector("#speaker-enroll-phrase");
const speakerSettingSummaryEl = document.querySelector("#speaker-setting-summary");
const speakerEnrollStartEl = document.querySelector("#speaker-enroll-start");
const speakerEnrollRetryEl = document.querySelector("#speaker-enroll-retry");
const speakerEnrollCancelEl = document.querySelector("#speaker-enroll-cancel");
const speakerEnrollCloseEl = document.querySelector("#speaker-enroll-close");
let typingNode = null;
let mediaStream = null;
let speechAudio = null;
let speechAudioUrl = null;
let speechPlayback = null;
let audioContext = null;
const AMBIENT_RETENTION = {
  maxSegments: 6,
  windowSeconds: 5 * 60,
};
const SPEAKER_ENROLLMENT_PHRASES = [
  "你好小忆，现在开始录入我的声音。",
  "清晨的街道很安静，我准备出门散步。",
  "明天下午三点，我们一起讨论新的计划。",
];
const USER_STORAGE_KEY = "ai-glasses-demo-user-id";
const ALL_SUBJECTS = "all";
const SELF_SUBJECT = "__self__";
const NEW_SUBJECT = "__new__";
const AUDIO_DISPATCH_POLL_INTERVAL_MS = 500;
const AUDIO_DISPATCH_RETRY_INTERVAL_MS = 1500;

function setStatus(text) {
  document.body.dataset.status = text;
}

function showToast(text) {
  toastEl.textContent = text;
  toastEl.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    toastEl.classList.remove("show");
  }, 2400);
}

function purgeSummary(payload) {
  return `彻底删除完成：清理 ${payload.purged_chunk_count || 0} 个原文片段、${payload.purged_parent_count || 0} 个空父记录、${payload.audit_records_removed || 0} 条 audit。`;
}

function timelineDeleteSummary(payload) {
  return `处理 ${payload.requested_count || 0} 个原文片段：软删除 ${payload.deleted_count || 0} 个，彻底删除 ${payload.purged_count || 0} 个，保留 ${payload.retained_count || 0} 个，未找到 ${payload.not_found_count || 0} 个。`;
}

function setVoiceStatus(text, status = "idle") {
  voiceStatusEl.textContent = text;
  voiceStatusEl.dataset.state = status;
}

function updateAmbientStatus() {
  if (!ambientModeLabelEl || !ambientStatusEl) return;
  const { enabled, wakePending, captureId, chunkCount, status, lastSegmentId, lastCapturedAt, wakeSession } = state.ambient;
  const wakePendingLabel = wakeSession && wakeSession.status !== "consumed" ? "正在等待问题" : "未唤醒";
  const kwsCapability = state.audio.capabilities?.components?.kws || {};
  const assistantWakeReady = isAndroidNative()
    ? androidModelsReady(["vad", "kws", "online_asr"])
    : Boolean(state.audio.capabilities?.assistant_query_ready ?? state.audio.capabilities?.assistant_wake_ready);
  const capabilityLabels = audioCapabilityLabels();
  const wakeTimeoutSeconds = Number(state.audio.capabilities?.wake_query_start_timeout_seconds || 10);
  const detectorState = wakePending
    ? "wake_detected_pending_query"
    : enabled && assistantWakeReady
      ? "wake_detector_listening"
      : "wake_query_unavailable";
  const statusLabels = {
    permission_pending: "等待麦克风授权",
    starting: "正在启动本地运行时",
    recording: "Android 后台收音中",
    paused_tts: "播报中，已暂停收音",
    stopping: "正在停止收音",
    listening: "待机监听中",
    speech_detected: "检测到现场语音",
    processing: "正在转写当前片段",
    ready_for_wake_context: "片段已加入现场语境",
    failed: "片段处理失败",
    wake_detector_listening: "唤醒词监听中",
    wake_detected: "已听到你好小忆",
    acknowledging: "已唤醒，正在回应",
    waiting_query: "已唤醒，等待你的问题",
    query_listening: "已唤醒，正在听你的问题",
    query_submitted: "问题已发送，正在生成回复",
    wake_timeout: "唤醒超时，已恢复待机",
  };
  let diagnostics = [];
  document.body.dataset.ambient = enabled ? "active" : "idle";
  if (wakePending) {
    const currentStatus = statusLabels[status] || "已唤醒";
    const remainingSeconds = Math.ceil(Number(state.audio.nativeStatus?.wake_timeout_remaining_ms || 0) / 1000);
    ambientModeLabelEl.textContent = currentStatus;
    ambientStatusEl.textContent = status === "waiting_query"
      ? `请在 ${remainingSeconds || wakeTimeoutSeconds} 秒内说出问题`
      : status === "query_listening"
        ? "正在接收你的问题"
        : "正在确认本次语音";
    diagnostics = [
      ambientStatusEl.textContent,
      ...capabilityLabels,
      `唤醒状态=${detectorState}`,
      `capture=${captureId || "准备中"}`,
    ];
  } else if (enabled) {
    const currentStatus = statusLabels[status] || "收音待机中";
    ambientModeLabelEl.textContent = currentStatus;
    ambientStatusEl.textContent = chunkCount ? `已记录 ${chunkCount} 段现场语境` : "正在等待现场语音";
    diagnostics = [
      `已缓存 ${chunkCount} 段现场语境`,
      `当前状态：${currentStatus}`,
      ...capabilityLabels,
      assistantWakeReady ? `唤醒状态=${detectorState}` : `KWS=${kwsCapability.status || "unavailable"}`,
      `capture=${captureId || "准备中"}`,
      lastSegmentId ? `最近片段=${lastSegmentId}` : "最近片段=暂无",
      lastCapturedAt ? `最近采集=${new Date(lastCapturedAt * 1000).toLocaleTimeString()}` : "最近采集=暂无",
      ...(isAndroidNative() ? nativeAudioDiagnosticLabels() : []),
      wakePendingLabel,
      "原始音频临时处理后即删除",
    ];
  } else {
    ambientModeLabelEl.textContent = "全天待机未开启";
    ambientStatusEl.textContent = chunkCount ? `本页保留 ${chunkCount} 段现场语境` : "原始音频不会持久化";
    diagnostics = [
      ambientStatusEl.textContent,
      wakePendingLabel,
      ...capabilityLabels,
    ];
  }
  if (settingsAudioDetailsEl) settingsAudioDetailsEl.textContent = diagnostics.join("；");
  if (ambientStandbyToggleEl) {
    ambientStandbyToggleEl.textContent = enabled ? "停止全天待机" : "开启全天待机";
    ambientStandbyToggleEl.setAttribute("aria-pressed", String(enabled));
    const capabilities = state.audio.capabilities;
    const canStart = isAndroidNative() || Boolean(
        browserAudioInputReady()
        && capabilities?.audio_input_ready
        && (capabilities.ambient_transcription_ready || capabilities.assistant_query_ready)
      );
    ambientStandbyToggleEl.disabled = !enabled && !canStart;
  }
}

function setSpeakerEnrollStatus(text, status = "idle") {
  if (!speakerEnrollStatusEl) return;
  speakerEnrollStatusEl.textContent = text;
  speakerEnrollStatusEl.dataset.state = status;
}

function updateSpeakerEnrollmentPrompt() {
  const targetCount = Math.max(1, Number(state.speaker.targetSampleCount || SPEAKER_ENROLLMENT_PHRASES.length));
  const continuingEnrollment = Boolean(state.speaker.enrollmentSessionId);
  const completedProfile = state.speaker.enrolled && !continuingEnrollment;
  const sampleIndex = completedProfile ? 0 : Math.min(Number(state.speaker.sampleCount || 0), targetCount - 1);
  const phraseIndex = Math.min(Math.max(sampleIndex, 0), SPEAKER_ENROLLMENT_PHRASES.length - 1);
  const displayIndex = phraseIndex + 1;
  if (speakerEnrollPhraseLabelEl) speakerEnrollPhraseLabelEl.textContent = `第 ${displayIndex} 段朗读内容`;
  if (speakerEnrollPhraseEl) speakerEnrollPhraseEl.textContent = SPEAKER_ENROLLMENT_PHRASES[phraseIndex];
  if (speakerEnrollStartEl) {
    speakerEnrollStartEl.textContent = completedProfile ? "重新录入第 1 段" : `录入第 ${displayIndex} 段`;
  }
}

function updateSpeakerProfileSummary() {
  if (!speakerEnrollSummaryEl || !speakerEnrollButtonEl) return;
  if (speakerEnrollProgressEl) {
    speakerEnrollProgressEl.textContent = `当前进度：${state.speaker.sampleCount || 0} / ${state.speaker.targetSampleCount || 3}`;
  }
  const enrollmentReady = isAndroidNative()
    ? state.audio.nativeStatus?.model_state === "ready"
    : Boolean(state.audio.capabilities?.speaker_enrollment_ready);
  speakerEnrollButtonEl.disabled = !enrollmentReady;
  if (state.speaker.enrolled) {
    const updatedLabel = state.speaker.updatedAt
      ? `最近更新：${new Date(state.speaker.updatedAt * 1000).toLocaleString()}`
      : "已录入";
    const calibratedLabel = state.speaker.calibrationStatus === "calibrated" ? "已完成 3 段校准" : "已录入但未完成校准";
    speakerEnrollSummaryEl.textContent = `当前已录入参考声纹。模型=${state.speaker.speakerModel || "cam++"}；${calibratedLabel}；${updatedLabel}`;
    if (speakerSettingSummaryEl) speakerSettingSummaryEl.textContent = `${calibratedLabel} · ${updatedLabel}`;
    speakerEnrollButtonEl.textContent = "已录入声纹";
  } else if (state.speaker.sampleCount > 0) {
    speakerEnrollSummaryEl.textContent = `当前已有 ${state.speaker.sampleCount} 段待校准样本，还差 ${Math.max(0, (state.speaker.targetSampleCount || 3) - state.speaker.sampleCount)} 段。`;
    if (speakerSettingSummaryEl) speakerSettingSummaryEl.textContent = `已录 ${state.speaker.sampleCount} / ${state.speaker.targetSampleCount || 3} 段`;
    speakerEnrollButtonEl.textContent = "继续录入声纹";
  } else {
    speakerEnrollSummaryEl.textContent = "当前还没有参考声纹。可以先用文字聊天；录入后才会在语音待机里区分用户本人和其他人。";
    if (speakerSettingSummaryEl) speakerSettingSummaryEl.textContent = "尚未录入";
    speakerEnrollButtonEl.textContent = "声纹录入";
  }
  updateSpeakerEnrollmentPrompt();
}

function setSpeakerEnrollModalOpen(open) {
  state.speaker.modalOpen = open;
  document.body.classList.toggle("speaker-open", open);
  speakerEnrollButtonEl?.setAttribute("aria-expanded", String(open));
  if (!speakerEnrollModalEl) return;
  speakerEnrollModalEl.hidden = !open;
  if (open) {
    updateSpeakerProfileSummary();
    setSpeakerEnrollStatus(
      state.speaker.sampleCount > 0
        ? `继续录到第 ${Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3)} 段，完成后才会整体替换旧声纹。`
        : state.speaker.enrolled
          ? "当前已有一份参考声纹，重新录入会在完成 3 段校准后整体覆盖旧样本。"
          : "点击开始录入后，依次完成三段朗读内容；暂时不录也可以直接用文字聊天。",
      "idle",
    );
  }
}

function hideButtonTooltip() {
  if (!buttonTooltipEl) return;
  buttonTooltipEl.hidden = true;
  buttonTooltipEl.textContent = "";
}

function showButtonTooltip(button) {
  if (!buttonTooltipEl) return;
  const text = button?.dataset.tooltip;
  if (!text) return;
  buttonTooltipEl.textContent = text;
  buttonTooltipEl.hidden = false;
  const buttonRect = button.getBoundingClientRect();
  const tooltipRect = buttonTooltipEl.getBoundingClientRect();
  const left = Math.min(Math.max(buttonRect.left + buttonRect.width / 2, tooltipRect.width / 2 + 8), window.innerWidth - tooltipRect.width / 2 - 8);
  const top = Math.max(buttonRect.top - 8, tooltipRect.height + 8);
  buttonTooltipEl.style.left = `${left}px`;
  buttonTooltipEl.style.top = `${top}px`;
}

function setupButtonTooltips() {
  document.addEventListener("pointerover", (event) => {
    const button = event.target.closest("button[data-tooltip]");
    if (!button || button.contains(event.relatedTarget)) return;
    showButtonTooltip(button);
  });
  document.addEventListener("pointerout", (event) => {
    const button = event.target.closest("button[data-tooltip]");
    if (!button || button.contains(event.relatedTarget)) return;
    hideButtonTooltip();
  });
  document.addEventListener("focusin", (event) => {
    showButtonTooltip(event.target.closest("button[data-tooltip]"));
  });
  document.addEventListener("focusout", (event) => {
    if (event.target.closest("button[data-tooltip]")) hideButtonTooltip();
  });
  document.addEventListener("click", hideButtonTooltip);
  window.addEventListener("scroll", hideButtonTooltip, true);
  window.addEventListener("resize", hideButtonTooltip);
}

const MOBILE_LAYOUT = window.matchMedia("(max-width: 900px)");
const UI_SURFACE_HISTORY_KEY = "aiGlassesUiSurface";
let surfaceReturnFocus = null;

function currentUiSurface() {
  return String(window.history.state?.[UI_SURFACE_HISTORY_KEY] || "");
}

function applyUiSurface(surface) {
  const nextSurface = MOBILE_LAYOUT.matches ? surface : "";
  const settingsOpen = nextSurface === "settings";
  const memoryOpen = nextSurface === "memory";
  const debugOpen = nextSurface === "debug";
  const speakerOpen = nextSurface === "speaker";
  document.body.classList.toggle("settings-open", settingsOpen);
  document.body.classList.toggle("memory-open", memoryOpen);
  settingsToggleEl?.setAttribute("aria-expanded", String(settingsOpen));
  memoryToggleEl?.setAttribute("aria-expanded", String(memoryOpen));
  memoryPaneEl.classList.toggle("open", memoryOpen);
  settingsBackdropEl.hidden = !settingsOpen;
  memoryBackdropEl.hidden = !memoryOpen;
  setDebugOpen(debugOpen);
  if (speakerOpen) {
    setSpeakerEnrollModalOpen(true);
  } else if (state.speaker.modalOpen) {
    finishSpeakerEnrollmentFlow(state.audio.active, true).catch((error) => showToast(error.message));
    setSpeakerEnrollModalOpen(false);
  }
  if (!nextSurface && surfaceReturnFocus?.isConnected) {
    surfaceReturnFocus.focus({ preventScroll: true });
    surfaceReturnFocus = null;
  }
}

function openUiSurface(surface, trigger = document.activeElement) {
  if (!MOBILE_LAYOUT.matches) {
    if (surface === "debug") setDebugOpen(true);
    if (surface === "speaker") setSpeakerEnrollModalOpen(true);
    return;
  }
  if (!currentUiSurface() && trigger instanceof HTMLElement) surfaceReturnFocus = trigger;
  const nextState = { ...(window.history.state || {}), [UI_SURFACE_HISTORY_KEY]: surface };
  if (currentUiSurface()) window.history.replaceState(nextState, "");
  else window.history.pushState(nextState, "");
  applyUiSurface(surface);
}

function closeUiSurface(surface) {
  if (surface === "speaker" && state.speaker.modalOpen) {
    finishSpeakerEnrollmentFlow(state.audio.active, true).catch((error) => showToast(error.message));
    setSpeakerEnrollModalOpen(false);
  }
  if (!MOBILE_LAYOUT.matches) {
    if (surface === "debug") setDebugOpen(false);
    if (surface === "speaker") setSpeakerEnrollModalOpen(false);
    return;
  }
  if (currentUiSurface() === surface) window.history.back();
  else applyUiSurface("");
}

window.addEventListener("popstate", (event) => {
  applyUiSurface(String(event.state?.[UI_SURFACE_HISTORY_KEY] || ""));
});

window.aiGlassesHandleBack = () => {
  const surface = currentUiSurface();
  if (surface) {
    closeUiSurface(surface);
    return true;
  }
  if (state.speaker.modalOpen) {
    finishSpeakerEnrollmentFlow(state.audio.active, true).catch((error) => showToast(error.message));
    setSpeakerEnrollModalOpen(false);
    return true;
  }
  if (document.body.classList.contains("debug-open")) {
    setDebugOpen(false);
    return true;
  }
  return false;
};

// Debug 面板只切换前端展示状态，不改变后端 debug/audit 数据结构。
function setDebugOpen(open) {
  document.body.classList.toggle("debug-open", open);
  debugToggleEl.setAttribute("aria-expanded", String(open));
  debugBackdropEl.hidden = !open;
}

function getLocalASRUnsupportedMessage() {
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    return "当前页面不是安全上下文，浏览器通常不会开放麦克风录音；请改用 HTTPS 或 localhost";
  }
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return "当前浏览器不支持麦克风录音，请改用新版 Chrome / Edge / Safari";
  }
  if (!supportsUnifiedAudio()) {
    return "当前浏览器不支持连续音频处理，请改用支持 AudioWorklet 的新版 Chrome / Edge / Safari";
  }
  return "当前环境无法启用本地 ASR 录音，请先检查浏览器权限";
}

function getLocalASRErrorMessage(error) {
  const name = String(error?.name || error?.error || "unknown");
  const messages = {
    NotAllowedError: "麦克风权限被拒绝，请在浏览器地址栏允许麦克风后重试",
    NotFoundError: "没有检测到可用麦克风，请检查系统输入设备",
    NotReadableError: "麦克风当前不可读，可能正被其他程序占用",
    SecurityError: "当前页面安全策略禁止麦克风录音，请改用 HTTPS 或 localhost",
    AbortError: "录音被中断，请稍后重试",
    InvalidStateError: "录音状态异常，请稍后重试",
  };
  return messages[name] || `本地 ASR 录音失败：${name}`;
}

function supportsUnifiedAudio() {
  return Boolean(window.AudioContext && window.AudioWorkletNode);
}

function browserAudioInputReady() {
  if (isAndroidNative()) return true;
  const secure = window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  return Boolean(secure && navigator.mediaDevices?.getUserMedia && supportsUnifiedAudio());
}

function audioCapabilityReason(componentNames) {
  const components = state.audio.capabilities?.components || {};
  const reasonLabels = {
    model_dir_missing: "模型未配置",
    model_dir_not_found: "模型目录不存在",
    model_or_keywords_missing: "模型或关键词未配置",
    funasr_not_installed: "FunASR 未安装",
    sherpa_onnx_not_installed: "Sherpa-ONNX 未安装",
    energy_fallback: "缺少 Silero VAD，仅能做收音诊断",
  };
  for (const name of componentNames) {
    const component = components[name] || {};
    if (component.status === "ready") continue;
    return reasonLabels[component.reason] || component.reason || "后端未就绪";
  }
  return "后端未就绪";
}

function audioCapabilityLabels() {
  const capabilities = state.audio.capabilities;
  if (isAndroidNative()) {
    return [
      "收音=Android 原生",
      androidModelsReady(["vad", "ambient_asr"]) ? "全天转写=可用" : "全天转写=等待本地模型",
      androidModelsReady(["vad", "kws", "online_asr"]) ? "语音唤醒问答=可用" : "语音唤醒问答=等待本地模型",
      androidModelsReady(["vad", "speaker"]) ? "声纹录入=可用" : "声纹录入=等待本地模型",
    ];
  }
  if (!capabilities) return ["收音=检测中", "全天转写=检测中", "语音唤醒问答=检测中", "声纹录入=检测中"];
  const receiveReady = browserAudioInputReady() && Boolean(capabilities.audio_input_ready);
  const ambientReady = Boolean(capabilities.ambient_transcription_ready);
  const queryReady = Boolean(capabilities.assistant_query_ready);
  const enrollmentReady = Boolean(capabilities.speaker_enrollment_ready);
  return [
    receiveReady ? "收音=可用" : `收音=不可用（${getLocalASRUnsupportedMessage()}）`,
    ambientReady ? "全天转写=可用" : `全天转写=不可用（${audioCapabilityReason(["vad", "utterance_asr"])}）`,
    queryReady ? "语音唤醒问答=可用" : `语音唤醒问答=不可用（${audioCapabilityReason(["vad", "kws", "streaming_asr"])}）`,
    enrollmentReady ? "声纹录入=可用" : `声纹录入=不可用（${audioCapabilityReason(["vad", "speaker"])}）`,
  ];
}

function androidModelsReady(names) {
  if (!isAndroidNative()) return false;
  const nativeStatus = state.audio.nativeStatus || {};
  const components = nativeStatus.model_self_test?.components || {};
  return nativeStatus.model_state === "ready" && names.every((name) => components[name]?.state === "ok");
}

function nativeAudioDiagnosticLabels() {
  const status = state.audio.nativeStatus || {};
  const rms = Number(status.audio_rms_dbfs);
  const peak = Number(status.audio_peak_dbfs);
  const level = Number.isFinite(rms) && Number.isFinite(peak)
    ? `音量 RMS=${rms.toFixed(1)} dBFS/峰值=${peak.toFixed(1)} dBFS`
    : "音量=暂无";
  return [
    level,
    `VAD片段=${Number(status.vad_segment_count || 0)}`,
    `ambient final=${Number(status.ambient_final_count || 0)}`,
    `拒绝片段=${Number(status.speech_rejected_count || 0)}`,
  ];
}

async function loadAudioCapabilities() {
  const payload = await requestJSON("/api/audio/capabilities");
  state.audio.capabilities = payload;
  updateAmbientStatus();
  updateSpeakerProfileSummary();
  return payload;
}

async function ensureMicrophoneStream() {
  if (mediaStream) return mediaStream;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(getLocalASRUnsupportedMessage());
  }
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  return mediaStream;
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return window.btoa(binary);
}

async function startUnifiedAudioSession(mode) {
  if (!state.userId) throw new Error("请先选择体验者 ID");
  if (!supportsUnifiedAudio()) return false;
  const capabilities = state.audio.capabilities;
  if (mode === "speaker_enroll" && !capabilities?.speaker_enrollment_ready) {
    throw new Error(`声纹录入不可用：${audioCapabilityReason(["vad", "speaker"])}`);
  }
  if (
    mode === "ambient"
    && !capabilities?.ambient_transcription_ready
    && !capabilities?.assistant_query_ready
  ) {
    throw new Error("全天转写和语音唤醒问答当前都不可用");
  }
  if (state.audio.active) {
    if (state.audio.active.mode === mode) return true;
    await stopUnifiedAudioSession({ interrupted: false });
  }
  const stream = await ensureMicrophoneStream();
  audioContext = audioContext || new window.AudioContext();
  if (!state.audio.workletLoaded) {
    await audioContext.audioWorklet.addModule("/static/audio-worklet.js");
    state.audio.workletLoaded = true;
  }
  await audioContext.resume();
  const sampleIndex = Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3);
  const started = await requestJSON("/api/audio/session/start", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      mode,
      enrollment_session_id: state.speaker.enrollmentSessionId || "",
      sample_index: sampleIndex,
      sample_total: state.speaker.targetSampleCount || 3,
    }),
  });
  const audioFormat = started.format || {};
  const node = new AudioWorkletNode(audioContext, "pcm16-capture", {
    processorOptions: {
      targetSampleRate: Number(audioFormat.sample_rate || 16000),
      chunkSamples: Number(audioFormat.recommended_push_samples || 4096),
    },
  });
  const source = audioContext.createMediaStreamSource(stream);
  const sink = audioContext.createGain();
  sink.gain.value = 0;
  source.connect(node);
  node.connect(sink);
  sink.connect(audioContext.destination);
  const active = {
    userId: state.userId,
    id: started.audio_session_id,
    token: started.session_token,
    mode,
    captureId: started.capture_id || "",
    sequence: 0,
    node,
    source,
    sink,
    pushChain: Promise.resolve(),
    stopping: false,
    stopRequested: false,
    acceptingPcm: true,
    stopPromise: null,
    flushWaiters: new Map(),
    wakeAckInFlight: false,
    playbackId: "",
    playbackSuspended: false,
    timeout: null,
  };
  state.audio.active = active;
  state.listening = true;
  document.body.dataset.voice = "listening";
  if (mode === "ambient") {
    state.ambient.enabled = true;
    state.ambient.captureId = active.captureId;
    state.ambient.status = "listening";
  }
  if (mode === "speaker_enroll") {
    active.timeout = window.setTimeout(() => {
      finishSpeakerEnrollmentFlow(active, false).catch((error) => showToast(error.message));
    }, 10000);
  }
  node.port.onmessage = (event) => {
    const message = event.data;
    if (message?.type === "pcm" && message.buffer instanceof ArrayBuffer) {
      enqueuePcmPush(active, message.buffer);
      return;
    }
    if (message?.type === "flushed") {
      const resolve = active.flushWaiters.get(String(message.requestId || ""));
      if (resolve) {
        active.flushWaiters.delete(String(message.requestId || ""));
        resolve();
      }
    }
  };
  stopSpeaking();
  setVoiceStatus(mode === "ambient" ? "全天待机收音中" : "声纹录入中", "listening");
  updateAmbientStatus();
  return true;
}

function enqueuePcmPush(active, buffer) {
  if (!active || !active.acceptingPcm || active.stopping || state.audio.active !== active) return;
  const sequence = ++active.sequence;
  active.pushChain = active.pushChain
    .then(async () => {
      const payload = await requestJSON("/api/audio/session/push", {
        method: "POST",
        body: JSON.stringify({
          user_id: state.userId,
          audio_session_id: active.id,
          session_token: active.token,
          sequence,
          pcm16_base64: arrayBufferToBase64(buffer),
        }),
      });
      await handleAudioPayload(payload, active);
    })
    .catch((error) => {
      setVoiceStatus(error.message, "error");
      showToast(error.message);
      window.setTimeout(() => {
        const stopPromise = active.mode === "speaker_enroll"
          ? finishSpeakerEnrollmentFlow(active, true)
          : stopUnifiedAudioSession({ interrupted: true, active });
        stopPromise.catch((stopError) => showToast(stopError.message));
      }, 0);
    });
}

async function handleAudioPayload(payload, active) {
  const events = Array.isArray(payload.events) ? payload.events : [];
  const eventById = new Map(events.map((event) => [event.event_id, event]));
  for (const event of events) {
    if (event.type === "speech_start") {
      setVoiceStatus("检测到语音", "listening");
      if (active.mode === "ambient") setAmbientRuntimeStatus("speech_detected");
    } else if (event.type === "wake_detected") {
      state.ambient.wakePending = true;
      state.ambient.wakeSession = {
        ambient_capture_id: active.captureId || "",
        wake_detected_at: Date.now() / 1000,
        wake_mode: "wake_word",
        wake_detector_backend: event.wake?.backend || "server_kws",
        status: "pending_query",
      };
      updateAmbientStatus();
      await acknowledgeWake(active, String(event.wake?.ack_text || ""));
    } else if (event.type === "transcript_partial") {
      setVoiceStatus(event.text ? `识别中：${event.text}` : "正在识别", "listening");
    } else if (event.type === "speech_rejected") {
      setVoiceStatus("未识别到可处理的语音", "idle");
    } else if (event.type === "session_state" && event.vad?.state === "wake_timeout") {
      state.ambient.wakePending = false;
      state.ambient.status = "wake_timeout";
      if (state.ambient.wakeSession) state.ambient.wakeSession.status = "expired";
      setVoiceStatus("等待提问超时，已恢复全天待机", "listening");
      updateAmbientStatus();
    }
  }
  for (const dispatch of Array.isArray(payload.dispatches) ? payload.dispatches : []) {
    const event = eventById.get(dispatch.event_id) || {};
    await handleAudioDispatch(dispatch, event, active);
  }
}

async function handleAudioDispatch(dispatch, event, active) {
  if (dispatch.action === "capture" && dispatch.result) {
    const result = dispatch.result;
    state.ambient.captureId = active.captureId || state.ambient.captureId;
    state.ambient.chunkCount = Number(result.chunk_count || state.ambient.chunkCount + 1);
    state.ambient.lastSegmentId = String(result.chunk_id || "");
    state.ambient.lastCapturedAt = Date.now() / 1000;
    state.ambient.status = "ready_for_wake_context";
    state.ambient.chunks.push({
      chunkId: state.ambient.lastSegmentId,
      text: String(event.text || ""),
      timestamp: Date.now() / 1000,
    });
    pruneAmbientContext();
    updateAmbientStatus();
  } else if (dispatch.action === "chat" && dispatch.job?.job_id) {
    state.ambient.wakePending = false;
    if (state.ambient.wakeSession) state.ambient.wakeSession.status = "dispatching";
    startAudioDispatchPolling(dispatch.job, String(event.text || ""));
  } else if (dispatch.action === "enroll" && dispatch.result) {
    applySpeakerEnrollmentPayload(dispatch.result);
    window.setTimeout(() => finishSpeakerEnrollmentFlow(active, false), 0);
  } else if (dispatch.action === "drop" && event.final) {
    setVoiceStatus("这段语音未进入聊天或长期记忆", "idle");
  }
}

function startAudioDispatchPolling(job, message) {
  const jobId = String(job?.job_id || "");
  if (!jobId || state.audio.dispatchJobs.has(jobId)) return;
  const userId = state.userId;
  state.audio.dispatchJobs.set(jobId, { userId, message });
  if (message) appendMessage("user", message);
  setVoiceStatus("已收到，正在生成回复", "listening");
  showTyping();
  pollAudioDispatchJob(jobId, userId, message).catch((error) => {
    state.audio.dispatchJobs.delete(jobId);
    if (state.userId !== userId) return;
    hideTyping();
    setVoiceStatus(error.message, "error");
    showToast(error.message);
  });
}

async function pollAudioDispatchJob(jobId, userId, message) {
  let retryDelay = AUDIO_DISPATCH_POLL_INTERVAL_MS;
  while (state.audio.dispatchJobs.has(jobId)) {
    await new Promise((resolve) => window.setTimeout(resolve, retryDelay));
    let payload;
    try {
      payload = await requestJSON(
        `/api/audio/dispatch/jobs?user_id=${encodeURIComponent(userId)}&job_id=${encodeURIComponent(jobId)}`,
      );
      retryDelay = AUDIO_DISPATCH_POLL_INTERVAL_MS;
    } catch (error) {
      if (error.status === 404) {
        state.audio.dispatchJobs.delete(jobId);
        if (state.userId === userId) {
          hideTyping();
          setVoiceStatus("语音回答结果已不可查询，请再试一次", "error");
          showToast("语音回答结果已不可查询，请再试一次");
        }
        return null;
      }
      retryDelay = AUDIO_DISPATCH_RETRY_INTERVAL_MS;
      console.warn("Audio dispatch polling failed; retrying.", error);
      continue;
    }
    const current = payload.job || {};
    if (current.status === "pending" || current.status === "running") continue;

    state.audio.dispatchJobs.delete(jobId);
    if (state.userId !== userId) return current;
    if (state.ambient.wakeSession) state.ambient.wakeSession.status = "consumed";
    if (current.status === "completed" && current.result) {
      await applyChatResponse(message, current.result);
      return current;
    }

    hideTyping();
    const statusMessages = {
      failed: "这次语音回答生成失败，请再试一次",
      cancelled: "这次语音回答已取消",
      interrupted: "这次语音回答因服务停止而中断",
    };
    const statusMessage = statusMessages[current.status] || "这次语音回答没有完成";
    setVoiceStatus(statusMessage, "error");
    showToast(statusMessage);
    return current;
  }
  return null;
}

function applySpeakerEnrollmentPayload(payload) {
  state.speaker.enrolled = Boolean(payload.enrolled);
  state.speaker.updatedAt = payload.updated_at || null;
  state.speaker.speakerModel = String(payload.speaker_model || "");
  state.speaker.speakerSource = String(payload.speaker_source || "campp_reference_enrollment");
  state.speaker.sampleCount = Number(payload.sample_count || 0);
  state.speaker.targetSampleCount = Number(payload.target_sample_count || 3);
  state.speaker.calibrationStatus = String(payload.calibration_status || "pending");
  state.speaker.speakerProfileVersion = payload.speaker_profile_version || null;
  state.speaker.enrollmentSessionId = String(payload.enrollment_session_id || state.speaker.enrollmentSessionId || "");
  updateSpeakerProfileSummary();
  setSpeakerEnrollStatus(
    payload.status === "ok" ? "声纹校准完成。" : `第 ${state.speaker.sampleCount} 段已保存，请继续录入。`,
    "success",
  );
}

async function stopUnifiedAudioSession({ interrupted = false, active = state.audio.active, stopReason = "" } = {}) {
  if (!active) return null;
  if (active.stopPromise) return active.stopPromise;
  active.stopPromise = (async () => {
    let stopInterrupted = interrupted;
    active.stopRequested = true;
    window.clearTimeout(active.timeout);
    try {
      active.source.disconnect(active.node);
    } catch {
      // The source may already be disconnected while TTS is playing.
    }
    if (!stopInterrupted) {
      try {
        await flushUnifiedAudio(active);
      } catch (error) {
        stopInterrupted = true;
        console.warn("Audio tail flush failed; stopping as interrupted.", error);
      }
    }
    active.acceptingPcm = false;
    await active.pushChain.catch(() => null);
    active.stopping = true;
    active.node.port.onmessage = null;
    active.node.disconnect();
    active.sink.disconnect();
    let payload = null;
    try {
      payload = await requestJSON("/api/audio/session/stop", {
        method: "POST",
        body: JSON.stringify({
          user_id: active.userId,
          audio_session_id: active.id,
          session_token: active.token,
          interrupted: stopInterrupted,
          stop_reason: stopReason,
        }),
      });
      await handleAudioPayload(payload, active);
    } finally {
      if (state.audio.active === active) state.audio.active = null;
      state.listening = false;
      document.body.dataset.voice = "idle";
      if (active.mode === "ambient") {
        state.ambient.enabled = false;
        state.ambient.wakePending = false;
        state.ambient.status = "idle";
        state.ambient.wakeSession = null;
        updateAmbientStatus();
      }
    }
    return payload;
  })();
  return active.stopPromise;
}

function flushUnifiedAudio(active) {
  const requestId = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  const timeoutMs = Number(state.audio.capabilities?.worklet_flush_timeout_seconds || 2) * 1000;
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      active.flushWaiters.delete(requestId);
      reject(new Error("音频尾包刷新超时"));
    }, timeoutMs);
    active.flushWaiters.set(requestId, () => {
      window.clearTimeout(timer);
      resolve();
    });
    active.node.port.postMessage({ type: "flush", requestId });
  });
}

async function postWakeAcknowledgementFinished(active) {
  const payload = await requestJSON("/api/audio/session/control", {
    method: "POST",
    body: JSON.stringify({
      user_id: active.userId,
      audio_session_id: active.id,
      session_token: active.token,
      action: "wake_ack_finished",
    }),
  });
  await handleAudioPayload(payload, active);
}

async function postAudioPlaybackState(active, action, playbackId) {
  return requestJSON("/api/audio/session/control", {
    method: "POST",
    body: JSON.stringify({
      user_id: active.userId,
      audio_session_id: active.id,
      session_token: active.token,
      action,
      playback_id: playbackId,
    }),
  });
}

async function beginUnifiedAudioPlayback(active = state.audio.active) {
  if (!active || active.stopping || active.stopRequested || state.audio.active !== active) return null;
  const playbackId = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  if (!active.playbackSuspended) {
    try {
      active.source.disconnect(active.node);
      active.playbackSuspended = true;
    } catch {
      active.playbackSuspended = false;
    }
  }
  active.playbackId = playbackId;
  const playback = { active, playbackId };
  try {
    await postAudioPlaybackState(active, "playback_started", playbackId);
    speechPlayback = playback;
    return playback;
  } catch (error) {
    resumeUnifiedAudioAfterPlayback(playback);
    throw error;
  }
}

async function finishUnifiedAudioPlayback(playback) {
  if (!playback?.active || !playback.playbackId) return;
  try {
    await postAudioPlaybackState(playback.active, "playback_finished", playback.playbackId);
  } catch (error) {
    console.warn("Audio playback finish notification failed; server timeout will clean up.", error);
  } finally {
    resumeUnifiedAudioAfterPlayback(playback);
  }
}

async function playBrowserSpeechUntilEnd(text, active) {
  if (!("speechSynthesis" in window)) throw new Error("浏览器不支持语音播报");
  const playback = await beginUnifiedAudioPlayback(active);
  try {
    await new Promise((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "zh-CN";
      utterance.rate = 1.05;
      utterance.addEventListener("end", resolve, { once: true });
      utterance.addEventListener("error", () => reject(new Error("浏览器语音播报失败")), { once: true });
      window.speechSynthesis.speak(utterance);
    });
  } finally {
    await finishUnifiedAudioPlayback(playback);
  }
}

async function acknowledgeWake(active, ackText) {
  const text = normalizeSpeechText(ackText);
  if (!text || active.wakeAckInFlight || active.stopping || state.audio.active !== active) return;
  active.wakeAckInFlight = true;
  stopSpeaking();
  setVoiceStatus(`已唤醒：${text}`, "listening");
  try {
    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`TTS HTTP ${response.status}`);
      const audioBlob = await response.blob();
      if (!audioBlob.size) throw new Error("TTS returned empty audio");
      speechAudioUrl = URL.createObjectURL(audioBlob);
      speechAudio = new Audio(speechAudioUrl);
      const playback = await beginUnifiedAudioPlayback(active);
      try {
        await new Promise((resolve, reject) => {
          speechAudio.addEventListener("ended", resolve, { once: true });
          speechAudio.addEventListener("error", () => reject(new Error("TTS playback failed")), { once: true });
          speechAudio.play().catch(reject);
        });
      } finally {
        clearSpeechAudioResources();
        await finishUnifiedAudioPlayback(playback);
      }
    } catch (error) {
      clearSpeechAudioResources();
      console.warn("Wake acknowledgement TTS unavailable; using browser speech.", error);
      await playBrowserSpeechUntilEnd(text, active);
    }
    if (state.audio.active === active && !active.stopping) {
      await postWakeAcknowledgementFinished(active);
      setVoiceStatus("请开始提问", "listening");
    }
  } catch (error) {
    setVoiceStatus(error.message, "error");
    showToast(error.message);
    window.setTimeout(() => stopUnifiedAudioSession({ interrupted: true, active }), 0);
  } finally {
    active.wakeAckInFlight = false;
  }
}

function stripSpeechMarkup(text) {
  let source = String(text || "").replace(/\r\n?/g, "\n");
  source = source.replace(/```[A-Za-z0-9_+-]*\n?/g, "\n").replace(/```/g, "\n");
  source = source.replace(/!?\[([^\]]*)\]\((?:[^)]+)\)/g, "$1");
  source = source.replace(/\b(?:https?:\/\/|www\.)\S+/gi, "");

  const lines = source
    .split("\n")
    .map((line) =>
      line
        .trim()
        .replace(/^#{1,6}\s*/, "")
        .replace(/^>\s*/, "")
        .replace(/^(?:[-*+]|\d+[.)])\s+/, "")
        .replace(/^\[[ xX]\]\s+/, "")
    )
    .filter(Boolean);

  return lines
    .join("\n")
    .replace(/`([^`\n]+?)`/g, "$1")
    .replace(/(\*\*|__)([^*_]+?)\1/g, "$2")
    .replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1$2")
    .replace(/~~([^~\n]+?)~~/g, "$1")
    .replace(/[*`]/g, "");
}

function normalizeSpeechText(text) {
  return stripSpeechMarkup(text).split(/\s+/).filter(Boolean).join(" ");
}

async function speakWithBrowserFallback(text) {
  const speechText = normalizeSpeechText(text);
  if (!("speechSynthesis" in window) || !speechText) return;
  window.speechSynthesis.cancel();
  const playback = await beginUnifiedAudioPlayback();
  const utterance = new SpeechSynthesisUtterance(speechText);
  utterance.lang = "zh-CN";
  utterance.rate = 1.05;
  utterance.addEventListener("end", () => finishUnifiedAudioPlayback(playback), { once: true });
  utterance.addEventListener("error", () => finishUnifiedAudioPlayback(playback), { once: true });
  window.speechSynthesis.speak(utterance);
}

function clearSpeechAudioResources() {
  if (speechAudio) {
    speechAudio.pause();
    speechAudio.removeAttribute("src");
    speechAudio = null;
  }
  if (speechAudioUrl) {
    URL.revokeObjectURL(speechAudioUrl);
    speechAudioUrl = null;
  }
}

function releaseSpeechAudio(playback = speechPlayback) {
  clearSpeechAudioResources();
  finishUnifiedAudioPlayback(playback).catch((error) => console.warn("Audio playback cleanup failed.", error));
}

function resumeUnifiedAudioAfterPlayback(playback) {
  const active = playback?.active;
  if (!active || active.playbackId !== playback.playbackId) return;
  active.playbackId = "";
  if (speechPlayback?.playbackId === playback.playbackId) speechPlayback = null;
  if (!active.stopRequested && !active.stopping && active.playbackSuspended && state.audio.active === active) {
    active.source.connect(active.node);
  }
  active.playbackSuspended = false;
}

function pruneAmbientContext(nowSeconds = Date.now() / 1000) {
  const cutoff = nowSeconds - AMBIENT_RETENTION.windowSeconds;
  const chunks = Array.isArray(state.ambient.chunks) ? state.ambient.chunks : [];
  const retained = chunks
    .filter((chunk) => Number(chunk.timestamp || 0) >= cutoff)
    .slice(-AMBIENT_RETENTION.maxSegments);
  const retainedIds = new Set(retained.map((chunk) => String(chunk.chunkId || "")).filter(Boolean));
  const expiredCount = Math.max(0, chunks.length - retained.length);
  state.ambient.chunks = retained;
  state.ambient.chunkCount = retained.length;
  state.ambient.lastSegmentId = retained.length ? String(retained[retained.length - 1].chunkId || "") : "";
  state.ambient.lastCapturedAt = retained.length ? Number(retained[retained.length - 1].timestamp || 0) : 0;
  if (state.ambient.wakeSession) {
    const nextPreWakeIds = (state.ambient.wakeSession.pre_wake_segment_ids || []).filter((id) => retainedIds.has(String(id || "")));
    const nextPostWakeIds = (state.ambient.wakeSession.post_wake_query_segment_ids || []).filter((id) => retainedIds.has(String(id || "")));
    const wakeDetectedAt = Number(state.ambient.wakeSession.wake_detected_at || 0);
    const wakeExpired = wakeDetectedAt > 0 && wakeDetectedAt < cutoff;
    state.ambient.wakeSession = {
      ...state.ambient.wakeSession,
      pre_wake_segment_ids: nextPreWakeIds,
      post_wake_query_segment_ids: nextPostWakeIds,
      expired: wakeExpired,
      status: wakeExpired ? "expired" : state.ambient.wakeSession.status,
    };
    if (wakeExpired || (!nextPreWakeIds.length && state.ambient.wakeSession.status !== "consumed")) {
      state.ambient.wakePending = false;
      state.ambient.wakeSession = null;
    }
  }
  if (!retained.length && !state.ambient.enabled && !state.ambient.wakePending) {
    state.ambient.status = "expired";
  }
  return {
    retainedCount: retained.length,
    expiredCount,
    retentionWindowSeconds: AMBIENT_RETENTION.windowSeconds,
    maxRetainedSegments: AMBIENT_RETENTION.maxSegments,
  };
}

function setAmbientRuntimeStatus(status) {
  state.ambient.status = status;
  updateAmbientStatus();
}

async function speak(text) {
  if (!state.voiceEnabled || !text) return;
  const speechText = normalizeSpeechText(text);
  if (!speechText) return;
  stopSpeaking();
  if (isAndroidNative()) {
    callAndroidBridge("speak", speechText);
    return;
  }
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: speechText }),
    });
    if (!res.ok) {
      throw new Error(`TTS HTTP ${res.status}`);
    }
    const audioBlob = await res.blob();
    if (!audioBlob.size) {
      throw new Error("TTS returned empty audio");
    }
    const playback = await beginUnifiedAudioPlayback();
    speechAudioUrl = URL.createObjectURL(audioBlob);
    speechAudio = new Audio(speechAudioUrl);
    speechAudio.addEventListener("ended", () => releaseSpeechAudio(playback), { once: true });
    speechAudio.addEventListener("error", () => releaseSpeechAudio(playback), { once: true });
    await speechAudio.play();
  } catch (error) {
    releaseSpeechAudio();
    console.warn("Backend TTS unavailable; falling back to browser speech.", error);
    try {
      await speakWithBrowserFallback(speechText);
    } catch (fallbackError) {
      console.warn("Browser speech fallback unavailable.", fallbackError);
    }
  }
}

function stopSpeaking() {
  releaseSpeechAudio();
  if (isAndroidNative()) {
    callAndroidBridge("stopSpeaking");
    return;
  }
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

function needsLocation(message) {
  const text = message.toLowerCase();
  return [
    "我在哪",
    "我现在在哪",
    "当前位置",
    "实际位置",
    "定位",
    "附近",
    "周边",
    "迷路",
    "导航",
    "带我去",
    "怎么去",
    "路线",
    "天气",
    "where am i",
    "current location",
    "nearby",
    "near me",
    "lost",
    "navigate",
    "directions",
    "route",
    "weather",
  ].some((trigger) => text.includes(trigger));
}

function locationUnavailable(status, error = "") {
  return {
    status,
    source: "browser_geolocation",
    error,
    timestamp: Date.now() / 1000,
  };
}

function isLocalHttpHost() {
  return location.hostname === "localhost" || location.hostname === "127.0.0.1";
}

function getInsecureLocationMessage() {
  return "局域网 HTTP 页面无法使用浏览器定位，请使用 HTTPS 地址";
}

function locationToastMessage(location) {
  if (location.error === "browser_geolocation_requires_https") {
    return getInsecureLocationMessage();
  }
  const messages = {
    denied: "定位权限被拒绝，本轮不会猜测地点",
    unavailable: "系统定位服务暂时没有可用位置，本轮不会猜测地点",
    timeout: "等待系统定位超时，本轮不会猜测地点",
    unsupported: "当前设备不支持网页定位，本轮不会猜测地点",
  };
  return messages[location.status] || "没有拿到当前位置，本轮不会猜测地点";
}

function getCurrentLocation() {
  // 局域网 IP 的 HTTP 页面不是安全上下文，浏览器会直接拒绝定位。
  if (!window.isSecureContext && !isLocalHttpHost()) {
    return Promise.resolve(locationUnavailable("denied", "browser_geolocation_requires_https"));
  }
  if (!("geolocation" in navigator)) {
    return Promise.resolve(locationUnavailable("unsupported", "browser_geolocation_unavailable"));
  }
  return new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          status: "available",
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          timestamp: position.timestamp / 1000,
          source: "browser_geolocation",
        });
      },
      (error) => {
        const statusByCode = {
          [error.PERMISSION_DENIED]: "denied",
          [error.POSITION_UNAVAILABLE]: "unavailable",
          [error.TIMEOUT]: "timeout",
        };
        const errorByCode = {
          [error.PERMISSION_DENIED]: "geolocation_permission_denied",
          [error.POSITION_UNAVAILABLE]: "geolocation_position_unavailable",
          [error.TIMEOUT]: "geolocation_timeout",
        };
        const status = statusByCode[error.code] || "error";
        resolve(locationUnavailable(status, errorByCode[error.code] || "geolocation_error"));
      },
      {
        enableHighAccuracy: true,
        timeout: 8000,
        maximumAge: 15000,
      },
    );
  });
}

async function refreshLocationForMessage(message) {
  if (!needsLocation(message)) {
    return null;
  }
  locationRefreshEl.disabled = true;
  locationRefreshEl.textContent = "定位中";
  const location = await getCurrentLocation();
  state.location = location;
  if (location.status === "available") {
    locationRefreshEl.textContent = "已定位";
    showToast(`已刷新定位，精度约 ${Math.round(location.accuracy || 0)} 米`);
  } else {
    locationRefreshEl.textContent = "定位";
    showToast(locationToastMessage(location));
  }
  return location;
}

function appendInlineMarkdown(parent, text) {
  const pattern = /(\*\*[^*\n]+?\*\*|__[^_\n]+?__|`[^`\n]+?`|\[[^\]\n]+?\]\((?:https?:\/\/|mailto:)[^) \n]+?\))/g;
  let offset = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > offset) {
      parent.appendChild(document.createTextNode(text.slice(offset, match.index)));
    }
    const token = match[0];
    if (token.startsWith("**") || token.startsWith("__")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.appendChild(strong);
    } else if (token.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.appendChild(code);
    } else {
      const linkMatch = token.match(/^\[([^\]\n]+?)\]\(((?:https?:\/\/|mailto:)[^) \n]+?)\)$/);
      if (linkMatch) {
        const link = document.createElement("a");
        link.href = linkMatch[2];
        link.textContent = linkMatch[1];
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        parent.appendChild(link);
      } else {
        parent.appendChild(document.createTextNode(token));
      }
    }
    offset = match.index + token.length;
  }
  if (offset < text.length) {
    parent.appendChild(document.createTextNode(text.slice(offset)));
  }
}

function appendParagraph(container, lines) {
  if (!lines.length) return;
  const paragraph = document.createElement("p");
  lines.forEach((line, index) => {
    if (index > 0) paragraph.appendChild(document.createElement("br"));
    appendInlineMarkdown(paragraph, line);
  });
  container.appendChild(paragraph);
}

// 助手回复只渲染受支持的 Markdown 结构，原始 HTML 始终按文本处理。
function renderMarkdown(text) {
  const container = document.createDocumentFragment();
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  let paragraphLines = [];
  let currentList = null;
  let codeBlock = null;

  const closeList = () => {
    currentList = null;
  };
  const closeParagraph = () => {
    appendParagraph(container, paragraphLines);
    paragraphLines = [];
  };

  for (const line of lines) {
    const codeFence = line.match(/^```/);
    if (codeFence) {
      if (codeBlock) {
        container.appendChild(codeBlock);
        codeBlock = null;
      } else {
        closeParagraph();
        closeList();
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        pre.appendChild(code);
        codeBlock = pre;
      }
      continue;
    }
    if (codeBlock) {
      const code = codeBlock.querySelector("code");
      code.textContent += `${code.textContent ? "\n" : ""}${line}`;
      continue;
    }
    if (!line.trim()) {
      closeParagraph();
      closeList();
      continue;
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      closeParagraph();
      closeList();
      const level = String(Math.min(headingMatch[1].length + 2, 5));
      const heading = document.createElement(`h${level}`);
      appendInlineMarkdown(heading, headingMatch[2].trim());
      container.appendChild(heading);
      continue;
    }

    const bulletMatch = line.match(/^\s*[-*+]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (bulletMatch || orderedMatch) {
      closeParagraph();
      const listTag = orderedMatch ? "ol" : "ul";
      if (!currentList || currentList.tagName.toLowerCase() !== listTag) {
        currentList = document.createElement(listTag);
        container.appendChild(currentList);
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, (bulletMatch || orderedMatch)[1].trim());
      currentList.appendChild(item);
      continue;
    }

    const quoteMatch = line.match(/^>\s?(.+)$/);
    if (quoteMatch) {
      closeParagraph();
      closeList();
      const quote = document.createElement("blockquote");
      appendInlineMarkdown(quote, quoteMatch[1].trim());
      container.appendChild(quote);
      continue;
    }

    closeList();
    paragraphLines.push(line.trim());
  }

  closeParagraph();
  if (codeBlock) container.appendChild(codeBlock);
  return container;
}

// 聊天气泡是唯一写入消息流的位置，按角色选择纯文本或 Markdown 展示。
function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : role === "assistant" ? "镜" : "!";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown-body");
    bubble.appendChild(renderMarkdown(text));
  } else {
    bubble.textContent = text;
  }
  node.append(avatar, bubble);
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

async function setupLocalASR() {
  await loadAudioCapabilities();
  if (isAndroidNative()) {
    await syncNativeAudioStatus();
    const modelReady = state.audio.nativeStatus?.model_state === "ready";
    const modelState = String(state.audio.nativeStatus?.model_state || "not_installed");
    setVoiceStatus(modelReady ? "Android 本地语音模型已就绪" : `Android 原生麦克风已就绪；本地模型状态：${modelState}`);
    return;
  }
  if (browserAudioInputReady() && state.audio.capabilities?.audio_input_ready) {
    if (!state.audio.capabilities.ambient_transcription_ready && !state.audio.capabilities.assistant_query_ready) {
      setVoiceStatus("麦克风可用，但全天转写和语音唤醒问答模型不可用", "error");
      return;
    }
    setVoiceStatus("统一音频引擎已就绪");
    return;
  }
  setVoiceStatus(getLocalASRUnsupportedMessage(), "error");
}

async function syncNativeAudioStatus() {
  if (!isAndroidNative()) return null;
  const status = callAndroidBridge("audioStatus") || {};
  applyNativeAudioUiStatus(status);
  state.audio.nativeStatus = { ...state.audio.nativeStatus, ...status };
  const enrollmentState = String(status.enrollment_state || "idle");
  if (status.enrollment_session_id) {
    state.speaker.enrollmentSessionId = String(status.enrollment_session_id);
  }
  if (enrollmentState !== "idle" && enrollmentState !== "cancelled") {
    state.speaker.sampleCount = Number(status.enrollment_sample_count || 0);
    state.speaker.targetSampleCount = Number(status.enrollment_sample_total || 3);
  }
  if (enrollmentState === "recording") {
    setSpeakerEnrollStatus(`录音中，请完成第 ${Math.min(state.speaker.sampleCount + 1, state.speaker.targetSampleCount)} 段朗读内容…`, "recording");
  } else if (enrollmentState === "processing") {
    setSpeakerEnrollStatus("三段录音已完成，正在安全保存声纹…", "processing");
  } else if (enrollmentState === "error") {
    setSpeakerEnrollStatus(String(status.last_error || "声纹处理失败，请重试"), "error");
  } else if (enrollmentState === "completed") {
    const completedSession = String(status.enrollment_session_id || "completed");
    setSpeakerEnrollStatus("声纹校准完成。", "completed");
    if (state.audio.nativeEnrollmentCompletionSession !== completedSession) {
      state.audio.nativeEnrollmentCompletionSession = completedSession;
      await loadSpeakerProfile();
    }
  } else if (enrollmentState === "cancelled") {
    setSpeakerEnrollStatus("声纹录入已取消。", "idle");
  }
  if (document.visibilityState === "visible" && state.userId) {
    const completed = callAndroidBridge("consumeCompletedReplies") || [];
    for (const item of completed) {
      const eventId = String(item?.event_id || "");
      showNativeFinalQuery(eventId, String(item?.query || ""));
      const reply = String(item?.reply || "").trim();
      if (!reply) continue;
      appendMessage("assistant", reply);
      speak(reply);
    }
  }
  updateSpeakerProfileSummary();
  updateAmbientStatus();
  return status;
}

function showNativeFinalQuery(eventId, rawQuery) {
  const query = String(rawQuery || "").trim();
  if (!eventId || !query || state.audio.nativeQueryEventIds.has(eventId)) return;
  state.audio.nativeQueryEventIds.add(eventId);
  if (state.audio.nativeQueryEventIds.size > 100) {
    state.audio.nativeQueryEventIds.delete(state.audio.nativeQueryEventIds.values().next().value);
  }
  appendMessage("user", query);
}

function applyNativeAudioUiStatus(status) {
  state.audio.nativeStatus = { ...state.audio.nativeStatus, ...status };
  const activeStates = new Set(["permission_pending", "starting", "recording", "paused_tts", "stopping"]);
  const enabled = activeStates.has(String(status.state || ""));
  const enrollmentState = String(state.audio.nativeStatus?.enrollment_state || "idle");
  const enrollmentActive = new Set(["recording", "processing", "error"]).has(enrollmentState);
  const interactionState = String(status.interaction_state || "ambient_listening");
  state.audio.active = enabled ? { mode: enrollmentActive ? "speaker_enroll" : "ambient", native: true } : null;
  state.ambient.enabled = enabled && !enrollmentActive;
  state.ambient.captureId = String(status.capture_id || "");
  state.ambient.wakePending = new Set(["wake_detected", "acknowledging", "waiting_query", "query_listening"]).has(interactionState);
  state.ambient.status = interactionState === "ambient_listening" ? String(status.state || "idle") : interactionState;
  const ambientContext = status.ambient_context || {};
  if (ambientContext.capture_id === state.ambient.captureId) {
    state.ambient.chunkCount = Number(ambientContext.chunk_count || 0);
    state.ambient.lastSegmentId = String(ambientContext.last_segment_id || ambientContext.last_chunk_id || "");
    state.ambient.lastCapturedAt = Number(ambientContext.last_captured_at || 0);
  }
  const partialSequence = Number(status.partial_sequence || 0);
  if (partialSequence !== state.audio.nativePartialSequence) {
    state.audio.nativePartialSequence = partialSequence;
    const partial = String(status.latest_partial || "").trim();
    if (partial) setVoiceStatus(`${state.ambient.wakePending ? "已唤醒，正在听" : "识别中"}：${partial}`);
    else if (interactionState === "waiting_query") setVoiceStatus("已唤醒，请说出你的问题", "listening");
    else if (state.audio.nativeStatus?.model_state === "ready" && !status.last_error) setVoiceStatus("Android 本地语音待机中");
  }
  const finalQuerySequence = Number(status.final_query_sequence || 0);
  if (finalQuerySequence !== state.audio.nativeFinalQuerySequence) {
    state.audio.nativeFinalQuerySequence = finalQuerySequence;
    showNativeFinalQuery(String(status.final_query_event_id || ""), String(status.final_query || ""));
    if (status.final_query) setVoiceStatus("问题已发送，正在生成回复", "listening");
  }
  if (status.last_error) setVoiceStatus(String(status.last_error), "error");
  updateAmbientStatus();
}

function syncNativeAudioUiStatus() {
  if (!isAndroidNative()) return null;
  const status = callAndroidBridge("audioUiStatus") || {};
  applyNativeAudioUiStatus(status);
  return status;
}

async function startNativeAmbient() {
  const status = callAndroidBridge("startAmbient") || {};
  state.audio.active = { mode: "ambient", native: true };
  state.ambient.enabled = true;
  state.ambient.status = String(status.state || "permission_pending");
  state.ambient.captureId = String(status.capture_id || "");
  updateAmbientStatus();
}

async function stopNativeAmbient() {
  callAndroidBridge("stopAmbient");
  state.ambient.wakePending = false;
  state.ambient.wakeSession = null;
  state.ambient.status = "stopping";
  await syncNativeAudioStatus();
}

function formatSeconds(value) {
  if (typeof value !== "number") return "n/a";
  if (value >= 1) return `${value.toFixed(3)}s`;
  return `${Math.round(value * 1000)}ms`;
}

function formatDebugBool(value) {
  if (value === true) return "是";
  if (value === false) return "否";
  return "未知";
}

function formatDebugValue(value, fallback = "无") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function formatDebugTime(value) {
  if (typeof value !== "number") return "未解析";
  return new Date(value * 1000).toLocaleString();
}

function formatDebugMemory(memory) {
  const kind = formatMemoryKind(memory.kind);
  const when = memory.start_at || memory.occurred_at;
  const timeText = memory.kind === "event" ? ` · 时间：${formatDebugTime(when)}` : "";
  return `- ${kind}: ${memory.content}${timeText}`;
}

function formatMemoryKind(kind) {
  if (kind === "profile") return "长期画像";
  if (kind === "assistant_preference") return "助手偏好";
  return "日常事件";
}

function formatMemoryJobStatus(status) {
  const labels = {
    pending: "等待后台保存",
    running: "后台保存中",
    saved: "已保存",
    skipped: "没有需要保存的长期记忆",
    rejected: "未保存，候选被拒绝",
    failed: "后台保存失败",
  };
  return labels[status] || formatDebugValue(status, "未处理");
}

function formatSourceSummary(summary = {}) {
  const sourceTypes = Array.isArray(summary.source_types) && summary.source_types.length
    ? summary.source_types.join(", ")
    : "无";
  const lines = [
    `- 输入来源：${formatDebugValue(summary.input_source, "chat")}`,
    `- 主依据：${formatDebugValue(summary.primary_source_label, "无可用来源")} (${formatDebugValue(summary.primary_source, "none")})`,
    `- 主依据说明：${formatDebugValue(summary.primary_source_explanation, "当前没有可用来源")}`,
    `- 结构化记忆：${summary.structured_memory_count ?? 0} 条（画像 ${summary.profile_count ?? 0}，事件 ${summary.event_count ?? 0}，observation ${summary.observation_count ?? 0}）`,
    `- Timeline 原文：${summary.timeline_chunk_count ?? 0} 段`,
    `- 文档：${summary.document_count ?? 0} 份`,
    `- 本轮保存：${summary.saved_memory_count ?? 0} 条`,
    `- 待确认 / 拒绝：${summary.pending_confirmation_count ?? 0} / ${summary.rejected_count ?? 0}`,
    `- 来源类型：${sourceTypes}`,
  ];
  const droppedSources = Array.isArray(summary.dropped_sources) ? summary.dropped_sources : [];
  if (droppedSources.length) {
    lines.push("- 被压掉的来源：");
    droppedSources.slice(0, 5).forEach((item) => {
      const source = formatDebugValue(item?.source, "unknown");
      const reason = formatDebugValue(item?.reason, "未记录");
      lines.push(`  - ${source}：${reason}`);
    });
  } else {
    lines.push("- 被压掉的来源：无");
  }
  if (Array.isArray(summary.deleted_or_inactive_source_ids) && summary.deleted_or_inactive_source_ids.length) {
    lines.push(`- 已删除/失活来源：${summary.deleted_or_inactive_source_ids.join(", ")}`);
  }
  return lines;
}

function formatRecallArbitration(arbitration = {}) {
  const decisions = Array.isArray(arbitration.decisions) ? arbitration.decisions : [];
  const lines = [
    `- 主来源：${formatDebugValue(arbitration.primary_source, "none")}`,
    `- 主来源原因：${formatDebugValue(arbitration.primary_source_reason, "未记录")}`,
    `- 召回目标：${formatDebugValue(arbitration.recall_goal, "未记录")}`,
  ];
  if (decisions.length) {
    lines.push("- 仲裁明细：");
    decisions.slice(0, 6).forEach((decision) => {
      lines.push(`  - ${formatDebugValue(decision?.source, "unknown")}：${formatDebugValue(decision?.reason, "未记录")}`);
    });
  } else {
    lines.push("- 仲裁明细：无");
  }
  return lines;
}

function formatRecentContextCapsule(capsule = {}) {
  const injected = capsule.injected_to_main_llm;
  const injectedLabel = injected === true ? "是" : injected === false ? "否" : "未记录";
  return [
    `- 是否可用：${formatDebugBool(capsule.available)}`,
    `- 是否注入主回答：${injectedLabel}`,
    `- 注入原因：${formatDebugValue(capsule.injection_reason, "未记录")}`,
    `- Timeline 片段数：${capsule.timeline_chunk_count ?? 0}`,
    `- 文档片段数：${capsule.document_count ?? 0}`,
  ];
}

function formatMemoryProcessingSummary(memoryProcessing = {}) {
  return [
    `- 写入状态：${formatMemoryJobStatus(memoryProcessing.status)}`,
    `- 当前阶段：${formatDebugValue(memoryProcessing.stage, "未记录")}`,
    `- 阶段原因：${formatDebugValue(memoryProcessing.stage_reason, "未记录")}`,
    `- 阶段解释：${formatDebugValue(memoryProcessing.stage_explanation, "未记录")}`,
    `- 后台任务：${formatDebugValue(memoryProcessing.job_id)}`,
    `- 后台模式：${formatDebugValue(memoryProcessing.mode)}`,
    `- 写入数量：${memoryProcessing.saved_count ?? 0}`,
    `- 拒绝数量：${memoryProcessing.rejected_count ?? 0}`,
    memoryProcessing.error_type ? `- 错误类型：${memoryProcessing.error_type}` : "- 错误类型：无",
  ];
}

function appendMemoryJobDebug(job) {
  const memoryProcessing = job.memory_processing || {};
  const lines = [
    "",
    "后台记忆任务结果",
    `- 任务 ID：${formatDebugValue(job.job_id)}`,
    `- 状态：${formatMemoryJobStatus(job.status)}`,
    `- 阶段：${formatDebugValue(memoryProcessing.stage, "未记录")}`,
    `- 阶段原因：${formatDebugValue(memoryProcessing.stage_reason, "未记录")}`,
    `- 阶段解释：${formatDebugValue(memoryProcessing.stage_explanation, "未记录")}`,
    `- 写入数量：${job.saved_count ?? 0}`,
    `- 拒绝数量：${job.rejected_count ?? 0}`,
  ];
  if (job.rejected_reasons?.length) {
    lines.push(`- 拒绝原因：${job.rejected_reasons.join(", ")}`);
  }
  if (job.error_type) {
    lines.push(`- 错误类型：${job.error_type}`);
  }
  debugOutputEl.textContent = `${debugOutputEl.textContent}\n${lines.join("\n")}`;
}

function appendMarkdownImportDebug(fileName, result) {
  const document = result.document || {};
  const lines = [
    "",
    "Markdown 导入结果",
    `- 文件名：${formatDebugValue(fileName)}`,
    `- 文档 ID：${formatDebugValue(document.id, "无")}`,
    `- 标题：${formatDebugValue(document.title, "无")}`,
    `- 归档文件名：${formatDebugValue(document.filename, "无")}`,
    `- ingestion_id：${formatDebugValue(document.ingestion_id || result.ingestion_id, "无")}`,
    `- 内容哈希：${formatDebugValue(document.content_hash, "无")}`,
    `- 候选数量：${result.candidate_count ?? 0}`,
    `- 保存数量：${result.saved_count ?? 0}`,
    `- 拒绝数量：${result.rejected_count ?? 0}`,
    `- 待确认数量：${result.pending_confirmation_count ?? 0}`,
    "来源摘要：",
    ...formatSourceSummary(result.source_summary || {}),
  ];
  debugOutputEl.textContent = `${debugOutputEl.textContent}\n${lines.join("\n")}`;
}

function isMarkdownFile(file) {
  const name = String(file?.name || "").toLowerCase();
  return name.endsWith(".md");
}

function formatDebugTool(tool) {
  const parts = [
    `- ${formatDebugValue(tool.name, "工具")}`,
    `触发：${formatDebugBool(tool.triggered)}`,
    `可用：${formatDebugBool(tool.available)}`,
  ];
  if (tool.backend) parts.push(`后端：${tool.backend}`);
  if (tool.query) parts.push(`查询：${tool.query}`);
  if (tool.result_count !== undefined) parts.push(`结果数：${tool.result_count}`);
  if (tool.error) parts.push(`错误：${tool.error}`);
  return parts.join(" · ");
}

function formatAgentToolCall(call) {
  const name = formatDebugValue(call.name, "工具");
  const parts = [`- ${name}`];
  if (call.argument_keys?.length) {
    parts.push(`参数键：${call.argument_keys.join(", ")}`);
  }
  return parts.join(" · ");
}

function formatAssistantTimingEndReason(reason) {
  const labels = {
    tool_call_started: "模型返回工具调用",
    run_completed: "模型返回最终回复",
    next_api_call_started: "下一轮开始前结束",
  };
  return labels[reason] || formatDebugValue(reason, "未记录");
}

function formatAssistantBottleneckKind(kind) {
  const labels = {
    llm_api_call: "模型轮次",
    agent_tool_call: "Agent 内部工具",
  };
  return labels[kind] || formatDebugValue(kind, "unknown");
}

function formatAssistantPhase(call) {
  if (call.phase_label) return call.phase_label;
  if (call.phase === "model_after_tool_results") return "工具结果后的模型续写";
  if (call.finish_type === "tool_calls") return "首轮模型判断并请求工具";
  if (call.finish_type === "final_response") return "首轮模型直接生成最终回复";
  return "模型请求";
}

function formatRequestedTools(tools) {
  if (!tools?.length) return "无";
  return tools.map((tool) => formatAgentToolCall(tool).replace(/^- /, "")).join("；");
}

function formatPreviousTools(tools) {
  if (!tools?.length) return "";
  const labels = tools.map((tool) => {
    const parts = [formatDebugValue(tool.name, "工具")];
    if (tool.argument_keys?.length) parts.push(`参数键：${tool.argument_keys.join(", ")}`);
    if (tool.result_chars !== undefined) parts.push(`结果长度：${tool.result_chars} 字符`);
    return parts.join("/");
  });
  return labels.join("；");
}

function formatAssistantApiTiming(call) {
  const parts = [
    `- API #${formatDebugValue(call.index, "?")}：${formatSeconds(call.seconds)}`,
    formatAssistantPhase(call),
    formatAssistantTimingEndReason(call.ended_by),
  ];
  if (call.finish_type) {
    parts.push(call.finish_type === "tool_calls" ? "返回类型：工具调用" : "返回类型：最终回复");
  }
  if (call.requested_tool_count !== undefined) {
    parts.push(`请求工具数：${call.requested_tool_count}`);
  }
  if (call.requested_tools?.length) {
    parts.push(`请求工具：${formatRequestedTools(call.requested_tools)}`);
  }
  if (call.previous_tool_result_count) {
    parts.push(`携带上一轮工具结果：${call.previous_tool_result_count}`);
  }
  const previousTools = formatPreviousTools(call.previous_tools);
  if (previousTools) parts.push(`上一轮工具：${previousTools}`);
  return parts.join(" · ");
}

function formatAssistantToolTiming(call) {
  const parts = [
    `- ${formatDebugValue(call.name, "工具")}：${formatSeconds(call.seconds)}`,
  ];
  if (call.argument_keys?.length) parts.push(`参数键：${call.argument_keys.join(", ")}`);
  if (call.result_chars !== undefined) parts.push(`结果长度：${call.result_chars} 字符`);
  if (call.is_error !== undefined) parts.push(`是否报错：${formatDebugBool(call.is_error)}`);
  return parts.join(" · ");
}

function formatAssistantTiming(debug, timing) {
  const detail = debug.assistant_response_timing;
  const assistantStage = (timing?.stages || []).find((stage) => stage.name === "assistant_response");
  if (!detail) {
    if (!assistantStage) return ["- 本轮没有进入主模型回复链路"];
    return [
      `- 生成助手回复总耗时：${formatSeconds(assistantStage.seconds)}`,
      "- 细分：旧版本 debug 未采集每轮 API/工具耗时",
    ];
  }
  const bottleneck = detail.bottleneck || {};
  const lines = [
    `- 生成助手回复总耗时：${formatSeconds(detail.total_seconds)}`,
    `- 模型轮次合计：${formatSeconds(detail.llm_wait_seconds)}`,
    `- Agent 内部工具合计：${formatSeconds(detail.agent_tool_seconds)}`,
    `- Loop/调度等其他开销：${formatSeconds(detail.loop_overhead_seconds)}`,
    bottleneck.seconds !== undefined
      ? `- 最大瓶颈：${formatDebugValue(bottleneck.name, "未知")}（${formatAssistantBottleneckKind(bottleneck.kind)}）${formatSeconds(bottleneck.seconds)}`
      : "- 最大瓶颈：未记录",
    "模型逐轮耗时：",
    ...(detail.api_calls?.length ? detail.api_calls.map(formatAssistantApiTiming) : ["- 无"]),
    "Agent 内部工具耗时：",
    ...(detail.agent_tool_calls?.length ? detail.agent_tool_calls.map(formatAssistantToolTiming) : ["- 无"]),
  ];
  if (detail.note) lines.push(`- 说明：${detail.note}`);
  return lines;
}

function formatDebugStep(step) {
  const labels = {
    created_new_chat_session: "创建新的对话会话",
    reused_chat_session: "复用已有对话会话",
    classified_turn_intent: "判断是否需要联网和写入记忆",
    resolved_query_temporal_scope: "解析用户问题中的时间范围",
    loaded_profile_and_event_memory: "读取长期画像和日常事件记忆",
    llm_completed: "模型完成回复",
    saved_memory_candidates: "保存通过校验的记忆候选",
    fast_path_greeting: "问候语快速回复",
    fast_path_identity_query: "身份查询快速读取画像",
    fast_path_profile_saved: "用户画像快速写入",
    fast_path_event_acknowledged: "日常事件快速确认，后台处理写入",
  };
  return labels[step] || step;
}

function formatDebugStageName(name) {
  const labels = {
    profile_identity_lookup: "读取身份画像",
    policy_profile_write: "快速写入用户画像",
    session_setup: "准备对话会话",
    intent_classification: "旧版意图判断",
    memory_extraction: "抽取记忆候选",
    query_temporal_resolution: "解析问题时间范围",
    memory_retrieval: "读取画像和事件记忆",
    web_context: "获取联网上下文",
    answer_directive: "规划回答组织",
    assistant_response: "生成助手回复",
    memory_write: "写入本轮记忆",
    memory_snapshot: "生成记忆快照",
    audit_write: "写入审计记录",
  };
  return labels[name] || name;
}

function summarizeSourceSummary(sourceSummary) {
  const parts = [];
  if (sourceSummary.profile_used) parts.push("长期画像");
  if (sourceSummary.event_used) parts.push("事件记忆");
  if (sourceSummary.timeline_used) parts.push("原始时间线");
  if (sourceSummary.document_used) parts.push("文档内容");
  if (sourceSummary.web_used) parts.push("联网结果");
  if (sourceSummary.location_used) parts.push("实时位置");
  if (sourceSummary.general_knowledge_used) parts.push("通用知识");
  return parts.length ? parts.join("、") : "当前轮直接推断";
}

function formatTextCleaningSummary(debug) {
  const textCleaning = debug.text_cleaning || {};
  const summary = textCleaning.summary || {};
  const extractionTrace = debug.memory_processing?.extraction_trace || {};
  const semanticCleaning = extractionTrace.semantic_cleaning || {};
  const hasCleaning =
    Object.keys(textCleaning).length > 0 || Object.keys(extractionTrace).length > 0;
  if (!hasCleaning) return "这轮没有暴露文本清洗摘要。";

  const parts = [];
  const segmentCount = summary.segment_count;
  if (segmentCount !== undefined) {
    parts.push(`先清洗成 ${segmentCount} 段`);
  }
  if (textCleaning.redacted) {
    const categories = textCleaning.redaction_categories?.length
      ? `，命中 ${textCleaning.redaction_categories.join("、")} 脱敏`
      : "";
    parts.push(`发现敏感内容并已脱敏${categories}`);
  } else {
    parts.push("没有命中敏感脱敏");
  }
  if ((summary.noise_marker_count ?? 0) > 0) {
    parts.push(`识别到 ${summary.noise_marker_count} 个噪声标记`);
  }
  if ((summary.correction_marker_count ?? 0) > 0) {
    parts.push(`识别到 ${summary.correction_marker_count} 个纠错标记`);
  }
  if ((summary.negation_marker_count ?? 0) > 0) {
    parts.push(`识别到 ${summary.negation_marker_count} 个否定标记`);
  }
  if (extractionTrace.segment_count !== undefined) {
    parts.push(`长输入抽取阶段查看了 ${extractionTrace.segment_count} 段`);
  }
  if (semanticCleaning.extractable_segment_count !== undefined) {
    parts.push(`其中 ${semanticCleaning.extractable_segment_count} 段被判定为值得继续抽取`);
  }
  if (semanticCleaning.skipped_segment_count !== undefined && semanticCleaning.skipped_segment_count > 0) {
    parts.push(`${semanticCleaning.skipped_segment_count} 段被暂时跳过`);
  }
  return parts.length ? `文本清洗方面：${parts.join("，")}。` : "这轮没有明显的文本清洗信号。";
}

function formatConversationSummary(debug, context) {
  const {
    planner,
    routing,
    memory,
    memoryProcessing,
    answerDirective,
    intent,
    location,
    timing,
  } = context;
  const sourceSummary = debug.source_summary || {};
  const routeMode = debug.fast_path ? "快速路径" : "完整对话路径";
  const replyMode = formatDebugValue(planner.fast_path_kind || planner.reply_mode, "未分类");
  const routeReason = formatDebugValue(planner.reason, "未记录");
  const evidenceSummary = summarizeSourceSummary(sourceSummary);
  const recalledCount = (memory.profile_count ?? 0) + (memory.event_recall_count ?? 0);
  const writeStatus = formatDebugValue(memoryProcessing.status, "未处理");
  const savedCount = memoryProcessing.saved_count ?? 0;
  const rejectedCount = memoryProcessing.rejected_count ?? 0;
  const organization = formatDebugValue(answerDirective.organization, "未记录");
  const textCleaningClause = formatTextCleaningSummary(debug);
  const webClause = intent.needs_web_search
    ? `这轮还额外查了网，查询词是“${formatDebugValue(intent.web_query, "未记录")}”`
    : "这轮没有额外联网";
  const locationClause = location.needed
    ? `位置能力${formatDebugValue(location.status, "未知") === "available" ? "已参与判断" : `需要但状态是${formatDebugValue(location.status, "未知")}`}`
    : "位置能力没有参与";
  const timingClause = timing?.total_seconds !== undefined
    ? `整轮耗时 ${formatSeconds(timing.total_seconds)}`
    : "总耗时未记录";

  return [
    `这轮对话可以先理解成：用户在问“${formatDebugValue(debug.query, "未记录")}”，系统判定走${routeMode}，当前轮次类型是“${replyMode}”，主要原因是“${routeReason}”。`,
    `回答组织方式偏向“${organization}”，主要依据来自${evidenceSummary}；本轮共读到 ${recalledCount} 条相关记忆。${webClause}，${locationClause}。`,
    textCleaningClause,
    `如果你想快速判断结果是否靠谱，先看这句：系统最终给出的记忆写入状态是“${writeStatus}”，本轮已保存 ${savedCount} 条、拒绝 ${rejectedCount} 条。${timingClause}。`,
  ].join("");
}

function renderDebug(debug) {
  const timing = debug?.timing;
  if (!timing) {
    debugOutputEl.textContent = JSON.stringify(debug || {}, null, 2);
    return;
  }
  const planner = debug.planner || {};
  const routing = debug.routing || {};
  const intent = debug.intent || {};
  const temporal = debug.temporal?.query || {};
  const memory = debug.memory || {};
  const memoryProcessing = debug.memory_processing || {};
  const answerDirective = debug.answer_directive || {};
  const location = debug.location || {};
  const llm = debug.llm || {};
  const runtime = debug.runtime || {};
  const recalledMemories = [
    ...(memory.assistant_memories || []),
    ...(memory.profile_memories || []),
    ...(memory.event_memories || []),
  ];
  const tools = debug.tools || [];
  const agentToolCalls = debug.agent_tool_calls || [];
  const skippedStages = debug.skipped_stages || [];
  const conversationSummary = formatConversationSummary(debug, {
    planner,
    routing,
    memory,
    memoryProcessing,
    answerDirective,
    intent,
    location,
    timing,
  });
  const lines = [
    "一段话看懂这次对话",
    conversationSummary,
    "",
    "本次 query 可以读出的信息",
    `- 用户问题：${formatDebugValue(debug.query)}`,
    `- 路由模式：${formatDebugValue(routing.mode, "llm_first")}`,
    `- Planner 角色：llm_first baseline / fast path`,
    `- PreReplyDecision 接管：${formatDebugBool(routing.pre_reply_decision_applied)}`,
    `- 执行路径：${debug.fast_path ? "快速路径，不调用完整主模型流程" : "完整对话路径"}`,
    `- 轮次类型：${formatDebugValue(planner.fast_path_kind || planner.reply_mode, "未分类")}`,
    `- 规划原因：${formatDebugValue(planner.reason)}`,
    `- 跳过阶段：${skippedStages.length ? skippedStages.join(", ") : "无"}`,
    "",
    "本次回答依据",
    ...formatSourceSummary(debug.source_summary || {}),
    "",
    "来源仲裁",
    ...formatRecallArbitration(debug.memory?.recall_arbitration || {}),
    "",
    "最近上下文注入",
    ...formatRecentContextCapsule(debug.recent_context_capsule || {}),
    "",
    "记忆读取",
    `- 长期画像读取数量：${memory.profile_count ?? 0}`,
    `- 日常事件召回数量：${memory.event_recall_count ?? 0}`,
    `- 事件召回策略：${formatDebugValue(memory.event_recall?.strategy, "未记录")}`,
    recalledMemories.length ? "召回内容：" : "召回内容：无",
    ...recalledMemories.map(formatDebugMemory),
    "",
    "回答组织",
    `- 规划后端：${formatDebugValue(answerDirective.backend, "未使用")}`,
    `- 回答意图：${formatDebugValue(answerDirective.answer_intent, "未记录")}`,
    `- 组织方式：${formatDebugValue(answerDirective.organization, "未记录")}`,
    `- 证据策略：${formatDebugValue(answerDirective.evidence_policy, "未记录")}`,
    `- 不确定性策略：${formatDebugValue(answerDirective.uncertainty_policy, "未记录")}`,
    `- 过滤规则：${answerDirective.filtering_rules?.length ? answerDirective.filtering_rules.join("；") : "无"}`,
    "",
    "记忆写入",
    ...formatMemoryProcessingSummary(memoryProcessing),
    `- 候选数量：${memory.extraction?.candidate_count ?? intent.memory_write_count ?? 0}`,
    `- 判断后端：${formatDebugValue(memory.extraction?.backend || intent.backend, "未记录")}`,
    "",
    "时间理解",
    `- 是否识别到时间表达：${formatDebugBool(temporal.has_temporal_expression)}`,
    `- 原始时间词：${formatDebugValue(temporal.temporal_text)}`,
    `- 可用时间范围：${formatDebugBool(temporal.usable_range)}`,
    `- 开始：${formatDebugTime(temporal.start_at)}`,
    `- 结束：${formatDebugTime(temporal.end_at)}`,
    `- 粒度：${formatDebugValue(temporal.granularity, "未知")}`,
    `- 置信度：${formatDebugValue(temporal.confidence, "未返回")}`,
    "",
    "联网和位置",
    `- 是否需要联网：${formatDebugBool(intent.needs_web_search)}`,
    `- 联网查询词：${formatDebugValue(intent.web_query)}`,
    `- 联网原因：${formatDebugValue(intent.web_reason)}`,
    `- 是否需要位置：${formatDebugBool(location.needed)}`,
    `- 位置状态：${formatDebugValue(location.status, "未知")}`,
    tools.length ? "Demo 前置工具调用：" : "Demo 前置工具调用：无",
    ...tools.map(formatDebugTool),
    agentToolCalls.length ? "Agent 内部工具调用：" : "Agent 内部工具调用：无",
    ...agentToolCalls.map(formatAgentToolCall),
    "",
    "模型与耗时",
    `- 模型：${formatDebugValue(runtime.model, "未记录")}`,
    `- Provider：${formatDebugValue(runtime.provider, "未记录")}`,
    `- API 模式：${formatDebugValue(runtime.api_mode, "未记录")}`,
    `- LLM API 调用次数：${formatDebugValue(llm.api_calls, "未记录")}`,
    `- 总耗时：${formatSeconds(timing.total_seconds)}`,
    "助手回复拆解：",
    ...formatAssistantTiming(debug, timing),
    "阶段耗时：",
    ...(timing.stages || []).map(
      (stage) => `- ${formatDebugStageName(stage.name)}：${formatSeconds(stage.seconds)}`,
    ),
    "",
    "执行步骤",
    ...(debug.steps || []).map((step, index) => `${index + 1}. ${formatDebugStep(step)}`),
    "",
    "完整 debug JSON",
    JSON.stringify(debug || {}, null, 2),
  ];
  debugOutputEl.textContent = lines.join("\n");
}

function formatAuditRecord(record) {
  const summary = record.audit_summary || {};
  const timestamp = formatTime(record.timestamp);
  const title = `${timestamp || "时间未知"} · ${formatDebugValue(summary.record_type || record.record_type, "chat_turn")}`;
  const message = record.message ? `- 用户：${record.message}` : "";
  const reply = record.reply ? `- 回复：${String(record.reply).slice(0, 120)}` : "";
  const sourceLines = formatSourceSummary(record.source_summary || {});
  return [
    title,
    message,
    reply,
    `- source：${formatDebugValue(summary.source, "chat")}`,
    `- 保存数量：${summary.saved_memory_count ?? 0}`,
    ...sourceLines,
  ].filter(Boolean).join("\n");
}

async function renderAuditRecords() {
  const payload = await requestJSON(`/api/debug/audit?user_id=${encodeURIComponent(state.userId)}&limit=20`);
  const records = payload.records || [];
  const lines = [
    "最近审计记录",
    `- 共读取：${records.length} 条`,
    "",
    ...(records.length ? records.map(formatAuditRecord) : ["暂无审计记录"]),
  ];
  debugOutputEl.textContent = lines.join("\n\n");
  setDebugOpen(true);
}

function downloadJSON(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function formatTime(value) {
  if (!value) return "";
  return new Date(value * 1000).toLocaleString();
}

function formatMemoryMeta(memory) {
  const kind = formatMemoryKind(memory.kind);
  const typeText = memory.memory_type && memory.memory_type !== "event"
    ? ` · 类型：${memory.memory_type}`
    : "";
  const recorded = formatTime(memory.created_at);
  if (memory.kind === "profile" || memory.kind === "assistant_preference") {
    return `${kind}${typeText} · 记录：${recorded} · ${memory.source}`;
  }
  const start = formatTime(memory.start_at || memory.occurred_at);
  const end = formatTime(memory.end_at);
  const eventTime = start
    ? `发生：${start}${end ? ` 至 ${end}` : ""}`
    : "发生：未解析";
  const granularity = memory.time_granularity && memory.time_granularity !== "unknown"
    ? ` · 粒度：${memory.time_granularity}`
    : "";
  return `${kind}${typeText} · ${eventTime}${granularity} · 记录：${recorded} · ${memory.source}`;
}

function formatDocumentMeta(documentRecord) {
  const uploaded = formatTime(documentRecord.created_at);
  const updated = formatTime(documentRecord.updated_at);
  const updatedText = updated && updated !== uploaded ? ` · 更新：${updated}` : "";
  return `文档 · 上传：${uploaded}${updatedText} · ${documentRecord.source || "markdown_upload"}`;
}

function formatTimelineMeta(chunk) {
  const when = formatTime(chunk.timestamp);
  const refs = `active 引用 ${chunk.active_refs || 0} · retained 引用 ${chunk.retained_refs || 0}`;
  return `${chunk.parent_type || "timeline"} · ${when || "时间未知"} · ${chunk.source || "unknown"} · ${refs}`;
}

function timelineReasonText(reason) {
  const labels = {
    active_memory_reference: "仍被 active 记忆引用，已保留",
    retained_inactive_memory_reference: "仍被历史记忆保留引用，已退出普通搜索但未物理删除",
    no_active_memory_reference: "没有 active 记忆引用，已软删除",
    no_retained_memory_reference: "没有 retained 记忆引用，已彻底删除",
    chunk_not_found: "未找到或不属于当前用户",
    chunk_already_deleted: "此前已经软删除",
  };
  return labels[reason] || reason || "未记录原因";
}

function renderTimelinePanel(title, chunks, emptyText = "没有找到相关原文。") {
  selectedTimelineChunkIds = new Set();
  timelineEvidencePanelEl.innerHTML = "";
  const header = document.createElement("div");
  header.className = "timeline-panel-header";
  const heading = document.createElement("h3");
  heading.textContent = title;
  header.appendChild(heading);
  const close = document.createElement("button");
  close.type = "button";
  close.className = "ghost";
  close.textContent = "收起";
  close.addEventListener("click", () => {
    selectedTimelineChunkIds = new Set();
    timelineEvidencePanelEl.innerHTML = "";
  });
  header.appendChild(close);
  timelineEvidencePanelEl.appendChild(header);

  if (!chunks.length) {
    const empty = document.createElement("div");
    empty.className = "empty-memory";
    empty.textContent = emptyText;
    timelineEvidencePanelEl.appendChild(empty);
    return;
  }

  const actions = document.createElement("div");
  actions.className = "timeline-bulk-actions";
  const softDelete = document.createElement("button");
  softDelete.type = "button";
  softDelete.textContent = "批量删除";
  softDelete.addEventListener("click", () => deleteSelectedTimelineChunks(false).catch((error) => showToast(error.message)));
  actions.appendChild(softDelete);
  const purge = document.createElement("button");
  purge.type = "button";
  purge.className = "hard-purge";
  purge.textContent = "批量彻底删除";
  purge.addEventListener("click", () => deleteSelectedTimelineChunks(true).catch((error) => showToast(error.message)));
  actions.appendChild(purge);
  timelineEvidencePanelEl.appendChild(actions);

  for (const chunk of chunks) {
    timelineEvidencePanelEl.appendChild(renderTimelineChunkRow(chunk));
  }
}

function renderTimelineChunkRow(chunk) {
  const row = document.createElement("label");
  row.className = "timeline-chunk-row";
  row.dataset.status = chunk.status || "active";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.value = chunk.id;
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      selectedTimelineChunkIds.add(chunk.id);
    } else {
      selectedTimelineChunkIds.delete(chunk.id);
    }
  });
  row.appendChild(checkbox);
  const body = document.createElement("div");
  const text = document.createElement("p");
  text.textContent = chunk.text || "";
  body.appendChild(text);
  const meta = document.createElement("div");
  meta.className = "memory-meta";
  meta.textContent = formatTimelineMeta(chunk);
  body.appendChild(meta);
  const badges = document.createElement("div");
  badges.className = "timeline-badges";
  badges.appendChild(timelineBadge((chunk.status || "active") === "active" ? "active" : "已删除"));
  if ((chunk.active_refs || 0) > 0) badges.appendChild(timelineBadge("active 引用"));
  if ((chunk.retained_refs || 0) > 0) badges.appendChild(timelineBadge("retained 引用"));
  body.appendChild(badges);
  row.appendChild(body);
  return row;
}

function timelineBadge(text) {
  const badge = document.createElement("span");
  badge.className = "timeline-badge";
  badge.textContent = text;
  return badge;
}

async function loadEvidenceForMemory(memory) {
  const ids = (memory.evidence_ids || []).filter((id) => String(id || "").startsWith("chunk_"));
  if (!ids.length) {
    showToast("这条记忆没有 timeline chunk 证据");
    return;
  }
  const payload = await requestJSON(
    `/api/timeline/chunks?user_id=${encodeURIComponent(state.userId)}&ids=${encodeURIComponent(ids.join(","))}`,
  );
  renderTimelinePanel("记忆证据原文", payload.chunks || [], "这条记忆的证据片段已不存在。");
}

async function searchTimeline(query) {
  const payload = await requestJSON(
    `/api/timeline/search?user_id=${encodeURIComponent(state.userId)}&q=${encodeURIComponent(query)}&limit=20`,
  );
  renderTimelinePanel("Timeline 搜索结果", payload.chunks || []);
}

async function deleteSelectedTimelineChunks(purge) {
  const ids = Array.from(selectedTimelineChunkIds);
  if (!ids.length) {
    showToast("先选择要处理的原文片段");
    return;
  }
  if (purge && !window.confirm("彻底删除不可恢复。仍被 retained 记忆引用的片段只会软删除并保留原因。是否继续？")) {
    return;
  }
  const payload = await requestJSON(`/api/timeline/chunks?user_id=${encodeURIComponent(state.userId)}&purge=${purge ? "true" : "false"}`, {
    method: "DELETE",
    body: JSON.stringify({ chunk_ids: ids }),
  });
  showToast(timelineDeleteSummary(payload));
  renderTimelineDeleteResults(payload);
  await loadMemories();
}

function renderTimelineDeleteResults(payload) {
  selectedTimelineChunkIds = new Set();
  timelineEvidencePanelEl.innerHTML = "";
  const header = document.createElement("div");
  header.className = "timeline-panel-header";
  const heading = document.createElement("h3");
  heading.textContent = "Timeline 删除结果";
  header.appendChild(heading);
  timelineEvidencePanelEl.appendChild(header);
  for (const result of payload.results || []) {
    const row = document.createElement("div");
    row.className = "timeline-result-row";
    row.dataset.action = result.action || "";
    const title = document.createElement("strong");
    title.textContent = `${result.chunk_id} · ${result.action}`;
    row.appendChild(title);
    const detail = document.createElement("p");
    detail.textContent = `${timelineReasonText(result.reason)} · active 引用 ${result.active_refs || 0} · retained 引用 ${result.retained_refs || 0}`;
    row.appendChild(detail);
    timelineEvidencePanelEl.appendChild(row);
  }
}

function showTyping() {
  typingNode = appendMessage("assistant typing", "正在思考...");
}

function hideTyping() {
  if (typingNode) {
    typingNode.remove();
    typingNode = null;
  }
}

async function requestJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await res.text();
  const payload = text ? JSON.parse(text) : {};
  if (!res.ok) {
    const error = new Error(payload.detail || `HTTP ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return payload;
}

async function pollMemoryJob(jobId) {
  const maxAttempts = 8;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, attempt === 0 ? 700 : 1200));
    const payload = await requestJSON(
      `/api/memory/jobs?user_id=${encodeURIComponent(state.userId)}&job_id=${encodeURIComponent(jobId)}`,
    );
    const job = payload.job || {};
    if (job.status === "pending" || job.status === "running") {
      if (attempt === 0) {
        showToast("正在整理记忆");
      }
      continue;
    }
    const processing = job.memory_processing || {};
    if (job.status === "saved") {
      await loadMemories();
      appendMemoryJobDebug(job);
      showToast(`后台已保存 ${job.saved_count || 0} 条记忆`);
      return job;
    }
    if (job.status === "skipped") {
      appendMemoryJobDebug(job);
      showToast(processing.stage_explanation || "没有需要长期保存的内容");
      return job;
    }
    if (job.status === "rejected") {
      appendMemoryJobDebug(job);
      showToast(processing.stage_explanation || "本次内容被门控拒绝，没有写入长期记忆");
      return job;
    }
    if (job.status === "failed") {
      appendMemoryJobDebug(job);
      const stage = formatDebugValue(processing.stage, "write_failure");
      const errorType = formatDebugValue(processing.error_type, "unknown");
      showToast(`后台保存失败，阶段：${stage}，错误：${errorType}`);
      return job;
    }
    return job;
  }
  showToast("正在整理记忆");
  return null;
}

async function applyChatResponse(message, payload, { appendUser = false } = {}) {
  if (appendUser && message) appendMessage("user", message);
  state.sessionId = payload.session_id || state.sessionId;
  hideTyping();
  const reply = payload.reply || "我收到了，但这次没有生成文字回复。";
  const replyNode = appendMessage("assistant", reply);
  const discussionEvidenceIds = Array.isArray(payload.discussion_recall?.evidence_ids)
    ? payload.discussion_recall.evidence_ids
    : [];
  if (discussionEvidenceIds.length) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost discussion-evidence-button";
    button.textContent = "查看本次依据";
    button.addEventListener("click", () => loadDiscussionEvidence(discussionEvidenceIds).catch((error) => showToast(error.message)));
    replyNode.querySelector(".bubble")?.appendChild(button);
  }
  setVoiceStatus("回复已生成");
  speak(reply);
  renderDebug(payload.debug);
  if (payload.recalled_memories?.length) showToast(`本次召回 ${payload.recalled_memories.length} 条记忆`);
  if (payload.saved_memories?.length) showToast(`已自动保存 ${payload.saved_memories.length} 条新记忆`);
  await loadMemories();
  const memoryJobId = payload.debug?.memory_processing?.job_id;
  if (memoryJobId) {
    pollMemoryJob(memoryJobId).catch((error) => showToast(error.message));
  } else if (payload.debug?.memory_processing?.status === "pending") {
    window.setTimeout(() => loadMemories().catch((error) => showToast(error.message)), 1600);
  }
}

async function sendMessage(message, options = {}) {
  appendMessage("user", message);
  setStatus("thinking");
  setVoiceStatus("已收到，正在生成回复");
  sendButtonEl.disabled = true;
  inputEl.disabled = true;
  ambientStandbyToggleEl.disabled = true;
  showTyping();
  try {
    const location = await refreshLocationForMessage(message);
    const ambientCaptureId = options.ambientCaptureId || "";
    const wakeSession = options.wakeSession || null;
    const payload = await requestJSON("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        user_id: state.userId,
        session_id: state.sessionId,
        location,
        defer_memory_writes: true,
        ambient_capture_id: ambientCaptureId,
        wake_session: wakeSession,
      }),
    });
    await applyChatResponse(message, payload);
  } finally {
    hideTyping();
    sendButtonEl.disabled = false;
    inputEl.disabled = false;
    ambientStandbyToggleEl.disabled = false;
    locationRefreshEl.disabled = false;
    inputEl.focus();
    setStatus("ready");
    updateAmbientStatus();
  }
}

function subjectId(subject) {
  if (subject?.id === undefined || subject?.id === null) return "";
  return String(subject.id);
}

function subjectDisplayName(subject) {
  const displayName = String(subject?.display_name || "").trim();
  const baseName = displayName || (subject?.subject_type === "self" ? "我" : "未命名人物");
  if (subject?.subject_type !== "provisional") return baseName;
  const createdAt = Number(subject?.created_at || 0);
  const timestamp = Number.isFinite(createdAt) && createdAt > 0
    ? new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date(createdAt * 1000))
    : "";
  const shortId = subjectId(subject).slice(0, 4);
  return [baseName, timestamp, shortId].filter(Boolean).join(" · ");
}

function orderedMemorySubjects(subjects) {
  return [...subjects].sort((left, right) => {
    const leftSelf = left.subject_type === "self" ? 0 : 1;
    const rightSelf = right.subject_type === "self" ? 0 : 1;
    if (leftSelf !== rightSelf) return leftSelf - rightSelf;
    return subjectDisplayName(left).localeCompare(subjectDisplayName(right), "zh-CN");
  });
}

function findSelfSubject() {
  return state.memory.subjects.find((subject) => subject.subject_type === "self") || null;
}

function syncNewSubjectField() {
  const isNewSubject = memorySubjectEl.value === NEW_SUBJECT;
  memoryNewSubjectFieldEl.hidden = !isNewSubject;
  memoryNewSubjectNameEl.required = isNewSubject;
  if (!isNewSubject) {
    memoryNewSubjectNameEl.value = "";
  }
}

function renderMemorySubjectSelect() {
  const previousValue = memorySubjectEl.value;
  const subjects = orderedMemorySubjects(state.memory.subjects);
  const selfSubject = subjects.find((subject) => subject.subject_type === "self");
  const selfValue = subjectId(selfSubject) || SELF_SUBJECT;
  memorySubjectEl.innerHTML = "";

  const selfOption = document.createElement("option");
  selfOption.value = selfValue;
  selfOption.textContent = "我";
  memorySubjectEl.appendChild(selfOption);

  for (const subject of subjects) {
    const id = subjectId(subject);
    if (!id || subject.subject_type === "self") continue;
    const option = document.createElement("option");
    option.value = id;
    option.textContent = subjectDisplayName(subject);
    memorySubjectEl.appendChild(option);
  }

  const newSubjectOption = document.createElement("option");
  newSubjectOption.value = NEW_SUBJECT;
  newSubjectOption.textContent = "新人物";
  memorySubjectEl.appendChild(newSubjectOption);

  const availableValues = new Set(Array.from(memorySubjectEl.options, (option) => option.value));
  memorySubjectEl.value = availableValues.has(previousValue) ? previousValue : selfValue;
  syncNewSubjectField();
}

function renderMemorySubjectFilter() {
  const subjects = orderedMemorySubjects(state.memory.subjects);
  const selfSubject = subjects.find((subject) => subject.subject_type === "self");
  const selfValue = subjectId(selfSubject) || SELF_SUBJECT;
  if (state.memory.activeSubjectId === SELF_SUBJECT && selfSubject) {
    state.memory.activeSubjectId = selfValue;
  }
  const availableValues = new Set([ALL_SUBJECTS, selfValue, ...subjects.map(subjectId).filter(Boolean)]);
  if (!availableValues.has(state.memory.activeSubjectId)) {
    state.memory.activeSubjectId = ALL_SUBJECTS;
  }

  const filters = [
    { id: ALL_SUBJECTS, label: "全部" },
    { id: selfValue, label: "我" },
    ...subjects
      .filter((subject) => subject.subject_type !== "self" && subjectId(subject))
      .map((subject) => ({ id: subjectId(subject), label: subjectDisplayName(subject) })),
  ];
  memorySubjectFilterEl.innerHTML = "";
  for (const filter of filters) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.subjectFilter = filter.id;
    button.setAttribute("aria-pressed", String(state.memory.activeSubjectId === filter.id));
    button.textContent = filter.label;
    button.title = filter.label;
    button.addEventListener("click", async () => {
      state.memory.activeSubjectId = filter.id;
      renderMemorySubjectFilter();
      try {
        await loadMemories();
      } catch (error) {
        showToast(error.message);
      }
    });
    memorySubjectFilterEl.appendChild(button);
  }
}

function memoryMatchesActiveSubject(memory) {
  const activeSubjectId = state.memory.activeSubjectId;
  if (activeSubjectId === ALL_SUBJECTS) return true;
  const activeSubject = state.memory.subjects.find((subject) => subjectId(subject) === activeSubjectId);
  const isSelf = activeSubjectId === SELF_SUBJECT || activeSubject?.subject_type === "self";
  if (isSelf) {
    const hasExplicitSubject = memory.subject_id !== undefined && memory.subject_id !== null;
    return memory.subject_type === "self"
      || (hasExplicitSubject && String(memory.subject_id) === activeSubjectId)
      || (!hasExplicitSubject && !memory.subject_name);
  }
  return memory.subject_id !== undefined
    && memory.subject_id !== null
    && String(memory.subject_id) === activeSubjectId;
}

function renderMemoryList() {
  memoryListEl.innerHTML = "";
  const memories = state.memory.memories.filter(memoryMatchesActiveSubject);
  const documents = state.memory.activeSubjectId === ALL_SUBJECTS ? state.memory.documents : [];
  const profileCount = memories.filter((memory) => memory.kind === "profile").length;
  const assistantPreferenceCount = memories.filter((memory) => memory.kind === "assistant_preference").length;
  const eventCount = memories.filter((memory) => !["profile", "assistant_preference"].includes(memory.kind)).length;
  memoryCountEl.textContent = `${profileCount} 画像 · ${assistantPreferenceCount} 助手偏好 · ${eventCount} 事件 · ${documents.length} 文档`;

  if (!memories.length && !documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-memory";
    empty.textContent = state.memory.activeSubjectId === ALL_SUBJECTS
      ? "还没有记忆或文档。聊天时说“记住...”，或上传一份 Markdown 文档。"
      : "这个人物还没有长期记忆。";
    memoryListEl.appendChild(empty);
    return;
  }

  const rows = [
    ...memories.map((memory) => ({ type: "memory", created_at: memory.created_at || 0, item: memory })),
    ...documents.map((documentRecord) => ({ type: "document", created_at: documentRecord.created_at || 0, item: documentRecord })),
  ].sort((left, right) => right.created_at - left.created_at);
  for (const row of rows) {
    memoryListEl.appendChild(row.type === "document" ? renderDocumentCard(row.item) : renderMemoryCard(row.item));
  }
}

async function loadMemories() {
  const params = new URLSearchParams({ user_id: state.userId });
  if (state.memory.activeSubjectId !== ALL_SUBJECTS) {
    const activeSubject = state.memory.subjects.find(
      (subject) => subjectId(subject) === state.memory.activeSubjectId,
    );
    const requestedSubjectId = subjectId(activeSubject)
      || (state.memory.activeSubjectId === SELF_SUBJECT ? subjectId(findSelfSubject()) : state.memory.activeSubjectId);
    if (requestedSubjectId && requestedSubjectId !== SELF_SUBJECT) {
      params.set("subject_id", requestedSubjectId);
    }
  }
  const payload = await requestJSON(`/api/memories?${params.toString()}`);
  state.memory.subjects = Array.isArray(payload.subjects) ? payload.subjects : [];
  state.memory.memories = Array.isArray(payload.memories) ? payload.memories : [];
  state.memory.documents = Array.isArray(payload.documents) ? payload.documents : [];
  renderMemorySubjectSelect();
  renderMemorySubjectFilter();
  if (state.memory.activeView === "long-term") renderMemoryList();
}

async function loadDiscussionEvidence(ids) {
  const normalized = Array.from(new Set((ids || []).map(String).filter(Boolean)));
  const payload = await requestJSON(
    `/api/timeline/chunks?user_id=${encodeURIComponent(state.userId)}&ids=${encodeURIComponent(normalized.join(","))}`,
  );
  renderTimelinePanel("讨论依据原文", payload.chunks || [], "原文已超过 30 天或已被用户删除，当前只保留讨论摘要。");
}

function setMemoryView(view) {
  const discussions = view === "discussions";
  state.memory.activeView = discussions ? "discussions" : "long-term";
  longTermMemoryViewEl.hidden = discussions;
  dailyDiscussionViewEl.hidden = !discussions;
  memoryViewLongTermEl.setAttribute("aria-selected", String(!discussions));
  memoryViewDiscussionsEl.setAttribute("aria-selected", String(discussions));
  memoryPaneTitleEl.textContent = discussions ? "每日回顾" : "当前记忆";
  refreshMemoryEl.dataset.tooltip = discussions ? "刷新每日回顾" : "刷新记忆列表";
  if (discussions) loadDiscussionDays().catch((error) => showToast(error.message));
}

async function loadDiscussionDays() {
  const payload = await requestJSON(`/api/discussions/days?user_id=${encodeURIComponent(state.userId)}&limit=30`);
  state.memory.discussionDays = Array.isArray(payload.days) ? payload.days : [];
  renderDiscussionDays();
}

function renderDiscussionDays() {
  dailyDiscussionListEl.innerHTML = "";
  memoryCountEl.textContent = `${state.memory.discussionDays.length} 天`;
  if (!state.memory.discussionDays.length) {
    const empty = document.createElement("div");
    empty.className = "empty-memory";
    empty.textContent = "还没有每日讨论摘要。全天待机产生 final 转写后，可直接询问今天讨论了什么。";
    dailyDiscussionListEl.appendChild(empty);
    return;
  }
  for (const day of state.memory.discussionDays) {
    dailyDiscussionListEl.appendChild(renderDiscussionDay(day));
  }
}

function renderDiscussionDay(day) {
  const section = document.createElement("section");
  section.className = "discussion-day";
  const header = document.createElement("div");
  header.className = "discussion-day-header";
  const heading = document.createElement("h3");
  heading.textContent = day.day || "日期未知";
  const meta = document.createElement("span");
  meta.className = "memory-meta";
  meta.textContent = `${day.topic_count || 0} 个话题`;
  header.append(heading, meta);
  section.appendChild(header);
  const overview = document.createElement("p");
  overview.className = "discussion-overview";
  overview.textContent = day.overview || "摘要生成中";
  section.appendChild(overview);

  const topics = document.createElement("div");
  section.appendChild(topics);
  const actions = document.createElement("div");
  actions.className = "discussion-actions";
  const detail = document.createElement("button");
  detail.type = "button";
  detail.className = "ghost";
  detail.textContent = "查看话题";
  detail.addEventListener("click", async () => {
    const payload = await requestJSON(
      `/api/discussions/day?user_id=${encodeURIComponent(state.userId)}&date=${encodeURIComponent(day.day)}`,
    );
    const loaded = payload.day || {};
    topics.innerHTML = "";
    for (const topic of loaded.topics || []) topics.appendChild(renderDiscussionTopic(topic));
    detail.disabled = true;
  });
  actions.appendChild(detail);
  for (const [scope, label, dangerous] of [
    ["raw", "删除原文", false],
    ["summary", "删除摘要", false],
    ["all", "全部删除", true],
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = dangerous ? "danger" : "ghost";
    button.textContent = label;
    button.addEventListener("click", () => deleteDiscussionDay(day.day, scope));
    actions.appendChild(button);
  }
  section.appendChild(actions);
  return section;
}

function renderDiscussionTopic(topic) {
  const node = document.createElement("section");
  node.className = "discussion-topic";
  const title = document.createElement("h4");
  title.textContent = topic.title || "未命名话题";
  const summary = document.createElement("p");
  summary.textContent = topic.summary || "";
  const meta = document.createElement("p");
  meta.className = "memory-meta";
  meta.textContent = `${formatTime(topic.start_at)} 至 ${formatTime(topic.end_at)} · 原文${topic.evidence_status === "available" ? "可用" : "已部分或全部过期"}`;
  node.append(title, summary, meta);
  if ((topic.available_evidence_ids || []).length) {
    const evidence = document.createElement("button");
    evidence.type = "button";
    evidence.className = "ghost discussion-evidence-button";
    evidence.textContent = "查看原文";
    evidence.addEventListener("click", () => loadDiscussionEvidence(topic.available_evidence_ids).catch((error) => showToast(error.message)));
    node.appendChild(evidence);
  }
  return node;
}

async function deleteDiscussionDay(day, scope) {
  const labels = { raw: "原文", summary: "摘要", all: "原文和摘要" };
  if (!window.confirm(`确认删除 ${day} 的${labels[scope]}？该操作不可恢复。`)) return;
  const payload = await requestJSON(
    `/api/discussions/day?user_id=${encodeURIComponent(state.userId)}&date=${encodeURIComponent(day)}&scope=${encodeURIComponent(scope)}`,
    { method: "DELETE" },
  );
  selectedTimelineChunkIds = new Set();
  timelineEvidencePanelEl.innerHTML = "";
  showToast(`已处理 ${payload.purged_raw_count || 0} 段原文和 ${payload.deleted_summary_record_count || 0} 条摘要记录`);
  await loadDiscussionDays();
}

async function loadSpeakerProfile() {
  const payload = await requestJSON(`/api/speaker/profile?user_id=${encodeURIComponent(state.userId)}`);
  state.speaker.enrolled = Boolean(payload.enrolled);
  state.speaker.updatedAt = payload.updated_at || null;
  state.speaker.speakerModel = String(payload.speaker_model || "");
  state.speaker.speakerSource = String(payload.speaker_source || "");
  state.speaker.sampleCount = Number(payload.sample_count || 0);
  state.speaker.targetSampleCount = Number(payload.target_sample_count || 3);
  state.speaker.calibrationStatus = String(payload.calibration_status || "not_enrolled");
  state.speaker.speakerProfileVersion = payload.speaker_profile_version || null;
  if (state.speaker.calibrationStatus !== "pending") {
    state.speaker.enrollmentSessionId = "";
  }
  updateSpeakerProfileSummary();
  return payload;
}

function ensureSpeakerEnrollmentSession() {
  if (state.speaker.enrollmentSessionId) return;
  state.speaker.enrollmentSessionId = `speaker_enroll_${Date.now()}`;
  state.speaker.sampleCount = 0;
  state.speaker.calibrationStatus = "pending";
  updateSpeakerProfileSummary();
}

async function discardPendingSpeakerEnrollment() {
  const enrollmentSessionId = state.speaker.enrollmentSessionId;
  await finishSpeakerEnrollmentFlow(state.audio.active, true);
  if (!isAndroidNative() && enrollmentSessionId) {
    await requestJSON(`/api/speaker/profile?user_id=${encodeURIComponent(state.userId)}&enrollment_session_id=${encodeURIComponent(enrollmentSessionId)}`, {
      method: "DELETE",
    });
  }
  state.speaker.enrollmentSessionId = "";
  await loadSpeakerProfile();
}

async function startSpeakerEnrollmentFlow() {
  if (isAndroidNative()) {
    if (state.audio.nativeStatus?.model_state !== "ready") {
      throw new Error("声纹录入不可用：请先在设置页完成五项模型自检");
    }
    ensureSpeakerEnrollmentSession();
    state.audio.nativeEnrollmentCompletionSession = "";
    setSpeakerEnrollStatus("正在启动原生声纹录入…", "recording");
    callAndroidBridge("startSpeakerEnrollment", state.speaker.enrollmentSessionId);
    await syncNativeAudioStatus();
    return;
  }
  if (!supportsUnifiedAudio()) throw new Error(getLocalASRUnsupportedMessage());
  if (!state.audio.capabilities?.speaker_enrollment_ready) {
    throw new Error(`声纹录入不可用：${audioCapabilityReason(["vad", "speaker"])}`);
  }
  const ambientSession = state.audio.active?.mode === "ambient" ? state.audio.active : null;
  const ambientWasRunning = Boolean(ambientSession);
  state.speaker.resumeAmbientAfterEnrollment = ambientWasRunning;
  state.speaker.resumeAmbientUserId = ambientSession?.userId || "";
  state.ambient.wakePending = false;
  state.ambient.wakeSession = null;
  state.ambient.enabled = false;
  state.ambient.status = "idle";
  ensureSpeakerEnrollmentSession();
  updateAmbientStatus();
  setSpeakerEnrollStatus(`录音中，请完成第 ${Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3)} 段朗读内容…`, "recording");
  try {
    if (ambientSession) {
      await stopUnifiedAudioSession({
        interrupted: false,
        active: ambientSession,
        stopReason: "pause_for_enrollment",
      });
    }
    await startUnifiedAudioSession("speaker_enroll");
  } catch (error) {
    await finishSpeakerEnrollmentFlow(null, true);
    throw error;
  }
}

async function finishSpeakerEnrollmentFlow(active = state.audio.active, interrupted = true) {
  if (isAndroidNative()) {
    const enrollmentState = String(state.audio.nativeStatus?.enrollment_state || "idle");
    if (new Set(["recording", "processing", "error"]).has(enrollmentState)) {
      callAndroidBridge("cancelSpeakerEnrollment");
      await syncNativeAudioStatus();
    }
    return;
  }
  const shouldResume = Boolean(state.speaker.resumeAmbientAfterEnrollment);
  const resumeUserId = state.speaker.resumeAmbientUserId;
  state.speaker.resumeAmbientAfterEnrollment = false;
  state.speaker.resumeAmbientUserId = "";
  if (active?.mode === "speaker_enroll") {
    await stopUnifiedAudioSession({ interrupted, active });
  }
  if (shouldResume && resumeUserId && state.userId === resumeUserId && !state.audio.active) {
    await startUnifiedAudioSession("ambient");
    showToast("声纹录入结束，已恢复全天待机");
  }
}

function renderMemoryCard(memory) {
  const card = document.createElement("div");
  card.className = "memory-card";
  card.dataset.kind = memory.kind || "event";

  const memorySubjectId = memory.subject_id === undefined || memory.subject_id === null
    ? ""
    : String(memory.subject_id);
  const subjectRecord = state.memory.subjects.find((subject) => subjectId(subject) === memorySubjectId);
  const hasSubjectMetadata = Boolean(memorySubjectId || memory.subject_name || memory.subject_type);
  const fallbackSubjectName = !hasSubjectMetadata || memory.subject_type === "self" ? "我" : "未命名人物";
  const subjectName = subjectRecord
    ? subjectDisplayName(subjectRecord)
    : String(memory.subject_name || "").trim() || fallbackSubjectName;
  const subjectType = subjectRecord?.subject_type || memory.subject_type || "self";
  const subjectLabel = document.createElement("span");
  subjectLabel.className = "memory-subject-label";
  subjectLabel.dataset.subjectType = subjectType;
  subjectLabel.textContent = subjectName;
  subjectLabel.title = `记忆归属：${subjectName}`;
  card.appendChild(subjectLabel);

  const text = document.createElement("p");
  text.textContent = memory.content;
  card.appendChild(text);

  const meta = document.createElement("div");
  meta.className = "memory-meta";
  meta.textContent = formatMemoryMeta(memory);
  card.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "card-actions";
  const timelineEvidenceIds = (memory.evidence_ids || []).filter((id) => String(id || "").startsWith("chunk_"));
  if (timelineEvidenceIds.length) {
    const evidence = document.createElement("button");
    evidence.type = "button";
    evidence.className = "secondary";
    evidence.textContent = "查看证据";
    evidence.addEventListener("click", () => loadEvidenceForMemory(memory).catch((error) => showToast(error.message)));
    actions.appendChild(evidence);
  }
  const del = document.createElement("button");
  del.type = "button";
  del.textContent = "删除";
  del.addEventListener("click", async () => {
    await requestJSON(`/api/memories/${memory.id}?user_id=${encodeURIComponent(state.userId)}`, {
      method: "DELETE",
    });
    showToast("已删除记忆");
    await loadMemories();
  });
  actions.appendChild(del);
  const purge = document.createElement("button");
  purge.type = "button";
  purge.className = "hard-purge";
  purge.textContent = "彻底删除";
  purge.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "彻底删除不可恢复。确认后会物理删除这条记忆，并清理未共享的原文证据和相关 audit 行。是否继续？",
    );
    if (!confirmed) return;
    const payload = await requestJSON(`/api/memories/${memory.id}?user_id=${encodeURIComponent(state.userId)}&purge=true`, {
      method: "DELETE",
    });
    showToast(purgeSummary(payload));
    await loadMemories();
  });
  actions.appendChild(purge);
  card.appendChild(actions);
  return card;
}

function renderDocumentCard(documentRecord) {
  const card = document.createElement("div");
  card.className = "memory-card";
  card.dataset.kind = "document";

  const text = document.createElement("p");
  text.textContent = `${documentRecord.filename}${documentRecord.title ? ` · ${documentRecord.title}` : ""}`;
  card.appendChild(text);

  if (documentRecord.summary) {
    const summary = document.createElement("div");
    summary.className = "document-summary";
    summary.textContent = documentRecord.summary;
    card.appendChild(summary);
  }

  const meta = document.createElement("div");
  meta.className = "memory-meta";
  meta.textContent = formatDocumentMeta(documentRecord);
  card.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "card-actions";
  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "secondary";
  edit.textContent = "编辑";
  edit.addEventListener("click", () => editDocument(card, documentRecord.id).catch((error) => showToast(error.message)));
  actions.appendChild(edit);

  const del = document.createElement("button");
  del.type = "button";
  del.textContent = "删除";
  del.addEventListener("click", async () => {
    await requestJSON(`/api/documents/${documentRecord.id}?user_id=${encodeURIComponent(state.userId)}`, {
      method: "DELETE",
    });
    showToast("已删除文档");
    await loadMemories();
  });
  actions.appendChild(del);
  const purge = document.createElement("button");
  purge.type = "button";
  purge.className = "hard-purge";
  purge.textContent = "彻底删除";
  purge.addEventListener("click", async () => {
    const confirmed = window.confirm(
      "彻底删除不可恢复。确认后会物理删除这份文档，并清理相关 audit 行；本轮不会清理 timeline 原文。是否继续？",
    );
    if (!confirmed) return;
    const payload = await requestJSON(`/api/documents/${documentRecord.id}?user_id=${encodeURIComponent(state.userId)}&purge=true`, {
      method: "DELETE",
    });
    showToast(purgeSummary(payload));
    await loadMemories();
  });
  actions.appendChild(purge);
  card.appendChild(actions);
  return card;
}

async function editDocument(card, documentId) {
  const payload = await requestJSON(`/api/documents/${documentId}?user_id=${encodeURIComponent(state.userId)}`);
  const documentRecord = payload.document || {};
  card.innerHTML = "";
  const form = document.createElement("form");
  form.className = "document-edit-form";

  const filename = document.createElement("input");
  filename.value = documentRecord.filename || "";
  filename.placeholder = "文件名";
  form.appendChild(filename);

  const title = document.createElement("input");
  title.value = documentRecord.title || "";
  title.placeholder = "标题";
  form.appendChild(title);

  const summary = document.createElement("textarea");
  summary.rows = 2;
  summary.value = documentRecord.summary || "";
  summary.placeholder = "摘要";
  form.appendChild(summary);

  const content = document.createElement("textarea");
  content.rows = 8;
  content.value = documentRecord.content || "";
  content.placeholder = "Markdown 原文";
  form.appendChild(content);

  const actions = document.createElement("div");
  actions.className = "card-actions";
  const save = document.createElement("button");
  save.type = "submit";
  save.className = "secondary";
  save.textContent = "保存";
  actions.appendChild(save);
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary";
  cancel.textContent = "取消";
  cancel.addEventListener("click", () => loadMemories().catch((error) => showToast(error.message)));
  actions.appendChild(cancel);
  form.appendChild(actions);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await requestJSON(`/api/documents/${documentId}`, {
      method: "PATCH",
      body: JSON.stringify({
        user_id: state.userId,
        filename: filename.value,
        title: title.value,
        summary: summary.value,
        content: content.value,
      }),
    });
    showToast("文档已更新");
    await loadMemories();
  });
  card.appendChild(form);
}

// Markdown 上传只负责读取文本，文档归档仍走统一 import API。
async function importMarkdownFile(file) {
  if (!isMarkdownFile(file)) {
    showToast("当前仅支持 Markdown .md 文件");
    markdownImportInputEl.value = "";
    return;
  }
  const text = (await file.text()).trim();
  if (!text) {
    showToast("文件内容为空，未导入");
    markdownImportInputEl.value = "";
    return;
  }
  markdownImportButtonEl.disabled = true;
  try {
    const result = await requestJSON("/api/memory/import", {
      method: "POST",
      body: JSON.stringify({
        user_id: state.userId,
        text,
        source: "markdown_upload",
        context: file.name,
        confirm: false,
      }),
    });
    await loadMemories();
    appendMarkdownImportDebug(file.name, result);
    if (result.reply) {
      appendMessage("assistant", result.reply);
      speak(result.reply);
    }
    showToast("文档已归档");
  } catch (error) {
    showToast(error.message);
    appendMessage("system", error.message);
  } finally {
    markdownImportButtonEl.disabled = false;
    markdownImportInputEl.value = "";
  }
}

function setUserSelectionOpen(open) {
  if (isAndroidNative()) {
    userSelectModalEl.hidden = true;
    return;
  }
  userSelectModalEl.hidden = !open;
  if (open) {
    userIdInputEl.value = state.userId || localStorage.getItem(USER_STORAGE_KEY) || "";
    userSelectErrorEl.textContent = "";
    window.setTimeout(() => userIdInputEl.focus(), 0);
  }
}

function resetUserScopedState() {
  state.sessionId = null;
  state.location = null;
  state.ambient = {
    enabled: false,
    wakePending: false,
    captureId: null,
    chunkCount: 0,
    status: "idle",
    lastSegmentId: "",
    lastCapturedAt: 0,
    chunks: [],
    wakeSession: null,
  };
  state.speaker = {
    enrolled: false,
    updatedAt: null,
    speakerModel: "",
    speakerSource: "",
    modalOpen: false,
    sampleCount: 0,
    targetSampleCount: 3,
    calibrationStatus: "not_enrolled",
    speakerProfileVersion: null,
    enrollmentSessionId: "",
    resumeAmbientAfterEnrollment: false,
    resumeAmbientUserId: "",
  };
  state.memory = { subjects: [], memories: [], documents: [], activeSubjectId: ALL_SUBJECTS };
  messagesEl.replaceChildren();
  debugOutputEl.textContent = "等待用户发送 query...";
  updateAmbientStatus();
  updateSpeakerProfileSummary();
}

async function selectUser(rawUserId) {
  const nextUserId = String(rawUserId || "").trim();
  if (!nextUserId || nextUserId.length > 64 || /[\r\n\t]/.test(nextUserId)) {
    throw new Error("体验者 ID 必须为 1 到 64 个字符，且不能包含换行或制表符");
  }
  if (state.audio.active) await stopUnifiedAudioSession({ interrupted: true });
  resetUserScopedState();
  state.userId = nextUserId;
  localStorage.setItem(USER_STORAGE_KEY, nextUserId);
  userSwitchEl.textContent = `用户 ${nextUserId.slice(0, 6)}`;
  userSwitchEl.title = nextUserId;
  inputEl.disabled = false;
  sendButtonEl.disabled = false;
  setUserSelectionOpen(false);
  appendMessage("assistant", `我在。当前体验者是 ${nextUserId}，可以使用文字聊天或开启全天待机。`);
  await Promise.all([loadSpeakerProfile(), loadMemories()]);
  inputEl.focus();
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  try {
    await sendMessage(message);
  } catch (error) {
    setStatus("error");
    appendMessage("system", error.message);
  }
});

userSwitchEl?.addEventListener("click", () => {
  if (!isAndroidNative()) {
    if (MOBILE_LAYOUT.matches) closeUiSurface("settings");
    setUserSelectionOpen(true);
  }
});

userSelectFormEl?.addEventListener("submit", async (event) => {
  event.preventDefault();
  userSelectErrorEl.textContent = "";
  try {
    await selectUser(userIdInputEl.value);
  } catch (error) {
    userSelectErrorEl.textContent = error.message;
    userSelectErrorEl.dataset.state = "error";
  }
});

markdownImportButtonEl.addEventListener("click", () => {
  markdownImportInputEl.click();
});

markdownImportInputEl.addEventListener("change", () => {
  const file = markdownImportInputEl.files?.[0];
  if (!file) return;
  importMarkdownFile(file).catch((error) => {
    showToast(error.message);
    appendMessage("system", error.message);
    markdownImportButtonEl.disabled = false;
    markdownImportInputEl.value = "";
  });
});

voiceToggleEl.addEventListener("click", () => {
  state.voiceEnabled = !state.voiceEnabled;
  voiceToggleEl.textContent = state.voiceEnabled ? "播报开" : "播报关";
  voiceToggleEl.dataset.tooltip = state.voiceEnabled ? "关闭语音播报" : "开启语音播报";
  voiceToggleEl.setAttribute("aria-pressed", String(state.voiceEnabled));
  voiceToggleEl.setAttribute("aria-checked", String(state.voiceEnabled));
  if (!state.voiceEnabled) {
    stopSpeaking();
  }
});

ambientStandbyToggleEl.addEventListener("click", async () => {
  const running = state.audio.active?.mode === "ambient";
  if (isAndroidNative()) {
    try {
      if (running) {
        await stopNativeAmbient();
        showToast("正在停止全天待机");
      } else {
        await startNativeAmbient();
        showToast("已请求开启 Android 后台收音");
      }
    } catch (error) {
      state.ambient.enabled = false;
      state.ambient.status = "error";
      showToast(error.message);
    }
    updateAmbientStatus();
    return;
  }
  if (!running) {
    try {
      if (!supportsUnifiedAudio()) throw new Error(getLocalASRUnsupportedMessage());
      await startUnifiedAudioSession("ambient");
      pruneAmbientContext();
      showToast("全天待机已开启");
    } catch (error) {
      state.ambient.enabled = false;
      state.ambient.status = "idle";
      showToast(error.message);
    }
  } else {
    state.ambient.wakePending = false;
    state.ambient.status = "idle";
    state.ambient.wakeSession = null;
    pruneAmbientContext();
    await stopUnifiedAudioSession({ interrupted: false });
    showToast("全天待机已停止");
  }
  updateAmbientStatus();
});

speakerEnrollButtonEl?.addEventListener("click", () => {
  openUiSurface("speaker", speakerEnrollButtonEl);
});

speakerEnrollCloseEl?.addEventListener("click", async () => {
  closeUiSurface("speaker");
});

speakerEnrollCancelEl?.addEventListener("click", async () => {
  try {
    await discardPendingSpeakerEnrollment();
  } catch (error) {
    showToast(error.message);
  }
  setSpeakerEnrollModalOpen(false);
  closeUiSurface("speaker");
});

speakerEnrollStartEl?.addEventListener("click", () => {
  startSpeakerEnrollmentFlow().catch((error) => {
    setSpeakerEnrollStatus(error.message, "error");
    showToast(error.message);
  });
});

speakerEnrollRetryEl?.addEventListener("click", async () => {
  try {
    await discardPendingSpeakerEnrollment();
    await startSpeakerEnrollmentFlow();
  } catch (error) {
    setSpeakerEnrollStatus(error.message, "error");
    showToast(error.message);
  }
});

memoryFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = memoryInputEl.value.trim();
  if (!content) return;
  const selectedSubjectId = memorySubjectEl.value;
  const newSubjectName = memoryNewSubjectNameEl.value.trim();
  if (selectedSubjectId === NEW_SUBJECT && !newSubjectName) {
    showToast("请输入人物姓名");
    memoryNewSubjectNameEl.focus();
    return;
  }
  const requestBody = {
    content,
    kind: memoryKindEl.value,
    user_id: state.userId,
    tags: ["manual"],
  };
  if (selectedSubjectId === NEW_SUBJECT) {
    requestBody.subject_name = newSubjectName;
    requestBody.subject_type = "named";
  } else if (selectedSubjectId !== SELF_SUBJECT) {
    requestBody.subject_id = selectedSubjectId;
  }
  try {
    await requestJSON("/api/memories", {
      method: "POST",
      body: JSON.stringify(requestBody),
    });
    memoryInputEl.value = "";
    memoryNewSubjectNameEl.value = "";
    await loadMemories();
    const selfSubject = findSelfSubject();
    memorySubjectEl.value = subjectId(selfSubject) || SELF_SUBJECT;
    syncNewSubjectField();
    showToast("记忆已保存");
  } catch (error) {
    showToast(error.message);
  }
});

memorySubjectEl.addEventListener("change", () => {
  syncNewSubjectField();
  if (!memoryNewSubjectFieldEl.hidden) {
    memoryNewSubjectNameEl.focus();
  }
});

refreshMemoryEl.addEventListener("click", () => {
  const request = state.memory.activeView === "discussions" ? loadDiscussionDays() : loadMemories();
  request
    .then(() => showToast(state.memory.activeView === "discussions" ? "每日回顾已刷新" : "记忆已刷新"))
    .catch((error) => showToast(error.message));
});

memoryViewLongTermEl.addEventListener("click", () => {
  setMemoryView("long-term");
  renderMemoryList();
});

memoryViewDiscussionsEl.addEventListener("click", () => setMemoryView("discussions"));

locationRefreshEl.addEventListener("click", async () => {
  locationRefreshEl.disabled = true;
  locationRefreshEl.textContent = "定位中";
  try {
    state.location = await getCurrentLocation();
    if (state.location.status === "available") {
      locationRefreshEl.textContent = "已定位";
      if (locationSettingSummaryEl) locationSettingSummaryEl.textContent = `定位可用 · 精度约 ${Math.round(state.location.accuracy || 0)} 米`;
      showToast(`已刷新定位，精度约 ${Math.round(state.location.accuracy || 0)} 米`);
    } else {
      locationRefreshEl.textContent = "定位";
      if (locationSettingSummaryEl) locationSettingSummaryEl.textContent = locationToastMessage(state.location);
      showToast(locationToastMessage(state.location));
    }
  } finally {
    locationRefreshEl.disabled = false;
  }
});

memoryToggleEl.addEventListener("click", () => {
  if (MOBILE_LAYOUT.matches) openUiSurface("memory", memoryToggleEl);
  else {
    const open = memoryPaneEl.classList.toggle("open");
    memoryToggleEl.setAttribute("aria-expanded", String(open));
  }
});

memoryCloseEl?.addEventListener("click", () => closeUiSurface("memory"));
memoryBackdropEl?.addEventListener("click", () => closeUiSurface("memory"));

settingsToggleEl?.addEventListener("click", () => openUiSurface("settings", settingsToggleEl));
settingsCloseEl?.addEventListener("click", () => closeUiSurface("settings"));
settingsBackdropEl?.addEventListener("click", () => closeUiSurface("settings"));
nativeSettingsButtonEl?.addEventListener("click", () => {
  try {
    callAndroidBridge("openSettings");
  } catch (error) {
    showToast(error.message);
  }
});

debugToggleEl.addEventListener("click", () => {
  if (MOBILE_LAYOUT.matches) openUiSurface("debug", debugToggleEl);
  else setDebugOpen(!document.body.classList.contains("debug-open"));
});

closeDebugEl.addEventListener("click", () => {
  closeUiSurface("debug");
});

debugBackdropEl.addEventListener("click", () => {
  closeUiSurface("debug");
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  const surface = currentUiSurface();
  if (surface) {
    event.preventDefault();
    closeUiSurface(surface);
  }
});

clearDebugEl.addEventListener("click", () => {
  debugOutputEl.textContent = "等待用户发送 query...";
});

viewAuditEl.addEventListener("click", () => {
  renderAuditRecords().catch((error) => showToast(error.message));
});

exportAuditEl.addEventListener("click", async () => {
  try {
    const payload = await requestJSON(`/api/debug/audit?user_id=${encodeURIComponent(state.userId)}&limit=50`);
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadJSON(`ai-glasses-audit-${timestamp}.json`, payload);
    showToast(`已导出 ${payload.records?.length || 0} 条诊断记录`);
  } catch (error) {
    showToast(error.message);
  }
});

document.body.classList.toggle("android-native", isAndroidNative());
if (MOBILE_LAYOUT.matches) {
  window.history.replaceState({ ...(window.history.state || {}), [UI_SURFACE_HISTORY_KEY]: "" }, "");
}
MOBILE_LAYOUT.addEventListener("change", () => {
  window.history.replaceState({ ...(window.history.state || {}), [UI_SURFACE_HISTORY_KEY]: "" }, "");
  applyUiSurface("");
});

setupButtonTooltips();
setupLocalASR().catch((error) => {
  setVoiceStatus(error.message, "error");
  showToast(error.message);
});
inputEl.disabled = true;
sendButtonEl.disabled = true;
updateSpeakerProfileSummary();
if (isAndroidNative()) {
  userSwitchEl.hidden = true;
  const runtimeDescriptionEl = document.querySelector("#runtime-description");
  if (runtimeDescriptionEl) runtimeDescriptionEl.textContent = "Android 本机运行 · 一机一位体验者";
  const ownerId = String(callAndroidBridge("ownerId") || "");
  selectUser(ownerId).catch((error) => {
    setVoiceStatus(error.message, "error");
    showToast(error.message);
  });
  window.setInterval(() => {
    syncNativeAudioStatus().catch((error) => console.warn("Android audio status unavailable", error));
  }, 1000);
  window.setInterval(() => {
    syncNativeAudioUiStatus();
  }, 250);
} else {
  setUserSelectionOpen(true);
}

window.addEventListener("pagehide", () => {
  if (isAndroidNative()) return;
  const active = state.audio.active;
  if (!active) return;
  fetch("/api/audio/session/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: state.userId,
      audio_session_id: active.id,
      session_token: active.token,
      interrupted: true,
    }),
    keepalive: true,
  }).catch(() => null);
});
