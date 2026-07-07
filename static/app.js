const state = {
  sessionId: null,
  userId: "local-user",
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
  },
  ambient: {
    enabled: false,
    wakePending: false,
    captureId: null,
    chunkCount: 0,
    status: "idle",
    lastSegmentId: "",
    chunks: [],
    wakeSession: null,
    wakeDetectorEnabled: true,
    wakeWordPhrases: ["hey hermes", "hi hermes", "hermes", "小赫墨斯", "赫耳墨斯"],
  },
};

const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const memoryFormEl = document.querySelector("#memory-form");
const memoryInputEl = document.querySelector("#memory-input");
const memoryKindEl = document.querySelector("#memory-kind");
const memoryListEl = document.querySelector("#memory-list");
const refreshMemoryEl = document.querySelector("#refresh-memory");
const memoryCountEl = document.querySelector("#memory-count");
const sendButtonEl = document.querySelector("#send-button");
const toastEl = document.querySelector("#toast");
const memoryPaneEl = document.querySelector("#memory-pane");
const memoryToggleEl = document.querySelector("#memory-toggle");
const locationRefreshEl = document.querySelector("#location-refresh");
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
const ambientModeLabelEl = document.querySelector("#ambient-mode-label");
const ambientStatusEl = document.querySelector("#ambient-status");
const ambientStandbyToggleEl = document.querySelector("#ambient-standby-toggle");
const ambientWakeButtonEl = document.querySelector("#ambient-wake-button");
const speakerEnrollButtonEl = document.querySelector("#speaker-enroll-button");
const ambientClearButtonEl = document.querySelector("#ambient-clear-button");
const markdownImportButtonEl = document.querySelector("#markdown-import-button");
const markdownImportInputEl = document.querySelector("#markdown-import-input");
const speakerEnrollModalEl = document.querySelector("#speaker-enroll-modal");
const speakerEnrollSummaryEl = document.querySelector("#speaker-enroll-summary");
const speakerEnrollProgressEl = document.querySelector("#speaker-enroll-progress");
const speakerEnrollStatusEl = document.querySelector("#speaker-enroll-status");
const speakerEnrollStartEl = document.querySelector("#speaker-enroll-start");
const speakerEnrollRetryEl = document.querySelector("#speaker-enroll-retry");
const speakerEnrollCancelEl = document.querySelector("#speaker-enroll-cancel");
const speakerEnrollCloseEl = document.querySelector("#speaker-enroll-close");
let typingNode = null;
const AUDIO_SEGMENT_MIME_CANDIDATES = ["audio/webm", "audio/ogg", "audio/mp3", "audio/mpeg", "audio/wav"];
let mediaStream = null;
let mediaRecorder = null;
let recorderMimeType = "";
let recorderStartedAt = 0;
let pendingRecorderMode = "";
let speechAudio = null;
let speechAudioUrl = null;
let recognitionRestartTimer = null;
let audioContext = null;
let analyserNode = null;
let analyserData = null;
let mediaStreamSource = null;
let ambientVadTimer = null;
let ambientSegmentStartedAt = 0;
let ambientSpeechDetectedAt = 0;
let ambientLastSpeechAt = 0;
let ambientSpeechActive = false;
let wakeSessionTimeoutTimer = null;
const AMBIENT_VAD = {
  minDecibels: -55,
  minSpeechMs: 220,
  silenceHangoverMs: 900,
  maxSegmentMs: 10000,
  idleSegmentMs: 3200,
  pollMs: 120,
};
const AMBIENT_RETENTION = {
  maxSegments: 6,
  windowSeconds: 5 * 60,
};
const WAKE_SESSION_TIMEOUT_SECONDS = 8;
const WAKE_DETECTOR_BACKEND = "local_asr_phrase_detector";
const SPEAKER_ENROLLMENT_PHRASE = "你好 Hermes，这是我的参考声纹样本。";

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
  const { enabled, wakePending, captureId, chunkCount, status, lastSegmentId, chunks, wakeSession, wakeDetectorEnabled } = state.ambient;
  const lastCapturedAt = chunks.length ? chunks[chunks.length - 1].timestamp : 0;
  const wakePendingLabel = wakeSession && wakeSession.status !== "consumed" ? "有待消费唤醒" : "无待消费唤醒";
  const wakeDetectorBackend = wakeSession?.wake_detector_backend || WAKE_DETECTOR_BACKEND;
  const detectorState = !wakeDetectorEnabled
    ? "wake_detector_disabled"
    : wakePending
      ? "wake_detected_pending_query"
      : enabled
        ? "wake_detector_listening"
        : chunkCount
          ? "wake_consumed"
          : "wake_detector_disabled";
  if (wakePending) {
    ambientModeLabelEl.textContent = wakeSession?.wake_mode === "wake_word" ? "已听到唤醒词，等待问题" : "等待唤醒后的问题";
    ambientStatusEl.textContent = chunkCount
      ? `下一句语音会作为 query 发送，并引用最近 ${chunkCount} 段现场语境。当前状态=${detectorState}；wake detector backend=${wakeDetectorBackend}；超时 ${WAKE_SESSION_TIMEOUT_SECONDS} 秒后会恢复待机。`
      : `下一句语音会作为 query 发送；当前没有最近语境，将按普通对话回答。当前状态=${detectorState}；wake detector backend=${wakeDetectorBackend}。`;
  } else if (enabled) {
    const statusLabels = {
      listening: "待机监听中",
      speech_detected: "检测到现场语音",
      processing: "正在转写当前片段",
      ready_for_wake_context: "片段已加入现场语境",
      failed: "片段处理失败",
      wake_detector_listening: "唤醒词监听中",
      wake_timeout: "唤醒超时，已恢复待机",
    };
    ambientModeLabelEl.textContent = statusLabels[status] || "收音待机中";
    ambientStatusEl.textContent = [
      `已缓存 ${chunkCount} 段现场语境`,
      `当前状态：${statusLabels[status] || "收音待机中"}`,
      `wake detector=${wakeDetectorEnabled ? "on" : "off"}`,
      `wake detector backend=${WAKE_DETECTOR_BACKEND}`,
      `wake detector state=${detectorState}`,
      `capture=${captureId || "准备中"}`,
      lastSegmentId ? `最近片段=${lastSegmentId}` : "最近片段=暂无",
      lastCapturedAt ? `最近采集=${new Date(lastCapturedAt * 1000).toLocaleTimeString()}` : "最近采集=暂无",
      wakePendingLabel,
      "原始音频临时处理后即删除",
    ].join("；");
  } else {
    ambientModeLabelEl.textContent = "收音待机已暂停";
    ambientStatusEl.textContent = chunkCount
      ? `已保留 ${chunkCount} 段本页语境；${wakePendingLabel}；wake detector=${wakeDetectorEnabled ? "on" : "off"}；backend=${WAKE_DETECTOR_BACKEND}；点击“唤醒提问”可在下一句 query 中引用。`
      : `本模式使用本地 ASR 转写音频片段；原始音频临时处理后即删除。wake detector=${wakeDetectorEnabled ? "on" : "off"}；backend=${WAKE_DETECTOR_BACKEND}。`;
  }
  if (ambientStandbyToggleEl) {
    ambientStandbyToggleEl.textContent = enabled ? "暂停待机" : "开始待机";
    ambientStandbyToggleEl.setAttribute("aria-pressed", String(enabled));
  }
  if (ambientWakeButtonEl) {
    ambientWakeButtonEl.setAttribute("aria-pressed", String(wakePending));
  }
}

function setSpeakerEnrollStatus(text, status = "idle") {
  if (!speakerEnrollStatusEl) return;
  speakerEnrollStatusEl.textContent = text;
  speakerEnrollStatusEl.dataset.state = status;
}

function updateSpeakerProfileSummary() {
  if (!speakerEnrollSummaryEl || !speakerEnrollButtonEl) return;
  if (speakerEnrollProgressEl) {
    speakerEnrollProgressEl.textContent = `当前进度：${state.speaker.sampleCount || 0} / ${state.speaker.targetSampleCount || 3}`;
  }
  if (state.speaker.enrolled) {
    const updatedLabel = state.speaker.updatedAt
      ? `最近更新：${new Date(state.speaker.updatedAt * 1000).toLocaleString()}`
      : "已录入";
    const calibratedLabel = state.speaker.calibrationStatus === "calibrated" ? "已完成 3 段校准" : "已录入但未完成校准";
    speakerEnrollSummaryEl.textContent = `当前已录入参考声纹。模型=${state.speaker.speakerModel || "cam++"}；${calibratedLabel}；${updatedLabel}`;
    speakerEnrollButtonEl.textContent = "已录入声纹";
  } else if (state.speaker.sampleCount > 0) {
    speakerEnrollSummaryEl.textContent = `当前已有 ${state.speaker.sampleCount} 段待校准样本，还差 ${Math.max(0, (state.speaker.targetSampleCount || 3) - state.speaker.sampleCount)} 段。`;
    speakerEnrollButtonEl.textContent = "继续录入声纹";
  } else {
    speakerEnrollSummaryEl.textContent = "当前还没有参考声纹。可以先用文字聊天；录入后才会在语音待机里区分用户本人和其他人。";
    speakerEnrollButtonEl.textContent = "声纹录入";
  }
}

function setSpeakerEnrollModalOpen(open) {
  state.speaker.modalOpen = open;
  if (!speakerEnrollModalEl) return;
  speakerEnrollModalEl.hidden = !open;
  if (open) {
    updateSpeakerProfileSummary();
    setSpeakerEnrollStatus(
      state.speaker.sampleCount > 0
        ? `继续录到第 ${Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3)} 段，完成后才会整体替换旧声纹。`
        : state.speaker.enrolled
          ? "当前已有一份参考声纹，重新录入会在完成 3 段校准后整体覆盖旧样本。"
          : "点击开始录入后，说完固定短句即可；暂时不录也可以直接用文字聊天。",
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
  if (typeof MediaRecorder === "undefined") {
    return "当前浏览器不支持本地录音上传，请改用新版 Chrome / Edge / Safari";
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

function pickRecorderMimeType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }
  return AUDIO_SEGMENT_MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function ensureAudioRecorderReady() {
  if (mediaRecorder && mediaStream) return true;
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    setVoiceStatus(getLocalASRUnsupportedMessage(), "error");
    return false;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorderMimeType = pickRecorderMimeType();
    mediaRecorder = recorderMimeType
      ? new MediaRecorder(mediaStream, { mimeType: recorderMimeType })
      : new MediaRecorder(mediaStream);
    audioContext = audioContext || new window.AudioContext();
    mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    analyserNode.smoothingTimeConstant = 0.15;
    analyserData = new Float32Array(analyserNode.fftSize);
    mediaStreamSource.connect(analyserNode);
    mediaRecorder.onstart = () => {
      clearRecognitionRestartTimer();
      clearAmbientVadTimer();
      resetAmbientVadState();
      recorderStartedAt = Date.now();
      state.listening = true;
      document.body.dataset.voice = "listening";
      setVoiceStatus(state.ambient.enabled ? "收音待机中，正在持续收音" : "正在听你说话", "listening");
      if (isAmbientModeActive()) {
        setAmbientRuntimeStatus("listening");
        scheduleAmbientVadLoop();
      }
      stopSpeaking();
    };
    mediaRecorder.onerror = (event) => {
      const message = getLocalASRErrorMessage(event?.error || event);
      setVoiceStatus(message, "error");
      showToast(message);
    };
    mediaRecorder.onstop = async (event) => {
      clearAmbientVadTimer();
      state.listening = false;
      document.body.dataset.voice = "idle";
      const chunks = Array.isArray(event?.target?._chunks) ? event.target._chunks : [];
      event.target._chunks = [];
      const mode = pendingRecorderMode;
      pendingRecorderMode = "";
      if (!chunks.length) {
        if (shouldAutoRestartRecognition()) {
          scheduleRecorderRestart();
          return;
        }
        if (!inputEl.disabled && voiceStatusEl.dataset.state !== "error") {
          setVoiceStatus("等待语音输入");
        }
        return;
      }
      const blob = new Blob(chunks, { type: recorderMimeType || mediaRecorder.mimeType || "audio/webm" });
      try {
        if (mode === "speaker_enroll") {
          setSpeakerEnrollStatus("录音完成，正在提取参考声纹…", "processing");
        } else {
          setVoiceStatus("录音完成，正在调用本地 ASR", "listening");
        }
        if (mode === "ambient") {
          setAmbientRuntimeStatus("processing");
        }
        if (mode === "speaker_enroll") {
          const payload = await enrollSpeakerFromBlob(blob);
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
          if (payload.status === "ok") {
            setSpeakerEnrollStatus("3 段校准完成，后续会按新的用户声纹中心做 user/other/unknown 比对。", "success");
            showToast("参考声纹 3 段校准成功");
          } else {
            setSpeakerEnrollStatus(`第 ${state.speaker.sampleCount} 段已保存，还需要继续录入。`, "success");
            showToast(`已保存第 ${state.speaker.sampleCount} 段声纹样本`);
          }
          return;
        }
        const transcript = await transcribeRecordedAudio(blob);
        if (!transcript) {
          setVoiceStatus("本地 ASR 未识别到有效内容", "error");
          showToast("本地 ASR 未识别到有效内容");
          if (mode === "ambient") {
            setAmbientRuntimeStatus("failed");
          }
        } else if (mode === "wake") {
          await sendWakeQuery(transcript);
        } else if (mode === "ambient") {
          await appendAmbientTranscript(transcript);
        } else {
          await sendMessage(transcript);
        }
      } catch (error) {
        setStatus("error");
        if (mode === "speaker_enroll") {
          setSpeakerEnrollStatus(error.message, "error");
        } else {
          appendMessage("system", error.message);
          setVoiceStatus("语音处理失败", "error");
        }
        if (mode === "ambient") {
          setAmbientRuntimeStatus("failed");
        }
      } finally {
        resetAmbientVadState();
        if (mode === "speaker_enroll") {
          return;
        }
        if (shouldAutoRestartRecognition()) {
          scheduleRecorderRestart();
        } else if (!inputEl.disabled && voiceStatusEl.dataset.state !== "error") {
          setVoiceStatus("等待语音输入");
        }
      }
    };
    return true;
  } catch (error) {
    setVoiceStatus(getLocalASRErrorMessage(error), "error");
    showToast(getLocalASRErrorMessage(error));
    return false;
  }
}

function scheduleRecorderRestart() {
  setVoiceStatus(state.ambient.wakePending ? "等待唤醒后的问题" : "收音待机中，准备继续收音", "listening");
  recognitionRestartTimer = window.setTimeout(() => {
    recognitionRestartTimer = null;
    startRecognitionSession({
      statusText: state.ambient.wakePending ? "等待唤醒后的问题" : "收音待机中，正在恢复收音",
    });
  }, 180);
}

async function transcribeRecordedAudio(blob) {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  const payload = await requestJSON("/api/audio/segment/process", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      source_type: pendingRecorderMode === "wake" ? "wake_query" : pendingRecorderMode === "ambient" ? "ambient_audio" : "chat",
      audio_base64: window.btoa(binary),
      audio_mime_type: blob.type || recorderMimeType || "audio/webm",
      audio_duration_ms: Math.max(1, Date.now() - recorderStartedAt),
    }),
  });
  if (payload.status !== "processed" || !String(payload.transcript || "").trim()) {
    throw new Error(payload.detail || payload.error_type || "本地 ASR 处理失败");
  }
  return String(payload.transcript || "").trim();
}

async function enrollSpeakerFromBlob(blob) {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return requestJSON("/api/speaker/enroll", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      audio_base64: window.btoa(binary),
      audio_mime_type: blob.type || recorderMimeType || "audio/webm",
      audio_duration_ms: Math.max(1, Date.now() - recorderStartedAt),
      enrollment_session_id: state.speaker.enrollmentSessionId || "",
      sample_index: Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3),
      sample_total: state.speaker.targetSampleCount || 3,
      finalize: Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3) >= (state.speaker.targetSampleCount || 3),
    }),
  });
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

function speakWithBrowserFallback(text) {
  const speechText = normalizeSpeechText(text);
  if (!("speechSynthesis" in window) || !speechText) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(speechText);
  utterance.lang = "zh-CN";
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

function releaseSpeechAudio() {
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

function clearRecognitionRestartTimer() {
  if (recognitionRestartTimer) {
    window.clearTimeout(recognitionRestartTimer);
    recognitionRestartTimer = null;
  }
}

function clearAmbientVadTimer() {
  if (ambientVadTimer) {
    window.clearTimeout(ambientVadTimer);
    ambientVadTimer = null;
  }
}

function clearWakeSessionTimeout() {
  if (wakeSessionTimeoutTimer) {
    window.clearTimeout(wakeSessionTimeoutTimer);
    wakeSessionTimeoutTimer = null;
  }
}

function scheduleWakeSessionTimeout() {
  clearWakeSessionTimeout();
  if (!state.ambient.wakePending || !state.ambient.wakeSession) return;
  wakeSessionTimeoutTimer = window.setTimeout(() => {
    const session = state.ambient.wakeSession;
    if (!session || session.status === "consumed") return;
    state.ambient.wakePending = false;
    state.ambient.enabled = true;
    state.ambient.status = "wake_timeout";
    state.ambient.wakeSession = {
      ...session,
      status: "expired",
      expired: true,
    };
    pruneAmbientContext();
    updateAmbientStatus();
    showToast("唤醒后没有收到有效问题，已恢复待机监听");
    if (!state.listening) {
      startRecognitionSession({ statusText: "唤醒超时，正在恢复收音", mode: "ambient" });
    }
  }, WAKE_SESSION_TIMEOUT_SECONDS * 1000);
}

function stripWakeWordPrefix(text) {
  let normalized = String(text || "").trim();
  if (!normalized) return "";
  state.ambient.wakeWordPhrases.forEach((phrase) => {
    const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    normalized = normalized.replace(new RegExp(`^${escaped}[,，!！\\s:：-]*`, "i"), "").trim();
  });
  return normalized;
}

function detectWakeWord(text) {
  const normalized = normalizeSpeechText(text).toLowerCase();
  if (!normalized) return false;
  return state.ambient.wakeWordPhrases.some((phrase) => normalized.includes(String(phrase || "").toLowerCase()));
}

function detectWakeWordFromTranscriptSegment(text) {
  const transcript = String(text || "").trim();
  if (!transcript) {
    return {
      detected: false,
      backend: WAKE_DETECTOR_BACKEND,
      reason: "empty_transcript_segment",
      wakeQueryText: "",
    };
  }
  if (!state.ambient.wakeDetectorEnabled) {
    return {
      detected: false,
      backend: WAKE_DETECTOR_BACKEND,
      reason: "wake_detector_disabled",
      wakeQueryText: transcript,
    };
  }
  const detected = detectWakeWord(transcript);
  return {
    detected,
    backend: WAKE_DETECTOR_BACKEND,
    reason: detected ? "wake_word_phrase_detected" : "wake_word_not_detected",
    wakeQueryText: detected ? stripWakeWordPrefix(transcript) : transcript,
  };
}

function beginWakeSession(wakeMode, options = {}) {
  if (state.ambient.wakeSession && state.ambient.wakeSession.status !== "consumed") {
    return false;
  }
  pruneAmbientContext();
  const detectedAt = typeof options.wakeDetectedAt === "number" ? options.wakeDetectedAt : Date.now() / 1000;
  state.ambient.wakePending = true;
  state.ambient.enabled = false;
  state.ambient.status = "idle";
  state.ambient.wakeSession = {
    ambient_capture_id: state.ambient.captureId || "",
    wake_detected_at: detectedAt,
    pre_wake_segment_ids: state.ambient.chunks.map((item) => String(item.chunkId || "")).filter(Boolean),
    post_wake_query_segment_ids: [],
    wake_query_text: "",
    wake_mode: wakeMode,
    wake_detector_backend: String(options.wakeDetectorBackend || ""),
    status: "pending_query",
    expired: false,
    timeout_seconds: WAKE_SESSION_TIMEOUT_SECONDS,
  };
  updateAmbientStatus();
  scheduleWakeSessionTimeout();
  return true;
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
      clearWakeSessionTimeout();
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

function resetAmbientVadState() {
  ambientSegmentStartedAt = 0;
  ambientSpeechDetectedAt = 0;
  ambientLastSpeechAt = 0;
  ambientSpeechActive = false;
}

function isAmbientModeActive(mode = pendingRecorderMode) {
  return mode === "ambient";
}

function setAmbientRuntimeStatus(status) {
  state.ambient.status = status;
  updateAmbientStatus();
}

function sampleAmbientLevel() {
  if (!analyserNode || !analyserData) return -100;
  analyserNode.getFloatTimeDomainData(analyserData);
  let sumSquares = 0;
  for (let index = 0; index < analyserData.length; index += 1) {
    const sample = analyserData[index];
    sumSquares += sample * sample;
  }
  const rms = Math.sqrt(sumSquares / analyserData.length);
  if (!rms) return -100;
  return 20 * Math.log10(rms);
}

function scheduleAmbientVadLoop() {
  clearAmbientVadTimer();
  if (!state.listening || !isAmbientModeActive()) return;
  ambientVadTimer = window.setTimeout(runAmbientVadLoop, AMBIENT_VAD.pollMs);
}

function runAmbientVadLoop() {
  if (!state.listening || !isAmbientModeActive()) return;
  const now = Date.now();
  if (!ambientSegmentStartedAt) {
    ambientSegmentStartedAt = now;
    setAmbientRuntimeStatus("listening");
  }
  const levelDb = sampleAmbientLevel();
  const isSpeechFrame = levelDb >= AMBIENT_VAD.minDecibels;
  if (isSpeechFrame) {
    ambientLastSpeechAt = now;
    if (!ambientSpeechDetectedAt) {
      ambientSpeechDetectedAt = now;
    }
    if (!ambientSpeechActive && now - ambientSpeechDetectedAt >= AMBIENT_VAD.minSpeechMs) {
      ambientSpeechActive = true;
      setAmbientRuntimeStatus("speech_detected");
    }
  } else if (!ambientSpeechActive) {
    ambientSpeechDetectedAt = 0;
  }

  const segmentAgeMs = now - ambientSegmentStartedAt;
  const speechSilenceMs = ambientLastSpeechAt ? now - ambientLastSpeechAt : 0;
  const shouldFlushSpeech = ambientSpeechActive && speechSilenceMs >= AMBIENT_VAD.silenceHangoverMs;
  const shouldFlushLongSegment = ambientSpeechActive && segmentAgeMs >= AMBIENT_VAD.maxSegmentMs;
  const shouldDropIdleSegment = !ambientSpeechActive && segmentAgeMs >= AMBIENT_VAD.idleSegmentMs;

  if (shouldFlushSpeech || shouldFlushLongSegment || shouldDropIdleSegment) {
    setAmbientRuntimeStatus(ambientSpeechActive ? "processing" : "listening");
    mediaRecorder.stop();
    return;
  }
  scheduleAmbientVadLoop();
}

function shouldAutoRestartRecognition() {
  return Boolean(mediaRecorder && !inputEl.disabled && (state.ambient.enabled || state.ambient.wakePending));
}

function startRecognitionSession({ statusText = "正在启动本地 ASR 录音", mode = "" } = {}) {
  if (!mediaRecorder || inputEl.disabled) return false;
  if (state.listening) return true;
  clearRecognitionRestartTimer();
  try {
    setVoiceStatus(statusText, "listening");
    mediaRecorder._chunks = [];
    pendingRecorderMode = mode || (state.ambient.wakePending ? "wake" : state.ambient.enabled ? "ambient" : "chat");
    resetAmbientVadState();
    mediaRecorder.start();
    return true;
  } catch (error) {
    setVoiceStatus("本地 ASR 录音未能启动，请稍后重试", "error");
    showToast("本地 ASR 录音未能启动，请稍后重试");
    return false;
  }
}

async function speak(text) {
  if (!state.voiceEnabled || !text) return;
  const speechText = normalizeSpeechText(text);
  if (!speechText) return;
  stopSpeaking();
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
    speechAudioUrl = URL.createObjectURL(audioBlob);
    speechAudio = new Audio(speechAudioUrl);
    speechAudio.addEventListener("ended", releaseSpeechAudio, { once: true });
    speechAudio.addEventListener("error", releaseSpeechAudio, { once: true });
    await speechAudio.play();
  } catch (error) {
    releaseSpeechAudio();
    console.warn("Backend TTS unavailable; falling back to browser speech.", error);
    speakWithBrowserFallback(speechText);
  }
}

function stopSpeaking() {
  releaseSpeechAudio();
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
  return "没有拿到当前位置，本轮不会猜测地点";
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
        const status = error.code === error.PERMISSION_DENIED ? "denied" : "error";
        resolve(locationUnavailable(status, error.message || "geolocation_error"));
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
  const ready = await ensureAudioRecorderReady();
  if (!ready) return;
  if (mediaRecorder) {
    mediaRecorder.ondataavailable = (event) => {
      if (!event.data || !event.data.size) return;
      if (!Array.isArray(mediaRecorder._chunks)) {
        mediaRecorder._chunks = [];
      }
      mediaRecorder._chunks.push(event.data);
    };
  }
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
    throw new Error(payload.detail || `HTTP ${res.status}`);
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

async function ensureAmbientCapture() {
  if (state.ambient.captureId) return state.ambient.captureId;
  const result = await requestJSON("/api/capture/start", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      source: "ambient_audio_text",
      context: "按钮模拟唤醒 MVP：本地 ASR 转写后的音频片段，原始音频临时处理后即删除",
    }),
  });
  state.ambient.captureId = result.capture_id;
  state.ambient.status = "listening";
  pruneAmbientContext();
  updateAmbientStatus();
  return state.ambient.captureId;
}

async function appendAmbientTranscript(text) {
  const transcript = String(text || "").trim();
  if (!transcript) return null;
  const wakeDetection = detectWakeWordFromTranscriptSegment(transcript);
  if (!state.ambient.wakePending && wakeDetection.detected) {
    const began = beginWakeSession("wake_word", {
      wakeDetectedAt: Date.now() / 1000,
      wakeDetectorBackend: wakeDetection.backend,
    });
    const wakeQueryText = wakeDetection.wakeQueryText;
    showToast("检测到唤醒词，正在等待你的问题");
    if (began && wakeQueryText) {
      await sendWakeQuery(wakeQueryText);
    }
    return {
      wake_detected: true,
      transcript,
      wake_detector_backend: wakeDetection.backend,
      wake_detector_reason: wakeDetection.reason,
    };
  }
  const captureId = await ensureAmbientCapture();
  const result = await requestJSON("/api/capture/append", {
    method: "POST",
    body: JSON.stringify({
      user_id: state.userId,
      capture_id: captureId,
      text: transcript,
      metadata: {
        source_type: "ambient_audio",
        audio_retention: "discarded_after_processing",
        emotion_enabled: false,
        wake_detector_backend: wakeDetection.backend,
        wake_detector_reason: wakeDetection.reason,
      },
    }),
  });
  state.ambient.chunkCount = result.chunk_count || state.ambient.chunkCount + 1;
  state.ambient.lastSegmentId = String(result.chunk_id || "");
  state.ambient.status = "ready_for_wake_context";
  state.ambient.chunks.push({
    chunkId: String(result.chunk_id || ""),
    text: transcript,
    timestamp: Date.now() / 1000,
  });
  pruneAmbientContext();
  updateAmbientStatus();
  showToast(`已加入最近语境：${state.ambient.chunkCount} 段`);
  return result;
}

async function sendWakeQuery(message) {
  const captureId = state.ambient.captureId || "";
  const wakeSession = state.ambient.wakeSession
    ? {
        ...state.ambient.wakeSession,
        post_wake_query_segment_ids: [String(state.ambient.lastSegmentId || "")].filter(Boolean),
        wake_query_text: String(message || "").trim(),
        status: "query_captured",
      }
    : null;
  clearWakeSessionTimeout();
  state.ambient.wakePending = false;
  state.ambient.status = state.ambient.enabled ? "listening" : "idle";
  state.ambient.wakeSession = wakeSession
    ? {
        ...wakeSession,
        status: "consumed",
      }
    : null;
  pruneAmbientContext();
  updateAmbientStatus();
  await sendMessage(message, { ambientCaptureId: captureId, wakeSession });
}

async function sendMessage(message, options = {}) {
  appendMessage("user", message);
  setStatus("thinking");
  setVoiceStatus("已收到，正在生成回复");
  sendButtonEl.disabled = true;
  inputEl.disabled = true;
  ambientStandbyToggleEl.disabled = true;
  ambientWakeButtonEl.disabled = true;
  ambientClearButtonEl.disabled = true;
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
    state.sessionId = payload.session_id;
    hideTyping();
    const reply = payload.reply || "我收到了，但这次没有生成文字回复。";
    appendMessage("assistant", reply);
    setVoiceStatus("回复已生成");
    speak(reply);
    renderDebug(payload.debug);
    if (payload.recalled_memories?.length) {
      showToast(`本次召回 ${payload.recalled_memories.length} 条记忆`);
    }
    if (payload.saved_memories?.length) {
      showToast(`已自动保存 ${payload.saved_memories.length} 条新记忆`);
    }
    await loadMemories();
    const memoryJobId = payload.debug?.memory_processing?.job_id;
    if (memoryJobId) {
      pollMemoryJob(memoryJobId).catch((error) => showToast(error.message));
    } else if (payload.debug?.memory_processing?.status === "pending") {
      window.setTimeout(() => {
        loadMemories().catch((error) => showToast(error.message));
      }, 1600);
    }
  } finally {
    hideTyping();
    sendButtonEl.disabled = false;
    inputEl.disabled = false;
    ambientStandbyToggleEl.disabled = false;
    ambientWakeButtonEl.disabled = false;
    ambientClearButtonEl.disabled = false;
    locationRefreshEl.disabled = false;
    inputEl.focus();
    setStatus("ready");
    updateAmbientStatus();
  }
}

async function loadMemories() {
  const payload = await requestJSON(`/api/memories?user_id=${encodeURIComponent(state.userId)}`);
  memoryListEl.innerHTML = "";
  const memories = payload.memories || [];
  const documents = payload.documents || [];
  const profileCount = memories.filter((m) => m.kind === "profile").length;
  const assistantPreferenceCount = memories.filter((m) => m.kind === "assistant_preference").length;
  const eventCount = memories.filter((m) => !["profile", "assistant_preference"].includes(m.kind)).length;
  memoryCountEl.textContent = `${profileCount} 画像 · ${assistantPreferenceCount} 助手偏好 · ${eventCount} 事件 · ${documents.length} 文档`;
  if (!memories.length && !documents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-memory";
    empty.textContent = "还没有记忆或文档。聊天时说“记住...”，或上传一份 Markdown 文档。";
    memoryListEl.appendChild(empty);
    return;
  }
  const rows = [
    ...memories.map((memory) => ({ type: "memory", created_at: memory.created_at || 0, item: memory })),
    ...documents.map((documentRecord) => ({ type: "document", created_at: documentRecord.created_at || 0, item: documentRecord })),
  ].sort((a, b) => b.created_at - a.created_at);
  for (const row of rows) {
    memoryListEl.appendChild(row.type === "document" ? renderDocumentCard(row.item) : renderMemoryCard(row.item));
  }
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

async function startSpeakerEnrollmentFlow() {
  const ready = await ensureAudioRecorderReady();
  if (!ready) {
    showToast(getLocalASRUnsupportedMessage());
    return;
  }
  clearRecognitionRestartTimer();
  clearWakeSessionTimeout();
  state.ambient.wakePending = false;
  state.ambient.wakeSession = null;
  state.ambient.enabled = false;
  state.ambient.status = "idle";
  if (!state.speaker.enrollmentSessionId) {
    state.speaker.enrollmentSessionId = `speaker_enroll_${Date.now()}`;
  }
  updateAmbientStatus();
  setSpeakerEnrollStatus(`录音中，请朗读第 ${Math.min((state.speaker.sampleCount || 0) + 1, state.speaker.targetSampleCount || 3)} 段固定短句…`, "recording");
  if (state.listening) {
    pendingRecorderMode = "speaker_enroll";
    mediaRecorder.stop();
    return;
  }
  startRecognitionSession({ statusText: "声纹录入中，正在录音", mode: "speaker_enroll" });
}

function renderMemoryCard(memory) {
  const card = document.createElement("div");
  card.className = "memory-card";
  card.dataset.kind = memory.kind || "event";

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
  if (!state.voiceEnabled) {
    stopSpeaking();
  }
});

ambientStandbyToggleEl.addEventListener("click", async () => {
  const ready = await ensureAudioRecorderReady();
  if (!ready) {
    showToast(getLocalASRUnsupportedMessage());
    return;
  }
  clearRecognitionRestartTimer();
  state.ambient.enabled = !state.ambient.enabled;
  if (state.ambient.enabled) {
    try {
      await ensureAmbientCapture();
      pruneAmbientContext();
      showToast("收音待机已开启，现在会直接开始持续收音");
      startRecognitionSession({ statusText: "收音待机中，正在启动收音" });
    } catch (error) {
      state.ambient.enabled = false;
      state.ambient.status = "idle";
      showToast(error.message);
    }
  } else {
    clearWakeSessionTimeout();
    state.ambient.wakePending = false;
    state.ambient.status = "idle";
    state.ambient.wakeSession = null;
    pruneAmbientContext();
    if (state.listening) {
      mediaRecorder.stop();
    }
    showToast("收音待机已暂停");
  }
  updateAmbientStatus();
});

ambientWakeButtonEl.addEventListener("click", () => {
  ensureAudioRecorderReady().then((ready) => {
    if (!ready) {
      showToast(getLocalASRUnsupportedMessage());
      return;
    }
    clearRecognitionRestartTimer();
    beginWakeSession("button");
    showToast(
      state.ambient.captureId && state.ambient.chunkCount > 0
        ? "下一句语音会作为问题发送，并引用最近语境"
        : "下一句语音会作为问题发送；当前没有最近语境，将按普通对话回答",
    );
    if (state.listening) {
      mediaRecorder.stop();
    } else {
      startRecognitionSession({ statusText: "正在启动本地 ASR 唤醒提问" });
    }
  }).catch((error) => {
    showToast(error.message);
  });
});

ambientClearButtonEl.addEventListener("click", () => {
  clearRecognitionRestartTimer();
  clearWakeSessionTimeout();
  state.ambient.captureId = null;
  state.ambient.chunkCount = 0;
  state.ambient.wakePending = false;
  state.ambient.enabled = false;
  state.ambient.status = "idle";
  state.ambient.lastSegmentId = "";
  state.ambient.chunks = [];
  state.ambient.wakeSession = null;
  if (state.listening) {
    mediaRecorder.stop();
  }
  updateAmbientStatus();
  showToast("已清空本页最近语境引用");
});

speakerEnrollButtonEl?.addEventListener("click", () => {
  setSpeakerEnrollModalOpen(true);
});

speakerEnrollCloseEl?.addEventListener("click", () => {
  setSpeakerEnrollModalOpen(false);
});

speakerEnrollCancelEl?.addEventListener("click", async () => {
  if (state.speaker.enrollmentSessionId) {
    try {
      await requestJSON(`/api/speaker/profile?user_id=${encodeURIComponent(state.userId)}&enrollment_session_id=${encodeURIComponent(state.speaker.enrollmentSessionId)}`, {
        method: "DELETE",
      });
      state.speaker.enrollmentSessionId = "";
      await loadSpeakerProfile();
    } catch (error) {
      showToast(error.message);
    }
  }
  setSpeakerEnrollModalOpen(false);
});

speakerEnrollStartEl?.addEventListener("click", () => {
  startSpeakerEnrollmentFlow().catch((error) => {
    setSpeakerEnrollStatus(error.message, "error");
    showToast(error.message);
  });
});

speakerEnrollRetryEl?.addEventListener("click", () => {
  startSpeakerEnrollmentFlow().catch((error) => {
    setSpeakerEnrollStatus(error.message, "error");
    showToast(error.message);
  });
});

memoryFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = memoryInputEl.value.trim();
  if (!content) return;
  memoryInputEl.value = "";
  try {
    await requestJSON("/api/memories", {
      method: "POST",
      body: JSON.stringify({
        content,
        kind: memoryKindEl.value,
        user_id: state.userId,
        tags: ["manual"],
      }),
    });
    await loadMemories();
    showToast("记忆已保存");
  } catch (error) {
    showToast(error.message);
  }
});

refreshMemoryEl.addEventListener("click", () => {
  loadMemories()
    .then(() => showToast("记忆已刷新"))
    .catch((error) => showToast(error.message));
});

locationRefreshEl.addEventListener("click", async () => {
  locationRefreshEl.disabled = true;
  locationRefreshEl.textContent = "定位中";
  try {
    state.location = await getCurrentLocation();
    if (state.location.status === "available") {
      locationRefreshEl.textContent = "已定位";
      showToast(`已刷新定位，精度约 ${Math.round(state.location.accuracy || 0)} 米`);
    } else {
      locationRefreshEl.textContent = "定位";
      showToast(locationToastMessage(state.location));
    }
  } finally {
    locationRefreshEl.disabled = false;
  }
});

memoryToggleEl.addEventListener("click", () => {
  memoryPaneEl.classList.toggle("open");
});

debugToggleEl.addEventListener("click", () => {
  setDebugOpen(!document.body.classList.contains("debug-open"));
});

closeDebugEl.addEventListener("click", () => {
  setDebugOpen(false);
});

debugBackdropEl.addEventListener("click", () => {
  setDebugOpen(false);
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

setupButtonTooltips();
setupLocalASR().catch((error) => {
  setVoiceStatus(error.message, "error");
  showToast(error.message);
});
appendMessage("assistant", "我在。你可以直接说话，也可以用文字输入。");
updateSpeakerProfileSummary();
loadSpeakerProfile().catch((error) => showToast(error.message));
loadMemories().catch((error) => showToast(error.message));
