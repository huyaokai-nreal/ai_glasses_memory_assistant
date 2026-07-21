package com.aiglasses.memoryassistant.demo

import org.json.JSONObject

data class NativeAudioSnapshot(
    val state: String = "idle",
    val captureId: String = "",
    val startedAtMillis: Long = 0,
    val capturedSamples: Long = 0,
    val networkOnline: Boolean = false,
    val lastError: String = "",
) {
    val running: Boolean
        get() = state in setOf("starting", "recording", "paused_tts", "stopping")

    fun toJson(nowMillis: Long = System.currentTimeMillis()): JSONObject = JSONObject()
        .put("platform", "android")
        .put("state", state)
        .put("running", running)
        .put("capture_id", captureId)
        .put("started_at_ms", startedAtMillis)
        .put("duration_seconds", if (startedAtMillis > 0) (nowMillis - startedAtMillis).coerceAtLeast(0) / 1000 else 0)
        .put("captured_samples", capturedSamples)
        .put("sample_rate", AudioRecorder.SAMPLE_RATE)
        .put("channels", 1)
        .put("encoding", "pcm16")
        .put("network_online", networkOnline)
        .put("model_state", ModelPackState.snapshot().state)
        .put("model_version", ModelPackState.snapshot().version)
        .put("transcription_ready", ModelPackState.snapshot().state == "ready")
        .put("last_error", lastError)
}

object NativeAudioState {
    private val lock = Any()
    private var snapshot = NativeAudioSnapshot()

    fun snapshot(): NativeAudioSnapshot = synchronized(lock) { snapshot }

    fun snapshotJson(): String = snapshot().toJson().toString()

    fun markStarting() = update {
        NativeAudioSnapshot(
            state = "starting",
            startedAtMillis = System.currentTimeMillis(),
            networkOnline = it.networkOnline,
        )
    }

    fun markPermissionPending() = update {
        it.copy(state = "permission_pending", lastError = "")
    }

    fun markRecording(captureId: String) = update {
        it.copy(state = "recording", captureId = captureId, lastError = "")
    }

    fun markPausedForTts() = update {
        if (it.running) it.copy(state = "paused_tts") else it
    }

    fun markResumed() = update {
        if (it.state == "paused_tts") it.copy(state = "recording") else it
    }

    fun markStopping() = update {
        if (it.running) it.copy(state = "stopping") else it
    }

    fun markIdle() = update { NativeAudioSnapshot(networkOnline = it.networkOnline) }

    fun markError(message: String) = update {
        it.copy(state = "error", lastError = message.take(500))
    }

    fun addCapturedSamples(count: Int) = update {
        it.copy(capturedSamples = it.capturedSamples + count.coerceAtLeast(0))
    }

    fun setNetworkOnline(online: Boolean) = update { it.copy(networkOnline = online) }

    private inline fun update(block: (NativeAudioSnapshot) -> NativeAudioSnapshot) {
        synchronized(lock) { snapshot = block(snapshot) }
    }
}
