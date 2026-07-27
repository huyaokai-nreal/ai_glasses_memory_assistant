package com.aiglasses.memoryassistant.demo

import org.json.JSONObject
import android.os.Debug

data class NativeAudioSnapshot(
    val state: String = "idle",
    val captureId: String = "",
    val startedAtMillis: Long = 0,
    val capturedSamples: Long = 0,
    val audioRmsDbfs: Double = -120.0,
    val audioPeakDbfs: Double = -120.0,
    val audioLevelAtMillis: Long = 0,
    val inputDeviceName: String = "",
    val inputDeviceType: String = "",
    val inputDeviceSource: String = "",
    val vadSegmentCount: Long = 0,
    val ambientFinalCount: Long = 0,
    val speechRejectedCount: Long = 0,
    val lastFinalAtMillis: Long = 0,
    val networkOnline: Boolean = false,
    val latestPartial: String = "",
    val partialSequence: Long = 0L,
    val interactionState: String = "ambient_listening",
    val wakeKeyword: String = "",
    val wakeDeadlineMillis: Long = 0L,
    val finalQuery: String = "",
    val finalQueryEventId: String = "",
    val finalQuerySequence: Long = 0L,
    val inferenceQueueDepth: Int = 0,
    val enrollmentState: String = "idle",
    val enrollmentSessionId: String = "",
    val enrollmentSampleCount: Int = 0,
    val enrollmentSampleTotal: Int = 3,
    val lastStoppedCaptureId: String = "",
    val lastStopStatus: String = "",
    val lastStopMemoryJobId: String = "",
    val lastStopMemoryJobStatus: String = "",
    val lastStopError: String = "",
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
        .put("audio_rms_dbfs", audioRmsDbfs)
        .put("audio_peak_dbfs", audioPeakDbfs)
        .put("audio_level_at_ms", audioLevelAtMillis)
        .put("input_device_name", inputDeviceName)
        .put("input_device_type", inputDeviceType)
        .put("input_device_source", inputDeviceSource)
        .put("vad_segment_count", vadSegmentCount)
        .put("ambient_final_count", ambientFinalCount)
        .put("speech_rejected_count", speechRejectedCount)
        .put("last_final_at_ms", lastFinalAtMillis)
        .put("sample_rate", AudioRecorder.SAMPLE_RATE)
        .put("channels", 1)
        .put("encoding", "pcm16")
        .put("network_online", networkOnline)
        .put("latest_partial", latestPartial)
        .put("partial_sequence", partialSequence)
        .put("interaction_state", interactionState)
        .put("wake_keyword", wakeKeyword)
        .put("wake_timeout_remaining_ms", (wakeDeadlineMillis - android.os.SystemClock.elapsedRealtime()).coerceAtLeast(0L))
        .put("final_query", finalQuery)
        .put("final_query_event_id", finalQueryEventId)
        .put("final_query_sequence", finalQuerySequence)
        .put("inference_queue_depth", inferenceQueueDepth)
        .put("enrollment_state", enrollmentState)
        .put("enrollment_session_id", enrollmentSessionId)
        .put("enrollment_sample_count", enrollmentSampleCount)
        .put("enrollment_sample_total", enrollmentSampleTotal)
        .put("last_stopped_capture_id", lastStoppedCaptureId)
        .put("last_stop_status", lastStopStatus)
        .put("last_stop_memory_job_id", lastStopMemoryJobId)
        .put("last_stop_memory_job_status", lastStopMemoryJobStatus)
        .put("last_stop_error", lastStopError)
        .put("model_state", ModelPackState.snapshot().state)
        .put("model_version", ModelPackState.snapshot().version)
        .put("model_self_test", ModelSelfTestState.snapshot().toJson())
        .put("transcription_ready", ModelPackState.snapshot().state == "ready")
        .put("pss_kb", Debug.getPss())
        .put("last_error", lastError)

    fun toUiJson(elapsedRealtimeMillis: Long = android.os.SystemClock.elapsedRealtime()): JSONObject = JSONObject()
        .put("state", state)
        .put("running", running)
        .put("capture_id", captureId)
        .put("input_device_name", inputDeviceName)
        .put("input_device_type", inputDeviceType)
        .put("input_device_source", inputDeviceSource)
        .put("ambient_final_count", ambientFinalCount)
        .put("latest_partial", latestPartial)
        .put("partial_sequence", partialSequence)
        .put("interaction_state", interactionState)
        .put("wake_keyword", wakeKeyword)
        .put("wake_timeout_remaining_ms", (wakeDeadlineMillis - elapsedRealtimeMillis).coerceAtLeast(0L))
        .put("final_query", finalQuery)
        .put("final_query_event_id", finalQueryEventId)
        .put("final_query_sequence", finalQuerySequence)
        .put("last_error", lastError)
}

object NativeAudioState {
    private val lock = Any()
    private var snapshot = NativeAudioSnapshot()

    fun snapshot(): NativeAudioSnapshot = synchronized(lock) { snapshot }

    fun snapshotJson(): String = snapshot().toJson().toString()

    fun markStarting() = update {
        it.copy(
            state = "starting",
            captureId = "",
            startedAtMillis = System.currentTimeMillis(),
            capturedSamples = 0,
            audioRmsDbfs = -120.0,
            audioPeakDbfs = -120.0,
            audioLevelAtMillis = 0,
            inputDeviceName = "",
            inputDeviceType = "",
            inputDeviceSource = "",
            vadSegmentCount = 0,
            ambientFinalCount = 0,
            speechRejectedCount = 0,
            lastFinalAtMillis = 0,
            lastError = "",
            interactionState = "ambient_listening",
            wakeKeyword = "",
            wakeDeadlineMillis = 0L,
            finalQuery = "",
            finalQueryEventId = "",
            lastStoppedCaptureId = "",
            lastStopStatus = "",
            lastStopMemoryJobId = "",
            lastStopMemoryJobStatus = "",
            lastStopError = "",
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

    fun markIdle() = update {
        NativeAudioSnapshot(
            captureId = it.captureId,
            capturedSamples = it.capturedSamples,
            audioRmsDbfs = it.audioRmsDbfs,
            audioPeakDbfs = it.audioPeakDbfs,
            audioLevelAtMillis = it.audioLevelAtMillis,
            inputDeviceName = it.inputDeviceName,
            inputDeviceType = it.inputDeviceType,
            inputDeviceSource = it.inputDeviceSource,
            vadSegmentCount = it.vadSegmentCount,
            ambientFinalCount = it.ambientFinalCount,
            speechRejectedCount = it.speechRejectedCount,
            lastFinalAtMillis = it.lastFinalAtMillis,
            networkOnline = it.networkOnline,
            enrollmentState = it.enrollmentState,
            enrollmentSessionId = it.enrollmentSessionId,
            enrollmentSampleCount = it.enrollmentSampleCount,
            enrollmentSampleTotal = it.enrollmentSampleTotal,
            lastStoppedCaptureId = it.lastStoppedCaptureId,
            lastStopStatus = it.lastStopStatus,
            lastStopMemoryJobId = it.lastStopMemoryJobId,
            lastStopMemoryJobStatus = it.lastStopMemoryJobStatus,
            lastStopError = it.lastStopError,
            lastError = it.lastError,
        )
    }

    fun markCaptureStopped(captureId: String, result: JSONObject) = update {
        val memoryJob = result.optJSONObject("import_result")?.optJSONObject("memory_job")
        it.copy(
            lastStoppedCaptureId = captureId.take(120),
            lastStopStatus = "completed",
            lastStopMemoryJobId = memoryJob?.optString("job_id").orEmpty().take(120),
            lastStopMemoryJobStatus = memoryJob?.optString("status").orEmpty().take(40),
            lastStopError = "",
        )
    }

    fun markCaptureStopFailed(captureId: String, message: String) = update {
        it.copy(
            lastStoppedCaptureId = captureId.take(120),
            lastStopStatus = "failed",
            lastStopMemoryJobId = "",
            lastStopMemoryJobStatus = "",
            lastStopError = message.take(500),
            lastError = message.take(500),
        )
    }

    fun markError(message: String) = update {
        it.copy(state = "error", lastError = message.take(500))
    }

    fun addCapturedSamples(count: Int) = update {
        it.copy(capturedSamples = it.capturedSamples + count.coerceAtLeast(0))
    }

    fun markAudioLevel(level: AudioLevel, measuredAtMillis: Long = System.currentTimeMillis()) = update {
        it.copy(
            audioRmsDbfs = level.rmsDbfs,
            audioPeakDbfs = level.peakDbfs,
            audioLevelAtMillis = measuredAtMillis,
        )
    }

    fun markInputRoute(route: AudioInputRoute) = update {
        it.copy(
            inputDeviceName = route.label.take(160),
            inputDeviceType = route.typeLabel.take(80),
            inputDeviceSource = route.source.wireName,
        )
    }

    fun addVadSegments(count: Int) = update {
        it.copy(vadSegmentCount = it.vadSegmentCount + count.coerceAtLeast(0))
    }

    fun markFinal(ambient: Boolean, rejected: Boolean, completedAtMillis: Long = System.currentTimeMillis()) = update {
        it.copy(
            ambientFinalCount = it.ambientFinalCount + if (ambient && !rejected) 1 else 0,
            speechRejectedCount = it.speechRejectedCount + if (rejected) 1 else 0,
            lastFinalAtMillis = if (rejected) it.lastFinalAtMillis else completedAtMillis,
        )
    }

    fun setNetworkOnline(online: Boolean) = update { it.copy(networkOnline = online) }

    fun markPartial(text: String) = update {
        it.copy(latestPartial = text.take(1_000), partialSequence = it.partialSequence + 1)
    }

    fun clearPartial() = update {
        if (it.latestPartial.isBlank()) it else it.copy(latestPartial = "", partialSequence = it.partialSequence + 1)
    }

    fun markInteraction(state: String, keyword: String = "", deadlineMillis: Long = 0L) = update {
        it.copy(
            interactionState = state,
            wakeKeyword = keyword.take(120),
            wakeDeadlineMillis = deadlineMillis.coerceAtLeast(0L),
        )
    }

    fun markFinalQuery(eventId: String, text: String) = update {
        it.copy(
            interactionState = "query_submitted",
            wakeDeadlineMillis = 0L,
            finalQuery = text.take(2_000),
            finalQueryEventId = eventId.take(120),
            finalQuerySequence = it.finalQuerySequence + 1,
        )
    }

    fun markQuerySubmissionFailed(eventId: String, message: String) = update {
        if (it.finalQueryEventId != eventId) it else it.copy(
            interactionState = "ambient_listening",
            wakeKeyword = "",
            wakeDeadlineMillis = 0L,
            lastError = message.take(500),
        )
    }

    fun setInferenceQueueDepth(depth: Int) = update { it.copy(inferenceQueueDepth = depth.coerceAtLeast(0)) }

    fun markEnrollment(
        state: String,
        sessionId: String,
        sampleCount: Int,
        sampleTotal: Int = 3,
        error: String = "",
    ) = update {
        it.copy(
            enrollmentState = state,
            enrollmentSessionId = sessionId.take(120),
            enrollmentSampleCount = sampleCount.coerceAtLeast(0),
            enrollmentSampleTotal = sampleTotal.coerceAtLeast(1),
            lastError = error.take(500),
        )
    }

    fun clearEnrollment(state: String = "idle") = update {
        it.copy(
            enrollmentState = state,
            enrollmentSessionId = "",
            enrollmentSampleCount = 0,
            enrollmentSampleTotal = 3,
            lastError = "",
        )
    }

    private inline fun update(block: (NativeAudioSnapshot) -> NativeAudioSnapshot) {
        synchronized(lock) { snapshot = block(snapshot) }
    }
}
