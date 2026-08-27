package com.aiglasses.memoryassistant.demo

import android.webkit.JavascriptInterface

class NativeAppBridge(private val activity: MainActivity) {
    @JavascriptInterface
    fun platform(): String = "android"

    @JavascriptInterface
    fun ownerId(): String = org.json.JSONObject()
        .put("owner_id", SecureSettings(activity).ownerId())
        .toString()

    @JavascriptInterface
    fun audioStatus(): String {
        val status = org.json.JSONObject(NativeAudioState.snapshotJson())
        val ownerId = SecureSettings(activity).ownerId()
        runCatching { PythonRuntime.queueStatus(ownerId) }
            .getOrNull()
            ?.optJSONObject("summary")
            ?.let { status.put("device_event_queue", it) }
        runCatching { PythonRuntime.captureStatus(ownerId, status.optString("capture_id")) }
            .getOrNull()
            ?.let { status.put("ambient_context", it) }
        runCatching { PythonRuntime.setDeviceState(status) }
        return status.toString()
    }

    @JavascriptInterface
    fun audioUiStatus(): String = NativeAudioState.snapshot().toUiJson().toString()

    @JavascriptInterface
    fun consumeCompletedReplies(): String = PendingReplyStore(activity).consumeCompleted()

    @JavascriptInterface
    fun startAmbient(): String {
        if (!NativeAudioState.snapshot().running) NativeAudioState.markPermissionPending()
        activity.runOnUiThread { activity.requestMicrophoneAndStart() }
        return NativeAudioState.snapshotJson()
    }

    @JavascriptInterface
    fun stopAmbient(): String {
        AudioCaptureService.stop(activity)
        return NativeAudioState.snapshotJson()
    }

    @JavascriptInterface
    fun createAdbDiagnosticSnapshot(): String = DiagnosticExporter(activity).createAdbSnapshot().toString()

    @JavascriptInterface
    fun deleteAdbDiagnosticSnapshot(relativePath: String): Boolean =
        DiagnosticExporter(activity).deleteAdbSnapshot(relativePath)

    @JavascriptInterface
    fun startSpeakerEnrollment(sessionId: String): String {
        if (!NativeAudioState.snapshot().running) NativeAudioState.markPermissionPending()
        activity.runOnUiThread { activity.requestMicrophoneAndStartEnrollment(sessionId) }
        return NativeAudioState.snapshotJson()
    }

    @JavascriptInterface
    fun cancelSpeakerEnrollment(): String {
        activity.runOnUiThread { AudioCaptureService.cancelEnrollment(activity) }
        return NativeAudioState.snapshotJson()
    }

    @JavascriptInterface
    fun speakerEnrollmentStatus(): String = NativeAudioState.snapshotJson()

    @JavascriptInterface
    fun speak(text: String) {
        activity.runOnUiThread { activity.speakWithSystemTts(text) }
    }

    @JavascriptInterface
    fun stopSpeaking() {
        activity.runOnUiThread { activity.stopSystemTts() }
    }

    @JavascriptInterface
    fun openSettings() {
        activity.runOnUiThread { activity.openNativeSettings() }
    }
}
