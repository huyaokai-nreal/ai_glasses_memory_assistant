package com.aiglasses.memoryassistant.demo

import android.webkit.JavascriptInterface

class NativeAppBridge(private val activity: MainActivity) {
    @JavascriptInterface
    fun platform(): String = "android"

    @JavascriptInterface
    fun ownerId(): String = SecureSettings(activity).ownerId()

    @JavascriptInterface
    fun audioStatus(): String = NativeAudioState.snapshotJson()

    @JavascriptInterface
    fun consumeCompletedReplies(): String = PendingReplyStore(activity).consumeCompleted()

    @JavascriptInterface
    fun startAmbient(): String {
        NativeAudioState.markPermissionPending()
        activity.runOnUiThread { activity.requestMicrophoneAndStart() }
        return NativeAudioState.snapshotJson()
    }

    @JavascriptInterface
    fun stopAmbient(): String {
        AudioCaptureService.stop(activity)
        return NativeAudioState.snapshotJson()
    }

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
