package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import org.json.JSONObject

class AndroidTtsController(context: Context) : TextToSpeech.OnInitListener {
    private val appContext = context.applicationContext
    private val engine = TextToSpeech(appContext, this)
    private var ready = false
    private var pendingText = ""
    private var pendingTimingIds = JSONObject()
    private val timingByUtterance = ConcurrentHashMap<String, JSONObject>()

    init {
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                utteranceId?.let { id -> timingByUtterance[id]?.let { AcceptanceTiming.mark("tts_start", it) } }
            }
            override fun onDone(utteranceId: String?) {
                if (utteranceId != null) timingByUtterance.remove(utteranceId)
                AudioCaptureService.resumeAfterTts(appContext)
            }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                if (utteranceId != null) timingByUtterance.remove(utteranceId)
                AudioCaptureService.resumeAfterTts(appContext)
            }
            override fun onError(utteranceId: String?, errorCode: Int) = onError(utteranceId)
        })
    }

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            engine.language = Locale.SIMPLIFIED_CHINESE
            engine.setSpeechRate(1.05f)
            pendingText.takeIf(String::isNotBlank)?.let {
                val timingIds = pendingTimingIds
                pendingText = ""
                pendingTimingIds = JSONObject()
                speak(it, timingIds)
            }
        } else {
            pendingText = ""
            AudioCaptureService.resumeAfterTts(appContext)
        }
    }

    fun speak(rawText: String, timingIds: JSONObject = JSONObject()) {
        val text = rawText.trim().take(MAX_TEXT_LENGTH)
        if (text.isEmpty()) return
        if (!ready) {
            pendingText = text
            pendingTimingIds = timingIds
            return
        }
        stop()
        AudioCaptureService.pauseForTts(appContext)
        val id = UUID.randomUUID().toString()
        if (AcceptanceTiming.enabled) {
            timingIds.put("utterance_id", timingIds.optString("utterance_id").ifBlank { id })
            timingByUtterance[id] = timingIds
        }
        val result = engine.speak(text, TextToSpeech.QUEUE_FLUSH, Bundle(), id)
        if (result == TextToSpeech.ERROR) {
            timingByUtterance.remove(id)
        }
    }

    fun stop() {
        pendingText = ""
        pendingTimingIds = JSONObject()
        timingByUtterance.clear()
        engine.stop()
        AudioCaptureService.resumeAfterTts(appContext)
    }

    fun shutdown() {
        stop()
        engine.shutdown()
        ready = false
    }

    companion object {
        private const val MAX_TEXT_LENGTH = 4_000
    }
}
