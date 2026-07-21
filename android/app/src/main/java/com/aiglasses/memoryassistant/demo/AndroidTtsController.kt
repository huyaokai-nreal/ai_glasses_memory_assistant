package com.aiglasses.memoryassistant.demo

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale
import java.util.UUID

class AndroidTtsController(context: Context) : TextToSpeech.OnInitListener {
    private val appContext = context.applicationContext
    private val engine = TextToSpeech(appContext, this)
    private var ready = false
    private var pendingText = ""

    init {
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = Unit
            override fun onDone(utteranceId: String?) = AudioCaptureService.resumeAfterTts(appContext)
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) = AudioCaptureService.resumeAfterTts(appContext)
            override fun onError(utteranceId: String?, errorCode: Int) = AudioCaptureService.resumeAfterTts(appContext)
        })
    }

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            engine.language = Locale.SIMPLIFIED_CHINESE
            engine.setSpeechRate(1.05f)
            pendingText.takeIf(String::isNotBlank)?.let {
                pendingText = ""
                speak(it)
            }
        } else {
            pendingText = ""
            AudioCaptureService.resumeAfterTts(appContext)
        }
    }

    fun speak(rawText: String) {
        val text = rawText.trim().take(MAX_TEXT_LENGTH)
        if (text.isEmpty()) return
        if (!ready) {
            pendingText = text
            return
        }
        stop()
        AudioCaptureService.pauseForTts(appContext)
        engine.speak(text, TextToSpeech.QUEUE_FLUSH, Bundle(), UUID.randomUUID().toString())
    }

    fun stop() {
        pendingText = ""
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
