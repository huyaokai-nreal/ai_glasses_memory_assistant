package com.aiglasses.memoryassistant.demo

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

class VadRuntimeProfileTest {
    @Test
    fun tenProfileUsesResolvedOptionsInEventTelemetry() {
        val profile = VadRuntimeProfile.from(
            backend = "ten_vad",
            modelPackVersion = "candidate-v2",
            options = JSONObject()
                .put("threshold", 0.35)
                .put("min_speech_seconds", 0.1)
                .put("num_threads", 2),
        )

        val event = profile.eventJson("speech_end")
        assertEquals("ten_vad", event.getString("backend"))
        assertEquals("candidate-v2", event.getString("model_pack_version"))
        assertEquals(0.35, event.getDouble("threshold"), 0.0001)
        assertEquals(0.1, event.getDouble("min_speech_seconds"), 0.0001)
        assertEquals(0.5, event.getDouble("min_silence_seconds"), 0.0001)
        assertEquals(256, event.getInt("window_size"))
        assertEquals(2, event.getInt("num_threads"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsUnsupportedBackend() {
        VadRuntimeProfile.from("unknown", "pack", JSONObject())
    }
}
