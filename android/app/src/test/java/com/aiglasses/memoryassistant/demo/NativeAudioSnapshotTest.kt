package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test

class NativeAudioSnapshotTest {
    @Test
    fun foregroundLifecycleStatesAreRunning() {
        listOf("starting", "recording", "paused_tts", "stopping").forEach { state ->
            assertTrue(state, NativeAudioSnapshot(state = state).running)
        }
    }

    @Test
    fun idlePermissionAndErrorStatesAreNotRunning() {
        listOf("idle", "permission_pending", "error").forEach { state ->
            assertFalse(state, NativeAudioSnapshot(state = state).running)
        }
    }

    @Test
    fun modelSelfTestRoundTripsPublicComponentStatus() {
        val original = ModelSelfTestSnapshot(
            state = "ok",
            version = "pack-v1",
            checkedAtMillis = 123L,
            pssKb = 456,
            components = mapOf("vad" to ModelComponentSelfTest("ok", 12L)),
        )

        val restored = ModelSelfTestSnapshot.fromJson(original.toJson().toString())

        assertEquals(original, restored)
    }

    @Test
    fun partialAndEnrollmentSnapshotExposeOnlyPublicProgress() {
        val snapshot = NativeAudioSnapshot(
            latestPartial = "正在识别",
            partialSequence = 4,
            inferenceQueueDepth = 2,
            enrollmentState = "recording",
            enrollmentSessionId = "session-1",
            enrollmentSampleCount = 1,
        )

        assertEquals("正在识别", snapshot.latestPartial)
        assertEquals(4L, snapshot.partialSequence)
        assertEquals(2, snapshot.inferenceQueueDepth)
        assertEquals(1, snapshot.enrollmentSampleCount)
        assertFalse(NativeAudioSnapshot::class.java.declaredFields.any { it.name.contains("embedding") })
    }

    @Test
    fun captureStopSnapshotExposesOnlyPublicCorrelationFields() {
        val result = org.json.JSONObject()
            .put(
                "import_result",
                org.json.JSONObject().put(
                    "memory_job",
                    org.json.JSONObject().put("job_id", "job-1").put("status", "pending"),
                ),
            )

        NativeAudioState.markCaptureStopped("capture-1", result)
        NativeAudioState.markIdle()
        val public = NativeAudioState.snapshot()

        assertEquals("capture-1", public.lastStoppedCaptureId)
        assertEquals("completed", public.lastStopStatus)
        assertEquals("job-1", public.lastStopMemoryJobId)
        assertEquals("pending", public.lastStopMemoryJobStatus)
        assertFalse(NativeAudioSnapshot::class.java.declaredFields.any { it.name.contains("embedding") })
    }

    @Test
    fun uiSnapshotExposesWakePhaseAndFinalQueryWithoutPrivateAudio() {
        val ui = NativeAudioSnapshot(
            interactionState = "query_submitted",
            finalQuery = "我的车停在哪里",
            finalQueryEventId = "event-1",
            finalQuerySequence = 3,
        ).toUiJson(elapsedRealtimeMillis = 0L)

        assertEquals("query_submitted", ui.getString("interaction_state"))
        assertEquals("我的车停在哪里", ui.getString("final_query"))
        assertEquals("event-1", ui.getString("final_query_event_id"))
        assertEquals(3L, ui.getLong("final_query_sequence"))
        assertFalse(ui.has("captured_samples"))
    }

    @Test
    fun connectedWakeKeepsOnlyTheQuestion() {
        assertEquals(
            "我的车停在哪里",
            WakeQueryText.extract("我的车停在哪里", "你好小忆，我的车停在哪里？", "你好小忆"),
        )
        assertEquals(
            "我的车停在哪里？",
            WakeQueryText.extract("", "你好小忆，我的车停在哪里？", "你好小忆"),
        )
    }

    @Test
    fun standaloneWakeDoesNotBecomeAUserQuery() {
        assertEquals("", WakeQueryText.extract("", "你好小忆。", "你好小忆"))
        assertEquals("", WakeQueryText.extract("", "你好小姨。", "你好小忆"))
    }

    @Test
    fun overlapRulesStayFailClosedWithoutEnoughEvidence() {
        assertEquals("unknown", SpeakerOverlapRules.classify(16_000, emptyList()).state)
        assertEquals("not_observed", SpeakerOverlapRules.classify(48_000, emptyList()).state)
        assertEquals("unknown", SpeakerOverlapRules.classify(80_000, emptyList()).state)
    }

    @Test
    fun overlapRulesDistinguishConsistentAndConflictingWindows() {
        assertEquals("not_observed", SpeakerOverlapRules.classify(80_000, listOf(0.82f, 0.77f)).state)
        assertEquals("suspected", SpeakerOverlapRules.classify(80_000, listOf(0.82f, 0.51f)).state)
    }

    @Test
    fun audioLevelMeterDistinguishesSilenceAndSignal() {
        val silence = AudioLevelMeter.measure(ShortArray(16), 16)
        val signal = AudioLevelMeter.measure(shortArrayOf(0, 8_192, -8_192, 16_384), 4)

        assertEquals(-120.0, silence.rmsDbfs, 0.001)
        assertEquals(-120.0, silence.peakDbfs, 0.001)
        assertTrue(signal.rmsDbfs > -20.0)
        assertEquals(-6.02, signal.peakDbfs, 0.05)
    }
}
