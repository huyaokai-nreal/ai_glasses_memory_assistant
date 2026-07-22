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
