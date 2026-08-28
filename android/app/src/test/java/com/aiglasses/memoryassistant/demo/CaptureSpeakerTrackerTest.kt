package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CaptureSpeakerTrackerTest {
    @Test
    fun assignsStableAnonymousTracksWithinOneCapture() {
        val tracker = CaptureSpeakerTracker()

        val first = tracker.assign(floatArrayOf(1f, 0f), "other", "not_observed")
        val second = tracker.assign(floatArrayOf(0f, 1f), "unknown", "not_observed")
        val third = tracker.assign(floatArrayOf(0.99f, 0.01f), "other", "not_observed")

        assertEquals("spk_01", first.voiceGroup)
        assertEquals("spk_02", second.voiceGroup)
        assertEquals("spk_01", third.voiceGroup)
        assertEquals("capture", third.scope)
    }

    @Test
    fun resetStartsAnonymousNumberingForNextCapture() {
        val tracker = CaptureSpeakerTracker()
        assertEquals("spk_01", tracker.assign(floatArrayOf(1f, 0f), "other", "not_observed").voiceGroup)

        tracker.reset()

        assertEquals("spk_01", tracker.assign(floatArrayOf(0f, 1f), "other", "not_observed").voiceGroup)
    }

    @Test
    fun overlapLowQualityAndSelfAreNeverForcedIntoAnonymousTrack() {
        val tracker = CaptureSpeakerTracker()

        assertTrue(tracker.assign(floatArrayOf(1f, 0f), "other", "suspected").voiceGroup.isEmpty())
        assertTrue(tracker.assign(null, "other", "not_observed").voiceGroup.isEmpty())
        assertTrue(tracker.assign(floatArrayOf(1f, 0f), "user", "not_observed").voiceGroup.isEmpty())
    }
}
