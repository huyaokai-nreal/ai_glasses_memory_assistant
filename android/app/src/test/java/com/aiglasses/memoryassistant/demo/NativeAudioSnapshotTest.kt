package com.aiglasses.memoryassistant.demo

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
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
}
